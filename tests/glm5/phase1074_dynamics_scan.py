#!/usr/bin/env python3
"""Map Phase1074 polarity-conditioned state writes and Attention routing."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1069_local_coordinate_scan as residual_tools
from phase1070_process_answer_scan import safe_cosine
import phase1074_polarity_dynamics_protocol as protocol


BATCH_SIZE = {"qwen3": 4, "glm4": 2, "deepseek7b": 2}
CONDITIONINGS = ("all", "behavior_conditioned")
ROUTING_METRICS = ("attention_mass", "av_norm")
EPSILON = 1e-12


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if (
        isinstance(output, (tuple, list))
        and output
        and torch.is_tensor(output[0])
    ):
        return output[0]
    raise TypeError(f"unsupported projection output {type(output)!r}")


def attention_tensor(output: Any) -> torch.Tensor:
    if not isinstance(output, (tuple, list)):
        raise TypeError("self-attention did not return a tuple")
    candidates = [
        value
        for value in output
        if torch.is_tensor(value) and value.ndim == 4
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one attention tensor, found {len(candidates)}"
        )
    return candidates[0]


def pad_rows(
    rows: list[dict[str, Any]],
    pad_id: int,
    device: torch.device,
    fixed_width: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    required = max(len(row["input_ids"]) for row in rows)
    if fixed_width < required:
        raise RuntimeError("shared width is smaller than a prompt")
    input_ids = torch.full(
        (len(rows), fixed_width),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    lengths = torch.zeros(
        len(rows), dtype=torch.long, device=device
    )
    positions = torch.zeros(
        (len(rows), len(protocol.CAPTURE_ROLES)),
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        values = torch.tensor(
            row["input_ids"], dtype=torch.long, device=device
        )
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
        positions[index] = torch.tensor(
            [
                int(row["role_positions"][role])
                for role in protocol.CAPTURE_ROLES
            ],
            dtype=torch.long,
            device=device,
        )
    return input_ids, attention_mask, lengths, positions


def routing_positions(
    rows: list[dict[str, Any]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    max_span = max(
        int(row["semantic_source_spans"][source][1])
        - int(row["semantic_source_spans"][source][0])
        + 1
        for row in rows
        for source in protocol.SEMANTIC_SOURCES
    )
    source_positions = torch.zeros(
        (
            len(rows),
            len(protocol.SEMANTIC_SOURCES),
            max_span,
        ),
        dtype=torch.long,
    )
    source_masks = torch.zeros_like(
        source_positions, dtype=torch.bool
    )
    destination_positions = torch.zeros(
        (len(rows), len(protocol.ATTENTION_DESTINATIONS)),
        dtype=torch.long,
    )
    for row_slot, row in enumerate(rows):
        for source_slot, source in enumerate(
            protocol.SEMANTIC_SOURCES
        ):
            start, end = (
                int(value)
                for value in row["semantic_source_spans"][source]
            )
            values = list(range(start, end + 1))
            source_positions[
                row_slot, source_slot, :len(values)
            ] = torch.tensor(values, dtype=torch.long)
            source_masks[
                row_slot, source_slot, :len(values)
            ] = True
        for destination_slot, destination in enumerate(
            protocol.ATTENTION_DESTINATIONS
        ):
            destination_positions[
                row_slot, destination_slot
            ] = int(row["role_positions"][destination])
    return (
        source_positions,
        source_masks,
        destination_positions,
    )


class BatchRoutingCapture:
    def __init__(self, layers: list[Any]) -> None:
        self.layers = layers
        self.handles = []
        self.v_cache: dict[int, torch.Tensor] = {}
        self.mass: dict[int, torch.Tensor] = {}
        self.av_norm: dict[int, torch.Tensor] = {}
        self.source_positions: torch.Tensor | None = None
        self.source_masks: torch.Tensor | None = None
        self.destination_positions: torch.Tensor | None = None

    def register(self) -> None:
        for depth, layer in enumerate(self.layers, 1):
            self.handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    self._v_hook(depth)
                )
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._attention_hook(depth)
                )
            )

    def begin(
        self,
        source_positions: torch.Tensor,
        source_masks: torch.Tensor,
        destination_positions: torch.Tensor,
    ) -> None:
        self.source_positions = source_positions
        self.source_masks = source_masks
        self.destination_positions = destination_positions
        self.v_cache = {}
        self.mass = {}
        self.av_norm = {}

    def validate(self) -> None:
        expected = set(range(1, len(self.layers) + 1))
        if (
            set(self.mass) != expected
            or set(self.av_norm) != expected
            or self.v_cache
        ):
            raise RuntimeError(
                "routing capture drift: "
                f"mass={sorted(self.mass)} "
                f"av={sorted(self.av_norm)} "
                f"v={sorted(self.v_cache)}"
            )

    def clear(self) -> None:
        self.source_positions = None
        self.source_masks = None
        self.destination_positions = None
        self.v_cache = {}
        self.mass = {}
        self.av_norm = {}

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.clear()

    def _v_hook(self, depth: int):
        def hook(_module, _inputs, output):
            self.v_cache[depth] = output_tensor(output)
        return hook

    def _attention_hook(self, depth: int):
        def hook(module, _inputs, output):
            if (
                self.source_positions is None
                or self.source_masks is None
                or self.destination_positions is None
            ):
                raise RuntimeError(
                    "routing hook fired outside an active batch"
                )
            values = self.v_cache.pop(depth)
            attention = attention_tensor(output).float()
            batch_size, n_heads, _, _ = attention.shape
            head_dim = int(getattr(module, "head_dim", 0))
            if head_dim <= 0:
                configured = int(
                    getattr(module, "num_key_value_heads", 0)
                )
                if configured <= 0:
                    configured = n_heads
                head_dim = int(values.shape[-1] // configured)
            n_kv_heads = int(values.shape[-1] // head_dim)
            if n_heads % n_kv_heads:
                raise RuntimeError("query/KV head grouping drift")
            values = values.reshape(
                batch_size,
                values.shape[1],
                n_kv_heads,
                head_dim,
            ).float()
            head_to_kv = (
                torch.arange(n_heads, device=values.device)
                // (n_heads // n_kv_heads)
            )
            mass = torch.zeros(
                (
                    batch_size,
                    n_heads,
                    len(protocol.ATTENTION_DESTINATIONS),
                    len(protocol.SEMANTIC_SOURCES),
                ),
                dtype=torch.float32,
                device=attention.device,
            )
            av = torch.zeros_like(mass)
            for batch_slot in range(batch_size):
                for destination_slot in range(
                    len(protocol.ATTENTION_DESTINATIONS)
                ):
                    destination = int(
                        self.destination_positions[
                            batch_slot, destination_slot
                        ]
                    )
                    for source_slot in range(
                        len(protocol.SEMANTIC_SOURCES)
                    ):
                        valid = self.source_masks[
                            batch_slot, source_slot
                        ].nonzero(as_tuple=False).flatten()
                        positions = self.source_positions[
                            batch_slot, source_slot, valid
                        ].tolist()
                        weights = attention[
                            batch_slot, :, destination, positions
                        ]
                        if weights.ndim == 1:
                            weights = weights[:, None]
                        selected_values = values[
                            batch_slot, positions, :, :
                        ][:, head_to_kv, :].permute(1, 0, 2)
                        contribution = (
                            weights.to(selected_values.device)[..., None]
                            * selected_values
                        ).sum(dim=1)
                        mass[
                            batch_slot,
                            :,
                            destination_slot,
                            source_slot,
                        ] = weights.sum(dim=-1)
                        av[
                            batch_slot,
                            :,
                            destination_slot,
                            source_slot,
                        ] = contribution.norm(dim=-1)
            self.mass[depth] = mass.detach().cpu()
            self.av_norm[depth] = av.detach().cpu()
        return hook


def state_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["orientation"]), int(row["lexical_branch"])


def relative_magnitude(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(value, dim=-1)
    result = torch.full_like(numerator, float("nan"))
    finite = (
        torch.isfinite(value).all(dim=-1)
        & torch.isfinite(numerator)
        & torch.isfinite(reference)
        & (reference > EPSILON)
    )
    result[finite] = numerator[finite] / reference[finite]
    return result


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    behavior_decision = protocol.read_json(
        protocol.OUT_ROOT
        / "analysis"
        / "behavior_decision.json"
    )
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1074 protocol audit failed")
    if not behavior_decision["should_run_internal_dynamics"]:
        raise RuntimeError(
            "Phase1074 behavior gate did not authorize dynamics"
        )
    all_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    rows = [
        row
        for row in all_rows
        if int(row["replicate"]) in protocol.INTERNAL_REPLICATES
    ]
    behavior_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "behavior"
        / model_name
        / "candidate_behavior.jsonl"
    )
    behavior_hit = {
        int(row["semantic_case_index"]): bool(row["candidate_hit"])
        for row in behavior_rows
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["unit_id"])].append(row)
    if (
        len(rows) != prereg["internal_case_count_per_model"]
        or len(grouped) != prereg["internal_unit_count_per_model"]
    ):
        raise RuntimeError("Phase1074 internal subset drift")

    started = time.time()
    model = tokenizer = residual_capture = routing_capture = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        n_layers = len(layers)
        n_heads = int(model.config.num_attention_heads)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos token")

        relation_index = {
            value: index
            for index, value in enumerate(protocol.RELATIONS)
        }
        split_index = {
            value: index
            for index, value in enumerate(protocol.SPLITS)
        }
        path_index = {
            value: index
            for index, value in enumerate(protocol.PATHS)
        }
        layout_index = {
            value: index
            for index, value in enumerate(protocol.LAYOUTS)
        }
        source_index = {
            value: index
            for index, value in enumerate(protocol.SEMANTIC_SOURCES)
        }
        source_pair_index = {
            value[0]: index
            for index, value in enumerate(protocol.SOURCE_PAIRS)
        }
        conditioning_index = {
            value: index
            for index, value in enumerate(CONDITIONINGS)
        }
        routing_shape = (
            len(protocol.RELATIONS),
            len(protocol.SPLITS),
            len(protocol.PATHS),
            len(protocol.LAYOUTS),
            n_layers,
            n_heads,
            len(protocol.ATTENTION_DESTINATIONS),
            len(protocol.SOURCE_PAIRS),
            len(ROUTING_METRICS),
            len(CONDITIONINGS),
        )
        routing_sums = np.zeros(routing_shape, dtype=np.float64)
        routing_counts = np.zeros(routing_shape, dtype=np.int32)
        routing_positive = np.zeros(routing_shape, dtype=np.int32)
        residual_records = []
        residual_attempts = 0
        residual_nonfinite = 0
        routing_attempts = 0
        routing_nonfinite = 0

        residual_capture = residual_tools.ResidualRoleCapture(
            model, layers
        )
        routing_capture = BatchRoutingCapture(layers)
        residual_capture.register()
        routing_capture.register()

        canonical_states = [
            (orientation, lexical)
            for orientation in protocol.ORIENTATIONS
            for lexical in protocol.LEXICAL_BRANCHES
        ]
        role_index = {
            role: index
            for index, role in enumerate(protocol.CAPTURE_ROLES)
        }
        source_pairs = {
            name: (
                source_index[positive],
                source_index[negative],
            )
            for name, positive, negative in protocol.SOURCE_PAIRS
        }

        with torch.inference_mode():
            for unit_number, (unit_id, unit_rows) in enumerate(
                sorted(grouped.items()), 1
            ):
                by_task: dict[
                    str, dict[tuple[int, int], dict[str, Any]]
                ] = defaultdict(dict)
                for row in unit_rows:
                    by_task[str(row["task"])][state_key(row)] = row
                if set(by_task) != set(protocol.TASKS) or any(
                    set(by_task[task]) != set(canonical_states)
                    for task in protocol.TASKS
                ):
                    raise RuntimeError(
                        f"incomplete Phase1074 unit: {unit_id}"
                    )
                reference = unit_rows[0]
                all_correct = all(
                    behavior_hit[int(row["semantic_case_index"])]
                    for row in unit_rows
                )
                conditionings = [conditioning_index["all"]]
                if all_correct:
                    conditionings.append(
                        conditioning_index["behavior_conditioned"]
                    )
                shared_width = max(
                    len(row["input_ids"]) for row in unit_rows
                )
                task_residual: dict[str, torch.Tensor] = {}
                task_mass: dict[str, torch.Tensor] = {}
                task_av: dict[str, torch.Tensor] = {}
                for task in protocol.TASKS:
                    ordered_rows = [
                        by_task[task][state]
                        for state in canonical_states
                    ]
                    residual_parts = []
                    mass_parts = []
                    av_parts = []
                    for batch in chunks(
                        ordered_rows, BATCH_SIZE[model_name]
                    ):
                        (
                            input_ids,
                            attention_mask,
                            _lengths,
                            positions,
                        ) = pad_rows(
                            batch,
                            int(pad_id),
                            device,
                            fixed_width=shared_width,
                        )
                        (
                            source_positions,
                            source_masks,
                            destination_positions,
                        ) = routing_positions(batch)
                        residual_capture.begin(positions)
                        routing_capture.begin(
                            source_positions,
                            source_masks,
                            destination_positions,
                        )
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            output_attentions=True,
                            return_dict=True,
                        )
                        residual_capture.validate()
                        routing_capture.validate()
                        residual_parts.append(torch.stack(
                            [
                                residual_capture.values[depth]
                                .float()
                                .cpu()
                                for depth in range(n_layers + 1)
                            ],
                            dim=1,
                        ))
                        mass_parts.append(torch.stack(
                            [
                                routing_capture.mass[depth]
                                for depth in range(1, n_layers + 1)
                            ],
                            dim=1,
                        ))
                        av_parts.append(torch.stack(
                            [
                                routing_capture.av_norm[depth]
                                for depth in range(1, n_layers + 1)
                            ],
                            dim=1,
                        ))
                        del (
                            output,
                            input_ids,
                            attention_mask,
                            positions,
                        )
                        residual_capture.values = {}
                        routing_capture.clear()
                    task_residual[task] = torch.cat(
                        residual_parts, dim=0
                    )
                    task_mass[task] = torch.cat(mass_parts, dim=0)
                    task_av[task] = torch.cat(av_parts, dim=0)

                stacked_states = torch.stack(
                    [
                        task_residual[task]
                        for task in protocol.TASKS
                    ],
                    dim=0,
                )
                state_reference = torch.linalg.vector_norm(
                    stacked_states, dim=-1
                ).mean(dim=(0, 1))
                max_state = task_residual["max"]
                min_state = task_residual["min"]
                selection_vectors = []
                task_vectors = []
                for lexical in protocol.LEXICAL_BRANCHES:
                    i0 = canonical_states.index((0, lexical))
                    i1 = canonical_states.index((1, lexical))
                    selection_vectors.append(
                        (max_state[i0] - min_state[i0])
                        - (max_state[i1] - min_state[i1])
                    )
                    task_vectors.append(
                        (
                            (max_state[i0] - min_state[i0])
                            + (max_state[i1] - min_state[i1])
                        )
                        / 2.0
                    )
                selection = torch.stack(selection_vectors)
                task_contrast = torch.stack(task_vectors)
                selection_relative = relative_magnitude(
                    selection, state_reference[None, ...]
                )
                task_relative = relative_magnitude(
                    task_contrast, state_reference[None, ...]
                )
                transition = selection[:, 1:] - selection[:, :-1]
                transition_relative = relative_magnitude(
                    transition, state_reference[None, 1:, ...]
                )
                lexical_cosine = safe_cosine(
                    selection[0], selection[1]
                )

                for depth in range(n_layers + 1):
                    for role, role_slot in role_index.items():
                        values = [
                            float(
                                selection_relative[
                                    lexical, depth, role_slot
                                ].item()
                            )
                            for lexical in protocol.LEXICAL_BRANCHES
                        ]
                        task_values = [
                            float(
                                task_relative[
                                    lexical, depth, role_slot
                                ].item()
                            )
                            for lexical in protocol.LEXICAL_BRANCHES
                        ]
                        transition_values = []
                        if depth < n_layers:
                            transition_values = [
                                float(
                                    transition_relative[
                                        lexical, depth, role_slot
                                    ].item()
                                )
                                for lexical in protocol.LEXICAL_BRANCHES
                            ]
                        cosine_value = float(
                            lexical_cosine[depth, role_slot].item()
                        )
                        attempted = (
                            len(values)
                            + len(task_values)
                            + len(transition_values)
                            + 1
                        )
                        finite_values = [
                            value
                            for value in (
                                values
                                + task_values
                                + transition_values
                                + [cosine_value]
                            )
                            if math.isfinite(value)
                        ]
                        residual_attempts += attempted
                        residual_nonfinite += (
                            attempted - len(finite_values)
                        )
                        residual_records.append({
                            "schema_version": (
                                "phase1074_residual_unit_metric.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "unit_id": unit_id,
                            "relation": reference["relation"],
                            "path": reference["path"],
                            "layout": reference["layout"],
                            "template_index": reference[
                                "template_index"
                            ],
                            "split": reference["split"],
                            "all_cases_correct": all_correct,
                            "depth": depth,
                            "relative_depth": depth / n_layers,
                            "role": role,
                            "selection_relative_magnitude": (
                                float(np.mean(values))
                                if all(
                                    math.isfinite(value)
                                    for value in values
                                )
                                else None
                            ),
                            "task_contrast_relative_magnitude": (
                                float(np.mean(task_values))
                                if all(
                                    math.isfinite(value)
                                    for value in task_values
                                )
                                else None
                            ),
                            "transition_relative_magnitude": (
                                float(np.mean(transition_values))
                                if transition_values
                                and all(
                                    math.isfinite(value)
                                    for value in transition_values
                                )
                                else None
                            ),
                            "selection_lexical_reuse_cosine": (
                                cosine_value
                                if math.isfinite(cosine_value)
                                else None
                            ),
                        })

                for metric_slot, metric_name in enumerate(
                    ROUTING_METRICS
                ):
                    task_values = (
                        task_mass
                        if metric_name == "attention_mass"
                        else task_av
                    )
                    max_values = task_values["max"]
                    min_values = task_values["min"]
                    pair_values = []
                    for pair_name, (positive, negative) in (
                        source_pairs.items()
                    ):
                        contrasts = []
                        for state_slot, _state in enumerate(
                            canonical_states
                        ):
                            contrasts.append(
                                (
                                    max_values[
                                        state_slot, ..., positive
                                    ]
                                    - max_values[
                                        state_slot, ..., negative
                                    ]
                                )
                                - (
                                    min_values[
                                        state_slot, ..., positive
                                    ]
                                    - min_values[
                                        state_slot, ..., negative
                                    ]
                                )
                            )
                        pair_values.append(
                            torch.stack(contrasts).mean(dim=0)
                        )
                    routing = torch.stack(
                        pair_values, dim=-1
                    ).numpy()
                    routing_attempts += int(routing.size)
                    routing_nonfinite += int(
                        (~np.isfinite(routing)).sum()
                    )
                    finite = np.isfinite(routing)
                    clean = np.where(finite, routing, 0.0)
                    positive = np.where(
                        finite & (routing > 0.0), 1, 0
                    )
                    base = (
                        relation_index[reference["relation"]],
                        split_index[reference["split"]],
                        path_index[reference["path"]],
                        layout_index[reference["layout"]],
                    )
                    for conditioning in conditionings:
                        index = (
                            *base,
                            slice(None),
                            slice(None),
                            slice(None),
                            slice(None),
                            metric_slot,
                            conditioning,
                        )
                        routing_sums[index] += clean
                        routing_counts[index] += finite.astype(
                            np.int32
                        )
                        routing_positive[index] += positive

                del (
                    task_residual,
                    task_mass,
                    task_av,
                    stacked_states,
                    selection,
                    task_contrast,
                    transition,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if unit_number % 5 == 0 or unit_number == len(grouped):
                    print(json.dumps({
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "internal_units_complete": unit_number,
                        "internal_units_total": len(grouped),
                    }), flush=True)

        residual_capture.close()
        routing_capture.close()
        residual_capture = routing_capture = None
        out_dir = protocol.OUT_ROOT / "dynamics" / model_name
        protocol.write_jsonl(
            out_dir / "residual_unit_metrics.jsonl",
            residual_records,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "routing_aggregates.npz",
            sums=routing_sums,
            counts=routing_counts,
            positive_counts=routing_positive,
            relations=np.asarray(protocol.RELATIONS),
            splits=np.asarray(protocol.SPLITS),
            paths=np.asarray(protocol.PATHS),
            layouts=np.asarray(protocol.LAYOUTS),
            destinations=np.asarray(
                protocol.ATTENTION_DESTINATIONS
            ),
            source_pairs=np.asarray(
                [value[0] for value in protocol.SOURCE_PAIRS]
            ),
            metrics=np.asarray(ROUTING_METRICS),
            conditionings=np.asarray(CONDITIONINGS),
        )
        summary = {
            "schema_version": "phase1074_dynamics_scan_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "internal_case_count": len(rows),
            "internal_unit_count": len(grouped),
            "behavior_model_selected": (
                model_name in behavior_decision["selected_models"]
            ),
            "residual_metric_attempt_count": residual_attempts,
            "residual_metric_finite_rate": (
                1.0
                - residual_nonfinite / residual_attempts
                if residual_attempts
                else 0.0
            ),
            "routing_metric_attempt_count": routing_attempts,
            "routing_metric_finite_rate": (
                1.0
                - routing_nonfinite / routing_attempts
                if routing_attempts
                else 0.0
            ),
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(out_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        if residual_capture is not None:
            residual_capture.close()
        if routing_capture is not None:
            routing_capture.close()
        if model is not None:
            release_fp16(model)
        del tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=protocol.MODELS, required=True
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
