#!/usr/bin/env python3
"""Map held-out polarity interactions, local readout, and Attention routing."""

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
import phase1074_dynamics_scan as routing_tools
import phase1075_relation_polarity_protocol as protocol


BATCH_SIZE = {"qwen3": 4, "glm4": 2, "deepseek7b": 2}
CONDITIONINGS = ("all", "behavior_conditioned")
ROUTING_METRICS = ("attention_mass", "av_norm")
EPSILON = 1e-12

# Reuse only the architecture-generic capture implementation. Its dimensions
# are read from this phase's frozen protocol at runtime.
routing_tools.protocol = protocol
BatchRoutingCapture = routing_tools.BatchRoutingCapture


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return source_positions, source_masks, destination_positions


def state_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["orientation"]), int(row["lexical_branch"])


def relative_magnitude(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(value, dim=-1)
    expanded_reference = torch.broadcast_to(
        reference, numerator.shape
    )
    result = torch.full_like(numerator, float("nan"))
    finite = (
        torch.isfinite(value).all(dim=-1)
        & torch.isfinite(numerator)
        & torch.isfinite(expanded_reference)
        & (expanded_reference > EPSILON)
    )
    result[finite] = (
        numerator[finite] / expanded_reference[finite]
    )
    return result


def output_weight(model) -> torch.Tensor:
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise RuntimeError("model has no output embedding weight")
    weight = output.weight
    if getattr(weight, "is_meta", False):
        hook = getattr(output, "_hf_hook", None)
        weights_map = getattr(hook, "weights_map", None)
        try:
            weight = (
                weights_map["weight"]
                if weights_map is not None
                else None
            )
        except (KeyError, TypeError):
            weight = None
        if weight is None or getattr(weight, "is_meta", False):
            raise RuntimeError(
                "unable to resolve offloaded output embedding"
            )
    return weight


def candidate_rows_for_units(
    model,
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, torch.Tensor]]:
    weight = output_weight(model)
    ids = sorted({
        int(token_id)
        for rows in grouped.values()
        for class_name in ("b0", "b1")
        for token_id in rows[0][
            "candidate_first_token_ids"
        ][class_name]
    })
    index = torch.tensor(ids, dtype=torch.long, device=weight.device)
    selected = (
        weight.index_select(0, index).detach().float().cpu()
    )
    by_id = {
        token_id: selected[position]
        for position, token_id in enumerate(ids)
    }
    result = {}
    for unit_id, rows in grouped.items():
        row = rows[0]
        result[unit_id] = {
            class_name: torch.stack([
                by_id[int(token_id)]
                for token_id in row[
                    "candidate_first_token_ids"
                ][class_name]
            ]).mean(dim=0)
            for class_name in ("b0", "b1")
        }
    return result


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    decision = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_decision.json"
    )
    internal_prereg = protocol.read_json(
        protocol.OUT_ROOT
        / "analysis"
        / "internal_preregistration.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1075 protocol audit failed")
    if not decision["should_run_internal_mapping"]:
        raise RuntimeError("Phase1075 internal mapping is not authorized")
    if (
        internal_prereg["protocol_digest"]
        != prereg["protocol_digest"]
        or internal_prereg["behavior_decision_digest"]
        != decision["decision_digest"]
        or protocol.digest({
            key: value
            for key, value in internal_prereg.items()
            if key != "internal_preregistration_digest"
        })
        != internal_prereg["internal_preregistration_digest"]
    ):
        raise RuntimeError("Phase1075 internal preregistration drift")
    authorized_relations = [
        relation
        for relation in decision["selected_relations"]
        if model_name in decision[
            "authorized_models_by_relation"
        ][relation]
    ]
    if not authorized_relations:
        raise RuntimeError(
            f"no authorized Phase1075 relation for {model_name}"
        )

    all_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    rows = [
        row
        for row in all_rows
        if row["relation"] in authorized_relations
        and int(row["replicate"]) in protocol.INTERNAL_REPLICATES
    ]
    expected_cases = (
        len(authorized_relations)
        * int(prereg["internal_cases_per_relation_model"])
    )
    if len(rows) != expected_cases:
        raise RuntimeError("Phase1075 internal case count drift")
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
    expected_units = (
        len(authorized_relations)
        * int(prereg["internal_units_per_relation_model"])
    )
    if len(grouped) != expected_units:
        raise RuntimeError("Phase1075 internal unit count drift")

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
        final_norm = residual_tools.final_norm_module(model)
        norm_device, norm_dtype = residual_tools.module_device_dtype(
            final_norm
        )
        candidates_cpu = candidate_rows_for_units(model, grouped)
        candidate_cache: dict[
            tuple[str, str], dict[str, torch.Tensor]
        ] = {}
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos token")

        relation_index = {
            relation: index
            for index, relation in enumerate(authorized_relations)
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
        conditioning_index = {
            value: index
            for index, value in enumerate(CONDITIONINGS)
        }
        source_pairs = {
            name: (
                source_index[positive],
                source_index[negative],
            )
            for name, positive, negative in protocol.SOURCE_PAIRS
        }
        routing_shape = (
            len(authorized_relations),
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
        internal_records = []
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
                        f"incomplete Phase1075 unit: {unit_id}"
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
                task_residual = {}
                task_mass = {}
                task_av = {}
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

                max_state = task_residual["max"]
                min_state = task_residual["min"]
                state_reference = torch.linalg.vector_norm(
                    torch.stack([max_state, min_state]),
                    dim=-1,
                ).mean(dim=(0, 1))
                raw_interactions = []
                for lexical in protocol.LEXICAL_BRANCHES:
                    i0 = canonical_states.index((0, lexical))
                    i1 = canonical_states.index((1, lexical))
                    raw_interactions.append(
                        (max_state[i0] - min_state[i0])
                        - (max_state[i1] - min_state[i1])
                    )
                raw_interaction = torch.stack(raw_interactions)
                raw_relative = relative_magnitude(
                    raw_interaction, state_reference[None, ...]
                )
                raw_transition = (
                    raw_interaction[:, 1:]
                    - raw_interaction[:, :-1]
                )
                raw_transition_relative = relative_magnitude(
                    raw_transition, state_reference[None, 1:, ...]
                )
                raw_lexical_cosine = safe_cosine(
                    raw_interaction[0], raw_interaction[1]
                )

                cache_key = (unit_id, str(norm_device))
                if cache_key not in candidate_cache:
                    candidate_cache[cache_key] = {
                        class_name: value.to(norm_device)
                        for class_name, value in candidates_cpu[
                            unit_id
                        ].items()
                    }
                candidates = candidate_cache[cache_key]
                local_by_task = {}
                for task, values in task_residual.items():
                    flattened = values.reshape(
                        -1, values.shape[-1]
                    )
                    normed = final_norm(
                        flattened.to(
                            device=norm_device,
                            dtype=norm_dtype,
                        )
                    ).float()
                    b0 = normed @ candidates["b0"].float()
                    b1 = normed @ candidates["b1"].float()
                    class_margin = (b1 - b0).reshape(
                        values.shape[:-1]
                    ).cpu()
                    high_low = class_margin.clone()
                    for state_slot, (orientation, _lexical) in enumerate(
                        canonical_states
                    ):
                        if orientation == 0:
                            high_low[state_slot] *= -1.0
                    local_by_task[task] = high_low
                    del flattened, normed, b0, b1, class_margin

                local_separations = []
                for lexical in protocol.LEXICAL_BRANCHES:
                    i0 = canonical_states.index((0, lexical))
                    i1 = canonical_states.index((1, lexical))
                    local_separations.append(0.5 * (
                        (
                            local_by_task["max"][i0]
                            - local_by_task["min"][i0]
                        )
                        + (
                            local_by_task["max"][i1]
                            - local_by_task["min"][i1]
                        )
                    ))
                local_selection = torch.stack(local_separations)

                for depth in range(n_layers + 1):
                    for role, role_slot in role_index.items():
                        raw_values = [
                            float(
                                raw_relative[
                                    lexical, depth, role_slot
                                ].item()
                            )
                            for lexical in protocol.LEXICAL_BRANCHES
                        ]
                        transition_values = []
                        if depth < n_layers:
                            transition_values = [
                                float(
                                    raw_transition_relative[
                                        lexical, depth, role_slot
                                    ].item()
                                )
                                for lexical in protocol.LEXICAL_BRANCHES
                            ]
                        local_values = [
                            float(
                                local_selection[
                                    lexical, depth, role_slot
                                ].item()
                            )
                            for lexical in protocol.LEXICAL_BRANCHES
                        ]
                        cosine_value = float(
                            raw_lexical_cosine[
                                depth, role_slot
                            ].item()
                        )
                        all_values = (
                            raw_values
                            + transition_values
                            + local_values
                            + [cosine_value]
                        )
                        residual_attempts += len(all_values)
                        residual_nonfinite += sum(
                            not math.isfinite(value)
                            for value in all_values
                        )
                        internal_records.append({
                            "schema_version": (
                                "phase1075_internal_unit_metric.v1"
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
                            "raw_interaction_relative_magnitude": (
                                float(np.mean(raw_values))
                                if all(
                                    math.isfinite(value)
                                    for value in raw_values
                                )
                                else None
                            ),
                            "raw_transition_relative_magnitude": (
                                float(np.mean(transition_values))
                                if transition_values
                                and all(
                                    math.isfinite(value)
                                    for value in transition_values
                                )
                                else None
                            ),
                            "raw_interaction_lexical_cosine": (
                                cosine_value
                                if math.isfinite(cosine_value)
                                else None
                            ),
                            "local_selection_separation": (
                                float(np.mean(local_values))
                                if all(
                                    math.isfinite(value)
                                    for value in local_values
                                )
                                else None
                            ),
                            "local_selection_positive_fraction": (
                                float(np.mean([
                                    value > 0.0
                                    for value in local_values
                                ]))
                                if all(
                                    math.isfinite(value)
                                    for value in local_values
                                )
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
                    for _pair_name, (positive, negative) in (
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
                    max_state,
                    min_state,
                    raw_interaction,
                    raw_transition,
                    local_by_task,
                    local_selection,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if unit_number % 4 == 0 or unit_number == len(grouped):
                    print(json.dumps({
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "internal_units_complete": unit_number,
                        "internal_units_total": len(grouped),
                    }), flush=True)

        residual_capture.close()
        routing_capture.close()
        residual_capture = routing_capture = None
        out_dir = protocol.OUT_ROOT / "internal" / model_name
        protocol.write_jsonl(
            out_dir / "unit_metrics.jsonl", internal_records
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "routing_aggregates.npz",
            sums=routing_sums,
            counts=routing_counts,
            positive_counts=routing_positive,
            relations=np.asarray(authorized_relations),
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
            "schema_version": "phase1075_internal_scan_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "authorized_relations": authorized_relations,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "internal_case_count": len(rows),
            "internal_unit_count": len(grouped),
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
            "final_norm_device": str(norm_device),
            "final_norm_dtype": str(norm_dtype),
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
