#!/usr/bin/env python3
"""Map all-layer post-query residual, attention, and MLP responses."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1034_post_query_component_protocol as protocol


BATCH_SIZE = {"qwen3": 24, "glm4": 8, "deepseek7b": 8}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def safe_cos(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum(left.float() * right.float(), dim=-1)
    denominator = (
        torch.linalg.vector_norm(left.float(), dim=-1)
        * torch.linalg.vector_norm(right.float(), dim=-1)
    )
    return numerator / torch.clamp(denominator, min=EPS)


def relative_norm(
    difference: torch.Tensor,
    states: torch.Tensor,
) -> torch.Tensor:
    scale = torch.linalg.vector_norm(states.float(), dim=-1).mean(dim=1)
    value = torch.linalg.vector_norm(difference.float(), dim=-1)
    return value / torch.clamp(scale, min=EPS)


def gather_positions(
    hidden: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    positions = positions.to(hidden.device)
    batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    return hidden[batch, positions, :]


def gather_span_means(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    positions = positions.to(hidden.device)
    masks = masks.to(hidden.device)
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    batch = batch[:, None, None].expand_as(positions)
    values = hidden[batch, positions, :]
    weights = masks[..., None].to(values.dtype)
    return (values * weights).sum(dim=2) / torch.clamp(
        weights.sum(dim=2), min=1
    )


def grouped(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] % 4:
        raise RuntimeError("batch does not preserve four-world units")
    return values.reshape(values.shape[0] // 4, 4, *values.shape[1:])


def make_batch(
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    expected_worlds = ["00", "10", "01", "11"]
    if len(rows) % 4:
        raise RuntimeError("batch size must be a multiple of four")
    unit_indices = []
    for start in range(0, len(rows), 4):
        group = rows[start:start + 4]
        if [row["world"] for row in group] != expected_worlds:
            raise RuntimeError("four-world order drift")
        if len({int(row["unit_index"]) for row in group}) != 1:
            raise RuntimeError("unit rows are not contiguous")
        unit_indices.append(int(group[0]["unit_index"]))

    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full(
        (len(rows), width), int(pad_token_id), dtype=torch.long
    )
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    anchors = torch.empty(
        (len(rows), len(protocol.ANCHORS)), dtype=torch.long
    )
    span_positions = torch.zeros(
        (len(rows), 2, 2), dtype=torch.long
    )
    span_masks = torch.zeros(
        (len(rows), 2, 2), dtype=torch.bool
    )

    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[index, :len(values)] = values
        mask[index, :len(values)] = 1
        anchors[index] = torch.tensor(
            protocol.anchor_positions(row), dtype=torch.long
        )
        for role_index, role in enumerate(("concept_a", "concept_b")):
            start, end = (
                int(value) for value in row["role_spans"][role]
            )
            positions = list(range(start, end + 1))
            if len(positions) > 2:
                raise RuntimeError("concept span exceeds frozen maximum")
            span_positions[index, role_index, :len(positions)] = (
                torch.tensor(positions, dtype=torch.long)
            )
            if len(positions) == 1:
                span_positions[index, role_index, 1] = positions[0]
            span_masks[index, role_index, :len(positions)] = True

    return (
        ids.to(device),
        mask.to(device),
        anchors,
        span_positions,
        span_masks,
        np.asarray(unit_indices, dtype=np.int64),
    )


class ComponentAtlasCapture:
    def __init__(
        self,
        layers: list[Any],
        response: np.memmap,
        source: np.memmap,
        closure: np.memmap,
        write: np.memmap,
    ):
        self.layers = layers
        self.response = response
        self.source = source
        self.closure = closure
        self.write = write
        self.anchors: torch.Tensor | None = None
        self.span_positions: torch.Tensor | None = None
        self.span_masks: torch.Tensor | None = None
        self.unit_indices: np.ndarray | None = None
        self.current: dict[int, dict[str, torch.Tensor]] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def begin(
        self,
        anchors: torch.Tensor,
        span_positions: torch.Tensor,
        span_masks: torch.Tensor,
        unit_indices: np.ndarray,
    ) -> None:
        self.anchors = anchors
        self.span_positions = span_positions
        self.span_masks = span_masks
        self.unit_indices = unit_indices
        self.current = {}
        self.counts = defaultdict(int)

    def _pre_hook(self, depth_index: int):
        def hook(module, args):
            hidden = args[0]
            if (
                self.anchors is None
                or self.span_positions is None
                or self.span_masks is None
            ):
                raise RuntimeError("capture context missing")
            anchor_states = grouped(
                gather_positions(hidden, self.anchors)
            ).detach()
            concepts = grouped(
                gather_span_means(
                    hidden, self.span_positions, self.span_masks
                )
            ).detach()
            source_a = concepts[:, 1, 0] - concepts[:, 0, 0]
            source_b = concepts[:, 1, 1] - concepts[:, 0, 1]
            source_relative = 0.5 * (source_a - source_b)
            state_scale = torch.linalg.vector_norm(
                concepts.float(), dim=-1
            ).mean(dim=(1, 2))
            source_strength = (
                torch.linalg.vector_norm(
                    source_relative.float(), dim=-1
                )
                / torch.clamp(state_scale, min=EPS)
            )
            query_leak = 0.5 * (
                torch.linalg.vector_norm(
                    (concepts[:, 2] - concepts[:, 0]).float(),
                    dim=-1,
                ).mean(dim=-1)
                + torch.linalg.vector_norm(
                    (concepts[:, 3] - concepts[:, 1]).float(),
                    dim=-1,
                ).mean(dim=-1)
            ) / torch.clamp(state_scale, min=EPS)
            pair_opposition = -safe_cos(source_a, source_b)
            source_metrics = torch.stack(
                [source_strength, query_leak, pair_opposition], dim=-1
            )
            self.source[
                self.unit_indices, depth_index, :
            ] = source_metrics.float().cpu().numpy()
            self.current[depth_index] = {
                "input": anchor_states,
                "source": source_relative,
            }
            self.counts[f"{depth_index}/pre"] += 1
        return hook

    def _response_metrics(
        self,
        states: torch.Tensor,
        source_relative: torch.Tensor,
    ) -> torch.Tensor:
        h00, h10, h01, h11 = (
            states[:, index] for index in range(4)
        )
        binding_q0 = h10 - h00
        binding_q1 = h11 - h01
        query_b0 = h01 - h00
        query_b1 = h11 - h10
        interaction = binding_q1 - binding_q0
        source = source_relative[:, None, :].expand_as(binding_q0)
        selected_alignment = 0.5 * (
            safe_cos(binding_q0, source)
            - safe_cos(binding_q1, source)
        )
        return torch.stack(
            [
                relative_norm(binding_q0, states),
                relative_norm(binding_q1, states),
                relative_norm(query_b0, states),
                relative_norm(query_b1, states),
                relative_norm(interaction, states),
                safe_cos(binding_q0, binding_q1),
                safe_cos(query_b0, query_b1),
                selected_alignment,
                safe_cos(interaction, source),
            ],
            dim=-1,
        )

    def _component_hook(self, depth_index: int, component_index: int):
        name = protocol.COMPONENTS[component_index]

        def hook(module, args, output):
            if self.anchors is None or self.unit_indices is None:
                raise RuntimeError("component context missing")
            current = self.current[depth_index]
            states = grouped(
                gather_positions(output_tensor(output), self.anchors)
            ).detach()
            metrics = self._response_metrics(
                states, current["source"]
            )
            self.response[
                self.unit_indices,
                depth_index,
                component_index,
                :,
                :,
            ] = metrics.float().cpu().numpy()
            current[name] = states
            self.counts[f"{depth_index}/{name}"] += 1
            return output
        return hook

    def _layer_hook(self, depth_index: int):
        def hook(module, args, output):
            if self.anchors is None or self.unit_indices is None:
                raise RuntimeError("layer context missing")
            current = self.current[depth_index]
            states = grouped(
                gather_positions(output_tensor(output), self.anchors)
            ).detach()
            metrics = self._response_metrics(
                states, current["source"]
            )
            self.response[
                self.unit_indices, depth_index, 2, :, :
            ] = metrics.float().cpu().numpy()

            input_states = current["input"]
            attention = current["attention"]
            mlp = current["mlp"]
            error = states - input_states - attention - mlp
            transition = states - input_states
            closure = (
                torch.linalg.vector_norm(error.float(), dim=-1)
                / torch.clamp(
                    torch.linalg.vector_norm(
                        transition.float(), dim=-1
                    ),
                    min=EPS,
                )
            ).mean(dim=1)
            self.closure[
                self.unit_indices, depth_index, :
            ] = closure.float().cpu().numpy()

            def interaction(value: torch.Tensor) -> torch.Tensor:
                return value[:, 3] - value[:, 2] - value[:, 1] + value[:, 0]

            residual_write = interaction(states) - interaction(input_states)
            denominator = torch.sum(
                residual_write.float() ** 2, dim=-1
            )
            for component_index, component in enumerate(
                (attention, mlp)
            ):
                component_interaction = interaction(component)
                alignment = safe_cos(
                    component_interaction, residual_write
                )
                fraction = torch.sum(
                    component_interaction.float()
                    * residual_write.float(),
                    dim=-1,
                ) / torch.clamp(denominator, min=EPS)
                values = torch.stack([alignment, fraction], dim=-1)
                self.write[
                    self.unit_indices,
                    depth_index,
                    component_index,
                    :,
                    :,
                ] = values.float().cpu().numpy()

            self.counts[f"{depth_index}/residual"] += 1
            del self.current[depth_index]
            return output
        return hook

    def register(self) -> None:
        for depth_index, layer in enumerate(self.layers):
            self.handles.append(
                layer.register_forward_pre_hook(
                    self._pre_hook(depth_index)
                )
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._component_hook(depth_index, 0)
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    self._component_hook(depth_index, 1)
                )
            )
            self.handles.append(
                layer.register_forward_hook(
                    self._layer_hook(depth_index)
                )
            )

    def end(self) -> None:
        expected = {}
        for depth_index in range(len(self.layers)):
            for name in ("pre", "attention", "mlp", "residual"):
                expected[f"{depth_index}/{name}"] = 1
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"component hook count drift: {dict(self.counts)}"
            )
        if self.current:
            raise RuntimeError(
                f"component state leak: {sorted(self.current)}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def finite_row(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "all_finite": bool(np.isfinite(array).all()),
        "finite_rate": float(np.isfinite(array).mean()),
    }


def group_indices(
    units: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {
        "all": list(range(len(units))),
        "template_0": [],
        "template_1": [],
        "bank_single": [],
        "bank_double": [],
    }
    for unit in units:
        index = int(unit["unit_index"])
        groups[f"template_{int(unit['template_index'])}"].append(index)
        groups[f"bank_{unit['bank_name']}"].append(index)
    return {
        name: np.asarray(values, dtype=np.int64)
        for name, values in groups.items()
    }


def summarize(
    response: np.ndarray,
    source: np.ndarray,
    closure: np.ndarray,
    write: np.ndarray,
    units: list[dict[str, Any]],
    n_layers: int,
) -> dict[str, Any]:
    groups = group_indices(units)
    response_index = {
        name: index
        for index, name in enumerate(protocol.RESPONSE_METRICS)
    }
    bin_rows = []
    for bin_index in range(protocol.DEPTH_BIN_COUNT):
        start = int(np.floor(bin_index * n_layers / protocol.DEPTH_BIN_COUNT))
        end = int(
            np.floor(
                (bin_index + 1)
                * n_layers
                / protocol.DEPTH_BIN_COUNT
            )
        )
        end = max(end, start + 1)
        for group_name, indices in groups.items():
            for component_index, component in enumerate(
                protocol.COMPONENTS
            ):
                for anchor_index, anchor in enumerate(protocol.ANCHORS):
                    values = np.asarray(
                        response[
                            indices,
                            start:end,
                            component_index,
                            anchor_index,
                            :,
                        ],
                        dtype=np.float32,
                    ).reshape(-1, len(protocol.RESPONSE_METRICS))
                    binding_cos = values[
                        :, response_index["binding_context_cosine"]
                    ]
                    row = {
                        "depth_bin": bin_index,
                        "depth_start": start + 1,
                        "depth_end": end,
                        "group": group_name,
                        "component": component,
                        "anchor": anchor,
                        "count": int(values.shape[0]),
                        "metrics": {},
                        "binding_context_negative_rate": float(
                            np.mean(binding_cos < 0)
                        ),
                    }
                    for metric_index, metric in enumerate(
                        protocol.RESPONSE_METRICS
                    ):
                        metric_values = values[:, metric_index]
                        row["metrics"][metric] = {
                            "mean": float(np.mean(metric_values)),
                            "median": float(np.median(metric_values)),
                        }
                    bin_rows.append(row)

    source_rows = []
    for group_name, indices in groups.items():
        values = np.asarray(source[indices], dtype=np.float32)
        source_rows.append(
            {
                "group": group_name,
                "metrics": {
                    metric: {
                        "mean": float(np.mean(values[..., index])),
                        "median": float(np.median(values[..., index])),
                    }
                    for index, metric in enumerate(
                        protocol.SOURCE_METRICS
                    )
                },
            }
        )
    closure_values = np.asarray(closure, dtype=np.float32).reshape(-1)
    write_values = np.asarray(write, dtype=np.float32)
    return {
        "depth_bins": bin_rows,
        "source_summary": source_rows,
        "instrumentation": {
            "residual_addition_relative_error": {
                "median": float(np.median(closure_values)),
                "p95": float(np.quantile(closure_values, 0.95)),
                "max": float(np.max(closure_values)),
            },
            "attention_signed_fraction_median": float(
                np.median(write_values[:, :, 0, :, 1])
            ),
            "mlp_signed_fraction_median": float(
                np.median(write_values[:, :, 1, :, 1])
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    cases = protocol.read_jsonl(
        protocol.SOURCE_ROOT
        / "protocol"
        / f"cases.{args.model}.jsonl"
    )
    units = protocol.read_jsonl(
        protocol.SOURCE_ROOT / "protocol" / "units.jsonl"
    )
    atlas_dir = protocol.OUT_ROOT / "atlas" / args.model
    atlas_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = None

    try:
        model, tokenizer, device, placement = load_fp16(args.model)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = get_layers(model)
        info = get_model_info(model, args.model)
        response = np.lib.format.open_memmap(
            atlas_dir / "component_response.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(units),
                info.n_layers,
                len(protocol.COMPONENTS),
                len(protocol.ANCHORS),
                len(protocol.RESPONSE_METRICS),
            ),
        )
        source_values = np.lib.format.open_memmap(
            atlas_dir / "source_relative.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(units),
                info.n_layers,
                len(protocol.SOURCE_METRICS),
            ),
        )
        closure = np.lib.format.open_memmap(
            atlas_dir / "residual_closure.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(units),
                info.n_layers,
                len(protocol.ANCHORS),
            ),
        )
        write = np.lib.format.open_memmap(
            atlas_dir / "component_write.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(units),
                info.n_layers,
                2,
                len(protocol.ANCHORS),
                len(protocol.WRITE_METRICS),
            ),
        )
        response[:] = 0
        source_values[:] = 0
        closure[:] = 0
        write[:] = 0

        capture = ComponentAtlasCapture(
            layers, response, source_values, closure, write
        )
        capture.register()
        try:
            for batch_number, row_batch in enumerate(
                chunks(cases, BATCH_SIZE[args.model]), 1
            ):
                (
                    input_ids,
                    attention_mask,
                    anchors,
                    span_positions,
                    span_masks,
                    unit_indices,
                ) = make_batch(
                    row_batch,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )
                capture.begin(
                    anchors, span_positions, span_masks, unit_indices
                )
                with torch.inference_mode():
                    output = model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                capture.end()
                del output
                if batch_number % 16 == 0:
                    print(
                        f"[phase1034] {args.model} "
                        f"units={int(unit_indices[-1]) + 1}/{len(units)}",
                        flush=True,
                    )
        finally:
            capture.close()
        for array in (response, source_values, closure, write):
            array.flush()

        metrics = summarize(
            response,
            source_values,
            closure,
            write,
            units,
            info.n_layers,
        )
        arrays = {
            "component_response": finite_row(response),
            "source_relative": finite_row(source_values),
            "residual_closure": finite_row(closure),
            "component_write": finite_row(write),
        }
        summary = {
            "schema_version": "phase1034_model_summary.v1",
            "phase": protocol.PHASE,
            "model": args.model,
            "protocol_digest": prereg["protocol_digest"],
            "source_protocol_digest": prereg[
                "source_protocol_digest"
            ],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "class": info.model_class,
                "n_layers": info.n_layers,
                "d_model": info.d_model,
            },
            "sample_counts": {
                "units": len(units),
                "cases": len(cases),
                "templates": 2,
                "banks": 2,
            },
            "arrays": arrays,
            "all_recorded_values_finite": all(
                row["all_finite"] for row in arrays.values()
            ),
            "instrumentation_gate_passed": (
                metrics["instrumentation"][
                    "residual_addition_relative_error"
                ]["p95"]
                <= prereg["instrumentation_gate"][
                    "residual_addition_relative_error_p95_max"
                ]
            ),
            "elapsed_seconds": time.time() - started,
        }
        protocol.write_json(atlas_dir / "metrics.json", metrics)
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    main()
