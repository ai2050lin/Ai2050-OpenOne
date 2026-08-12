#!/usr/bin/env python3
"""Observe real attention heads and MLP neuron writes in repeated regions."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import (
    MODEL_CONFIGS,
    get_layers,
    get_model_info,
    load_model,
    release_model,
)
from phase1008_global_response_atlas_protocol import (
    ANALYSIS_OPERATIONS,
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase1008_global_response_atlas_scan import (
    STATE_ORDER,
    case_tensors,
    operation_deltas,
    stage_case,
)


MODELS = ("qwen3", "glm4")
OPERATIONS = ("B", "Q", "BQ", "X")
GLOBAL_OP_INDEX = {
    name: index for index, name in enumerate(ANALYSIS_OPERATIONS)
}
LOCAL_OP_INDEX = {name: index for index, name in enumerate(OPERATIONS)}
EPSILON = 1e-12


def materialize_runtime_weight(
    weight: torch.Tensor,
    module=None,
) -> torch.Tensor:
    """Explicitly materialize the matrix used by the quantized runtime."""
    if weight.dtype == torch.int8 and hasattr(weight, "SCB"):
        import bitsandbytes.functional as bnb_functional

        state = None if module is None else getattr(module, "state", None)
        quantized = getattr(weight, "CB", None)
        if quantized is None and state is not None:
            quantized = getattr(state, "CB", None)
        if quantized is None:
            quantized = weight.data
        statistics = getattr(weight, "SCB", None)
        if statistics is None and state is not None:
            statistics = getattr(state, "SCB", None)
        if statistics is None:
            raise RuntimeError("8-bit weight is missing row statistics")
        return bnb_functional.int8_vectorwise_dequant(
            quantized, statistics
        ).detach().float()
    return weight.detach().float()


def original_weight(
    model_name: str,
    layer_number: int,
    component: str,
) -> torch.Tensor:
    """Read the original trained matrix, bypassing runtime int8 bytes."""
    from safetensors import safe_open

    model_root = Path(MODEL_CONFIGS[model_name]["path"])
    index_path = model_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    key = f"model.layers.{layer_number - 1}.{component}.weight"
    shard_name = index["weight_map"].get(key)
    if shard_name is None:
        raise RuntimeError(f"{model_name}: missing original weight {key}")
    with safe_open(
        str(model_root / shard_name),
        framework="pt",
        device="cpu",
    ) as handle:
        value = handle.get_tensor(key).float()
    return value


def state_cases_for_unit(
    unit: dict[str, Any],
    case_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    base = case_by_id[unit["case_ids"]["base"]]
    return [
        base,
        case_by_id[unit["case_ids"]["B"]],
        case_by_id[unit["case_ids"]["Q"]],
        case_by_id[unit["case_ids"]["BQ"]],
        case_by_id[unit["case_ids"]["E"]],
        case_by_id[unit["case_ids"]["O"]],
        case_by_id[unit["case_ids"]["N"]],
        dict(base),
    ]


class RefinementCapture:
    def __init__(
        self,
        layers,
        attention_layers: list[int],
        mlp_layers: list[int],
    ):
        self.layers = layers
        self.attention_layers = attention_layers
        self.mlp_layers = mlp_layers
        self.positions: torch.Tensor | None = None
        self.head_inputs: dict[int, torch.Tensor] = {}
        self.attention_outputs: dict[int, torch.Tensor] = {}
        self.mlp_activations: dict[int, torch.Tensor] = {}
        self.mlp_outputs: dict[int, torch.Tensor] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def _select(self, value: torch.Tensor) -> torch.Tensor:
        if self.positions is None:
            raise RuntimeError("refinement positions not set")
        positions = self.positions.to(value.device)
        batch = torch.arange(value.shape[0], device=value.device)
        return value[batch, positions, :].detach()

    def register(self) -> None:
        for layer_number in self.attention_layers:
            layer = self.layers[layer_number - 1]

            def make_head_hook(number):
                def hook(module, args):
                    self.head_inputs[number] = self._select(args[0])
                    self.counts[f"head/{number}"] += 1
                return hook

            def make_attention_hook(number):
                def hook(module, args, output):
                    value = output[0] if isinstance(output, tuple) else output
                    self.attention_outputs[number] = self._select(value)
                    self.counts[f"attention/{number}"] += 1
                    return output
                return hook

            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(layer_number)
                )
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    make_attention_hook(layer_number)
                )
            )
        for layer_number in self.mlp_layers:
            layer = self.layers[layer_number - 1]

            def make_activation_hook(number):
                def hook(module, args):
                    self.mlp_activations[number] = self._select(args[0])
                    self.counts[f"mlp_activation/{number}"] += 1
                return hook

            def make_mlp_hook(number):
                def hook(module, args, output):
                    value = output[0] if isinstance(output, tuple) else output
                    self.mlp_outputs[number] = self._select(value)
                    self.counts[f"mlp_output/{number}"] += 1
                    return output
                return hook

            self.handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    make_activation_hook(layer_number)
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    make_mlp_hook(layer_number)
                )
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.head_inputs = {}
        self.attention_outputs = {}
        self.mlp_activations = {}
        self.mlp_outputs = {}
        self.counts = defaultdict(int)

    def validate(self) -> None:
        expected = {
            *{f"head/{layer}" for layer in self.attention_layers},
            *{f"attention/{layer}" for layer in self.attention_layers},
            *{f"mlp_activation/{layer}" for layer in self.mlp_layers},
            *{f"mlp_output/{layer}" for layer in self.mlp_layers},
        }
        bad = {
            key: self.counts[key]
            for key in expected
            if self.counts[key] != 1
        }
        if bad:
            raise RuntimeError(f"refinement hook count drift: {bad}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.positions = None
        self.begin(torch.empty(0, dtype=torch.long))
        self.positions = None


def attention_grams(
    model_name: str,
    layers,
    layer_numbers: list[int],
    head_count: int,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, int],
]:
    runtime_result = {}
    reference_result = {}
    head_dims = {}
    for layer_number in layer_numbers:
        module = layers[layer_number - 1].self_attn.o_proj
        weight = materialize_runtime_weight(
            module.weight,
            module,
        )
        reference_weight = original_weight(
            model_name,
            layer_number,
            "self_attn.o_proj",
        )
        input_width = int(weight.shape[1])
        if input_width % head_count:
            raise RuntimeError(
                f"L{layer_number}: o_proj width {input_width} "
                f"not divisible by {head_count}"
            )
        head_dim = input_width // head_count
        reshaped = weight.reshape(weight.shape[0], head_count, head_dim)
        reference_reshaped = reference_weight.reshape(
            reference_weight.shape[0], head_count, head_dim
        )
        gram = torch.einsum(
            "ohd,ohe->hde", reshaped, reshaped
        ).detach().cpu().numpy().astype(np.float32)
        reference_gram = torch.einsum(
            "ohd,ohe->hde", reference_reshaped, reference_reshaped
        ).detach().cpu().numpy().astype(np.float32)
        runtime_result[layer_number] = gram
        reference_result[layer_number] = reference_gram
        head_dims[layer_number] = head_dim
        del (
            weight,
            reference_weight,
            reshaped,
            reference_reshaped,
            gram,
            reference_gram,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return runtime_result, reference_result, head_dims


def mlp_column_norms(
    model_name: str,
    layers,
    layer_numbers: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    runtime_result = {}
    reference_result = {}
    for layer_number in layer_numbers:
        module = layers[layer_number - 1].mlp.down_proj
        weight = materialize_runtime_weight(
            module.weight,
            module,
        )
        reference_weight = original_weight(
            model_name,
            layer_number,
            "mlp.down_proj",
        )
        runtime_result[layer_number] = (
            torch.linalg.vector_norm(weight, dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        reference_result[layer_number] = (
            torch.linalg.vector_norm(reference_weight, dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        del weight, reference_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return runtime_result, reference_result


def contribution_rank_audit(
    *,
    model_name: str,
    component: str,
    unit_id: str,
    operation: str,
    stage: str,
    role: str,
    layer: int,
    runtime_values: np.ndarray,
    reference_values: np.ndarray,
    top_fraction: float,
) -> dict[str, Any]:
    count = max(1, int(np.ceil(runtime_values.size * top_fraction)))
    runtime_top = set(
        np.argpartition(runtime_values, -count)[-count:].tolist()
    )
    reference_top = set(
        np.argpartition(reference_values, -count)[-count:].tolist()
    )
    union = runtime_top | reference_top
    jaccard = len(runtime_top & reference_top) / max(len(union), 1)
    runtime_std = float(np.std(runtime_values))
    reference_std = float(np.std(reference_values))
    if runtime_std <= EPSILON or reference_std <= EPSILON:
        correlation = float(
            np.allclose(runtime_values, reference_values)
        )
    else:
        correlation = float(
            np.corrcoef(runtime_values, reference_values)[0, 1]
        )
    return {
        "schema_version": "phase1008_dual_weight_rank_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "component": component,
        "unit_id": unit_id,
        "operation": operation,
        "stage": stage,
        "role": role,
        "layer": int(layer),
        "population_size": int(runtime_values.size),
        "top_fraction": float(top_fraction),
        "top_count": int(count),
        "top_set_jaccard": float(jaccard),
        "magnitude_correlation": correlation,
    }


def reconstruction_row(
    *,
    model_name: str,
    component: str,
    layer_number: int,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    module,
) -> dict[str, Any]:
    source_component = (
        "self_attn.o_proj"
        if component == "attention_o_proj"
        else "mlp.down_proj"
    )
    runtime_weight = materialize_runtime_weight(
        module.weight,
        module,
    ).to(inputs.device)
    reference_weight = original_weight(
        model_name,
        layer_number,
        source_component,
    ).to(inputs.device)
    bias = (
        None
        if getattr(module, "bias", None) is None
        else module.bias.detach().float()
    )
    runtime_predicted = torch.nn.functional.linear(
        inputs.float(),
        runtime_weight,
        bias,
    )
    reference_predicted = torch.nn.functional.linear(
        inputs.float(),
        reference_weight,
        bias,
    )
    scales = torch.linalg.vector_norm(outputs.float(), dim=-1)
    runtime_relative = torch.linalg.vector_norm(
        runtime_predicted - outputs.float(), dim=-1
    ) / torch.clamp(scales, min=EPSILON)
    reference_relative = torch.linalg.vector_norm(
        reference_predicted - outputs.float(), dim=-1
    ) / torch.clamp(scales, min=EPSILON)
    runtime_maximum = float(runtime_relative.max().item())
    reference_maximum = float(reference_relative.max().item())
    runtime_threshold = (
        0.06 if component == "attention_o_proj" else 0.13
    )
    row = {
        "schema_version": "phase1008_weight_reconstruction_audit.v2",
        "phase": PHASE,
        "model": model_name,
        "component": component,
        "layer": int(layer_number),
        "n": int(runtime_relative.numel()),
        "runtime_weight_source": "explicit_CB_SCB_dequantization",
        "runtime_mean_relative_error": float(
            runtime_relative.mean().item()
        ),
        "runtime_maximum_relative_error": runtime_maximum,
        "original_bf16_mean_relative_error": float(
            reference_relative.mean().item()
        ),
        "original_bf16_maximum_relative_error": reference_maximum,
        "runtime_gate_threshold": runtime_threshold,
        "original_reference_gate_threshold": 0.15,
        "gate_pass": bool(
            runtime_maximum <= runtime_threshold
            and reference_maximum <= 0.15
        ),
    }
    del (
        runtime_weight,
        reference_weight,
        runtime_predicted,
        reference_predicted,
        scales,
        runtime_relative,
        reference_relative,
    )
    return row


def run_model(
    model_name: str,
    *,
    limit_units: int | None = None,
    scope: str = "formal",
) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "refinement" / "protocol.json")
    model_spec = protocol["model_targets"][model_name]
    cases = read_jsonl(OUT_ROOT / "protocol" / model_name / "cases.jsonl")
    units = read_jsonl(OUT_ROOT / "protocol" / model_name / "units.jsonl")
    scan_units = read_jsonl(OUT_ROOT / "scan" / model_name / "units.jsonl")
    if [row["unit_id"] for row in units] != [
        row["unit_id"] for row in scan_units
    ]:
        raise RuntimeError("global/refinement unit ordering drift")
    if limit_units is not None:
        units = units[:limit_units]
        scan_units = scan_units[:limit_units]
    case_by_id = {case["record_id"]: case for case in cases}
    attention_layers = [
        int(value) for value in model_spec["attention"]["scan_layers"]
    ]
    mlp_targets = model_spec["mlp"]["targets"]
    mlp_layers = sorted({int(row["layer"]) for row in mlp_targets})
    output_root = (
        OUT_ROOT
        / ("refinement_scan" if scope == "formal" else "refinement_smoke")
        / model_name
    )
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = device = capture = None
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        head_count = int(model.config.num_attention_heads)
        grams, reference_grams, head_dims = attention_grams(
            model_name, layers, attention_layers, head_count
        )
        column_norms, reference_column_norms = mlp_column_norms(
            model_name, layers, mlp_layers
        )
        intermediate_size = int(info.intermediate_size)
        if any(
            len(column_norms[layer]) != intermediate_size
            for layer in mlp_layers
        ):
            raise RuntimeError("MLP intermediate width drift")

        unit_count = len(units)
        operation_count = len(OPERATIONS)
        head_write = np.full(
            (
                unit_count,
                operation_count,
                len(attention_layers),
                head_count,
            ),
            np.nan,
            dtype=np.float32,
        )
        head_fraction = np.full_like(head_write, np.nan)
        attention_full_norm = np.full(
            (unit_count, operation_count, len(attention_layers)),
            np.nan,
            dtype=np.float32,
        )
        neuron_write = np.full(
            (
                len(mlp_targets),
                unit_count,
                operation_count,
                intermediate_size,
            ),
            np.nan,
            dtype=np.float32,
        )
        mlp_full_norm = np.full(
            (len(mlp_targets), unit_count, operation_count),
            np.nan,
            dtype=np.float32,
        )
        semantic_qualified = np.array([
            [
                bool(unit["semantic_qualified"][operation])
                for operation in OPERATIONS
            ]
            for unit in scan_units
        ], dtype=np.bool_)
        rollout_qualified = np.array([
            [
                bool(unit["rollout_qualified"][operation])
                for operation in OPERATIONS
            ]
            for unit in scan_units
        ], dtype=np.bool_)
        capture = RefinementCapture(
            layers, attention_layers, mlp_layers
        )
        capture.register()
        reconstruction_rows: list[dict[str, Any]] = []
        rank_audit_rows: list[dict[str, Any]] = []
        targets_by_stage: dict[str, list[tuple[int, dict[str, Any]]]] = (
            defaultdict(list)
        )
        for target_index, target in enumerate(mlp_targets):
            targets_by_stage[target["stage"]].append((target_index, target))

        for unit_index, unit in enumerate(units):
            state_cases = state_cases_for_unit(unit, case_by_id)
            for stage in ("prompt", "semantic0"):
                staged = [stage_case(case, stage) for case in state_cases]
                role = (
                    "answer_boundary"
                    if stage == "prompt"
                    else "decision_boundary"
                )
                positions = torch.tensor(
                    [
                        int(case["scan_role_positions"][role])
                        for case in staged
                    ],
                    dtype=torch.long,
                    device=device,
                )
                input_ids, attention = case_tensors(staged, device)
                capture.begin(positions)
                try:
                    with torch.inference_mode():
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention,
                            use_cache=False,
                            return_dict=True,
                        )
                    capture.validate()
                    if unit_index == 0:
                        if stage == "semantic0":
                            for layer_number in attention_layers:
                                reconstruction_rows.append(
                                    reconstruction_row(
                                        model_name=model_name,
                                        component="attention_o_proj",
                                        layer_number=layer_number,
                                        inputs=capture.head_inputs[layer_number],
                                        outputs=capture.attention_outputs[
                                            layer_number
                                        ],
                                        module=layers[
                                            layer_number - 1
                                        ].self_attn.o_proj,
                                    )
                                )
                        for _, target in targets_by_stage[stage]:
                            layer_number = int(target["layer"])
                            reconstruction_rows.append(
                                reconstruction_row(
                                    model_name=model_name,
                                    component="mlp_down_proj",
                                    layer_number=layer_number,
                                    inputs=capture.mlp_activations[
                                        layer_number
                                    ],
                                    outputs=capture.mlp_outputs[
                                        layer_number
                                    ],
                                    module=layers[
                                        layer_number - 1
                                    ].mlp.down_proj,
                                )
                            )
                    if stage == "semantic0":
                        for layer_offset, layer_number in enumerate(
                            attention_layers
                        ):
                            values = capture.head_inputs[layer_number]
                            head_dim = head_dims[layer_number]
                            values = values.reshape(
                                len(STATE_ORDER), head_count, head_dim
                            )
                            deltas = operation_deltas(values)
                            full_deltas = operation_deltas(
                                capture.attention_outputs[layer_number]
                            )
                            gram = grams[layer_number]
                            reference_gram = reference_grams[layer_number]
                            for operation in OPERATIONS:
                                operation_offset = LOCAL_OP_INDEX[operation]
                                delta = (
                                    deltas[operation]
                                    .float()
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                                squared = np.einsum(
                                    "hd,hde,he->h",
                                    delta,
                                    gram,
                                    delta,
                                    optimize=True,
                                )
                                write = np.sqrt(np.maximum(squared, 0.0))
                                reference_squared = np.einsum(
                                    "hd,hde,he->h",
                                    delta,
                                    reference_gram,
                                    delta,
                                    optimize=True,
                                )
                                reference_write = np.sqrt(
                                    np.maximum(reference_squared, 0.0)
                                )
                                full_norm = float(torch.linalg.vector_norm(
                                    full_deltas[operation].float()
                                ).item())
                                head_write[
                                    unit_index,
                                    operation_offset,
                                    layer_offset,
                                ] = write
                                head_fraction[
                                    unit_index,
                                    operation_offset,
                                    layer_offset,
                                ] = write / max(full_norm, EPSILON)
                                attention_full_norm[
                                    unit_index,
                                    operation_offset,
                                    layer_offset,
                                ] = full_norm
                                rank_audit_rows.append(
                                    contribution_rank_audit(
                                        model_name=model_name,
                                        component="attention_head",
                                        unit_id=unit["unit_id"],
                                        operation=operation,
                                        stage="semantic0",
                                        role="decision_boundary",
                                        layer=layer_number,
                                        runtime_values=write,
                                        reference_values=reference_write,
                                        top_fraction=0.25,
                                    )
                                )
                    for target_index, target in targets_by_stage[stage]:
                        layer_number = int(target["layer"])
                        activations = capture.mlp_activations[layer_number]
                        activation_deltas = operation_deltas(activations)
                        output_deltas = operation_deltas(
                            capture.mlp_outputs[layer_number]
                        )
                        norms = column_norms[layer_number]
                        reference_norms = reference_column_norms[
                            layer_number
                        ]
                        for operation in OPERATIONS:
                            operation_offset = LOCAL_OP_INDEX[operation]
                            delta = (
                                activation_deltas[operation]
                                .float()
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            write = np.abs(delta) * norms
                            reference_write = (
                                np.abs(delta) * reference_norms
                            )
                            full_norm = float(torch.linalg.vector_norm(
                                output_deltas[operation].float()
                            ).item())
                            neuron_write[
                                target_index,
                                unit_index,
                                operation_offset,
                            ] = write
                            mlp_full_norm[
                                target_index,
                                unit_index,
                                operation_offset,
                            ] = full_norm
                            rank_audit_rows.append(
                                contribution_rank_audit(
                                    model_name=model_name,
                                    component="mlp_neuron",
                                    unit_id=unit["unit_id"],
                                    operation=operation,
                                    stage=stage,
                                    role=target["role"],
                                    layer=layer_number,
                                    runtime_values=write,
                                    reference_values=reference_write,
                                    top_fraction=0.01,
                                )
                            )
                    del output
                finally:
                    del input_ids, attention, positions
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if (unit_index + 1) % 4 == 0 or unit_index + 1 == unit_count:
                print(
                    f"[refinement] {model_name} "
                    f"{unit_index + 1}/{unit_count}",
                    flush=True,
                )

        np.savez_compressed(
            output_root / "head_observations.npz",
            write_magnitude=head_write,
            contribution_fraction=head_fraction,
            full_attention_delta_norm=attention_full_norm,
            semantic_qualified=semantic_qualified,
            rollout_qualified=rollout_qualified,
        )
        np.savez_compressed(
            output_root / "neuron_observations.npz",
            write_magnitude=neuron_write,
            full_mlp_delta_norm=mlp_full_norm,
            semantic_qualified=semantic_qualified,
            rollout_qualified=rollout_qualified,
        )
        write_jsonl(output_root / "units.jsonl", scan_units)
        write_jsonl(
            output_root / "attention_targets.jsonl",
            [
                {
                    "schema_version": "phase1008_attention_target.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "target_index": index,
                    "stage": "semantic0",
                    "role": "decision_boundary",
                    "layer": layer,
                    "head_count": head_count,
                    "head_dim": head_dims[layer],
                }
                for index, layer in enumerate(attention_layers)
            ],
        )
        write_jsonl(
            output_root / "mlp_targets.jsonl",
            [
                {
                    "schema_version": "phase1008_mlp_target.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "target_index": index,
                    "stage": target["stage"],
                    "role": target["role"],
                    "layer": int(target["layer"]),
                    "intermediate_size": intermediate_size,
                }
                for index, target in enumerate(mlp_targets)
            ],
        )
        write_jsonl(
            output_root / "weight_reconstruction_audit.jsonl",
            reconstruction_rows,
        )
        write_jsonl(
            output_root / "dual_weight_rank_audit.jsonl",
            rank_audit_rows,
        )
        if not reconstruction_rows or not all(
            row["gate_pass"] for row in reconstruction_rows
        ):
            raise RuntimeError(
                f"{model_name}: weight reconstruction instrument failed"
            )
        head_rank_rows = [
            row for row in rank_audit_rows
            if row["component"] == "attention_head"
        ]
        neuron_rank_rows = [
            row for row in rank_audit_rows
            if row["component"] == "mlp_neuron"
        ]
        rank_gate = {
            "attention_median_jaccard": float(np.median([
                row["top_set_jaccard"] for row in head_rank_rows
            ])),
            "attention_minimum_jaccard": float(min(
                row["top_set_jaccard"] for row in head_rank_rows
            )),
            "mlp_median_jaccard": float(np.median([
                row["top_set_jaccard"] for row in neuron_rank_rows
            ])),
            "mlp_minimum_jaccard": float(min(
                row["top_set_jaccard"] for row in neuron_rank_rows
            )),
            "median_magnitude_correlation": float(np.median([
                row["magnitude_correlation"] for row in rank_audit_rows
            ])),
        }
        rank_gate["all_pass"] = bool(
            rank_gate["attention_median_jaccard"] >= 0.75
            and rank_gate["attention_minimum_jaccard"] >= 0.45
            and rank_gate["mlp_median_jaccard"] >= 0.90
            and rank_gate["mlp_minimum_jaccard"] >= 0.75
            and rank_gate["median_magnitude_correlation"] >= 0.99
        )
        if not rank_gate["all_pass"]:
            raise RuntimeError(
                f"{model_name}: dual-weight ranking instrument failed: "
                f"{rank_gate}"
            )
        summary = {
            "schema_version": "phase1008_refinement_scan_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "scope": scope,
            "refinement_protocol_digest": protocol["preregistration_digest"],
            "unit_count": unit_count,
            "operations": list(OPERATIONS),
            "attention_layers": attention_layers,
            "head_count": head_count,
            "head_observation_count": int(head_write.size),
            "mlp_targets": mlp_targets,
            "intermediate_size": intermediate_size,
            "neuron_observation_count": int(neuron_write.size),
            "raw_head_vectors_persisted": 0,
            "raw_neuron_activations_persisted": 0,
            "causal_interventions_performed": 0,
            "weight_reconstruction_audit": {
                "row_count": len(reconstruction_rows),
                "all_pass": True,
                "maximum_runtime_relative_error": max(
                    row["runtime_maximum_relative_error"]
                    for row in reconstruction_rows
                ),
                "maximum_original_bf16_relative_error": max(
                    row["original_bf16_maximum_relative_error"]
                    for row in reconstruction_rows
                ),
            },
            "dual_weight_rank_audit": rank_gate,
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_model(model)
        model = tokenizer = device = capture = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--scope", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--limit-units", type=int)
    args = parser.parse_args()
    limit = args.limit_units
    if args.scope == "smoke" and limit is None:
        limit = 2
    run_model(args.model, limit_units=limit, scope=args.scope)


if __name__ == "__main__":
    main()
