#!/usr/bin/env python3
"""Record the authorized Phase429 architecture path without head/neuron scans."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase429_typed_route_analysis import wilson  # noqa: E402
from phase429_typed_route_protocol import MODELS, OUT, SCHEMA_VERSION  # noqa: E402


PHASE_ID = "Phase429-ArchitecturePhysical"
PHYSICAL_SCHEMA = "phase429_architecture_physical.v1"
VIS = ROOT / "frontend/public/vis_data/phase429_architecture_path"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
BATCH_SIZE = {"qwen3": 4, "glm4": 1, "deepseek7b": 2}
CHECKPOINT_CONDITIONS = 32
QUERY_LINE = "Question: Which item is selected?"
COMPONENTS = (
    "residual_pre",
    "query_projection",
    "source_key_projection",
    "source_value_projection",
    "attention_write",
    "mlp_write",
    "residual_post",
)
COMPONENT_COLORS = {
    "residual_pre": "#64748b",
    "query_projection": "#06b6d4",
    "source_key_projection": "#22c55e",
    "source_value_projection": "#10b981",
    "attention_write": "#f59e0b",
    "mlp_write": "#ef4444",
    "residual_post": "#8b5cf6",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def digest_rows(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase429 physical non-finite scalar: {value}")
    return round(float(value), 10)


def median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.median(rows)) if rows else 0.0


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.fmean(rows)) if rows else 0.0


def freeze_physical_protocol() -> dict[str, Any]:
    path = OUT / "phase429_physical_protocol.json"
    if path.exists():
        protocol = read_json(path)
        if protocol.get("schema_version") == PHYSICAL_SCHEMA and protocol.get("valid"):
            return protocol
    open_gate = read_json(OUT / "phase429_open_behavior_gate.json")
    authorized = open_gate["authorized_candidates"]
    if not authorized:
        raise RuntimeError("Phase429 open behavior gate did not authorize physical recording")
    rows = []
    authorized_models = sorted({row["model"] for row in authorized})
    for model in authorized_models:
        materialized = read_jsonl(
            OUT / "models" / model / "behavior" / "phase429_materialized_conditions.jsonl"
        )
        allowed = {
            (row["block_id"], row["contract_variant"])
            for row in authorized
            if row["model"] == model
        }
        controls = {
            (row["matched_control_block_id"], row["contract_variant"])
            for row in authorized
            if row["model"] == model
        }
        rows.extend(
            row
            for row in materialized
            if (row["block_id"], row["contract_variant"]) in allowed | controls
            and row["split"] in {"behavior_calibration", "behavior_holdout"}
        )
    rows = sorted(rows, key=lambda row: row["condition_id"])
    condition_ids = {row["condition_id"] for row in rows}
    expected = len(authorized) * 2 * 2 * 96 * 2 * 5
    valid = bool(
        len(rows) == expected
        and len(condition_ids) == len(rows)
        and all(not row["physical"] and not row["causal"] for row in rows)
        and all(row["contract_variant"] == "no_examples" for row in rows)
    )
    if not valid:
        raise RuntimeError(
            json.dumps(
                {"row_count": len(rows), "expected": expected, "unique": len(condition_ids)},
                ensure_ascii=False,
                indent=2,
            )
        )
    write_jsonl(OUT / "phase429_physical_conditions_open.jsonl", rows)
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {
        "schema_version": PHYSICAL_SCHEMA,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "valid": valid,
        "authorized_candidates": authorized,
        "authorized_models": authorized_models,
        "condition_count": len(rows),
        "independent_group_count": len({row["semantic_group_id"] for row in rows}),
        "condition_rows_sha256": digest_rows(rows),
        "splits": ["behavior_calibration", "behavior_holdout"],
        "routes": ["none", "source_only", "query_only", "consistent", "conflict"],
        "components": list(COMPONENTS),
        "record_contract": {
            "all_transformer_layers": True,
            "source_positions": "two registered result-item spans",
            "query_position": "last token of the registered question span",
            "residual_conservation": "post - pre = attention_write + mlp_write",
            "qkv_summary": "whole projection-vector RMS; no head or channel index",
            "output_readout": "registered target-versus-opposite first-token margin",
            "head_channel_neuron_scan": False,
            "intervention": False,
        },
        "prediction_contract": {
            "calibration_fit": "none; zero-sign terminal readout fixed by architecture",
            "target": "natural target-first event on candidate conditions",
            "independent_unit": "semantic group",
            "group_success": "at least 8 of 10 paired role-route conditions predicted",
            "holdout_group_success_lcb_min": 0.70,
            "majority_baseline_group_improvement_min": 0.10,
            "reconstruction_relative_error_median_max": 0.01,
        },
        "sealed_contract": {
            "main_behavior_collector_sealed_path_authorized": False,
            "reason": "use an exact-tuple physical subprotocol so controls cannot widen to another contract variant",
            "sealed_read_requires_open_reconstruction_and_prediction": True,
            "sealed_group_commitment": read_json(OUT / "phase429_sealed_commitment.json"),
        },
        "evidence_contract": {
            "physical": True,
            "observer": True,
            "predictive_only_if_holdout_gate_passes": True,
            "causal": False,
            "strict_double_blind": False,
            "single_neuron": False,
        },
        "implementation_sha256": implementation_hash,
    }
    write_json(path, protocol)
    return protocol


def token_positions(
    rendered: str, offsets: list[tuple[int, int]], value: str, start_at: int = 0
) -> list[int]:
    left = rendered.find(value, start_at)
    if left < 0:
        raise RuntimeError(f"Physical span not found: {value!r}")
    right = left + len(value)
    return [
        index
        for index, (token_left, token_right) in enumerate(offsets)
        if token_right > token_left and token_left < right and token_right > left
    ]


def registered_positions(fast_tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    encoded = fast_tokenizer(
        row["rendered_prompt"], add_special_tokens=False, return_offsets_mapping=True
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    first_positions = token_positions(row["rendered_prompt"], offsets, row["first_item"])
    second_positions = token_positions(row["rendered_prompt"], offsets, row["second_item"])
    query_positions = token_positions(row["rendered_prompt"], offsets, QUERY_LINE)
    if not first_positions or not second_positions or not query_positions:
        raise RuntimeError(f"Empty Phase429 physical positions: {row['condition_id']}")
    return {
        "input_ids": ids,
        "source_positions": sorted(set(first_positions + second_positions)),
        "query_position": query_positions[-1],
    }


def padded_batch(
    positioned: list[dict[str, Any]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], list[int]]:
    width = max(len(row["input_ids"]) for row in positioned)
    input_ids = torch.full(
        (len(positioned), width), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    source_positions = []
    query_positions = []
    for index, row in enumerate(positioned):
        ids = row["input_ids"]
        pad = width - len(ids)
        input_ids[index, pad:] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[index, pad:] = 1
        source_positions.append([pad + value for value in row["source_positions"]])
        query_positions.append(pad + int(row["query_position"]))
    return input_ids, attention_mask, source_positions, query_positions


def rms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(tensor.float() * tensor.float(), dim=-1).clamp_min(1e-20))


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum(left.float() * right.float(), dim=-1)
    denominator = torch.linalg.vector_norm(left.float(), dim=-1) * torch.linalg.vector_norm(right.float(), dim=-1)
    return numerator / denominator.clamp_min(1e-20)


def trace_batch(
    loaded: Any,
    fast_tokenizer: Any,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    positioned = [registered_positions(fast_tokenizer, row) for row in rows]
    slow_ids = [
        [
            int(value)
            for value in loaded.tokenizer(row["rendered_prompt"], add_special_tokens=False)[
                "input_ids"
            ]
        ]
        for row in rows
    ]
    if any(item["input_ids"] != slow for item, slow in zip(positioned, slow_ids)):
        raise RuntimeError("Fast and execution tokenizers disagree in Phase429 physical trace")
    pad_id = int(loaded.tokenizer.pad_token_id)
    input_ids, attention_mask, source_positions, query_positions = padded_batch(
        positioned, pad_id, loaded.input_device
    )
    layers = get_layers(loaded.model)
    captures: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    hooks = []

    def query_vectors(tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [tensor[index, position] for index, position in enumerate(query_positions)]
        ).detach()

    def source_vectors(tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                tensor[index, positions].mean(dim=0)
                for index, positions in enumerate(source_positions)
            ]
        ).detach()

    for layer_index, layer in enumerate(layers):
        hooks.append(
            layer.register_forward_pre_hook(
                lambda module, args, index=layer_index: captures[index].__setitem__(
                    "pre", query_vectors(args[0])
                )
            )
        )
        hooks.append(
            layer.self_attn.q_proj.register_forward_hook(
                lambda module, args, output, index=layer_index: captures[index].__setitem__(
                    "q", query_vectors(output)
                )
            )
        )
        hooks.append(
            layer.self_attn.k_proj.register_forward_hook(
                lambda module, args, output, index=layer_index: captures[index].__setitem__(
                    "k", source_vectors(output)
                )
            )
        )
        hooks.append(
            layer.self_attn.v_proj.register_forward_hook(
                lambda module, args, output, index=layer_index: captures[index].__setitem__(
                    "v", source_vectors(output)
                )
            )
        )
        hooks.append(
            layer.self_attn.register_forward_hook(
                lambda module, args, output, index=layer_index: captures[index].__setitem__(
                    "attention", query_vectors(output[0] if isinstance(output, tuple) else output)
                )
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                lambda module, args, output, index=layer_index: captures[index].__setitem__(
                    "mlp", query_vectors(output)
                )
            )
        )
        hooks.append(
            layer.register_forward_hook(
                lambda module, args, output, index=layer_index: captures[index].__setitem__(
                    "post", query_vectors(output[0] if isinstance(output, tuple) else output)
                )
            )
        )
    try:
        base = loaded.model.model if hasattr(loaded.model, "model") else loaded.model.transformer
        with torch.inference_mode():
            result = base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        del result
    finally:
        for hook in hooks:
            hook.remove()
    target_ids = torch.tensor(
        [int(row["target_sequence_token_ids"][0]) for row in rows],
        dtype=torch.long,
        device=loaded.input_device,
    )
    opposite_ids = torch.tensor(
        [int(row["opposite_sequence_token_ids"][0]) for row in rows],
        dtype=torch.long,
        device=loaded.input_device,
    )
    final_norm = loaded.model.model.norm
    output_weight = loaded.model.lm_head.weight
    output_bias = getattr(loaded.model.lm_head, "bias", None)
    trace_rows = []
    for layer_index in range(len(layers)):
        capture = captures[layer_index]
        expected_keys = {"pre", "q", "k", "v", "attention", "mlp", "post"}
        if set(capture) != expected_keys:
            raise RuntimeError(f"Missing Phase429 physical hooks at layer {layer_index}: {set(capture)}")
        transition = capture["post"] - capture["pre"]
        reconstructed = capture["attention"] + capture["mlp"]
        error = transition - reconstructed
        normalized = final_norm(capture["post"])
        target_logits = torch.sum(normalized.float() * output_weight[target_ids].float(), dim=-1)
        opposite_logits = torch.sum(normalized.float() * output_weight[opposite_ids].float(), dim=-1)
        if output_bias is not None:
            target_logits = target_logits + output_bias[target_ids].float()
            opposite_logits = opposite_logits + output_bias[opposite_ids].float()
        metrics = {
            "residual_pre_rms": rms(capture["pre"]),
            "query_projection_rms": rms(capture["q"]),
            "source_key_projection_rms": rms(capture["k"]),
            "source_value_projection_rms": rms(capture["v"]),
            "attention_write_rms": rms(capture["attention"]),
            "mlp_write_rms": rms(capture["mlp"]),
            "residual_post_rms": rms(capture["post"]),
            "transition_rms": rms(transition),
            "reconstruction_error_rms": rms(error),
            "attention_mlp_cosine": cosine(capture["attention"], capture["mlp"]),
            "transition_attention_cosine": cosine(transition, capture["attention"]),
            "transition_mlp_cosine": cosine(transition, capture["mlp"]),
            "target_first_token_margin": target_logits - opposite_logits,
        }
        relative = metrics["reconstruction_error_rms"] / metrics["transition_rms"].clamp_min(1e-20)
        for batch_index, row in enumerate(rows):
            trace_rows.append(
                {
                    "schema_version": PHYSICAL_SCHEMA,
                    "phase_id": PHASE_ID,
                    "created_at": now(),
                    "model": loaded.key,
                    "condition_id": row["condition_id"],
                    "semantic_group_id": row["semantic_group_id"],
                    "block_id": row["block_id"],
                    "family_id": row["family_id"],
                    "mechanism_id": row["mechanism_id"],
                    "candidate": row["candidate"],
                    "matched_control_block_id": row["matched_control_block_id"],
                    "contract_variant": row["contract_variant"],
                    "split": row["split"],
                    "route_mode": row["route_mode"],
                    "role": row["role"],
                    "interface": row["interface"],
                    "layer": layer_index,
                    "layer_fraction": clean(layer_index / max(1, len(layers) - 1)),
                    "source_token_count": len(positioned[batch_index]["source_positions"]),
                    "query_position": positioned[batch_index]["query_position"],
                    **{
                        key: clean(value[batch_index].item())
                        for key, value in metrics.items()
                    },
                    "reconstruction_relative_error": clean(relative[batch_index].item()),
                    "physical": True,
                    "observer": True,
                    "native_compute_event": True,
                    "compute_edge": False,
                    "predictive": False,
                    "causal": False,
                    "pipeline_sealed": False,
                    "strict_double_blind": False,
                    "single_neuron": False,
                }
            )
    del input_ids, attention_mask, captures
    return trace_rows


def run_open_model(model: str) -> dict[str, Any]:
    protocol = freeze_physical_protocol()
    if model not in protocol["authorized_models"]:
        raise RuntimeError(f"Phase429 physical model not authorized: {model}")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != protocol["implementation_sha256"]:
        raise RuntimeError("Phase429 physical implementation changed after freeze")
    model_root = OUT / "physical" / "open" / model
    complete_path = model_root / "phase429_physical_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        complete = read_json(complete_path)
        print(json.dumps(complete, ensure_ascii=False, indent=2))
        return complete
    conditions = [
        row
        for row in read_jsonl(OUT / "phase429_physical_conditions_open.jsonl")
        if row["model"] == model
    ]
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        actual_dtype = str(next(loaded.model.parameters()).dtype).removeprefix("torch.")
        fast_tokenizer = AutoTokenizer.from_pretrained(
            str(loaded.spec.local_dir),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True,
        )
        checkpoint_root = model_root / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        layer_count = len(get_layers(loaded.model))
        existing_parts = sorted(checkpoint_root.glob("phase429_physical_part_*.jsonl"))
        existing = [row for path in existing_parts for row in read_jsonl(path)]
        counts = Counter(row["condition_id"] for row in existing)
        completed_ids = {condition_id for condition_id, count in counts.items() if count == layer_count}
        pending = [row for row in conditions if row["condition_id"] not in completed_ids]
        part_number = len(existing_parts)
        buffer: list[dict[str, Any]] = []
        processed = len(completed_ids)
        print(
            f"[Phase429 physical] loading {model}; conditions={len(conditions)}; "
            f"pending={len(pending)}; layers={layer_count}",
            flush=True,
        )
        for start in range(0, len(pending), BATCH_SIZE[model]):
            batch = pending[start : start + BATCH_SIZE[model]]
            buffer.extend(trace_batch(loaded, fast_tokenizer, batch))
            processed += len(batch)
            if processed % CHECKPOINT_CONDITIONS < len(batch) or processed == len(conditions):
                write_jsonl(
                    checkpoint_root / f"phase429_physical_part_{part_number:05d}.jsonl",
                    buffer,
                )
                part_number += 1
                buffer.clear()
            if processed == len(batch) or processed % 256 < len(batch):
                allocated, reserved = vram_gb()
                print(
                    f"[Phase429 physical] {model} {processed}/{len(conditions)}; "
                    f"VRAM={allocated:.2f}/{reserved:.2f} GiB",
                    flush=True,
                )
        if buffer:
            write_jsonl(
                checkpoint_root / f"phase429_physical_part_{part_number:05d}.jsonl",
                buffer,
            )
        rows = [
            row
            for path in sorted(checkpoint_root.glob("phase429_physical_part_*.jsonl"))
            for row in read_jsonl(path)
        ]
        unique = {(row["condition_id"], row["layer"]): row for row in rows}
        rows = [unique[key] for key in sorted(unique)]
        expected_rows = len(conditions) * layer_count
        if len(rows) != expected_rows:
            raise RuntimeError(f"Phase429 physical incomplete: {len(rows)} != {expected_rows}")
        write_jsonl(model_root / "phase429_physical_rows.jsonl", rows)
        complete = {
            "schema_version": PHYSICAL_SCHEMA,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model,
            "execution_dtype": actual_dtype,
            "condition_count": len(conditions),
            "trace_row_count": len(rows),
            "layer_count": layer_count,
            "independent_group_count": len({row["semantic_group_id"] for row in conditions}),
            "finite": all(
                math.isfinite(float(row["reconstruction_relative_error"])) for row in rows
            ),
            "all_rows_complete": len(rows) == expected_rows,
            "elapsed_seconds": clean(time.monotonic() - started),
            "head_channel_neuron_scan": False,
            "intervention": False,
            "causal_tested": False,
            "sealed_read": False,
        }
        write_json(complete_path, complete)
        print(json.dumps(complete, ensure_ascii=False, indent=2))
        return complete
    finally:
        if loaded is not None:
            release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def publish_visual(
    model: str, rows: list[dict[str, Any]], prediction_gate: dict[str, Any]
) -> None:
    selected = [
        row
        for row in rows
        if row["candidate"]
        and row["split"] == "behavior_holdout"
        and row["route_mode"] == "consistent"
    ]
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    field_by_component = {
        "residual_pre": "residual_pre_rms",
        "query_projection": "query_projection_rms",
        "source_key_projection": "source_key_projection_rms",
        "source_value_projection": "source_value_projection_rms",
        "attention_write": "attention_write_rms",
        "mlp_write": "mlp_write_rms",
        "residual_post": "residual_post_rms",
    }
    for row in selected:
        for component in COMPONENTS:
            grouped[(int(row["layer"]), component)].append(row)
    nodes = []
    edges = []
    component_index = {component: index for index, component in enumerate(COMPONENTS)}
    layer_count = max(int(row["layer"]) for row in selected) + 1
    for (layer, component), values in sorted(grouped.items()):
        node_id = f"phase429:{model}:layer{layer}:{component}"
        score = mean(row[field_by_component[component]] for row in values)
        nodes.append(
            {
                "id": node_id,
                "label": f"L{layer} / {component}",
                "type": "architecture_component_trace",
                "model": model,
                "layer": layer,
                "layer_fraction": clean(layer / max(1, layer_count - 1)),
                "component": component,
                "block_id": values[0]["block_id"],
                "contract_variant": values[0]["contract_variant"],
                "route_mode": "consistent",
                "mean_rms": score,
                "reconstruction_relative_error_median": median(
                    row["reconstruction_relative_error"] for row in values
                ),
                "score": score,
                "size": 0.48,
                "color": COMPONENT_COLORS[component],
                "position": [float((component_index[component] - 3) * 7), float(layer * 3), 0.0],
                "physical": True,
                "observer": True,
                "native_compute_event": True,
                "compute_edge": False,
                "predictive": bool(component == "residual_post" and layer == layer_count - 1 and prediction_gate["prediction_gate_pass"]),
                "causal": False,
                "pipeline_sealed": False,
                "strict_double_blind": False,
                "single_neuron": False,
                "evidence_level": "architecture_physical_observation",
                "show_label": layer % max(1, layer_count // 6) == 0,
            }
        )
    within = (
        ("residual_pre", "query_projection"),
        ("query_projection", "attention_write"),
        ("source_key_projection", "attention_write"),
        ("source_value_projection", "attention_write"),
        ("attention_write", "residual_post"),
        ("mlp_write", "residual_post"),
    )
    for layer in range(layer_count):
        for source_component, target_component in within:
            source = f"phase429:{model}:layer{layer}:{source_component}"
            target = f"phase429:{model}:layer{layer}:{target_component}"
            edges.append(
                {
                    "id": f"{source}->{target}",
                    "source": source,
                    "target": target,
                    "type": "registered_architecture_compute_edge",
                    "physical": True,
                    "observer": True,
                    "compute_edge": True,
                    "predictive": False,
                    "causal": False,
                    "pipeline_sealed": False,
                    "strict_double_blind": False,
                    "single_neuron": False,
                    "evidence_level": "architecture_identity",
                    "color": "#64748b",
                    "weight": 0.7,
                }
            )
        if layer + 1 < layer_count:
            source = f"phase429:{model}:layer{layer}:residual_post"
            target = f"phase429:{model}:layer{layer + 1}:residual_pre"
            edges.append(
                {
                    "id": f"{source}->{target}",
                    "source": source,
                    "target": target,
                    "type": "residual_transport",
                    "physical": True,
                    "observer": True,
                    "compute_edge": True,
                    "predictive": False,
                    "causal": False,
                    "pipeline_sealed": False,
                    "strict_double_blind": False,
                    "single_neuron": False,
                    "evidence_level": "architecture_identity",
                    "color": "#8b5cf6",
                    "weight": 1.0,
                }
            )
    VIS.mkdir(parents=True, exist_ok=True)
    filename = f"phase429_{model}_architecture_path.json"
    write_json(
        VIS / filename,
        {
            "schema_version": PHYSICAL_SCHEMA,
            "phase_id": PHASE_ID,
            "model": model,
            "title": f"Phase429 {model} 架构级来源—查询物理路径",
            "evidence_scope": "architecture-level physical observation and terminal prediction; non-causal, non-neuronal",
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "meta": {
                    "phase": 429,
                    "physical_node_count": len(nodes),
                    "compute_edge_count": len(edges),
                    "prediction_gate_pass": prediction_gate["prediction_gate_pass"],
                    "pipeline_sealed": False,
                    "strict_double_blind": False,
                    "single_neuron": False,
                    "causal": False,
                },
            },
        },
    )
    manifest = {
        "schema_version": "phase429_architecture_path_manifest.v1",
        "generated_at": now(),
        "default_item_id": f"phase429_{model}_architecture_path",
        "items": [
            {
                "id": f"phase429_{model}_architecture_path",
                "label": f"Phase429 {model} 架构级物理路径",
                "filename": filename,
                "model": model,
                "phase": 429,
                "evidence_scope": "architecture physical observation; non-causal and non-neuronal",
            }
        ],
    }
    write_json(VIS / "manifest.json", manifest)
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase429_architecture_path",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase429 架构级来源—查询物理路径",
        "description": "仅显示内容门授权块的残差、查询、来源键值、注意力和多层感知机写入，不含头、通道或神经元扫描。",
        "manifest_path": "/vis_data/phase429_architecture_path/manifest.json",
        "manifest_schema": manifest["schema_version"],
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase429_architecture_path",
        "models": [model],
        "evidence_scope": "架构级物理观察与末端预测；非因果、非神经元闭合",
        "color": "#22c55e",
    }
    registry["sources"] = [row for row in registry["sources"] if row["id"] != source["id"]] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def analyze_open() -> dict[str, Any]:
    protocol = freeze_physical_protocol()
    trace_rows = []
    for model in protocol["authorized_models"]:
        complete = read_json(
            OUT / "physical" / "open" / model / "phase429_physical_complete.json"
        )
        if not complete["all_rows_complete"]:
            raise RuntimeError(f"Phase429 physical incomplete for {model}")
        trace_rows.extend(
            read_jsonl(OUT / "physical" / "open" / model / "phase429_physical_rows.jsonl")
        )
    behavior = {
        row["condition_id"]: row
        for model in protocol["authorized_models"]
        for row in read_jsonl(OUT / "models" / model / "behavior" / "phase429_rows.jsonl")
    }
    layer_count = max(int(row["layer"]) for row in trace_rows) + 1
    reconstruction = {}
    for split in ("behavior_calibration", "behavior_holdout"):
        values = [
            row["reconstruction_relative_error"]
            for row in trace_rows
            if row["split"] == split
        ]
        reconstruction[split] = {
            "trace_row_count": len(values),
            "relative_error_median": median(values),
            "relative_error_p95": clean(sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]),
        }
    last_rows = [
        row
        for row in trace_rows
        if int(row["layer"]) == layer_count - 1 and row["candidate"]
    ]
    calibration_events = [
        bool(behavior[row["condition_id"]]["natural_target_first"])
        for row in last_rows
        if row["split"] == "behavior_calibration"
    ]
    majority_label = sum(calibration_events) * 2 >= len(calibration_events)
    holdout_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in last_rows:
        if row["split"] == "behavior_holdout":
            holdout_by_group[row["semantic_group_id"]].append(row)
    physical_group_success = 0
    baseline_group_success = 0
    group_rows = []
    for group_id, values in sorted(holdout_by_group.items()):
        if len(values) != 10:
            raise RuntimeError(f"Phase429 physical prediction group is not 10 paired conditions: {group_id}")
        physical_correct = sum(
            (float(row["target_first_token_margin"]) > 0)
            == bool(behavior[row["condition_id"]]["natural_target_first"])
            for row in values
        )
        baseline_correct = sum(
            majority_label == bool(behavior[row["condition_id"]]["natural_target_first"])
            for row in values
        )
        physical_pass = physical_correct >= 8
        baseline_pass = baseline_correct >= 8
        physical_group_success += int(physical_pass)
        baseline_group_success += int(baseline_pass)
        group_rows.append(
            {
                "semantic_group_id": group_id,
                "condition_count": len(values),
                "physical_correct_count": physical_correct,
                "baseline_correct_count": baseline_correct,
                "physical_group_pass": physical_pass,
                "baseline_group_pass": baseline_pass,
            }
        )
    total_groups = len(group_rows)
    physical_interval = wilson(physical_group_success, total_groups)
    baseline_interval = wilson(baseline_group_success, total_groups)
    improvement = clean(physical_interval["estimate"] - baseline_interval["estimate"])
    reconstruction_pass = all(
        row["relative_error_median"]
        <= protocol["prediction_contract"]["reconstruction_relative_error_median_max"]
        for row in reconstruction.values()
    )
    prediction_pass = bool(
        physical_interval["lcb"]
        >= protocol["prediction_contract"]["holdout_group_success_lcb_min"]
        and improvement
        >= protocol["prediction_contract"]["majority_baseline_group_improvement_min"]
    )
    gate = {
        "schema_version": PHYSICAL_SCHEMA,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "reconstruction": reconstruction,
        "reconstruction_gate_pass": reconstruction_pass,
        "prediction": {
            "layer": layer_count - 1,
            "independent_holdout_group_count": total_groups,
            "physical_group_success": physical_interval,
            "majority_label_from_calibration": majority_label,
            "majority_baseline_group_success": baseline_interval,
            "estimate_improvement": improvement,
        },
        "prediction_gate_pass": prediction_pass,
        "sealed_unlock": bool(reconstruction_pass and prediction_pass),
        "sealed_authorized_candidates": protocol["authorized_candidates"],
        "sealed_rows_read": False,
        "physical_hooks_run": True,
        "head_channel_neuron_scan": False,
        "intervention": False,
        "causal_tested": False,
    }
    write_jsonl(OUT / "phase429_physical_prediction_group_rows.jsonl", group_rows)
    write_json(OUT / "phase429_open_physical_gate.json", gate)
    for model in protocol["authorized_models"]:
        publish_visual(
            model,
            [row for row in trace_rows if row["model"] == model],
            gate,
        )
    global_summary = read_json(OUT / "phase429_global_summary.json")
    global_summary.update(
        {
            "physical_tested": True,
            "predictive_tested": True,
            "open_physical_reconstruction_gate_pass": reconstruction_pass,
            "open_physical_prediction_gate_pass": prediction_pass,
            "sealed_unlock": gate["sealed_unlock"],
            "conclusion": (
                "The single Qwen3 no-example language-action candidate passed architecture conservation and independent terminal prediction; an exact-tuple sealed subprotocol is authorized, but no causal or neuronal mechanism is established."
                if gate["sealed_unlock"]
                else "The authorized Qwen3 candidate was physically recorded, but open architecture reconstruction or independent terminal prediction failed; sealed, causal, head, channel and neuron stages remain closed."
            ),
        }
    )
    write_json(OUT / "phase429_global_summary.json", global_summary)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "open", "analyze-open"), required=True)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.stage == "freeze":
        print(json.dumps(freeze_physical_protocol(), ensure_ascii=False, indent=2))
    elif args.stage == "open":
        if not args.model:
            raise SystemExit("--model is required for --stage open")
        run_open_model(args.model)
    else:
        analyze_open()


if __name__ == "__main__":
    main()
