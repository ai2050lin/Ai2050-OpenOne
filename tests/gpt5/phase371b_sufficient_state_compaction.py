#!/usr/bin/env python3
"""Compact existing Phase371B files and prove on-demand exact-tree replay."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase365_dynamic_bundle_extraction import load_weight  # noqa: E402
from phase358_multiresolution_component_conservation import relative_error  # noqa: E402


PHASE369_RAW = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/raw_collection"
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
SOURCE = PHASE371 / "anchor_qk_engineering"
OUT = PHASE371 / "anchor_qk_sufficient_state"
PROTOCOL = PHASE371 / "phase371b_sufficient_state_protocol.json"
SUMMARY = PHASE371 / "phase371b_sufficient_state_summary.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(actual: torch.Tensor, replay: torch.Tensor) -> float:
    return relative_error(actual.float(), replay.float())[1]


def repeat_key_value(value: torch.Tensor, head_count: int) -> torch.Tensor:
    if value.shape[1] == head_count:
        return value
    return value.repeat_interleave(head_count // value.shape[1], dim=1)


def causal_probabilities(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    repeated_key = repeat_key_value(key, int(query.shape[1]))
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * scaling
    sequence = int(query.shape[-2])
    causal = torch.triu(
        torch.full((sequence, sequence), torch.finfo(query.dtype).min, dtype=query.dtype),
        diagonal=1,
    )
    scores = scores + causal.view(1, 1, sequence, sequence)
    return torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)


def replay_payload(
    model: str,
    payload: dict[str, Any],
    source_payload: dict[str, Any],
    o_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> dict[str, float]:
    attention = payload["attention"]
    query = attention["query_states_all_positions"]
    key = attention["key_states_all_positions"]
    value = attention["value_states_all_positions"]
    probabilities = attention["probabilities_all_receivers_all_sources"]
    replayed_probabilities = causal_probabilities(query, key, float(attention["scaling"]))
    qk_error = relative(probabilities, replayed_probabilities)

    head_count = int(attention["head_count"])
    head_dim = int(attention["head_dim"])
    repeated_value = repeat_key_value(value.float(), head_count)
    weighted = torch.matmul(probabilities.float(), repeated_value)
    weight = o_weight.float()
    blocks = weight.view(weight.shape[0], head_count, head_dim)
    head_writes = torch.einsum("bhqd,ohd->bqho", weighted, blocks)
    attention_direct = head_writes.sum(dim=2)
    expected_attention = payload["component_vectors"]["attention_output_all_positions"]
    attention_error = relative(expected_attention, attention_direct)
    head_partitions = [tuple(item) for item in attention["head_partitions"]]
    attention_partition_writes = torch.stack([
        head_writes[:, :, start:end].sum(dim=2) for start, end in head_partitions
    ])
    attention_tree_error = relative(attention_direct, attention_partition_writes.sum(dim=0))
    removed_head_error = relative(
        source_payload["attention"]["head_writes_all_receivers"],
        head_writes.to(source_payload["attention"]["head_writes_all_receivers"].dtype),
    )
    removed_attention_partition_error = relative(
        source_payload["attention"]["head_partition_writes_all_receivers"],
        attention_partition_writes.to(source_payload["attention"]["head_partition_writes_all_receivers"].dtype),
    )

    mlp = payload["mlp"]
    product = mlp["down_projection_input_product_all_positions"].float()
    down_weight = down_weight.float()
    mlp_direct = F.linear(product, down_weight)
    expected_mlp = payload["component_vectors"]["mlp_output_all_positions"]
    mlp_error = relative(expected_mlp, mlp_direct)
    channel_partitions = [tuple(item) for item in mlp["channel_partitions"]]
    mlp_partition_writes = torch.stack([
        F.linear(product[..., start:end], down_weight[:, start:end])
        for start, end in channel_partitions
    ])
    mlp_tree_error = relative(mlp_direct, mlp_partition_writes.sum(dim=0))
    removed_mlp_partition_error = relative(
        source_payload["mlp"]["partition_writes_all_receivers"],
        mlp_partition_writes.to(source_payload["mlp"]["partition_writes_all_receivers"].dtype),
    )
    components = payload["component_vectors"]
    block_replay = (
        components["layer_input_all_positions"].float()
        + components["attention_output_all_positions"].float()
        + components["mlp_output_all_positions"].float()
    )
    block_error = relative(components["layer_output_all_positions"], block_replay)
    return {
        "query_key_probability": qk_error,
        "attention_output": attention_error,
        "attention_tree": attention_tree_error,
        "removed_head_writes": removed_head_error,
        "removed_attention_partitions": removed_attention_partition_error,
        "mlp_output": mlp_error,
        "mlp_tree": mlp_tree_error,
        "removed_mlp_partitions": removed_mlp_partition_error,
        "block_output": block_error,
    }


def compact_payload(source: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source)
    payload["schema_version"] = "47.5.0"
    payload["phase_id"] = "Phase371B-R"
    payload["attention"] = dict(source["attention"])
    payload["mlp"] = dict(source["mlp"])
    del payload["attention"]["head_writes_all_receivers"]
    del payload["attention"]["head_partition_writes_all_receivers"]
    del payload["mlp"]["partition_writes_all_receivers"]
    payload["derivation_contract"] = {
        "attention_head_writes": "matmul(probabilities,value_then_repeat_kv) followed by o_projection_head_block",
        "attention_partition_writes": "sum_exact_head_writes_over_frozen_contiguous_head_partitions",
        "mlp_single_neuron_write": "channel_product[channel] times down_projection_weight[:,channel]",
        "mlp_partition_writes": "sum_single_neuron_writes_over_frozen_contiguous_channel_partitions",
        "removed_tensors_are_caches_not_independent_state": True,
    }
    return payload


def gates(errors: dict[str, float], protocol: dict[str, Any]) -> dict[str, bool]:
    limits = protocol["lossless_replay_gates"]
    return {
        "query_key_probability": errors["query_key_probability"] <= limits["query_key_probability_relative_error_max"],
        "attention_output": errors["attention_output"] <= limits["attention_output_relative_error_max"],
        "attention_tree": errors["attention_tree"] <= limits["attention_tree_conservation_relative_error_max"],
        "removed_head_writes": errors["removed_head_writes"] <= limits["removed_materialization_reconstruction_relative_error_max"],
        "removed_attention_partitions": errors["removed_attention_partitions"] <= limits["removed_materialization_reconstruction_relative_error_max"],
        "mlp_output": errors["mlp_output"] <= limits["mlp_output_relative_error_max"],
        "mlp_tree": errors["mlp_tree"] <= limits["mlp_tree_conservation_relative_error_max"],
        "removed_mlp_partitions": errors["removed_mlp_partitions"] <= limits["removed_materialization_reconstruction_relative_error_max"],
        "block_output": errors["block_output"] <= limits["block_output_relative_error_max"],
    }


def main() -> None:
    protocol = read_json(PROTOCOL)
    if OUT.exists():
        shutil.rmtree(OUT)
    model_rows = []
    projected_bytes = 0
    all_files = []
    for model in MODELS:
        source_manifest = read_json(SOURCE / "models" / model / "manifest.json")
        phase369_manifest = read_json(PHASE369_RAW / "models" / model / "manifest.json")
        rows = []
        files = []
        weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for file_row in source_manifest["files"]:
            source_path = SOURCE / file_row["relative_path"]
            source_payload = torch.load(source_path, map_location="cpu", weights_only=True)
            layer = int(source_payload["layer_index"])
            if layer not in weight_cache:
                weight_cache[layer] = (
                    load_weight(model, f"model.layers.{layer}.self_attn.o_proj.weight"),
                    load_weight(model, f"model.layers.{layer}.mlp.down_proj.weight"),
                )
            payload = compact_payload(source_payload)
            errors = replay_payload(model, payload, source_payload, *weight_cache[layer])
            row_gates = gates(errors, protocol)
            payload["compaction_validation"] = {
                "errors": errors,
                "gates": row_gates,
                "all_gates_pass": all(row_gates.values()),
            }
            path = OUT / file_row["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
            compact_file = {
                "generation_time": int(payload["generation_time"]),
                "layer_index": layer,
                "relative_path": str(path.relative_to(OUT)),
                "byte_count": path.stat().st_size,
                "sha256": sha256_file(path),
                "all_gates_pass": all(row_gates.values()),
            }
            files.append(compact_file)
            all_files.append({"model": model, **compact_file})
            rows.append({
                "generation_time": int(payload["generation_time"]),
                "layer_index": layer,
                "sequence_length": int(payload["sequence_length"]),
                "errors": errors,
                "all_gates_pass": all(row_gates.values()),
            })
        total_bytes = sum(row["byte_count"] for row in files)
        case_count = int(phase369_manifest["case_count"])
        model_projection = total_bytes * case_count
        projected_bytes += model_projection
        maxima = {key: max(row["errors"][key] for row in rows) for key in rows[0]["errors"]}
        manifest = {
            "schema_version": "47.5.0",
            "phase_id": "Phase371B-R",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "blind_case_id": source_manifest["blind_case_id"],
            "row_count": len(rows),
            "file_count": len(files),
            "total_byte_count": total_bytes,
            "projected_discovery_case_count": case_count,
            "projected_discovery_byte_count": model_projection,
            "max_errors": maxima,
            "all_replay_gates_pass": all(row["all_gates_pass"] for row in rows),
            "files": files,
            "rows": rows,
        }
        write_json(OUT / "models" / model / "manifest.json", manifest)
        model_rows.append({key: value for key, value in manifest.items() if key not in {"files", "rows"}})
    free = int(shutil.disk_usage(ROOT).free)
    budget = int(protocol["storage_gate"]["full_discovery_budget_bytes"])
    reserve = int(protocol["storage_gate"]["minimum_free_reserve_bytes"])
    replay_pass = all(row["all_replay_gates_pass"] for row in model_rows)
    storage_pass = projected_bytes <= budget and free - projected_bytes >= reserve
    summary = {
        "schema_version": "47.5.0",
        "phase_id": "Phase371B-R",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "validate_lossless_sufficient_state_storage_and_on_demand_exact_tree_replay",
        "execution": {
            "new_model_execution": False,
            "source": "existing_phase371b_engineering_files",
            "model_order": list(MODELS),
        },
        "denominator": {
            "model_count": 3,
            "engineering_case_count": 3,
            "audit_row_count": sum(row["row_count"] for row in model_rows),
            "file_count": sum(row["file_count"] for row in model_rows),
            "projected_discovery_case_count": sum(row["projected_discovery_case_count"] for row in model_rows),
        },
        "results": {
            "all_on_demand_replay_gates_pass": replay_pass,
            "actual_compact_storage_projection_pass": storage_pass,
            "phase371b_repaired_full_gate_pass": replay_pass and storage_pass,
            "language_mechanism_claimed": False,
            "phase371c_protocol_design_authorized": replay_pass and storage_pass,
            "phase371c_model_execution_authorized": False,
        },
        "storage": {
            "engineering_compact_bytes": sum(row["total_byte_count"] for row in model_rows),
            "projected_discovery_bytes": projected_bytes,
            "budget_bytes": budget,
            "free_disk_bytes_at_audit": free,
            "minimum_free_reserve_bytes": reserve,
        },
        "models": model_rows,
        "claim_boundary": {
            "exact_state_capture_and_replay_engineering_feasible": replay_pass,
            "three_anchor_sufficient_state_within_budget": storage_pass,
            "future_or_language_prediction_tested": False,
            "new_language_path_discovered": False,
            "calibration_or_physical_holdout_opened": False,
        },
        "next_decision": "freeze_independent_phase371c_discovery_calibration_protocol_without_opening_data",
    }
    write_json(SUMMARY, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
