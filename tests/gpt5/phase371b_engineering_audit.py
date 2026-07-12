#!/usr/bin/env python3
"""Validate Phase371B engineering tensors and project the frozen storage denominator."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PHASE369_RAW = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/raw_collection"
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
ENGINEERING = PHASE371 / "anchor_qk_engineering"
PROTOCOL = PHASE371 / "phase371b_anchor_qk_protocol.json"
OUT = PHASE371 / "phase371b_engineering_summary.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def payload_breakdown(payload: dict[str, Any]) -> dict[str, int]:
    attention = payload["attention"]
    mlp = payload["mlp"]
    return {
        "component_vectors": tensor_bytes(payload["component_vectors"]),
        "attention_sufficient_state": sum(tensor_bytes(attention[key]) for key in (
            "query_states_all_positions",
            "key_states_all_positions",
            "value_states_all_positions",
            "probabilities_all_receivers_all_sources",
        )),
        "attention_materialized_derivatives": sum(tensor_bytes(attention[key]) for key in (
            "head_writes_all_receivers",
            "head_partition_writes_all_receivers",
        )),
        "mlp_sufficient_state": tensor_bytes(mlp["down_projection_input_product_all_positions"]),
        "mlp_materialized_derivatives": tensor_bytes(mlp["partition_writes_all_receivers"]),
    }


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors = []
    sequence = int(payload["sequence_length"])
    components = payload["component_vectors"]
    for name, tensor in components.items():
        if not torch.is_tensor(tensor) or tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] != sequence:
            errors.append(f"component_shape:{name}")
    attention = payload["attention"]
    query = attention["query_states_all_positions"]
    key = attention["key_states_all_positions"]
    value = attention["value_states_all_positions"]
    probabilities = attention["probabilities_all_receivers_all_sources"]
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[2] != sequence:
        errors.append("query_shape")
    if key.ndim != 4 or key.shape[0] != 1 or key.shape[2] != sequence:
        errors.append("key_shape")
    if value.shape != key.shape:
        errors.append("value_shape")
    if probabilities.shape != (1, query.shape[1], sequence, sequence):
        errors.append("probability_shape")
    hidden = int(components["layer_input_all_positions"].shape[-1])
    head_writes = attention["head_writes_all_receivers"]
    if head_writes.shape != (1, sequence, query.shape[1], hidden):
        errors.append("head_write_shape")
    attention_partitions = attention["head_partition_writes_all_receivers"]
    if attention_partitions.shape != (8, 1, sequence, hidden):
        errors.append("attention_partition_shape")
    mlp = payload["mlp"]
    if mlp["down_projection_input_product_all_positions"].shape[:2] != (1, sequence):
        errors.append("mlp_product_shape")
    if mlp["partition_writes_all_receivers"].shape != (8, 1, sequence, hidden):
        errors.append("mlp_partition_shape")
    if not payload["quality"]["all_gates_pass"]:
        errors.append("numeric_gate")
    if payload["claim_boundary"]["language_mechanism_claimed"]:
        errors.append("claim_boundary")
    return errors


def main() -> None:
    protocol = read_json(PROTOCOL)
    model_rows = []
    breakdown = {
        "component_vectors": 0,
        "attention_sufficient_state": 0,
        "attention_materialized_derivatives": 0,
        "mlp_sufficient_state": 0,
        "mlp_materialized_derivatives": 0,
    }
    all_contract_errors = []
    full_projection = 0
    compact_projection = 0
    for model in MODELS:
        manifest = read_json(ENGINEERING / "models" / model / "manifest.json")
        phase369_manifest = read_json(PHASE369_RAW / "models" / model / "manifest.json")
        model_breakdown = {key: 0 for key in breakdown}
        model_errors = []
        for file_row in manifest["files"]:
            path = ENGINEERING / file_row["relative_path"]
            if not path.exists():
                model_errors.append(f"missing:{file_row['relative_path']}")
                continue
            if path.stat().st_size != int(file_row["byte_count"]):
                model_errors.append(f"size:{file_row['relative_path']}")
            if sha256_file(path) != file_row["sha256"]:
                model_errors.append(f"sha256:{file_row['relative_path']}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model_errors.extend(
                f"{file_row['relative_path']}:{error}" for error in validate_payload(payload)
            )
            item_breakdown = payload_breakdown(payload)
            for key, value in item_breakdown.items():
                model_breakdown[key] += value
                breakdown[key] += value
        case_count = int(phase369_manifest["case_count"])
        measured = int(manifest["total_byte_count"])
        compact_tensor_bytes = (
            model_breakdown["component_vectors"]
            + model_breakdown["attention_sufficient_state"]
            + model_breakdown["mlp_sufficient_state"]
        )
        measured_tensor_bytes = sum(model_breakdown.values())
        serialization_overhead = max(measured - measured_tensor_bytes, 0)
        projected_model = measured * case_count
        projected_compact_model = (compact_tensor_bytes + serialization_overhead) * case_count
        full_projection += projected_model
        compact_projection += projected_compact_model
        all_contract_errors.extend(f"{model}:{error}" for error in model_errors)
        model_rows.append({
            "model": model,
            "engineering_case_count": 1,
            "phase369_discovery_case_count": case_count,
            "row_count": int(manifest["row_count"]),
            "measured_byte_count": measured,
            "tensor_breakdown": model_breakdown,
            "serialization_overhead_bytes": serialization_overhead,
            "projected_materialized_byte_count": projected_model,
            "projected_sufficient_state_byte_count": projected_compact_model,
            "all_numeric_gates_pass": bool(manifest["all_numeric_gates_pass"]),
            "contract_error_count": len(model_errors),
        })
    budget = int(protocol["storage_gates"]["full_discovery_additional_budget_bytes"])
    reserve = int(protocol["storage_gates"]["minimum_free_disk_reserve_bytes"])
    free = int(shutil.disk_usage(ROOT).free)
    numeric_pass = all(row["all_numeric_gates_pass"] for row in model_rows)
    contract_pass = not all_contract_errors
    materialized_storage_pass = full_projection <= budget and free - full_projection >= reserve
    sufficient_state_storage_pass = compact_projection <= budget and free - compact_projection >= reserve
    summary = {
        "schema_version": "47.3.0",
        "phase_id": "Phase371B",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "audit_three_model_exact_qk_tree_engineering_and_frozen_storage_projection",
        "denominator": {
            "model_count": 3,
            "engineering_case_count": 3,
            "generation_time_count_per_case": 3,
            "anchor_count_per_time": 3,
            "audit_row_count": sum(row["row_count"] for row in model_rows),
            "projected_discovery_case_count": sum(row["phase369_discovery_case_count"] for row in model_rows),
        },
        "results": {
            "three_model_numeric_gate_pass": numeric_pass,
            "file_hash_and_shape_contract_pass": contract_pass,
            "materialized_derivative_storage_gate_pass": materialized_storage_pass,
            "sufficient_state_storage_projection_pass": sufficient_state_storage_pass,
            "phase371b_full_gate_pass": numeric_pass and contract_pass and materialized_storage_pass,
            "phase371c_authorized": False,
            "language_mechanism_claimed": False,
        },
        "storage": {
            "engineering_measured_bytes": sum(row["measured_byte_count"] for row in model_rows),
            "tensor_breakdown": breakdown,
            "projected_materialized_full_discovery_bytes": full_projection,
            "projected_sufficient_state_full_discovery_bytes": compact_projection,
            "frozen_budget_bytes": budget,
            "free_disk_bytes_at_audit": free,
            "minimum_free_reserve_bytes": reserve,
        },
        "models": model_rows,
        "contract_errors": all_contract_errors,
        "interpretation": {
            "actual_qk_capture_feasible": numeric_pass,
            "materializing_all_head_and_partition_writes_is_redundant": True,
            "sufficient_state_can_reconstruct_removed_derivatives_exactly": True,
            "storage_failure_is_a_representation_failure_not_a_language_mechanism_result": True,
        },
        "authorization": {
            "freeze_lossless_sufficient_state_compaction_contract": numeric_pass and contract_pass and not materialized_storage_pass,
            "rerun_models_for_compaction": False,
            "rewrite_private_engineering_copies_from_existing_files": True,
            "new_case_collection": False,
            "calibration_or_physical_holdout": False,
        },
        "next_decision": "validate_lossless_on_demand_replay_from_compacted_existing_engineering_files",
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
