#!/usr/bin/env python3
"""Audit all Phase371C discovery files before exact-path extraction."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
RAW = PHASE371 / "phase371c_internal_discovery"
FREEZE = PHASE371 / "phase371c_internal_execution_freeze.json"
OUT = PHASE371 / "phase371c_internal_collection_audit.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_time(payload: dict[str, Any]) -> list[str]:
    errors = []
    if payload.get("semantic_labels_available") is not False:
        errors.append("semantic_label_boundary")
    logits = payload.get("full_vocabulary_logits")
    if not torch.is_tensor(logits) or logits.ndim != 1 or logits.numel() < 1000:
        errors.append("vocabulary_logits_shape")
    if int(payload.get("sequence_length", 0)) <= 0:
        errors.append("sequence_length")
    return errors


def validate_layer(payload: dict[str, Any]) -> list[str]:
    errors = []
    sequence = int(payload.get("sequence_length", 0))
    components = payload.get("component_vectors", {})
    if len(components) != 7:
        errors.append("component_count")
    for name, tensor in components.items():
        if not torch.is_tensor(tensor) or tensor.ndim != 3 or tensor.shape[:2] != (1, sequence):
            errors.append(f"component_shape:{name}")
    attention = payload.get("attention", {})
    query = attention.get("query_states_all_positions")
    key = attention.get("key_states_all_positions")
    value = attention.get("value_states_all_positions")
    probabilities = attention.get("probabilities_all_receivers_all_sources")
    if not torch.is_tensor(query) or query.ndim != 4 or query.shape[2] != sequence:
        errors.append("query_shape")
    if not torch.is_tensor(key) or key.ndim != 4 or key.shape[2] != sequence:
        errors.append("key_shape")
    if not torch.is_tensor(value) or not torch.is_tensor(key) or value.shape != key.shape:
        errors.append("value_shape")
    if torch.is_tensor(query) and (
        not torch.is_tensor(probabilities)
        or probabilities.shape != (1, query.shape[1], sequence, sequence)
    ):
        errors.append("probability_shape")
    if "head_writes_all_receivers" in attention or "head_partition_writes_all_receivers" in attention:
        errors.append("materialized_attention_derivative")
    mlp = payload.get("mlp", {})
    product = mlp.get("down_projection_input_product_all_positions")
    if not torch.is_tensor(product) or product.ndim != 3 or product.shape[:2] != (1, sequence):
        errors.append("mlp_product_shape")
    if "partition_writes_all_receivers" in mlp:
        errors.append("materialized_mlp_derivative")
    if payload.get("claim_boundary", {}).get("semantic_labels_available") is not False:
        errors.append("semantic_label_boundary")
    if not payload.get("quality", {}).get("all_gates_pass"):
        errors.append("numeric_gate")
    return errors


def main() -> None:
    freeze = read_json(FREEZE)
    errors = []
    model_rows = []
    total_bytes = 0
    total_files = 0
    total_cases = 0
    for model in MODELS:
        manifest = read_json(RAW / "models" / model / "manifest.json")
        model_errors = []
        if int(manifest.get("case_count", 0)) != 88:
            model_errors.append("case_count")
        if int(manifest.get("file_count", 0)) != 1056:
            model_errors.append("file_count")
        if manifest.get("semantic_labels_available") is not False:
            model_errors.append("manifest_semantic_label_boundary")
        if not manifest.get("all_numeric_gates_pass"):
            model_errors.append("manifest_numeric_gate")
        seen_paths = set()
        for file_row in manifest["files"]:
            path = RAW / file_row["relative_path"]
            if file_row["relative_path"] in seen_paths:
                model_errors.append(f"duplicate:{file_row['relative_path']}")
                continue
            seen_paths.add(file_row["relative_path"])
            if not path.is_file():
                model_errors.append(f"missing:{file_row['relative_path']}")
                continue
            if path.stat().st_size != int(file_row["byte_count"]):
                model_errors.append(f"size:{file_row['relative_path']}")
            if sha256_file(path) != file_row["sha256"]:
                model_errors.append(f"sha256:{file_row['relative_path']}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload_errors = validate_time(payload) if file_row["kind"] == "time_meta" else validate_layer(payload)
            model_errors.extend(f"{file_row['relative_path']}:{error}" for error in payload_errors)
        errors.extend(f"{model}:{error}" for error in model_errors)
        total_bytes += int(manifest["total_byte_count"])
        total_files += int(manifest["file_count"])
        total_cases += int(manifest["case_count"])
        model_rows.append({
            "model": model,
            "case_count": int(manifest["case_count"]),
            "file_count": int(manifest["file_count"]),
            "byte_count": int(manifest["total_byte_count"]),
            "anchor_layers": manifest["anchor_layers"],
            "max_errors": manifest["max_errors"],
            "contract_error_count": len(model_errors),
        })
    valid = (
        bool(freeze["valid"])
        and not errors
        and total_cases == 264
        and total_files == 3168
        and total_bytes <= 64 * 1024**3
    )
    summary = {
        "schema_version": "47.12.0",
        "phase_id": "Phase371C",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "verify_complete_label_free_exact_sufficient_state_discovery_ledger",
        "valid": valid,
        "denominator": {
            "model_count": 3,
            "case_count": total_cases,
            "file_count": total_files,
            "generation_time_count": 3,
            "anchor_layer_count": 3,
            "anchor_layer_row_count": total_cases * 3 * 3,
        },
        "storage": {
            "total_byte_count": total_bytes,
            "budget_bytes": 64 * 1024**3,
            "free_disk_bytes_after_collection": shutil.disk_usage(ROOT).free,
        },
        "models": model_rows,
        "errors": errors,
        "claim_boundary": {
            "measurement_ledger_complete": valid,
            "exact_language_path_discovered": False,
            "calibration_internal_states_opened": False,
            "physical_holdout_opened": False,
        },
        "authorization": {
            "extract_label_free_exact_path_objects": valid,
            "use_semantic_labels_during_extraction": False,
            "open_calibration": False,
            "open_physical": False,
        },
        "next_decision": "extract_all_deterministic_branch_merge_events_without_top_k" if valid else "repair_ledger_before_analysis",
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
