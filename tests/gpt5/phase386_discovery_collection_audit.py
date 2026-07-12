#!/usr/bin/env python3
"""Audit every Phase386 incremental discovery artifact before feature extraction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
COLLECTION_ROOT = PHASE_ROOT / "collection"
STAGE_ROOT = COLLECTION_ROOT / "discovery"
MODELS = ("qwen3", "glm4", "deepseek7b")
EXPECTED_LAYERS = {"qwen3": 36, "glm4": 40, "deepseek7b": 28}
COORDINATES = (
    "source_encoded",
    "query_integrated",
    "pre_decision",
    "target_encoded",
    "post_decision_next_token",
)
MECHANISMS = ("relation_binding", "entity_recency", "field_extraction")
MAX_ERROR = 0.01


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_model(model: str) -> dict[str, Any]:
    manifest = read_json(STAGE_ROOT / "models" / model / "manifest.json")
    failures: list[str] = []
    expected_layers = EXPECTED_LAYERS[model]
    expected_files = 96 * (expected_layers + 1)
    if manifest["case_count"] != 96 or manifest["parallel_group_count"] != 24:
        failures.append("denominator")
    if manifest["layer_count"] != expected_layers:
        failures.append("layer_count")
    if manifest["file_count"] != expected_files:
        failures.append("file_count")
    if (
        not manifest["valid"]
        or not manifest["all_case_gates_pass"]
        or manifest["required_transition_pass_count"] != 96
    ):
        failures.append("runtime_gate")
    if (
        not manifest["incremental_kv_cache_path"]
        or manifest["fixed_three_forward_pass_claimed"]
        or manifest["model_call_count"] < 96 * 3
    ):
        failures.append("incremental_contract")
    if manifest["neuron_replay_audited"]:
        failures.append("unexpected_full_neuron_replay")
    if (
        manifest["top_k_used"]
        or manifest["pairwise_gram_materialized"]
        or manifest["physical_holdout_opened"]
    ):
        failures.append("forbidden_operation")
    if any(value > MAX_ERROR for value in manifest["gate_maxima"].values()):
        failures.append("error_threshold")

    mechanisms = Counter(row["mechanism_id_private"] for row in manifest["case_rows"])
    conditions = Counter(row["contrast_condition_private"] for row in manifest["case_rows"])
    groups = Counter(row["public_parallel_group_id"] for row in manifest["case_rows"])
    if mechanisms != Counter({mechanism: 32 for mechanism in MECHANISMS}):
        failures.append("mechanism_balance")
    if sorted(conditions.values()) != [24, 24, 24, 24]:
        failures.append("condition_balance")
    if len(groups) != 24 or set(groups.values()) != {4}:
        failures.append("group_completeness")

    checksum_failures = 0
    shape_failures = 0
    payload_gate_failures = 0
    layer_files = 0
    meta_files = 0
    observed_calls = 0
    for file_row in manifest["files"]:
        path = COLLECTION_ROOT / file_row["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != file_row["byte_count"]
            or sha256_file(path) != file_row["sha256"]
        ):
            checksum_failures += 1
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if path.name == "multitime_meta.pt":
            meta_files += 1
            calls = payload["generation_calls"]
            observed_calls += len(calls)
            flat = [name for call in calls for name in call["coordinate_names"]]
            if (
                payload["schema_version"] != "60.5.0"
                or not payload["incremental_kv_cache_path"]
                or payload["model_call_count"] != len(calls)
                or flat != list(COORDINATES)
                or not payload["required_transition_pass"]
                or payload["physical_holdout"]
            ):
                shape_failures += 1
            continue
        layer_files += 1
        component_shapes = {
            tuple(value.shape) for value in payload["component_vectors"].values()
        }
        attention_frames = payload["attention"]["frames"]
        attention_coordinates = [
            name for frame in attention_frames for name in frame["coordinate_names"]
        ]
        if (
            payload["coordinate_names"] != list(COORDINATES)
            or len(component_shapes) != 1
            or next(iter(component_shapes))[1] != 5
            or payload["mlp"][
                "down_projection_input_product_at_coordinates"
            ].shape[1]
            != 5
            or attention_coordinates != list(COORDINATES)
            or sum(
                frame["probabilities_receivers_all_sources"].shape[2]
                for frame in attention_frames
            )
            != 5
        ):
            shape_failures += 1
        quality = payload["quality"]
        if (
            not quality["all_required_gates_pass"]
            or not all(
                frame["required_gates_pass"]
                and not frame["neuron_replay_audited"]
                for frame in quality["coordinate_frames"]
            )
        ):
            payload_gate_failures += 1
    if layer_files != 96 * expected_layers or meta_files != 96:
        failures.append("artifact_counts")
    if observed_calls != manifest["model_call_count"]:
        failures.append("model_call_count")
    if checksum_failures:
        failures.append("checksum")
    if shape_failures:
        failures.append("shape")
    if payload_gate_failures:
        failures.append("payload_gate")
    return {
        "model": model,
        "case_count": manifest["case_count"],
        "parallel_group_count": manifest["parallel_group_count"],
        "layer_count": manifest["layer_count"],
        "model_call_count": manifest["model_call_count"],
        "layer_file_count": layer_files,
        "meta_file_count": meta_files,
        "total_byte_count": manifest["total_byte_count"],
        "gate_maxima": manifest["gate_maxima"],
        "mechanism_case_counts_private": dict(sorted(mechanisms.items())),
        "checksum_failure_count": checksum_failures,
        "shape_failure_count": shape_failures,
        "payload_gate_failure_count": payload_gate_failures,
        "failures": failures,
        "valid": not failures,
    }


def main() -> None:
    rows = [audit_model(model) for model in MODELS]
    valid = all(row["valid"] for row in rows)
    summary = {
        "schema_version": "60.6.0",
        "phase_id": "Phase386-DiscoveryCollectionAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "mechanisms": list(MECHANISMS),
            "parallel_group_count": 24,
            "case_count": sum(row["case_count"] for row in rows),
            "model_call_count": sum(row["model_call_count"] for row in rows),
            "semantic_coordinate_observation_count": sum(
                row["case_count"] * 5 for row in rows
            ),
            "layer_file_count": sum(row["layer_file_count"] for row in rows),
            "total_byte_count": sum(row["total_byte_count"] for row in rows),
        },
        "models": rows,
        "results": {
            "all_discovery_artifacts_valid": valid,
            "all_required_generation_transitions_replayed": valid,
            "all_component_ledgers_conserved": valid,
            "all_five_coordinates_present": valid,
            "top_k_used": False,
            "pairwise_gram_materialized": False,
            "physical_holdout_opened": False,
            "language_path_discovered": False,
        },
        "authorization": {
            "discovery_relation_extraction": valid,
            "calibration_collection": False,
            "physical_holdout_collection": False,
            "causal_intervention": False,
        },
    }
    write_json(PHASE_ROOT / "phase386_discovery_collection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
