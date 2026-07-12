#!/usr/bin/env python3
"""Audit Phase386 five-coordinate event files before discovery is authorized."""

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
COLLECTION = COLLECTION_ROOT / "instrument_audit"
MODELS = ("qwen3", "glm4", "deepseek7b")
COORDINATES = (
    "source_encoded",
    "query_integrated",
    "pre_decision",
    "target_encoded",
    "post_decision_next_token",
)
EXPECTED_LAYERS = {"qwen3": 36, "glm4": 40, "deepseek7b": 28}
MAX_ERROR = 0.01


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    manifest_path = COLLECTION / "models" / model / "manifest.json"
    manifest = read_json(manifest_path)
    failures: list[str] = []
    if manifest["case_count"] != 12:
        failures.append("case_count")
    if manifest["parallel_group_count"] != 3:
        failures.append("parallel_group_count")
    if manifest["layer_count"] != EXPECTED_LAYERS[model]:
        failures.append("layer_count")
    if manifest["semantic_coordinate_count"] != 5:
        failures.append("coordinate_count")
    if (
        not manifest.get("incremental_kv_cache_path", False)
        or manifest.get("fixed_three_forward_pass_claimed", True)
        or manifest.get("model_call_count", 0) < manifest["case_count"] * 3
    ):
        failures.append("incremental_call_contract")
    if manifest["required_transition_pass_count"] != 12:
        failures.append("transition_replay")
    if not manifest["all_case_gates_pass"] or not manifest["valid"]:
        failures.append("manifest_gate")
    if not manifest["neuron_replay_audited"]:
        failures.append("neuron_replay_missing")
    if manifest["top_k_used"] or manifest["pairwise_gram_materialized"]:
        failures.append("forbidden_reduction")
    if manifest["physical_holdout_opened"]:
        failures.append("physical_holdout_opened")
    if any(value > MAX_ERROR for value in manifest["gate_maxima"].values()):
        failures.append("error_threshold")

    cases = manifest["case_rows"]
    conditions = Counter(row["contrast_condition_private"] for row in cases)
    mechanisms = Counter(row["mechanism_id_private"] for row in cases)
    if sorted(conditions.values()) != [3, 3, 3, 3]:
        failures.append("condition_balance")
    if sorted(mechanisms.values()) != [4, 4, 4]:
        failures.append("mechanism_balance")

    layer_file_count = 0
    meta_file_count = 0
    checksum_failure_count = 0
    shape_failure_count = 0
    gate_failure_count = 0
    for file_row in manifest["files"]:
        path = COLLECTION_ROOT / file_row["relative_path"]
        if not path.is_file() or path.stat().st_size != file_row["byte_count"]:
            checksum_failure_count += 1
            continue
        if sha256_file(path) != file_row["sha256"]:
            checksum_failure_count += 1
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if path.name == "multitime_meta.pt":
            meta_file_count += 1
            calls = payload["generation_calls"]
            flat = [name for call in calls for name in call["coordinate_names"]]
            if (
                flat != list(COORDINATES)
                or len(calls) != payload["model_call_count"]
                or len(calls) < 3
                or not payload["required_transition_pass"]
                or payload["physical_holdout"]
                or not payload["incremental_kv_cache_path"]
            ):
                shape_failure_count += 1
            continue
        layer_file_count += 1
        if payload["coordinate_names"] != list(COORDINATES):
            shape_failure_count += 1
            continue
        component_shapes = {
            tuple(value.shape) for value in payload["component_vectors"].values()
        }
        mlp = payload["mlp"]
        attention_frames = payload["attention"]["frames"]
        if (
            len(component_shapes) != 1
            or next(iter(component_shapes))[1] != 5
            or mlp["down_projection_input_product_at_coordinates"].shape[1] != 5
            or sum(
                frame["probabilities_receivers_all_sources"].shape[2]
                for frame in attention_frames
            )
            != 5
            or [
                name
                for frame in attention_frames
                for name in frame["coordinate_names"]
            ]
            != list(COORDINATES)
        ):
            shape_failure_count += 1
        quality = payload["quality"]
        if (
            not quality["all_required_gates_pass"]
            or not all(
                frame["neuron_replay_audited"]
                and frame["required_gates_pass"]
                for frame in quality["coordinate_frames"]
            )
        ):
            gate_failure_count += 1
    if layer_file_count != 12 * EXPECTED_LAYERS[model]:
        failures.append("layer_file_count")
    if meta_file_count != 12:
        failures.append("meta_file_count")
    if checksum_failure_count:
        failures.append("checksum")
    if shape_failure_count:
        failures.append("shape")
    if gate_failure_count:
        failures.append("payload_gate")
    return {
        "model": model,
        "case_count": manifest["case_count"],
        "parallel_group_count": manifest["parallel_group_count"],
        "layer_count": manifest["layer_count"],
        "model_call_count": manifest["model_call_count"],
        "semantic_coordinate_observation_count": manifest["case_count"] * 5,
        "layer_file_count": layer_file_count,
        "meta_file_count": meta_file_count,
        "total_byte_count": manifest["total_byte_count"],
        "gate_maxima": manifest["gate_maxima"],
        "checksum_failure_count": checksum_failure_count,
        "shape_failure_count": shape_failure_count,
        "payload_gate_failure_count": gate_failure_count,
        "mechanism_case_counts_private": dict(sorted(mechanisms.items())),
        "condition_case_counts_private": dict(sorted(conditions.items())),
        "failures": failures,
        "valid": not failures,
    }


def main() -> None:
    model_rows = [audit_model(model) for model in MODELS]
    valid = all(row["valid"] for row in model_rows)
    summary = {
        "schema_version": "60.4.0",
        "phase_id": "Phase386-InstrumentAudit",
        "created_at": now(),
        "denominator": {
            "models": list(MODELS),
            "eligible_mechanisms": [
                "relation_binding",
                "entity_recency",
                "field_extraction",
            ],
            "parallel_group_count": 3,
            "case_count": sum(row["case_count"] for row in model_rows),
            "model_call_count": sum(
                row["model_call_count"] for row in model_rows
            ),
            "semantic_coordinate_observation_count": sum(
                row["semantic_coordinate_observation_count"] for row in model_rows
            ),
            "layer_file_count": sum(row["layer_file_count"] for row in model_rows),
            "total_byte_count": sum(row["total_byte_count"] for row in model_rows),
        },
        "semantic_coordinates": list(COORDINATES),
        "results": {
            "all_three_model_instruments_valid": valid,
            "all_required_generation_transitions_replayed": valid,
            "attention_head_source_events_exactly_replayable": valid,
            "mlp_channel_events_exactly_replayable": valid,
            "residual_component_conservation_pass": valid,
            "top_k_used": False,
            "pairwise_gram_materialized": False,
            "five_independent_clock_times_claimed": False,
            "language_path_discovered": False,
        },
        "models": model_rows,
        "authorization": {
            "discovery_collection": valid,
            "calibration_collection": False,
            "physical_holdout_collection": False,
            "causal_intervention": False,
        },
        "claim_boundary": {
            "instrument_validity_is_language_path": False,
            "event_replayability_is_causal_specificity": False,
            "semantic_coordinates_are_independent_times": False,
        },
    }
    write_json(PHASE_ROOT / "phase386_instrument_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
