#!/usr/bin/env python3
"""Audit Phase390 exact events and semantic role partitions before discovery."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase390_role_mapping import semantic_role_indices  # noqa: E402


PHASE_ROOT = ROOT / "tests/gpt5/result/phase390_joint_formation_graph"
COLLECTION_ROOT = PHASE_ROOT / "collection"
COLLECTION = COLLECTION_ROOT / "instrument_audit"
CASE_FILE = PHASE_ROOT / "protocol/private/phase390_instrument_audit_cases.jsonl"
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_model(model: str, cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    manifest = read_json(COLLECTION / "models" / model / "manifest.json")
    failures: list[str] = []
    if manifest["case_count"] != 4 or manifest["parallel_group_count"] != 1:
        failures.append("denominator")
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
    if manifest["required_transition_pass_count"] != 4:
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

    conditions = Counter(
        row["contrast_condition_private"] for row in manifest["case_rows"]
    )
    mechanisms = Counter(row["mechanism_id_private"] for row in manifest["case_rows"])
    if sorted(conditions.values()) != [1, 1, 1, 1]:
        failures.append("condition_balance")
    if mechanisms != {"field_extraction": 4}:
        failures.append("mechanism_balance")

    layer_file_count = 0
    meta_file_count = 0
    checksum_failure_count = 0
    shape_failure_count = 0
    gate_failure_count = 0
    for file_row in manifest["files"]:
        path = COLLECTION_ROOT / file_row["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != file_row["byte_count"]
            or sha256_file(path) != file_row["sha256"]
        ):
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
        attention_frames = payload["attention"]["frames"]
        if (
            len(component_shapes) != 1
            or next(iter(component_shapes))[1] != 5
            or payload["mlp"]["down_projection_input_product_at_coordinates"].shape[1]
            != 5
            or sum(
                frame["probabilities_receivers_all_sources"].shape[2]
                for frame in attention_frames
            )
            != 5
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

    role_failures: list[dict[str, Any]] = []
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=False,
    )
    for row in manifest["case_rows"]:
        case = cases[row["blind_case_id"]]
        sample_path = (
            COLLECTION
            / "private/models"
            / model
            / row["blind_case_id"]
            / "layer_000.pt"
        )
        payload = torch.load(sample_path, map_location="cpu", weights_only=False)
        frame = next(
            item
            for item in payload["attention"]["frames"]
            if "query_integrated" in item["coordinate_names"]
        )
        receiver = frame["coordinate_names"].index("query_integrated")
        query_position = int(frame["global_positions"][receiver])
        partition, audit = semantic_role_indices(tokenizer, case, query_position)
        if (
            audit["missing_fragments"]
            or not audit["partition_conserved"]
            or any(not partition[role] for role in ("entities", "attributes_items", "relations", "query_window"))
        ):
            role_failures.append(
                {
                    "blind_case_id": row["blind_case_id"],
                    "audit": audit,
                    "role_sizes": {role: len(value) for role, value in partition.items()},
                }
            )
    if checksum_failure_count:
        failures.append("checksum")
    if shape_failure_count:
        failures.append("shape")
    if gate_failure_count:
        failures.append("payload_gate")
    if role_failures:
        failures.append("semantic_role_mapping")
    if layer_file_count != 4 * EXPECTED_LAYERS[model] or meta_file_count != 4:
        failures.append("file_count")
    return {
        "model": model,
        "case_count": manifest["case_count"],
        "parallel_group_count": manifest["parallel_group_count"],
        "layer_count": manifest["layer_count"],
        "model_call_count": manifest["model_call_count"],
        "layer_file_count": layer_file_count,
        "meta_file_count": meta_file_count,
        "total_byte_count": manifest["total_byte_count"],
        "gate_maxima": manifest["gate_maxima"],
        "checksum_failure_count": checksum_failure_count,
        "shape_failure_count": shape_failure_count,
        "payload_gate_failure_count": gate_failure_count,
        "role_mapping_failure_count": len(role_failures),
        "role_mapping_failures_private": role_failures,
        "failures": failures,
        "valid": not failures,
    }


def main() -> None:
    cases = {row["blind_case_id"]: row for row in read_jsonl(CASE_FILE)}
    model_rows = [audit_model(model, cases) for model in MODELS]
    valid = all(row["valid"] for row in model_rows)
    summary = {
        "schema_version": "64.4.0",
        "phase_id": "Phase390-InstrumentAudit",
        "created_at": now(),
        "denominator": {
            "models": list(MODELS),
            "eligible_mechanisms": ["field_extraction"],
            "parallel_group_count": 1,
            "case_count": sum(row["case_count"] for row in model_rows),
            "model_call_count": sum(row["model_call_count"] for row in model_rows),
            "layer_file_count": sum(row["layer_file_count"] for row in model_rows),
            "total_byte_count": sum(row["total_byte_count"] for row in model_rows),
        },
        "results": {
            "all_three_model_instruments_valid": valid,
            "actual_incremental_transitions_replayed": valid,
            "exact_component_conservation_pass": valid,
            "exact_source_role_partition_pass": valid,
            "all_heads_and_sources_retained": valid,
            "learned_basis_used": False,
            "language_path_discovered": False,
        },
        "models": model_rows,
        "authorization": {
            "discovery_collection": valid,
            "calibration_collection": False,
            "physical_holdout_collection": False,
            "causal_replay": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "instrument_validity_is_language_path": False,
            "role_partition_is_multi_source_cooperation": False,
            "event_replayability_is_causal_specificity": False,
        },
    }
    write_json(PHASE_ROOT / "phase390_instrument_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
