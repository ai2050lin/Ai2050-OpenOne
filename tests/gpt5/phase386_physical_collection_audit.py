#!/usr/bin/env python3
"""Audit the one-time Phase386 physical holdout collection before evaluation."""

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
STAGE = COLLECTION_ROOT / "physical_holdout"
MODELS = ("qwen3", "glm4", "deepseek7b")
LAYERS = {"qwen3": 36, "glm4": 40, "deepseek7b": 28}
COORDINATES = (
    "source_encoded",
    "query_integrated",
    "pre_decision",
    "target_encoded",
    "post_decision_next_token",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(model: str) -> dict[str, Any]:
    manifest = read_json(STAGE / "models" / model / "manifest.json")
    failures: list[str] = []
    layer_count = LAYERS[model]
    if (
        manifest["case_count"] != 48
        or manifest["parallel_group_count"] != 12
        or manifest["layer_count"] != layer_count
        or manifest["file_count"] != 48 * (layer_count + 1)
    ):
        failures.append("denominator")
    if (
        not manifest["valid"]
        or not manifest["all_case_gates_pass"]
        or manifest["required_transition_pass_count"] != 48
        or not manifest["incremental_kv_cache_path"]
        or not manifest["physical_holdout_opened"]
        or manifest["top_k_used"]
        or manifest["pairwise_gram_materialized"]
        or any(value > 0.01 for value in manifest["gate_maxima"].values())
    ):
        failures.append("runtime_contract")
    groups = Counter(row["public_parallel_group_id"] for row in manifest["case_rows"])
    mechanisms = Counter(row["mechanism_id_private"] for row in manifest["case_rows"])
    if len(groups) != 12 or set(groups.values()) != {4}:
        failures.append("group_completeness")
    if len(mechanisms) != 3 or set(mechanisms.values()) != {16}:
        failures.append("mechanism_balance")
    checksum_failures = 0
    shape_failures = 0
    layer_files = 0
    meta_files = 0
    calls = 0
    for row in manifest["files"]:
        path = COLLECTION_ROOT / row["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["byte_count"]
            or sha256_file(path) != row["sha256"]
        ):
            checksum_failures += 1
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if path.name == "multitime_meta.pt":
            meta_files += 1
            calls += payload["model_call_count"]
            if (
                not payload["physical_holdout"]
                or not payload["required_transition_pass"]
                or [
                    name
                    for call in payload["generation_calls"]
                    for name in call["coordinate_names"]
                ]
                != list(COORDINATES)
            ):
                shape_failures += 1
            continue
        layer_files += 1
        if (
            payload["coordinate_names"] != list(COORDINATES)
            or any(
                value.shape[1] != 5
                for value in payload["component_vectors"].values()
            )
            or payload["mlp"][
                "down_projection_input_product_at_coordinates"
            ].shape[1]
            != 5
            or not payload["quality"]["all_required_gates_pass"]
        ):
            shape_failures += 1
    if layer_files != 48 * layer_count or meta_files != 48:
        failures.append("artifact_count")
    if calls != manifest["model_call_count"]:
        failures.append("model_call_count")
    if checksum_failures:
        failures.append("checksum")
    if shape_failures:
        failures.append("shape")
    return {
        "model": model,
        "case_count": manifest["case_count"],
        "model_call_count": manifest["model_call_count"],
        "layer_file_count": layer_files,
        "meta_file_count": meta_files,
        "total_byte_count": manifest["total_byte_count"],
        "gate_maxima": manifest["gate_maxima"],
        "checksum_failure_count": checksum_failures,
        "shape_failure_count": shape_failures,
        "failures": failures,
        "valid": not failures,
    }


def main() -> None:
    protocol = read_json(PHASE_ROOT / "phase386_physical_holdout_protocol.json")
    frozen = PHASE_ROOT / protocol["candidate_file"]
    candidate_checksum_valid = (
        hashlib.sha256(frozen.read_bytes()).hexdigest()
        == protocol["candidate_file_sha256"]
    )
    rows = [audit(model) for model in MODELS]
    valid = candidate_checksum_valid and all(row["valid"] for row in rows)
    summary = {
        "schema_version": "60.12.0",
        "phase_id": "Phase386-PhysicalCollectionAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "case_count": sum(row["case_count"] for row in rows),
            "model_call_count": sum(row["model_call_count"] for row in rows),
            "layer_file_count": sum(row["layer_file_count"] for row in rows),
            "total_byte_count": sum(row["total_byte_count"] for row in rows),
            "frozen_candidate_count": protocol["frozen_candidate_count"],
        },
        "candidate_checksum_valid": candidate_checksum_valid,
        "models": rows,
        "results": {
            "all_physical_holdout_artifacts_valid": valid,
            "candidate_values_evaluated_by_this_audit": False,
            "holdout_opened_once": True,
        },
        "authorization": {
            "physical_holdout_candidate_evaluation": valid,
            "candidate_replacement": False,
            "causal_intervention": False,
        },
    }
    path = PHASE_ROOT / "phase386_physical_collection_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
