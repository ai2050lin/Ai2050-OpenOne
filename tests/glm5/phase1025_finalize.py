#!/usr/bin/env python3
"""Finalize Phase1025 binding-specificity controls."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1025_binding_specificity_protocol as protocol


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def control_margin(block: dict[str, Any], field: str) -> float:
    bound = float(block["target_bound"][field])
    controls = [
        float(block[condition][field])
        for condition in protocol.CONDITIONS
        if condition != "target_bound"
    ]
    return bound - max(controls)


def select_discovery(
    rows: list[dict[str, Any]],
    branch: str,
) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if row["role"] == "focus_end" and int(row["depth"]) >= 1
    ]
    if branch == "alignment":
        def key(row):
            block = row["target_query_alignment"]["discovery"]
            return (
                control_margin(block, "target_query_top1"),
                block["target_bound"]["target_query_top1"],
                -int(row["depth"]),
            )
    elif branch == "retrieval":
        def key(row):
            block = row["condition_metrics"]["discovery"]
            return (
                control_margin(block, "target_cross_surface_top1"),
                block["target_bound"]["target_cross_surface_top1"],
                -int(row["depth"]),
            )
    else:
        raise ValueError(branch)
    return max(candidates, key=key)


def row_extract(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "depth": row["depth"],
        "relative_depth": row["relative_depth"],
        "condition_metrics": row["condition_metrics"],
        "target_query_alignment": row["target_query_alignment"],
    }


def model_summary(model: str) -> dict[str, Any]:
    atlas_dir = protocol.OUT_ROOT / "atlas" / model
    run = read_json(atlas_dir / "summary.json")
    rows = read_jsonl(atlas_dir / "metrics.jsonl")
    alignment = select_discovery(rows, "alignment")
    retrieval = select_discovery(rows, "retrieval")
    confirm_align = alignment["target_query_alignment"]["confirmation"]
    confirm_retr = retrieval["condition_metrics"]["confirmation"]
    align_margin = control_margin(confirm_align, "target_query_top1")
    retrieval_margin = control_margin(
        confirm_retr, "target_cross_surface_top1"
    )
    gate = {
        "alignment_bound_at_least_half": (
            confirm_align["target_bound"]["target_query_top1"] >= 0.50
        ),
        "alignment_control_margin_at_least_point_two": (
            align_margin >= 0.20
        ),
        "retrieval_bound_at_least_point_three": (
            confirm_retr["target_bound"][
                "target_cross_surface_top1"
            ] >= 0.30
        ),
        "retrieval_control_margin_at_least_point_one_five": (
            retrieval_margin >= 0.15
        ),
        "captured_tensors_finite": run["tensor_finiteness"]["all_finite"],
    }
    return {
        "model": model,
        "run": run,
        "discovery_selected_alignment": row_extract(alignment),
        "discovery_selected_retrieval": row_extract(retrieval),
        "confirmation_alignment_control_margin": align_margin,
        "confirmation_retrieval_control_margin": retrieval_margin,
        "gate": gate,
        "relation_specific_repeat": all(gate.values()),
    }


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def artifact_manifest() -> dict[str, Any]:
    manifest_path = protocol.OUT_ROOT / "final" / "artifact_manifest.json"
    files = []
    for path in sorted(
        item for item in protocol.OUT_ROOT.rglob("*")
        if item.is_file() and item != manifest_path
    ):
        files.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        })
    return {
        "schema_version": "phase1025_artifact_manifest.v1",
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def main() -> None:
    prereg = read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    models = {
        model: model_summary(model) for model in protocol.MODELS
    }
    repeat_count = sum(
        row["relation_specific_repeat"] for row in models.values()
    )
    summary = {
        "schema_version": "phase1025_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": models,
        "cross_model": {
            "relation_specific_repeat_count": repeat_count,
            "relation_specific_models": [
                model for model, row in models.items()
                if row["relation_specific_repeat"]
            ],
            "interpretation": (
                "Only GLM4 passes the complete preregistered discovery-to-"
                "confirmation gate. Qwen3 has strong secondary evidence at "
                "depth 13, but its discovery-selected retrieval depth 5 "
                "misses the confirmation control-margin gate. DeepSeek7B "
                "does not independently confirm the full gate under the "
                "current FP16 protocol."
            ),
            "late_layer_caveat": (
                "in Qwen3, co-occurrence and reversed-relation controls "
                "become much more readable at late depths, showing global "
                "prompt mixing after the more specific middle-depth window"
            ),
        },
        "theoretical_progress": {
            "supported_observational_shape": (
                "surface identity + relation-specific temporary semantic "
                "binding + later broad contextual integration"
            ),
            "relative_differential_reuse": (
                "the same nonce surface keeps its lexical identity while "
                "its query state acquires a context-dependent difference "
                "linked to the currently assigned concept"
            ),
            "not_supported": (
                "semantic replacement of surface identity, one-neuron "
                "storage, complete token mechanism, correct output "
                "execution, brain homology, or optimality"
            ),
        },
    }
    final_dir = protocol.OUT_ROOT / "final"
    protocol.write_json(final_dir / "summary.json", summary)

    prior = read_json(
        ROOT
        / "tests"
        / "glm5"
        / "result"
        / "phase1024_lexical_semantic_orthogonal_atlas"
        / "final"
        / "summary.json"
    )
    automatic = {
        "schema_version": "phase1025_automatic_next_action.v1",
        "automatic_causal_execution_authorized": False,
        "blockers": (
            "Only GLM4 passed the complete Phase1025 preregistered gate; "
            "Qwen3 missed the discovery-selected retrieval control margin "
            "despite strong depth-13 secondary evidence; DeepSeek7B did not "
            "replicate the relation-specific gate; only Qwen3 has fully "
            "finite Phase1024 internal tensors and candidate logits; all "
            "three models failed the Phase1024 behavior-claim gate"
        ),
        "recommended_next_large_task": (
            "run an independently generated, larger frozen-depth replication "
            "before any causal transport study; separately repair native "
            "FP16 numerical qualification for GLM4 and DeepSeek7B"
        ),
        "phase1024_fully_finite_model_count": prior["cross_model"][
            "fully_finite_model_count"
        ],
    }
    protocol.write_json(
        final_dir / "automatic_next_action.json", automatic
    )

    checks = {
        "protocol_common_audit": read_json(
            protocol.OUT_ROOT / "protocol" / "audit.common.json"
        )["all_checks_passed"],
        "protocol_model_audits": read_json(
            protocol.OUT_ROOT / "protocol" / "audit.models.json"
        )["all_checks_passed"],
        "all_model_runs_present": all(
            (
                protocol.OUT_ROOT / "atlas" / model / "summary.json"
            ).exists()
            for model in protocol.MODELS
        ),
        "all_depths_frozen_from_phase1024": all(
            row["run"]["selection_source"] == "phase1024_only"
            for row in models.values()
        ),
        "all_phase1025_captures_finite": all(
            row["run"]["tensor_finiteness"]["all_finite"]
            for row in models.values()
        ),
        "fp16_no_quantization": all(
            row["run"]["precision"] == "fp16"
            and row["run"]["quantization"] == "none"
            for row in models.values()
        ),
    }
    audit = {
        "schema_version": "phase1025_final_audit.v1",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    protocol.write_json(final_dir / "audit.json", audit)
    manifest = artifact_manifest()
    protocol.write_json(
        final_dir / "artifact_manifest.json", manifest
    )
    print(json.dumps({
        "cross_model": summary["cross_model"],
        "automatic_next_action": automatic,
        "audit": audit,
        "manifest": {
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
