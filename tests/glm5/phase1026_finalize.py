#!/usr/bin/env python3
"""Finalize the enlarged independent Phase1026 replication."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1026_binding_replication_protocol as protocol


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def control_margin(block: dict[str, Any], field: str) -> float:
    return float(block["target_bound"][field]) - max(
        float(block[condition][field])
        for condition in protocol.CONDITIONS
        if condition != "target_bound"
    )


def split_gate(
    row: dict[str, Any],
    split: str,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    alignment = row["target_query_alignment"][split]
    retrieval = row["condition_metrics"][split]
    values = {
        "alignment_bound": float(
            alignment["target_bound"]["target_query_top1"]
        ),
        "alignment_max_control": max(
            float(alignment[condition]["target_query_top1"])
            for condition in protocol.CONDITIONS
            if condition != "target_bound"
        ),
        "alignment_control_margin": control_margin(
            alignment, "target_query_top1"
        ),
        "retrieval_bound": float(
            retrieval["target_bound"]["target_cross_surface_top1"]
        ),
        "retrieval_max_control": max(
            float(
                retrieval[condition]["target_cross_surface_top1"]
            )
            for condition in protocol.CONDITIONS
            if condition != "target_bound"
        ),
        "retrieval_control_margin": control_margin(
            retrieval, "target_cross_surface_top1"
        ),
    }
    checks = {
        "alignment_bound": (
            values["alignment_bound"]
            >= thresholds["alignment_bound_minimum"]
        ),
        "alignment_control_margin": (
            values["alignment_control_margin"]
            >= thresholds["alignment_control_margin_minimum"]
        ),
        "retrieval_bound": (
            values["retrieval_bound"]
            >= thresholds["retrieval_bound_minimum"]
        ),
        "retrieval_control_margin": (
            values["retrieval_control_margin"]
            >= thresholds["retrieval_control_margin_minimum"]
        ),
    }
    return {
        "values": values,
        "checks": checks,
        "passed": all(checks.values()),
    }


def model_summary(
    model: str,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    atlas_dir = protocol.OUT_ROOT / "atlas" / model
    run = read_json(atlas_dir / "summary.json")
    rows = read_jsonl(atlas_dir / "metrics.jsonl")
    primary_depth = int(
        prereg["primary_depth_frozen_from_phase1025"][model]
    )
    matching = [
        row for row in rows
        if row["role"] == "focus_end"
        and int(row["depth"]) == primary_depth
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"{model}: expected one primary row, got {len(matching)}"
        )
    row = matching[0]
    split_results = {
        split: split_gate(
            row,
            split,
            prereg["replication_gate"],
        )
        for split in protocol.SPLITS
    }
    finite = bool(run["tensor_finiteness"]["all_finite"])
    return {
        "model": model,
        "primary_depth": primary_depth,
        "split_results": split_results,
        "captured_tensors_finite": finite,
        "replication_passed": (
            finite
            and all(value["passed"] for value in split_results.values())
        ),
        "run": run,
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
        "schema_version": "phase1026_artifact_manifest.v1",
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def main() -> None:
    prereg = read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    models = {
        model: model_summary(model, prereg)
        for model in protocol.MODELS
    }
    passed_models = [
        model for model, row in models.items()
        if row["replication_passed"]
    ]
    causal_authorized = len(passed_models) >= 2
    summary = {
        "schema_version": "phase1026_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": models,
        "cross_model": {
            "replication_pass_count": len(passed_models),
            "replication_pass_models": passed_models,
            "two_model_replication_gate": len(passed_models) >= 2,
        },
        "interpretation_policy": (
            "passing supports a repeated relation-specific observational "
            "state at the frozen depth; failure does not prove absence"
        ),
        "claim_limit": prereg["claim_limit"],
    }
    final_dir = protocol.OUT_ROOT / "final"
    protocol.write_json(final_dir / "summary.json", summary)
    next_action = {
        "schema_version": "phase1026_automatic_next_action.v1",
        "automatic_causal_execution_authorized": causal_authorized,
        "authorization_rule": (
            "at least two models must pass both independent splits at "
            "their preregistered joint depths"
        ),
        "passed_models": passed_models,
        "decision": (
            "a narrowly scoped causal transport protocol may be designed"
            if causal_authorized
            else "do not auto-run causal transport; first resolve "
                 "replication and FP16 numerical qualification"
        ),
    }
    protocol.write_json(
        final_dir / "automatic_next_action.json",
        next_action,
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
        "all_depths_frozen_from_phase1025": all(
            row["run"]["selection_source"] == "phase1025_only"
            for row in models.values()
        ),
        "all_captures_finite": all(
            row["captured_tensors_finite"] for row in models.values()
        ),
        "fp16_no_quantization": all(
            row["run"]["precision"] == "fp16"
            and row["run"]["quantization"] == "none"
            for row in models.values()
        ),
    }
    audit = {
        "schema_version": "phase1026_final_audit.v1",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    protocol.write_json(final_dir / "audit.json", audit)
    manifest = artifact_manifest()
    protocol.write_json(
        final_dir / "artifact_manifest.json",
        manifest,
    )
    print(json.dumps({
        "cross_model": summary["cross_model"],
        "next_action": next_action,
        "audit": audit,
        "manifest": {
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
