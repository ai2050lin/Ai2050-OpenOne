#!/usr/bin/env python3
"""Finalize Phase1027 local binding-state transport."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1027_binding_transport_protocol as protocol


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_gate(
    metrics: dict[str, Any],
    split: str,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    clean = metrics["clean"][split]
    matched = metrics["interventions"]["matched_focus"][split]
    scrambled = metrics["interventions"]["scrambled_focus"][split]
    bos = metrics["interventions"]["matched_bos_delta"][split]
    values = {
        "clean_target_top1": float(clean["target_top1"]),
        "matched_donor_top1": float(matched["donor_top1"]),
        "matched_margin_shift": float(
            matched["donor_vs_target_margin_shift_from_clean"]
        ),
        "matched_minus_scrambled_donor_top1": float(
            matched["donor_top1"] - scrambled["donor_top1"]
        ),
        "matched_minus_bos_donor_top1": float(
            matched["donor_top1"] - bos["donor_top1"]
        ),
        "scrambled_intended_donor_top1": float(
            scrambled["donor_top1"]
        ),
        "bos_intended_donor_top1": float(bos["donor_top1"]),
    }
    checks = {
        "clean_target_top1": (
            values["clean_target_top1"]
            >= thresholds["clean_target_top1_minimum"]
        ),
        "matched_donor_top1": (
            values["matched_donor_top1"]
            >= thresholds["matched_donor_top1_minimum"]
        ),
        "matched_margin_shift": (
            values["matched_margin_shift"]
            >= thresholds["matched_margin_shift_minimum"]
        ),
        "matched_vs_scrambled": (
            values["matched_minus_scrambled_donor_top1"]
            >= thresholds[
                "matched_minus_scrambled_donor_top1_minimum"
            ]
        ),
        "matched_vs_bos": (
            values["matched_minus_bos_donor_top1"]
            >= thresholds["matched_minus_bos_donor_top1_minimum"]
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
    metrics = read_json(atlas_dir / "metrics.json")
    split_results = {
        split: split_gate(
            metrics,
            split,
            prereg["replication_gate"],
        )
        for split in protocol.SPLITS
    }
    finite = bool(run["finiteness"]["all_finite"])
    return {
        "model": model,
        "patch_depth": run["patch_depth"],
        "readout_depth": run["readout_depth"],
        "split_results": split_results,
        "all_arrays_finite": finite,
        "transport_gate_passed": (
            finite
            and all(row["passed"] for row in split_results.values())
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
        "schema_version": "phase1027_artifact_manifest.v1",
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
        if row["transport_gate_passed"]
    ]
    summary = {
        "schema_version": "phase1027_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": models,
        "cross_model": {
            "transport_pass_count": len(passed_models),
            "transport_pass_models": passed_models,
            "two_model_causal_repeat": len(passed_models) >= 2,
        },
        "interpretation": (
            "a pass means that replacing the repeated-label state at the "
            "frozen middle depth transports concept identity into a later "
            "internal answer-boundary state more specifically than a third-"
            "concept transplant or the same vector at position 0"
        ),
        "claim_limit": prereg["claim_limit"],
    }
    final_dir = protocol.OUT_ROOT / "final"
    protocol.write_json(final_dir / "summary.json", summary)
    next_action = {
        "schema_version": "phase1027_automatic_next_action.v1",
        "automatic_next_execution_authorized": False,
        "reason": (
            "the requested lexical-semantic mechanism question now has "
            "observational replication plus a preregistered local causal "
            "test; further automation would require choosing between "
            "subspace decomposition, output validation, or a different "
            "language-pattern family"
        ),
        "recommended_large_tasks": (
            "repeat in numerically native precision and decompose the "
            "transported state into reusable semantic and surface "
            "subspaces; then extend the same protocol to translation, "
            "contrast, and punctuation families"
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
        "all_depths_frozen": all(
            row["run"]["selection_source"]
            == "phase1026_and_prior_finite_atlas"
            for row in models.values()
        ),
        "all_arrays_finite": all(
            row["all_arrays_finite"] for row in models.values()
        ),
        "fp16_no_quantization": all(
            row["run"]["precision"] == "fp16"
            and row["run"]["quantization"] == "none"
            for row in models.values()
        ),
    }
    audit = {
        "schema_version": "phase1027_final_audit.v1",
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
