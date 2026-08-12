#!/usr/bin/env python3
"""Finalize Phase1028 without assuming that alliances are necessary."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1028_role_depth_causal_map_protocol as protocol


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_gate(
    row: dict[str, Any],
    clean_target_top1: float,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    matched = row["confirmation"]["matched"]
    scrambled = row["confirmation"]["scrambled_concept"]
    wrong = row["confirmation"]["matched_wrong_position"]
    values = {
        "clean_target_top1": float(clean_target_top1),
        "matched_donor_top1": float(matched["donor_top1"]),
        "matched_margin_shift": float(
            matched["donor_vs_target_margin_shift_from_clean"]
        ),
        "matched_minus_scrambled_donor_top1": float(
            matched["donor_top1"] - scrambled["donor_top1"]
        ),
        "matched_minus_wrong_position_donor_top1": float(
            matched["donor_top1"] - wrong["donor_top1"]
        ),
        "scrambled_intended_donor_top1": float(
            scrambled["donor_top1"]
        ),
        "wrong_position_donor_top1": float(wrong["donor_top1"]),
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
        "matched_vs_wrong_position": (
            values["matched_minus_wrong_position_donor_top1"]
            >= thresholds[
                "matched_minus_wrong_position_donor_top1_minimum"
            ]
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
    clean = float(
        metrics["clean_readout"]["confirmation"]["target_top1"]
    )
    candidates = []
    for row in metrics["confirmation_candidates"]:
        gate = candidate_gate(
            row,
            clean,
            prereg["confirmation_gate"],
        )
        candidates.append({
            "role": row["role"],
            "depth": row["depth"],
            "wrong_role": row["wrong_role"],
            "discovery_metrics": row["discovery_metrics"],
            "confirmation": row["confirmation"],
            "gate": gate,
        })
    passing = [row for row in candidates if row["gate"]["passed"]]
    passing_upstream = [
        row for row in passing if row["role"] != "pre_output"
    ]
    return {
        "model": model,
        "clean_readout_confirmation": metrics[
            "clean_readout"
        ]["confirmation"],
        "selected_candidates": candidates,
        "passing_candidates": [
            {"role": row["role"], "depth": row["depth"]}
            for row in passing
        ],
        "passing_upstream_candidates": [
            {"role": row["role"], "depth": row["depth"]}
            for row in passing_upstream
        ],
        "any_singleton_pass": bool(passing),
        "any_upstream_singleton_pass": bool(passing_upstream),
        "all_arrays_finite": bool(run["finiteness"]["all_finite"]),
        "run": run,
        "observational_role_depth": metrics[
            "observational_role_depth"
        ],
    }


def repeated_roles(
    models: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, int]:
    count = Counter()
    for row in models.values():
        count.update({
            value["role"] for value in row[field]
        })
    return dict(sorted(count.items()))


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
        "schema_version": "phase1028_artifact_manifest.v1",
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
    singleton_models = [
        model for model, row in models.items()
        if row["any_singleton_pass"]
    ]
    upstream_models = [
        model for model, row in models.items()
        if row["any_upstream_singleton_pass"]
    ]
    role_counts = repeated_roles(models, "passing_candidates")
    upstream_role_counts = repeated_roles(
        models, "passing_upstream_candidates"
    )
    summary = {
        "schema_version": "phase1028_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": models,
        "cross_model": {
            "singleton_pass_models": singleton_models,
            "singleton_pass_count": len(singleton_models),
            "upstream_singleton_pass_models": upstream_models,
            "upstream_singleton_pass_count": len(upstream_models),
            "passing_role_model_counts": role_counts,
            "passing_upstream_role_model_counts": upstream_role_counts,
            "two_model_upstream_singleton_repeat": (
                len(upstream_models) >= 2
                and any(value >= 2 for value in upstream_role_counts.values())
            ),
        },
        "interpretation_policy": (
            "Phase1027 falsified only query_nonce_end sufficiency. "
            "Phase1028 determines whether another singleton role has "
            "causal leverage before inferring that multi-position "
            "alliances are necessary."
        ),
        "claim_limit": prereg["claim_limit"],
    }
    final_dir = protocol.OUT_ROOT / "final"
    protocol.write_json(final_dir / "summary.json", summary)

    upstream_repeat = summary["cross_model"][
        "two_model_upstream_singleton_repeat"
    ]
    if upstream_repeat:
        decision = (
            "do not assume a multi-position alliance; independently "
            "replicate the repeated upstream singleton role and then "
            "decompose its component path"
        )
        route = "singleton_replication"
    else:
        decision = (
            "no upstream singleton repeated across two models; proceed to "
            "a preregistered two-position alliance search using only "
            "discovery-frozen role-depth candidates"
        )
        route = "two_position_alliance"
    next_action = {
        "schema_version": "phase1028_automatic_next_action.v1",
        "automatic_next_execution_authorized": True,
        "route": route,
        "decision": decision,
        "authorization_basis": (
            "all five causal roles were scanned over frozen finite depth "
            "grids, then checked on independent confirmation with concept "
            "and position controls"
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
        "discovery_only_selection": all(
            row["run"]["selection_source"] == "discovery_only"
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
        "schema_version": "phase1028_final_audit.v1",
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
