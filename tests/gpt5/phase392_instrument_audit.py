#!/usr/bin/env python3
"""Audit Phase392 parent-boundary hooks before the 24-group causal test."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase392_parent_boundary_replay"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    rows = []
    complete = []
    for model in MODELS:
        root = OUT / "collection/instrument_audit" / model
        complete.append(read_json(root / "complete.json"))
        rows.extend(read_jsonl(root / "direction_rows.jsonl"))
    failures = []
    if len(rows) != 12 or any(not row["valid"] for row in complete):
        failures.append("denominator")
    identity_errors = []
    patch_failures = 0
    generation_patch_failures = 0
    for row in rows:
        scenarios = {item["scenario"]: item for item in row["scenario_rows"]}
        identity = scenarios["identity_semantic_joint"]
        identity_errors.extend(
            [
                abs(identity["query_shift_norm"]),
                abs(identity["donor_margin_shift"]),
                identity["patch_audit"]["max_patch_error"],
                identity["patch_audit"]["max_outside_error"],
            ]
        )
        if scenarios["no_intervention"]["patch_audit"]["patch_call_count"] != 0:
            patch_failures += 1
        for name, scenario in scenarios.items():
            expected = 0 if name == "no_intervention" else 1
            if (
                scenario["patch_audit"]["patch_call_count"] != expected
                or scenario["patch_audit"]["max_patch_error"] > 0.01
                or scenario["patch_audit"]["max_outside_error"] > 0.01
            ):
                patch_failures += 1
        generation = row["joint_generation"]["audit"]
        if (
            generation["patch_call_count"] != 1
            or generation["max_patch_error"] > 0.01
            or generation["max_outside_error"] > 0.01
        ):
            generation_patch_failures += 1
    max_identity_error = max(identity_errors, default=float("inf"))
    if max_identity_error > 0.01:
        failures.append("identity")
    if patch_failures or generation_patch_failures:
        failures.append("patch_audit")
    valid = not failures
    summary = {
        "schema_version": "66.4.0",
        "phase_id": "Phase392-InstrumentAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "group_count": 2,
            "direction_count": len(rows),
            "scenario_count": len(rows) * 8,
            "joint_generation_count": len(rows),
        },
        "results": {
            "max_identity_error": max_identity_error,
            "patch_failure_count": patch_failures,
            "generation_patch_failure_count": generation_patch_failures,
            "valid": valid,
            "language_path_established": False,
        },
        "failures": failures,
        "authorization": {
            "causal_test": valid,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "instrument_validity_is_causal_result": False,
            "parent_boundary_patch_is_language_path": False,
        },
    }
    (OUT / "phase392_instrument_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
