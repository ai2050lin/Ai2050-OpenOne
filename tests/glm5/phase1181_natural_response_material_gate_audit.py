#!/usr/bin/env python3
"""Independent integrity and numerical audit for Phase1181."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1181_natural_response_material_gate as phase  # noqa: E402


AUDIT_PATH = phase.OUT_ROOT / "audit/independent_audit.json"


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-split", type=int, default=8)
    args = parser.parse_args()
    protocol = phase.validate_protocol()
    checks: list[dict[str, Any]] = []
    add(checks, "final_exists", phase.FINAL_PATH.exists())
    if not phase.FINAL_PATH.exists():
        raise RuntimeError("missing final result")
    final = phase.read_json(phase.FINAL_PATH)
    for split, rows_path, summary_path, expected_count in (
        ("discovery", phase.DISCOVERY_ROWS, phase.DISCOVERY_SUMMARY, 64),
        ("confirmation", phase.CONFIRMATION_ROWS, phase.CONFIRMATION_SUMMARY, 32),
    ):
        rows = phase.read_jsonl(rows_path)
        summary = phase.read_json(summary_path)
        add(checks, f"{split}_row_count", len(rows) == expected_count, len(rows))
        add(checks, f"{split}_unique_checkpoints", len({row["checkpoint"] for row in rows}) == expected_count)
        add(checks, f"{split}_rows_digest", phase.digest(rows) == summary["rows_digest"])
        recomputed_summary = phase.summarize(rows, split, protocol["thresholds"])
        comparison_keys = (
            "system_count",
            "qualified_system_count",
            "maximum_replay_error",
            "maximum_gauge_fp32_logit_error",
            "maximum_gauge_ordered_response_error",
            "response_scale_coefficient_of_variation",
            "numerical_pass",
            "passing_task_count",
            "split_pass",
        )
        for key in comparison_keys:
            left, right = summary[key], recomputed_summary[key]
            passed = abs(left - right) <= 1e-12 if isinstance(left, float) else left == right
            add(checks, f"{split}_summary_{key}", passed, {"stored": left, "recomputed": right})
        manifest = protocol["checkpoint_manifests"][split]
        add(
            checks,
            f"{split}_checkpoint_hashes",
            all(row["checkpoint_sha256"] == manifest[next(key for key in manifest if key.endswith(row["checkpoint"]))] for row in rows),
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for numerical audit")
        device = torch.device("cuda")
        selected_indices = np.linspace(0, len(rows) - 1, num=min(args.sample_per_split, len(rows)), dtype=int)
        maximum_response_error = 0.0
        maximum_behavior_error = 0.0
        source_paths = {path.name: path for path in phase.endpoint_paths(split)}
        for index in selected_indices:
            row = rows[int(index)]
            payload = torch.load(source_paths[row["checkpoint"]], map_location="cpu", weights_only=False)
            model = phase.load_model(payload, device)
            panel = phase.load_panel(payload, split)
            response = phase.response_spectrum(model, panel, device)
            behavior = phase.behavior_metrics(model, panel, device)
            maximum_response_error = max(
                maximum_response_error,
                float(np.max(np.abs(np.asarray(response["ordered"]) - np.asarray(row["response"]["ordered"])))),
            )
            maximum_behavior_error = max(
                maximum_behavior_error,
                max(abs(float(behavior[name]) - float(row["behavior"][name])) for name in phase.BEHAVIOR_FEATURES),
            )
            del model
            torch.cuda.empty_cache()
        add(checks, f"{split}_sample_response_recompute", maximum_response_error <= 1e-7, maximum_response_error)
        add(checks, f"{split}_sample_behavior_recompute", maximum_behavior_error <= 1e-7, maximum_behavior_error)
    add(
        checks,
        "final_primary_decision",
        final["primary_pass"]
        == bool(final["discovery"]["split_pass"] and final["confirmation"]["split_pass"]),
    )
    add(checks, "auto_continue_matches_primary", final["auto_continue"]["authorized"] == final["primary_pass"])
    integrity_pass = all(check["passed"] for check in checks)
    audit = {
        "phase": phase.PHASE,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol["protocol_digest"],
        "integrity_and_recompute_pass": integrity_pass,
        "scientific_primary_pass": final["primary_pass"],
        "check_count": len(checks),
        "passed_check_count": sum(check["passed"] for check in checks),
        "checks": checks,
    }
    audit["audit_digest"] = phase.digest(audit)
    phase.write_json(AUDIT_PATH, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not integrity_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
