#!/usr/bin/env python3
"""Independent result audit for Phase1358/C054."""
from __future__ import annotations

import json
import math
import py_compile
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1358_c054_identity_camera_calibration"
ROUTES = ("duplicate_no_hook", "same_batch_exact_token", "cached_fixed_shape_exact_token",
          "same_batch_span_mean_diagnostic", "same_batch_zero_delta")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    manifest = load(OUT / "protocol/execution_manifest.json")
    summary = load(OUT / "analysis/qwen3_camera_summary.json")
    final = load(OUT / "analysis/final.json")
    data = rows(OUT / "raw/qwen3_identity_camera.jsonl")
    metrics_ok = True
    for route in ROUTES:
        for layer in manifest["layers"]:
            values = [row for row in data if row["route"] == route and row["layer"] == layer]
            expected = summary["route_metrics"][route][str(layer)]
            metrics_ok &= len(values) == manifest["case_count"]
            metrics_ok &= abs(max(abs(row["margin_diff"]) for row in values) - expected["max_abs_margin_diff"]) <= 1e-12
            metrics_ok &= abs(statistics.median(abs(row["margin_diff"]) for row in values)
                              - expected["median_abs_margin_diff"]) <= 1e-12
    expected_authorization = ("run_phase1359_c054_same_batch_causal_replay" if summary["camera_qualified"]
                              else "close_c054_camera_unqualified_without_mechanism_claim")
    checks = {
        "record_count": len(data) == manifest["case_count"] * len(manifest["layers"]) * len(ROUTES),
        "finite": all(math.isfinite(row["margin_diff"]) and math.isfinite(row["candidate_logit_max_abs_diff"])
                      for row in data),
        "metrics_recomputed": metrics_ok,
        "duplicate_control": summary["route_checks"]["duplicate_no_hook"],
        "zero_delta_control": summary["route_checks"]["same_batch_zero_delta"],
        "priority_selection": summary["selected_camera_route"] == next(
            (route for route in manifest["authorized_priority"] if summary["route_checks"][route]), None),
        "qualification": summary["camera_qualified"] == (
            summary["selected_camera_route"] is not None
            and summary["route_checks"]["duplicate_no_hook"]
            and summary["route_checks"]["same_batch_zero_delta"]),
        "authorization": final["authorization"] == expected_authorization,
        "script_compiles": True,
    }
    try:
        py_compile.compile(str(TESTS / "phase1358_c054_identity_camera_calibration.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    result = {"phase": 1358, "campaign": "C054", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
