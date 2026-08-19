#!/usr/bin/env python3
"""Independent audit for Phase1362/C055."""
from __future__ import annotations

import json
import math
import py_compile
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1362_c055_coalition_identity_camera"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    manifest = load(OUT / "protocol/execution_manifest.json")
    summary = load(OUT / "analysis/qwen3_coalition_camera.json")
    final = load(OUT / "analysis/final.json")
    data = rows(OUT / "raw/qwen3_coalition_identity.jsonl")
    routes = ["duplicate_no_hook"] + list(manifest["coalitions"])
    metrics_ok = True
    for route in routes:
        for layer in manifest["layers"]:
            values = [row for row in data if row["coalition"] == route and row["layer"] == layer]
            observed = summary["metrics"][route][str(layer)]
            metrics_ok &= len(values) == manifest["calibration_cases"]
            metrics_ok &= abs(max(abs(row["margin_diff"]) for row in values) - observed["max_abs_margin_diff"]) <= 1e-12
            metrics_ok &= abs(statistics.median(abs(row["margin_diff"]) for row in values)
                              - observed["median_abs_margin_diff"]) <= 1e-12
    expected = ("run_phase1363_c055_coalition_causal" if summary["camera_qualified"]
                else "close_c055_camera_unqualified_without_mechanism_claim")
    checks = {
        "record_count": len(data) == manifest["calibration_cases"] * len(manifest["layers"]) * len(routes),
        "finite": all(math.isfinite(row["margin_diff"]) for row in data),
        "metrics_recomputed": metrics_ok,
        "all_routes_registered": set(summary["checks"]) == set(routes),
        "qualification": summary["camera_qualified"] == all(summary["checks"].values()),
        "authorization": final["authorization"] == expected,
        "script_compiles": True,
    }
    try:
        py_compile.compile(str(TESTS / "phase1362_c055_coalition_identity_camera.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    audit = {"phase": 1362, "campaign": "C055", "checks": checks,
             "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
