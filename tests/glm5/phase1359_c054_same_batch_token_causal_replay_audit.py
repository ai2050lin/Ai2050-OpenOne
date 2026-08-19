#!/usr/bin/env python3
"""Independent result audit for Phase1359/C054."""
from __future__ import annotations

import json
import math
import py_compile
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1359_c054_same_batch_token_causal_replay"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recompute(data: list[dict], route: str):
    if route == "state_transport":
        correct = [row["gains"]["state_correct"] for row in data]
        wrong = [[row["gains"]["state_wrong_true"], row["gains"]["state_same_false"]] for row in data]
    else:
        correct = [row["gains"]["delta_correct"] for row in data]
        wrong = [[row["gains"]["delta_wrong"]] for row in data]
    advantage = [value - max(values) for value, values in zip(correct, wrong)]
    wins = [value > max(values) for value, values in zip(correct, wrong)]
    return {
        "correct_gain_median": statistics.median(correct),
        "correct_direction_fraction": sum(value > 0 for value in correct) / len(correct),
        "correct_over_wrong_median": statistics.median(advantage),
        "correct_over_wrong_win_fraction": sum(wins) / len(wins),
    }


def main() -> None:
    manifest = load(OUT / "protocol/execution_manifest.json")
    summary = load(OUT / "analysis/qwen3_causal_summary.json")
    final = load(OUT / "analysis/final.json")
    data = rows(OUT / "raw/qwen3_same_batch_causal.jsonl")
    metrics_ok = True
    for route in manifest["routes"]:
        expected = recompute(data, route)
        observed = summary["route_metrics"][route]
        metrics_ok &= all(abs(expected[key] - observed[key]) <= 1e-12 for key in expected)
    self_max = max(abs(row["gains"]["self"]) for row in data)
    zero_max = max(abs(row["gains"]["zero_delta"]) for row in data)
    expected_authorization = ("close_c054_with_calibrated_causal_candidate" if summary["any_route_qualified"]
                              else "close_c054_at_calibrated_causal_selectivity_boundary")
    checks = {
        "record_count": len(data) == manifest["recipient_count"] == 324,
        "finite": all(math.isfinite(value) for row in data
                      for values in (row["margins"], row["gains"]) for value in values.values()),
        "metrics_recomputed": metrics_ok,
        "self_recomputed": abs(self_max - summary["identity"]["self_max_abs_diff"]) <= 1e-12,
        "zero_recomputed": abs(zero_max - summary["identity"]["zero_delta_max_abs_diff"]) <= 1e-12,
        "identity_qualified": all(summary["identity_checks"].values()),
        "route_qualification": all(summary["route_qualified"][route] == all(summary["route_checks"][route].values())
                                   for route in manifest["routes"]),
        "global_qualification": summary["any_route_qualified"] == any(summary["route_qualified"].values()),
        "authorization": final["authorization"] == expected_authorization,
        "script_compiles": True,
    }
    try:
        py_compile.compile(str(TESTS / "phase1359_c054_same_batch_token_causal_replay.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    result = {"phase": 1359, "campaign": "C054", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
