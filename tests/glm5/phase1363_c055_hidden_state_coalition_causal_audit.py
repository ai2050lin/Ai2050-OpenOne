#!/usr/bin/env python3
"""Independent audit for Phase1363/C055."""
from __future__ import annotations

import json
import math
import py_compile
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1363_c055_hidden_state_coalition_causal"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    manifest = load(OUT / "protocol/execution_manifest.json")
    summary = load(OUT / "analysis/qwen3_coalition_causal.json")
    final = load(OUT / "analysis/final.json")
    data = rows(OUT / "raw/qwen3_coalition_causal.jsonl")
    metrics_ok = True
    for name in manifest["coalitions"]:
        correct = [row["gains"][name]["correct_true"] for row in data]
        controls = [[row["gains"][name][arm] for arm in
                     ("wrong_family_true", "same_family_false", "status_true")] for row in data]
        advantage = [value - max(wrong) for value, wrong in zip(correct, controls)]
        wins = [value > max(wrong) for value, wrong in zip(correct, controls)]
        expected = summary["route_metrics"][name]
        metrics_ok &= abs(statistics.median(correct) - expected["correct_gain_median"]) <= 1e-12
        metrics_ok &= abs(sum(value > 0 for value in correct) / len(correct)
                          - expected["correct_direction_fraction"]) <= 1e-12
        metrics_ok &= abs(statistics.median(advantage) - expected["correct_over_controls_median"]) <= 1e-12
        metrics_ok &= abs(sum(wins) / len(wins) - expected["correct_over_controls_win_fraction"]) <= 1e-12
    expected_auth = ("run_phase1364_c055_necessity_rescue" if summary["any_multi_qualified"]
                     else "close_c055_at_hidden_state_coalition_boundary")
    checks = {
        "record_count": len(data) == manifest["recipient_count"],
        "finite": all(math.isfinite(value) for row in data
                      for coalition in row["gains"].values() for value in coalition.values()),
        "metrics_recomputed": metrics_ok,
        "self_controls": all(summary["route_checks"][name]["self"] for name in manifest["coalitions"]),
        "route_qualification": all(summary["route_qualified"][name] == all(summary["route_checks"][name].values())
                                   for name in manifest["coalitions"]),
        "multi_qualification": all(summary["multi_qualified"][name] == all(summary["multi_synergy"][name]["checks"].values())
                                   for name in summary["multi_qualified"]),
        "authorization": final["authorization"] == expected_auth,
        "script_compiles": True,
    }
    try:
        py_compile.compile(str(TESTS / "phase1363_c055_hidden_state_coalition_causal.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    audit = {"phase": 1363, "campaign": "C055", "checks": checks,
             "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
