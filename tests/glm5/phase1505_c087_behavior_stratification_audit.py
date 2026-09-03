#!/usr/bin/env python3
"""Independent audit for Phase1505."""
from __future__ import annotations

import json
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1505_c087_behavior_stratification"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    rows = core.rows(OUT / "raw/behavior.jsonl")
    groups = core.rows(OUT / "material/stratified_composition_sets.jsonl")
    summary = core.load(OUT / "analysis/behavior_stratification_summary.json")
    py_compile.compile(str(TESTS / "phase1505_c087_behavior_stratification.py"), doraise=True)
    checks = {
        "prediction": all(row["prediction"] == max(range(2), key=lambda i: row["scores"][i]) for row in rows),
        "correct": all(row["correct"] == (row["prediction"] == row["gold_position"]) for row in rows),
        "summary": abs(sum(row["correct"] for row in rows) / len(rows) - summary["global_accuracy"]) < 1e-12,
        "strata": Counter(row["stratum"] for row in groups) == summary["stratum_counts"],
        "integrity": all(summary["checks"].values()),
    }
    result = {"phase": 1505, "campaign": "C087", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
