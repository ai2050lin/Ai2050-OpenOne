#!/usr/bin/env python3
"""Independent audit for Phase1374."""
from __future__ import annotations

import json
import math
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1374_c058_qwen_behavior_qualification"
SCRIPT = TESTS / "phase1374_c058_qwen_behavior_qualification.py"


def main() -> None:
    summary = core.load(OUT / "analysis/qwen3_behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    active = core.rows(OUT / "raw/active_behavior.jsonl")
    status = core.rows(OUT / "raw/status_behavior.jsonl")
    selected = core.rows(OUT / "material/eligible_pairs.jsonl")
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    checks = {
        "counts": len(active) == 864 and len(status) == 288,
        "finite": all(math.isfinite(r["margin"]) for r in active + status),
        "active_accuracy": abs(summary["active"]["accuracy"] - sum(r["correct"] for r in active) / len(active)) < 1e-12,
        "status_accuracy": abs(summary["status"]["accuracy"] - sum(r["correct"] for r in status) / len(status)) < 1e-12,
        "selected_matches_gate": (len(selected) == 288 and len({r["pair_id"] for r in selected}) == 288)
                                 if summary["behavior_qualified"] else len(selected) == 0,
        "partition_matches_gate": (Counter(r["partition"] for r in selected) ==
                                   {"response_discovery": 96, "confirmation": 96, "lockbox": 96})
                                  if summary["behavior_qualified"] else True,
        "cell_matches_gate": (set(Counter((r["target_family"], r["partition"], r["surface"])
                                          for r in selected).values()) == {8})
                             if summary["behavior_qualified"] else True,
        "checks_consistent": summary["behavior_qualified"] == all(summary["checks"].values()),
        "final_consistent": final["behavior_qualified"] == summary["behavior_qualified"],
        "authorization": final["authorization"] ==
            ("run_phase1375_c058_instrument_calibration" if summary["behavior_qualified"]
             else "close_c058_behavior_unqualified_before_hidden_access"),
        "scripts_compile": True,
    }
    audit = {"phase": 1374, "campaign": "C058", "checks": checks,
             "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
