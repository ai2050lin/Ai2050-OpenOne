#!/usr/bin/env python3
"""Independent audit for Phase1461 C078 behavior."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1460_c078_colon_label_contract"
OUT = TESTS / "result/phase1461_c078_behavior"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/active_behavior.jsonl")
    eligible = core.rows(OUT / "material/eligible_composition_sets.jsonl")
    composition = core.rows(CONTRACT / "material/composition_sets.jsonl")
    by_case = {row["case_id"]: row for row in rows}
    set_keys = tuple(f"{surface}_{cell}" for surface in protocol["surfaces"] for cell in protocol["cells"])
    expected_eligible = [row for row in composition if all(by_case[row[key]]["correct"] for key in set_keys)]
    expected_auth = "run_phase1462_c078_discovery_full_field_capture" if summary["behavior_qualified"] else "close_c078_at_behavior_gate"
    checks = {
        "rows": len(rows) == 3456,
        "truth": Counter(row["truth"] for row in rows) == {True: 1728, False: 1728},
        "semantic": all(row["truth"] == (row["record_label"] == row["query_label"]) for row in rows),
        "finite": all(math.isfinite(value) for row in rows for value in row["scores"]),
        "eligible_exact": [row["set_id"] for row in eligible] == [row["set_id"] for row in expected_eligible],
        "eligible_complete": all(all(by_case[row[key]]["correct"] for key in set_keys) for row in eligible),
        "counts": len(eligible) == summary["eligible_count"] and Counter(row["record_relation_id"] for row in eligible) == Counter(summary["eligible_relation_counts"]),
        "decision": summary["behavior_qualified"] == all(summary["checks"].values()),
        "authorization": final["authorization"] == expected_auth,
        "hidden": summary["hidden_state_accessed"] is False,
    }
    result = {"phase": 1461, "campaign": "C078", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
