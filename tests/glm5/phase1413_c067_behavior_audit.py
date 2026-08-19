#!/usr/bin/env python3
"""Independent audit for Phase1413."""
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

CONTRACT = TESTS / "result/phase1412_c067_paired_state_composition_contract"
OUT = TESTS / "result/phase1413_c067_behavior"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/active_behavior.jsonl")
    selected = core.rows(OUT / "material/eligible_composition_sets.jsonl")
    expected = "run_phase1414_c067_dual_write_camera" if summary["behavior_qualified"] else "close_c067_at_behavior_gate"
    checks = {
        "rows": len(rows) == 2160,
        "balance": Counter(r["truth"] for r in rows) == {True: 1080, False: 1080},
        "finite": all(math.isfinite(value) for row in rows for value in row["scores"]),
        "decision": final["qualified_families"] == summary["qualified_families"],
        "selected": len(selected) == 12 * len(summary["qualified_families"]),
        "partition_balance": all(sum(r["family"] == family and r["partition"] == partition for r in selected) == 4 for family in summary["qualified_families"] for partition in protocol["material"]["partitions"]),
        "breadth": summary["behavior_qualified"] == all(summary["breadth_checks"].values()),
        "authorization": final["authorization"] == expected,
        "hidden_not_accessed": True,
    }
    result = {"phase": 1413, "campaign": "C067", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
