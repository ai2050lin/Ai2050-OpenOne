#!/usr/bin/env python3
"""Independent audit for Phase1421."""
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

CONTRACT = TESTS / "result/phase1420_c069_catalog_four_role_contract"
OUT = TESTS / "result/phase1421_c069_catalog_behavior"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/active_behavior.jsonl")
    selected = core.rows(OUT / "material/eligible_composition_sets.jsonl")
    expected = "run_phase1422_c069_quartet_camera" if summary["behavior_qualified"] else "close_c069_at_behavior_gate"
    recomputed_qualified = [
        family for family, result in summary["family_results"].items()
        if all(result["checks"].values())
    ]
    checks = {
        "rows": len(rows) == 2880,
        "cells": Counter(row["cell"] for row in rows) == {
            cell: 360 for cell in ("aa", "ab", "ac", "ad", "bb", "ba", "bc", "bd")
        },
        "truth": Counter(row["truth"] for row in rows) == {True: 720, False: 2160},
        "surfaces": Counter(row["surface"] for row in rows) == {"ordinary": 1440, "catalog": 1440},
        "finite": all(math.isfinite(value) for row in rows for value in row["scores"]),
        "decision": final["qualified_families"] == summary["qualified_families"] == recomputed_qualified,
        "selected": len(selected) == 12 * len(summary["qualified_families"]),
        "partition_balance": all(
            sum(row["family"] == family and row["partition"] == partition for row in selected) == 4
            for family in summary["qualified_families"]
            for partition in protocol["material"]["partitions"]
        ),
        "breadth": summary["behavior_qualified"] == all(summary["breadth_checks"].values()),
        "authorization": final["authorization"] == expected,
        "hidden_not_accessed": summary["hidden_state_accessed"] is False,
    }
    result = {
        "phase": 1421,
        "campaign": "C069",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
