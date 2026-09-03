#!/usr/bin/env python3
"""Independent audit for Phase1446 C074 behavior qualification."""
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

PHASE, CAMPAIGN = 1446, "C074"
CONTRACT = TESTS / "result/phase1445_c074_directional_domain_contract"
OUT = TESTS / "result/phase1446_c074_behavior"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/active_behavior.jsonl")
    selected = core.rows(OUT / "material/eligible_composition_sets.jsonl")
    expected = "run_phase1447_c074_identity_camera" if summary["behavior_qualified"] else "close_c074_at_behavior_gate"
    recomputed = [family for family, result in summary["family_results"].items() if all(result["checks"].values()) and all(value["qualified"] for value in result["surfaces"].values())]
    checks = {
        "rows": len(rows) == 5760,
        "surfaces": Counter(row["surface"] for row in rows) == {surface: 1440 for surface in protocol["surfaces"]},
        "cells": Counter(row["cell"] for row in rows) == {cell: 720 for cell in ("aa", "ab", "ac", "ad", "bb", "ba", "bc", "bd")},
        "truth": Counter(row["truth"] for row in rows) == {True: 1440, False: 4320},
        "semantic": all(row["truth"] == (row["record_target"] == row["query_target"] and row["record_family"] == row["query_family"]) for row in rows),
        "finite": all(math.isfinite(value) for row in rows for value in row["scores"]),
        "decision": final["qualified_families"] == summary["qualified_families"] == recomputed,
        "selected": len(selected) == 12 * len(summary["qualified_families"]),
        "partition_balance": all(sum(row["family"] == family and row["partition"] == split for row in selected) == 4 for family in summary["qualified_families"] for split in protocol["partitions"]),
        "surface_gate": all(value["balanced_accuracy"] >= protocol["zero_model_gate"]["required_model_balanced_accuracy_min"] for value in summary["surface_global"].values()),
        "zero_model_gap": min(value["balanced_accuracy"] for value in summary["surface_global"].values()) > protocol["zero_model_gate"]["maximum_incomplete_balanced_accuracy"],
        "breadth": summary["behavior_qualified"] == all(summary["breadth_checks"].values()),
        "authorization": final["authorization"] == expected,
        "hidden_not_accessed": summary["hidden_state_accessed"] is False,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
