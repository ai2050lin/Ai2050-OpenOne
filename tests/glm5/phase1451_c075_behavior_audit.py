#!/usr/bin/env python3
"""Independent audit for Phase1451 C075 behavior qualification."""
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

PHASE, CAMPAIGN = 1451, "C075"
CONTRACT = TESTS / "result/phase1450_c075_full_field_atlas_contract"
OUT = TESTS / "result/phase1451_c075_behavior"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/active_behavior.jsonl")
    selected = core.rows(OUT / "material/eligible_composition_sets.jsonl")
    expected = "run_phase1452_c075_discovery_full_field_capture" if summary["behavior_qualified"] else "close_c075_at_behavior_gate"
    recomputed = [relation for relation, result in summary["relation_results"].items() if result["qualified"] and all(result["checks"].values())]
    checks = {
        "rows": len(rows) == 3456,
        "surfaces": Counter(row["surface"] for row in rows) == {surface: 1728 for surface in protocol["surfaces"]},
        "cells": Counter(row["cell"] for row in rows) == {cell: 432 for cell in protocol["cells"]},
        "truth": Counter(row["truth"] for row in rows) == {True: 432, False: 3024},
        "semantic": all(row["truth"] == (row["entity_match"] and row["object_match"] and row["relation_match"]) for row in rows),
        "finite": all(math.isfinite(value) for row in rows for value in row["scores"]),
        "decision": final["qualified_relations"] == summary["qualified_relations"] == recomputed,
        "selected": len(selected) == summary["selected_count"] and len(selected) == 36 * len(summary["qualified_relations"]),
        "partition_balance": all(sum(row["record_relation"] == relation and row["partition"] == split for row in selected) == 12 for relation in summary["qualified_relations"] for split in protocol["partitions"]),
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
