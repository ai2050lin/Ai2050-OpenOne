#!/usr/bin/env python3
"""Independent audit for Phase1454 C076 behavior."""
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

PHASE, CAMPAIGN = 1454, "C076"
CONTRACT = TESTS / "result/phase1453_c076_relation_discrimination_contract"
OUT = TESTS / "result/phase1454_c076_behavior"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/active_behavior.jsonl")
    eligible = core.rows(OUT / "material/eligible_composition_sets.jsonl")
    expected = "run_phase1455_c076_discovery_full_field_capture" if summary["behavior_qualified"] else "close_c076_at_behavior_gate"
    checks = {
        "rows": len(rows) == 3456,
        "surfaces": Counter(row["surface"] for row in rows) == {surface: 1728 for surface in protocol["surfaces"]},
        "truth": Counter(row["truth"] for row in rows) == {True: 1728, False: 1728},
        "semantic": all(row["truth"] == (row["record_relation_id"] == row["query_relation_id"]) for row in rows),
        "nuisance": all(Counter(row[key] for row in rows) == {True: 1728, False: 1728} for key in ("entity_match", "object_match", "relation_match")),
        "finite": all(math.isfinite(value) for row in rows for value in row["scores"]),
        "eligible": len(eligible) == summary["eligible_count"] and {split: sum(row["partition"] == split for row in eligible) for split in protocol["partitions"]} == summary["eligible_partition_counts"],
        "relations": final["qualified_relations"] == summary["qualified_relations"] and set(summary["qualified_relations"]) <= set(protocol["relations"]),
        "surface_gate": summary["checks"]["all_surfaces"] == all(value["balanced_accuracy"] >= protocol["behavior"]["global_surface_balanced_accuracy_min"] for value in summary["surface_global"].values()),
        "decision": summary["behavior_qualified"] == all(summary["checks"].values()),
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
