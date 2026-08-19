#!/usr/bin/env python3
"""Independent audit for Phase1415."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1415_c067_paired_composition as phase

CONTRACT = TESTS / "result/phase1412_c067_paired_state_composition_contract"
OUT = TESTS / "result/phase1415_c067_paired_composition"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/composition_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/paired_composition.jsonl")
    gate = protocol["mechanism"]
    recomputed_splits = {
        split: phase.group_metrics([row for row in rows if row["partition"] == split], gate)
        for split in ("confirmation", "lockbox")
    }
    recomputed_families = []
    for family in summary["family_metrics"]:
        values = [phase.group_metrics([row for row in rows if row["partition"] == split and row["family"] == family], gate) for split in ("confirmation", "lockbox")]
        if all(value["qualified"] for value in values):
            recomputed_families.append(family)
    recomputed_confirmed = all(value["qualified"] for value in recomputed_splits.values()) and len(recomputed_families) >= gate["minimum_family_breadth"]
    checks = {
        "rows": len(rows) == 432,
        "sets": len({row["set_id"] for row in rows}) == 48,
        "arms": {row["arm"] for row in rows} == set(phase.ARMS) and all(sum(other["set_id"] == row["set_id"] for other in rows) == 9 for row in rows[::9]),
        "holdout_only": {row["partition"] for row in rows} == {"confirmation", "lockbox"},
        "split_count": all(len([row for row in rows if row["partition"] == split]) == 216 for split in ("confirmation", "lockbox")),
        "state16": {row["state_index"] for row in rows} == {16},
        "finite": all(math.isfinite(row[key]) for row in rows for key in ("baseline_margin", "swap_margin", "signed_damage", "loss_fraction")),
        "split_recompute": recomputed_splits == summary["split_metrics"],
        "family_recompute": sorted(recomputed_families) == sorted(summary["qualified_families"]),
        "decision": recomputed_confirmed == summary["composition_confirmed"] == final["composition_confirmed"],
        "authorization": final["authorization"] == "run_phase1416_c067_campaign_closure",
    }
    result = {
        "phase": 1415,
        "campaign": "C067",
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
