#!/usr/bin/env python3
"""Independent audit for Phase1423."""
from __future__ import annotations

import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1423_c069_bidirectional_composition as phase

CONTRACT = TESTS / "result/phase1420_c069_catalog_four_role_contract"
OUT = TESTS / "result/phase1423_c069_bidirectional_composition"
SPLITS = ("confirmation", "lockbox")


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/composition_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/bidirectional_composition.jsonl")
    gate = protocol["mechanism"]
    recomputed = {
        split: {
            direction: phase.group_metrics(
                [row for row in rows if row["partition"] == split and row["direction"] == direction], gate, direction
            )
            for direction in phase.DIRECTIONS
        }
        for split in SPLITS
    }
    graded_families, discrete_families, strong_families = [], [], []
    for family in summary["family_metrics"]:
        groups = [
            phase.group_metrics(
                [row for row in rows if row["partition"] == split and row["direction"] == direction and row["family"] == family],
                gate, direction,
            )
            for split in SPLITS for direction in phase.DIRECTIONS
        ]
        if all(group["graded_qualified"] for group in groups): graded_families.append(family)
        if all(group["discrete_qualified"] for group in groups): discrete_families.append(family)
        if all(group["strong_qualified"] for group in groups): strong_families.append(family)
    aggregate = [recomputed[split][direction] for split in SPLITS for direction in phase.DIRECTIONS]
    graded = all(group["graded_qualified"] for group in aggregate) and len(graded_families) >= gate["minimum_family_breadth"]
    discrete = all(group["discrete_qualified"] for group in aggregate) and len(discrete_families) >= gate["minimum_family_breadth"]
    strong = graded and discrete and len(strong_families) >= gate["minimum_family_breadth"]
    checks = {
        "rows": len(rows) == 864,
        "sets": len({row["set_id"] for row in rows}) == 48,
        "balanced_arms": all(
            sum(other["set_id"] == set_id and other["direction"] == direction for other in rows) == 9
            for set_id in {row["set_id"] for row in rows} for direction in phase.DIRECTIONS
        ),
        "arms": {row["arm"] for row in rows} == set(phase.ARMS),
        "holdout_only": {row["partition"] for row in rows} == set(SPLITS),
        "split_direction_count": all(
            len([row for row in rows if row["partition"] == split and row["direction"] == direction]) == 216
            for split in SPLITS for direction in phase.DIRECTIONS
        ),
        "state16": {row["state_index"] for row in rows} == {16},
        "finite": all(math.isfinite(row[key]) for row in rows for key in ("baseline_margin", "swap_margin", "relative_deviation")),
        "split_recompute": recomputed == summary["split_direction_metrics"],
        "family_recompute": sorted(graded_families) == sorted(summary["graded_qualified_families"])
            and sorted(discrete_families) == sorted(summary["discrete_qualified_families"])
            and sorted(strong_families) == sorted(summary["strong_qualified_families"]),
        "decision": (graded, discrete, strong) == (
            summary["graded_confirmed"], summary["discrete_confirmed"], summary["strong_confirmed"]
        ) == (final["graded_confirmed"], final["discrete_confirmed"], final["strong_confirmed"]),
        "authorization": final["authorization"] == "run_phase1424_c069_campaign_closure",
    }
    result = {"phase": 1423, "campaign": "C069", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
