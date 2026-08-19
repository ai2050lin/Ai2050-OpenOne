#!/usr/bin/env python3
"""Independent audit for Phase1428."""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1425_c070_quartet_complement_contract"
BEHAVIOR = TESTS / "result/phase1426_c070_roster_behavior"
OUT = TESTS / "result/phase1428_c070_support_partition"
ARMS = ("self", "quartet_only", "complement_only", "full_state", "wrong_full_state")
DIRECTIONS = ("true_to_false", "false_to_true")
SPLITS = ("confirmation", "lockbox")


def median(values):
    return float(statistics.median(values))


def metrics(rows, gate, partition_threshold, full_threshold):
    arms = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    desired = {arm: sum(row["desired_sign"] for row in values) / len(values) for arm, values in arms.items()}
    full = {
        "self": max(row["recipient_output_max_abs_diff"] for row in arms["self"]) <= gate["self_max_abs_diff"],
        "desired": desired["full_state"] >= full_threshold,
        "wrong": sum(row["wrong_expected_sign"] for row in arms["wrong_full_state"]) / len(arms["wrong_full_state"]) >= full_threshold,
        "full_donor": median([row["desired_donor_relative_deviation"] for row in arms["full_state"]]) <= gate["full_donor_relative_deviation_median_max"],
        "wrong_donor": median([row["wrong_donor_relative_deviation"] for row in arms["wrong_full_state"]]) <= gate["wrong_donor_relative_deviation_median_max"],
        "gain": median([row["oriented_gain"] for row in arms["full_state"]]) >= gate["full_oriented_gain_median_min"],
    }
    return {
        "full": all(full.values()),
        "quartet": desired["quartet_only"] >= partition_threshold,
        "complement": desired["complement_only"] >= partition_threshold,
    }


def classify(records, families, gate, direction):
    aggregate = {
        split: metrics(
            [row for row in records if row["partition"] == split and row["direction"] == direction],
            gate, gate["aggregate_partition_desired_sign_fraction_min"], gate["full_desired_sign_fraction_min"],
        ) for split in SPLITS
    }
    family = {
        name: {
            split: metrics(
                [row for row in records if row["partition"] == split and row["direction"] == direction and row["family"] == name],
                gate, gate["family_partition_desired_sign_fraction_min"], gate["family_partition_desired_sign_fraction_min"],
            ) for split in SPLITS
        } for name in families
    }
    breadth = {
        key: [name for name, values in family.items() if all(values[split][key] for split in SPLITS)]
        for key in ("full", "quartet", "complement")
    }
    full = all(aggregate[split]["full"] for split in SPLITS) and len(breadth["full"]) >= gate["minimum_family_breadth"]
    quartet = all(aggregate[split]["quartet"] for split in SPLITS) and len(breadth["quartet"]) >= gate["minimum_family_breadth"]
    complement = all(aggregate[split]["complement"] for split in SPLITS) and len(breadth["complement"]) >= gate["minimum_family_breadth"]
    if not full:
        label = "full_transport_failed"
    elif quartet and complement:
        label = "redundant_dual_support"
    elif quartet:
        label = "quartet_dominant"
    elif complement:
        label = "complement_dominant"
    else:
        label = "joint_only_or_unresolved"
    return label, breadth


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    behavior = core.load(BEHAVIOR / "analysis/final.json")
    summary = core.load(OUT / "analysis/support_partition_summary.json")
    final = core.load(OUT / "analysis/final.json")
    records = core.rows(OUT / "raw/support_partition.jsonl")
    gate = protocol["mechanism"]
    recomputed = {}
    breadth = {}
    for direction in DIRECTIONS:
        recomputed[direction], breadth[direction] = classify(records, behavior["qualified_families"], gate, direction)
    classes = set(recomputed.values())
    overall = next(iter(classes)) if len(classes) == 1 else "direction_asymmetric"
    checks = {
        "records": len(records) == 480,
        "arm_balance": all(sum(row["arm"] == arm for row in records) == 96 for arm in ARMS),
        "split_direction_balance": all(sum(row["partition"] == split and row["direction"] == direction for row in records) == 120 for split in SPLITS for direction in DIRECTIONS),
        "set_arm_balance": all(sum(row["set_id"] == set_id and row["direction"] == direction for row in records) == 5 for set_id in {row["set_id"] for row in records} for direction in DIRECTIONS),
        "state16": {row["state_index"] for row in records} == {16},
        "partition": all(len(row["quartet_points"]) == 4 and row["complement_count"] == 66 for row in records),
        "finite": all(math.isfinite(row[key]) for row in records for key in (
            "recipient_margin", "desired_donor_margin", "wrong_donor_margin", "swap_margin", "oriented_gain",
            "desired_donor_relative_deviation", "wrong_donor_relative_deviation", "recipient_output_max_abs_diff",
            "desired_output_max_abs_diff", "wrong_output_max_abs_diff",
        )),
        "direction_classification": all(final["direction_results"][direction]["classification"] == recomputed[direction] for direction in DIRECTIONS),
        "family_breadth": all(
            final["direction_results"][direction][f"{key}_qualified_families"] == breadth[direction][key]
            for direction in DIRECTIONS for key in ("full", "quartet", "complement")
        ),
        "overall": final["overall_classification"] == summary["overall_classification"] == overall,
        "contract": summary["contract_sha256"] == protocol["contract_sha256"],
        "execution": summary["all_execution_checks_passed"] and all(summary["checks"].values()),
        "authorization": final["authorization"] == "run_phase1429_c070_campaign_closure",
    }
    result = {
        "phase": 1428,
        "campaign": "C070",
        "checks": checks,
        "recomputed_classification": recomputed,
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
