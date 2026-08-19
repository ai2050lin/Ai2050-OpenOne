#!/usr/bin/env python3
"""Independent audit for Phase1443 C073 one-shot side/phase competition."""
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

CONTRACT = TESTS / "result/phase1440_c073_side_phase_contract"
BEHAVIOR = TESTS / "result/phase1441_c073_behavior"
OUT = TESTS / "result/phase1443_c073_side_phase_competition"
SPLITS = ("confirmation", "lockbox")
DIRECTIONS = ("true_to_false", "false_to_true")


def med(values):
    return float(statistics.median(values))


def frac(rows, field):
    return sum(bool(row[field]) for row in rows) / len(rows)


def recompute_cell(rows, families, gate):
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in gate["arms"]}
    threshold = gate["family_paired_win_fraction_min"]
    identity_breadth = [family for family in families if frac([row for row in by_arm["correct_identity"] if row["family"] == family], "desired_sign") >= threshold]
    wrong_breadth = [family for family in families if frac([row for row in by_arm["wrong_identity"] if row["family"] == family], "wrong_expected_sign") >= threshold]
    executor = (
        max(row["recipient_output_max_abs_diff"] for row in by_arm["self"]) <= gate["self_max_abs_diff"]
        and frac(by_arm["correct_identity"], "desired_sign") >= gate["identity_desired_sign_fraction_min"]
        and frac(by_arm["wrong_identity"], "wrong_expected_sign") >= gate["wrong_expected_sign_fraction_min"]
        and len(identity_breadth) >= gate["minimum_family_breadth"]
        and len(wrong_breadth) >= gate["minimum_family_breadth"]
    )
    indexed = {arm: {row["set_id"]: row for row in values} for arm, values in by_arm.items()}
    set_ids = sorted(indexed["p07"])
    gaps = [indexed["p07"][set_id]["oriented_gain"] - indexed["p23"][set_id]["oriented_gain"] for set_id in set_ids]
    semantic_breadth, physical_breadth = [], []
    for family in families:
        ids = [set_id for set_id in set_ids if indexed["p07"][set_id]["family"] == family]
        if sum(indexed["p07"][set_id]["oriented_gain"] > indexed["p23"][set_id]["oriented_gain"] for set_id in ids) / len(ids) >= gate["family_paired_win_fraction_min"]:
            semantic_breadth.append(family)
        if sum(indexed["p23"][set_id]["oriented_gain"] > indexed["p07"][set_id]["oriented_gain"] for set_id in ids) / len(ids) >= gate["family_paired_win_fraction_min"]:
            physical_breadth.append(family)
    semantic_fraction = sum(value > 0 for value in gaps) / len(gaps)
    physical_fraction = sum(value < 0 for value in gaps) / len(gaps)
    gap = med(gaps)
    semantic = executor and frac(by_arm["p07"], "desired_sign") >= gate["arm_desired_sign_fraction_min"] and semantic_fraction >= gate["paired_win_fraction_min"] and gap >= gate["paired_gain_gap_median_min"] and len(semantic_breadth) >= gate["minimum_family_breadth"]
    physical = executor and frac(by_arm["p23"], "desired_sign") >= gate["arm_desired_sign_fraction_min"] and physical_fraction >= gate["paired_win_fraction_min"] and gap <= -gate["paired_gain_gap_median_min"] and len(physical_breadth) >= gate["minimum_family_breadth"]
    return executor, semantic, physical


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    behavior = core.load(BEHAVIOR / "analysis/final.json")
    rows = core.rows(OUT / "raw/side_phase_competition.jsonl")
    summary = core.load(OUT / "analysis/side_phase_summary.json")
    final = core.load(OUT / "analysis/final.json")
    reveal = core.load(OUT / "protocol/reveal_manifest.json")
    gate = protocol["mechanism"]
    recomputed = {}
    executor_count = semantic_reversed = physical_reversed = semantic_same = physical_same = 0
    for route in gate["routes"]:
        recomputed[route] = {}
        for direction in DIRECTIONS:
            recomputed[route][direction] = {}
            for split in SPLITS:
                cell_rows = [row for row in rows if row["route"] == route and row["direction"] == direction and row["partition"] == split]
                executor, semantic, physical = recompute_cell(cell_rows, behavior["qualified_families"], gate)
                recomputed[route][direction][split] = {"executor": executor, "semantic": semantic, "physical": physical}
                executor_count += executor
                if route in gate["reversed_routes"]:
                    semantic_reversed += semantic; physical_reversed += physical
                else:
                    semantic_same += semantic; physical_same += physical
    counts = {
        "total_executor_pass": executor_count,
        "total_cells": 16,
        "reversed_semantic_winners": semantic_reversed,
        "reversed_physical_winners": physical_reversed,
        "same_order_semantic_winners": semantic_same,
        "same_order_physical_winners": physical_same,
    }
    execution_ok = all(summary["checks"].values())
    if not execution_ok or executor_count != 16:
        expected = "executor_failed"
    elif semantic_reversed == gate["strong_required_reversed_cells"] and semantic_same >= gate["strong_required_same_order_cells"]:
        expected = "semantic_side_confirmed"
    elif physical_reversed == gate["strong_required_reversed_cells"]:
        expected = "physical_phase_confirmed"
    elif semantic_reversed >= gate["conditional_required_reversed_cells"] and physical_reversed == 0:
        expected = "conditional_semantic_side"
    elif physical_reversed >= gate["conditional_required_reversed_cells"] and semantic_reversed == 0:
        expected = "conditional_physical_phase"
    else:
        expected = "mixed_or_no_stable_separation"
    checks = {
        "count": len(rows) == gate["holdout_sets"] * len(gate["routes"]) * len(gate["directions"]) * len(gate["arms"]),
        "balance": all(sum(row["partition"] == split and row["route"] == route and row["direction"] == direction and row["arm"] == arm for row in rows) == 24 for split in SPLITS for route in gate["routes"] for direction in DIRECTIONS for arm in gate["arms"]),
        "holdout": {row["partition"] for row in rows} == set(SPLITS),
        "state16": {row["state_index"] for row in rows} == {16},
        "routes": {row["route"] for row in rows} == set(gate["routes"]),
        "arms": {row["arm"] for row in rows} == set(gate["arms"]),
        "route_shapes": all((row["source_length"] == row["target_length"]) == (row["route_order"] == "same_order") for row in rows),
        "write_errors": max(max(row["write_max_abs_diff"], row["complement_max_abs_diff"]) for row in rows) <= protocol["camera"]["write_max_abs_diff"],
        "finite": all(math.isfinite(row[key]) for row in rows for key in ("recipient_margin", "swap_margin", "oriented_gain", "recipient_output_max_abs_diff")),
        "cells": all(recomputed[route][direction][split]["executor"] == summary["cell_results"][route][direction][split]["executor_pass"] and recomputed[route][direction][split]["semantic"] == summary["cell_results"][route][direction][split]["semantic_side_winner"] and recomputed[route][direction][split]["physical"] == summary["cell_results"][route][direction][split]["physical_phase_winner"] for route in gate["routes"] for direction in DIRECTIONS for split in SPLITS),
        "counts": counts == summary["classification_counts"] == final["classification_counts"],
        "classification": expected == summary["overall_classification"] == final["overall_classification"],
        "contract": summary["contract_sha256"] == reveal["contract_sha256"] == protocol["contract_sha256"],
        "one_shot": reveal["one_shot"] and reveal["holdout_count"] == 48,
        "execution": summary["all_execution_checks_passed"],
        "authorization": final["authorization"] == "run_phase1444_c073_campaign_closure",
    }
    result = {
        "phase": 1443,
        "campaign": "C073",
        "recomputed_cells": recomputed,
        "recomputed_counts": counts,
        "recomputed_classification": expected,
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
