#!/usr/bin/env python3
"""Independent audit for Phase1433 C071 cross-surface mechanism."""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1433, "C071"
CONTRACT = TESTS / "result/phase1430_c071_cross_surface_role_contract"
BEHAVIOR = TESTS / "result/phase1431_c071_cross_surface_behavior"
CAMERA = TESTS / "result/phase1432_c071_role_map_camera"
OUT = TESTS / "result/phase1433_c071_cross_surface_mechanism"
SPLITS = ("confirmation", "lockbox")
DIRECTIONS = ("true_to_false", "false_to_true")
ARMS = (
    "self", "same_surface_quartet", "cross_surface_role_mapped",
    "cross_surface_role_permuted", "wrong_cross_surface_role_mapped",
)


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def metrics(rows: list[dict]) -> dict:
    arms = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    ids = {arm: {row["set_id"] for row in values} for arm, values in arms.items()}
    if not all(arms.values()) or len({frozenset(value) for value in ids.values()}) != 1:
        raise RuntimeError("arm imbalance")
    lookup = {arm: {row["set_id"]: row for row in values} for arm, values in arms.items()}
    set_ids = sorted(next(iter(ids.values())))
    sign = {arm: sum(row["desired_sign"] for row in values) / len(values) for arm, values in arms.items()}
    return {
        "count": len(set_ids),
        "self": max(row["recipient_output_max_abs_diff"] for row in arms["self"]),
        "same_sign": sign["same_surface_quartet"],
        "mapped_sign": sign["cross_surface_role_mapped"],
        "permuted_sign": sign["cross_surface_role_permuted"],
        "wrong_sign": sum(row["wrong_expected_sign"] for row in arms["wrong_cross_surface_role_mapped"]) / len(arms["wrong_cross_surface_role_mapped"]),
        "mapped_gain": med([row["oriented_gain"] for row in arms["cross_surface_role_mapped"]]),
        "sign_gap": sign["cross_surface_role_mapped"] - sign["cross_surface_role_permuted"],
        "gain_gap": med([
            lookup["cross_surface_role_mapped"][set_id]["oriented_gain"]
            - lookup["cross_surface_role_permuted"][set_id]["oriented_gain"]
            for set_id in set_ids
        ]),
    }


def classify(rows: list[dict], families: list[str], transfer: str, direction: str, gate: dict) -> str:
    split = {
        name: metrics([row for row in rows if row["surface_transfer"] == transfer and row["direction"] == direction and row["partition"] == name])
        for name in SPLITS
    }
    family = {
        value: {
            name: metrics([row for row in rows if row["surface_transfer"] == transfer and row["direction"] == direction and row["partition"] == name and row["family"] == value])
            for name in SPLITS
        }
        for value in families
    }
    self_pass = all(split[name]["self"] <= gate["self_max_abs_diff"] for name in SPLITS)
    same_pass = all(split[name]["same_sign"] >= gate["same_surface_desired_sign_fraction_min"] for name in SPLITS)
    mapped_pass = all(split[name]["mapped_sign"] >= gate["cross_surface_desired_sign_fraction_min"] and split[name]["mapped_gain"] >= gate["mapped_oriented_gain_median_min"] for name in SPLITS)
    wrong_pass = all(split[name]["wrong_sign"] >= gate["wrong_expected_sign_fraction_min"] for name in SPLITS)
    selective = all(split[name]["sign_gap"] >= gate["mapped_vs_permuted_sign_gap_min"] and split[name]["gain_gap"] >= gate["mapped_vs_permuted_gain_gap_median_min"] for name in SPLITS)
    same_breadth = sum(all(family[value][name]["same_sign"] >= gate["family_desired_sign_fraction_min"] for name in SPLITS) for value in families)
    mapped_breadth = sum(all(family[value][name]["mapped_sign"] >= gate["family_desired_sign_fraction_min"] for name in SPLITS) for value in families)
    wrong_breadth = sum(all(family[value][name]["wrong_sign"] >= gate["family_desired_sign_fraction_min"] for name in SPLITS) for value in families)
    executor = self_pass and same_pass and wrong_pass and same_breadth >= gate["minimum_family_breadth"] and wrong_breadth >= gate["minimum_family_breadth"]
    cross = mapped_pass and mapped_breadth >= gate["minimum_family_breadth"]
    if not executor:
        return "same_surface_executor_failed"
    if not cross:
        return "same_surface_only"
    if not selective:
        return "cross_surface_nonspecific"
    return "role_isomorphic_selective"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    behavior = core.load(BEHAVIOR / "analysis/final.json")
    camera = core.load(CAMERA / "audit/independent_final_audit.json")
    summary = core.load(OUT / "analysis/mechanism_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/cross_surface_mechanism.jsonl")
    gate = protocol["mechanism"]
    recomputed = {
        transfer: {
            direction: classify(rows, behavior["qualified_families"], transfer, direction, gate)
            for direction in DIRECTIONS
        }
        for transfer in gate["surface_transfers"]
    }
    classes = {recomputed[transfer][direction] for transfer in recomputed for direction in DIRECTIONS}
    overall = next(iter(classes)) if len(classes) == 1 else "transfer_or_direction_asymmetric"
    expected_cells = {
        transfer: {direction: summary["cell_results"][transfer][direction]["classification"] for direction in DIRECTIONS}
        for transfer in gate["surface_transfers"]
    }
    counts = Counter((row["partition"], row["surface_transfer"], row["direction"], row["arm"]) for row in rows)
    checks = {
        "camera": camera["all_checks_passed"],
        "contract": summary["contract_sha256"] == protocol["contract_sha256"],
        "count": len(rows) == 960,
        "balance": len(counts) == 2 * 2 * 2 * 5 and set(counts.values()) == {24},
        "holdout_only": {row["partition"] for row in rows} == set(SPLITS),
        "state16": {row["state_index"] for row in rows} == {16},
        "shapes": all(row["source_length"] != row["target_length"] for row in rows),
        "quartet": all(row["quartet_size"] == 4 for row in rows),
        "writes": max(row["mapped_role_max_abs_diff"] for row in rows) <= 1e-4,
        "complement": max(max(row["mapped_complement_max_abs_diff"], row["permuted_complement_max_abs_diff"]) for row in rows) <= 1e-4,
        "finite": all(math.isfinite(row[key]) for row in rows for key in ("recipient_margin", "swap_margin", "oriented_gain", "recipient_output_max_abs_diff")),
        "cell_recompute": recomputed == expected_cells,
        "overall_recompute": overall == summary["overall_classification"] == final["overall_classification"],
        "execution": summary["all_execution_checks_passed"],
        "authorization": final["authorization"] == "run_phase1434_c071_campaign_closure",
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "recomputed_cells": recomputed,
        "recomputed_overall": overall,
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
