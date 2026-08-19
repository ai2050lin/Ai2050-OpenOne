#!/usr/bin/env python3
"""Independent audit for Phase1438 C072 exhaustive permutation spectrum."""
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

CONTRACT = TESTS / "result/phase1435_c072_permutation_spectrum_contract"
BEHAVIOR = TESTS / "result/phase1436_c072_behavior"
OUT = TESTS / "result/phase1438_c072_permutation_spectrum"
SPLITS = ("confirmation", "lockbox")
DIRECTIONS = ("true_to_false", "false_to_true")


def med(values):
    return float(statistics.median(values))


def compose(p, q):
    return tuple(p[q[i]] for i in range(4))


def inverse(p):
    out = [0] * 4
    for i, value in enumerate(p):
        out[value] = i
    return tuple(out)


def is_subgroup(values):
    return tuple(range(4)) in values and all(inverse(value) in values for value in values) and all(compose(a, b) in values for a in values for b in values)


def metrics(rows, pid):
    values = [row for row in rows if row["permutation_id"] == pid]
    return {"sign": sum(row["desired_sign"] for row in values) / len(values), "gain": med([row["oriented_gain"] for row in values])}


def recompute_cell(rows, transfer, direction, registry, families, gate):
    selected = [row for row in rows if row["surface_transfer"] == transfer and row["direction"] == direction]
    self_pass = all(max(row["recipient_output_max_abs_diff"] for row in selected if row["partition"] == split and row["arm"] == "self") <= gate["self_max_abs_diff"] for split in SPLITS)
    same_pass = all(sum(row["desired_sign"] for row in selected if row["partition"] == split and row["arm"] == "same_surface_identity") / 24 >= gate["same_surface_desired_sign_fraction_min"] for split in SPLITS)
    wrong_pass = all(sum(row["wrong_expected_sign"] for row in selected if row["partition"] == split and row["arm"] == "wrong_cross_surface_identity") / 24 >= gate["wrong_expected_sign_fraction_min"] for split in SPLITS)
    same_breadth = sum(all(sum(row["desired_sign"] for row in selected if row["partition"] == split and row["family"] == family and row["arm"] == "same_surface_identity") / 4 >= gate["family_sign_fraction_min"] for split in SPLITS) for family in families)
    wrong_breadth = sum(all(sum(row["wrong_expected_sign"] for row in selected if row["partition"] == split and row["family"] == family and row["arm"] == "wrong_cross_surface_identity") / 4 >= gate["family_sign_fraction_min"] for split in SPLITS) for family in families)
    executor = self_pass and same_pass and wrong_pass and same_breadth >= gate["minimum_family_breadth"] and wrong_breadth >= gate["minimum_family_breadth"]
    qualified = []
    for permutation in registry:
        pid = permutation["permutation_id"]
        split_pass = all((value := metrics([row for row in selected if row["partition"] == split and row["record_type"] == "permutation"], pid))["sign"] >= gate["permutation_desired_sign_fraction_min"] and value["gain"] >= gate["permutation_oriented_gain_median_min"] for split in SPLITS)
        breadth = sum(all(metrics([row for row in selected if row["partition"] == split and row["family"] == family and row["record_type"] == "permutation"], pid)["sign"] >= gate["family_sign_fraction_min"] for split in SPLITS) for family in families)
        if split_pass and breadth >= gate["minimum_family_breadth"]:
            qualified.append(pid)
    return executor, set(qualified)


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    registry = core.rows(CONTRACT / "material/permutation_registry.jsonl")
    behavior = core.load(BEHAVIOR / "analysis/final.json")
    rows = core.rows(OUT / "raw/permutation_spectrum.jsonl")
    summary = core.load(OUT / "analysis/permutation_spectrum_summary.json")
    final = core.load(OUT / "analysis/final.json")
    gate = protocol["mechanism"]
    tuple_by_id = {row["permutation_id"]: tuple(row["source_indices_by_target"]) for row in registry}
    identity = protocol["permutations"]["identity_id"]
    all_ids = set(tuple_by_id)
    recomputed = {}
    executors, sets = [], []
    for transfer in gate["surface_transfers"]:
        recomputed[transfer] = {}
        for direction in DIRECTIONS:
            executor, qualified = recompute_cell(rows, transfer, direction, registry, behavior["qualified_families"], gate)
            recomputed[transfer][direction] = {"executor": executor, "qualified": sorted(qualified)}
            executors.append(executor); sets.append(qualified)
    identity_only = all(executors) and all(values == {identity} for values in sets)
    all_permutations = all(executors) and all(values == all_ids for values in sets)
    symmetry = all(summary["cell_results"][transfer][direction]["symmetric_gain_range_ratio"][split] <= gate["symmetric_gain_range_ratio_max"] for transfer in gate["surface_transfers"] for direction in DIRECTIONS for split in SPLITS)
    identity_gap = all(summary["cell_results"][transfer][direction]["identity_vs_best_nonidentity"][split]["sign"] >= gate["identity_vs_best_nonidentity_sign_gap_min"] and summary["cell_results"][transfer][direction]["identity_vs_best_nonidentity"][split]["paired_gain"] >= gate["identity_vs_best_nonidentity_gain_gap_median_min"] for transfer in gate["surface_transfers"] for direction in DIRECTIONS for split in SPLITS)
    stable = len({frozenset(value) for value in sets}) == 1
    common = sets[0] if stable else set()
    subgroup = all(executors) and stable and 1 < len(common) < 24 and is_subgroup({tuple_by_id[pid] for pid in common})
    expected = "role_order_selective" if identity_only and identity_gap else "permutation_symmetric_multiset" if all_permutations and symmetry else "subgroup_structured" if subgroup else "heterogeneous_or_executor_failed"
    counts = Counter((row["partition"], row["surface_transfer"], row["direction"], row["record_type"]) for row in rows)
    checks = {
        "count": len(rows) == 5184,
        "types": sum(row["record_type"] == "control" for row in rows) == 576 and sum(row["record_type"] == "permutation" for row in rows) == 4608,
        "balance": all(sum(row["partition"] == split and row["surface_transfer"] == transfer and row["direction"] == direction and row["permutation_id"] == permutation["permutation_id"] for row in rows) == 24 for split in SPLITS for transfer in gate["surface_transfers"] for direction in DIRECTIONS for permutation in registry),
        "holdout": {row["partition"] for row in rows} == set(SPLITS),
        "state16": {row["state_index"] for row in rows} == {16},
        "finite": all(math.isfinite(row[key]) for row in rows for key in ("recipient_margin", "swap_margin", "oriented_gain", "recipient_output_max_abs_diff")),
        "cells": all(recomputed[transfer][direction]["executor"] == summary["cell_results"][transfer][direction]["executor_pass"] and recomputed[transfer][direction]["qualified"] == summary["cell_results"][transfer][direction]["qualified_permutations"] for transfer in gate["surface_transfers"] for direction in DIRECTIONS),
        "classification": expected == summary["overall_classification"] == final["overall_classification"],
        "contract": summary["contract_sha256"] == protocol["contract_sha256"],
        "execution": summary["all_execution_checks_passed"],
        "authorization": final["authorization"] == "run_phase1439_c072_campaign_closure",
    }
    result = {"phase": 1438, "campaign": "C072", "recomputed_cells": recomputed, "recomputed_classification": expected, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
