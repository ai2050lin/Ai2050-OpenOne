#!/usr/bin/env python3
"""Non-upgrading exact A* sublattice and coverage diagnostic for Phase1163.

This script is intentionally post-hoc.  It searches no new schedule: every
subset of the already frozen four-site A* set was predeclared as calibration
or leave-one-out data.  The diagnostic separates algebraic high-order terms
from minimal causal sufficiency and compares a simple max-lower-pair coverage
rule with the frozen additive pairwise estimator.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1163_high_order_exception_replication as phase  # noqa: E402


SCRIPT = Path(__file__).resolve()
OUT_PATH = phase.OUT_ROOT / "analysis/posthoc_sublattice_coverage_diagnostic.json"
SUFFICIENCY_THRESHOLD = 0.80


def all_subsets(base: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [
        tuple(row)
        for cardinality in range(len(base) + 1)
        for row in itertools.combinations(base, cardinality)
    ]


def mobius(values: dict[tuple[int, ...], np.ndarray], target: tuple[int, ...]) -> np.ndarray:
    result = np.zeros_like(next(iter(values.values())), dtype=np.float64)
    target_set = set(target)
    for subset, value in values.items():
        if set(subset).issubset(target_set):
            result += ((-1.0) ** (len(target) - len(subset))) * value
    return result


def main() -> None:
    if OUT_PATH.exists():
        raise RuntimeError("refusing to overwrite post-hoc diagnostic")
    protocol = phase.read_json(phase.OUT_ROOT / "protocol/preregistration.json")
    score = phase.read_json(phase.OUT_ROOT / "analysis/score.json")
    with np.load(phase.OUT_ROOT / "runs/calibration/calibration_responses.npz") as pack:
        calibration = np.asarray(pack["response"], dtype=np.float64)
    with np.load(phase.OUT_ROOT / "predictions/diagnostic_predictions.npz") as pack:
        pairwise_prediction = np.asarray(pack["prediction"], dtype=np.float64)
    with np.load(phase.OUT_ROOT / "runs/diagnostics/diagnostic_responses.npz") as pack:
        matched = np.asarray(pack["matched"], dtype=np.float64)

    calibration_lookup = {
        subset: index for index, subset in enumerate(phase.calibration_subsets())
    }
    diagnostic_lookup = {
        tuple(row["subset"]): int(row["index"])
        for row in protocol["diagnostic_registry"]
    }
    lattice = all_subsets(phase.A_STAR)
    values: dict[tuple[int, ...], np.ndarray] = {}
    sources = {}
    for subset in lattice:
        if len(subset) <= 2:
            values[subset] = calibration[:, :, calibration_lookup[subset]]
            sources[phase.subset_id(subset)] = "calibration"
        else:
            values[subset] = matched[:, :, diagnostic_lookup[subset]]
            sources[phase.subset_id(subset)] = "predeclared_diagnostic"
    coefficients = {subset: mobius(values, subset) for subset in lattice}
    reconstructed = {}
    reconstruction_error = 0.0
    for target in lattice:
        target_set = set(target)
        total = sum(
            value for subset, value in coefficients.items() if set(subset).issubset(target_set)
        )
        reconstructed[target] = total
        reconstruction_error = max(
            reconstruction_error, float(np.max(np.abs(total - values[target])))
        )

    order_contributions = {
        order: sum(value for subset, value in coefficients.items() if len(subset) == order)
        for order in range(5)
    }
    exact_low_order = sum(order_contributions[order] for order in range(3))
    exact_high_order = order_contributions[3] + order_contributions[4]
    a_index = phase.registry_index("frozen_a_star")
    ridge_a = pairwise_prediction[:, :, a_index]
    a_actual = values[phase.A_STAR]

    minimum_cardinality = np.full(a_actual.shape, 5, dtype=np.int64)
    sufficient_subsets = []
    for subset in lattice:
        sufficient = values[subset] >= SUFFICIENCY_THRESHOLD
        minimum_cardinality = np.minimum(
            minimum_cardinality,
            np.where(sufficient, len(subset), 5),
        )
        sufficient_subsets.append(
            {
                "subset_id": phase.subset_id(subset),
                "cardinality": len(subset),
                "actual_median": float(np.median(values[subset])),
                "unit_sufficient_count": int(np.sum(sufficient)),
            }
        )

    registry_subsets = phase.registry_subsets()
    max_pair_prediction = np.zeros_like(matched, dtype=np.float64)
    for schedule_index, schedule in enumerate(registry_subsets):
        schedule_set = set(schedule)
        lower_indices = [
            index
            for subset, index in calibration_lookup.items()
            if len(subset) <= 2 and set(subset).issubset(schedule_set)
        ]
        max_pair_prediction[:, :, schedule_index] = np.max(
            calibration[:, :, lower_indices], axis=2
        )
    pairwise_abs = np.abs(pairwise_prediction - matched)
    max_pair_abs = np.abs(max_pair_prediction - matched)
    pairwise_unit_mae = np.mean(pairwise_abs, axis=2).reshape(-1)
    max_pair_unit_mae = np.mean(max_pair_abs, axis=2).reshape(-1)
    random_indices = [
        row["index"]
        for row in protocol["diagnostic_registry"]
        if "matched_cardinality_control" in row["categories"]
    ]
    random_pairwise_mae = np.mean(pairwise_abs[:, :, random_indices], axis=2).reshape(-1)
    random_max_pair_mae = np.mean(max_pair_abs[:, :, random_indices], axis=2).reshape(-1)

    checks = {
        "phase1163_primary_audit_passed": phase.read_json(
            phase.OUT_ROOT / "audit/independent_audit.json"
        )["all_checks_passed"],
        "frozen_target_replicated": score["results"]["operational_exception_replication_confirmed"],
        "complete_a_star_lattice": len(values) == 16,
        "calibration_supplies_orders_zero_to_two": all(
            sources[phase.subset_id(subset)] == "calibration"
            for subset in lattice if len(subset) <= 2
        ),
        "diagnostic_supplies_orders_three_to_four": all(
            sources[phase.subset_id(subset)] == "predeclared_diagnostic"
            for subset in lattice if len(subset) >= 3
        ),
        "exact_reconstruction": reconstruction_error <= 1e-10,
        "all_finite": all(np.isfinite(value).all() for value in values.values()),
        "no_new_schedule_search": True,
        "evidence_upgrade_forbidden": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"diagnostic checks failed: {checks}")

    report: dict[str, Any] = {
        "phase": phase.PHASE,
        "created_at_utc": phase.now(),
        "status": "posthoc_non_upgrading_sublattice_diagnostic",
        "evidence_upgrade_forbidden": True,
        "source_protocol_digest": protocol["protocol_digest"],
        "source_score_digest": score["score_digest"],
        "script_sha256": phase.sha256_file(SCRIPT),
        "a_star_subset_id": phase.subset_id(phase.A_STAR),
        "lattice_subset_count": len(lattice),
        "unit_count": int(a_actual.size),
        "sufficiency_threshold": SUFFICIENCY_THRESHOLD,
        "minimum_sufficient_cardinality_counts": {
            str(cardinality): int(np.sum(minimum_cardinality == cardinality))
            for cardinality in sorted(set(minimum_cardinality.reshape(-1).tolist()))
        },
        "sufficient_subsets": sufficient_subsets,
        "mobius_order_contribution_medians_at_a_star": {
            str(order): float(np.median(order_contributions[order]))
            for order in range(5)
        },
        "mobius_order_contribution_abs_medians_at_a_star": {
            str(order): float(np.median(np.abs(order_contributions[order])))
            for order in range(5)
        },
        "exact_low_order_prediction_median": float(np.median(exact_low_order)),
        "exact_high_order_correction_median": float(np.median(exact_high_order)),
        "ridge_pairwise_prediction_median": float(np.median(ridge_a)),
        "ridge_vs_exact_low_order_abs_difference_median": float(
            np.median(np.abs(ridge_a - exact_low_order))
        ),
        "a_star_actual_median": float(np.median(a_actual)),
        "exact_reconstruction_max_error": reconstruction_error,
        "coverage_baseline": {
            "definition": "maximum observed response among all null/single/pair subsets contained in the target schedule",
            "all_registry_pairwise_unit_mae_median": float(np.median(pairwise_unit_mae)),
            "all_registry_max_pair_unit_mae_median": float(np.median(max_pair_unit_mae)),
            "all_registry_max_pair_advantage_median": float(
                np.median(pairwise_unit_mae - max_pair_unit_mae)
            ),
            "a_star_max_pair_prediction_median": float(
                np.median(max_pair_prediction[:, :, a_index])
            ),
            "a_star_max_pair_abs_error_median": float(
                np.median(max_pair_abs[:, :, a_index])
            ),
            "matched_cardinality_pairwise_unit_mae_median": float(
                np.median(random_pairwise_mae)
            ),
            "matched_cardinality_max_pair_unit_mae_median": float(
                np.median(random_max_pair_mae)
            ),
        },
        "interpretation": (
            "A large algebraic order-3/4 correction can coexist with a two-site sufficient intervention. "
            "The correction therefore measures failure of additive inclusion-exclusion extrapolation under "
            "saturation/redundancy; it does not establish four-site causal necessity."
        ),
        "non_implications": [
            "The max-lower-pair rule was examined after observing the Phase1163 mechanism split and is not confirmed.",
            "Exact Mobius coefficients on the patch-response sublattice are not natural neural interaction edges.",
            "Sufficiency under full-state replacement does not establish necessity in the unpatched forward pass.",
        ],
        "checks": checks,
    }
    report["diagnostic_digest"] = phase.digest(report)
    phase.write_json(OUT_PATH, report)
    print(phase.canonical({
        "minimum_sufficient_cardinality_counts": report["minimum_sufficient_cardinality_counts"],
        "mobius_order_contribution_medians_at_a_star": report["mobius_order_contribution_medians_at_a_star"],
        "coverage_baseline": report["coverage_baseline"],
        "diagnostic_digest": report["diagnostic_digest"],
    }))


if __name__ == "__main__":
    main()
