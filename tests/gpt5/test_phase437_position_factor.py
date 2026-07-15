#!/usr/bin/env python3
"""Deterministic protocol and gate tests for Phase437."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase437_position_factor_analysis as analysis  # noqa: E402
import phase437_position_factor_collect as collect  # noqa: E402
import phase437_position_factor_protocol as protocol  # noqa: E402


def test_denominator_and_geometry_pairs_are_frozen() -> None:
    frozen = protocol.freeze()
    audit = frozen["denominator_audit"]
    assert audit["valid"]
    assert audit["groups_by_split"] == {
        "observer_calibration": 1152,
        "behavior_discovery": 576,
        "behavior_holdout": 1152,
        "physical_calibration": 1152,
        "sealed_physical_holdout": 1152,
    }
    for payload in audit["geometry_pair_audit"].values():
        assert payload["all_pairs_have_two_groups"]
        assert payload["different_relation_matched_factor_pairs"]
    assert all(
        payload["valid"]
        for payload in audit["observer_pairwise_balance"].values()
    )


def test_phase436_threshold_audit_is_explicit_and_not_rewritten() -> None:
    audit = protocol.freeze()["phase436_threshold_audit"]
    assert audit["interface_other_ucb"] == 0.10
    assert audit["behavior_other_ucb"] == 0.05
    assert audit["phase437_unified_other_ucb"] == 0.05
    assert audit["historical_summary_consistent_with_code"]
    assert audit["phase436_results_not_rewritten"]


def test_behavior_groups_are_full_factor_balanced() -> None:
    groups = protocol.build_groups()
    for split in protocol.BEHAVIOR_SPLITS:
        for contract in protocol.CONTRACTS:
            selected = [row for row in groups[split] if row["contract"] == contract]
            for factor in ("boundary", "connector", "record_length", "label_order"):
                counts = {value: sum(row[factor] == value for row in selected) for value in {row[factor] for row in selected}}
                assert len(set(counts.values())) == 1


def test_semantic_format_and_stop_are_separate() -> None:
    row = {
        "teacher_sequence_correct": True,
        "actual_choice": "source_1",
        "semantic_target_source": "source_1",
        "natural_target_first": True,
        "natural_opposite_first": False,
        "natural_revision": False,
        "natural_interface_valid": False,
        "natural_exact_target_contract": False,
        "natural_stop_good": False,
    }
    assert analysis.semantic_good(row)
    value = analysis.metrics([row])
    assert value["semantic_content"]["estimate"] == 1.0
    assert value["exact_format"]["estimate"] == 0.0
    assert value["stop"]["estimate"] == 0.0


def synthetic_rows(*, fail_first_near: bool = False) -> list[dict[str, Any]]:
    rows = []
    variants = {
        "first_natural_near": ("first", "natural", "near", 20),
        "second_natural_near": ("second", "natural", "near", 4),
        "first_natural_far": ("first", "natural", "far", 40),
        "second_natural_far": ("second", "natural", "far", 24),
        "second_matched_near": ("second", "matched", "near", 20),
        "second_matched_far": ("second", "matched", "far", 40),
    }
    for index in range(400):
        for variant, (position, recency, gap, distance) in variants.items():
            good = not (fail_first_near and variant == "first_natural_near" and index % 5 == 0)
            rows.append(
                {
                    "semantic_group_id": f"synthetic_{index:04d}",
                    "variant": variant,
                    "target_position": position,
                    "recency_control": recency,
                    "post_gap": gap,
                    "boundary": protocol.BOUNDARIES[index % 4],
                    "connector": protocol.CONNECTORS[index % 3],
                    "record_length": protocol.RECORD_LENGTHS[index % 2],
                    "label_order": protocol.LABEL_ORDERS[(index // 2) % 2],
                    "target_to_question_token_distance": distance + index % 3,
                    "semantic_content_good": good,
                    "teacher_sequence_correct": True,
                    "actual_choice": "source_1" if good else "source_2",
                    "semantic_target_source": "source_1",
                    "natural_target_first": good,
                    "natural_opposite_first": not good,
                    "natural_revision": False,
                    "natural_interface_valid": True,
                    "natural_exact_target_contract": True,
                    "natural_stop_good": True,
                }
            )
    return rows


def test_factorized_behavior_gate_can_pass_clean_synthetic_data() -> None:
    thresholds = protocol.freeze()["behavior_gate"]
    audit = analysis.analyze_behavior_split(synthetic_rows(), thresholds)
    assert audit["pass"]
    assert audit["gate_components"]["actual_token_distance_registration"]
    assert audit["maximum_effects"] == {
        "ordinal": 0.0,
        "matched": 0.0,
        "post_gap": 0.0,
        "outer_factor": 0.0,
    }


def test_factorized_behavior_gate_rejects_position_dependent_data() -> None:
    thresholds = protocol.freeze()["behavior_gate"]
    audit = analysis.analyze_behavior_split(
        synthetic_rows(fail_first_near=True), thresholds
    )
    assert not audit["pass"]
    assert not audit["gate_components"]["natural_position_gap"]


def test_distance_design_does_not_claim_impossible_full_orthogonality() -> None:
    audit = analysis.distance_audit(synthetic_rows())
    assert audit["first_record_is_structurally_farther_than_second"]
    assert audit["distance_claim_is_conditional_not_fully_orthogonal"]
    assert all(value["median_absolute_error_tokens"] == 0.0 for value in audit["matched"].values())


def test_all_registered_controls_change_the_prompt_surface() -> None:
    group = next(
        row for row in protocol.build_groups()[protocol.PHYSICAL_SPLIT]
        if row["contract"] == "natural_qa"
    )
    values = collect.engine.mapping_values(group, group["mapping"])
    candidate, _, _, _ = collect.controlled_surface(
        group, "natural_qa", values, "second_natural_near", "candidate"
    )
    candidate_signature = (
        candidate["record_block"], candidate["distance_spacer"],
        candidate["post_gap_text"], candidate["question_line"],
    )
    for control_type in protocol.CONTROL_TYPES:
        surface, _, _, design = collect.controlled_surface(
            group, "natural_qa", values, "second_natural_near",
            f"control_{control_type}",
        )
        signature = (
            surface["record_block"], surface["distance_spacer"],
            surface["post_gap_text"], surface["question_line"],
        )
        assert design["prompt_changed"]
        assert signature != candidate_signature


def test_sealed_rows_remain_locked_before_open_physical_pass() -> None:
    frozen = protocol.freeze()
    assert frozen["sealed_commitment"]["read_requires_all_open_physical_gates"]
    assert frozen["sealed_commitment"]["causal_and_single_neuron_forbidden"]
    assert frozen["physical_stage"]["causal_and_single_neuron_forbidden"]


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"Phase437 tests passed: {len(tests)}")


if __name__ == "__main__":
    main()
