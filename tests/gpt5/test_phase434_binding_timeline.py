#!/usr/bin/env python3
"""Focused invariants for the Phase434 binding-timeline protocol."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase434_binding_timeline_analysis as analysis  # noqa: E402
import phase434_binding_timeline_collect as collect  # noqa: E402
import phase434_binding_timeline_protocol as protocol  # noqa: E402


def test_denominator_is_large_balanced_and_split_disjoint() -> None:
    rows = protocol.build_groups()
    audit = protocol.denominator_audit(rows)
    assert audit["valid"]
    assert audit["conditions_by_split_per_model"]["behavior_discovery"] == 4800
    assert audit["conditions_by_split_per_model"]["behavior_holdout"] == 9600
    assert audit["three_model_behavior_open_conditions"] == 44352
    assert audit["physical_open_conditions_per_eligible_model"] == 9600
    assert audit["vocabulary_disjoint_across_splits"]


def test_factor_marginals_are_balanced_in_formal_splits() -> None:
    rows = protocol.build_groups()
    for split in ("behavior_holdout", "physical_calibration", "sealed_physical_holdout"):
        candidates = [row for row in rows[split] if row["candidate"]]
        assert {sum(row["role_alias_index"] == value for row in candidates) for value in range(4)} == {48}
        assert {sum(row["cue_alias_index"] == value for row in candidates) for value in range(4)} == {48}
        assert sum(row["baseline_record_order"] == "ab" for row in candidates) == 96
        assert sum(row["baseline_mapping"] == "direct" for row in candidates) == 96


def test_multi_token_event_surface_contract() -> None:
    rows = protocol.build_groups()
    for values in rows.values():
        for row in values:
            assert len(row["source_1"]) == len(row["source_2"])
            assert row["source_1"].startswith(row["shared_stem"])
            assert row["source_2"].startswith(row["shared_stem"])
            assert row["source_1"] != row["source_2"]


def test_exactly_one_active_slot_for_main_conditions() -> None:
    aliases = {"a": "cue-ax", "b": "cue-by"}
    for timing in protocol.TIMINGS:
        slots = collect.selector_slots(timing, "a", aliases)
        assert set(slots) == set(protocol.TIMINGS)
        assert sum(value != protocol.NEUTRAL_CUE for value in slots.values()) == 1
        assert slots[timing] == "cue-ax"


def test_candidate_crosses_order_and_mapping_control_does_not_fake_independence() -> None:
    assert protocol.conditions_per_group(True, "behavior_holdout") == 40
    assert protocol.conditions_per_group(False, "behavior_holdout") == 10
    assert protocol.conditions_per_group(True, protocol.STRESS_SPLIT) == 4


def test_cosine_geometry_is_label_blind_and_well_formed() -> None:
    assert analysis.cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert analysis.cosine_distance([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert round(analysis.cosine_distance([1.0, 0.0], [-1.0, 0.0]), 6) == 2.0


def test_complete_event_gate_rejects_branch_only_success() -> None:
    row = {
        "teacher_sequence_correct": True,
        "natural_target_first": True,
        "natural_opposite_first": False,
        "natural_interface_valid": True,
        "natural_exact_target_contract": True,
        "natural_revision": False,
        "natural_boundary": True,
        "natural_stop": True,
        "natural_censoring": False,
        "natural_common_prefix_exact": True,
        "natural_reaches_branch_boundary": True,
        "natural_branch_correct": True,
        "natural_complete_event_correct": True,
    }
    assert analysis.condition_good(row)
    row["natural_complete_event_correct"] = False
    assert not analysis.condition_good(row)


def test_source_transport_is_explicitly_not_inferred_from_geometry() -> None:
    frozen = protocol.freeze()
    assert frozen["source_transport_is_not_inferred_from_state_geometry"]
    assert "G4_source_specific_transport" in frozen["gate_order"]

