#!/usr/bin/env python3
"""Focused invariants for the Phase435 natural-relation protocol."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase435_natural_relation_analysis as analysis  # noqa: E402
import phase435_natural_relation_collect as collect  # noqa: E402
import phase435_natural_relation_protocol as protocol  # noqa: E402
from phase429_typed_route_collect import interface_parse  # noqa: E402


def test_denominator_is_large_balanced_and_split_disjoint() -> None:
    rows = protocol.build_groups()
    audit = protocol.denominator_audit(rows)
    assert audit["valid"]
    assert audit["conditions_by_split_per_model"]["interface_calibration"] == 1152
    assert audit["conditions_by_split_per_model"]["behavior_discovery"] == 2304
    assert audit["conditions_by_split_per_model"]["behavior_holdout"] == 4608
    assert audit["open_behavior_conditions_per_model"] == 8064
    assert audit["three_model_open_behavior_conditions"] == 24192
    assert audit["maximum_open_physical_conditions_per_model"] == 6144
    assert audit["vocabulary_disjoint_across_splits"]


def test_interface_calibration_is_contract_and_factor_balanced() -> None:
    rows = protocol.build_groups()[protocol.INTERFACE_SPLIT]
    for contract in protocol.CONTRACTS:
        selected = [row for row in rows if row["contract_variants"] == [contract]]
        assert len(selected) == 96
        assert {sum(row["baseline_record_order"] == order for row in selected) for order in protocol.RECORD_ORDERS} == {48}
        assert {sum(row["baseline_mapping"] == mapping for row in selected) for mapping in protocol.MAPPINGS} == {48}
        assert {sum(row["baseline_query_role"] == role for row in selected) for role in protocol.QUERY_ROLES} == {48}
        assert {sum(row["relation_family"] == family for row in selected) for family in protocol.RELATION_FAMILIES} == {24}


def test_natural_relation_surfaces_are_exactly_registered() -> None:
    for family_index, family in enumerate(protocol.RELATION_FAMILIES):
        group = protocol.build_group("behavior_discovery", family_index)
        assert group["relation_family"] == family
        values = collect.mapping_values(group, "direct")
        for contract in (*protocol.CONTRACTS, protocol.GENERIC_CONTROL):
            surface = collect.contract_surface(group, contract, values, "ab", "a")
            for entry in surface["record_entries"]:
                assert entry["relation_surface"] in entry["line"]
                assert entry["entity"] in entry["line"]
                assert entry["value"] in entry["line"]
            assert surface["query_relation_surface"] in surface["question_line"]
            assert surface["query_entity"] in surface["question_line"]


def test_output_interfaces_are_parseable_without_shared_state() -> None:
    target = "Alder Basin City"
    opposite = "Bronze Ridge City"
    for interface in protocol.INTERFACES:
        payload = collect.interface_payload(interface, target, opposite)
        row = {
            "interface": interface,
            "target": payload["target"],
            "opposite_target": payload["opposite"],
        }
        parsed = interface_parse(str(payload["target"]), row)
        assert parsed["natural_interface_valid"]
        assert parsed["natural_exact_target_contract"]


def test_stop_is_reported_but_not_folded_into_content() -> None:
    base = {
        "natural_content_good": True,
        "teacher_sequence_correct": True,
        "natural_first_answer_good": True,
        "natural_complete_answer_good": True,
        "natural_interface_valid": True,
        "natural_stop_good": False,
        "natural_other": False,
        "actual_choice": "source_1",
    }
    metrics = analysis.behavior_metrics([base])
    assert metrics["content"]["estimate"] == 1.0
    assert metrics["stop_separate"]["estimate"] == 0.0


def test_position_balance_gap_is_symmetric() -> None:
    rows = []
    for position, good in (("first", True), ("first", True), ("second", True), ("second", False)):
        rows.append(
            {
                "target_position": position,
                "natural_content_good": good,
                "teacher_sequence_correct": True,
                "natural_first_answer_good": good,
                "natural_complete_answer_good": good,
                "natural_interface_valid": True,
                "natural_stop_good": True,
                "natural_other": not good,
                "actual_choice": "source_1" if good else "other",
            }
        )
    metrics = analysis.position_metrics(rows)
    assert metrics["first"]["content"]["estimate"] == 1.0
    assert metrics["second"]["content"]["estimate"] == 0.5
    assert metrics["content_position_gap"] == 0.5


def test_label_blind_geometry_primitive_is_well_formed() -> None:
    assert analysis.cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert analysis.cosine_distance([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert round(analysis.cosine_distance([1.0, 0.0], [-1.0, 0.0]), 6) == 2.0


def test_source_transport_is_not_inferred_from_geometry_and_sealed_is_locked() -> None:
    frozen = protocol.freeze()
    assert frozen["source_transport_is_not_inferred_from_state_geometry"]
    assert frozen["causal_and_single_neuron_forbidden_in_phase435"]
    assert frozen["sealed_commitment"]["read_requires_open_gate"]
    assert frozen["gate_order"].index("G3_label_blind_order_geometry") < frozen["gate_order"].index("G4_semantic_source_transport")


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"Phase435 tests passed: {len(tests)}")


if __name__ == "__main__":
    main()
