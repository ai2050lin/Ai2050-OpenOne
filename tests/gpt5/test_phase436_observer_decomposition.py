#!/usr/bin/env python3
"""Focused invariants for Phase436 observer decomposition."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase436_observer_decomposition_analysis as analysis  # noqa: E402
import phase436_observer_decomposition_protocol as protocol  # noqa: E402


def test_fresh_calibration_is_large_balanced_and_disjoint() -> None:
    rows = protocol.build_calibration_groups()
    audit = protocol.denominator_audit(rows)
    assert audit["valid"]
    assert audit["calibration_group_count"] == 576
    assert audit["calibration_conditions_per_model"] == 2304
    assert audit["three_model_calibration_conditions"] == 6912
    assert audit["fresh_calibration_vocabulary_disjoint_from_phase435"]


def test_selection_unit_is_model_contract_not_global_model() -> None:
    frozen = protocol.freeze()
    assert frozen["interface_selection_unit"] == "model_x_contract"
    assert frozen["semantic_content_excludes_exact_format_and_stop"]


def test_each_contract_has_balanced_factors() -> None:
    rows = protocol.build_calibration_groups()
    for contract in protocol.CONTRACTS:
        selected = [row for row in rows if row["contract_variants"] == [contract]]
        assert len(selected) == 192
        assert {sum(row["baseline_record_order"] == value for row in selected) for value in ("ab", "ba")} == {96}
        assert {sum(row["baseline_mapping"] == value for row in selected) for value in ("direct", "swapped")} == {96}
        assert {sum(row["baseline_query_role"] == value for row in selected) for value in ("a", "b")} == {96}
        assert {sum(row["relation_family"] == value for row in selected) for value in protocol.p435.RELATION_FAMILIES} == {48}


def test_semantic_content_does_not_require_format_or_stop() -> None:
    row = {
        "teacher_sequence_correct": True,
        "actual_choice": "source_1",
        "semantic_target_source": "source_1",
        "natural_target_first": True,
        "natural_opposite_first": False,
        "natural_revision": False,
        "natural_interface_valid": False,
        "natural_stop_good": False,
    }
    assert analysis.semantic_content_good(row)
    row["actual_choice"] = "source_2"
    assert not analysis.semantic_content_good(row)


def test_metrics_keep_three_axes_separate() -> None:
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
    value = analysis.metrics([row])
    assert value["semantic_content"]["estimate"] == 1.0
    assert value["exact_format"]["estimate"] == 0.0
    assert value["stop"]["estimate"] == 0.0


def test_phase435_behavior_and_sealed_rows_remain_gate_locked() -> None:
    frozen = protocol.freeze()
    assert frozen["denominator_audit"]["phase435_behavior_discovery_and_holdout_were_not_used_for_observer_selection"]
    assert frozen["phase435_sealed_commitment"]["read_requires_open_gate"]
    assert frozen["source_transport_is_not_inferred_from_geometry"]


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"Phase436 tests passed: {len(tests)}")


if __name__ == "__main__":
    main()
