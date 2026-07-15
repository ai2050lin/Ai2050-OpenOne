#!/usr/bin/env python3
"""Focused invariants for the Phase433 shared-prefix protocol."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase433_shared_prefix_analysis as analysis  # noqa: E402
import phase433_shared_prefix_collect as collect  # noqa: E402
import phase433_shared_prefix_protocol as protocol  # noqa: E402


def test_denominator_is_balanced_and_split_disjoint() -> None:
    open_rows, sealed_rows, stress_rows = protocol.build_groups()
    audit = protocol.denominator_audit(open_rows, sealed_rows, stress_rows)
    assert audit["valid"]
    assert audit["main_open_conditions_per_model"] == 4608
    assert audit["stress_open_conditions_per_model"] == 1024
    assert audit["three_model_open_conditions"] == 16896
    assert audit["sealed_conditions_qwen"] == 1536
    assert audit["vocabulary_disjoint_across_splits"]


def test_source_pairs_share_surface_prefix_and_length() -> None:
    open_rows, sealed_rows, stress_rows = protocol.build_groups()
    for row in [*open_rows, *sealed_rows, *stress_rows]:
        assert len(row["source_1"]) == len(row["source_2"])
        assert row["source_1"][:-1] == row["source_2"][:-1]
        assert row["source_1"][-1] == "A"
        assert row["source_2"][-1] == "B"


def test_common_prefix_length_stops_at_first_difference() -> None:
    assert collect.common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert collect.common_prefix_length([1, 2], [1, 2, 3]) == 2
    assert collect.common_prefix_length([1], [2]) == 0


def test_route_contract_separates_main_and_stress() -> None:
    assert set(protocol.MAIN_ROUTES).isdisjoint(protocol.STRESS_ROUTES)
    assert protocol.route_tags("source_only", "a")[2:] == ("a", "none")
    assert protocol.route_tags("query_only", "b")[2:] == ("none", "b")
    assert protocol.route_tags("conflict", "a")[2:] == ("a", "b")


def test_wilson_is_conservative() -> None:
    perfect = analysis.wilson(128, 128)
    assert perfect["estimate"] == 1.0
    assert 0.95 < perfect["lcb"] < 1.0
    empty = analysis.wilson(0, 0)
    assert empty["lcb"] == 0.0
    assert empty["ucb"] == 1.0


def test_condition_good_requires_complete_event_not_branch_only() -> None:
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


def test_window_metric_distinguishes_candidate_flip_and_control_invariance() -> None:
    rows = []
    for candidate in (True, False):
        for group in range(8):
            for role in ("a", "b"):
                actual = (
                    "source_1"
                    if (candidate and role == "a") or (not candidate and group % 2 == 0)
                    else "source_2"
                )
                margin = 1.0 if actual == "source_1" else -1.0
                rows.append(
                    {
                        "split": "behavior_holdout",
                        "layer": 26,
                        "candidate": candidate,
                        "actual_choice": actual,
                        "semantic_group_id": f"g{candidate}-{group}",
                        "route_mode": "consistent",
                        "role": role,
                        "natural_common_prefix_exact": True,
                        "natural_complete_event_correct": True,
                        "position_metrics": {
                            "prompt_terminal": {
                                "source_1_minus_source_2_branch_margin": margin
                            }
                        },
                    }
                )
    candidate = analysis.window_metrics(
        rows, "behavior_holdout", 26, "prompt_terminal", True
    )
    control = analysis.window_metrics(
        rows, "behavior_holdout", 26, "prompt_terminal", False
    )
    assert candidate["choice"]["balanced_accuracy"] == 1.0
    assert candidate["predicted_role_flip"]["estimate"] == 1.0
    assert control["predicted_role_invariance"]["estimate"] == 1.0


def test_position_roles_keep_teacher_boundary_explicit() -> None:
    assert "prompt_terminal" in collect.POSITION_ROLES
    assert "teacher_branch_boundary" in collect.POSITION_ROLES
    assert collect.POSITION_ROLES.index("prompt_terminal") != collect.POSITION_ROLES.index(
        "teacher_branch_boundary"
    )
