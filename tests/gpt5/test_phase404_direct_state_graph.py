#!/usr/bin/env python3
"""Contract tests for Phase404 direct endpoint states."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase404_direct_state_analysis import group_audit  # noqa: E402
from phase404_direct_state_protocol import (  # noqa: E402
    FAMILIES,
    QUERIES,
    STATE_IDS,
    SURFACE_REPLICAS,
    expected_answer,
    semantic_transition_table,
    state_truth,
)


class Phase404ContractTest(unittest.TestCase):
    def test_state_fingerprints_are_pairwise_distinct(self) -> None:
        for family in FAMILIES:
            fingerprints = {
                tuple(
                    expected_answer(family, state_id, query)
                    for query in QUERIES[family]
                )
                for state_id in STATE_IDS[family]
            }
            self.assertEqual(len(fingerprints), len(STATE_IDS[family]))

    def test_surface_axes_are_balanced(self) -> None:
        for axis in ("lexical", "syntax", "order"):
            self.assertEqual(
                [surface[axis] for surface in SURFACE_REPLICAS].count(0), 2
            )
            self.assertEqual(
                [surface[axis] for surface in SURFACE_REPLICAS].count(1), 2
            )

    def test_state_truth_is_deterministic(self) -> None:
        for family in FAMILIES:
            for state_id in STATE_IDS[family]:
                self.assertEqual(
                    state_truth(family, state_id), state_truth(family, state_id)
                )

    def test_semantic_edges_reference_registered_states(self) -> None:
        for family, edges in semantic_transition_table().items():
            states = set(STATE_IDS[family])
            for edge in edges:
                self.assertIn(edge["source"], states)
                self.assertIn(edge["target"], states)

    def perfect_rows(self, family: str) -> list[dict]:
        rows = []
        for state_id in STATE_IDS[family]:
            for surface in SURFACE_REPLICAS:
                for query in QUERIES[family]:
                    target = expected_answer(family, state_id, query)
                    rows.append(
                        {
                            "state_id_private": state_id,
                            "surface_id_private": surface["surface_id"],
                            "future_query_private": query,
                            "target_private": target,
                            "predicted_candidate_private": target,
                            "finite_candidate_correct": True,
                            "target_minus_best_distractor_logit": 1.0,
                        }
                    )
        return rows

    def test_perfect_group_passes(self) -> None:
        for family in FAMILIES:
            audit = group_audit(self.perfect_rows(family), family)
            self.assertTrue(audit["truth_predictive_group_pass"])
            self.assertTrue(audit["observed_structure_group_pass"])

    def test_one_surface_error_is_tolerated(self) -> None:
        family = "knowledge_binding"
        rows = self.perfect_rows(family)
        state = STATE_IDS[family][0]
        surface = SURFACE_REPLICAS[0]["surface_id"]
        for row in rows:
            if (
                row["state_id_private"] == state
                and row["surface_id_private"] == surface
            ):
                row["finite_candidate_correct"] = False
                row["predicted_candidate_private"] = "blue"
                row["target_minus_best_distractor_logit"] = -1.0
        self.assertTrue(group_audit(rows, family)["truth_predictive_group_pass"])

    def test_two_surface_fingerprint_errors_fail(self) -> None:
        family = "knowledge_binding"
        rows = self.perfect_rows(family)
        state = STATE_IDS[family][0]
        failed_surfaces = {
            SURFACE_REPLICAS[0]["surface_id"],
            SURFACE_REPLICAS[1]["surface_id"],
        }
        for row in rows:
            if (
                row["state_id_private"] == state
                and row["surface_id_private"] in failed_surfaces
            ):
                row["finite_candidate_correct"] = False
                row["predicted_candidate_private"] = "blue"
                row["target_minus_best_distractor_logit"] = -1.0
        self.assertFalse(group_audit(rows, family)["truth_predictive_group_pass"])

    def test_nonfinite_candidate_row_is_an_explicit_group_failure(self) -> None:
        rows = self.perfect_rows("rule_reasoning")
        failed_state = STATE_IDS["rule_reasoning"][0]
        failed_surfaces = {
            SURFACE_REPLICAS[0]["surface_id"],
            SURFACE_REPLICAS[1]["surface_id"],
        }
        failed_count = 0
        for row in rows:
            if (
                row["state_id_private"] == failed_state
                and row["surface_id_private"] in failed_surfaces
            ):
                row["predicted_candidate_private"] = None
                row["finite_candidate_correct"] = False
                row["target_minus_best_distractor_logit"] = None
                failed_count += 1

        audit = group_audit(rows, "rule_reasoning")

        self.assertEqual(
            audit["positive_target_margin_count"], len(rows) - failed_count
        )
        self.assertFalse(audit["truth_predictive_group_pass"])


if __name__ == "__main__":
    unittest.main()
