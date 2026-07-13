#!/usr/bin/env python3
"""Contract tests for Phase406 condition-response sequence tables."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase406_conditioned_sequence_analysis import (  # noqa: E402
    enrich_row,
    extract_semantic_label,
    group_audit,
)
from phase406_conditioned_sequence_protocol import (  # noqa: E402
    CANDIDATE_SETS,
    FAMILIES,
    INTERFACES,
    QUERIES,
    STATE_IDS,
    SURFACE_REPLICAS,
    condition_ids,
    condition_prompt,
    expected_answer,
    package_for,
)


class Phase406ConditionedSequenceTest(unittest.TestCase):
    def test_exactly_six_conditions_per_family(self) -> None:
        for family in FAMILIES:
            self.assertEqual(len(condition_ids(family)), 6)
            self.assertEqual(len(set(condition_ids(family))), 6)

    def test_prompts_do_not_list_candidates_or_imperative_updates(self) -> None:
        forbidden = (
            "choose ",
            "return exactly",
            "swap ",
            "copy ",
            "delete ",
            "update ",
            "apply the rule",
        )
        for family in FAMILIES:
            state_id = STATE_IDS[family][0]
            for surface in SURFACE_REPLICAS:
                package = package_for(family, 0, surface["lexical"])
                for query in QUERIES[family]:
                    for interface in INTERFACES:
                        prompt, _branch, _protocol = condition_prompt(
                            family, package, state_id, query, surface, interface
                        )
                        lowered = prompt.lower()
                        self.assertFalse(any(item in lowered for item in forbidden))

    def test_conservative_parser(self) -> None:
        self.assertEqual(
            extract_semantic_label(" red. It is explicit.", ["red", "blue"])[
                "semantic_label_private"
            ],
            "red",
        )
        self.assertEqual(
            extract_semantic_label("The answer is 'were'.", ["is", "are", "was", "were"])[
                "semantic_label_private"
            ],
            "were",
        )
        self.assertEqual(
            extract_semantic_label("Person B receives it.", ["A", "B"])[
                "semantic_label_private"
            ],
            "B",
        )
        self.assertEqual(
            extract_semantic_label(
                "Galen receives it.",
                ["A", "B"],
                {"A": ["A", "Arlo"], "B": ["B", "Galen"]},
            )["semantic_label_private"],
            "B",
        )
        self.assertIsNone(
            extract_semantic_label("The answer is unclear.", ["is", "are", "was", "were"])[
                "semantic_label_private"
            ]
        )

    def perfect_rows(self, family: str) -> list[dict]:
        rows = []
        for state_id in STATE_IDS[family]:
            for surface in SURFACE_REPLICAS:
                for query in QUERIES[family]:
                    target = expected_answer(family, state_id, query)
                    candidates = list(CANDIDATE_SETS[family][query])
                    for interface in INTERFACES:
                        raw = {
                            "state_id_private": state_id,
                            "surface_id_private": surface["surface_id"],
                            "future_query_private": query,
                            "interface_private": interface,
                            "target_semantic_label_private": target,
                            "semantic_candidate_labels_private": candidates,
                            "semantic_aliases_private": {
                                candidate: [candidate] for candidate in candidates
                            },
                            "generated_text_clean_private": f" {target}.",
                            "eos_observed": True,
                            "all_generated_step_logits_valid": True,
                            "first_step_candidate_correct": True,
                            "first_step_global_top_is_target": True,
                        }
                        rows.append(enrich_row(raw))
        return rows

    def test_perfect_group_and_leave_one_folds_pass(self) -> None:
        for family in FAMILIES:
            audit = group_audit(self.perfect_rows(family), family)
            self.assertTrue(audit["group_pass"])
            self.assertTrue(audit["all_leave_one_interface_folds_pass"])

    def test_two_surface_failures_close_group(self) -> None:
        family = "knowledge_binding"
        rows = self.perfect_rows(family)
        state = STATE_IDS[family][0]
        query = QUERIES[family][0]
        interface = INTERFACES[0]
        failed_surfaces = {
            SURFACE_REPLICAS[0]["surface_id"],
            SURFACE_REPLICAS[1]["surface_id"],
        }
        for row in rows:
            if (
                row["state_id_private"] == state
                and row["future_query_private"] == query
                and row["interface_private"] == interface
                and row["surface_id_private"] in failed_surfaces
            ):
                row["semantic_label_private"] = None
                row["short_sequence_semantic_correct"] = False
        self.assertFalse(group_audit(rows, family)["group_pass"])


if __name__ == "__main__":
    unittest.main()
