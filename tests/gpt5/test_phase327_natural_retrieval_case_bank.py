#!/usr/bin/env python3
from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase327_natural_retrieval_case_bank as bank


class Phase327CaseBankTest(unittest.TestCase):
    def test_registered_denominator_and_independence(self) -> None:
        cases = bank.build_cases()
        validation = bank.validate_cases(cases)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["independent_object_count"], 54)
        self.assertEqual(validation["prompt_case_count"], 108)
        self.assertEqual(validation["natural_variant_count"], 540)
        self.assertEqual(validation["phase326_same_mechanism_subject_overlap_count"], 0)

    def test_controls_have_registered_roles(self) -> None:
        for case in bank.build_cases():
            self.assertEqual(set(case["variants"]), set(bank.VARIANTS))
            self.assertEqual(
                case["variants"]["same_target_object"]["natural_target"],
                case["target"],
            )
            for name in (
                "same_semantic_wrong_target",
                "token_length_wrong_target",
                "unrelated_wrong_target",
            ):
                self.assertNotEqual(case["variants"][name]["natural_target"], case["target"])

    def test_targets_are_absent_from_prompts(self) -> None:
        self.assertTrue(all(case["target_absent_from_prompt"] for case in bank.build_cases()))

    def test_source_precedes_query_for_causal_propagation(self) -> None:
        for case in bank.build_cases():
            self.assertLess(
                case["prompt"].index(case["source_fragments"][0]),
                case["prompt"].index(case["query_fragment"]),
            )


if __name__ == "__main__":
    unittest.main()
