#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase327_natural_retrieval_case_bank as bank
import phase327_natural_retrieval_path as path


class Phase327PathTest(unittest.TestCase):
    def test_variant_case_preserves_registered_target(self) -> None:
        case = bank.build_cases()[0]
        variant = path.variant_case(case, "same_semantic_wrong_target")
        self.assertEqual(variant["target"], case["target"])
        self.assertNotEqual(variant["natural_target"], case["target"])
        self.assertIn(variant["source_fragments"][0], variant["prompt"])

    def test_role_conditions_are_separated(self) -> None:
        specs = [
            {"position_role": role, "component_type": "x", "component_layer": 0,
             "component_start": index, "component_end": index + 1}
            for index, role in enumerate(path.ROLES)
        ]
        self.assertEqual(len(path.role_specs(specs, ("source",))), 1)
        self.assertEqual(len(path.role_specs(specs, ("source", "query"))), 2)

    def test_normalized_generation_contract(self) -> None:
        self.assertEqual(path.normalized_words(" Blue. "), ["blue"])
        self.assertEqual(path.normalized_words("instrument\nextra"), ["instrument", "extra"])


if __name__ == "__main__":
    unittest.main()
