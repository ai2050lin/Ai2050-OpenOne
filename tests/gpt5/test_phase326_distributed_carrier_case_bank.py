#!/usr/bin/env python3
from __future__ import annotations

import unittest

import phase326_distributed_carrier_case_bank as p326


class Phase326CaseBankTest(unittest.TestCase):
    def test_frozen_denominator_and_splits(self) -> None:
        rows = p326.build_cases()
        validation = p326.validate_cases(rows)
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(validation["prompt_case_count"], 288)
        self.assertEqual(validation["independent_base_case_count"], 96)
        self.assertEqual(validation["split_counts"], {"discovery": 96, "calibration": 96, "heldout": 96})

    def test_implicit_knowledge_has_no_answer_leak(self) -> None:
        rows = p326.build_cases()
        knowledge = [row for row in rows if row["family_id"] == "content_knowledge"]
        self.assertEqual(len(knowledge), 144)
        for row in knowledge:
            self.assertNotIn(row["target"].lower(), row["prompt"].lower())

    def test_expanded_confirmation_is_independent_and_leak_free(self) -> None:
        rows = p326.build_confirmation_cases()
        validation = p326.validate_confirmation_cases(rows)
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(validation["prompt_case_count"], 128)
        self.assertEqual(validation["independent_base_case_count"], 64)
        self.assertEqual(validation["target_leak_count"], 0)


if __name__ == "__main__":
    unittest.main()
