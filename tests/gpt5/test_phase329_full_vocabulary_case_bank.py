import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase329_full_vocabulary_case_bank as bank


class Phase329CaseBankTests(unittest.TestCase):
    def test_registered_independent_matrix_is_valid(self) -> None:
        cases = bank.build_cases()
        validation = bank.validate_cases(cases)
        self.assertTrue(validation["valid"], validation)
        self.assertEqual(len(cases), 72)
        self.assertEqual(validation["independent_object_count"], 36)
        self.assertEqual(validation["natural_variant_count"], 288)

    def test_controls_and_role_order_are_registered(self) -> None:
        for case in bank.build_cases():
            self.assertEqual(set(case["variants"]), set(bank.VARIANTS))
            self.assertNotIn(case["target"].lower(), case["prompt"].lower())
            self.assertLess(
                case["prompt"].index(case["source_fragments"][0]),
                case["prompt"].index(case["query_fragment"]),
            )

    def test_protocol_freezes_taxonomy_and_thresholds(self) -> None:
        protocol = bank.protocol()
        self.assertEqual(protocol["top_k_competitor_count"], 50)
        self.assertEqual(tuple(protocol["blocker_taxonomy"]), bank.BLOCKER_TAXONOMY)
        self.assertFalse(protocol["selection_updates_allowed"])


if __name__ == "__main__":
    unittest.main()
