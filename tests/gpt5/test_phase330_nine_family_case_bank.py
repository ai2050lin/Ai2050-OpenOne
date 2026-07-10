import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase330_nine_family_case_bank as bank


class Phase330CaseBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = bank.build_cases()
        cls.validation = bank.validate_cases(cls.rows)

    def test_total_denominator_is_exact(self) -> None:
        self.assertTrue(self.validation["valid"], self.validation)
        self.assertEqual(len(self.rows), 5184)
        self.assertEqual(self.validation["prompt_model_plan_count"], 15552)

    def test_each_family_and_mechanism_has_equal_weight(self) -> None:
        self.assertEqual(set(self.validation["family_prompt_counts"].values()), {576})
        self.assertEqual(self.validation["mechanism_prompt_count_values"], [72])

    def test_item_and_template_splits_are_frozen(self) -> None:
        self.assertEqual(
            self.validation["split_item_count_values"],
            {"discovery": [12], "calibration": [6], "heldout": [6]},
        )
        self.assertEqual(set(self.validation["template_counts"].values()), {1728})
        self.assertTrue(all(not row["selection_updates_allowed"] for row in self.rows))

    def test_source_precedes_query(self) -> None:
        self.assertEqual(self.validation["source_query_order_error_count"], 0)


if __name__ == "__main__":
    unittest.main()
