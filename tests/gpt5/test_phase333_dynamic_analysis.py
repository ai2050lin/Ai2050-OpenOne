from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas"


class Phase333AnalysisTests(unittest.TestCase):
    def test_execution_denominator(self) -> None:
        quality = json.loads((RESULT / "phase333_execution_quality.json").read_text())
        self.assertTrue(quality["valid"])
        self.assertEqual(quality["registered_case_count"], 648)
        self.assertEqual(quality["registered_exchange_case_count"], 108)
        self.assertEqual(quality["condition_row_count"], 972)

    def test_claim_boundary(self) -> None:
        summary = json.loads((RESULT / "phase333_global_summary.json").read_text())
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)
        self.assertFalse(summary["language_encoding_mechanism_closed"])


if __name__ == "__main__":
    unittest.main()
