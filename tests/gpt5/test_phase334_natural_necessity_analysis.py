from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas"


class Phase334AnalysisTests(unittest.TestCase):
    def test_execution_denominator(self) -> None:
        quality = json.loads((RESULT / "phase334_execution_quality.json").read_text())
        self.assertTrue(quality["valid"])
        self.assertEqual(quality["registered_case_count"], 1944)
        self.assertEqual(quality["heldout_condition_row_count"], 5346)
        self.assertEqual(quality["frozen_necessity_plan_count"], 54)

    def test_claim_boundary(self) -> None:
        summary = json.loads((RESULT / "phase334_global_summary.json").read_text())
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)
        self.assertFalse(summary["language_encoding_mechanism_closed"])

    def test_common_valid_denominator_reported(self) -> None:
        rows = [
            json.loads(line)
            for line in (RESULT / "phase334_local_necessity_summary.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 54)
        self.assertTrue(all("common_valid_case_count" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
