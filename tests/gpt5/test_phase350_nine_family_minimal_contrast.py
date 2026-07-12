from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase350_nine_family_minimal_contrast/nine_family_minimal_contrast_qualification"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase350NineFamilyMinimalContrastTests(unittest.TestCase):
    def test_denominator(self) -> None:
        rows = read_jsonl(RESULT / "phase350_registered_cases.jsonl")
        self.assertEqual(len(rows), 2592)
        self.assertEqual(len({row["case_id"] for row in rows}), 2592)
        self.assertEqual(len({row["family_id"] for row in rows}), 9)
        self.assertEqual(len({row["contrast_condition"] for row in rows}), 4)
        self.assertFalse(any(row["internal_intervention_allowed"] for row in rows))
        validation = json.loads((RESULT / "phase350_case_bank_validation.json").read_text())
        self.assertEqual(validation["within_pair_target_mismatch_count"], 0)

    def test_execution_boundary(self) -> None:
        path = RESULT / "phase350_global_summary.json"
        if not path.exists():
            self.skipTest("Phase350 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["phrase_row_count"], 2592)
        self.assertEqual(summary["denominator"]["rollout_row_count"], 2592)
        self.assertEqual(summary["denominator"]["actual_model_batch_size"], 1)
        self.assertFalse(summary["results"]["physical_heldout_trace_revealed"])
        self.assertFalse(summary["results"]["causal_sealed_trace_revealed"])
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)


if __name__ == "__main__":
    unittest.main()
