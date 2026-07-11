from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase348_adjusted_block_screen/adjusted_natural_candidate_block_screen"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase348AdjustedBlockScreenTests(unittest.TestCase):
    def test_denominator_and_seal(self) -> None:
        cases = read_jsonl(RESULT / "phase348_registered_cases.jsonl")
        self.assertEqual(len(cases), 1296)
        self.assertEqual(len({row["case_id"] for row in cases}), 1296)
        self.assertEqual(len(read_jsonl(RESULT / "phase348_frozen_blocks.jsonl")), 6)
        self.assertEqual(sum(row["split"] in {"discovery", "calibration"} for row in cases), 918)
        self.assertEqual(sum(row["split"] in {"heldout", "private_heldout"} for row in cases), 378)

    def test_screen_boundary(self) -> None:
        path = RESULT / "phase348_global_summary.json"
        if not path.exists():
            self.skipTest("Phase348 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["screen_condition_row_count"], 4590)
        self.assertEqual(summary["denominator"]["actual_model_batch_size"], 1)
        self.assertFalse(summary["results"]["heldout_causal_outcome_revealed"])
        self.assertFalse(summary["results"]["mcue_entry_gate_open"])
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)


if __name__ == "__main__":
    unittest.main()
