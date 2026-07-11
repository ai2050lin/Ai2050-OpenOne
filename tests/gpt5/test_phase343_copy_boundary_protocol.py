from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase343_copy_boundary_protocol/copy_boundary_protocol_qualification"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase343CopyBoundaryProtocolTests(unittest.TestCase):
    def test_denominator(self) -> None:
        cases = read_jsonl(RESULT / "phase343_registered_cases.jsonl")
        self.assertEqual(len(cases), 2592)
        self.assertEqual(len({row["case_id"] for row in cases}), 2592)
        self.assertEqual(len({row["mechanism_id"] for row in cases}), 16)
        self.assertEqual({row["official_execution_mode"] for row in cases}, {"b1_left_cache0"})
        self.assertFalse(any(row["internal_intervention_allowed"] for row in cases))

    def test_counts(self) -> None:
        validation = json.loads((RESULT / "phase343_case_bank_validation.json").read_text())
        self.assertEqual(validation["model_case_count"], {"qwen3": 864, "glm4": 864, "deepseek7b": 864})
        self.assertEqual(validation["split_case_count"], {
            "discovery": 1296, "calibration": 576,
            "heldout": 432, "private_heldout": 288,
        })

    def test_execution_boundary(self) -> None:
        path = RESULT / "phase343_global_summary.json"
        if not path.exists():
            self.skipTest("Phase343 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["phrase_row_count"], 2592)
        self.assertEqual(summary["denominator"]["rollout_row_count"], 2592)
        self.assertEqual(summary["results"]["internal_intervention_executed_count"], 0)
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)
        self.assertEqual(len(read_jsonl(RESULT / "phase343_protocol_nodes.jsonl")), 48)


if __name__ == "__main__":
    unittest.main()
