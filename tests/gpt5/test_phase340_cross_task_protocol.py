from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase340_cross_task_protocol/fresh_cross_task_protocol_repair"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase340CrossTaskProtocolTests(unittest.TestCase):
    def test_fresh_baseline_denominator(self) -> None:
        cases = read_jsonl(RESULT / "phase340_registered_cases.jsonl")
        self.assertEqual(len(cases), 972)
        self.assertEqual(len({row["case_id"] for row in cases}), 972)
        self.assertEqual(len({row["mechanism_id"] for row in cases}), 9)
        self.assertEqual({row["template_id"] for row in cases}, {"template_b", "template_c"})
        self.assertTrue(all(row["baseline_only"] for row in cases))
        self.assertFalse(any(row["internal_intervention_allowed"] for row in cases))

    def test_counts(self) -> None:
        cases = read_jsonl(RESULT / "phase340_registered_cases.jsonl")
        self.assertEqual(
            {model: sum(row["model"] == model for row in cases)
             for model in ("qwen3", "glm4", "deepseek7b")},
            {"qwen3": 324, "glm4": 324, "deepseek7b": 324},
        )
        self.assertEqual(
            {split: sum(row["split"] == split for row in cases)
             for split in ("discovery", "calibration", "heldout", "private_heldout")},
            {"discovery": 486, "calibration": 216, "heldout": 162, "private_heldout": 108},
        )

    def test_execution_boundary(self) -> None:
        path = RESULT / "phase340_global_summary.json"
        if not path.exists():
            self.skipTest("Phase340 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["phrase_row_count"], 972)
        self.assertEqual(summary["denominator"]["rollout_row_count"], 972)
        diagnostic = summary["denominator"]["glm4_batch_invariance_diagnostic"]
        self.assertIsNotNone(diagnostic)
        self.assertFalse(diagnostic["batch_invariant"])
        self.assertEqual(summary["results"]["internal_intervention_executed_count"], 0)
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)
        self.assertEqual(len(read_jsonl(RESULT / "phase340_protocol_nodes.jsonl")), 27)


if __name__ == "__main__":
    unittest.main()
