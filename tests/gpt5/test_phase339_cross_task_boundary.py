from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase339_cross_task_boundary/early_source_cross_task_boundary"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase339CrossTaskBoundaryTests(unittest.TestCase):
    def test_frozen_denominator(self) -> None:
        cases = read_jsonl(RESULT / "phase339_registered_cases.jsonl")
        self.assertEqual(len(cases), 1458)
        self.assertEqual(len({row["case_id"] for row in cases}), 1458)
        self.assertEqual(len({row["mechanism_id"] for row in cases}), 9)
        self.assertEqual(
            {model: sum(row["model"] == model for row in cases)
             for model in ("qwen3", "glm4", "deepseek7b")},
            {"qwen3": 486, "glm4": 486, "deepseek7b": 486},
        )
        self.assertFalse(any(row["block_reselection_allowed"] for row in cases))
        self.assertFalse(any(row["layer_shrink_allowed"] for row in cases))
        self.assertFalse(any(row["single_unit_intervention_allowed"] for row in cases))

    def test_split_counts_and_protocol(self) -> None:
        cases = read_jsonl(RESULT / "phase339_registered_cases.jsonl")
        expected = {
            "discovery": 729, "calibration": 324,
            "heldout": 243, "private_heldout": 162,
        }
        self.assertEqual(
            {split: sum(row["split"] == split for row in cases) for split in expected},
            expected,
        )
        protocol = json.loads(
            (RESULT / "phase339_registered_protocol.json").read_text()
        )
        self.assertEqual(protocol["registered_case_count"], 1458)
        self.assertEqual(protocol["thresholds"]["task_phrase_score_valid_rate_min"], 1.0)
        self.assertTrue(any("No block is reselected" in value
                            for value in protocol["claim_boundaries"]))
        self.assertTrue(any("neuron shrinking remain closed" in value
                            for value in protocol["claim_boundaries"]))

    def test_execution_boundaries(self) -> None:
        path = RESULT / "phase339_global_summary.json"
        if not path.exists():
            self.skipTest("Phase339 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["phrase_row_count"], 8748)
        self.assertEqual(summary["denominator"]["rollout_row_count"], 1620)
        self.assertEqual(summary["results"]["layer_shrink_executed_count"], 0)
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)
        nodes = read_jsonl(RESULT / "phase339_task_boundary_nodes.jsonl")
        self.assertEqual(len(nodes), 27)
        self.assertFalse(any(row["single_unit_causal"] for row in nodes))


if __name__ == "__main__":
    unittest.main()
