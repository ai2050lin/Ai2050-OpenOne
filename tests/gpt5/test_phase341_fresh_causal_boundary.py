from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase341_fresh_causal_boundary/qualified_six_task_causal_boundary"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase341FreshCausalBoundaryTests(unittest.TestCase):
    def test_frozen_denominator(self) -> None:
        cases = read_jsonl(RESULT / "phase341_registered_cases.jsonl")
        self.assertEqual(len(cases), 648)
        self.assertEqual(len({row["case_id"] for row in cases}), 648)
        self.assertEqual(len({row["mechanism_id"] for row in cases}), 6)
        self.assertFalse(any(row["block_reselection_allowed"] for row in cases))
        self.assertFalse(any(row["layer_shrink_allowed"] for row in cases))
        self.assertFalse(any(row["single_unit_intervention_allowed"] for row in cases))

    def test_protocol(self) -> None:
        protocol = json.loads((RESULT / "phase341_registered_protocol.json").read_text())
        self.assertEqual(protocol["registered_case_count"], 648)
        self.assertEqual(protocol["rollout_batch_size"], 1)
        self.assertTrue(any("not reselected" in value for value in protocol["claim_boundaries"]))

    def test_execution_boundary(self) -> None:
        path = RESULT / "phase341_global_summary.json"
        if not path.exists():
            self.skipTest("Phase341 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["phrase_row_count"], 3888)
        self.assertEqual(summary["denominator"]["rollout_row_count"], 720)
        self.assertEqual(summary["denominator"]["rollout_batch_size"], 1)
        self.assertEqual(summary["results"]["layer_shrink_executed_count"], 0)
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)
        nodes = read_jsonl(RESULT / "phase341_task_boundary_nodes.jsonl")
        self.assertEqual(len(nodes), 18)
        self.assertFalse(any(row["single_unit_causal"] for row in nodes))


if __name__ == "__main__":
    unittest.main()
