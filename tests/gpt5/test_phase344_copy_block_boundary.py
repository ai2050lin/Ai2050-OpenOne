from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase344_copy_block_boundary/copy_block_heldout_boundary"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase344CopyBlockBoundaryTests(unittest.TestCase):
    def test_frozen_denominator(self) -> None:
        cases = read_jsonl(RESULT / "phase344_registered_cases.jsonl")
        self.assertEqual(len(cases), 585)
        self.assertEqual(len({row["case_id"] for row in cases}), 585)
        self.assertEqual(len({row["mechanism_id"] for row in cases}), 13)
        self.assertEqual({row["split"] for row in cases}, {"heldout", "private_heldout"})
        self.assertFalse(any(row["block_reselection_allowed"] for row in cases))
        self.assertFalse(any(row["layer_shrink_allowed"] for row in cases))
        self.assertFalse(any(row["single_unit_intervention_allowed"] for row in cases))

    def test_protocol(self) -> None:
        protocol = json.loads((RESULT / "phase344_registered_protocol.json").read_text())
        self.assertEqual(protocol["execution_mode"], "b1_left_cache0")
        self.assertEqual(len(protocol["lexical_generalization_required_tasks"]), 4)
        self.assertTrue(any("No natural-state" in value for value in protocol["claim_boundaries"]))

    def test_execution_boundary(self) -> None:
        path = RESULT / "phase344_global_summary.json"
        if not path.exists():
            self.skipTest("Phase344 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["phrase_row_count"], 3510)
        self.assertEqual(summary["denominator"]["rollout_row_count"], 2340)
        self.assertEqual(summary["results"]["layer_shrink_executed_count"], 0)
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)
        self.assertEqual(len(read_jsonl(RESULT / "phase344_copy_boundary_nodes.jsonl")), 39)


if __name__ == "__main__":
    unittest.main()
