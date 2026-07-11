from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase338BlockCausalScreenTests(unittest.TestCase):
    def test_frozen_denominator(self) -> None:
        cases = read_jsonl(RESULT / "phase338_registered_cases.jsonl")
        blocks = read_jsonl(RESULT / "phase338_registered_blocks.jsonl")
        self.assertEqual(len(cases), 216)
        self.assertEqual(len({row["case_id"] for row in cases}), 216)
        self.assertEqual(len(blocks), 27)
        self.assertEqual({row["interface"] for row in cases}, {"answer_aligned_chat"})
        self.assertFalse(any(row["single_unit_intervention_allowed"] for row in cases))

    def test_split_and_claim_boundaries(self) -> None:
        cases = read_jsonl(RESULT / "phase338_registered_cases.jsonl")
        expected = {
            "discovery": 108, "calibration": 54,
            "heldout": 36, "private_heldout": 18,
        }
        self.assertEqual(
            {split: sum(row["split"] == split for row in cases) for split in expected}, expected
        )
        protocol = json.loads((RESULT / "phase338_registered_protocol.json").read_text())
        self.assertTrue(any("real mechanism" in value for value in protocol["claim_boundaries"]))
        self.assertTrue(any("No single-neuron" in value for value in protocol["claim_boundaries"]))

    def test_execution_boundary(self) -> None:
        path = RESULT / "phase338_global_summary.json"
        if not path.exists():
            self.skipTest("Phase338 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_stage_completions_valid"])
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)
        self.assertEqual(summary["progress_vector"]["coarse_block_deep_audit_attempted"], "1/72")
        self.assertFalse(summary["results"]["cross_model_block_gate_pass"])
        self.assertFalse(summary["results"]["minimal_causal_set_entry_gate_open"])
        nodes = read_jsonl(RESULT / "phase338_physical_block_nodes.jsonl")
        self.assertEqual(len(nodes), 81)
        self.assertEqual(sum(row["local_heldout_private_gate_pass"] for row in nodes), 1)
        self.assertFalse(any(row["single_unit_causal"] for row in nodes))


if __name__ == "__main__":
    unittest.main()
