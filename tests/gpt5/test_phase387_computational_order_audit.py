from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P387 = ROOT / "tests/gpt5/result/phase387_computational_order_audit"


def read_json(name: str) -> dict:
    return json.loads((P387 / name).read_text(encoding="utf-8"))


def read_jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (P387 / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase387ComputationalOrderTests(unittest.TestCase):
    def test_all_phase386_physical_survivors_are_audited_once(self) -> None:
        summary = read_json("phase387_summary.json")
        rows = read_jsonl("phase387_candidate_order_rows.jsonl")
        self.assertEqual(summary["denominator"]["phase386_physical_survivor_count"], 10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({row["candidate_id"] for row in rows}), 10)
        self.assertTrue(all(row["physical_holdout_used"] for row in rows))

    def test_predictive_trajectories_are_not_promoted_to_direct_edges(self) -> None:
        summary = read_json("phase387_summary.json")
        rows = read_jsonl("phase387_candidate_order_rows.jsonl")
        self.assertEqual(summary["results"]["predictive_trajectory_count"], 10)
        self.assertEqual(summary["results"]["direct_computational_edge_admissible_count"], 0)
        self.assertEqual(summary["results"]["upstream_direct_physical_edge_count"], 0)
        self.assertTrue(all(not row["direct_computational_edge_admissible"] for row in rows))
        self.assertTrue(all(not row["causal_claim"] for row in rows))

    def test_contract_names_the_actual_same_layer_dependency(self) -> None:
        contract = read_json("phase387_computational_order_contract.json")
        self.assertIn("layer input", contract["cross_position_rule"].lower())
        self.assertIn("k/v", contract["cross_position_rule"].lower())
        self.assertFalse(contract["composite_score_used"])
        self.assertFalse(contract["model_run_required"])

    def test_next_stage_remains_fresh_and_causally_admissible(self) -> None:
        summary = read_json("phase387_summary.json")
        self.assertFalse(summary["authorization"]["reuse_phase386_physical_holdout"])
        self.assertFalse(summary["authorization"]["run_single_neuron_scan"])
        self.assertEqual(summary["next_stage"]["phase"], 388)
        self.assertIn("wrong_source_position", summary["next_stage"]["required_controls"])


if __name__ == "__main__":
    unittest.main()
