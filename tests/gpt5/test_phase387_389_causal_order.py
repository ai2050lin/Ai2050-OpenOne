from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P387 = ROOT / "tests/gpt5/result/phase387_computational_order_audit"
P388 = ROOT / "tests/gpt5/result/phase388_source_kv_transport"
P389 = ROOT / "tests/gpt5/result/phase389_head_source_decomposition"
ATLAS = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase387To389CausalOrderTests(unittest.TestCase):
    def test_phase387_downgrades_every_relation_without_deleting_evidence(self) -> None:
        summary = read_json(P387 / "phase387_summary.json")
        self.assertEqual(summary["results"]["predictive_trajectory_count"], 10)
        self.assertEqual(summary["results"]["direct_computational_edge_admissible_count"], 0)
        self.assertTrue(summary["authorization"]["display_predictive_trajectories"])
        self.assertFalse(summary["authorization"]["display_direct_physical_edge"])

    def test_phase388_uses_single_sample_three_model_denominator(self) -> None:
        protocol = read_json(P388 / "phase388_protocol.json")
        freeze = read_json(P388 / "phase388_intervention_freeze.json")
        self.assertEqual(protocol["runtime_contract"]["execution_batch_size"], 1)
        self.assertEqual(freeze["denominator"]["three_model_paired_behavior_and_position_qualified_group_count"], 22)
        self.assertEqual(freeze["denominator"]["instrument_group_count"], 2)
        self.assertEqual(freeze["denominator"]["causal_test_group_count"], 16)
        self.assertEqual(freeze["denominator"]["causal_test_direction_count"], 96)
        for model in ("qwen3", "glm4", "deepseek7b"):
            behavior = read_json(P388 / "behavior" / model / "complete.json")
            self.assertEqual(behavior["case_count"], 48)
            self.assertEqual(behavior["batch_size"], 1)

    def test_phase388_instrument_is_exact_and_causal_result_is_negative(self) -> None:
        instrument = read_json(P388 / "phase388_instrument_audit_summary.json")
        causal = read_json(P388 / "phase388_causal_summary.json")
        self.assertTrue(instrument["results"]["valid"])
        self.assertEqual(instrument["results"]["patch_failure_count"], 0)
        self.assertEqual(causal["denominator"]["direction_count"], 96)
        self.assertEqual(causal["denominator"]["scenario_count"], 672)
        self.assertEqual(causal["results"]["strict_donor_target_switch_count"], 0)
        self.assertEqual(causal["results"]["models_passing_all_three_outcomes"], 0)
        self.assertFalse(causal["results"]["causal_source_kv_transport_path_established"])

    def test_phase389_preserves_broad_activity_but_rejects_crossmodel_specificity(self) -> None:
        summary = read_json(P389 / "phase389_summary.json")
        self.assertEqual(summary["denominator"]["all_heads_evaluated"], 92)
        self.assertEqual(summary["results"]["replicated_head_relation_count"], 56)
        self.assertEqual(summary["results"]["replicated_source_anchor_specificity_by_model"], {
            "qwen3": 0,
            "glm4": 2,
            "deepseek7b": 0,
        })
        self.assertFalse(summary["authorization"]["run_new_head_specific_intervention"])
        self.assertFalse(summary["authorization"]["run_single_neuron_scan"])
        self.assertFalse(summary["denominator"]["physical_holdout_reused"])

    def test_phase389_atlas_and_client_match_strict_latest_boundary(self) -> None:
        for root in (ATLAS, CLIENT):
            manifest = read_json(root / "manifest.json")
            progress = read_json(root / "progress.json")
            stage = read_json(root / "phase389_causal_order_stage_summary.json")
            self.assertEqual(manifest["last_phase"], "Phase389-StageMerge")
            self.assertEqual(progress["last_phase"], "Phase389-StageMerge")
            self.assertEqual(stage["results"]["direct_computational_edge_count"], 0)
            self.assertEqual(stage["results"]["strict_donor_answer_switch_count"], 0)
            self.assertFalse(stage["authorization"]["show_specific_neuron_path"])
            self.assertEqual(progress["causal_order_kv_stage"]["complete_language_paths"], {
                "numerator": 0,
                "denominator": 72,
            })

    def test_public_evidence_edges_never_claim_causality(self) -> None:
        rows = read_jsonl(ATLAS / "phase389_evidence_edges.jsonl")
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["causal_path"] is False for row in rows))


if __name__ == "__main__":
    unittest.main()
