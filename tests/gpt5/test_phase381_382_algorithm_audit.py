from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P381 = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
P382 = ROOT / "tests/gpt5/result/phase382_transition_event_audit"
ATLAS = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Phase381382AlgorithmAuditTests(unittest.TestCase):
    def test_phase381_fresh_denominator_and_behavior_expansion_are_sealed(self) -> None:
        bank = read_json(P381 / "phase381_case_bank_summary.json")
        expansion = read_json(P381 / "target_expansion/phase381x_protocol.json")
        behavior = read_json(P381 / "phase381_behavior_analysis_final_summary.json")
        self.assertEqual(bank["case_count"], 864)
        self.assertEqual(bank["phase380_prompt_overlap_count"], 0)
        self.assertFalse(bank["internal_trace_started"])
        self.assertEqual(expansion["case_count"], 288)
        self.assertFalse(expansion["failed_original_groups_replaced"])
        self.assertFalse(expansion["threshold_lowered"])
        self.assertEqual(behavior["denominator"]["selected_parallel_group_count"], 24)
        self.assertEqual(behavior["denominator"]["selected_trace_case_count"], 288)
        self.assertTrue(all(row["passed"] for row in behavior["gates"]))
        self.assertFalse(behavior["threshold_retuned"])
        self.assertFalse(behavior["internal_trace_started_before_final_gate"])

    def test_phase381_trace_replay_gate_excludes_mismatch_groups(self) -> None:
        freeze = read_json(P381 / "phase381_joint_scan_freeze.json")
        self.assertEqual(freeze["denominator"]["trace_case_count"], 288)
        self.assertEqual(freeze["denominator"]["replay_match_case_count"], 285)
        self.assertEqual(freeze["denominator"]["replay_qualified_group_count"], 22)
        self.assertEqual(
            freeze["denominator"]["replay_groups_by_mechanism"],
            {"relation_binding": 7, "entity_recency": 7, "target_vs_wrong": 8},
        )
        self.assertEqual(len(freeze["replay_mismatch_groups"]), 2)
        self.assertFalse(freeze["quality"]["mismatch_groups_used_for_causal_claims"])
        self.assertFalse(freeze["quality"]["threshold_retuned"])

    def test_phase381_all_cuda_runs_match_the_frozen_grid(self) -> None:
        for model in ("qwen3", "glm4", "deepseek7b"):
            complete = read_json(P381 / "causal/models" / model / "complete.json")
            self.assertEqual(complete["condition_row_count"], 23040)
            self.assertEqual(complete["transfer_task_count"], 144)
            self.assertTrue(complete["all_patch_hooks_reached"])
            self.assertFalse(complete["top_k_used"])
            self.assertFalse(complete["single_neuron_scan"])
            self.assertTrue(complete["valid"])

    def test_phase381_joint_state_is_a_strong_negative(self) -> None:
        summary = read_json(P381 / "phase381_joint_state_summary.json")
        result = summary["results"]
        self.assertEqual(summary["denominator"]["condition_row_count"], 69120)
        self.assertEqual(summary["denominator"]["joint_direction_gate_row_count"], 8640)
        self.assertEqual(result["joint_direction_gate_pass_count"], 563)
        self.assertEqual(result["maximum_all_four_direction_group_pass_count_in_any_model_cell"], 1)
        self.assertEqual(result["model_cell_pass_count"], 0)
        self.assertEqual(result["heterogeneous_level2_cell_count"], 0)
        self.assertEqual(result["heterogeneous_upstream_joint_state_cell_count"], 0)
        self.assertFalse(result["joint_distributed_upstream_state_established"])
        self.assertEqual(result["single_neuron_causal_count"], 0)
        self.assertFalse(result["language_encoding_mechanism_closed"])
        self.assertLess(summary["descriptive"]["joint_synergy_gain_median"], 0.0)

    def test_phase382_protocol_is_frozen_and_noncausal(self) -> None:
        protocol = read_json(P382 / "phase382_transition_protocol.json")
        self.assertFalse(protocol["fresh_cuda_execution"])
        self.assertEqual(protocol["profile_grid"]["profile_width"], 15)
        self.assertFalse(protocol["profile_grid"]["top_k_used"])
        self.assertFalse(protocol["parameter_free_identifiability_gate"]["threshold_fitting_allowed"])
        self.assertFalse(protocol["claim_boundary"]["offline_identifiability_is_causal_path"])
        self.assertFalse(protocol["claim_boundary"]["positive_result_authorizes_immediate_neuron_scan"])
        self.assertTrue(
            all(
                len(splits["offline_discovery"]) == 4
                and len(splits["offline_validation"]) >= 3
                for splits in protocol["frozen_group_splits"].values()
            )
        )

    def test_phase382_transition_update_fails_all_improvement_gates(self) -> None:
        summary = read_json(P382 / "phase382_transition_summary.json")
        result = summary["results"]
        transition = result["metrics"]["transition_update"]
        static = result["metrics"]["static_layer_input"]
        self.assertEqual(summary["denominator"]["transition_event_row_count"], 20592)
        self.assertEqual(summary["denominator"]["profile_count"], 108)
        self.assertEqual(transition["own_profile_win_count"], 12)
        self.assertEqual(static["own_profile_win_count"], 13)
        self.assertLess(
            transition["within_mechanism_cosine_median"],
            static["within_mechanism_cosine_median"],
        )
        self.assertLess(
            transition["heterogeneous_crossmodel_cosine_median"],
            static["heterogeneous_crossmodel_cosine_median"],
        )
        self.assertTrue(not any(result["parameter_free_gate_vector"].values()))
        self.assertFalse(result["transition_update_more_identifiable_than_static_state"])
        self.assertFalse(result["causal_intervention_authorized"])

    def test_latest_atlas_mirrors_preserve_negative_evidence(self) -> None:
        names = (
            "manifest.json",
            "progress.json",
            "client_index.json",
            "phase381_joint_state_stage_summary.json",
            "phase382_transition_stage_summary.json",
            "phase382_transition_summary.json",
            "phase382_evidence_nodes.jsonl",
            "phase382_evidence_edges.jsonl",
        )
        for name in names:
            self.assertEqual((ATLAS / name).read_bytes(), (CLIENT / name).read_bytes(), name)
        manifest = read_json(ATLAS / "manifest.json")
        progress = read_json(ATLAS / "progress.json")
        self.assertEqual(manifest["last_phase"], "Phase385-StageMerge")
        self.assertEqual(manifest["phase381"]["model_cell_pass_count"], 0)
        self.assertEqual(manifest["phase382"]["parameter_free_gate_pass_count"], 0)
        self.assertFalse(progress["single_global_progress_percentage_valid"])
        self.assertEqual(
            progress["transition_operator_stage"]["causal_intervention_authorized"]["numerator"],
            0,
        )
        for root in (NEURON, NEURON_CLIENT):
            neuron_manifest = read_json(root / "manifest.json")
            self.assertEqual(neuron_manifest["phase"], 385)
            self.assertEqual(
                neuron_manifest["phase381_audit"]["new_neuron_path_nodes_promoted"], 0
            )
            self.assertEqual(
                neuron_manifest["phase382_audit"]["new_neuron_path_nodes_promoted"], 0
            )
            self.assertEqual(
                neuron_manifest["phase383_385_audit"]["new_neuron_path_nodes_promoted"],
                0,
            )
            self.assertFalse(
                neuron_manifest["evidence_boundary"]["upstream_language_path_available"]
            )


if __name__ == "__main__":
    unittest.main()
