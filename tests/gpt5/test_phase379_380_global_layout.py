from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P379 = ROOT / "tests/gpt5/result/phase379_global_reuse_difference_layout"
P380 = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
ATLAS = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON_ATLAS = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase379380GlobalLayoutTests(unittest.TestCase):
    def test_phase379_rejects_old_denominator_and_raw_backbone_claim(self) -> None:
        bank = read_json(P379 / "phase379_case_bank_summary.json")
        audit = read_json(P379 / "phase379_backbone_confound_audit.json")
        self.assertEqual(bank["denominator"]["case_count"], 516)
        self.assertEqual(bank["quality"]["old_phase330_common_strict_decision_cases"], 1)
        self.assertFalse(bank["quality"]["phase330_reused_as_scientific_layout"])
        self.assertTrue(audit["results"]["common_backbone_confound_detected"])
        self.assertFalse(audit["results"]["raw_profile_replication_is_function_specific_evidence"])
        self.assertFalse(audit["results"]["causal_scan_authorized"])

    def test_phase380_behavior_is_frozen_before_exact_trace(self) -> None:
        behavior = read_json(P380 / "phase380_behavior_analysis_final_summary.json")
        self.assertEqual(behavior["denominator"]["initial_behavior_case_count"], 1152)
        self.assertEqual(behavior["denominator"]["expansion_behavior_case_count"], 864)
        self.assertEqual(behavior["denominator"]["qualified_parallel_group_count"], 65)
        self.assertEqual(behavior["denominator"]["qualified_trace_case_count"], 780)
        self.assertTrue(behavior["all_mechanism_gates_passed"])
        self.assertTrue(behavior["original_number_agreement_groups_retired"])
        self.assertFalse(behavior["failed_groups_replaced"])
        self.assertFalse(behavior["internal_trace_started_before_final_behavior_gate"])

    def test_phase380_exact_trace_and_residual_denominators_are_complete(self) -> None:
        residual = read_json(P380 / "phase380_residual_validation_summary.json")
        self.assertEqual(residual["denominator"]["registered_case_count"], 780)
        self.assertEqual(residual["denominator"]["exact_event_vector_count"], 324480)
        self.assertEqual(residual["denominator"]["all_pair_event_row_count"], 486720)
        self.assertEqual(residual["denominator"]["replay_qualified_parallel_group_count"], 57)
        self.assertEqual(residual["results"]["heterogeneous_level2_stable_object_count"], 5)
        self.assertFalse(residual["results"]["causal_reuse_established"])
        self.assertFalse(residual["claim_boundary"]["four_function_validation_completes_nine_family_atlas"])
        summaries = [
            read_json(P380 / "trace/models" / model / "complete.json")
            for model in ("qwen3", "glm4", "deepseek7b")
        ]
        self.assertEqual(sum(row["case_count"] for row in summaries), 780)
        self.assertEqual(sum(row["baseline_replay_match_count"] for row in summaries), 770)
        self.assertTrue(all(row["valid"] for row in summaries))

    def test_causal_result_is_terminal_interface_not_upstream_path(self) -> None:
        summary = read_json(P380 / "phase380_causal_layout_summary.json")
        result = summary["results"]
        self.assertEqual(summary["denominator"]["condition_row_count"], 57600)
        self.assertEqual(result["heterogeneous_level2_cell_count"], 10)
        self.assertEqual(result["heterogeneous_terminal_interface_cell_count"], 10)
        self.assertEqual(result["heterogeneous_upstream_cell_count"], 0)
        self.assertEqual(result["shared_cross_mechanism_territory_count"], 0)
        self.assertEqual(result["shared_terminal_interface_territory_count"], 2)
        self.assertEqual(result["complete_upstream_language_path_count"], 0)
        self.assertEqual(result["single_neuron_causal_count"], 0)
        self.assertFalse(result["language_encoding_mechanism_closed"])
        rows = read_jsonl(P380 / "causal/phase380_crossmodel_cell_rows.jsonl")
        level2 = [row for row in rows if row["heterogeneous_level2"]]
        self.assertEqual(len(level2), 10)
        self.assertTrue(all(row["terminal_interface_cell"] for row in level2))
        self.assertTrue(all(not row["upstream_path_claimed"] for row in level2))

    def test_all_three_cuda_causal_runs_reached_frozen_denominator(self) -> None:
        for model in ("qwen3", "glm4", "deepseek7b"):
            complete = read_json(P380 / "causal/models" / model / "complete.json")
            self.assertEqual(complete["condition_row_count"], 19200)
            self.assertTrue(complete["all_patch_hooks_reached"])
            self.assertTrue(complete["valid"])

    def test_pattern_atlas_and_client_are_identical(self) -> None:
        names = (
            "manifest.json",
            "progress.json",
            "client_index.json",
            "phase380_global_layout_stage_summary.json",
            "phase380_causal_layout_summary.json",
            "phase380_evidence_nodes.jsonl",
            "phase380_evidence_edges.jsonl",
        )
        for name in names:
            self.assertEqual(
                (ATLAS / name).read_bytes(),
                (CLIENT / name).read_bytes(),
                name,
            )
        manifest = read_json(ATLAS / "manifest.json")
        progress = read_json(ATLAS / "progress.json")
        self.assertEqual(manifest["last_phase"], "Phase380-GlobalLayoutStageMerge")
        self.assertEqual(progress["global_layout_stage"]["upstream_causal_mechanisms"]["numerator"], 0)
        self.assertFalse(progress["single_global_progress_percentage_valid"])

    def test_stage_and_neuron_atlas_preserve_claim_boundary(self) -> None:
        stage = read_json(ATLAS / "phase380_global_layout_stage_summary.json")
        self.assertTrue(stage["assessment"]["global_layout_is_higher_priority_than_single_family_closure"])
        self.assertTrue(stage["assessment"]["crossmodel_causal_effect_is_terminal_interface_only"])
        self.assertFalse(stage["assessment"]["crossmodel_causal_effect_is_upstream"])
        self.assertFalse(stage["authorization"]["show_any_upstream_language_rule"])
        self.assertFalse(stage["authorization"]["promote_any_single_neuron"])
        self.assertFalse(stage["authorization"]["claim_global_layout_complete"])
        for root in (NEURON_ATLAS, NEURON_CLIENT):
            manifest = read_json(root / "manifest.json")
            self.assertEqual(manifest["phase"], 380)
            self.assertEqual(manifest["phase380_audit"]["new_neuron_path_nodes_promoted"], 0)
            self.assertEqual(manifest["phase380_audit"]["upstream_crossmodel_cell_count"], 0)
            self.assertFalse(manifest["evidence_boundary"]["upstream_language_path_available"])
            self.assertFalse(manifest["evidence_boundary"]["single_unit_causal_closure"])
        self.assertEqual(
            (NEURON_ATLAS / "phase380_global_layout_stage_summary.json").read_bytes(),
            (NEURON_CLIENT / "phase380_global_layout_stage_summary.json").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
