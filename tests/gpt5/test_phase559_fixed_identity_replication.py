#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase559_fixed_identity_replication_protocol as protocol  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase559FixedIdentityReplicationTests(unittest.TestCase):
    def test_static_denominator_and_seal(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_case_count"], 55296)
        self.assertEqual(audit["open_case_count"], 46080)
        self.assertEqual(audit["sealed_case_count"], 9216)
        self.assertEqual(audit["behavior_case_count_per_model"], 8192)
        self.assertEqual(commitment["sealed_case_count"], 9216)
        self.assertFalse(commitment["sealed_split_read_for_analysis"])

    def test_phase558_contract_was_not_relaxed(self) -> None:
        frozen = read_json(protocol.PROTOCOL_PATH)
        self.assertEqual(frozen["phase558_contract_changes"], {
            "surface_templates_changed": False,
            "fact_order_conditions_changed": False,
            "classifier_changed": False,
            "behavior_thresholds_changed": False,
            "only_new_disjoint_objects_and_larger_denominator": True,
        })
        self.assertEqual(frozen["behavior_gate"]["world_all_32_rate_min_per_behavior_split"], 0.80)
        self.assertEqual(frozen["behavior_gate"]["minimum_cell_wilson_95_lcb"], 0.90)
        self.assertEqual(frozen["behavior_gate"]["unrecoverable_wilson_95_ucb_max"], 0.05)

    def test_phase558_open_objects_do_not_overlap(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        self.assertEqual(audit["phase558_open_object_overlap_count"], 0)

    def test_counterfactual_pair_invariants(self) -> None:
        rows = [
            row for row in read_jsonl(protocol.OPEN_CASES_PATH)
            if row["model"] == "qwen3"
            and row["pair_id"] == "phase559_behavior_discovery_000_query0_surface1_order1"
        ]
        self.assertEqual(len(rows), 2)
        left, right = sorted(rows, key=lambda row: row["binding"])
        for key in (
            "object_a", "object_b", "color_a", "color_b", "query_object", "surface_id",
            "fact_order", "fact_token_multiset_key", "prompt_token_multiset_key",
        ):
            self.assertEqual(left[key], right[key])
        self.assertEqual(left["target"], right["nontarget_color"])
        self.assertEqual(right["target"], left["nontarget_color"])

    def test_behavior_script_selects_only_behavior_splits(self) -> None:
        source = (ROOT / "tests/gpt5/phase559_fixed_identity_replication_behavior.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("BEHAVIOR_SPLITS", source)
        self.assertIn("EXPECTED_MODEL_ROWS = 8192", source)
        self.assertNotIn("phase559_sealed_cases", source)

    def test_fine_scan_stays_closed(self) -> None:
        frozen = read_json(protocol.PROTOCOL_PATH)
        policy = frozen["evidence_policy"]
        self.assertTrue(policy["internal_collection_requires_phase559_behavior_pass"])
        self.assertTrue(policy["path_behavior_runs_only_after_model_authorization"])
        self.assertTrue(policy["compute_edge_requires_source_delete_restore_and_exclusion"])
        self.assertFalse(policy["single_neuron_scan_before_compute_edge"])
        self.assertFalse(policy["sealed_split_read"])

    def test_behavior_authorization_requires_both_splits(self) -> None:
        summary_path = protocol.OUT_DIR / "phase559_behavior_summary.json"
        if not summary_path.exists():
            self.skipTest("Phase559 behavior has not run")
        summary = read_json(summary_path)
        for report in summary["model_reports"]:
            expected = all(
                report["split_reports"][split]["behavior_gate_pass"]
                for split in protocol.BEHAVIOR_SPLITS
            )
            self.assertEqual(report["authorized_for_path_behavior"], expected)
        self.assertFalse(summary["sealed_split_read"])

    def test_path_summary_obeys_frozen_gate(self) -> None:
        summary_path = protocol.OUT_DIR / "phase559_path_behavior_summary.json"
        if not summary_path.exists():
            self.skipTest("Phase559 path behavior has not been analyzed")
        summary = read_json(summary_path)
        contract = read_json(protocol.OUT_DIR / "phase559_path_behavior_frozen_contract.json")
        gate = contract["path_gate"]
        expected_all = True
        for split in contract["selected_splits"]:
            report = summary["split_reports"][split]
            expected = bool(
                report["all_32_correct_world_rate"] >= gate["world_all_32_rate_min_per_split"]
                and report["minimum_cell_wilson_95_lcb"] >= gate["minimum_cell_wilson_95_lcb"]
                and report["unrecoverable_wilson_95_ucb"] <= gate["unrecoverable_wilson_95_ucb_max"]
            )
            self.assertEqual(report["path_gate_pass"], expected)
            expected_all = expected_all and expected
        self.assertEqual(summary["authorized_for_internal_collection"], expected_all)
        self.assertFalse(summary["sealed_split_read"])

    def test_event_candidates_are_discovery_only_and_coarse(self) -> None:
        registry_path = protocol.OUT_DIR / "phase559_binding_candidate_registry.json"
        if not registry_path.exists():
            self.skipTest("Phase559 binding candidates have not been frozen")
        registry = read_json(registry_path)
        self.assertEqual(registry["candidate_count"], 6)
        self.assertFalse(registry["head_channel_parameter_neuron_scan_authorized"])
        self.assertFalse(registry["sealed_split_read"])
        self.assertEqual({row["boundary"] for row in registry["candidates"]}, {"source", "query"})
        self.assertTrue(all(row["selection_split"] == "path_discovery" for row in registry["candidates"]))
        self.assertTrue(all(not row["confirmation_used_for_selection"] for row in registry["candidates"]))

    def test_causal_screen_is_confirmation_only_and_not_a_compute_edge(self) -> None:
        contract_path = protocol.OUT_DIR / "phase559_causal_screen_frozen_contract.json"
        if not contract_path.exists():
            self.skipTest("Phase559 causal screen has not been frozen")
        contract = read_json(contract_path)
        self.assertEqual(contract["split"], "path_confirmation")
        self.assertEqual(contract["selected_anchor_count"], 24)
        self.assertEqual(contract["conditions"], [
            "same_case_restore",
            "correct_paired_donor_replace",
            "channel_roll_donor_replace",
        ])
        self.assertTrue(contract["evidence_policy"]["screen_pass_is_sufficiency_candidate_only"])
        self.assertFalse(contract["evidence_policy"]["head_channel_parameter_neuron_scan_authorized"])
        self.assertFalse(contract["evidence_policy"]["sealed_split_read"])

    def test_screen_analysis_never_promotes_without_all_frozen_gates(self) -> None:
        path = protocol.OUT_DIR / "phase559_causal_screen_analysis.json"
        if not path.exists():
            self.skipTest("Phase559 causal screen has not been analyzed")
        summary = read_json(path)
        contract = read_json(protocol.OUT_DIR / "phase559_causal_screen_frozen_contract.json")
        gate = contract["screen_gate"]
        for row in summary["candidate_reports"]:
            expected = bool(
                row["same_case_max_absolute_switch_effect"] <= gate["same_case_max_absolute_switch_effect"]
                and row["correct_donor_win_rate"] >= gate["correct_donor_win_rate_min"]
                and row["minimum_factorial_cell_donor_win_rate"] >= gate["minimum_factorial_cell_donor_win_rate"]
                and row["correct_donor_mean_switch_effect"] >= gate["correct_donor_mean_switch_effect_min"]
                and row["correct_minus_channel_roll_mean_switch_effect"] >= gate[
                    "correct_minus_channel_roll_mean_switch_effect_min"
                ]
            )
            self.assertEqual(row["screen_gate_pass"], expected)
            self.assertFalse(row["compute_edge"])
        self.assertFalse(summary["sealed_split_read"])

    def test_phase560_uses_semantic_color_and_fresh_confirmation_worlds(self) -> None:
        phase560 = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
        contract_path = phase560 / "phase560_semantic_color_screen_frozen_contract.json"
        if not contract_path.exists():
            self.skipTest("Phase560 semantic color protocol has not been frozen")
        contract = read_json(contract_path)
        candidates = read_json(phase560 / "phase560_semantic_color_candidate_registry.json")
        prior = read_json(protocol.OUT_DIR / "phase559_causal_screen_frozen_contract.json")
        self.assertFalse(set(contract["selected_anchor_ids"]) & set(prior["selected_anchor_ids"]))
        self.assertEqual(contract["selected_anchor_count"], 24)
        self.assertTrue(all(row["semantic_position"] == "source_color_end" for row in candidates["candidates"]))
        self.assertTrue(all(not row["confirmation_used_for_selection"] for row in candidates["candidates"]))
        self.assertFalse(contract["evidence_policy"]["head_channel_parameter_neuron_scan_authorized"])

    def test_phase560_unseen_contract_requires_delete_and_exclusion(self) -> None:
        phase560 = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
        contract_path = phase560 / "phase560_semantic_color_unseen_frozen_contract.json"
        if not contract_path.exists():
            self.skipTest("Phase560 unseen contract has not been frozen")
        contract = read_json(contract_path)
        self.assertEqual(contract["split"], "unseen_recombination")
        self.assertEqual(contract["selected_anchor_count"], 40)
        self.assertIn("paired_contrast_neutralize", contract["conditions"])
        self.assertIn("wrong_position_donor_replace", contract["conditions"])
        self.assertIn("wrong_depth_donor_replace", contract["conditions"])
        self.assertTrue(contract["evidence_policy"]["object_color_binding_operation_not_identified"])
        self.assertFalse(contract["evidence_policy"]["head_channel_parameter_neuron_scan_authorized"])

    def test_phase560_parent_contract_uses_fresh_unseen_worlds(self) -> None:
        phase560 = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
        parent_path = phase560 / "phase560_parent_decomposition_frozen_contract.json"
        if not parent_path.exists():
            self.skipTest("Phase560 parent decomposition has not been frozen")
        parent = read_json(parent_path)
        unseen = read_json(phase560 / "phase560_semantic_color_unseen_frozen_contract.json")
        self.assertFalse(set(parent["selected_anchor_ids"]) & set(unseen["selected_anchor_ids"]))
        self.assertEqual(parent["selected_anchor_count"], 20)
        self.assertEqual(parent["conditions"], [
            "same_case_restore",
            "layer_input_donor_replace",
            "attention_output_donor_replace",
            "mlp_output_donor_replace",
            "layer_output_donor_replace",
        ])
        self.assertFalse(parent["evidence_policy"]["head_channel_parameter_neuron_scan_authorized"])

    def test_phase561_trace_and_phase562_validation_are_disjoint(self) -> None:
        phase561 = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
        trace_path = phase561 / "phase561_source_to_query_trace_frozen_contract.json"
        reader_path = phase561 / "phase562_reader_validation_frozen_contract.json"
        if not trace_path.exists() or not reader_path.exists():
            self.skipTest("Phase561/562 contracts have not been frozen")
        trace = read_json(trace_path)
        reader = read_json(reader_path)
        candidates = read_json(phase561 / "phase562_reader_candidate_registry.json")
        self.assertFalse(set(trace["selected_anchor_ids"]) & set(reader["selected_anchor_ids"]))
        self.assertTrue(all(row["component"] == "attention_output" for row in candidates["candidates"]))
        self.assertTrue(reader["evidence_policy"]["trajectory_onset_does_not_prejudge_reader_validation"])
        self.assertFalse(reader["evidence_policy"]["head_channel_parameter_neuron_scan_authorized"])

    def test_phase562_reader_analysis_applies_every_frozen_gate(self) -> None:
        phase561 = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
        analysis_path = phase561 / "phase562_reader_validation_analysis.json"
        if not analysis_path.exists():
            self.skipTest("Phase562 reader validation has not been analyzed")
        analysis = read_json(analysis_path)
        contract = read_json(phase561 / "phase562_reader_validation_frozen_contract.json")
        gate = contract["validation_gate"]
        for row in analysis["candidate_reports"]:
            expected = bool(
                row["same_case_max_absolute_switch_effect"]
                <= gate["same_case_max_absolute_switch_effect"]
                and row["correct_donor_win_rate"] >= gate["correct_donor_win_rate_min"]
                and row["minimum_factorial_cell_donor_win_rate"]
                >= gate["minimum_factorial_cell_donor_win_rate"]
                and row["correct_donor_mean_switch_effect"]
                >= gate["correct_donor_mean_switch_effect_min"]
                and row["paired_neutralize_mean_switch_effect"]
                >= gate["paired_neutralize_mean_switch_effect_min"]
                and row["correct_minus_channel_roll_mean_switch_effect"]
                >= gate["correct_minus_channel_roll_mean_switch_effect_min"]
                and row["correct_minus_wrong_position_mean_switch_effect"]
                >= gate["correct_minus_wrong_position_mean_switch_effect_min"]
            )
            self.assertEqual(row["validation_gate_pass"], expected)
            self.assertFalse(row["compute_edge_confirmed"])
        self.assertEqual(analysis["qualified_reader_edge_count"], 0)
        self.assertTrue(analysis["static_single_position_reader_route_closed"])

    def test_phase563_uses_only_untouched_unseen_worlds(self) -> None:
        phase560 = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
        phase561 = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
        contract_path = phase561 / "phase563_multiposition_reader_frozen_contract.json"
        if not contract_path.exists():
            self.skipTest("Phase563 contract has not been frozen")
        contract = read_json(contract_path)
        registry = read_json(phase561 / "phase563_multiposition_reader_candidate_registry.json")
        used = set()
        for path in (
            phase560 / "phase560_semantic_color_unseen_frozen_contract.json",
            phase560 / "phase560_parent_decomposition_frozen_contract.json",
            phase561 / "phase561_source_to_query_trace_frozen_contract.json",
        ):
            used.update(read_json(path)["selected_anchor_ids"])
        self.assertFalse(set(contract["selected_anchor_ids"]) & used)
        self.assertEqual(contract["selected_anchor_count"], 15)
        self.assertEqual(contract["recipient_case_count"], 480)
        self.assertEqual(contract["candidate_count"], 4)
        self.assertEqual(contract["expected_intervention_rows"], 11520)
        self.assertEqual(
            {(row["layer"], row["position_block"]) for row in registry["candidates"]},
            {
                (4, "query_answer_roles"),
                (4, "all_semantic_roles"),
                (10, "query_answer_roles"),
                (10, "all_semantic_roles"),
            },
        )
        self.assertTrue(registry["candidate_family_frozen_before_model_execution"])
        self.assertFalse(contract["evidence_policy"]["head_channel_parameter_neuron_scan_authorized"])
        self.assertFalse(contract["evidence_policy"]["sealed_split_read"])

    def test_phase563_analysis_never_promotes_without_all_gates(self) -> None:
        phase561 = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
        analysis_path = phase561 / "phase563_multiposition_reader_analysis.json"
        if not analysis_path.exists():
            self.skipTest("Phase563 has not completed")
        analysis = read_json(analysis_path)
        contract = read_json(phase561 / "phase563_multiposition_reader_frozen_contract.json")
        gate = contract["validation_gate"]
        for row in analysis["candidate_reports"]:
            expected = bool(
                row["same_case_max_absolute_switch_effect"]
                <= gate["same_case_max_absolute_switch_effect"]
                and row["correct_donor_win_rate"] >= gate["correct_donor_win_rate_min"]
                and row["minimum_factorial_cell_donor_win_rate"]
                >= gate["minimum_factorial_cell_donor_win_rate"]
                and row["correct_donor_mean_switch_effect"]
                >= gate["correct_donor_mean_switch_effect_min"]
                and row["paired_neutralize_mean_switch_effect"]
                >= gate["paired_neutralize_mean_switch_effect_min"]
                and row["correct_minus_channel_roll_mean_switch_effect"]
                >= gate["correct_minus_channel_roll_mean_switch_effect_min"]
                and row["correct_minus_wrong_position_mean_switch_effect"]
                >= gate["correct_minus_wrong_position_mean_switch_effect_min"]
            )
            self.assertEqual(row["validation_gate_pass"], expected)
            self.assertFalse(row["compute_edge_confirmed"])
        self.assertFalse(analysis["head_channel_parameter_neuron_scan_authorized"])
        self.assertFalse(analysis["sealed_split_read"])


if __name__ == "__main__":
    unittest.main()
