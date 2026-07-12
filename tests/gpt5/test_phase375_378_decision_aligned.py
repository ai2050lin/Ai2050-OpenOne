#!/usr/bin/env python3
"""Regression tests for Phase375-378 decision-aligned evidence boundaries."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P375 = ROOT / "tests/gpt5/result/phase375_finite_exact_subgraphs"
P376 = ROOT / "tests/gpt5/result/phase376_decision_aligned_subgraphs"
P377 = ROOT / "tests/gpt5/result/phase377_decision_aligned_calibration"
P378 = ROOT / "tests/gpt5/result/phase378_physical_confirmation"
ATLAS = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Phase375To378Tests(unittest.TestCase):
    def test_phase375_protocol_keeps_diagnostics_out_of_state(self):
        protocol = read_json(P375 / "phase375_protocol.json")
        self.assertFalse(
            protocol["object_separation"][
                "formation_children_eligible_as_sufficient_state"
            ]
        )
        self.assertFalse(
            protocol["object_separation"]["arbitrary_head_or_neuron_subsets_allowed"]
        )
        self.assertIn(
            "gram_rank_and_cancellation_are_navigation_diagnostics_not_state",
            protocol["attachment_audit"]["corrections"],
        )

    def test_blind_inventory_is_complete_and_label_free(self):
        summary = read_json(P375 / "phase375_blind_inventory_summary.json")
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["denominator"]["total_inventory_row_count"], 16632)
        self.assertEqual(summary["quality"]["forbidden_semantic_field_count"], 0)
        self.assertFalse(summary["quality"]["semantic_labels_available"])

    def test_phase375_frozen_gate_rejects_all_templates(self):
        summary = read_json(
            P375 / "phase375_discovery/phase375_discovery_summary.json"
        )
        diagnostic = read_json(
            P375 / "phase375_discovery/phase375_negative_result_diagnostic.json"
        )
        self.assertEqual(summary["denominator"]["total_lexical_evaluation_count"], 1584)
        self.assertEqual(summary["results"]["heterogeneous_level2_count"], 0)
        self.assertFalse(summary["results"]["causal_replay_authorized"])
        self.assertEqual(diagnostic["overall"]["gate_pass_counts"]["absolute_error"], 0)
        self.assertGreater(diagnostic["overall"]["current_error"]["minimum"], 0.75)

    def test_fixed_offsets_are_not_semantic_alignment(self):
        summary = read_json(P376 / "phase376_decision_time_alignment_summary.json")
        self.assertEqual(summary["denominator"]["case_count"], 264)
        self.assertEqual(summary["crossmodel"]["semantic_condition_count"], 88)
        self.assertEqual(summary["crossmodel"]["all_models_within_fixed_t0_t2_count"], 16)
        self.assertFalse(
            summary["results"]["fixed_t0_t2_is_crossmodel_semantic_alignment"]
        )

    def test_decision_aligned_discovery_is_crossmodel_but_not_mechanism(self):
        summary = read_json(
            P376 / "phase376_intervention/phase376_intervention_summary.json"
        )
        self.assertEqual(summary["denominator"]["patched_forward_condition_count"], 9504)
        self.assertEqual(summary["results"]["heterogeneous_level2_transfer_count"], 6)
        self.assertEqual(summary["results"]["heterogeneous_level2_winner_flip_count"], 4)
        self.assertFalse(summary["results"]["language_mechanism_claimed"])

    def test_calibration_replication_keeps_physical_separate(self):
        summary = read_json(
            P377 / "phase377_intervention/phase377_calibration_summary.json"
        )
        self.assertEqual(summary["denominator"]["case_count"], 132)
        self.assertEqual(summary["results"]["heterogeneous_level2_calibration_count"], 4)
        self.assertTrue(summary["results"]["physical_holdout_authorized"])
        self.assertFalse(summary["results"]["language_mechanism_claimed"])

    def test_physical_behavior_denominator_is_complete(self):
        summary = read_json(P378 / "phase378_physical_behavior_analysis_summary.json")
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["denominator"]["case_count"], 96)
        self.assertEqual(summary["denominator"]["strict_correct_case_count"], 96)
        self.assertEqual(
            summary["denominator"]["common_group_counts"],
            {"entity_recency": 4, "relation_binding": 4},
        )
        self.assertFalse(summary["quality"]["failed_groups_replaced"])

    def test_physical_result_is_terminal_carrier_only(self):
        summary = read_json(P378 / "phase378_intervention/phase378_physical_summary.json")
        self.assertEqual(summary["denominator"]["patched_forward_condition_count"], 768)
        self.assertEqual(summary["results"]["physically_confirmed_terminal_carrier_count"], 4)
        self.assertEqual(summary["results"]["upstream_encoding_rule_count"], 0)
        self.assertEqual(summary["results"]["language_path_candidate_count"], 0)
        self.assertFalse(summary["results"]["language_mechanism_claimed"])

    def test_minimality_collapses_expanded_template(self):
        summary = read_json(P378 / "phase378_terminal_carrier_minimality_summary.json")
        results = summary["results"]
        self.assertEqual(
            results["sealed_calibration_and_physical_winner_disagreement_count"], 0
        )
        self.assertFalse(
            results["source_query_additions_required_for_single_token_winner_transfer"]
        )
        self.assertTrue(results["generic_terminal_content_carrier_supported"])
        self.assertFalse(results["multi_route_encoding_mechanism_supported"])

    def test_public_stage_has_strict_claim_boundary(self):
        stage = read_json(ATLAS / "phase378_decision_aligned_stage_summary.json")
        self.assertTrue(stage["assessment"]["carrier_physically_confirmed"])
        self.assertFalse(stage["assessment"]["carrier_is_mechanism_specific"])
        self.assertFalse(stage["assessment"]["upstream_encoding_rule_discovered"])
        self.assertFalse(stage["assessment"]["language_encoding_mechanism_closed"])
        self.assertEqual(stage["objective_denominators"]["minimal_distinct_terminal_carriers"], 2)
        self.assertEqual(stage["objective_denominators"]["strictly_closed_registered_cells"], 0)
        self.assertFalse(stage["single_global_progress_percentage_valid"])

    def test_atlas_and_client_mirrors_match(self):
        names = (
            "phase378_decision_aligned_stage_summary.json",
            "phase378_evidence_nodes.jsonl",
            "phase378_evidence_edges.jsonl",
            "progress.json",
        )
        for name in names:
            self.assertEqual(
                (ATLAS / name).read_bytes(),
                (CLIENT / name).read_bytes(),
                name,
            )

    def test_neuron_atlas_promotes_no_neuron_path(self):
        for root in (NEURON, NEURON_CLIENT):
            manifest = read_json(root / "manifest.json")
            audit = manifest["phase378_audit"]
            self.assertEqual(audit["new_neuron_path_nodes_promoted"], 0)
            self.assertEqual(audit["single_unit_causal_count"], 0)
            self.assertEqual(audit["upstream_encoding_rule_count"], 0)
            self.assertEqual(audit["language_path_count"], 0)


if __name__ == "__main__":
    unittest.main()
