#!/usr/bin/env python3
"""Contract tests for Phase401 execution and local-edge evidence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase401_local_edge_graph"
ATLAS = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase401LocalEdgeGraphTest(unittest.TestCase):
    def test_protocol_freezes_execution_models_splits_and_controls(self):
        protocol = read_json(RESULT / "phase401_local_edge_protocol.json")
        self.assertEqual(
            protocol["models_in_execution_order"],
            ["qwen3", "glm4", "deepseek7b"],
        )
        self.assertEqual(protocol["execution_contract"]["batch_size"], 1)
        self.assertEqual(protocol["behavior_denominator"]["candidate_case_count"], 4608)
        self.assertEqual(
            len(protocol["counterfactual_controls"]["required_separately"]), 8
        )
        self.assertFalse(
            protocol["authorization"]["head_channel_or_neuron_scan"]
        )
        amendment = protocol["protocol_amendment_001"]
        self.assertTrue(amendment["all_formal_models_reexecuted_after_amendment"])
        self.assertIn("all_numeric_gates", amendment["unchanged"])

    def test_behavior_batch_and_trace_denominators_are_separate(self):
        behavior = read_json(RESULT / "phase401_behavior_freeze_summary.json")
        denominator = behavior["denominator"]
        self.assertEqual(denominator["candidate_case_count"], 4608)
        self.assertEqual(denominator["qualified_parallel_group_count"], 66)
        self.assertEqual(denominator["selected_case_count"], 1536)
        self.assertEqual(
            behavior["eligible_surfaces"], ["possession_relation", "role_filling"]
        )
        self.assertEqual(
            sum(row["semantic_correct_count"] for row in behavior["model_results"].values()),
            4557,
        )
        batch = read_json(RESULT / "phase401_batch_sensitivity_audit.json")
        self.assertEqual(batch["case_count"], 192)
        self.assertEqual(batch["batch_sensitive_case_count"], 7)
        self.assertEqual(batch["semantic_correctness_difference_count"], 1)
        self.assertFalse(batch["result"]["batch_shape_is_empirically_invariant_on_pilot"])
        self.assertFalse(batch["result"]["pilot_rows_enter_formal_mechanism_denominator"])

    def test_same_shape_ledger_is_complete_without_mechanism_claim(self):
        instrument = read_json(RESULT / "phase401_instrument_audit.json")
        self.assertTrue(instrument["joint_gate"]["pass"])
        self.assertEqual(
            sum(row["case_count"] for row in instrument["models"].values()), 96
        )
        self.assertEqual(
            sum(row["quality_pass_case_count"] for row in instrument["models"].values()),
            96,
        )
        self.assertTrue(instrument["authorization"]["run_discovery_local_edges"])
        self.assertFalse(instrument["authorization"]["run_calibration"])
        self.assertFalse(instrument["claim_boundary"]["instrument_pass_is_a_language_edge"])

    def test_discovery_has_complete_sources_and_zero_candidates(self):
        audit = read_json(RESULT / "phase401_local_edge_discovery_audit.json")
        self.assertTrue(audit["all_source_denominators_complete"])
        self.assertEqual(
            sum(item["aggregate_row_count"] for item in audit["source_completeness"].values()),
            14976,
        )
        self.assertTrue(audit["protocol_contradiction"]["present"])
        self.assertFalse(
            audit["protocol_contradiction"]["sensitivity_has_authorization_power"]
        )
        for model in audit["model_surface_summary"].values():
            for surface in model.values():
                self.assertEqual(surface["strict_passing_layer_count"], 0)
                self.assertEqual(surface["sensitivity_passing_layer_count"], 0)
                self.assertIsNone(surface["strict_candidate"])
                self.assertIsNone(surface["sensitivity_candidate_non_authorizing"])
        self.assertEqual(audit["strict_crossmodel_candidates"], [])
        self.assertEqual(audit["sensitivity_crossmodel_candidates_non_authorizing"], [])
        self.assertFalse(audit["authorization"]["run_calibration"])

    def test_group_first_stage_profile_rejects_function_specific_edge(self):
        profile = read_json(RESULT / "phase401_local_edge_stage_profile.json")
        self.assertEqual(profile["total_pair_row_count"], 239616)
        self.assertEqual(profile["group_stage_row_count"], 59904)
        self.assertTrue(profile["group_first_independent_denominator"])
        self.assertEqual(
            profile["registered_direct_attention_local_physical_candidate_count"], 0
        )
        for model in profile["stage_summary"].values():
            for surface in model.values():
                self.assertEqual(surface["attention"]["physical_passing_layer_count"], 0)
                self.assertIsNone(surface["attention"]["earliest_physical_candidate"])
                self.assertTrue(
                    surface["attention"]["eligible_for_local_edge_registration"]
                )
                for stage in ("post_attention", "mlp", "layer_output"):
                    self.assertFalse(surface[stage]["eligible_for_local_edge_registration"])

    def test_public_atlas_matches_and_excludes_private_pair_rows(self):
        names = [
            "phase401_local_edge_protocol.json",
            "phase401_behavior_protocol.json",
            "phase401_batch_sensitivity_audit.json",
            "phase401_behavior_freeze_summary.json",
            "phase401_trace_protocol.json",
            "phase401_instrument_audit.json",
            "phase401_local_edge_execution_freeze.json",
            "phase401_local_edge_discovery_audit.json",
            "phase401_local_edge_stage_profile.json",
            "phase401_execution_semantic_local_edge_stage_summary.json",
        ]
        for name in names:
            self.assertEqual(read_json(ATLAS / name), read_json(CLIENT / name))
        self.assertFalse((CLIENT / "pair_rows.jsonl").exists())
        self.assertFalse((CLIENT / "phase401_local_edge_stage_group_rows.jsonl").exists())
        self.assertFalse((CLIENT / "protocol/private").exists())

    def test_evidence_and_neuron_atlas_preserve_negative_boundary(self):
        nodes = read_jsonl(ATLAS / "phase401_evidence_nodes.jsonl")
        edges = read_jsonl(ATLAS / "phase401_evidence_edges.jsonl")
        self.assertEqual(len(nodes), 6)
        self.assertEqual(len(edges), 6)
        self.assertTrue(all(not row["causal"] for row in nodes))
        self.assertTrue(all(not row["causal_path"] for row in edges))
        for root in (NEURON, NEURON_CLIENT):
            manifest = read_json(root / "manifest.json")
            self.assertGreaterEqual(manifest["phase"], 401)
            self.assertEqual(manifest["phase401_audit"]["new_neuron_path_nodes_promoted"], 0)
            self.assertIn("phase401_audit", manifest)
            self.assertFalse(manifest["evidence_boundary"]["single_unit_causal_closure"])

    def test_progress_and_client_preserve_phase401_stage(self):
        progress = read_json(ATLAS / "progress.json")
        self.assertIn("local_edge_stage", progress)
        self.assertFalse(progress["single_global_progress_percentage_valid"])
        self.assertEqual(
            progress["local_edge_stage"]["strict_local_edge_layers"],
            {"numerator": 0, "denominator": 208},
        )
        dashboard = (
            ROOT / "frontend/src/blueprint/AtlasControlDashboard.jsx"
        ).read_text(encoding="utf-8")
        kernel = (
            ROOT / "frontend/src/blueprint/EvidenceKernelDashboard.jsx"
        ).read_text(encoding="utf-8")
        self.assertIn("progress?.local_edge_stage", dashboard)
        self.assertIn("0/208 local-edge layers", dashboard)
        self.assertIn("atlas?.phase401_audit", kernel)
        self.assertIn("Phase 401 严格局部边层", kernel)


if __name__ == "__main__":
    unittest.main()
