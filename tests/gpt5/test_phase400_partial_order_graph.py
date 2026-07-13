#!/usr/bin/env python3
"""Contract tests for Phase400 partial-order evidence and publication."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase400_partial_order"
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


class Phase400PartialOrderGraphTest(unittest.TestCase):
    def test_protocol_was_frozen_and_forbids_fine_search(self):
        protocol = read_json(RESULT / "phase400_partial_order_protocol.json")
        self.assertTrue(
            protocol["prior_information_boundary"][
                "phase400_thresholds_frozen_before_phase400_behavior_execution"
            ]
        )
        self.assertEqual(len(protocol["required_nodes"]), 6)
        self.assertEqual(len(protocol["required_edges"]), 5)
        self.assertFalse(
            protocol["causal_authorization"]["head_channel_neuron_scan_before_joint_gate"]
        )

    def test_behavior_denominator_and_split_selection_are_fixed(self):
        payload = read_json(RESULT / "phase400_behavior_freeze_summary.json")
        denominator = payload["denominator"]
        self.assertEqual(denominator["candidate_case_count"], 4608)
        self.assertEqual(denominator["qualified_parallel_group_count"], 74)
        self.assertEqual(denominator["selected_case_count"], 1536)
        self.assertEqual(
            payload["eligible_surfaces"], ["possession_relation", "role_filling"]
        )
        gates = {row["task_surface"]: row for row in payload["surface_gates"]}
        self.assertFalse(gates["coreference_resolution"]["eligible"])
        self.assertFalse(gates["field_extraction"]["eligible"])

    def test_discovery_partial_order_is_observational_and_nonpredictive(self):
        discovery = read_json(RESULT / "phase400_partial_order_discovery.json")
        self.assertEqual(discovery["denominator"]["case_count"], 768)
        self.assertEqual(discovery["denominator"]["group_model_cell_count"], 48)
        self.assertTrue(discovery["denominator"]["all_collection_quality_gates_pass"])
        self.assertEqual(discovery["results"]["partial_order_graph_cell_count"], 5)
        self.assertEqual(
            discovery["results"]["crossmodel_isomorphism_surface_count"], 1
        )
        self.assertEqual(discovery["results"]["prediction_pass_cell_count"], 0)
        passing = [
            row["surface"]
            for row in discovery["crossmodel_surfaces"]
            if row["crossmodel_functional_isomorphism_pass"]
        ]
        self.assertEqual(passing, ["possession_relation"])
        self.assertTrue(all(not cell["prediction"]["prediction_pass"] for cell in discovery["cells"]))
        self.assertFalse(discovery["authorization"]["run_joint_causal_intervention"])

    def test_calibration_failure_is_preserved_without_override(self):
        audit = read_json(
            RESULT / "phase400_calibration_collection_quality_audit.json"
        )
        self.assertEqual(audit["denominator"]["case_count"], 384)
        self.assertEqual(audit["denominator"]["quality_group_model_cell_count"], 23)
        self.assertEqual(audit["denominator"]["group_model_cell_count"], 24)
        self.assertEqual(audit["denominator"]["first_answer_replay_match_count"], 383)
        self.assertEqual(
            audit["denominator"]["target_completion_replay_match_count"], 384
        )
        self.assertFalse(audit["diagnosis"]["numeric_conservation_gate_failed"])
        self.assertFalse(
            audit["diagnosis"]["parent_capture_hooks_changed_current_single_case_top1"]
        )
        self.assertFalse(
            audit["diagnosis"]["batch_size_1_vs_8_first_token_invariance"]
        )
        self.assertTrue(all(not value for value in audit["non_override_audit"].values()))
        self.assertFalse(audit["authorization"]["open_physical_holdout"])

    def test_physical_holdout_and_fine_resolution_remain_closed(self):
        physical = read_json(RESULT / "phase400_partial_order_physical.json")
        self.assertTrue(physical["physical_holdout_remains_unopened"])
        self.assertEqual(physical["case_count_consumed"], 0)
        self.assertFalse(physical["authorization"]["run_joint_causal_intervention"])
        self.assertFalse(physical["authorization"]["head_channel_or_neuron_scan"])

    def test_public_artifacts_match_and_private_data_are_not_published(self):
        names = [
            "phase400_protocol.json",
            "phase400_behavior_freeze_summary.json",
            "phase400_partial_order_protocol.json",
            "phase400_dynamic_trace_protocol.json",
            "phase400_instrument_audit.json",
            "phase400_partial_order_discovery.json",
            "phase400_partial_order_candidate_freeze.json",
            "phase400_calibration_collection_quality_audit.json",
            "phase400_partial_order_calibration.json",
            "phase400_partial_order_physical.json",
            "phase400_dynamic_partial_order_stage_summary.json",
        ]
        for name in names:
            self.assertEqual(read_json(ATLAS / name), read_json(CLIENT / name))
        self.assertFalse((CLIENT / "phase400_failed_replay_diagnostic.json").exists())
        self.assertFalse((CLIENT / "event_trajectory_rows.jsonl").exists())
        self.assertFalse((CLIENT / "raw_anchors").exists())

    def test_evidence_graph_and_neuron_atlas_keep_claim_boundary(self):
        nodes = read_jsonl(ATLAS / "phase400_evidence_nodes.jsonl")
        edges = read_jsonl(ATLAS / "phase400_evidence_edges.jsonl")
        self.assertEqual(len(nodes), 6)
        self.assertEqual(len(edges), 6)
        self.assertTrue(all(not node["causal"] for node in nodes))
        self.assertTrue(all(not edge["causal_path"] for edge in edges))
        for root in (NEURON, NEURON_CLIENT):
            promoted = []
            for path in (root / "partitions").glob("*/*.json"):
                promoted.extend(
                    node
                    for node in read_json(path).get("nodes", [])
                    if node.get("phase400_tested")
                )
            self.assertEqual(promoted, [])
            manifest = read_json(root / "manifest.json")
            self.assertEqual(
                manifest["phase400_audit"]["new_neuron_path_nodes_promoted"], 0
            )

    def test_stage_summary_and_progress_are_vectorized(self):
        stage = read_json(ATLAS / "phase400_dynamic_partial_order_stage_summary.json")
        self.assertFalse(stage["assessment"]["terminal_answer_prediction_pass"])
        self.assertFalse(stage["assessment"]["calibration_collection_quality_pass"])
        self.assertFalse(stage["assessment"]["physical_holdout_opened"])
        self.assertFalse(stage["authorization"]["show_candidate_as_causal_path"])
        progress = read_json(ATLAS / "progress.json")
        self.assertFalse(progress["single_global_progress_percentage_valid"])
        partial = progress["dynamic_partial_order_stage"]
        self.assertEqual(
            partial["discovery_partial_order_graph_cells"],
            {"numerator": 5, "denominator": 6},
        )
        self.assertEqual(
            partial["discovery_prediction_cells"],
            {"numerator": 0, "denominator": 6},
        )

    def test_client_reads_phase400_as_latest_stage(self):
        dashboard_source = (
            ROOT
            / "frontend"
            / "src"
            / "blueprint"
            / "AtlasControlDashboard.jsx"
        ).read_text(encoding="utf-8")
        kernel_source = (
            ROOT
            / "frontend"
            / "src"
            / "blueprint"
            / "EvidenceKernelDashboard.jsx"
        ).read_text(encoding="utf-8")
        self.assertIn("progress?.dynamic_partial_order_stage", dashboard_source)
        self.assertIn("dynamicPartialOrderRows", dashboard_source)
        self.assertIn("0/6 prediction cells", dashboard_source)
        self.assertIn("物理留出仍为 0/384", dashboard_source)
        self.assertIn("atlas?.phase400_audit", kernel_source)
        self.assertIn("Phase 400 预测门合格单元", kernel_source)
        self.assertIn("物理留出保持 0/384", kernel_source)


if __name__ == "__main__":
    unittest.main()
