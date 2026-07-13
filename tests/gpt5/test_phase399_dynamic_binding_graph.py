#!/usr/bin/env python3
"""Contract tests for Phase399 dynamic-event publication."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase399_dynamic_binding"
ATLAS = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase399DynamicBindingGraphTest(unittest.TestCase):
    def test_behavior_denominator_and_surface_gate_are_frozen(self):
        payload = read_json(RESULT / "phase399_behavior_freeze_summary.json")
        self.assertEqual(payload["denominator"]["candidate_case_count"], 5376)
        self.assertEqual(payload["denominator"]["qualified_parallel_group_count"], 82)
        self.assertEqual(payload["denominator"]["selected_case_count"], 2880)
        gates = {row["task_surface"]: row for row in payload["surface_gates"]}
        self.assertEqual(gates["field_extraction"]["qualified_group_count"], 1)
        self.assertFalse(gates["field_extraction"]["eligible"])
        self.assertTrue(all(gates[name]["eligible"] for name in (
            "possession_relation", "role_filling", "coreference_resolution"
        )))

    def test_three_split_result_is_model_specific_and_noncausal(self):
        discovery = read_json(RESULT / "phase399_dynamic_discovery_analysis.json")
        calibration = read_json(RESULT / "phase399_dynamic_calibration_validation.json")
        physical = read_json(RESULT / "phase399_dynamic_physical_validation.json")
        self.assertEqual(discovery["results"]["dynamic_chain_discovery_cell_count"], 1)
        self.assertEqual(calibration["results"]["dynamic_chain_validation_cell_count"], 1)
        self.assertEqual(physical["results"]["dynamic_chain_validation_cell_count"], 1)
        self.assertEqual(physical["results"]["crossmodel_surface_count"], 0)
        passing = [row for row in physical["cells"] if row["dynamic_chain_validation_pass"]]
        self.assertEqual([(row["model"], row["surface"]) for row in passing], [("deepseek7b", "role_filling")])
        self.assertFalse(physical["authorization"]["run_joint_causal_intervention"])
        self.assertFalse(physical["authorization"]["head_channel_or_neuron_scan"])

    def test_all_required_event_classes_replicate_without_implying_ordered_chain(self):
        physical = read_json(RESULT / "phase399_dynamic_physical_validation.json")
        for cell in physical["cells"]:
            required = [event for event in cell["event_classes"].values() if event["required_for_chain"]]
            self.assertEqual(len(required), 3)
            self.assertTrue(all(event["validation_pass"] for event in required))
        self.assertEqual(sum(cell["ordered_peak_layer_gate_pass"] for cell in physical["cells"]), 1)

    def test_public_artifacts_match_and_private_rows_are_not_published(self):
        public_names = [
            "phase399_protocol.json",
            "phase399_behavior_freeze_summary.json",
            "phase399_dynamic_trace_protocol.json",
            "phase399_instrument_audit.json",
            "phase399_dynamic_candidate_protocol.json",
            "phase399_dynamic_candidate_freeze.json",
            "phase399_dynamic_discovery_analysis.json",
            "phase399_dynamic_calibration_validation.json",
            "phase399_dynamic_physical_validation.json",
            "phase399_dynamic_binding_stage_summary.json",
        ]
        for name in public_names:
            self.assertEqual(read_json(ATLAS / name), read_json(CLIENT / name))
        self.assertFalse((CLIENT / "event_trajectory_rows.jsonl").exists())
        self.assertFalse((CLIENT / "phase399_dynamic_trace_cases.jsonl").exists())

    def test_neuron_atlas_contains_only_three_aggregate_phase399_events(self):
        nodes = []
        for root in (NEURON, NEURON_CLIENT):
            for path in (root / "partitions").glob("*/*.json"):
                nodes.extend(node for node in read_json(path).get("nodes", []) if node.get("phase399_tested"))
        self.assertEqual(len(nodes), 6)
        for node in nodes:
            self.assertEqual(node["model"], "deepseek7b")
            self.assertEqual(node["family_id"], "language_action")
            self.assertEqual(node["node_type"], "aggregate_dynamic_route_event")
            self.assertFalse(node["is_real_unit"])
            self.assertFalse(node["single_neuron_claim"])
            self.assertFalse(node["phase399_crossmodel_chain_pass"])
            self.assertFalse(node["phase399_causal_gate_open"])

    def test_stage_summary_and_progress_keep_claim_boundary(self):
        stage = read_json(ATLAS / "phase399_dynamic_binding_stage_summary.json")
        self.assertEqual(stage["results"]["ordered_chain_physical_cell_count"], 1)
        self.assertEqual(stage["results"]["ordered_chain_crossmodel_surface_count"], 0)
        self.assertFalse(stage["authorization"]["run_joint_causal_intervention"])
        self.assertFalse(stage["next_stage"]["automatic_continuation_authorized"])
        progress = read_json(ATLAS / "progress.json")
        self.assertFalse(progress["single_global_progress_percentage_valid"])
        dynamic = progress["multi_position_dynamic_binding_stage"]
        self.assertEqual(dynamic["crossmodel_ordered_chain_surfaces"], {"numerator": 0, "denominator": 3})
        self.assertEqual(dynamic["complete_language_paths"], {"numerator": 0, "denominator": 72})

    def test_evidence_graph_does_not_promote_a_causal_path(self):
        nodes = read_jsonl(ATLAS / "phase399_evidence_nodes.jsonl")
        edges = read_jsonl(ATLAS / "phase399_evidence_edges.jsonl")
        self.assertTrue(nodes)
        self.assertTrue(edges)
        self.assertTrue(all(not node["causal"] for node in nodes))
        self.assertTrue(all(not edge["causal_path"] for edge in edges))


if __name__ == "__main__":
    unittest.main()
