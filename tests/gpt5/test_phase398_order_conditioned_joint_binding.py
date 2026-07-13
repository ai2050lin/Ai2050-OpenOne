#!/usr/bin/env python3
"""Regression checks for frozen Phase398 public evidence boundaries."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"
ATLAS = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Phase398RegressionTest(unittest.TestCase):
    def test_behavior_and_trace_denominators(self) -> None:
        behavior = read_json(OUT / "phase398_behavior_freeze_summary.json")
        trace = read_json(OUT / "phase398_query_trace_protocol.json")
        self.assertEqual(behavior["denominator"]["candidate_case_count"], 3456)
        self.assertEqual(behavior["denominator"]["qualified_parallel_group_count"], 68)
        self.assertEqual(behavior["denominator"]["selected_case_count"], 2304)
        self.assertEqual(trace["denominator"]["discovery_case_count"], 1152)
        self.assertEqual(trace["denominator"]["calibration_case_count"], 576)
        self.assertEqual(trace["denominator"]["physical_holdout_case_count"], 576)

    def test_observational_and_causal_boundaries(self) -> None:
        discovery = read_json(OUT / "phase398_discovery_analysis.json")
        calibration = read_json(OUT / "phase398_order_conditioned_calibration_validation.json")
        physical = read_json(OUT / "phase398_order_conditioned_physical_validation.json")
        causal = read_json(OUT / "phase398_order_conditioned_causal_analysis.json")
        self.assertEqual(discovery["results"]["qualified_model_surface_cell_count"], 0)
        self.assertEqual(calibration["results"]["passing_model_surface_cell_count"], 9)
        self.assertEqual(physical["results"]["passing_model_surface_cell_count"], 9)
        self.assertEqual(causal["results"]["passing_causal_cell_count"], 0)
        self.assertEqual(causal["results"]["same_order_total_answer_switch_count"], 10)
        self.assertFalse(causal["authorization"]["run_single_neuron_localization"])

    def test_public_atlas_matches_research(self) -> None:
        stage = read_json(ATLAS / "phase398_order_conditioned_joint_binding_stage_summary.json")
        progress = read_json(ATLAS / "progress.json")
        manifest = read_json(NEURON / "manifest.json")
        self.assertEqual(stage["results"]["order_conditioned_roq_physical_cells"], 9)
        self.assertEqual(stage["results"]["order_conditioned_single_position_causal_cells"], 0)
        self.assertEqual(progress["last_phase"], "Phase399-MultiPositionDynamicBindingStage")
        self.assertFalse(progress["single_global_progress_percentage_valid"])
        self.assertEqual(manifest["phase"], 399)
        self.assertEqual(manifest["phase398_audit"]["new_aggregate_state_anchor_count"], 9)
        self.assertEqual(manifest["phase398_audit"]["new_neuron_path_nodes_promoted"], 0)
        self.assertEqual(manifest["phase399_audit"]["new_aggregate_dynamic_event_count"], 3)
        self.assertEqual(manifest["phase399_audit"]["new_neuron_path_nodes_promoted"], 0)

    def test_phase398_nodes_are_not_neurons(self) -> None:
        models = ("qwen3", "glm4", "deepseek7b")
        families = ("content_knowledge", "language_action", "reasoning_constraint")
        nodes = []
        for family in families:
            for model in models:
                partition = read_json(NEURON / f"partitions/{family}/{model}.json")
                phase_nodes = [node for node in partition["nodes"] if node.get("phase398_tested")]
                self.assertEqual(len(phase_nodes), 1)
                nodes.extend(phase_nodes)
        self.assertEqual(len(nodes), 9)
        self.assertTrue(all(node["node_type"] == "aggregate_interaction_trajectory_anchor" for node in nodes))
        self.assertTrue(all(node["is_real_unit"] is False for node in nodes))
        self.assertTrue(all(node["single_neuron_claim"] is False for node in nodes))
        self.assertTrue(all(node["phase398_causal_gate_pass"] is False for node in nodes))


if __name__ == "__main__":
    unittest.main()
