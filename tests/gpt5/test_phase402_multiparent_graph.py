#!/usr/bin/env python3
"""Contract tests for Phase402 multi-parent direct-child evidence."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase402_multiparent_graph"
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


class Phase402MultiParentGraphTest(unittest.TestCase):
    def test_protocol_freezes_real_partitions_and_downstream_gates(self):
        protocol = read_json(RESULT / "phase402_multiparent_protocol.json")
        self.assertEqual(
            protocol["models_in_execution_order"],
            ["qwen3", "glm4", "deepseek7b"],
        )
        self.assertEqual(protocol["execution_contract"]["batch_size"], 1)
        self.assertEqual(
            protocol["fresh_behavior_denominator"]["candidate_case_count"], 6912
        )
        partitions = protocol["executable_parent_partition"]
        self.assertTrue(partitions["partition_is_disjoint"])
        self.assertTrue(partitions["partition_conserves_every_position_up_to_receiver"])
        self.assertFalse(partitions["remaining_prefix_is_generated_history"])
        self.assertFalse(protocol["authorization"]["run_physical_holdout"])
        self.assertFalse(protocol["authorization"]["run_head_channel_or_neuron_scan"])

    def test_behavior_qualification_preserves_six_surface_denominator(self):
        behavior = read_json(RESULT / "phase402_behavior_freeze_summary.json")
        denominator = behavior["denominator"]
        self.assertEqual(denominator["candidate_case_count"], 6912)
        self.assertEqual(denominator["candidate_parallel_group_count"], 144)
        self.assertEqual(denominator["qualified_parallel_group_count"], 68)
        self.assertEqual(denominator["selected_case_count"], 1152)
        self.assertEqual(
            behavior["eligible_surfaces"],
            ["role_filling", "conditional_presence"],
        )
        self.assertEqual(
            sum(
                row["semantic_correct_count"]
                for row in behavior["model_results"].values()
            ),
            5585,
        )
        surfaces = {row["task_surface"]: row for row in behavior["surface_gates"]}
        self.assertEqual(surfaces["two_step_composition"]["qualified_group_count"], 0)
        self.assertEqual(surfaces["number_agreement"]["qualified_group_count"], 0)

    def test_partition_instrument_is_complete(self):
        trace = read_json(RESULT / "phase402_trace_protocol.json")
        self.assertTrue(trace["parent_partition"]["disjoint_and_prefix_conserving"])
        self.assertFalse(
            trace["parent_partition"]["remaining_prefix_is_generated_history"]
        )
        instrument = read_json(RESULT / "phase402_instrument_audit.json")
        self.assertTrue(instrument["valid"])
        self.assertEqual(
            sum(row["row_count"] for row in instrument["models"].values()), 3328
        )
        self.assertEqual(
            sum(row["passing_row_count"] for row in instrument["models"].values()),
            3328,
        )
        self.assertLessEqual(
            max(
                row["max_empty_subset_attention_relative_error"]
                for row in instrument["models"].values()
            ),
            0.01,
        )

    def test_discovery_denominator_and_gate_flow_are_frozen(self):
        freeze = read_json(RESULT / "phase402_discovery_execution_freeze.json")
        self.assertEqual(freeze["discovery_denominator"]["case_count"], 576)
        self.assertEqual(
            freeze["discovery_denominator"]["subset_count_including_empty"], 16
        )
        self.assertFalse(freeze["authorization"]["run_calibration"])
        audit = read_json(RESULT / "phase402_discovery_audit.json")
        self.assertEqual(audit["denominator"]["pair_metric_count"], 2875392)
        self.assertEqual(
            audit["denominator"]["group_layer_subset_metric_count"], 179712
        )
        self.assertEqual(audit["denominator"]["joint_group_layer_subset_count"], 13728)
        self.assertEqual(audit["gate_flow"]["true_base_gate_pass_count"], 5909)
        self.assertEqual(audit["gate_flow"]["joint_above_best_singleton_count"], 641)
        self.assertEqual(
            audit["gate_flow"]["all_controls_pass_count_independent_of_other_gates"],
            18,
        )
        self.assertEqual(audit["gate_flow"]["strict_group_layer_candidate_count"], 8)

    def test_local_hints_do_not_become_model_or_crossmodel_candidates(self):
        rows = read_jsonl(
            RESULT / "discovery_analysis/phase402_group_layer_candidate_rows.jsonl"
        )
        strict = [row for row in rows if row["strict_group_layer_candidate"]]
        self.assertEqual(len(strict), 8)
        self.assertEqual({row["subset_id"] for row in strict}, {"S0110"})
        self.assertEqual({row["depth_zone"] for row in strict}, {"early"})
        self.assertEqual(sum(row["model"] == "qwen3" for row in strict), 0)
        self.assertEqual(sum(row["model"] == "glm4" for row in strict), 1)
        self.assertEqual(sum(row["model"] == "deepseek7b" for row in strict), 7)
        audit = read_json(RESULT / "phase402_discovery_audit.json")
        self.assertEqual(audit["gate_flow"]["model_candidate_count"], 0)
        self.assertEqual(audit["crossmodel_candidates"], [])
        self.assertEqual(audit["partial_two_model_candidates"], [])
        self.assertFalse(audit["authorization"]["freeze_and_run_calibration"])

    def test_decimal_threshold_is_audited_without_posthoc_relaxation(self):
        precision = read_json(RESULT / "phase402_discovery_audit.json")[
            "protocol_precision_audit"
        ]
        self.assertEqual(precision["stored_group_rate_threshold"], 0.666666667)
        self.assertEqual(precision["groups_per_surface"], 6)
        self.assertEqual(precision["effective_required_count"], 5)
        self.assertIn("honored_without_posthoc_relaxation", precision["note"])

    def test_public_atlas_is_compact_and_matches_client(self):
        names = [
            "phase402_multiparent_protocol.json",
            "phase402_behavior_protocol.json",
            "phase402_behavior_freeze_summary.json",
            "phase402_trace_protocol.json",
            "phase402_instrument_audit.json",
            "phase402_discovery_execution_freeze.json",
            "phase402_discovery_audit.json",
            "phase402_multiparent_direct_child_stage_summary.json",
        ]
        for name in names:
            self.assertEqual(read_json(ATLAS / name), read_json(CLIENT / name))
        strict = read_jsonl(ATLAS / "phase402_strict_local_cells.jsonl")
        self.assertEqual(strict, read_jsonl(CLIENT / "phase402_strict_local_cells.jsonl"))
        self.assertEqual(len(strict), 8)
        self.assertFalse((CLIENT / "phase402_group_layer_candidate_rows.jsonl").exists())
        self.assertFalse((CLIENT / "phase402_model_candidate_rows.jsonl").exists())
        self.assertFalse((CLIENT / "multiparent").exists())

    def test_evidence_and_neuron_atlas_preserve_negative_boundary(self):
        nodes = read_jsonl(ATLAS / "phase402_evidence_nodes.jsonl")
        edges = read_jsonl(ATLAS / "phase402_evidence_edges.jsonl")
        self.assertEqual(len(nodes), 6)
        self.assertEqual(len(edges), 6)
        self.assertTrue(all(not row["causal"] for row in nodes))
        self.assertTrue(all(not row["causal_path"] for row in edges))
        for root in (NEURON, NEURON_CLIENT):
            manifest = read_json(root / "manifest.json")
            self.assertEqual(manifest["phase"], 402)
            self.assertEqual(
                manifest["phase402_audit"]["new_neuron_path_nodes_promoted"], 0
            )
            self.assertFalse(
                manifest["evidence_boundary"][
                    "crossmodel_joint_parent_candidate_available"
                ]
            )
            self.assertFalse(manifest["evidence_boundary"]["single_unit_causal_closure"])

    def test_progress_and_client_use_phase402_as_latest(self):
        progress = read_json(ATLAS / "progress.json")
        self.assertEqual(progress["last_phase"], "Phase402-MultiParentDirectChildStage")
        self.assertFalse(progress["single_global_progress_percentage_valid"])
        self.assertEqual(
            progress["multiparent_direct_child_stage"]["strict_local_joint_cells"],
            {"numerator": 8, "denominator": 13728},
        )
        dashboard = (
            ROOT / "frontend/src/blueprint/AtlasControlDashboard.jsx"
        ).read_text(encoding="utf-8")
        kernel = (
            ROOT / "frontend/src/blueprint/EvidenceKernelDashboard.jsx"
        ).read_text(encoding="utf-8")
        self.assertIn("progress?.multiparent_direct_child_stage", dashboard)
        self.assertIn("8/13,728", dashboard)
        self.assertIn("atlas?.phase402_audit", kernel)
        self.assertIn("Phase 402 严格局部联合单元", kernel)


if __name__ == "__main__":
    unittest.main()
