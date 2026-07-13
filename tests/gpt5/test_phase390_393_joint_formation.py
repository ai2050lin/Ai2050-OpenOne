from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P390 = ROOT / "tests/gpt5/result/phase390_joint_formation_graph"
P391 = ROOT / "tests/gpt5/result/phase391_local_parent_graph"
P392 = ROOT / "tests/gpt5/result/phase392_parent_boundary_replay"
P393 = ROOT / "tests/gpt5/result/phase393_attribute_content_holdout"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class Phase390To393JointFormationTests(unittest.TestCase):
    def test_phase390_freezes_large_behavior_denominator_before_internal_trace(self) -> None:
        freeze = read_json(P390 / "phase390_behavior_freeze_summary.json")
        discovery = read_json(P390 / "phase390_discovery_candidate_freeze.json")
        self.assertEqual(freeze["denominator"]["candidate_case_count"], 1728)
        self.assertEqual(freeze["denominator"]["qualified_parallel_group_count"], 88)
        self.assertEqual(freeze["eligible_mechanisms"], ["field_extraction"])
        self.assertEqual(discovery["denominator"]["model_candidate_count"], 144)
        self.assertEqual(discovery["denominator"]["crossmodel_candidate_count"], 48)
        self.assertEqual(discovery["denominator"]["passing_crossmodel_candidate_count"], 0)
        self.assertFalse(discovery["authorization"]["single_neuron_scan"])

    def test_phase391_local_parent_layout_replicates_without_causal_promotion(self) -> None:
        physical = read_json(P391 / "phase391_physical_summary.json")
        self.assertEqual(physical["denominator"], {
            "frozen_physical_candidate_count": 1,
            "passing_physical_candidate_count": 1,
        })
        candidate = physical["physical_crossmodel_candidates"][0]
        self.assertEqual(candidate["receiver_coordinate"], "query_integrated")
        self.assertEqual(set(candidate["models_passing"]), {"qwen3", "glm4", "deepseek7b"})
        self.assertFalse(candidate["causal_language_path_claim"])
        self.assertFalse(physical["authorization"]["single_neuron_scan"])

    def test_phase392_joint_switches_fail_specificity_controls(self) -> None:
        causal = read_json(P392 / "phase392_causal_summary.json")
        self.assertEqual(causal["denominator"]["direction_count"], 144)
        self.assertEqual(causal["results"]["strict_answer_switch_count"], 139)
        self.assertEqual(causal["results"]["models_passing_parent_boundary_participation"], 0)
        self.assertEqual(causal["results"]["models_passing_language_function_path"], 0)
        self.assertFalse(causal["authorization"]["promote_parent_boundary_causal_edge"])

    def test_phase393_separates_attribute_transport_from_depth_specificity(self) -> None:
        summary = read_json(P393 / "phase393_summary.json")
        self.assertEqual(summary["denominator"]["direction_count"], 72)
        self.assertEqual(summary["denominator"]["phase392_group_overlap"], 0)
        self.assertEqual(summary["results"]["models_passing_attribute_transport"], 3)
        self.assertEqual(summary["results"]["models_passing_depth_specificity"], 0)
        attribute_switches = sum(
            round(row["attribute_answer_switch_rate"] * row["direction_count"])
            for row in summary["models"]
        )
        wrong_depth_switches = sum(
            round(row["wrong_depth_attribute_switch_rate"] * row["direction_count"])
            for row in summary["models"]
        )
        structure_switches = sum(
            round(row["structure_answer_switch_rate"] * row["direction_count"])
            for row in summary["models"]
        )
        random_switches = sum(
            round(row["random_answer_switch_rate"] * row["direction_count"])
            for row in summary["models"]
        )
        self.assertEqual(attribute_switches, 71)
        self.assertEqual(wrong_depth_switches, 72)
        self.assertEqual(structure_switches, 0)
        self.assertEqual(random_switches, 0)
        self.assertTrue(summary["authorization"]["promote_attribute_content_transport_edge"])
        self.assertFalse(summary["authorization"]["promote_depth_specific_specialized_path"])
        self.assertFalse(summary["authorization"]["single_neuron_scan"])

    def test_atlas_and_client_publish_identical_strict_stage_boundary(self) -> None:
        expected_results = {
            "phase393_attribute_answer_switch_count": 71,
            "phase393_wrong_depth_attribute_switch_count": 72,
            "models_passing_attribute_transport": 3,
            "models_passing_depth_specificity": 0,
            "complete_language_path_count": 0,
            "single_neuron_causal_path_count": 0,
        }
        stages = []
        for root in (ATLAS, CLIENT):
            manifest = read_json(root / "manifest.json")
            progress = read_json(root / "progress.json")
            stage = read_json(root / "phase393_joint_formation_stage_summary.json")
            stages.append(stage)
            self.assertEqual(manifest["last_phase"], "Phase399-MultiPositionDynamicBindingStage")
            self.assertEqual(progress["last_phase"], "Phase399-MultiPositionDynamicBindingStage")
            self.assertIn("phase397", manifest)
            self.assertIn("phase399", manifest)
            for key, value in expected_results.items():
                self.assertEqual(stage["results"][key], value)
            self.assertFalse(stage["authorization"]["show_depth_specialized_path"])
            self.assertFalse(stage["authorization"]["show_specific_neuron_path"])
        self.assertEqual(stages[0], stages[1])

    def test_public_graph_promotes_only_scoped_attribute_transport(self) -> None:
        edges = read_jsonl(ATLAS / "phase393_evidence_edges.jsonl")
        causal_edges = [row for row in edges if row["causal_path"]]
        self.assertEqual(len(causal_edges), 1)
        self.assertEqual(causal_edges[0]["edge_type"], "controlled_attribute_state_transport")
        self.assertFalse(causal_edges[0]["complete_language_path"])

    def test_neuron_atlas_stays_at_zero_promoted_paths(self) -> None:
        for root in (NEURON_ATLAS, NEURON_CLIENT):
            manifest = read_json(root / "manifest.json")
            self.assertGreaterEqual(manifest["phase"], 393)
            audit = manifest["phase393_audit"]
            self.assertEqual(audit["new_neuron_path_nodes_promoted"], 0)
            self.assertEqual(audit["single_unit_causal_count"], 0)
            self.assertEqual(audit["language_path_count"], 0)
            self.assertFalse(manifest["evidence_boundary"]["candidate_depth_specificity_available"])
            checksums = read_json(root / "checksums.json")
            entries = {row["path"]: row["sha256"] for row in checksums["files"]}
            phase393 = "phase393_joint_formation_stage_summary.json"
            phase396 = "phase396_binding_separation_stage_summary.json"
            latest = "phase397_factor_separated_binding_stage_summary.json"
            self.assertEqual(entries[phase393], sha256(root / phase393))
            self.assertEqual(entries[phase396], sha256(root / phase396))
            self.assertEqual(entries[latest], sha256(root / latest))


if __name__ == "__main__":
    unittest.main()
