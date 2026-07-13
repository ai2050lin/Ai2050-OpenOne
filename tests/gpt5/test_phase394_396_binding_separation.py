from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P394 = ROOT / "tests/gpt5/result/phase394_binding_separation"
P395 = ROOT / "tests/gpt5/result/phase395_natural_binding"
P396 = ROOT / "tests/gpt5/result/phase396_field_binding_physical"
ATLAS_ROOTS = (
    ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
    ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
)
NEURON_ROOTS = (
    ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1",
    ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1",
)
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase394To396BindingSeparationTests(unittest.TestCase):
    def test_formal_pointer_failure_is_not_exported_as_natural_binding_absence(self) -> None:
        freeze = read_json(P394 / "phase394_behavior_freeze_summary.json")
        self.assertEqual(freeze["denominator"]["candidate_case_count"], 864)
        self.assertEqual(freeze["denominator"]["qualified_parallel_group_count"], 0)
        self.assertFalse(freeze["authorization"]["run_internal_event_collection"])
        self.assertFalse(freeze["claim_boundary"]["formal_pointer_failure_means_no_binding_state"])

    def test_natural_denominator_and_observational_candidate_are_frozen(self) -> None:
        behavior = read_json(P395 / "phase395_behavior_freeze_summary.json")
        candidate = read_json(P395 / "phase395_discovery_candidate_freeze.json")
        self.assertEqual(behavior["denominator"]["candidate_case_count"], 864)
        self.assertEqual(behavior["denominator"]["qualified_parallel_group_count"], 65)
        self.assertEqual(behavior["eligible_surfaces"], ["field_extraction", "entity_recency"])
        self.assertTrue(candidate["frozen_candidate"]["crossmodel_crosssurface_discovery_gate_pass"])
        self.assertEqual(
            {model: rows["candidate_layer"] for model, rows in candidate["frozen_candidate"]["model_layers"].items()},
            {"qwen3": 20, "glm4": 22, "deepseek7b": 15},
        )
        self.assertFalse(candidate["frozen_candidate"]["causal_binding_claim"])

    def test_calibration_is_local_positive_and_crosssurface_negative(self) -> None:
        analysis = read_json(P395 / "phase395_causal_calibration_analysis.json")
        self.assertEqual(analysis["denominator"]["direction_count"], 144)
        self.assertEqual(analysis["denominator"]["scenario_count"], 1296)
        self.assertEqual(analysis["results"]["local_static_same_literal_context_transport_cell_count"], 4)
        self.assertTrue(analysis["results"]["crossmodel_field_extraction_gate_pass"])
        self.assertFalse(analysis["results"]["crossmodel_entity_recency_gate_pass"])
        self.assertFalse(analysis["results"]["crossmodel_crosssurface_shared_state_gate_pass"])
        self.assertEqual(analysis["results"]["maximum_identity_effect"], 0.0)
        self.assertEqual(analysis["results"]["maximum_patch_locality_error"], 0.0)
        self.assertFalse(analysis["authorization"]["phase395_physical_holdout"])
        self.assertFalse(analysis["authorization"]["single_neuron_scan"])

    def test_field_specific_physical_replication_passes_all_frozen_controls(self) -> None:
        analysis = read_json(P396 / "phase396_physical_analysis.json")
        self.assertEqual(analysis["denominator"]["direction_count"], 72)
        self.assertEqual(analysis["denominator"]["scenario_count"], 648)
        self.assertEqual(analysis["results"]["physical_model_cell_pass_count"], 3)
        self.assertTrue(analysis["results"]["crossmodel_field_specific_physical_replication_gate_pass"])
        self.assertEqual(analysis["results"]["crosssurface_shared_binding_rule_count"], 0)
        self.assertEqual(analysis["results"]["single_neuron_mechanism_count"], 0)
        context_switches = sum(
            cell["scenario_summaries"]["donor_same_literal_candidate"]["answer_switch_count"]
            for cell in analysis["cells"]
        )
        content_switches = sum(
            cell["scenario_summaries"]["donor_same_position_candidate"]["answer_switch_count"]
            for cell in analysis["cells"]
        )
        self.assertEqual(context_switches, 46)
        self.assertEqual(content_switches, 71)
        for cell in analysis["cells"]:
            self.assertTrue(cell["physical_static_same_literal_context_transport_gate_pass"])
            self.assertTrue(cell["candidate_depth_specific"])

    def test_public_atlas_mirrors_preserve_positive_and_negative_results(self) -> None:
        stages = []
        for root in ATLAS_ROOTS:
            manifest = read_json(root / "manifest.json")
            progress = read_json(root / "progress.json")
            stage = read_json(root / "phase396_binding_separation_stage_summary.json")
            stages.append(stage)
            self.assertEqual(manifest["last_phase"], "Phase399-MultiPositionDynamicBindingStage")
            self.assertEqual(progress["last_phase"], "Phase399-MultiPositionDynamicBindingStage")
            self.assertIn("phase397", manifest)
            self.assertIn("phase398", manifest)
            self.assertIn("phase399", manifest)
            self.assertIn("factor_separated_binding_stage", progress)
            self.assertEqual(stage["results"]["phase396_same_literal_answer_switches"], 46)
            self.assertEqual(stage["results"]["phase396_same_position_content_switches"], 71)
            self.assertEqual(stage["results"]["phase395_crosssurface_shared_state_count"], 0)
            self.assertFalse(stage["authorization"]["show_specific_neuron_path"])
        self.assertEqual(stages[0], stages[1])

    def test_3d_nodes_are_aggregate_state_anchors_not_neurons(self) -> None:
        for root in NEURON_ROOTS:
            manifest = read_json(root / "manifest.json")
            self.assertGreaterEqual(manifest["phase"], 396)
            self.assertEqual(manifest["phase396_audit"]["new_aggregate_state_anchor_count"], 6)
            self.assertEqual(manifest["phase396_audit"]["new_neuron_path_nodes_promoted"], 0)
            for model in MODELS:
                partition = read_json(root / f"partitions/content_knowledge/{model}.json")
                anchors = [node for node in partition["nodes"] if node.get("phase396_tested")]
                self.assertEqual(len(anchors), 2)
                self.assertTrue(all(node["node_type"] == "aggregate_token_state_anchor" for node in anchors))
                self.assertTrue(all(node["unit_kind"] == "token_state_aggregate" for node in anchors))
                self.assertTrue(all(node["is_real_unit"] is False for node in anchors))
                self.assertTrue(all(node["single_neuron_claim"] is False for node in anchors))
            checksums = read_json(root / "checksums.json")
            entries = {row["path"]: row["sha256"] for row in checksums["files"]}
            latest = "phase396_binding_separation_stage_summary.json"
            self.assertEqual(entries[latest], sha256(root / latest))
            phase397 = "phase397_factor_separated_binding_stage_summary.json"
            self.assertEqual(entries[phase397], sha256(root / phase397))


if __name__ == "__main__":
    unittest.main()
