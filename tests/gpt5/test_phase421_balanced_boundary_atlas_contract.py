from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase421_balanced_boundary_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase421_balanced_boundary_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class Phase421BalancedBoundaryAtlasContractTest(unittest.TestCase):
    def test_global_evidence_boundary(self) -> None:
        summary = read_json(OUT / "phase421_global_summary.json")
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["registered_group_count"], 96)
        self.assertEqual(summary["registered_behavior_condition_count"], 31_104)
        self.assertEqual(summary["measured_behavior_condition_count"], 31_104)
        self.assertEqual(summary["measured_generation_condition_count"], 1_728)
        self.assertEqual(summary["development_physical_condition_count"], 4_032)
        self.assertEqual(summary["physical_holdout_condition_count"], 0)
        self.assertEqual(summary["selected_source_coordinate_count"], 96)
        self.assertEqual(summary["fixed_path_feature_row_count"], 2_688)
        self.assertEqual(summary["strict_mechanism_closure_count"], 0)
        self.assertEqual(summary["strict_mechanism_denominator"], 72)
        self.assertTrue(summary["gates"]["balanced_behavior_boundary"])
        self.assertTrue(summary["gates"]["source_write_replication"])
        self.assertTrue(summary["gates"]["structural_history_current_coordinate_separation"])
        self.assertFalse(summary["gates"]["incremental_continuous_prediction"])
        self.assertFalse(summary["gates"]["physical_holdout_authorized"])
        self.assertFalse(summary["gates"]["causal_intervention_authorized"])
        self.assertFalse(summary["gates"]["single_neuron_scan_authorized"])

    def test_model_measurement_contracts(self) -> None:
        expected_layers = {"qwen3": 48_384, "glm4": 53_760, "deepseek7b": 37_632}
        for model in MODELS:
            model_root = OUT / "models" / model
            behavior = read_json(model_root / "phase421_behavior_complete.json")
            physical = read_json(model_root / "phase421_physical_complete.json")
            self.assertEqual(behavior["margin_condition_count"], 10_368)
            self.assertEqual(behavior["generation_panel_count"], 576)
            self.assertTrue(behavior["all_behavior_rows_pass"])
            self.assertEqual(physical["physical_development_condition_count"], 1_344)
            self.assertEqual(physical["physical_holdout_condition_count"], 0)
            self.assertEqual(physical["physical_condition_layer_row_count"], expected_layers[model])
            self.assertEqual(physical["selected_coordinate_count"], 32)
            self.assertEqual(physical["fixed_path_feature_row_count"], 896)
            self.assertTrue(physical["all_development_rows_pass"])
            self.assertTrue(physical["physical_holdout_remains_sealed"])

    def test_analysis_row_denominators(self) -> None:
        source = read_jsonl(OUT / "phase421_source_replication_audit.jsonl")
        geometry = read_jsonl(OUT / "phase421_independent_geometry_summary.jsonl")
        prediction = read_jsonl(OUT / "phase421_incremental_prediction_audit.jsonl")
        self.assertEqual(len(source), 288)
        self.assertEqual(len(geometry), 54)
        self.assertEqual(len(prediction), 9)
        self.assertTrue(all(not row["causal"] for row in source + geometry + prediction))
        validation = [row for row in prediction if row["split"] != "discovery"]
        self.assertEqual(len(validation), 6)
        self.assertFalse(all(row["split_gate_pass"] for row in validation))
        authorization = read_json(OUT / "phase421_physical_holdout_authorization.json")
        self.assertFalse(authorization["physical_holdout_collection_authorized"])

    def test_public_manifest_and_graphs(self) -> None:
        manifest = read_json(PUBLIC / "manifest.json")
        self.assertEqual(
            manifest["schema_version"], "phase421_balanced_boundary_atlas_manifest.v1"
        )
        self.assertEqual(len(manifest["items"]), 12)
        ids = [item["id"] for item in manifest["items"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({item["model"] for item in manifest["items"]}, set(MODELS))
        for item in manifest["items"]:
            payload = read_json(PUBLIC / item["filename"])
            self.assertEqual(payload["schema_version"], "atlas_graph_v1")
            self.assertEqual(payload["model"], item["model"])
            self.assertGreater(len(payload["graph"]["nodes"]), 0)
            self.assertTrue(all(not node.get("causal", False) for node in payload["graph"]["nodes"]))
            self.assertTrue(all(not edge.get("causal", False) for edge in payload["graph"]["edges"]))
            self.assertGreater(len(payload["evidence_boundary"]), 0)

    def test_multi_route_registry_entry(self) -> None:
        registry = read_json(ROOT / "frontend/public/vis_data/source_registry.json")
        sources = {
            source["id"]: source for source in registry["sources"]
        }
        source = sources["gpt5_phase421_balanced_boundary_atlas"]
        self.assertEqual(source["route_id"], "gpt5")
        self.assertEqual(
            source["manifest_path"], "/vis_data/phase421_balanced_boundary_atlas/manifest.json"
        )
        self.assertEqual(source["models"], list(MODELS))


if __name__ == "__main__":
    unittest.main()
