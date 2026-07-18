#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result"
ATLAS = ROOT / "frontend/public/vis_data/phase533_world_geometry_role_binding_atlas"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Phase526To534WorldGeometryTest(unittest.TestCase):
    def test_protocols_are_balanced_and_sealed(self):
        audits = [
            read_json(
                RESULT
                / "phase526_role_normalized_world_geometry_protocol"
                / "phase526_static_audit.json"
            ),
            read_json(
                RESULT
                / "phase528_relation_contract_factorial_protocol"
                / "phase528_static_audit.json"
            ),
            read_json(
                RESULT
                / "phase530_glm4_fresh_world_geometry_protocol"
                / "phase530_static_audit.json"
            ),
        ]
        for audit in audits:
            self.assertEqual(audit["status"], "static_pass_no_model_run")
            self.assertFalse(audit["model_run"])
            self.assertFalse(audit["sealed_split_read_by_downstream"])
            for split in audit["splits"].values():
                self.assertTrue(split["row_count_pass"])
                self.assertTrue(split["four_way_group_pass"])

    def test_initial_gate_failure_and_factorial_diagnosis_are_distinct(self):
        initial = read_json(
            RESULT
            / "phase527_world_geometry_behavior_qualification"
            / "phase527_physical_authorization.json"
        )
        factorial = read_json(
            RESULT
            / "phase529_relation_contract_factorial_behavior"
            / "phase529_factorial_authorization.json"
        )
        self.assertEqual(initial["physical_authorized_models"], [])
        self.assertEqual(factorial["models_with_any_confirmed_condition"], ["glm4"])
        self.assertEqual(factorial["shared_confirmed_conditions"], [])
        self.assertEqual(len(factorial["confirmed_conditions_by_model"]["glm4"]), 5)
        self.assertEqual(factorial["confirmed_conditions_by_model"]["qwen3"], [])
        self.assertEqual(factorial["confirmed_conditions_by_model"]["deepseek7b"], [])

    def test_fresh_glm_behavior_authorizes_measurement_only(self):
        authorization = read_json(
            RESULT
            / "phase531_glm4_fresh_world_geometry_behavior"
            / "phase531_fresh_physical_authorization.json"
        )
        summary = read_json(
            RESULT
            / "phase531_glm4_fresh_world_geometry_behavior"
            / "phase531_glm4_fresh_behavior_summary.json"
        )
        self.assertEqual(authorization["fresh_physical_authorized_models"], ["glm4"])
        for split in ("discovery", "entity_prediction", "relation_prediction"):
            self.assertTrue(summary["split_reports"][split]["gate_pass"])
        self.assertEqual(summary["split_reports"]["discovery"]["overall"]["count"], 384)
        self.assertEqual(summary["split_reports"]["relation_prediction"]["overall"]["count"], 768)
        self.assertFalse(summary["sealed_split_read"])

    def test_role_polarity_is_not_pair_binding(self):
        geometry = read_json(
            RESULT
            / "phase532_glm4_role_normalized_world_geometry"
            / "phase532_glm4_world_geometry_summary.json"
        )
        ledger = read_json(
            RESULT
            / "phase532_glm4_role_normalized_world_geometry"
            / "phase532_glm4_frozen_world_geometry_ledger.json"
        )
        audit = read_json(
            RESULT
            / "phase533_world_geometry_role_binding_audit"
            / "phase533_world_geometry_role_binding_audit.json"
        )
        self.assertEqual(geometry["status"], "stopped_no_discovery_platform")
        self.assertEqual(ledger["platform_count"], 0)
        self.assertFalse(ledger["prediction_splits_read"])
        self.assertFalse(ledger["sealed_split_read"])
        self.assertTrue(all(cell["orientation_accuracy"] == 1.0 for cell in audit["selected_cells"]))
        self.assertGreater(
            audit["mean_source_to_target_disconnected_false_positive"],
            0.99,
        )
        self.assertFalse(audit["evidence_boundary"]["edge_binding_confirmed"])
        self.assertFalse(audit["evidence_boundary"]["causal"])

    def test_stage_audit_keeps_progress_and_stop_accounting_honest(self):
        audit = read_json(
            RESULT
            / "phase534_world_geometry_stage_audit"
            / "phase534_world_geometry_stage_audit.json"
        )
        self.assertEqual(audit["status"], "complete_stopped_before_prediction")
        self.assertEqual(audit["physical_findings"]["pair_specific_edge_binding_models"], [])
        self.assertEqual(audit["physical_findings"]["world_relation_platform_models"], [])
        self.assertEqual(audit["stop_accounting"]["permutation_replicates_run"], 0)
        self.assertFalse(audit["stop_accounting"]["entity_prediction_split_read"])
        self.assertFalse(audit["stop_accounting"]["relation_prediction_split_read"])
        self.assertFalse(audit["stop_accounting"]["sealed_split_read"])
        self.assertEqual(audit["progress"]["strict_closed_mechanisms"], 0)
        self.assertEqual(audit["progress"]["mechanism_denominator"], 72)
        self.assertEqual(audit["progress"]["global_physical_atlas_percent"], 31)
        self.assertEqual(audit["progress"]["overall_research_percent"], 26)

    def test_atlas_preserves_observer_and_negative_evidence(self):
        manifest = read_json(ATLAS / "manifest.json")
        self.assertEqual(
            manifest["schema_version"],
            "phase533_world_geometry_role_binding_atlas_manifest.v1",
        )
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            payload = read_json(ATLAS / item["path"])
            graph = payload["graph"]
            self.assertTrue(all(not node["causal"] for node in graph["nodes"]))
            self.assertTrue(all(not node["compute_edge"] for node in graph["nodes"]))
            self.assertTrue(all(not node["single_neuron"] for node in graph["nodes"]))
            self.assertTrue(all(not node["pipeline_sealed"] for node in graph["nodes"]))
            self.assertTrue(all(not edge["causal"] for edge in graph["edges"]))
            self.assertTrue(all(not edge["compute_edge"] for edge in graph["edges"]))
        glm = read_json(ATLAS / "phase533_glm4_world_geometry_role_binding.json")
        node_types = {node["type"] for node in glm["graph"]["nodes"]}
        self.assertIn("scaffold_conditioned_node_role_signature", node_types)
        self.assertIn("binding_confound", node_types)
        self.assertIn("pair_specific_binding_missing", node_types)
        self.assertEqual(glm["graph"]["meta"]["permutation_replicates_run"], 0)

    def test_source_registry_and_v2_summary_match_stage_boundary(self):
        registry = read_json(ROOT / "frontend/public/vis_data/source_registry.json")
        sources = {item["id"]: item for item in registry["sources"]}
        source = sources["gpt5_phase533_world_geometry_role_binding_atlas"]
        self.assertEqual(source["payload_adapter"], "atlas_graph")
        self.assertEqual(
            source["manifest_path"],
            "/vis_data/phase533_world_geometry_role_binding_atlas/manifest.json",
        )
        summary = read_json(
            ROOT
            / "frontend/public/vis_data/pattern_family_atlas/v2"
            / "phase533_world_geometry_role_binding_summary.json"
        )
        self.assertEqual(summary["node_role_polarity_models"], ["glm4"])
        self.assertEqual(summary["pair_specific_edge_binding_models"], [])
        self.assertEqual(summary["permutation_replicates_run"], 0)
        self.assertEqual(summary["strict_closed_mechanisms"], 0)
        self.assertFalse(summary["sealed_split_read"])


if __name__ == "__main__":
    unittest.main()
