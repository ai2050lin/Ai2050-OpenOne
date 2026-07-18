from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase517_dual_contract_stage_audit"
PUBLIC = ROOT / "frontend/public/vis_data/phase517_relation_binding_decomposition_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Phase517DualContractStageAuditTest(unittest.TestCase):
    def test_frozen_stage_decision_and_denominators(self) -> None:
        audit = read_json(OUT / "phase517_dual_contract_stage_audit.json")
        self.assertEqual(audit["status"], "stage_complete_prediction_gate_stopped")
        self.assertEqual(audit["denominators"]["calibration_model_rows"], 2_112)
        self.assertEqual(audit["denominators"]["confirmation_model_rows"], 384)
        self.assertEqual(audit["denominators"]["joint_model_rows"], 0)
        self.assertEqual(audit["denominators"]["physical_fit_rows"], 192)
        self.assertEqual(audit["denominators"]["physical_prediction_rows"], 384)
        self.assertEqual(audit["denominators"]["strict_closed_mechanisms"], 0)
        self.assertEqual(audit["denominators"]["mechanism_denominator"], 72)
        self.assertEqual(audit["gates"]["relation_calibration_models"], ["glm4"])
        self.assertEqual(audit["gates"]["relation_confirmation_models"], ["glm4"])
        self.assertEqual(audit["gates"]["binding_calibration_models"], [])
        self.assertEqual(audit["gates"]["joint_confirmation_models"], [])
        self.assertFalse(audit["gates"]["shared_relation_physical"])
        self.assertFalse(audit["gates"]["glm4_relation_primary_prediction"])
        self.assertFalse(audit["stage_decision"]["same_stage_has_authorized_next_step"])

    def test_primary_failure_is_not_replaced_posthoc(self) -> None:
        audit = read_json(OUT / "phase517_dual_contract_stage_audit.json")
        reanalysis = audit["physical"]["reanalysis"]
        primary = reanalysis["pre_registered_primary"]
        self.assertTrue(reanalysis["primary_matches_phase516_summary"])
        self.assertEqual(primary["layer_with_embedding"], 10)
        self.assertEqual(primary["position_role"], "claim_entity_end")
        self.assertEqual(primary["metrics"]["overall"]["count"], 350)
        self.assertEqual(primary["metrics"]["overall"]["n"], 384)
        self.assertEqual(primary["metrics"]["four_way_pair"]["count"], 71)
        self.assertEqual(primary["metrics"]["four_way_pair"]["n"], 96)
        self.assertFalse(primary["passes_numeric_gate"])
        self.assertEqual(reanalysis["numeric_gate_passing_window_count"], 65)
        self.assertTrue(
            all(
                not row["pre_registered_primary"]
                for row in reanalysis["all_prediction_windows"]
                if row["passes_numeric_gate"]
            )
        )
        by_role = {
            row["position_role"]: row
            for row in reanalysis["numeric_gate_passing_windows_by_role"]
        }
        self.assertEqual(
            {role: row["count"] for role, row in by_role.items()},
            {
                "claim_entity_end": 15,
                "claim_relation_end": 15,
                "claim_end": 13,
                "prompt_end": 22,
            },
        )
        self.assertNotIn("target_evidence_end", by_role)
        self.assertNotIn("distractor_evidence_end", by_role)

    def test_control_and_evidence_boundaries(self) -> None:
        audit = read_json(OUT / "phase517_dual_contract_stage_audit.json")
        reanalysis = audit["physical"]["reanalysis"]
        self.assertFalse(
            reanalysis["phase516_unbalanced_random_control_audit"]["diagnostic"]
        )
        balanced = reanalysis["balanced_sign_controls"]
        self.assertEqual(balanced["seed_count"], 32)
        self.assertEqual(balanced["train_pair_count"], 24)
        self.assertEqual(balanced["flipped_pair_count_each_seed"], 12)
        boundary = audit["evidence_boundary"]
        self.assertTrue(boundary["hidden_state_collected"])
        self.assertTrue(boundary["model_specific_observational_trajectory"])
        self.assertFalse(boundary["shared_cross_model_physical_rule"])
        self.assertFalse(boundary["compute_transport_measured"])
        self.assertFalse(boundary["causal_intervention"])
        self.assertFalse(boundary["head_channel_neuron_scan"])
        self.assertFalse(boundary["sealed_split_read"])

    def test_atlas_preserves_observation_only_scope(self) -> None:
        manifest = read_json(PUBLIC / "manifest.json")
        self.assertEqual(
            manifest["schema_version"],
            "phase517_relation_binding_decomposition_atlas_manifest.v1",
        )
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            payload = read_json(PUBLIC / item["path"])
            nodes = payload["graph"]["nodes"]
            edges = payload["graph"]["edges"]
            self.assertGreater(len(nodes), 0)
            self.assertTrue(all(not node.get("causal", False) for node in nodes))
            self.assertTrue(all(not node.get("single_neuron", False) for node in nodes))
            self.assertTrue(all(not edge.get("compute_edge", False) for edge in edges))
            self.assertTrue(all(not edge.get("causal", False) for edge in edges))
            physical = [node for node in nodes if node.get("physical", False)]
            if item["model"] == "glm4":
                self.assertGreater(len(physical), 0)
                primary = [node for node in physical if node.get("pre_registered_primary")]
                self.assertEqual(len(primary), 1)
                self.assertFalse(primary[0]["gate_pass"])
                for node in physical:
                    if node.get("posthoc_descriptive"):
                        self.assertFalse(node["gate_pass"])
            else:
                self.assertEqual(physical, [])

    def test_registry_entry(self) -> None:
        registry = read_json(ROOT / "frontend/public/vis_data/source_registry.json")
        sources = {source["id"]: source for source in registry["sources"]}
        source = sources["gpt5_phase517_relation_binding_decomposition_atlas"]
        self.assertEqual(source["route_id"], "gpt5")
        self.assertEqual(source["models"], list(MODELS))
        self.assertEqual(
            source["manifest_path"],
            "/vis_data/phase517_relation_binding_decomposition_atlas/manifest.json",
        )


if __name__ == "__main__":
    unittest.main()
