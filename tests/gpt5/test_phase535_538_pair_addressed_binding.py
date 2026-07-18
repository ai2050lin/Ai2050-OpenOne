#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result"
ATLAS = ROOT / "frontend/public/vis_data/phase537_pair_addressed_behavior_atlas"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Phase535To538PairAddressedBindingTest(unittest.TestCase):
    def test_pair_address_protocol_flips_labels_without_moving_slots(self):
        audit = read_json(
            RESULT
            / "phase535_pair_addressed_binding_protocol"
            / "phase535_static_audit.json"
        )
        self.assertEqual(audit["status"], "static_pass_no_model_run")
        self.assertTrue(audit["entity_pool_disjoint_pass"])
        self.assertTrue(audit["relation_holdout_design_pass"])
        self.assertFalse(audit["model_run"])
        self.assertFalse(audit["sealed_split_read_by_downstream"])
        for report in audit["splits"].values():
            self.assertTrue(report["row_count_pass"])
            self.assertTrue(report["sixteen_way_group_pass"])
            self.assertTrue(report["world_surface_four_way_pass"])
            self.assertTrue(report["pair_status_flip_pass"])
            self.assertTrue(report["pair_ledger_world_identity_pass"])
            self.assertTrue(report["slot_label_balance_pass"])
            self.assertTrue(report["query_section_separated_from_world_prefix_pass"])

    def test_all_models_ran_in_order_and_none_was_authorized(self):
        authorization = read_json(
            RESULT
            / "phase536_pair_addressed_binding_behavior"
            / "phase536_physical_authorization.json"
        )
        self.assertEqual(
            authorization["models_in_required_order"],
            ["qwen3", "glm4", "deepseek7b"],
        )
        self.assertEqual(authorization["physical_authorized_models"], [])
        self.assertFalse(authorization["sealed_split_read"])
        for model in ("qwen3", "glm4", "deepseek7b"):
            summary = read_json(
                RESULT
                / "phase536_pair_addressed_binding_behavior"
                / f"phase536_{model}_behavior_summary.json"
            )
            self.assertEqual(summary["row_count"], 3840)
            self.assertTrue(summary["cuda_used"])
            self.assertFalse(summary["physical_authorized"])
            self.assertFalse(summary["sealed_split_read"])

    def test_strongest_qwen_split_does_not_override_all_split_gate(self):
        qwen = read_json(
            RESULT
            / "phase536_pair_addressed_binding_behavior"
            / "phase536_qwen3_behavior_summary.json"
        )
        self.assertFalse(qwen["split_reports"]["discovery"]["gate_pass"])
        self.assertFalse(qwen["split_reports"]["entity_prediction"]["gate_pass"])
        self.assertTrue(qwen["split_reports"]["relation_prediction"]["gate_pass"])
        self.assertFalse(qwen["physical_authorized"])
        self.assertEqual(qwen["split_reports"]["discovery"]["pair_flip_exact"]["count"], 276)
        self.assertEqual(qwen["split_reports"]["relation_prediction"]["pair_flip_exact"]["count"], 696)

    def test_diagnostics_preserve_polarity_bias_and_physical_stop(self):
        diagnostics = read_json(
            RESULT
            / "phase537_pair_addressed_behavior_diagnostics"
            / "phase537_pair_addressed_behavior_diagnostics.json"
        )
        self.assertTrue(diagnostics["stage_findings"]["truth_polarity_bias_present"])
        self.assertFalse(diagnostics["stage_findings"]["physical_collection_run"])
        self.assertFalse(diagnostics["stage_findings"]["prediction_split_hidden_states_read"])
        self.assertEqual(diagnostics["stage_findings"]["pipeline_permutations_run"], 0)
        self.assertFalse(diagnostics["stage_findings"]["sealed_split_read"])
        glm_outcomes = diagnostics["model_reports"]["glm4"]["splits"]["discovery"]["pair_flip"]["outcome_counts"]
        self.assertEqual(glm_outcomes["both_correct"], 152)
        self.assertEqual(glm_outcomes["true_only"], 232)

    def test_stage_audit_and_progress_do_not_promote_behavior(self):
        audit = read_json(
            RESULT
            / "phase538_pair_addressed_binding_stage_audit"
            / "phase538_pair_addressed_binding_stage_audit.json"
        )
        self.assertEqual(audit["status"], "complete_stopped_before_physical_collection")
        self.assertEqual(audit["stage_findings"]["physical_authorized_models"], [])
        self.assertFalse(audit["stage_findings"]["hidden_state_collection_run"])
        self.assertEqual(audit["stage_findings"]["pipeline_permutation_replicates_run"], 0)
        self.assertFalse(audit["evidence_boundary"]["physical"])
        self.assertEqual(audit["progress"]["strict_closed_mechanisms"], 0)
        self.assertEqual(audit["progress"]["global_physical_atlas_percent"], 31)
        self.assertEqual(audit["progress"]["overall_research_percent"], 26)

    def test_atlas_and_registry_preserve_behavior_only_scope(self):
        manifest = read_json(ATLAS / "manifest.json")
        self.assertEqual(
            manifest["schema_version"],
            "phase537_pair_addressed_behavior_atlas_manifest.v1",
        )
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            payload = read_json(ATLAS / item["path"])
            graph = payload["graph"]
            self.assertTrue(all(not node["physical"] for node in graph["nodes"]))
            self.assertTrue(all(not node["causal"] for node in graph["nodes"]))
            self.assertTrue(all(not node["compute_edge"] for node in graph["nodes"]))
            self.assertTrue(all(not node["single_neuron"] for node in graph["nodes"]))
            self.assertTrue(all(not edge["causal"] for edge in graph["edges"]))
            self.assertTrue(all(not edge["compute_edge"] for edge in graph["edges"]))
            self.assertFalse(graph["meta"]["hidden_state_collection_run"])
        registry = read_json(ROOT / "frontend/public/vis_data/source_registry.json")
        sources = {source["id"]: source for source in registry["sources"]}
        self.assertIn("gpt5_phase537_pair_addressed_behavior_atlas", sources)
        summary = read_json(
            ROOT
            / "frontend/public/vis_data/pattern_family_atlas/v2"
            / "phase537_pair_addressed_behavior_summary.json"
        )
        self.assertEqual(summary["physical_authorized_models"], [])
        self.assertEqual(summary["pair_binding_models"], [])
        self.assertFalse(summary["hidden_state_collection_run"])
        self.assertEqual(summary["pipeline_permutation_replicates_run"], 0)


if __name__ == "__main__":
    unittest.main()
