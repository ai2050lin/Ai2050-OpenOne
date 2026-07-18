#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result"
ATLAS = ROOT / "frontend/public/vis_data/phase524_world_query_platform_atlas"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Phase518To525AuditTest(unittest.TestCase):
    def test_behavior_authorization_is_split_by_contract(self):
        discovery = read_json(
            RESULT / "phase520_behavior_authorization/phase520_behavior_authorization.json"
        )
        confirmation = read_json(
            RESULT
            / "phase522_semantic_event_confirmation/phase522_physical_authorization.json"
        )
        self.assertEqual(discovery["relation_models"], ["qwen3", "glm4"])
        self.assertEqual(confirmation["relation_models"], ["qwen3", "glm4"])
        self.assertEqual(discovery["binding_models"], [])
        self.assertEqual(confirmation["binding_models"], [])
        self.assertTrue(confirmation["physical_authorized"])
        for model in ("qwen3", "glm4"):
            report = confirmation["model_reports"][model]
            self.assertFalse(report["strict_whole_response_relation_gate_pass"])
            self.assertTrue(report["relation_first_event"]["first_event_gate_pass"])
            self.assertEqual(
                report["relation_first_event"]["first_event_four_way"]["count"],
                192,
            )

    def test_platform_ledgers_freeze_before_prediction(self):
        for model in ("qwen3", "glm4"):
            ledger = read_json(
                RESULT
                / "phase523_world_query_platform_physical"
                / f"phase523_{model}_frozen_platform_ledger.json"
            )
            self.assertEqual(ledger["status"], "frozen_before_prediction_read")
            self.assertFalse(ledger["prediction_split_read"])
            self.assertFalse(ledger["sealed_split_read"])
            self.assertEqual(ledger["tasks"]["world_topology"]["platform_count"], 0)

    def test_familywise_result_is_narrowly_scoped(self):
        qwen = read_json(
            RESULT
            / "phase524_platform_permutation_audit"
            / "phase524_qwen3_platform_permutation_summary.json"
        )
        glm = read_json(
            RESULT
            / "phase524_platform_permutation_audit"
            / "phase524_glm4_platform_permutation_summary.json"
        )
        self.assertTrue(qwen["observational_platform_confirmed"])
        self.assertFalse(glm["observational_platform_confirmed"])
        self.assertTrue(qwen["tasks"]["query_evaluation"]["familywise_significant"])
        self.assertFalse(qwen["tasks"]["world_topology"]["familywise_significant"])
        self.assertEqual(qwen["tasks"]["query_evaluation"]["null_replicate_count"], 128)
        self.assertEqual(qwen["tasks"]["query_evaluation"]["permutation_p_value"], 1 / 129)
        self.assertFalse(qwen["causal"])
        self.assertFalse(qwen["sealed_split_read"])

    def test_atlas_preserves_negative_and_noncausal_evidence(self):
        manifest = read_json(ATLAS / "manifest.json")
        self.assertEqual(
            manifest["schema_version"],
            "phase524_world_query_platform_atlas_manifest.v1",
        )
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            payload = read_json(ATLAS / item["path"])
            graph = payload["graph"]
            self.assertTrue(any(node["type"] == "world_platform_missing" for node in graph["nodes"]))
            self.assertTrue(all(not node["causal"] for node in graph["nodes"]))
            self.assertTrue(all(not edge["causal"] for edge in graph["edges"]))
            self.assertTrue(all(not edge["compute_edge"] for edge in graph["edges"]))
        qwen = read_json(ATLAS / "phase524_qwen3_world_query_platform.json")
        qwen_layers = [
            node for node in qwen["graph"]["nodes"]
            if node["type"] == "query_evaluation_platform_layer"
            and node["evidence_level"] == "familywise_controlled_observational_query_platform"
        ]
        self.assertGreaterEqual(len(qwen_layers), 11)
        self.assertTrue(all(node["physical"] and node["predictive"] for node in qwen_layers))

    def test_source_registry_and_progress_are_consistent(self):
        registry = read_json(ROOT / "frontend/public/vis_data/source_registry.json")
        sources = {item["id"]: item for item in registry["sources"]}
        source = sources["gpt5_phase524_world_query_platform_atlas"]
        self.assertEqual(source["payload_adapter"], "atlas_graph")
        self.assertEqual(
            source["manifest_path"],
            "/vis_data/phase524_world_query_platform_atlas/manifest.json",
        )
        progress = read_json(
            ROOT
            / "frontend/public/vis_data/pattern_family_atlas/v2"
            / "phase524_world_query_platform_summary.json"
        )
        self.assertEqual(progress["strict_closed_mechanisms"], 0)
        self.assertEqual(progress["mechanism_denominator"], 72)
        self.assertEqual(progress["overall_research_percent"], 26)
        self.assertFalse(progress["sealed_split_read"])


if __name__ == "__main__":
    unittest.main()
