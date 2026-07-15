from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase423_workspace_observer_qualification"
PUBLIC = ROOT / "frontend/public/vis_data/phase423_workspace_observer_qualification"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class Phase423WorkspaceObserverQualificationTest(unittest.TestCase):
    def test_protocol_denominators_are_frozen(self):
        protocol = read_json(OUT / "phase423_protocol.json")
        self.assertEqual(protocol["fit_contract"]["prompt_count"], 100)
        self.assertEqual(protocol["evaluation_contract"]["total_items"], 192)
        self.assertFalse(protocol["workspace_claim_allowed"])
        fit_rows = read_jsonl(OUT / "phase423_fit_prompts.jsonl")
        eval_rows = read_jsonl(OUT / "phase423_evaluation_items.jsonl")
        self.assertEqual(len(fit_rows), 100)
        self.assertEqual(len(eval_rows), 192)
        self.assertEqual({row["split"] for row in fit_rows}, {"fit_a", "fit_b"})

    def test_model_outputs_are_complete_and_observer_only(self):
        for model in MODELS:
            root = OUT / "models" / model
            fit = read_json(root / "phase423_fit_summary.json")
            evaluation = read_json(root / "phase423_observer_evaluation_summary.json")
            self.assertEqual(fit["n_prompts_merged"], 100)
            self.assertEqual(len(fit["source_layers"]), 9)
            self.assertEqual(evaluation["evaluation_item_count"], 192)
            self.assertTrue(evaluation["observer_only"])
            self.assertFalse(evaluation["workspace_claim_allowed"])
            self.assertFalse(evaluation["compute_edge_claim_allowed"])
            self.assertFalse(evaluation["causal_claim_allowed"])
            rows = read_jsonl(root / "phase423_observer_evaluation_rows.jsonl")
            eligible = next(row for row in rows if row["eligible_single_token"])
            self.assertEqual(len(eligible["source_layers"]), 9)
            self.assertEqual(len(eligible["jlens_ranks_by_layer"]), 9)
            self.assertEqual(len(eligible["logit_lens_ranks_by_layer"]), 9)

    def test_global_summary_never_promotes_observer_to_mechanism(self):
        summary = read_json(OUT / "phase423_global_summary.json")
        self.assertEqual(summary["strict_mechanism_closure_count"], 0)
        self.assertEqual(summary["strict_mechanism_denominator"], 72)
        self.assertFalse(summary["workspace_claim_allowed"])
        self.assertFalse(summary["compute_edge_claim_allowed"])
        self.assertFalse(summary["causal_claim_allowed"])
        for model in MODELS:
            audit = summary["models"][model]["posthoc_fixed_layer_holdout_audit"]
            self.assertEqual(audit["status"], "posthoc_not_preregistered")
            self.assertFalse(audit["changes_frozen_authorization"])

    def test_atlas_is_fixed_format_and_noncausal(self):
        manifest = read_json(PUBLIC / "manifest.json")
        self.assertEqual(manifest["schema_version"], "phase423_workspace_observer_manifest.v1")
        self.assertEqual(len(manifest["items"]), 6)
        for item in manifest["items"]:
            payload = read_json(PUBLIC / item["filename"])
            self.assertEqual(payload["schema_version"], "atlas_graph_v1")
            for node in payload["graph"]["nodes"]:
                self.assertTrue(node["observer"])
                self.assertFalse(node["compute_edge"])
                self.assertFalse(node["causal"])
            for edge in payload["graph"]["edges"]:
                self.assertTrue(edge["observer"])
                self.assertFalse(edge["compute_edge"])
                self.assertFalse(edge["causal"])

    def test_source_registry_contains_phase423(self):
        registry = read_json(ROOT / "frontend/public/vis_data/source_registry.json")
        source = next(
            item
            for item in registry["sources"]
            if item["id"] == "gpt5_phase423_workspace_observer_qualification"
        )
        self.assertEqual(source["route_id"], "gpt5")
        self.assertEqual(
            source["manifest_path"],
            "/vis_data/phase423_workspace_observer_qualification/manifest.json",
        )


if __name__ == "__main__":
    unittest.main()
