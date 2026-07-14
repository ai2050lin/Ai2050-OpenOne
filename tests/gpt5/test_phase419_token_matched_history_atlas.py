from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase419_token_matched_history_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase419_token_matched_history_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


class Phase419AtlasTest(unittest.TestCase):
    def test_denominator(self) -> None:
        result = read_json(OUT / "phase419_denominator_qualification.json")
        self.assertTrue(result["valid"])
        self.assertEqual(result["qualified_cross_model_semantic_case_count"], 33)
        self.assertEqual(result["condition_count"], 396)
        self.assertEqual(result["exact_prompt_token_count_pair_count"], 198)

    def test_model_outputs(self) -> None:
        for model in MODELS:
            root = OUT / "models" / model
            result = read_json(root / "phase419_trace_complete.json")
            self.assertTrue(result["all_conditions_pass"])
            self.assertEqual(result["condition_count"], 132)
            self.assertEqual(result["exact_prompt_token_count_pair_count"], 66)
            self.assertEqual(count_jsonl(root / "phase419_vector_contrast_rows.jsonl"), 1485)
            self.assertEqual(count_jsonl(root / "phase419_direction_consistency_rows.jsonl"), 360)

    def test_global_claim_boundary(self) -> None:
        summary = read_json(OUT / "phase419_global_summary.json")
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["condition_pass_count"], 396)
        self.assertEqual(summary["exact_prompt_token_count_pair_count"], 198)
        self.assertEqual(summary["strict_mechanism_closure_count"], 0)
        self.assertFalse(summary["authorization"]["claim_history_identity_mechanism"])
        self.assertFalse(summary["authorization"]["run_single_neuron_scan_from_phase419_alone"])

    def test_public_graphs(self) -> None:
        manifest = read_json(PUBLIC / "manifest.json")
        self.assertEqual(manifest["schema_version"], "phase419_token_matched_history_atlas_manifest.v1")
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            graph = read_json(PUBLIC / item["filename"])
            self.assertEqual(graph["schema_version"], "atlas_graph_v1")
            self.assertEqual(len(graph["graph"]["nodes"]), 180)
            self.assertTrue(all(not node["causal"] for node in graph["graph"]["nodes"]))
            self.assertTrue(all(not edge["causal"] for edge in graph["graph"]["edges"]))

    def test_source_registry(self) -> None:
        registry = read_json(ROOT / "frontend/public/vis_data/source_registry.json")
        matches = [item for item in registry["sources"] if item["id"] == "gpt5_phase419_token_matched_history_atlas"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["payload_adapter"], "atlas_graph")


if __name__ == "__main__":
    unittest.main()
