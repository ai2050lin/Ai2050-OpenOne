from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase418_interface_history_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase418_interface_history_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


class Phase418AtlasTest(unittest.TestCase):
    def test_registered_denominator_is_frozen(self) -> None:
        qualification = read_json(OUT / "phase418_denominator_qualification.json")
        self.assertTrue(qualification["valid"])
        self.assertEqual(qualification["case_count"], 1200)
        self.assertEqual(qualification["semantic_case_count"], 40)
        self.assertEqual(count_jsonl(OUT / "phase418_registered_conditions.jsonl"), 1200)

    def test_model_outputs_are_complete(self) -> None:
        for model in MODELS:
            root = OUT / "models" / model
            complete = read_json(root / "phase418_trace_complete.json")
            self.assertTrue(complete["all_conditions_pass"])
            self.assertEqual(complete["condition_count"], 400)
            self.assertEqual(complete["terminal_suffix_alignment_pass_count"], 400)
            self.assertEqual(count_jsonl(root / "phase418_condition_rows.jsonl"), 400)
            self.assertEqual(count_jsonl(root / "phase418_prefill_physical_rows.jsonl"), complete["physical_row_count"])
            self.assertEqual(count_jsonl(root / "phase418_vector_contrast_rows.jsonl"), 10200)
            self.assertGreater(complete["lossless_anchor_vector_count"], 0)

    def test_global_summary_keeps_claim_boundary(self) -> None:
        summary = read_json(OUT / "phase418_global_summary.json")
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["condition_count"], 1200)
        self.assertEqual(summary["condition_pass_count"], 1200)
        self.assertEqual(summary["strict_mechanism_closure_count"], 0)
        self.assertFalse(summary["authorization"]["publish_interface_history_mechanism"])
        self.assertFalse(summary["authorization"]["run_single_neuron_scan_from_phase418_alone"])

    def test_public_graphs_are_noncausal(self) -> None:
        manifest = read_json(PUBLIC / "manifest.json")
        self.assertEqual(manifest["schema_version"], "phase418_interface_history_atlas_manifest.v1")
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            graph = read_json(PUBLIC / item["filename"])
            self.assertEqual(graph["schema_version"], "atlas_graph_v1")
            self.assertGreater(len(graph["graph"]["nodes"]), 0)
            self.assertGreater(len(graph["graph"]["edges"]), 0)
            self.assertTrue(all(not node["causal"] for node in graph["graph"]["nodes"]))
            self.assertTrue(all(not edge["causal"] for edge in graph["graph"]["edges"]))

    def test_source_registry_contains_phase418(self) -> None:
        registry = read_json(ROOT / "frontend/public/vis_data/source_registry.json")
        matches = [source for source in registry["sources"] if source["id"] == "gpt5_phase418_interface_history_atlas"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["payload_adapter"], "atlas_graph")


if __name__ == "__main__":
    unittest.main()
