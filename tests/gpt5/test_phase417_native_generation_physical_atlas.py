#!/usr/bin/env python3
"""Evidence-contract tests for the Phase417 native-generation atlas."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase417_native_generation_physical_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase417_native_generation_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class Phase417NativeGenerationPhysicalAtlasTest(unittest.TestCase):
    def test_same_native_contract_is_zero_difference_for_all_cases(self) -> None:
        rows = []
        for model in MODELS:
            model_rows = read_jsonl(
                OUT / "models" / model / "phase417_native_generation_case_rows.jsonl"
            )
            self.assertEqual(len(model_rows), 55)
            rows.extend(model_rows)
        self.assertEqual(len(rows), 165)
        self.assertTrue(all(row["native_generation_case_pass"] for row in rows))
        self.assertTrue(all(row["comparison"]["token_exact"] for row in rows))
        self.assertTrue(all(row["comparison"]["score_max_abs"] == 0.0 for row in rows))
        self.assertTrue(all(row["comparison"]["score_js"] == 0.0 for row in rows))
        self.assertTrue(all(row["component_ledger_max_relative_error"] == 0.0 for row in rows))

    def test_global_denominators_and_right_censoring_are_preserved(self) -> None:
        summary = read_json(OUT / "phase417_global_summary.json")
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["case_count"], 165)
        self.assertEqual(summary["native_generation_qualification_pass_count"], 165)
        self.assertEqual(summary["same_contract_zero_score_difference_case_count"], 165)
        self.assertEqual(summary["physical_row_count"], 100300)
        self.assertEqual(summary["native_call_count"], 575)
        self.assertEqual(summary["prompt_prefill_call_count"], 165)
        self.assertEqual(summary["cached_incremental_call_count"], 410)
        self.assertEqual(summary["right_censored_case_count"], 46)
        self.assertEqual(summary["region_cell_count"], 360)
        self.assertEqual(summary["prompt_to_cached_transition_measurement_count"], 71700)
        self.assertTrue(summary["authorization"]["publish_native_generation_physical_atlas"])
        self.assertFalse(summary["authorization"]["publish_functional_generation_mechanism"])
        self.assertFalse(summary["authorization"]["run_causal_intervention"])
        self.assertFalse(summary["authorization"]["run_neuron_scan"])

    def test_transition_rows_do_not_claim_isolated_history(self) -> None:
        rows = read_jsonl(OUT / "phase417_prompt_to_cached_transition_rows.jsonl")
        self.assertEqual(len(rows), 180)
        self.assertTrue(all(row["contrast_mixes_generated_token_identity_and_cache_history"] for row in rows))
        self.assertTrue(all(row["causal"] is False for row in rows))
        layout = read_json(OUT / "phase417_cross_model_layout.json")["rows"]
        self.assertEqual(len(layout), 40)
        self.assertTrue(all(row["dominant_depth"] == "late" for row in layout))

    def test_public_graphs_are_generation_time_but_noncausal(self) -> None:
        manifest = read_json(PUBLIC / "manifest.json")
        self.assertEqual(manifest["schema_version"], "phase417_native_generation_atlas_manifest.v1")
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            graph = read_json(PUBLIC / item["filename"])
            self.assertEqual(graph["schema_version"], "atlas_graph_v1")
            self.assertEqual(len(graph["graph"]["nodes"]), 120)
            self.assertEqual(len(graph["graph"]["edges"]), 140)
            self.assertTrue(all(node["generation_time"] for node in graph["graph"]["nodes"]))
            self.assertTrue(all(node["observer_id"] is None for node in graph["graph"]["nodes"]))
            self.assertTrue(all(node["causal"] is False for node in graph["graph"]["nodes"]))
            self.assertTrue(all(edge["causal"] is False for edge in graph["graph"]["edges"]))


if __name__ == "__main__":
    unittest.main()
