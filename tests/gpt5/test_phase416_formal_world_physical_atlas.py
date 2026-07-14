#!/usr/bin/env python3
"""Evidence-contract tests for the Phase416 formal prefill physical atlas."""

from __future__ import annotations

import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase416_formal_world_physical_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase416_formal_prefill_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class Phase416FormalWorldPhysicalAtlasTest(unittest.TestCase):
    def test_frozen_case_bank_is_balanced_and_cross_model_aligned(self) -> None:
        rows = read_jsonl(OUT / "phase416_registered_cases.jsonl")
        self.assertEqual(len(rows), 165)
        self.assertEqual(Counter(row["model"] for row in rows), Counter({model: 55 for model in MODELS}))
        by_model: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            by_model[row["model"]].add(row["semantic_case_id"])
            self.assertTrue(row["formal_semantics_executable"])
            self.assertFalse(row["causal_intervention_authorized"])
            self.assertFalse(row["single_neuron_scan_authorized"])
        self.assertEqual(by_model["qwen3"], by_model["glm4"])
        self.assertEqual(by_model["qwen3"], by_model["deepseek7b"])

    def test_instrument_domains_are_not_collapsed(self) -> None:
        qualification = read_json(OUT / "phase416_instrument_domain_qualification.json")
        results = qualification["results"]
        authorization = qualification["authorization"]
        self.assertTrue(qualification["valid"])
        self.assertEqual(results["prefill_collector_pass_count"], 165)
        self.assertEqual(results["prefill_qualified_model_count"], 3)
        self.assertEqual(results["incremental_cache_pass_count"], 45)
        self.assertEqual(results["incremental_cache_qualified_model_count"], 0)
        self.assertEqual(results["greedy_generation_pass_count"], 61)
        self.assertEqual(results["greedy_generation_qualified_model_count"], 0)
        self.assertTrue(authorization["collect_observer_free_prefill_physical_atlas"])
        self.assertFalse(authorization["collect_generation_time_physical_atlas"])
        self.assertFalse(authorization["run_causal_intervention"])
        self.assertFalse(authorization["run_neuron_scan"])

    def test_fixed_format_physical_collection_is_complete(self) -> None:
        summary = read_json(OUT / "phase416_global_summary.json")
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["physical_case_count"], 165)
        self.assertEqual(summary["physical_row_count"], 204160)
        self.assertEqual(summary["lossless_anchor_vector_count"], 6240)
        self.assertEqual(summary["region_cell_count"], 540)
        self.assertEqual(summary["model_physical_case_count"], {model: 55 for model in MODELS})
        self.assertTrue(summary["authorization"]["publish_descriptive_prefill_atlas"])
        self.assertFalse(summary["authorization"]["publish_generation_time_atlas"])
        self.assertFalse(summary["authorization"]["run_causal_intervention"])

    def test_public_atlas_preserves_noncausal_boundary(self) -> None:
        manifest = read_json(PUBLIC / "manifest.json")
        self.assertEqual(manifest["schema_version"], "phase416_prefill_physical_atlas_manifest.v1")
        self.assertEqual(len(manifest["items"]), 3)
        self.assertEqual({item["model"] for item in manifest["items"]}, set(MODELS))
        for item in manifest["items"]:
            payload = read_json(PUBLIC / item["filename"])
            self.assertEqual(payload["schema_version"], "atlas_graph_v1")
            self.assertGreater(len(payload["graph"]["nodes"]), 0)
            self.assertGreater(len(payload["graph"]["edges"]), 0)
            self.assertTrue(all(edge["causal"] is False for edge in payload["graph"]["edges"]))
            self.assertTrue(all(node["observer_id"] is None for node in payload["graph"]["nodes"]))
            self.assertTrue(all(node["generation_time"] is False for node in payload["graph"]["nodes"]))


if __name__ == "__main__":
    unittest.main()
