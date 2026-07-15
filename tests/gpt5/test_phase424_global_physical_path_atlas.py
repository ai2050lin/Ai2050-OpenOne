#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase424_global_physical_path_atlas"
VIS = ROOT / "frontend/public/vis_data/phase424_global_physical_path_atlas"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase424ContractTest(unittest.TestCase):
    def test_frozen_denominator(self) -> None:
        protocol = read_json(OUT / "phase424_protocol.json")
        validation = protocol["validation"]
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["family_count"], 9)
        self.assertEqual(validation["mechanism_count"], 72)
        self.assertEqual(validation["pair_count"], 864)
        self.assertEqual(validation["condition_count"], 5184)
        self.assertEqual(validation["new_double_blind_pair_count"], 0)
        self.assertFalse(protocol["evidence_contract"]["causal_claim_allowed"])

    def test_model_collections(self) -> None:
        for model in ("qwen3", "glm4", "deepseek7b"):
            complete = read_json(
                OUT / "models" / model / "phase424_collection_complete.json"
            )
            self.assertTrue(complete["all_rows_complete"])
            self.assertTrue(complete["component_ledger_gate_pass"])
            self.assertEqual(complete["condition_count"], 1728)
            self.assertEqual(complete["pair_count"], 864)
            self.assertFalse(complete["causal"])

    def test_analysis_keeps_closure_closed(self) -> None:
        summary = read_json(OUT / "phase424_global_summary.json")
        self.assertEqual(summary["strict_mechanism_closure"], "0/72")
        self.assertEqual(summary["strict_double_blind_mechanism_count"], 0)
        self.assertEqual(summary["causally_closed_mechanism_count"], 0)
        maps = read_jsonl(OUT / "phase424_mechanism_maps.jsonl")
        self.assertEqual(len(maps), 216)
        self.assertFalse(any(row["mechanism_closed"] for row in maps))

    def test_visual_manifest(self) -> None:
        manifest = read_json(VIS / "manifest.json")
        self.assertEqual(
            manifest["schema_version"],
            "phase424_global_physical_path_manifest.v1",
        )
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            graph = read_json(VIS / item["filename"])
            self.assertEqual(graph["schema_version"], "atlas_graph_v1")
            self.assertEqual(len(graph["graph"]["nodes"]), 216)
            self.assertEqual(len(graph["graph"]["edges"]), 144)
            self.assertFalse(graph["graph"]["meta"]["causal"])


if __name__ == "__main__":
    unittest.main()
