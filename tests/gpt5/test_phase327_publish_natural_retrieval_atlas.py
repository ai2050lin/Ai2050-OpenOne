#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"


class Phase327PublishedAtlasTest(unittest.TestCase):
    def test_manifest_preserves_physical_counts_and_adds_paths(self) -> None:
        manifest = json.loads((ATLAS / "manifest.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(manifest["phase"], 327)
        self.assertEqual(manifest["metrics"]["unique_unit_count"], 1121)
        self.assertEqual(manifest["metrics"]["component_set_member_count"], 288)
        self.assertEqual(manifest["metrics"]["natural_retrieval_path_count"], 9)
        self.assertEqual(manifest["metrics"]["full_natural_chain_pass_count"], 0)
        self.assertEqual(manifest["metrics"]["single_unit_causal_count"], 0)

    def test_content_partition_annotates_without_promoting_members(self) -> None:
        partition = json.loads(
            (ATLAS / "partitions/content_knowledge/qwen3.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(partition["path"]["natural_retrieval_paths"]), 3)
        self.assertTrue(all(not row["causal"] for row in partition["path"]["natural_retrieval_paths"]))
        set_members = [node for node in partition["nodes"] if node["node_type"] == "component_set_member"]
        self.assertTrue(set_members)
        self.assertTrue(all(node["single_unit_causal"] is False for node in set_members))
        self.assertTrue(all("phase327_status" in node for node in set_members))


if __name__ == "__main__":
    unittest.main()
