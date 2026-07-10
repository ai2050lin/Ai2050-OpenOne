#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"


class Phase326PublishedAtlasTest(unittest.TestCase):
    def test_manifest_separates_neurons_and_component_sets(self) -> None:
        manifest = json.loads((ATLAS / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["metrics"]["mapped_family_count"], 2)
        self.assertEqual(manifest["metrics"]["single_unit_causal_count"], 0)
        self.assertGreater(manifest["metrics"]["single_neuron_candidate_count"], 0)
        self.assertEqual(manifest["metrics"]["component_set_member_count"], 288)
        self.assertEqual(manifest["metrics"]["expanded_confirmed_candidate_count"], 72)

    def test_reasoning_partition_contains_only_noncausal_set_members(self) -> None:
        partition = json.loads((ATLAS / "partitions/reasoning_constraint/qwen3.json").read_text(encoding="utf-8"))
        self.assertTrue(partition["nodes"])
        self.assertTrue(all(node["node_type"] == "component_set_member" for node in partition["nodes"]))
        self.assertTrue(all(node["single_unit_causal"] is False for node in partition["nodes"]))
        self.assertTrue(all(edge["causal"] is False for edge in partition["membership_edges"]))


if __name__ == "__main__":
    unittest.main()
