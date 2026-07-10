#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"


class Phase328PublishedAtlasTest(unittest.TestCase):
    def test_manifest_keeps_all_causal_counts_at_zero(self) -> None:
        manifest = json.loads((ATLAS / "manifest.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(manifest["phase"], 328)
        self.assertEqual(manifest["metrics"]["unique_unit_count"], 1121)
        self.assertEqual(manifest["metrics"]["upstream_residual_mediation_edge_count"], 3)
        self.assertEqual(manifest["metrics"]["cross_model_causal_path_edge_count"], 0)
        self.assertEqual(manifest["metrics"]["single_unit_causal_count"], 0)

    def test_partition_exposes_noncausal_residual_edge(self) -> None:
        partition = json.loads(
            (ATLAS / "partitions/content_knowledge/glm4.json").read_text(encoding="utf-8")
        )
        edges = partition["path"]["upstream_residual_mediation_edges"]
        self.assertEqual(len(edges), 1)
        self.assertFalse(edges[0]["causal"])
        category = [
            node for node in partition["nodes"]
            if node.get("mechanism_id") == "category_retrieval"
            and node.get("node_type") == "component_set_member"
        ]
        self.assertTrue(category)
        self.assertTrue(all(node["phase328_upstream_mediation_pass"] for node in category))
        self.assertTrue(all(not node["phase328_causal_edge"] for node in category))


if __name__ == "__main__":
    unittest.main()
