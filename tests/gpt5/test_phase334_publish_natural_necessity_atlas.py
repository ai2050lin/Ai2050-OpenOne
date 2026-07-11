from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"


class Phase334AtlasTests(unittest.TestCase):
    def test_component_candidate_boundary(self) -> None:
        rows = [
            json.loads(line)
            for line in (ATLAS / "phase334_natural_necessity_nodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 54)
        self.assertTrue(all(row["node_type"] == "natural_necessity_component_candidate" for row in rows))
        self.assertTrue(all(not row["single_unit_causal"] for row in rows))

    def test_manifest_sync(self) -> None:
        manifest = json.loads((ATLAS / "manifest.json").read_text())
        self.assertEqual(manifest["phase"], 334)
        self.assertEqual(manifest["metrics"]["phase334_candidate_node_count"], 54)
        self.assertEqual(manifest["metrics"]["single_unit_causal_count"], 0)


if __name__ == "__main__":
    unittest.main()
