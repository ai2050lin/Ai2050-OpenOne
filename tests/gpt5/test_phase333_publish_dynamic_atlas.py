from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"


class Phase333AtlasTests(unittest.TestCase):
    def test_dynamic_event_boundary(self) -> None:
        nodes = [json.loads(line) for line in (ATLAS / "neuron_nodes.jsonl").read_text().splitlines() if line.strip()]
        phase333 = [row for row in nodes if row.get("phase333_tested")]
        self.assertEqual(len(phase333), 18)
        self.assertTrue(all(row["node_type"] == "dynamic_path_event" for row in phase333))
        self.assertTrue(all(not row["single_unit_causal"] for row in phase333))

    def test_manifest_sync(self) -> None:
        manifest = json.loads((ATLAS / "manifest.json").read_text())
        self.assertEqual(manifest["phase"], 333)
        self.assertEqual(manifest["metrics"]["phase333_dynamic_event_count"], 18)
        mapped = [row for row in manifest["partitions"] if row.get("phase333_dynamic_event_count")]
        self.assertEqual(len(mapped), 3)


if __name__ == "__main__":
    unittest.main()
