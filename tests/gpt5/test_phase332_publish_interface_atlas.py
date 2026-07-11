from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"


class Phase332AtlasTests(unittest.TestCase):
    def test_interface_path_overlay_preserves_evidence_boundary(self) -> None:
        nodes = [json.loads(line) for line in (ATLAS / "neuron_nodes.jsonl").read_text().splitlines() if line.strip()]
        phase332 = [row for row in nodes if row.get("phase332_tested")]
        self.assertEqual(len(nodes) - len(phase332), 1985)
        self.assertEqual(len(phase332), 286)
        self.assertTrue(all(row["node_type"] == "interface_path_member" for row in phase332))
        self.assertTrue(all(row["phase332_heldout_stable"] for row in phase332))
        self.assertTrue(all(not row["single_unit_causal"] for row in phase332))

    def test_manifest_and_partitions_are_synchronized(self) -> None:
        manifest = json.loads((ATLAS / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["phase"], 332)
        self.assertEqual(manifest["metrics"]["phase332_interface_path_member_count"], 286)
        self.assertEqual(manifest["metrics"]["phase332_full_gate_pass_count"], 0)
        mapped = [row for row in manifest["partitions"] if row.get("phase332_interface_path_member_count")]
        self.assertEqual(len(mapped), 6)
        self.assertEqual(sum(row["phase332_interface_path_member_count"] for row in mapped), 286)


if __name__ == "__main__":
    unittest.main()
