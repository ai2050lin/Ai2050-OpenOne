from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
ATLAS = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase331AtlasTests(unittest.TestCase):
    def test_phase331_overlay_preserves_single_unit_boundary(self):
        manifest = json.loads((ATLAS / "manifest.json").read_text(encoding="utf-8"))
        nodes = read_jsonl(ATLAS / "neuron_nodes.jsonl")
        refined = [row for row in nodes if row.get("phase331_tested")]
        self.assertEqual(manifest["phase"], 331)
        self.assertEqual(len(refined), 60)
        self.assertTrue(all(not row["single_unit_causal"] for row in refined))
        self.assertEqual(manifest["metrics"]["phase331_refined_mechanism_count"], 5)
        self.assertEqual(manifest["metrics"]["phase331_full_gate_pass_count"], 0)
        self.assertEqual(manifest["metrics"]["phase331_behavior_mechanism_closed_count"], 0)
        self.assertEqual(manifest["metrics"]["single_unit_causal_count"], 0)


if __name__ == "__main__":
    unittest.main()
