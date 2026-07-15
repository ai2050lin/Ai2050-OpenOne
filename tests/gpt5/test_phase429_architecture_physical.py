#!/usr/bin/env python3
"""Evidence-boundary tests for the Phase429 architecture trace."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase429_typed_route"
VIS = ROOT / "frontend/public/vis_data/phase429_architecture_path"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase429ArchitecturePhysicalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = read_json(OUT / "phase429_physical_protocol.json")
        cls.gate = read_json(OUT / "phase429_open_physical_gate.json")
        cls.audit = read_json(OUT / "phase429_posthoc_audit.json")
        cls.rows = read_jsonl(
            OUT / "physical/open/qwen3/phase429_physical_rows.jsonl"
        )

    def test_frozen_physical_denominator(self) -> None:
        self.assertTrue(self.protocol["valid"])
        self.assertEqual(self.protocol["condition_count"], 3840)
        self.assertEqual(self.protocol["independent_group_count"], 384)
        self.assertEqual(
            self.protocol["condition_rows_sha256"],
            "3399c0e8dd5050d9cb7f6411399c62c5c9c77a5fd265541a306a9968b2856a00",
        )
        self.assertFalse(
            self.protocol["record_contract"]["head_channel_neuron_scan"]
        )
        self.assertFalse(self.protocol["record_contract"]["intervention"])

    def test_complete_architecture_trace(self) -> None:
        self.assertEqual(len(self.rows), 138240)
        counts = Counter(row["condition_id"] for row in self.rows)
        self.assertEqual(set(counts.values()), {36})
        self.assertTrue(all(row["physical"] and row["observer"] for row in self.rows))
        self.assertTrue(all(not row["pipeline_sealed"] for row in self.rows))
        self.assertTrue(all(not row["causal"] for row in self.rows))
        self.assertTrue(all(not row["single_neuron"] for row in self.rows))

    def test_registered_stop_and_interpretability_correction(self) -> None:
        self.assertTrue(self.gate["reconstruction_gate_pass"])
        self.assertFalse(self.gate["prediction_gate_pass"])
        self.assertFalse(self.gate["sealed_unlock"])
        self.assertEqual(
            self.gate["prediction"]["physical_group_success"]["estimate"], 0.0
        )
        self.assertEqual(
            self.gate["prediction"]["majority_baseline_group_success"]["estimate"],
            1.0,
        )
        physical = self.audit["physical_audit"]
        self.assertTrue(physical["query_precedes_output_instruction"])
        self.assertFalse(physical["readout_is_true_autoregressive_terminal"])
        self.assertFalse(physical["terminal_prediction_interpretation_valid"])
        self.assertFalse(self.audit["stage_decision"]["sealed_stage_authorized"])

    def test_visual_graph_preserves_evidence_boundary(self) -> None:
        manifest = read_json(VIS / "manifest.json")
        payload = read_json(VIS / manifest["items"][0]["filename"])
        nodes = payload["graph"]["nodes"]
        edges = payload["graph"]["edges"]
        meta = payload["graph"]["meta"]
        self.assertEqual(len(nodes), 252)
        self.assertEqual(len(edges), 251)
        self.assertTrue(all(node["physical"] for node in nodes))
        self.assertTrue(all(not node["predictive"] for node in nodes))
        self.assertTrue(all(not node["causal"] for node in nodes))
        self.assertTrue(all(not node["single_neuron"] for node in nodes))
        self.assertFalse(meta["readout_is_autoregressive_terminal"])
        self.assertFalse(meta["terminal_prediction_interpretation_valid"])


if __name__ == "__main__":
    unittest.main()
