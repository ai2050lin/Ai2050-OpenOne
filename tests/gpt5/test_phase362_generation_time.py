from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time"
P361_CANDIDATES = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace/four_admitted_balanced_trace/phase361_frozen_predictive_candidates.jsonl"


class Phase362GenerationTimeTests(unittest.TestCase):
    def test_independent_denominator_and_candidate_freeze(self) -> None:
        summary = json.loads((OUT / "phase362_case_summary.json").read_text())
        self.assertEqual(summary["denominator"]["case_count"], 384)
        self.assertEqual(summary["denominator"]["independent_calibration_case_count"], 288)
        self.assertEqual(summary["denominator"]["physical_confirmation_case_count"], 96)
        self.assertEqual(summary["quality"]["phase361_case_overlap_count"], 0)
        digest = hashlib.sha256(P361_CANDIDATES.read_bytes()).hexdigest()
        self.assertEqual(summary["frozen_phase361_candidates"]["sha256"], digest)

    def test_anchor_edge_replay(self) -> None:
        summary = json.loads((OUT / "phase362_anchor_replay_summary.json").read_text())
        self.assertEqual(summary["denominator"]["anchor_count"], 9)
        self.assertEqual(summary["denominator"]["layer_file_count"], 520)
        self.assertTrue(summary["quality"]["all_offline_gates_pass"])
        self.assertFalse(summary["claim_boundary"]["physical_confirmation_opened"])

    def test_frozen_candidate_boundary(self) -> None:
        frozen = json.loads((OUT / "phase362_frozen_candidate_summary.json").read_text())
        posthoc = json.loads((OUT / "phase362_posthoc_survivor_summary.json").read_text())
        self.assertEqual(frozen["results"]["b3_independently_best_all_models_count"], 7)
        self.assertFalse(frozen["identifiability_audit"]["next_generation_predictive_gate_identifiable_without_new_rule"])
        self.assertEqual(posthoc["results"]["selective_next_layer_association_count"], 6)
        self.assertEqual(posthoc["results"]["temporal_predictive_survivor_count"], 0)
        self.assertFalse(posthoc["claim_boundary"]["physical_confirmation_opened"])


if __name__ == "__main__":
    unittest.main()
