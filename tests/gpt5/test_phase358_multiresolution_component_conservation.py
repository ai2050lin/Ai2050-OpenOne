from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase358_multiresolution_full_trace/format_development_component_conservation"


class Phase358ComponentConservationTests(unittest.TestCase):
    def test_fixed_format_denominator(self) -> None:
        path = RESULT / "phase358_global_summary.json"
        if not path.exists():
            self.skipTest("Phase358 format development has not run")
        summary = json.loads(path.read_text())
        self.assertEqual(summary["denominator"]["format_case_count"], 6)
        self.assertEqual(summary["denominator"]["mlp_shard_count"], 16)
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertFalse(summary["results"]["semantic_label_used"])
        self.assertFalse(summary["results"]["top_k_selection_used"])
        self.assertTrue(summary["results"]["all_attention_heads_recorded"])
        self.assertTrue(summary["results"]["all_mlp_channels_partitioned"])

    def test_claim_boundary(self) -> None:
        path = RESULT / "phase358_global_summary.json"
        if not path.exists():
            self.skipTest("Phase358 format development has not run")
        summary = json.loads(path.read_text())
        self.assertFalse(summary["claim_boundary"]["format_development_is_full_phase358"])
        self.assertFalse(summary["claim_boundary"]["component_conservation_is_language_mechanism"])
        self.assertFalse(summary["results"]["physical_heldout_revealed"])
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)

    def test_expanded_ledger_boundary(self) -> None:
        path = RESULT / "phase358_expanded_ledger_summary.json"
        if not path.exists():
            self.skipTest("Phase358 expanded ledger has not run")
        summary = json.loads(path.read_text())
        self.assertEqual(summary["denominator"]["blind_discovery_case_count"], 9)
        self.assertEqual(summary["denominator"]["blind_calibration_case_count"], 3)
        self.assertFalse(summary["claim_boundary"]["expanded_ledger_is_blind_motif_discovery"])
        self.assertFalse(summary["claim_boundary"]["physical_heldout_tested"])
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)


if __name__ == "__main__":
    unittest.main()
