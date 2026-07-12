from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase357_residual_reconstruction_audit/pre_registered_anchor_reconstruction"


class Phase357ReconstructionTests(unittest.TestCase):
    def test_fixed_anchor_denominator_and_boundaries(self) -> None:
        path = RESULT / "phase357_global_summary.json"
        if not path.exists():
            self.skipTest("Phase357 has not run")
        summary = json.loads(path.read_text())
        self.assertEqual(summary["denominator"]["anchor_case_count"], 36)
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertFalse(summary["results"]["target_direction_used"])
        self.assertFalse(summary["results"]["semantic_label_used_for_selection"])
        self.assertFalse(summary["results"]["physical_heldout_revealed"])
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)

    def test_claim_is_limited_to_block_decomposition(self) -> None:
        path = RESULT / "phase357_global_summary.json"
        if not path.exists():
            self.skipTest("Phase357 has not run")
        summary = json.loads(path.read_text())
        self.assertFalse(summary["claim_boundary"]["block_reconstruction_validates_full_trace_schema"])
        self.assertFalse(summary["claim_boundary"]["block_reconstruction_validates_attention_edges"])


if __name__ == "__main__":
    unittest.main()
