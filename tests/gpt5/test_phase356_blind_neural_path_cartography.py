from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase356_blind_neural_path_cartography/coarse_trace_feasibility"
BANNED = {
    "case_id", "model", "family_id", "mechanism_id", "target", "operation_demanded",
    "contrast_condition", "signed_competition_margin", "actual_token_id", "expected_token_id",
}


class Phase356BlindCartographyTests(unittest.TestCase):
    def test_blind_rows_have_no_semantic_keys(self) -> None:
        path = RESULT / "phase356_blind_skeleton_rows.jsonl"
        if not path.exists():
            self.skipTest("Phase356 has not run")
        for line in path.read_text().splitlines()[:1000]:
            self.assertFalse(BANNED & set(json.loads(line)))

    def test_conservation_and_claim_boundary(self) -> None:
        path = RESULT / "phase356_global_summary.json"
        if not path.exists():
            self.skipTest("Phase356 has not run")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["quality"]["source_row_conservation_valid"])
        self.assertEqual(summary["quality"]["label_leakage_key_count"], 0)
        self.assertFalse(summary["claim_boundary"]["coarse_motif_is_full_neural_path"])
        self.assertFalse(summary["claim_boundary"]["phase356_full_success_gate"])
        self.assertEqual(summary["results"]["physical_heldout_stable_motif_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)

    def test_discovery_source_does_not_open_sealed_labels(self) -> None:
        source = (ROOT / "tests/gpt5/phase356_blind_motif_discovery.py").read_text()
        self.assertNotIn("private_label_key", source)
        self.assertNotIn("sealed_labels", source)


if __name__ == "__main__":
    unittest.main()
