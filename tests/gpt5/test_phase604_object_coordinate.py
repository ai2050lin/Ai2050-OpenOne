from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase604_object_coordinate_control"


class Phase604ObjectCoordinateTest(unittest.TestCase):
    def test_extractions_are_complete_and_pre_option(self) -> None:
        for model, count in (("qwen3", 1440), ("glm4", 480)):
            summary = json.loads((OUT / f"phase604_{model}_object_coordinate_summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["case_count"], count)
            self.assertGreaterEqual(summary["minimum_exact_occurrence_count"], 1)
            self.assertTrue(summary["future_option_tokens_excluded"])
            self.assertFalse(summary["causal_intervention"])

    def test_analysis_preserves_evidence_boundary(self) -> None:
        analysis = json.loads((OUT / "phase604_coordinate_analysis.json").read_text())
        self.assertEqual(len(analysis["branches"]), 4)
        self.assertTrue(analysis["future_option_tokens_excluded"])
        self.assertFalse(analysis["causal_intervention_authorized"])
        self.assertFalse(analysis["mechanism_claim_authorized"])
        self.assertFalse(analysis["theory_or_formula_update_authorized"])


if __name__ == "__main__":
    unittest.main()
