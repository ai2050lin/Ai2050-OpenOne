from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase603_fruit_residual_observer"


class Phase603FruitResidualTest(unittest.TestCase):
    def test_only_qualified_branches_were_collected(self) -> None:
        expected = {"qwen3": ["daily", "explicit_evidence", "technical"], "glm4": ["daily"]}
        for model, tracks in expected.items():
            summary = json.loads((OUT / f"phase603_{model}_qualified_residual_summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["tracks"], tracks)
            self.assertTrue(summary["internal_state_collected"])
            self.assertFalse(summary["attention_or_mlp_collected"])
            self.assertFalse(summary["causal_intervention"])
        self.assertFalse((OUT / "phase603_deepseek7b_qualified_residual_summary.json").exists())

    def test_analysis_keeps_observational_boundary(self) -> None:
        analysis = json.loads((OUT / "phase603_residual_analysis.json").read_text())
        self.assertEqual(set(analysis["branches"]), {
            "qwen3/technical", "qwen3/daily", "qwen3/explicit_evidence", "glm4/daily",
        })
        self.assertFalse(analysis["cross_model_unit_identity_comparison_authorized"])
        self.assertFalse(analysis["causal_intervention_authorized"])
        self.assertFalse(analysis["mechanism_claim_authorized"])
        self.assertFalse(analysis["theory_or_formula_update_authorized"])


if __name__ == "__main__":
    unittest.main()
