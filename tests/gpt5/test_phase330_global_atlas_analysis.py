import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase330_global_atlas_analysis as analysis


class Phase330GlobalAnalysisTest(unittest.TestCase):
    def test_complete_analysis(self):
        result = analysis.analyze("nine_family_global_atlas")
        self.assertEqual(result["counts"]["prompt_model_cases"], 15552)
        self.assertEqual(result["counts"]["component_event_rows"], 4852224)
        self.assertEqual(result["counts"]["causal_condition_rows"], 4320)
        self.assertEqual(result["counts"]["heldout_prediction_rows"], 3888)
        self.assertFalse(result["single_unit_intervention_gate_open"])

    def test_claim_registry_is_explicit(self):
        path = analysis.OUT / "nine_family_global_atlas" / "claim_registry.jsonl"
        claims = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(row["status"] == "not_supported" for row in claims))
        self.assertTrue(any("single_neuron" in row["claim_id"] for row in claims))


if __name__ == "__main__":
    unittest.main()
