import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase330_global_atlas_survey as survey
import phase330_nine_family_case_bank as bank


class Phase330SurveyTest(unittest.TestCase):
    def test_denominator(self):
        rows = bank.build_cases()
        self.assertEqual(len(rows), 5184)
        self.assertEqual(sum(row["selection_eligible"] for row in rows), 2592)

    def test_protocol_checks(self):
        case = {"expected_structure": "json", "protocol": "json", "target": "red"}
        self.assertTrue(survey.protocol_ok(case, '{"answer":"red"}'))
        self.assertFalse(survey.protocol_ok(case, "red"))

    def test_path_signature(self):
        rows = []
        for layer, value in enumerate((0.1, 1.0, 0.4)):
            rows.append({
                "schema_version": "8", "phase_id": "Phase330", "model": "qwen3",
                "case_id": "c", "item_id": "i", "family_id": "f", "mechanism_id": "m",
                "split": "discovery", "template_id": "template_a", "language": "en",
                "target_bucket": "lexical", "target_absent_from_prompt": True,
                "selection_eligible": True, "component_type": "mlp", "position_role": "last",
                "layer": layer, "projection": value,
            })
        result = survey.path_signatures(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["peak_layer"], 1)
        self.assertEqual(result[0]["sign_flip_count"], 0)


if __name__ == "__main__":
    unittest.main()
