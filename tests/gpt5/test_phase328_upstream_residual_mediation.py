#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase328_upstream_residual_mediation as phase328


class Phase328MediationTest(unittest.TestCase):
    def test_selection_uses_primary_query_rows_only(self) -> None:
        rows = []
        for layer in range(1, 5):
            for variant, value in (("same_target_object", 0.2), ("unrelated_wrong_target", 0.2 + layer)):
                rows.append({
                    "model": "qwen3",
                    "mechanism_id": phase328.MECHANISM,
                    "split": "registered_primary",
                    "position_role": phase328.ROLE,
                    "layer": layer,
                    "comparison_variant": variant,
                    "residual_rms_delta": value,
                })
        selection = phase328.freeze_residual_selection("qwen3", rows)
        self.assertEqual(selection["layer"], 3)
        self.assertEqual(selection["position_role"], "query")
        self.assertFalse(selection["selection_updates_allowed"])

    def test_validation_split_has_six_objects_and_two_templates(self) -> None:
        cases = phase328.validation_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case["base_case_id"] for case in cases}), 6)
        self.assertTrue(all(case["split"] == "registered_confirmation" for case in cases))


if __name__ == "__main__":
    unittest.main()
