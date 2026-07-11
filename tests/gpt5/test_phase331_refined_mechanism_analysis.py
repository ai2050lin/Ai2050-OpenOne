from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase331_refined_mechanism_analysis import direction_consistency


class Phase331AnalysisTests(unittest.TestCase):
    def test_direction_consistency_is_computed_per_heldout_item(self):
        rows = [
            {"item_index": 19, "delta_target_margin_vs_baseline": -1.0},
            {"item_index": 19, "delta_target_margin_vs_baseline": 0.2},
            {"item_index": 20, "delta_target_margin_vs_baseline": -0.1},
            {"item_index": 21, "delta_target_margin_vs_baseline": -0.2},
            {"item_index": 22, "delta_target_margin_vs_baseline": 0.3},
        ]
        self.assertEqual(direction_consistency(rows), 0.75)

    def test_completed_summary_keeps_closure_gate_closed(self):
        result = ROOT / "tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit"
        execution = json.loads(
            (result / "phase331_execution_quality.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (result / "phase331_global_summary.json").read_text(encoding="utf-8")
        )
        self.assertTrue(execution["valid"])
        self.assertEqual(execution["interface_case_count"], 720)
        self.assertEqual(execution["condition_row_count"], 9360)
        self.assertEqual(summary["results"]["full_gate_pass_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)


if __name__ == "__main__":
    unittest.main()
