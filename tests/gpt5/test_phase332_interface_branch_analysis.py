from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas"


class Phase332InterfaceBranchAnalysisTests(unittest.TestCase):
    def test_execution_denominator_is_complete(self) -> None:
        quality = json.loads((RESULT / "phase332_execution_quality.json").read_text(encoding="utf-8"))
        self.assertTrue(quality["valid"])
        self.assertEqual(quality["registered_interface_case_count"], 1152)
        self.assertEqual(quality["registered_exchange_case_count"], 288)
        self.assertEqual(quality["exchange_condition_row_count"], 1728)
        self.assertFalse(quality["selection_updates_allowed"])

    def test_claims_keep_single_unit_boundary_closed(self) -> None:
        summary = json.loads((RESULT / "phase332_global_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["single_unit_intervention_gate_open_count"], 0)
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertEqual(summary["results"]["incomplete_exchange_cell_count"], 2)
        self.assertFalse(summary["language_encoding_mechanism_closed"])

    def test_glm_equivalent_interfaces_are_registered(self) -> None:
        protocol = json.loads((RESULT / "phase332_registered_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["glm4_native_no_think_equivalent_case_count"], 96)
        self.assertEqual(protocol["unique_prompt_case_count"], 1056)


if __name__ == "__main__":
    unittest.main()
