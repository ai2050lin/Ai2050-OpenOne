from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase331_refined_mechanism_case_bank as phase331


ROUND = ROOT / "tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase331CaseBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(ROUND / "phase331_registered_cases.jsonl")

    def test_registered_denominator_is_frozen_and_balanced(self):
        rows = self.rows
        self.assertEqual(len(rows), 720)
        self.assertEqual(len({row["audit_case_id"] for row in rows}), 720)
        self.assertEqual(sum(row["cohort"] == "positive" for row in rows), 360)
        self.assertEqual(sum(row["cohort"] == "matched_negative_control" for row in rows), 360)
        self.assertEqual({row["item_index"] for row in rows}, {19, 20, 21, 22})
        self.assertEqual({row["template_id"] for row in rows}, set(phase331.TEMPLATES))
        self.assertEqual({row["interface"] for row in rows}, set(phase331.INTERFACES))
        self.assertTrue(all(not row["selection_updates_allowed"] for row in rows))
        self.assertTrue(all(not row["single_unit_intervention_gate_open"] for row in rows))

    def test_positive_and_control_pairing_is_exact(self):
        pairs = {
            (row["family_id"], row["mechanism_id"], row["paired_mechanism_id"], row["cohort"])
            for row in self.rows
        }
        for family, positive, control in phase331.PAIR_REGISTRY:
            self.assertIn((family, positive, control, "positive"), pairs)
            self.assertIn((family, control, positive, "matched_negative_control"), pairs)

    def test_protocol_separates_coverage_from_closure(self):
        protocol = json.loads((ROUND / "phase331_registered_protocol.json").read_text(encoding="utf-8"))
        self.assertTrue(protocol["denominator_frozen"])
        self.assertFalse(protocol["single_unit_intervention_gate_open"])
        self.assertFalse(protocol["theory_update_gate_open"])
        self.assertEqual(protocol["success_gate"], [
            "readout_specific", "expanded_heldout", "cross_interface", "cross_model",
            "member_localized", "compensation_accounted", "full_generation_changed", "low_side_effect",
        ])


if __name__ == "__main__":
    unittest.main()
