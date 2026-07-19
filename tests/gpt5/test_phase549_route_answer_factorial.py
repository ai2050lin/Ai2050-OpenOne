#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase549_factorial_observer_analysis as analysis  # noqa: E402
import phase549_route_answer_factorial_protocol as protocol  # noqa: E402


class Phase549FactorialTests(unittest.TestCase):
    def test_factorial_target_relations(self) -> None:
        for mechanism in protocol.MECHANISMS:
            rows = {
                cell: protocol.case_spec(mechanism, "discovery", 9, cell)
                for cell in protocol.CELLS
            }
            self.assertEqual(rows["route0_answer_a"]["target"], rows["route1_answer_a"]["target"])
            self.assertEqual(rows["route0_answer_b"]["target"], rows["route1_answer_b"]["target"])
            self.assertNotEqual(rows["route0_answer_a"]["target"], rows["route0_answer_b"]["target"])

    def test_registered_protocol_is_valid(self) -> None:
        payload = json.loads(protocol.AUDIT_PATH.read_text(encoding="utf-8"))
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["registered_case_count"], 3504)
        self.assertEqual(payload["factorial_relation_error_count"], 0)

    @staticmethod
    def rows(margin: float) -> list[dict[str, float]]:
        return [
            {
                "route_effect": 1.0 + margin,
                "answer_identity_effect": 1.0,
                "route_minus_answer_effect": margin,
            }
            for _ in range(73)
        ]

    def test_route_dominance(self) -> None:
        result = analysis.report(self.rows(0.4), "route")
        self.assertEqual(result["classification"], "route_dominant")

    def test_answer_dominance(self) -> None:
        result = analysis.report(self.rows(-0.4), "answer")
        self.assertEqual(result["classification"], "answer_identity_dominant")


if __name__ == "__main__":
    unittest.main()
