#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase551_model_specific_route_protocol as protocol  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase551ModelSpecificRouteTests(unittest.TestCase):
    def test_all_mechanisms_preserve_answer_identity_across_routes(self) -> None:
        for mechanism in protocol.MECHANISMS:
            for scaffold in protocol.SCAFFOLDS:
                rows = {
                    cell: protocol.case_spec(
                        mechanism, "discovery", 13, cell, scaffold,
                    )
                    for cell in protocol.CELLS
                }
                self.assertEqual(
                    rows["route0_answer_a"]["target"],
                    rows["route1_answer_a"]["target"],
                )
                self.assertEqual(
                    rows["route0_answer_b"]["target"],
                    rows["route1_answer_b"]["target"],
                )
                self.assertNotEqual(
                    rows["route0_answer_a"]["target"],
                    rows["route0_answer_b"]["target"],
                )

    def test_registered_denominators_are_complete_and_disjoint(self) -> None:
        calibration = read_json(protocol.CALIBRATION_AUDIT_PATH)
        validation = read_json(protocol.VALIDATION_AUDIT_PATH)
        self.assertTrue(calibration["valid"])
        self.assertEqual(calibration["registered_case_count"], 5760)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["registered_case_count"], 4672)
        self.assertEqual(validation["duplicate_prompt_count"], 0)
        self.assertEqual(validation["calibration_entity_overlap_count"], 0)

    def test_validation_freeze_and_behavior_gate(self) -> None:
        frozen = read_json(protocol.FROZEN_SCAFFOLDS_PATH)
        qualification = read_jsonl(
            protocol.OUT_DIR / "phase551_validation_behavior_qualification.jsonl"
        )
        self.assertEqual(len(frozen["selections"]), 12)
        self.assertEqual(
            sum(row["validation_authorized"] for row in frozen["selections"]),
            8,
        )
        passed = {
            (row["model"], row["mechanism_id"])
            for row in qualification
            if row["observer_collection_authorized"]
        }
        self.assertEqual(
            passed,
            {
                ("qwen3", "category"),
                ("qwen3", "negated_attribute"),
                ("glm4", "category"),
                ("glm4", "negated_attribute"),
            },
        )

    def test_uncontrolled_events_remain_observer_only(self) -> None:
        summary = read_json(
            protocol.OUT_DIR / "phase551_full_layer_route_summary.json"
        )
        events = read_jsonl(
            protocol.OUT_DIR / "phase551_confirmed_route_observer_events.jsonl"
        )
        self.assertEqual(summary["tested_layer_feature_event_count"], 1824)
        self.assertEqual(summary["independently_confirmed_route_event_count"], 1053)
        self.assertEqual(len(events), 1053)
        self.assertTrue(all(row["observer_only"] for row in events))
        self.assertTrue(all(not row["compute_edge"] for row in events))
        self.assertTrue(all(not row["causal"] for row in events))


if __name__ == "__main__":
    unittest.main()
