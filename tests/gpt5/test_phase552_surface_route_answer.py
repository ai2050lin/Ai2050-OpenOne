#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase552_full_layer_factorial_analysis as analysis  # noqa: E402
import phase552_surface_route_answer_protocol as protocol  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase552SurfaceRouteAnswerTests(unittest.TestCase):
    def test_factorial_contract_relations(self) -> None:
        registered = read_jsonl(protocol.CASES_PATH)
        anchor_ids = []
        for row in registered:
            key = (row["model"], row["mechanism_id"])
            if key not in {(item[0], item[1]) for item in anchor_ids}:
                anchor_ids.append((row["model"], row["mechanism_id"], row["anchor_id"]))
        self.assertEqual(len(anchor_ids), 4)
        for _model, _mechanism, anchor_id in anchor_ids:
            rows = {
                row["factorial_cell"]: row
                for row in registered
                if row["anchor_id"] == anchor_id
            }
            for answer in ("a", "b"):
                targets = {
                    rows[f"route{route}_surface{surface}_answer_{answer}"]["target"]
                    for route in (0, 1)
                    for surface in (0, 1)
                }
                self.assertEqual(len(targets), 1)
            self.assertNotEqual(
                rows["route0_surface0_answer_a"]["target"],
                rows["route0_surface0_answer_b"]["target"],
            )

    def test_registered_protocol_is_complete_and_fresh(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_case_count"], 4672)
        self.assertEqual(audit["anchor_count"], 584)
        self.assertEqual(audit["rows_per_anchor"], [8])
        self.assertEqual(audit["duplicate_prompt_count"], 0)
        self.assertEqual(audit["phase551_entity_overlap_count"], 0)

    def test_behavior_gate_passed_before_observation(self) -> None:
        rows = read_jsonl(
            protocol.OUT_DIR / "phase552_behavior_qualification.jsonl"
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["behavior_gate_pass"] for row in rows))
        self.assertTrue(all(row["observer_collection_authorized"] for row in rows))
        for row in rows:
            for split in row["split_reports"].values():
                self.assertEqual(split["all_cells_exact"]["count"], 73)
                self.assertEqual(split["all_cells_exact"]["n"], 73)

    @staticmethod
    def synthetic_rows(route: float, surface: float, answer: float) -> list[dict]:
        return [
            {
                "features": {
                    "layer_output__query": {
                        "semantic_route_effect": route,
                        "surface_form_effect": surface,
                        "answer_identity_effect": answer,
                        "route_minus_max_control": route - max(surface, answer),
                        "route_to_max_control_ratio": route / max(surface, answer),
                    }
                }
            }
            for _ in range(73)
        ]

    def test_factorial_gate_requires_route_to_beat_both_controls(self) -> None:
        passed = analysis.event_report(
            self.synthetic_rows(0.4, 0.2, 0.25), "layer_output__query",
        )
        failed = analysis.event_report(
            self.synthetic_rows(0.4, 0.2, 0.39), "layer_output__query",
        )
        self.assertTrue(passed["semantic_route_gate_pass"])
        self.assertFalse(failed["semantic_route_gate_pass"])

    def test_real_result_stops_before_intervention(self) -> None:
        summary = read_json(analysis.SUMMARY_PATH)
        self.assertEqual(summary["tested_layer_feature_event_count"], 1824)
        self.assertEqual(summary["phase551_uncontrolled_confirmed_event_count"], 1053)
        self.assertEqual(summary["independently_confirmed_semantic_route_event_count"], 0)
        self.assertEqual(summary["event_count_reduction_from_phase551"], 1053)
        self.assertFalse(summary["intervention_authorized"])
        self.assertEqual(summary["stop_reason"], "zero_controlled_semantic_route_candidates")
        self.assertEqual(len(summary["contract_overview"]), 4)
        self.assertGreater(len(summary["top_near_miss_coordinates"]), 0)


if __name__ == "__main__":
    unittest.main()
