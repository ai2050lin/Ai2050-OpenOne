#!/usr/bin/env python3
"""Contract tests for the frozen Phase427 behavior qualification."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase427_behavior_analysis import behavior_gate  # noqa: E402
from phase427_behavior_collect import parse_generation  # noqa: E402
from phase427_dual_route_protocol import MODELS, OUT, freeze  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase427ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = freeze()

    def test_frozen_denominator_and_implementation(self) -> None:
        validation = self.protocol["validation"]
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["formal_group_count"], 1024)
        self.assertEqual(validation["formal_condition_count"], 30720)
        self.assertEqual(validation["open_condition_count"], 23040)
        self.assertEqual(validation["sealed_condition_count"], 7680)
        self.assertEqual(validation["instrument_group_count"], 8)
        self.assertEqual(validation["instrument_condition_count"], 240)
        self.assertEqual(validation["route_position_mismatch_count"], 0)
        self.assertTrue(
            self.protocol["future_physical_contract"]["full_hidden_rank_r_map_prohibited"]
        )
        self.assertFalse(
            self.protocol["evidence_contract"]["strict_human_double_blind"]
        )
        for filename, expected in self.protocol["implementation_commitments"].items():
            actual = hashlib.sha256((ROOT / "tests/gpt5" / filename).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, filename)

    def test_route_positions_normative_scope_and_balance(self) -> None:
        rows = read_jsonl(OUT / "phase427_registered_conditions_open.jsonl")
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["semantic_group_id"], row["model"], row["role"])].append(row)
            self.assertEqual(
                len(row["target_sequence_token_ids"]),
                len(row["opposite_sequence_token_ids"]),
            )
            if row["candidate"] and row["route_mode"] in {"none", "conflict"}:
                self.assertFalse(row["normative_target"])
            else:
                self.assertTrue(row["normative_target"])
        self.assertTrue(grouped)
        for values in grouped.values():
            self.assertEqual(len(values), 5)
            reference = values[0]
            for row in values[1:]:
                self.assertEqual(row["source_positions"], reference["source_positions"])
                self.assertEqual(row["query_positions"], reference["query_positions"])
                self.assertEqual(row["before_tag_positions"], reference["before_tag_positions"])
                self.assertEqual(row["after_tag_positions"], reference["after_tag_positions"])
                self.assertEqual(row["prompt_token_count"], reference["prompt_token_count"])
        groups = read_jsonl(OUT / "phase427_registered_groups.jsonl")
        balance = Counter(
            (row["block_id"], row["split"], row["interface"], row["history"])
            for row in groups
        )
        self.assertTrue(balance)
        self.assertTrue(all(value == 16 for value in balance.values()))

    def test_behavior_gate_is_frozen_and_conjunctive(self) -> None:
        thresholds = self.protocol["registered_thresholds"]
        passing = {
            "independent_group_count": 64,
            "teacher_sequence_correct_fraction": 0.80,
            "teacher_sequence_margin_median": 0.01,
            "natural_target_first_fraction": 0.70,
            "natural_opposite_first_fraction": 0.10,
            "natural_revision_fraction": 0.05,
            "natural_boundary_fraction": 0.70,
            "natural_stop_fraction": 0.70,
            "natural_censoring_fraction": 0.25,
        }
        self.assertTrue(behavior_gate(passing, thresholds)["gate_pass"])
        for key in (
            "teacher_sequence_correct_fraction",
            "natural_target_first_fraction",
            "natural_boundary_fraction",
            "natural_stop_fraction",
        ):
            failed = dict(passing)
            failed[key] = 0.0
            self.assertFalse(behavior_gate(failed, thresholds)["gate_pass"], key)
        failed_margin = dict(passing)
        failed_margin["teacher_sequence_margin_median"] = 0.0
        self.assertFalse(behavior_gate(failed_margin, thresholds)["gate_pass"])

    def test_natural_event_parser_keeps_events_separate(self) -> None:
        row = {
            "target": "A000001",
            "opposite_target": "B000001",
            "natural_generation_max_new_tokens": 24,
            "interface": "direct",
        }
        parsed = parse_generation("A000001. B000001", [1, 2, 3], row, {99})
        self.assertTrue(parsed["natural_target_first"])
        self.assertFalse(parsed["natural_opposite_first"])
        self.assertTrue(parsed["natural_revision"])
        self.assertTrue(parsed["natural_boundary"])
        self.assertTrue(parsed["natural_stop"])
        self.assertFalse(parsed["natural_exact_contract"])

    def test_completed_outputs_if_present(self) -> None:
        summary_path = OUT / "phase427_global_summary.json"
        if not summary_path.exists():
            return
        summary = read_json(summary_path)
        self.assertEqual(summary["strict_mechanism_closure"], "0/72")
        self.assertFalse(summary["physical_tested"])
        self.assertFalse(summary["causal_tested"])
        gate = read_json(OUT / "phase427_open_gate_freeze.json")
        self.assertFalse(gate["physical_hooks_run"])
        for model in MODELS:
            complete = read_json(
                OUT / "models" / model / "open" / "phase427_collection_complete.json"
            )
            self.assertTrue(complete["all_rows_complete"])
            self.assertEqual(complete["condition_count"], 7680)
            rows = read_jsonl(
                OUT / "models" / model / "open" / "phase427_behavior_rows.jsonl"
            )
            self.assertEqual(len(rows), 7680)
            self.assertTrue(
                all(
                    math.isfinite(float(row["teacher_sequence_logprob_margin"]))
                    for row in rows
                )
            )


if __name__ == "__main__":
    unittest.main()
