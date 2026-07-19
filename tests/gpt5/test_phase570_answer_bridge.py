#!/usr/bin/env python3

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase570_answer_bridge_protocol as protocol  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_cases() -> list[dict]:
    with gzip.open(protocol.CASES_PATH, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Phase570AnswerBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = read_cases()
        cls.frozen = read_json(protocol.PROTOCOL_PATH)
        cls.audit = read_json(protocol.AUDIT_PATH)

    def test_static_denominator(self) -> None:
        self.assertTrue(self.audit["valid"])
        self.assertEqual(self.audit["registered_case_count"], 2304)
        self.assertEqual(self.audit["registered_case_count_per_model"], 768)
        self.assertEqual(len(self.rows), 2304)
        self.assertEqual(len({row["case_id"] for row in self.rows}), 2304)
        self.assertFalse(any(row["sealed"] for row in self.rows))

    def test_balanced_model_phenotype_denominator(self) -> None:
        counts = Counter((row["model"], row["intended_phenotype"]) for row in self.rows)
        for model in protocol.MODELS:
            self.assertEqual(counts[(model, "stable_correct")], 384)
            self.assertEqual(counts[(model, "stable_relation_confusion")], 384)
            selected = self.frozen["selected_cells_by_model"][model]
            self.assertEqual(len(selected["stable_correct"]), 4)
            self.assertEqual(len(selected["stable_relation_confusion"]), 4)
            self.assertFalse(
                set(selected["stable_correct"]) & set(selected["stable_relation_confusion"])
            )

    def test_relation_competition_is_observable(self) -> None:
        for row in self.rows:
            self.assertNotEqual(row["target"], row["other_relation_target"])
            self.assertEqual({len(ids) for ids in row["candidate_token_ids"].values()}, {1})
            self.assertEqual(len({tuple(ids) for ids in row["candidate_token_ids"].values()}), 4)

    def test_candidate_layers_match_shared_late_attention_topology(self) -> None:
        self.assertEqual(
            self.frozen["selected_layers_by_model"],
            {
                "qwen3": {"target_layer": 29, "wrong_layer_control": 9, "layer_count": 36},
                "glm4": {"target_layer": 34, "wrong_layer_control": 10, "layer_count": 40},
                "deepseek7b": {"target_layer": 23, "wrong_layer_control": 7, "layer_count": 28},
            },
        )
        self.assertIn("not upstream relation encoding", self.frozen["causal_screen_scope"])

    def test_causal_runner_has_controls_and_no_sealed_path(self) -> None:
        source = (ROOT / "tests/gpt5/phase570_answer_bridge_causal.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("sealed_cases", source)
        self.assertIn("target_projection_remove", source)
        self.assertIn("random_matched_remove", source)
        self.assertIn("wrong_layer_projection_remove", source)

    def test_completed_causal_screen_is_specific_but_behavior_negative(self) -> None:
        summary = read_json(protocol.OUT_DIR / "phase570_causal_summary.json")
        self.assertEqual(summary["passed_models"], [])
        self.assertFalse(summary["cross_model_late_answer_bridge_screen_pass"])
        for report in summary["model_reports"]:
            checks = report["screen_checks"]
            self.assertTrue(checks["minimum_cases"])
            self.assertTrue(checks["bidirectional_margin"])
            self.assertTrue(checks["specific_over_random_and_wrong_layer"])
            self.assertFalse(checks["correct_behavior_degrades"])
            self.assertFalse(checks["confusion_behavior_improves"])
            correct_shift = report["paired_margin_shifts_by_phenotype"][
                "stable_correct"
            ]["target_projection_remove"]["mean_margin_shift_from_baseline"]
            confusion_shift = report["paired_margin_shifts_by_phenotype"][
                "stable_relation_confusion"
            ]["target_projection_remove"]["mean_margin_shift_from_baseline"]
            self.assertLess(correct_shift, 0.0)
            self.assertGreater(confusion_shift, 0.0)
            self.assertFalse(report["late_answer_bridge_causal_screen_pass"])

    def test_completed_execution_hashes_and_paired_counts(self) -> None:
        expected_retained = {
            "qwen3": {"stable_correct": 64, "stable_relation_confusion": 62},
            "glm4": {"stable_correct": 62, "stable_relation_confusion": 59},
            "deepseek7b": {"stable_correct": 62, "stable_relation_confusion": 61},
        }
        for model, counts in expected_retained.items():
            execution = read_json(
                protocol.OUT_DIR / f"phase570_{model}_execution_summary.json"
            )
            rows_path = protocol.OUT_DIR / f"phase570_{model}_causal_rows.jsonl"
            self.assertEqual(execution["retained_paired_case_counts"], counts)
            self.assertEqual(execution["causal_row_count"], sum(counts.values()) * 4)
            self.assertEqual(execution["causal_rows_sha256"], sha256_file(rows_path))
            self.assertFalse(execution["sealed_split_read"])


if __name__ == "__main__":
    unittest.main()
