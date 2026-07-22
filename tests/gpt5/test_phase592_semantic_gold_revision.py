#!/usr/bin/env python3
"""Tests for the Phase592 pre-review protocol revision."""

from __future__ import annotations

import gzip
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase592_semantic_gold_revision_analysis as analysis  # noqa: E402
import phase592_semantic_gold_revision_protocol as protocol  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Phase592SemanticGoldRevisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        protocol.register()
        analysis.analyze()

    def test_revision_precedes_all_human_submissions(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["pre_execution_revision"])
        self.assertEqual(audit["phase591_completed_file_count_before_revision"], 0)
        self.assertFalse(audit["private_answer_key_read"])
        self.assertFalse(audit["phase590_sealed_cases_read"])

    def test_packets_have_fixed_batches_and_revised_spans(self) -> None:
        orders = []
        for slot in protocol.REVIEWER_SLOTS:
            packet = read_jsonl_gz(protocol.packet_path(slot))
            template = read_jsonl(protocol.response_template_path(slot))
            self.assertEqual(len(packet), 288)
            self.assertEqual(len(template), 288)
            self.assertEqual(len({row["batch_id"] for row in packet}), 6)
            self.assertTrue(
                all(
                    "decisive_span_start" in row
                    and "decisive_span_end" in row
                    and "later_text_changes_final_semantics" in row
                    and "factuality_source_tier" in row
                    and "batch_started_at" in row
                    and "batch_completed_at" in row
                    for row in template
                )
            )
            orders.append(tuple(row["review_id"] for row in packet))
        self.assertEqual(len(set(orders)), 3)

    def test_semantic_consensus_and_anchor_are_separate(self) -> None:
        self.assertEqual(
            analysis.direct_consensus(["positive", "positive", "unresolved"], "unresolved"),
            ("positive", "nonopposed_majority"),
        )
        self.assertEqual(
            analysis.direct_consensus(["positive", "positive", "negative"], "unresolved"),
            (None, "requires_independent_adjudication"),
        )
        rows = [
            {
                "semantic_polarity": "positive",
                "response_complete": True,
                "later_text_changes_final_semantics": False,
                "decisive_span_start": 4,
                "decisive_span_end": 10,
            },
            {
                "semantic_polarity": "positive",
                "response_complete": True,
                "later_text_changes_final_semantics": False,
                "decisive_span_start": 7,
                "decisive_span_end": 12,
            },
            {
                "semantic_polarity": "unresolved",
                "response_complete": True,
                "later_text_changes_final_semantics": False,
                "decisive_span_start": None,
                "decisive_span_end": None,
            },
        ]
        self.assertEqual(analysis.overlapping_anchor(rows, "positive"), (True, [7, 10]))

    def test_no_humans_keep_all_scientific_gates_closed(self) -> None:
        status = read_json(analysis.STATUS_PATH)
        stage = read_json(analysis.STAGE_PATH)
        self.assertEqual(status["directly_resolved_item_count"], 0)
        self.assertEqual(status["workflow_unresolved_item_count"], 288)
        self.assertFalse(status["semantic_gold_complete"])
        self.assertFalse(status["resolved_gold_artifact_written"])
        self.assertEqual(status["event_anchor_qualified_item_count"], 0)
        self.assertEqual(stage["status"], "blocked_pending_external_human_review")
        self.assertEqual(stage["denominators"]["model_case_count_consumed"], 0)
        self.assertEqual(stage["denominators"]["sealed_case_count_consumed"], 0)
        self.assertFalse(stage["automatic_execution_now"])
        self.assertTrue(all(not value for value in stage["authorization"].values()))


if __name__ == "__main__":
    unittest.main()
