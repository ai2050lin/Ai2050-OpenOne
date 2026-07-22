#!/usr/bin/env python3
"""Tests for the Phase591 external human-review gate."""

from __future__ import annotations

import gzip
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase591_blind_semantic_gold_analysis as analysis  # noqa: E402
import phase591_blind_semantic_gold_protocol as protocol  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Phase591BlindSemanticGoldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        protocol.register()
        analysis.analyze()

    def test_source_and_packets_are_blind_and_complete(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["source_item_count"], 288)
        self.assertEqual(audit["hidden_metadata_key_count"], 0)
        self.assertFalse(audit["private_answer_key_read"])
        orders = []
        for slot in protocol.REVIEWER_SLOTS:
            packet = read_jsonl_gz(protocol.packet_path(slot))
            self.assertEqual(len(packet), 288)
            self.assertEqual(len({row["review_id"] for row in packet}), 288)
            self.assertTrue(all(row["reviewer_slot"] == slot for row in packet))
            orders.append(tuple(row["review_id"] for row in packet))
        self.assertEqual(len(set(orders)), 3)

    def test_response_templates_require_semantic_structure(self) -> None:
        for slot in protocol.REVIEWER_SLOTS:
            rows = read_jsonl(protocol.response_template_path(slot))
            self.assertEqual(len(rows), 288)
            self.assertTrue(
                all(
                    "semantic_polarity" in row
                    and "negation_scope_text" in row
                    and "condition_scope_text" in row
                    and "first_decisive_character_offset" in row
                    and "factuality" in row
                    and "factuality_basis" in row
                    and "factuality_evidence" in row
                    and "confidence_1_to_5" in row
                    and "rationale" in row
                    for row in rows
                )
            )

    def test_absent_humans_close_every_scientific_gate(self) -> None:
        status = read_json(analysis.STATUS_PATH)
        stage = read_json(analysis.STAGE_PATH)
        self.assertEqual(status["completed_structurally_valid_reviewer_count"], 0)
        self.assertFalse(status["independent_human_semantic_gold_complete"])
        self.assertFalse(status["private_answer_key_read"])
        self.assertEqual(
            status["consensus_status_counts"]["pending_three_independent_reviewers"],
            288,
        )
        self.assertEqual(stage["status"], "blocked_pending_external_human_review")
        self.assertEqual(stage["denominators"]["model_case_count_consumed"], 0)
        self.assertEqual(stage["denominators"]["sealed_case_count_consumed"], 0)
        self.assertFalse(stage["automatic_execution_now"])
        self.assertTrue(all(not value for value in stage["authorization"].values()))

    def test_consensus_rule_rejects_opposite_and_conditional_conflicts(self) -> None:
        def rows(labels: tuple[str, str, str]) -> list[dict]:
            return [{"semantic_polarity": label} for label in labels]

        self.assertEqual(
            analysis.label_consensus(rows(("positive", "positive", "unresolved"))),
            ("positive", "nonopposed_majority"),
        )
        self.assertEqual(
            analysis.label_consensus(rows(("positive", "positive", "negative"))),
            (None, "semantic_label_conflict"),
        )
        self.assertEqual(
            analysis.label_consensus(rows(("conditional", "conditional", "positive"))),
            (None, "semantic_label_conflict"),
        )


if __name__ == "__main__":
    unittest.main()
