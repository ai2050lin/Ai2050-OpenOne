#!/usr/bin/env python3
"""Tests for Phase593 pre-review integrity controls."""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase593_human_semantic_review_integrity_analysis as analysis  # noqa: E402
import phase593_human_semantic_review_integrity_protocol as protocol  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def anchor_row(label: str, start: int | None, end: int | None) -> dict:
    return {
        "semantic_polarity": label,
        "response_complete": True,
        "later_text_changes_final_semantics": False,
        "decisive_span_start": start,
        "decisive_span_end": end,
    }


class Phase593HumanSemanticReviewIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        protocol.register()
        analysis.analyze()

    def test_pre_review_freeze_is_clean(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["pre_review_freeze"])
        self.assertEqual(audit["prior_or_current_completed_artifact_count"], 0)
        self.assertEqual(audit["reviewer_packet_forbidden_key_count"], 0)
        self.assertFalse(audit["private_answer_key_read"])
        self.assertFalse(audit["phase590_sealed_cases_read"])

    def test_packets_keep_main_and_repeat_denominators_separate(self) -> None:
        private_rows = read_json(protocol.PRIVATE_MAP_PATH)["rows"]
        packet_orders = []
        for slot in protocol.REVIEWER_SLOTS:
            packet = read_jsonl_gz(protocol.packet_path(slot))
            template = read_jsonl(protocol.response_template_path(slot))
            slot_map = [row for row in private_rows if row["reviewer_slot"] == slot]
            self.assertEqual(len(packet), 300)
            self.assertEqual(len(template), 300)
            self.assertEqual(
                sum(row["item_role"] == "main" for row in slot_map), 288
            )
            self.assertEqual(
                sum(row["item_role"] == "repeat_control" for row in slot_map), 12
            )
            self.assertEqual(
                {row["batch_id"] for row in packet},
                {f"batch_{index:02d}" for index in range(1, 7)},
            )
            self.assertTrue(
                all(
                    sum(row["batch_id"] == batch_id for row in packet) == 50
                    for batch_id in {row["batch_id"] for row in packet}
                )
            )
            self.assertTrue(
                all(
                    "canonical_review_id" not in row and "item_role" not in row
                    for row in packet
                )
            )
            self.assertTrue(all("condition_types" in row for row in template))
            packet_orders.append(tuple(row["submission_id"] for row in packet))
        self.assertEqual(len(set(packet_orders)), 3)

    def test_majority_anchor_uses_all_supporters(self) -> None:
        two_supporters = [
            anchor_row("positive", 4, 10),
            anchor_row("positive", 7, 12),
            anchor_row("unresolved", None, None),
        ]
        self.assertEqual(
            analysis.majority_covered_anchor(two_supporters, "positive"),
            (True, [7, 10]),
        )

        connected_three = [
            anchor_row("positive", 4, 12),
            anchor_row("positive", 7, 14),
            anchor_row("positive", 9, 13),
        ]
        self.assertEqual(
            analysis.majority_covered_anchor(connected_three, "positive"),
            (True, [7, 13]),
        )

    def test_disconnected_pairwise_overlaps_do_not_create_selected_anchor(self) -> None:
        disconnected = [
            anchor_row("positive", 0, 2),
            anchor_row("positive", 1, 4),
            anchor_row("positive", 3, 5),
        ]
        self.assertEqual(
            analysis.majority_covered_anchor(disconnected, "positive"),
            (False, None),
        )

    def test_submission_lock_detects_later_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = root / "completed.jsonl"
            lock = root / "lock.json"
            completed.write_text('{"value":1}\n', encoding="utf-8")
            first = analysis.lock_or_verify_submission(
                completed, lock, "human-1", 1, "packet-digest"
            )
            second = analysis.lock_or_verify_submission(
                completed, lock, "human-1", 1, "packet-digest"
            )
            completed.write_text('{"value":2}\n', encoding="utf-8")
            changed = analysis.lock_or_verify_submission(
                completed, lock, "human-1", 1, "packet-digest"
            )
            self.assertTrue(first["lock_created_now"])
            self.assertTrue(first["lock_valid"])
            self.assertFalse(second["lock_created_now"])
            self.assertTrue(second["lock_valid"])
            self.assertFalse(changed["lock_valid"])
            self.assertFalse(changed["comparison_pass"]["completed_file_sha256"])

    def test_condition_types_are_structural_not_free_text(self) -> None:
        base = {
            "semantic_polarity": "conditional",
            "condition_scope_text": ["after cooking"],
            "condition_types": ["processing_required"],
        }
        self.assertTrue(analysis.valid_condition_types(base))
        self.assertFalse(
            analysis.valid_condition_types(
                {**base, "condition_types": ["none", "processing_required"]}
            )
        )
        self.assertFalse(
            analysis.valid_condition_types(
                {
                    "semantic_polarity": "conditional",
                    "condition_scope_text": [],
                    "condition_types": ["none"],
                }
            )
        )
        self.assertFalse(
            analysis.valid_condition_types(
                {
                    "semantic_polarity": "positive",
                    "factuality": "conditional",
                    "condition_scope_text": [],
                    "condition_types": ["none"],
                }
            )
        )

    def test_no_human_input_keeps_every_scientific_gate_closed(self) -> None:
        status = read_json(analysis.STATUS_PATH)
        stage = read_json(analysis.STAGE_PATH)
        self.assertEqual(status["valid_locked_first_pass_main_label_count"], 0)
        self.assertEqual(status["valid_locked_repeat_control_label_count"], 0)
        self.assertEqual(status["workflow_unresolved_item_count"], 288)
        self.assertEqual(status["event_anchor_qualified_item_count"], 0)
        self.assertFalse(status["semantic_gold_complete"])
        self.assertFalse(status["authoritative_factual_gold_complete"])
        self.assertFalse(status["resolved_gold_artifact_written"])
        self.assertEqual(stage["status"], "blocked_pending_external_human_review")
        self.assertFalse(stage["automatic_model_execution_now"])
        self.assertTrue(all(not value for value in stage["authorization"].values()))


if __name__ == "__main__":
    unittest.main()
