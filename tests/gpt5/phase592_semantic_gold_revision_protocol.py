#!/usr/bin/env python3
"""Freeze the pre-review Phase592 semantic-gold and event-anchor revision."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase591_blind_semantic_gold_protocol as phase591  # noqa: E402


PHASE = "Phase592"
SOURCE_QUEUE_PATH = phase591.SOURCE_QUEUE_PATH
OUT_DIR = ROOT / "tests/gpt5/result/phase592_semantic_gold_revision"
REVIEW_DIR = OUT_DIR / "external_review"
PROTOCOL_PATH = OUT_DIR / "phase592_semantic_gold_revision_protocol.json"
AUDIT_PATH = OUT_DIR / "phase592_static_audit.json"

REVIEWER_SLOTS = phase591.REVIEWER_SLOTS
POLARITY_LABELS = phase591.POLARITY_LABELS
FACTUALITY_LABELS = phase591.FACTUALITY_LABELS
EXPECTED_ITEM_COUNT = phase591.EXPECTED_ITEM_COUNT
ITEMS_PER_BATCH = 48
BATCH_COUNT = EXPECTED_ITEM_COUNT // ITEMS_PER_BATCH
CONFIDENCE_MIN = 1
CONFIDENCE_MAX = 5

FACTUALITY_SOURCE_TIERS = (
    "tier1_public_health_toxicology_food_safety_government",
    "tier2_botany_agriculture_medicine_food_science",
    "tier3_professional_academic_encyclopedia",
    "tier4_stable_ordinary_knowledge",
    "tier5_unconfirmed",
)

REVIEW_ATTESTATION = (
    "I reviewed this packet independently without consulting another reviewer's "
    "answers, model identity, semantic group, expected polarity, frozen parser output, "
    "or the private answer key."
)

ADJUDICATOR_ATTESTATION = (
    "I adjudicated independently without seeing the three prior labels, model identity, "
    "semantic group, expected polarity, frozen parser output, or the private answer key."
)

SLOT_SALTS = {
    "reviewer_a": "phase592-independent-review-a-v1",
    "reviewer_b": "phase592-independent-review-b-v1",
    "reviewer_c": "phase592-independent-review-c-v1",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def packet_path(slot: str) -> Path:
    return REVIEW_DIR / f"{slot}_packet_v2.jsonl.gz"


def response_template_path(slot: str) -> Path:
    return REVIEW_DIR / f"{slot}_response_template_v2.jsonl"


def completed_response_path(slot: str) -> Path:
    return REVIEW_DIR / f"{slot}_completed_v2.jsonl"


def adjudicator_completed_path() -> Path:
    return REVIEW_DIR / "adjudicator_completed_v2.jsonl"


def instruction(slot: str) -> dict[str, Any]:
    return {
        "phase_id": PHASE,
        "reviewer_slot": slot,
        "task": (
            "Independently judge what the response finally commits to about ordinary human "
            "food use, then separately judge whether that proposition is factually sound."
        ),
        "semantic_labels": list(POLARITY_LABELS),
        "factuality_labels": list(FACTUALITY_LABELS),
        "factuality_source_tiers": list(FACTUALITY_SOURCE_TIERS),
        "decisive_span_rule": (
            "Use zero-based Unicode code-point offsets and a half-open [start,end) span. "
            "Mark the earliest minimal text span that stably determines the final semantic label; "
            "use null/null for unresolved."
        ),
        "semantic_stability_rule": (
            "Set later_text_changes_final_semantics=true when later text reverses or materially "
            "conditions the label suggested by the marked span."
        ),
        "scope_rule": (
            "Copy exact response substrings governed by negation or a condition; use an empty list "
            "when no such scope exists."
        ),
        "batch_rule": (
            "Complete six fixed 48-item batches in packet order. Record batch start and completion "
            "times. Do not inspect another reviewer or change the ontology between batches."
        ),
        "attestation": REVIEW_ATTESTATION,
    }


def response_template(packet: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "review_id": row["review_id"],
            "reviewer_slot": row["reviewer_slot"],
            "packet_digest": row["packet_digest"],
            "batch_id": row["batch_id"],
            "reviewer_id": None,
            "semantic_polarity": None,
            "negation_scope_text": None,
            "condition_scope_text": None,
            "has_contrast": None,
            "response_complete": None,
            "decisive_span_start": None,
            "decisive_span_end": None,
            "later_text_changes_final_semantics": None,
            "factuality": None,
            "factuality_source_tier": None,
            "factuality_evidence": None,
            "confidence_1_to_5": None,
            "rationale": None,
            "batch_started_at": None,
            "batch_completed_at": None,
            "attestation": REVIEW_ATTESTATION,
            "reviewed_at": None,
        }
        for row in packet
    ]


def old_review_started() -> list[str]:
    return [
        slot
        for slot in REVIEWER_SLOTS
        if phase591.completed_response_path(slot).exists()
    ]


def register() -> dict[str, Any]:
    old_started = old_review_started()
    if old_started:
        raise RuntimeError(
            "Phase592 protocol revision is forbidden after Phase591 human review starts: "
            + ",".join(old_started)
        )
    source_rows = phase591.read_jsonl_gz(SOURCE_QUEUE_PATH)
    source_audit = phase591.validate_source(source_rows)
    if not source_audit["valid"]:
        raise RuntimeError("Phase592 source blind queue failed the Phase591 audit")

    packet_digests: dict[str, str] = {}
    packet_orders: dict[str, list[str]] = {}
    batch_counts: dict[str, dict[str, int]] = {}
    for slot in REVIEWER_SLOTS:
        ordered = sorted(
            source_rows,
            key=lambda row: digest(SLOT_SALTS[slot] + "|" + row["review_id"]),
        )
        rows_with_batches = [
            {**row, "batch_id": f"batch_{index // ITEMS_PER_BATCH + 1:02d}"}
            for index, row in enumerate(ordered)
        ]
        review_instruction = instruction(slot)
        packet_digest = digest(
            canonical_json(review_instruction)
            + "\n"
            + "\n".join(canonical_json(row) for row in rows_with_batches)
        )
        packet = [
            {
                **row,
                "phase_id": PHASE,
                "reviewer_slot": slot,
                "packet_digest": packet_digest,
                "review_instruction": review_instruction,
            }
            for row in rows_with_batches
        ]
        phase591.write_jsonl_gz(packet_path(slot), packet)
        if not completed_response_path(slot).exists():
            phase591.write_jsonl(response_template_path(slot), response_template(packet))
        packet_digests[slot] = packet_digest
        packet_orders[slot] = [row["review_id"] for row in packet]
        batch_counts[slot] = dict(
            sorted(
                {
                    batch_id: sum(row["batch_id"] == batch_id for row in packet)
                    for batch_id in {row["batch_id"] for row in packet}
                }.items()
            )
        )

    audit = {
        "schema_version": "phase592_semantic_gold_revision_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "pre_execution_revision": True,
        "phase591_completed_file_count_before_revision": len(old_started),
        "phase591_completed_slots_before_revision": old_started,
        "source_item_count": len(source_rows),
        "hidden_metadata_key_count": source_audit["hidden_metadata_key_count"],
        "packet_item_count_by_slot": {
            slot: len(packet_orders[slot]) for slot in REVIEWER_SLOTS
        },
        "packet_order_pairwise_distinct": len(
            {tuple(packet_orders[slot]) for slot in REVIEWER_SLOTS}
        )
        == len(REVIEWER_SLOTS),
        "packet_digest_count": len(set(packet_digests.values())),
        "batch_count_by_slot": {
            slot: len(batch_counts[slot]) for slot in REVIEWER_SLOTS
        },
        "batch_item_counts_by_slot": batch_counts,
        "private_answer_key_read": False,
        "phase590_sealed_cases_read": False,
    }
    audit["valid"] = bool(
        audit["phase591_completed_file_count_before_revision"] == 0
        and len(source_rows) == EXPECTED_ITEM_COUNT
        and audit["hidden_metadata_key_count"] == 0
        and audit["packet_order_pairwise_distinct"]
        and audit["packet_digest_count"] == len(REVIEWER_SLOTS)
        and all(
            count == EXPECTED_ITEM_COUNT
            for count in audit["packet_item_count_by_slot"].values()
        )
        and all(count == BATCH_COUNT for count in audit["batch_count_by_slot"].values())
        and all(
            batch_count == ITEMS_PER_BATCH
            for slot_counts in batch_counts.values()
            for batch_count in slot_counts.values()
        )
    )
    phase591.write_json(AUDIT_PATH, audit)
    frozen = {
        "schema_version": "phase592_semantic_gold_revision_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Pre-review semantic-gold and event-anchor gate revision",
        "supersedes_review_packets_from": "Phase591",
        "revision_reason": (
            "Separate semantic resolution from event-anchor qualification and allow preserved, "
            "independent adjudication after all first-pass reviews."
        ),
        "source_queue_path": str(SOURCE_QUEUE_PATH.relative_to(ROOT)),
        "source_queue_sha256": phase591.sha256_file(SOURCE_QUEUE_PATH),
        "source_item_count": len(source_rows),
        "reviewer_slots": list(REVIEWER_SLOTS),
        "required_distinct_reviewer_count": len(REVIEWER_SLOTS),
        "items_per_reviewer": EXPECTED_ITEM_COUNT,
        "items_per_batch": ITEMS_PER_BATCH,
        "batch_count_per_reviewer": BATCH_COUNT,
        "packet_digests": packet_digests,
        "completed_response_paths": {
            slot: str(completed_response_path(slot).relative_to(ROOT))
            for slot in REVIEWER_SLOTS
        },
        "semantic_consensus_rule": {
            "unanimous": "directly_resolved",
            "two_same_nonunresolved_plus_unresolved": "directly_resolved",
            "positive_negative_conflict": "independent_adjudication",
            "conditional_positive_or_negative_conflict": "independent_adjudication",
            "unresolved_majority_with_decisive_minority": "independent_adjudication",
            "raw_reviews_must_never_be_overwritten": True,
        },
        "gold_gate": {
            "three_distinct_reviewers_complete_all_items": True,
            "directly_resolved_plus_independently_adjudicated_must_equal": EXPECTED_ITEM_COUNT,
            "workflow_unresolved_item_count_must_equal": 0,
            "first_pass_zero_disagreement_not_required": True,
        },
        "event_anchor_gate": {
            "separate_from_semantic_gold": True,
            "requires_nonunresolved_direct_semantic_consensus": True,
            "requires_two_complete_response_votes": True,
            "requires_two_no_later_semantic_change_votes": True,
            "requires_overlap_of_two_supporting_half_open_unicode_spans": True,
            "failure_does_not_destroy_semantic_gold": True,
            "minimum_balanced_anchor_subset_not_yet_frozen": True,
        },
        "evidence_policy": {
            "model_or_agent_annotation_cannot_substitute": True,
            "adjudicator_must_be_distinct_from_three_reviewers": True,
            "adjudicator_cannot_see_prior_labels_or_hidden_metadata": True,
            "private_answer_key_must_remain_unread_until_semantic_gold_complete": True,
            "no_model_or_internal_execution_before_semantic_gold_complete": True,
            "phase590_sealed_set_remains_unread": True,
        },
        "static_audit_sha256": phase591.sha256_file(AUDIT_PATH),
    }
    phase591.write_json(PROTOCOL_PATH, frozen)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
