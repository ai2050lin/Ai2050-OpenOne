#!/usr/bin/env python3
"""Freeze Phase593 human-review integrity controls before any annotation starts."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase591_blind_semantic_gold_protocol as io_helpers  # noqa: E402
import phase592_semantic_gold_revision_protocol as phase592  # noqa: E402


PHASE = "Phase593"
SOURCE_QUEUE_PATH = phase592.SOURCE_QUEUE_PATH
OUT_DIR = ROOT / "tests/gpt5/result/phase593_human_semantic_review_integrity"
REVIEW_DIR = OUT_DIR / "external_review"
PROTOCOL_PATH = OUT_DIR / "phase593_human_semantic_review_integrity_protocol.json"
AUDIT_PATH = OUT_DIR / "phase593_static_audit.json"
PRIVATE_MAP_PATH = OUT_DIR / "phase593_private_submission_map.json"

REVIEWER_SLOTS = phase592.REVIEWER_SLOTS
POLARITY_LABELS = phase592.POLARITY_LABELS
FACTUALITY_LABELS = phase592.FACTUALITY_LABELS
FACTUALITY_SOURCE_TIERS = phase592.FACTUALITY_SOURCE_TIERS
CONFIDENCE_MIN = phase592.CONFIDENCE_MIN
CONFIDENCE_MAX = phase592.CONFIDENCE_MAX

MAIN_ITEM_COUNT = phase592.EXPECTED_ITEM_COUNT
BATCH_COUNT = 6
MAIN_ITEMS_PER_BATCH = 48
REPEAT_ITEMS_PER_BATCH = 2
REPEAT_ITEM_COUNT = BATCH_COUNT * REPEAT_ITEMS_PER_BATCH
PACKET_ITEM_COUNT = MAIN_ITEM_COUNT + REPEAT_ITEM_COUNT
PACKET_ITEMS_PER_BATCH = MAIN_ITEMS_PER_BATCH + REPEAT_ITEMS_PER_BATCH

CONDITION_TYPES = (
    "none",
    "part_specific",
    "processing_required",
    "dose_or_quantity_limited",
    "variety_specific",
    "culture_or_use_context",
    "ripeness_or_life_stage",
    "other_explicit_condition",
)

REVIEW_ATTESTATION = (
    "I reviewed this packet independently without consulting another reviewer's "
    "answers, model identity, semantic group, expected polarity, frozen parser output, "
    "the private answer key, or the private repeat-control map."
)

ADJUDICATOR_ATTESTATION = (
    "I adjudicated independently without seeing the three prior labels, model identity, "
    "semantic group, expected polarity, frozen parser output, private answer key, or "
    "repeat-control outcomes."
)

SLOT_SALTS = {
    "reviewer_a": "phase593-independent-review-a-v1",
    "reviewer_b": "phase593-independent-review-b-v1",
    "reviewer_c": "phase593-independent-review-c-v1",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def submission_id(slot: str, canonical_review_id: str, role: str, ordinal: int) -> str:
    value = f"phase593|{slot}|{role}|{ordinal}|{canonical_review_id}"
    return "p593s_" + io_helpers.sha256_text(value)[:20]


def packet_path(slot: str) -> Path:
    return REVIEW_DIR / f"{slot}_packet_v3.jsonl.gz"


def response_template_path(slot: str) -> Path:
    return REVIEW_DIR / f"{slot}_response_template_v3.jsonl"


def completed_response_path(slot: str) -> Path:
    return REVIEW_DIR / f"{slot}_completed_v3.jsonl"


def submission_lock_path(slot: str) -> Path:
    return REVIEW_DIR / f"{slot}_submission_lock_v3.json"


def adjudicator_completed_path() -> Path:
    return REVIEW_DIR / "adjudicator_completed_v3.jsonl"


def adjudicator_lock_path() -> Path:
    return REVIEW_DIR / "adjudicator_submission_lock_v3.json"


def instruction(slot: str) -> dict[str, Any]:
    return {
        "phase_id": PHASE,
        "reviewer_slot": slot,
        "task": (
            "Independently judge the response's final proposition about ordinary human food use, "
            "then separately judge factuality. Repeated consistency items, if any, are not "
            "identified."
        ),
        "semantic_labels": list(POLARITY_LABELS),
        "factuality_labels": list(FACTUALITY_LABELS),
        "condition_types": list(CONDITION_TYPES),
        "condition_type_rule": (
            "Use ['none'] only when condition_scope_text is empty. Otherwise choose one or more "
            "non-none condition types. A conditional semantic or factuality label requires an "
            "explicit non-none type."
        ),
        "factuality_source_tiers": list(FACTUALITY_SOURCE_TIERS),
        "decisive_span_rule": (
            "Use zero-based Unicode code-point offsets and a half-open [start,end) span. Mark the "
            "earliest minimal span that stably determines the final semantic label; use null/null "
            "for unresolved. Later reversal or material qualification must be marked separately."
        ),
        "scope_rule": (
            "Copy exact response substrings governed by negation or a condition; use an empty list "
            "when no such scope exists."
        ),
        "batch_rule": (
            "Complete six fixed 50-item batches in packet order and record one start and completion "
            "time per batch. Each batch contains 48 main items and two blinded repeat controls."
        ),
        "attestation": REVIEW_ATTESTATION,
    }


def response_template(packet: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "submission_id": row["submission_id"],
            "reviewer_slot": row["reviewer_slot"],
            "packet_digest": row["packet_digest"],
            "batch_id": row["batch_id"],
            "reviewer_id": None,
            "semantic_polarity": None,
            "negation_scope_text": None,
            "condition_scope_text": None,
            "condition_types": None,
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


def review_already_started() -> list[str]:
    started = [
        f"phase592:{slot}"
        for slot in REVIEWER_SLOTS
        if phase592.completed_response_path(slot).exists()
    ]
    started.extend(
        f"phase593:{slot}"
        for slot in REVIEWER_SLOTS
        if completed_response_path(slot).exists()
    )
    started.extend(
        f"phase593_lock:{slot}"
        for slot in REVIEWER_SLOTS
        if submission_lock_path(slot).exists()
    )
    return started


def _build_slot(
    source_rows: list[dict[str, Any]], slot: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        source_rows,
        key=lambda row: io_helpers.sha256_text(SLOT_SALTS[slot] + "|" + row["review_id"]),
    )
    packet_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for batch_index in range(BATCH_COUNT):
        batch_id = f"batch_{batch_index + 1:02d}"
        main_chunk = ordered[
            batch_index * MAIN_ITEMS_PER_BATCH : (batch_index + 1) * MAIN_ITEMS_PER_BATCH
        ]
        batch_rows: list[dict[str, Any]] = []
        for item_index, source in enumerate(main_chunk):
            sid = submission_id(
                slot,
                source["review_id"],
                "main",
                batch_index * MAIN_ITEMS_PER_BATCH + item_index,
            )
            batch_rows.append(
                {
                    "schema_version": "phase593_blind_review_item.v1",
                    "phase_id": PHASE,
                    "submission_id": sid,
                    "reviewer_slot": slot,
                    "batch_id": batch_id,
                    "prompt": source["prompt"],
                    "response": source["response"],
                }
            )
            private_rows.append(
                {
                    "reviewer_slot": slot,
                    "submission_id": sid,
                    "canonical_review_id": source["review_id"],
                    "item_role": "main",
                    "batch_id": batch_id,
                }
            )

        repeat_source_batch = (batch_index + 2) % BATCH_COUNT
        repeat_indices = (
            repeat_source_batch * MAIN_ITEMS_PER_BATCH + 11,
            repeat_source_batch * MAIN_ITEMS_PER_BATCH + 35,
        )
        for repeat_index, source_index in enumerate(repeat_indices):
            source = ordered[source_index]
            sid = submission_id(
                slot,
                source["review_id"],
                "repeat",
                batch_index * REPEAT_ITEMS_PER_BATCH + repeat_index,
            )
            batch_rows.append(
                {
                    "schema_version": "phase593_blind_review_item.v1",
                    "phase_id": PHASE,
                    "submission_id": sid,
                    "reviewer_slot": slot,
                    "batch_id": batch_id,
                    "prompt": source["prompt"],
                    "response": source["response"],
                }
            )
            private_rows.append(
                {
                    "reviewer_slot": slot,
                    "submission_id": sid,
                    "canonical_review_id": source["review_id"],
                    "item_role": "repeat_control",
                    "batch_id": batch_id,
                }
            )

        packet_rows.extend(
            sorted(
                batch_rows,
                key=lambda row: io_helpers.sha256_text(
                    SLOT_SALTS[slot] + "|" + batch_id + "|" + row["submission_id"]
                ),
            )
        )
    return packet_rows, private_rows


def register() -> dict[str, Any]:
    started = review_already_started()
    if started:
        raise RuntimeError(
            "Phase593 protocol cannot be regenerated after a v2/v3 review or lock exists: "
            + ",".join(started)
        )

    phase592_audit = json.loads(phase592.AUDIT_PATH.read_text(encoding="utf-8"))
    if not phase592_audit.get("valid"):
        raise RuntimeError("Phase592 source audit is not valid")
    source_rows = io_helpers.read_jsonl_gz(SOURCE_QUEUE_PATH)
    source_audit = io_helpers.validate_source(source_rows)
    if not source_audit["valid"]:
        raise RuntimeError("Phase593 source blind queue failed the Phase591 source audit")

    all_private_rows: list[dict[str, Any]] = []
    packet_digests: dict[str, str] = {}
    packet_orders: dict[str, list[str]] = {}
    batch_counts: dict[str, dict[str, int]] = {}
    for slot in REVIEWER_SLOTS:
        base_packet, private_rows = _build_slot(source_rows, slot)
        review_instruction = instruction(slot)
        packet_digest = io_helpers.sha256_text(
            io_helpers.canonical_json(review_instruction)
            + "\n"
            + "\n".join(io_helpers.canonical_json(row) for row in base_packet)
        )
        packet = [
            {**row, "packet_digest": packet_digest, "review_instruction": review_instruction}
            for row in base_packet
        ]
        io_helpers.write_jsonl_gz(packet_path(slot), packet)
        io_helpers.write_jsonl(response_template_path(slot), response_template(packet))
        all_private_rows.extend(private_rows)
        packet_digests[slot] = packet_digest
        packet_orders[slot] = [row["submission_id"] for row in packet]
        batch_counts[slot] = dict(
            sorted(Counter(row["batch_id"] for row in packet).items())
        )

    private_payload = {
        "schema_version": "phase593_private_submission_map.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "share_with_first_pass_reviewers": False,
        "contains_model_or_expected_label_metadata": False,
        "rows": all_private_rows,
    }
    io_helpers.write_json(PRIVATE_MAP_PATH, private_payload)

    private_main_counts = Counter(
        row["reviewer_slot"] for row in all_private_rows if row["item_role"] == "main"
    )
    private_repeat_counts = Counter(
        row["reviewer_slot"]
        for row in all_private_rows
        if row["item_role"] == "repeat_control"
    )
    exposed_forbidden_keys = {
        "canonical_review_id",
        "item_role",
        "model",
        "case_id",
        "split",
        "semantic_group",
        "expected_polarity",
        "frozen_parser_polarity",
    }
    exposed_forbidden_count = 0
    for slot in REVIEWER_SLOTS:
        exposed_forbidden_count += sum(
            len(exposed_forbidden_keys & set(row))
            for row in io_helpers.read_jsonl_gz(packet_path(slot))
        )

    audit = {
        "schema_version": "phase593_human_semantic_review_integrity_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "pre_review_freeze": True,
        "prior_or_current_completed_artifact_count": len(started),
        "source_item_count": len(source_rows),
        "packet_item_count_by_slot": {
            slot: len(packet_orders[slot]) for slot in REVIEWER_SLOTS
        },
        "main_item_count_by_slot": dict(private_main_counts),
        "repeat_control_count_by_slot": dict(private_repeat_counts),
        "batch_item_counts_by_slot": batch_counts,
        "packet_order_pairwise_distinct": len(
            {tuple(packet_orders[slot]) for slot in REVIEWER_SLOTS}
        )
        == len(REVIEWER_SLOTS),
        "packet_digest_count": len(set(packet_digests.values())),
        "reviewer_packet_forbidden_key_count": exposed_forbidden_count,
        "private_answer_key_read": False,
        "phase590_sealed_cases_read": False,
        "model_case_count_consumed": 0,
        "internal_state_case_count_consumed": 0,
    }
    audit["valid"] = bool(
        audit["prior_or_current_completed_artifact_count"] == 0
        and len(source_rows) == MAIN_ITEM_COUNT
        and all(
            count == PACKET_ITEM_COUNT
            for count in audit["packet_item_count_by_slot"].values()
        )
        and all(count == MAIN_ITEM_COUNT for count in private_main_counts.values())
        and all(count == REPEAT_ITEM_COUNT for count in private_repeat_counts.values())
        and all(
            count == PACKET_ITEMS_PER_BATCH
            for slot_counts in batch_counts.values()
            for count in slot_counts.values()
        )
        and audit["packet_order_pairwise_distinct"]
        and audit["packet_digest_count"] == len(REVIEWER_SLOTS)
        and audit["reviewer_packet_forbidden_key_count"] == 0
    )
    io_helpers.write_json(AUDIT_PATH, audit)

    frozen = {
        "schema_version": "phase593_human_semantic_review_integrity_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Pre-review submission locking, repeat controls, and majority-covered anchors",
        "supersedes_review_packets_from": "Phase592",
        "source_queue_path": str(SOURCE_QUEUE_PATH.relative_to(ROOT)),
        "source_queue_sha256": io_helpers.sha256_file(SOURCE_QUEUE_PATH),
        "main_gold_item_count": MAIN_ITEM_COUNT,
        "repeat_control_item_count_per_reviewer": REPEAT_ITEM_COUNT,
        "repeat_controls_change_gold_denominator": False,
        "packet_item_count_per_reviewer": PACKET_ITEM_COUNT,
        "batch_count_per_reviewer": BATCH_COUNT,
        "main_items_per_batch": MAIN_ITEMS_PER_BATCH,
        "repeat_items_per_batch": REPEAT_ITEMS_PER_BATCH,
        "packet_digests": packet_digests,
        "condition_types": list(CONDITION_TYPES),
        "event_anchor_rule": {
            "name": "contiguous_majority_character_coverage",
            "coverage_threshold": 2,
            "requires_nonempty": True,
            "requires_single_contiguous_interval": True,
            "pair_selection_after_results_forbidden": True,
            "adjudication_cannot_create_anchor": True,
        },
        "repeat_control_rule": {
            "descriptive_count_per_reviewer": REPEAT_ITEM_COUNT,
            "mismatch_does_not_change_main_denominator": True,
            "any_mismatching_main_item_requires_independent_adjudication": True,
            "no_post_result_agreement_threshold": True,
        },
        "submission_lock_rule": {
            "first_structurally_valid_file_is_sha256_locked": True,
            "lock_is_never_overwritten": True,
            "later_digest_change_invalidates_submission": True,
            "aggregation_requires_all_three_valid_locks": True,
        },
        "adjudication_rule": {
            "generated_only_after_three_valid_locked_first_pass_submissions": True,
            "adjudicator_must_be_distinct": True,
            "prior_labels_and_repeat_outcomes_hidden": True,
            "allowed_semantic_outputs": list(POLARITY_LABELS),
            "allowed_factuality_outputs": list(FACTUALITY_LABELS),
            "semantic_unresolved_is_a_resolved_workflow_label_but_not_an_anchor": True,
            "factuality_uncertain_is_a_resolved_workflow_label_not_a_fact_claim": True,
        },
        "factual_reference_policy": {
            "source_tiers_frozen": list(FACTUALITY_SOURCE_TIERS),
            "per_item_evidence_required": True,
            "fixed_object_level_authoritative_registry_present": False,
            "impact": (
                "semantic review may proceed, but factuality cannot be promoted to an authoritative "
                "fact registry without a separately audited source collection"
            ),
        },
        "evidence_policy": {
            "model_or_agent_annotation_cannot_substitute": True,
            "same_workspace_cannot_prove_human_independence": True,
            "private_submission_map_must_not_be_shared_with_reviewers": True,
            "no_model_or_internal_execution_before_semantic_gold_complete": True,
            "private_answer_key_remains_unread": True,
            "phase590_sealed_set_remains_unread": True,
        },
        "static_audit_sha256": io_helpers.sha256_file(AUDIT_PATH),
        "private_submission_map_sha256": io_helpers.sha256_file(PRIVATE_MAP_PATH),
    }
    io_helpers.write_json(PROTOCOL_PATH, frozen)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
