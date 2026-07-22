#!/usr/bin/env python3
"""Freeze the final pre-human amendment for the Phase593 v3 review packets."""

from __future__ import annotations

import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase591_blind_semantic_gold_protocol as io_helpers  # noqa: E402
import phase592_semantic_gold_revision_protocol as phase592  # noqa: E402
import phase593_human_semantic_review_integrity_protocol as phase593  # noqa: E402


PHASE = "Phase594"
OUT_DIR = ROOT / "tests/gpt5/result/phase594_human_review_execution"
REVIEW_DIR = OUT_DIR / "external_review"
PROTOCOL_PATH = OUT_DIR / "phase594_human_review_execution_protocol.json"
AUDIT_PATH = OUT_DIR / "phase594_static_audit.json"
FACT_REGISTRY_TEMPLATE_PATH = OUT_DIR / "phase594_authoritative_fact_registry_template.json"
FACT_REGISTRY_COMPLETED_PATH = (
    OUT_DIR
    / "external_fact_audit/phase594_authoritative_fact_registry_completed.json"
)
FACT_REGISTRY_LOCK_PATH = (
    OUT_DIR / "external_fact_audit/phase594_authoritative_fact_registry_lock.json"
)
PUBLIC_OBJECT_BANK_PATH = (
    ROOT
    / "tests/gpt5/result/phase590_natural_semantic_event/phase590_public_object_bank.json"
)

REVIEWER_SLOTS = phase593.REVIEWER_SLOTS
POLARITY_LABELS = phase593.POLARITY_LABELS
FACTUALITY_LABELS = phase593.FACTUALITY_LABELS
FACTUALITY_SOURCE_TIERS = phase593.FACTUALITY_SOURCE_TIERS
CONDITION_TYPES = phase593.CONDITION_TYPES
CONFIDENCE_MIN = phase593.CONFIDENCE_MIN
CONFIDENCE_MAX = phase593.CONFIDENCE_MAX
MAIN_ITEM_COUNT = phase593.MAIN_ITEM_COUNT
REPEAT_ITEM_COUNT = phase593.REPEAT_ITEM_COUNT
PACKET_ITEM_COUNT = phase593.PACKET_ITEM_COUNT
BATCH_COUNT = phase593.BATCH_COUNT
PACKET_ITEMS_PER_BATCH = phase593.PACKET_ITEMS_PER_BATCH

MAX_SUBSTANTIVE_REPEAT_MISMATCHES = 2
MIN_SUBSTANTIVE_REPEAT_AGREEMENTS = REPEAT_ITEM_COUNT - MAX_SUBSTANTIVE_REPEAT_MISMATCHES

FACT_CLAIM_POLARITIES = ("positive", "negative", "conditional", "uncertain")
FACT_SOURCE_EVIDENCE_RELATIONS = ("supports", "refutes", "qualifies", "mixed")
FACT_SOURCE_INDEPENDENCE = ("independent", "not_independent", "unknown")
FACT_OBJECT_AUDIT_STATUSES = (
    "completed_with_claims",
    "completed_no_required_claims",
    "uncertain",
)
FACT_REVIEW_DISPOSITIONS = (
    "claims_sufficient",
    "insufficient_external_evidence",
    "not_fact_checkable",
)
FACT_PROPOSITION_STATUSES = (
    "fact_checkable",
    "insufficient_external_evidence",
    "not_fact_checkable",
)
FACT_CLAIM_COVERAGE_RELATIONS = (
    "supports",
    "refutes",
    "qualifies",
    "insufficient_evidence",
)

SUBSTANTIVE_REPEAT_FIELDS = (
    "semantic_polarity",
    "factuality",
    "condition_types",
    "response_complete",
    "later_text_changes_final_semantics",
)
ANCHOR_REPEAT_FIELDS = (
    "decisive_span_start",
    "decisive_span_end",
)

ADJUDICATOR_ATTESTATION = (
    "I adjudicated independently without seeing the three prior labels, model identity, "
    "semantic group, expected polarity, frozen parser output, private answer key, repeat-control "
    "outcomes, or the authoritative fact-registry draft."
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def adjudicator_completed_path() -> Path:
    return REVIEW_DIR / "adjudicator_completed_v4.jsonl"


def adjudicator_lock_path() -> Path:
    return REVIEW_DIR / "adjudicator_submission_lock_v4.json"


def prior_review_started() -> list[str]:
    started = [
        f"phase592:{slot}"
        for slot in REVIEWER_SLOTS
        if phase592.completed_response_path(slot).exists()
    ]
    started.extend(
        f"phase593:{slot}"
        for slot in REVIEWER_SLOTS
        if phase593.completed_response_path(slot).exists()
    )
    started.extend(
        f"phase593_lock:{slot}"
        for slot in REVIEWER_SLOTS
        if phase593.submission_lock_path(slot).exists()
    )
    if phase593.adjudicator_completed_path().exists():
        started.append("phase593:adjudicator")
    if phase593.adjudicator_lock_path().exists():
        started.append("phase593_lock:adjudicator")
    return started


def build_fact_registry_template() -> dict[str, Any]:
    public_bank = json.loads(PUBLIC_OBJECT_BANK_PATH.read_text(encoding="utf-8"))
    objects = sorted(
        {
            object_name
            for group_objects in public_bank["objects_by_group"].values()
            for object_name in group_objects
        }
    )
    payload = {
        "schema_version": "phase594_authoritative_fact_registry_template.v3",
        "phase_id": PHASE,
        "created_at": now(),
        "source_public_object_bank": str(PUBLIC_OBJECT_BANK_PATH.relative_to(ROOT)),
        "contains_expected_polarity_or_group": False,
        "registry_complete": False,
        "completion_requires_external_human_source_audit": True,
        "one_object_can_have_multiple_fact_claims": True,
        "claim_record_schema": {
            "required_claim_fields": [
                "claim_id",
                "relation",
                "value_status",
                "polarity",
                "subject_part",
                "processing_condition",
                "dose_or_quantity_condition",
                "variety_condition",
                "ripeness_or_life_stage",
                "culture_or_use_context",
                "temporal_scope",
                "geographic_scope",
                "risk_or_exception",
                "dispute_status",
                "sources",
                "confidence_1_to_5",
                "supersedes_claim_ids",
                "auditor_id",
                "audit_rationale",
                "audited_at",
            ],
            "required_source_fields": [
                "source_tier",
                "source_title",
                "source_locator",
                "source_version_or_date",
                "source_accessed_at",
                "evidence_relation",
                "source_independence",
            ],
            "allowed_polarities": list(FACT_CLAIM_POLARITIES),
            "field_semantics": {
                "value_status": "normalized content value asserted by the claim",
                "polarity": "polarity of the claim itself, not source support direction",
                "source_evidence_relation": (
                    "each source's support, refutation, qualification, or mixed relation "
                    "to the claim"
                ),
            },
            "allowed_source_evidence_relations": list(
                FACT_SOURCE_EVIDENCE_RELATIONS
            ),
            "allowed_source_independence": list(FACT_SOURCE_INDEPENDENCE),
            "allowed_object_audit_statuses": list(FACT_OBJECT_AUDIT_STATUSES),
            "claims_must_be_independently_sourced": True,
            "uncertain_or_disputed_claims_must_remain_explicit": True,
        },
        "review_disposition_schema": {
            "required_fields": [
                "review_id",
                "disposition",
                "claim_ids",
                "propositions",
                "auditor_id",
                "audit_rationale",
                "audited_at",
            ],
            "allowed_dispositions": list(FACT_REVIEW_DISPOSITIONS),
            "all_288_review_ids_must_have_one_disposition": True,
            "claims_sufficient_requires_nonempty_claim_ids": True,
            "claims_sufficient_requires_every_proposition_covered": True,
        },
        "proposition_schema": {
            "required_fields": [
                "proposition_id",
                "text_span_start",
                "text_span_end",
                "proposition_text",
                "fact_checkability",
                "claim_links",
            ],
            "offset_unit": "zero_based_nfc_unicode_code_point_half_open_interval",
            "allowed_fact_checkability": list(FACT_PROPOSITION_STATUSES),
            "allowed_claim_coverage_relations": list(
                FACT_CLAIM_COVERAGE_RELATIONS
            ),
            "fact_checkable_requires_evidentiary_claim_link": True,
            "claim_links_must_match_answer_level_claim_ids": True,
        },
        "supersedes_policy": {
            "self_reference_forbidden": True,
            "unknown_claim_reference_forbidden": True,
            "directed_cycles_forbidden": True,
        },
        "lock_policy": {
            "completed_file_digest_basis": "exact_file_bytes",
            "template_digest_basis": "exact_file_bytes",
            "metadata_fields_compared_separately_not_concatenated": True,
            "any_reordering_or_byte_change_after_lock_invalidates_submission": True,
        },
        "completed_registry_path": str(
            FACT_REGISTRY_COMPLETED_PATH.relative_to(ROOT)
        ),
        "completed_registry_is_separate_from_template": True,
        "records": [
            {
                "object": object_name,
                "audit_status": "not_started",
                "claims": [],
            }
            for object_name in objects
        ],
        "review_dispositions": [],
    }
    io_helpers.write_json(FACT_REGISTRY_TEMPLATE_PATH, payload)
    return payload


def register() -> dict[str, Any]:
    started = prior_review_started()
    if started:
        raise RuntimeError(
            "Phase594 amendment is forbidden after any Phase592/593 review or lock starts: "
            + ",".join(started)
        )

    phase593_audit = json.loads(phase593.AUDIT_PATH.read_text(encoding="utf-8"))
    phase593_manifest = json.loads(
        phase593.PROTOCOL_PATH.read_text(encoding="utf-8")
    )
    if not phase593_audit.get("valid"):
        raise RuntimeError("Phase593 static audit is not valid")

    non_nfc_field_count = 0
    packet_item_counts: dict[str, int] = {}
    packet_digest_mismatch_count = 0
    packet_file_sha256: dict[str, str] = {}
    response_template_file_sha256: dict[str, str] = {}
    for slot in REVIEWER_SLOTS:
        packet_path = phase593.packet_path(slot)
        template_path = phase593.response_template_path(slot)
        packet_rows = io_helpers.read_jsonl_gz(packet_path)
        packet_item_counts[slot] = len(packet_rows)
        packet_file_sha256[slot] = io_helpers.sha256_file(packet_path)
        response_template_file_sha256[slot] = io_helpers.sha256_file(template_path)
        for row in packet_rows:
            packet_digest_mismatch_count += int(
                row.get("packet_digest") != phase593_manifest["packet_digests"][slot]
            )
            for field in ("prompt", "response"):
                text = row[field]
                non_nfc_field_count += int(unicodedata.normalize("NFC", text) != text)

    fact_registry = build_fact_registry_template()
    fact_registry_object_count = len(fact_registry["records"])
    duplicate_fact_object_count = fact_registry_object_count - len(
        {row["object"] for row in fact_registry["records"]}
    )
    fact_registry_prefilled_claim_count = sum(
        len(row.get("claims", [])) for row in fact_registry["records"]
    )
    fact_registry_invalid_claim_container_count = sum(
        not isinstance(row.get("claims"), list) for row in fact_registry["records"]
    )

    audit = {
        "schema_version": "phase594_human_review_execution_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "pre_review_amendment": True,
        "prior_review_or_lock_artifact_count": len(started),
        "phase593_static_audit_valid": phase593_audit["valid"],
        "packet_item_count_by_slot": packet_item_counts,
        "packet_digest_mismatch_count": packet_digest_mismatch_count,
        "non_nfc_prompt_or_response_field_count": non_nfc_field_count,
        "fact_registry_template_object_count": fact_registry_object_count,
        "fact_registry_duplicate_object_count": duplicate_fact_object_count,
        "fact_registry_prefilled_claim_count": fact_registry_prefilled_claim_count,
        "fact_registry_invalid_claim_container_count": (
            fact_registry_invalid_claim_container_count
        ),
        "fact_registry_one_to_many_claim_structure": fact_registry[
            "one_object_can_have_multiple_fact_claims"
        ],
        "fact_registry_prefilled_review_disposition_count": len(
            fact_registry["review_dispositions"]
        ),
        "fact_registry_complete": False,
        "private_answer_key_read": False,
        "phase590_sealed_cases_read": False,
        "model_case_count_consumed": 0,
        "internal_state_case_count_consumed": 0,
    }
    audit["valid"] = bool(
        audit["prior_review_or_lock_artifact_count"] == 0
        and audit["phase593_static_audit_valid"]
        and all(count == PACKET_ITEM_COUNT for count in packet_item_counts.values())
        and audit["packet_digest_mismatch_count"] == 0
        and audit["non_nfc_prompt_or_response_field_count"] == 0
        and audit["fact_registry_template_object_count"] == 96
        and audit["fact_registry_duplicate_object_count"] == 0
        and audit["fact_registry_prefilled_claim_count"] == 0
        and audit["fact_registry_invalid_claim_container_count"] == 0
        and audit["fact_registry_one_to_many_claim_structure"]
        and audit["fact_registry_prefilled_review_disposition_count"] == 0
    )
    io_helpers.write_json(AUDIT_PATH, audit)

    frozen = {
        "schema_version": "phase594_human_review_execution_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Final pre-human quality amendment for Phase593 v3 review packets",
        "amends_protocol": str(phase593.PROTOCOL_PATH.relative_to(ROOT)),
        "reuses_phase593_v3_packets_without_content_change": True,
        "phase593_packet_digests": phase593_manifest["packet_digests"],
        "phase593_packet_file_sha256": packet_file_sha256,
        "phase593_response_template_file_sha256": response_template_file_sha256,
        "unicode_policy": {
            "required_normalization": "NFC",
            "all_existing_prompt_and_response_fields_pass": non_nfc_field_count == 0,
            "offset_unit": "zero_based_unicode_code_point_half_open_interval",
            "future_token_mapping_must_be_separate_and_tokenizer_specific": True,
        },
        "event_anchor_policy": {
            "majority_coverage_rule_from_phase593_retained": True,
            "must_be_nonempty_and_contiguous": True,
            "must_contain_non_whitespace_non_punctuation_character": True,
            "pure_whitespace_or_punctuation_overlap_rejected": True,
            "surface_anchor_is_not_internal_decision_time": True,
        },
        "repeat_control_policy": {
            "substantive_fields": list(SUBSTANTIVE_REPEAT_FIELDS),
            "anchor_fields": list(ANCHOR_REPEAT_FIELDS),
            "substantive_and_anchor_ledgers_separate": True,
            "substantive_mismatch_forces_item_adjudication": True,
            "anchor_mismatch_only_removes_event_anchor_qualification": True,
            "anchor_mismatch_does_not_destroy_semantic_label": True,
        },
        "reviewer_quality_policy": {
            "repeat_control_count": REPEAT_ITEM_COUNT,
            "minimum_substantive_repeat_agreements": MIN_SUBSTANTIVE_REPEAT_AGREEMENTS,
            "maximum_substantive_repeat_mismatches": MAX_SUBSTANTIVE_REPEAT_MISMATCHES,
            "three_or_more_substantive_mismatches_invalidate_entire_submission": True,
            "invalid_submission_requires_full_re_review_or_replacement_reviewer": True,
            "anchor_repeat_mismatches_do_not_invalidate_reviewer": True,
            "threshold_is_minimum_integrity_gate_not_accuracy_proof": True,
        },
        "time_policy": {
            "timezone_aware_iso8601_required": True,
            "batch_duration_must_be_strictly_positive": True,
            "batches_must_not_overlap_and_must_follow_packet_order": True,
            "no_minimum_minutes_claimed": True,
            "time_records_do_not_prove_attention_or_identity": True,
        },
        "artifact_policy": {
            "first_structurally_and_quality_valid_submission_is_digest_locked": True,
            "completed_and_lock_files_become_read_only_after_lock": True,
            "all_three_valid_locks_required_before_aggregation": True,
            "raw_first_pass_and_adjudication_files_never_overwritten": True,
        },
        "gold_artifact_policy": {
            "semantic_gold_is_separate": True,
            "reviewed_factuality_gold_is_separate": True,
            "event_anchor_subset_is_separate": True,
            "reviewed_factuality_is_not_authoritative_fact_registry": True,
        },
        "fact_registry_policy": {
            "template_path": str(FACT_REGISTRY_TEMPLATE_PATH.relative_to(ROOT)),
            "template_object_count": fact_registry_object_count,
            "template_contains_expected_polarity_or_group": False,
            "one_object_can_have_multiple_fact_claims": True,
            "completed_registry_is_separate_from_template": True,
            "completed_registry_path": str(
                FACT_REGISTRY_COMPLETED_PATH.relative_to(ROOT)
            ),
            "completed_registry_lock_path": str(
                FACT_REGISTRY_LOCK_PATH.relative_to(ROOT)
            ),
            "registry_complete": False,
            "must_be_completed_by_external_source_audit": True,
            "current_model_or_agent_must_not_prefill_expected_truth": True,
        },
        "execution_policy": {
            "three_distinct_external_people_required": True,
            "same_agent_multiple_passes_forbidden": True,
            "no_model_generation_observer_internal_trace_or_causal_test": True,
            "private_answer_key_remains_unread": True,
            "phase590_confirmation_heldout_and_sealed_data_remain_unread": True,
        },
        "static_audit_sha256": io_helpers.sha256_file(AUDIT_PATH),
        "fact_registry_template_sha256": io_helpers.sha256_file(
            FACT_REGISTRY_TEMPLATE_PATH
        ),
    }
    io_helpers.write_json(PROTOCOL_PATH, frozen)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
