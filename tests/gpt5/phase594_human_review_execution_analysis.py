#!/usr/bin/env python3
"""Apply the frozen Phase594 quality rules to Phase593 v3 human submissions."""

from __future__ import annotations

import gzip
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase591_blind_semantic_gold_protocol as io_helpers  # noqa: E402
import phase593_human_semantic_review_integrity_analysis as phase593_analysis  # noqa: E402
import phase593_human_semantic_review_integrity_protocol as phase593  # noqa: E402
import phase594_human_review_execution_protocol as protocol  # noqa: E402


STATUS_PATH = protocol.OUT_DIR / "phase594_external_review_status.json"
STAGE_PATH = protocol.OUT_DIR / "phase594_stage_summary.json"
ADJUDICATION_PACKET_PATH = protocol.REVIEW_DIR / "adjudicator_packet_v4.jsonl.gz"
ADJUDICATION_TEMPLATE_PATH = protocol.REVIEW_DIR / "adjudicator_response_template_v4.jsonl"
SEMANTIC_GOLD_PATH = protocol.OUT_DIR / "phase594_semantic_gold.jsonl.gz"
FACTUALITY_GOLD_PATH = protocol.OUT_DIR / "phase594_reviewed_factuality_gold.jsonl.gz"
ANCHOR_GOLD_PATH = protocol.OUT_DIR / "phase594_event_anchor_subset.jsonl.gz"
PROVENANCE_PATH = protocol.OUT_DIR / "phase594_resolution_provenance.jsonl.gz"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_aware_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def lock_or_verify_fact_registry(
    auditor_ids: list[str], claim_count: int, review_disposition_count: int
) -> dict[str, Any]:
    completed_path = protocol.FACT_REGISTRY_COMPLETED_PATH
    lock_path = protocol.FACT_REGISTRY_LOCK_PATH
    file_digest = io_helpers.sha256_file(completed_path)
    template_digest = io_helpers.sha256_file(protocol.FACT_REGISTRY_TEMPLATE_PATH)
    expected = {
        "schema_version": "phase594_fact_registry_lock.v1",
        "phase_id": protocol.PHASE,
        "auditor_ids": sorted(auditor_ids),
        "claim_count": claim_count,
        "review_disposition_count": review_disposition_count,
        "template_sha256": template_digest,
        "completed_file_sha256": file_digest,
    }
    if lock_path.exists():
        existing = read_json(lock_path)
        comparison_pass = {
            key: existing.get(key) == value
            for key, value in expected.items()
            if key not in {"schema_version", "phase_id"}
        }
        return {
            "lock_present": True,
            "lock_created_now": False,
            "lock_valid": all(comparison_pass.values()),
            "completed_file_sha256": file_digest,
            "comparison_pass": comparison_pass,
        }

    lock_payload = {**expected, "locked_at": now(), "immutable_after_creation": True}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    lock_payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    except FileExistsError:
        return lock_or_verify_fact_registry(
            auditor_ids, claim_count, review_disposition_count
        )
    return {
        "lock_present": True,
        "lock_created_now": True,
        "lock_valid": True,
        "completed_file_sha256": file_digest,
        "comparison_pass": {
            "auditor_ids": True,
            "claim_count": True,
            "review_disposition_count": True,
            "template_sha256": True,
            "completed_file_sha256": True,
        },
    }


def validate_fact_registry() -> dict[str, Any]:
    path = protocol.FACT_REGISTRY_COMPLETED_PATH
    if not path.exists():
        return {
            "completed_file_present": False,
            "structurally_valid": False,
            "submission_lock_valid": False,
            "registry_complete": False,
            "object_count": 0,
            "claim_count": 0,
            "review_disposition_count": 0,
            "missing_review_disposition_count": protocol.MAIN_ITEM_COUNT,
            "structural_error_count": 0,
        }

    try:
        registry = read_json(path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {
            "completed_file_present": True,
            "structurally_valid": False,
            "submission_lock_valid": False,
            "registry_complete": False,
            "object_count": 0,
            "claim_count": 0,
            "review_disposition_count": 0,
            "missing_review_disposition_count": protocol.MAIN_ITEM_COUNT,
            "structural_error_count": 1,
        }

    template = read_json(protocol.FACT_REGISTRY_TEMPLATE_PATH)
    expected_objects = {row["object"] for row in template["records"]}
    source_rows = io_helpers.read_jsonl_gz(phase593.SOURCE_QUEUE_PATH)
    source_by_review_id = {row["review_id"]: row for row in source_rows}
    source_review_ids = set(source_by_review_id)
    errors: Counter[str] = Counter()
    errors["schema_version"] += int(
        registry.get("schema_version")
        != "phase594_authoritative_fact_registry_completed.v1"
    )
    errors["template_digest"] += int(
        registry.get("source_template_sha256")
        != io_helpers.sha256_file(protocol.FACT_REGISTRY_TEMPLATE_PATH)
    )
    errors["registry_complete_flag"] += int(
        registry.get("registry_complete") is not True
    )
    errors["expected_label_leak"] += int(
        registry.get("contains_expected_polarity_or_group") is not False
    )

    records = registry.get("records")
    if not isinstance(records, list):
        records = []
        errors["records_container"] += 1
    object_names = [row.get("object") for row in records if isinstance(row, dict)]
    errors["object_count"] += int(len(records) != len(expected_objects))
    errors["object_set"] += int(set(object_names) != expected_objects)
    errors["duplicate_objects"] += len(object_names) - len(set(object_names))

    claim_ids: list[str] = []
    claims: list[dict[str, Any]] = []
    auditor_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            errors["record_type"] += 1
            continue
        record_claims = record.get("claims")
        if not isinstance(record_claims, list):
            errors["claims_container"] += 1
            continue
        audit_status = record.get("audit_status")
        errors["object_audit_status"] += int(
            audit_status not in protocol.FACT_OBJECT_AUDIT_STATUSES
        )
        errors["object_claim_status_mismatch"] += int(
            audit_status == "completed_with_claims" and not record_claims
        )
        errors["object_empty_status_mismatch"] += int(
            audit_status == "completed_no_required_claims" and bool(record_claims)
        )
        for claim in record_claims:
            if not isinstance(claim, dict):
                errors["claim_type"] += 1
                continue
            claims.append(claim)
            claim_id = claim.get("claim_id")
            if not isinstance(claim_id, str) or not claim_id.strip():
                errors["claim_id"] += 1
            else:
                claim_ids.append(claim_id)
            for field in ("relation", "value_status", "subject_part"):
                errors[f"claim_{field}"] += int(
                    not isinstance(claim.get(field), str)
                    or not claim.get(field).strip()
                )
            errors["claim_polarity"] += int(
                claim.get("polarity") not in protocol.FACT_CLAIM_POLARITIES
            )
            errors["claim_dispute_status"] += int(
                not isinstance(claim.get("dispute_status"), str)
                or not claim.get("dispute_status").strip()
            )
            confidence = claim.get("confidence_1_to_5")
            errors["claim_confidence"] += int(
                not isinstance(confidence, int)
                or isinstance(confidence, bool)
                or not 1 <= confidence <= 5
            )
            supersedes = claim.get("supersedes_claim_ids")
            if not isinstance(supersedes, list):
                errors["claim_supersedes"] += 1
            else:
                valid_supersedes = [
                    value for value in supersedes if isinstance(value, str)
                ]
                errors["claim_supersedes_type"] += len(supersedes) - len(
                    valid_supersedes
                )
                errors["claim_supersedes_duplicate"] += len(valid_supersedes) - len(
                    set(valid_supersedes)
                )
            auditor_id = claim.get("auditor_id")
            errors["claim_auditor"] += int(
                not isinstance(auditor_id, str) or not auditor_id.strip()
            )
            if isinstance(auditor_id, str) and auditor_id.strip():
                auditor_ids.add(auditor_id)
            errors["claim_rationale"] += int(
                not isinstance(claim.get("audit_rationale"), str)
                or not claim.get("audit_rationale").strip()
            )
            errors["claim_audited_at"] += int(
                parse_aware_time(claim.get("audited_at")) is None
            )
            sources = claim.get("sources")
            if not isinstance(sources, list) or not sources:
                errors["claim_sources"] += 1
                continue
            for source in sources:
                if not isinstance(source, dict):
                    errors["source_type"] += 1
                    continue
                for field in (
                    "source_tier",
                    "source_title",
                    "source_locator",
                    "source_version_or_date",
                ):
                    errors[f"source_{field}"] += int(
                        not isinstance(source.get(field), str)
                        or not source.get(field).strip()
                    )
                errors["source_tier_value"] += int(
                    source.get("source_tier") not in protocol.FACTUALITY_SOURCE_TIERS
                )
                errors["source_accessed_at"] += int(
                    parse_aware_time(source.get("source_accessed_at")) is None
                )
                errors["source_evidence_relation"] += int(
                    source.get("evidence_relation")
                    not in protocol.FACT_SOURCE_EVIDENCE_RELATIONS
                )
                errors["source_independence"] += int(
                    source.get("source_independence")
                    not in protocol.FACT_SOURCE_INDEPENDENCE
                )

    errors["duplicate_claim_ids"] += len(claim_ids) - len(set(claim_ids))
    claim_id_set = set(claim_ids)
    supersedes_graph: dict[str, list[str]] = {claim_id: [] for claim_id in claim_id_set}
    for claim in claims:
        claim_id = claim.get("claim_id")
        supersedes = claim.get("supersedes_claim_ids")
        if isinstance(supersedes, list):
            errors["unknown_superseded_claim"] += sum(
                isinstance(superseded_id, str)
                and superseded_id not in claim_id_set
                for superseded_id in supersedes
            )
            errors["self_superseded_claim"] += sum(
                superseded_id == claim_id for superseded_id in supersedes
            )
            if isinstance(claim_id, str) and claim_id in supersedes_graph:
                supersedes_graph[claim_id] = [
                    superseded_id
                    for superseded_id in supersedes
                    if superseded_id in claim_id_set
                ]

    visit_state: dict[str, int] = {}

    def visit_supersedes(claim_id: str) -> bool:
        state = visit_state.get(claim_id, 0)
        if state == 1:
            return True
        if state == 2:
            return False
        visit_state[claim_id] = 1
        if any(visit_supersedes(target) for target in supersedes_graph[claim_id]):
            return True
        visit_state[claim_id] = 2
        return False

    errors["supersedes_cycle"] += int(
        any(visit_supersedes(claim_id) for claim_id in sorted(claim_id_set))
    )

    dispositions = registry.get("review_dispositions")
    if not isinstance(dispositions, list):
        dispositions = []
        errors["review_dispositions_container"] += 1
    disposition_ids: list[str] = []
    proposition_ids: list[str] = []
    proposition_count = 0
    for disposition in dispositions:
        if not isinstance(disposition, dict):
            errors["review_disposition_type"] += 1
            continue
        review_id = disposition.get("review_id")
        if isinstance(review_id, str):
            disposition_ids.append(review_id)
        errors["review_id"] += int(review_id not in source_review_ids)
        disposition_value = disposition.get("disposition")
        errors["review_disposition_value"] += int(
            disposition_value not in protocol.FACT_REVIEW_DISPOSITIONS
        )
        linked_claim_ids = disposition.get("claim_ids")
        if not isinstance(linked_claim_ids, list):
            errors["review_claim_ids"] += 1
            linked_claim_ids = []
        else:
            valid_linked_claim_ids = [
                claim_id
                for claim_id in linked_claim_ids
                if isinstance(claim_id, str)
            ]
            errors["review_claim_id_type"] += len(linked_claim_ids) - len(
                valid_linked_claim_ids
            )
            errors["duplicate_review_claim_ids"] += len(
                valid_linked_claim_ids
            ) - len(
                set(valid_linked_claim_ids)
            )
            errors["unknown_review_claim_ids"] += sum(
                claim_id not in claim_id_set for claim_id in valid_linked_claim_ids
            )
            linked_claim_ids = valid_linked_claim_ids
        errors["sufficient_without_claim"] += int(
            disposition_value == "claims_sufficient" and not linked_claim_ids
        )
        propositions = disposition.get("propositions")
        if not isinstance(propositions, list) or not propositions:
            errors["review_propositions"] += 1
            propositions = []
        covered_claim_ids: set[str] = set()
        proposition_statuses: list[str] = []
        for proposition in propositions:
            proposition_count += 1
            if not isinstance(proposition, dict):
                errors["proposition_type"] += 1
                continue
            proposition_id = proposition.get("proposition_id")
            if not isinstance(proposition_id, str) or not proposition_id.strip():
                errors["proposition_id"] += 1
            else:
                proposition_ids.append(proposition_id)
            start = proposition.get("text_span_start")
            end = proposition.get("text_span_end")
            source = source_by_review_id.get(review_id)
            valid_span = bool(
                source is not None
                and isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and 0 <= start < end <= len(source["response"])
            )
            errors["proposition_span"] += int(not valid_span)
            errors["proposition_text"] += int(
                not valid_span
                or proposition.get("proposition_text")
                != source["response"][start:end]
            )
            fact_checkability = proposition.get("fact_checkability")
            proposition_statuses.append(fact_checkability)
            errors["proposition_fact_checkability"] += int(
                fact_checkability not in protocol.FACT_PROPOSITION_STATUSES
            )
            claim_links = proposition.get("claim_links")
            if not isinstance(claim_links, list):
                errors["proposition_claim_links"] += 1
                claim_links = []
            evidentiary_link_present = False
            for link in claim_links:
                if not isinstance(link, dict):
                    errors["proposition_claim_link_type"] += 1
                    continue
                linked_claim_id = link.get("claim_id")
                coverage_relation = link.get("coverage_relation")
                errors["unknown_proposition_claim_id"] += int(
                    not isinstance(linked_claim_id, str)
                    or linked_claim_id not in claim_id_set
                )
                errors["claim_coverage_relation"] += int(
                    coverage_relation
                    not in protocol.FACT_CLAIM_COVERAGE_RELATIONS
                )
                if isinstance(linked_claim_id, str) and linked_claim_id in claim_id_set:
                    covered_claim_ids.add(linked_claim_id)
                evidentiary_link_present = bool(
                    evidentiary_link_present
                    or coverage_relation in {"supports", "refutes", "qualifies"}
                )
            errors["fact_checkable_without_evidence"] += int(
                fact_checkability == "fact_checkable"
                and not evidentiary_link_present
            )
            errors["not_fact_checkable_with_claim_link"] += int(
                fact_checkability == "not_fact_checkable" and bool(claim_links)
            )
        errors["answer_claim_link_mismatch"] += int(
            set(linked_claim_ids) != covered_claim_ids
        )
        errors["sufficient_without_full_proposition_coverage"] += int(
            disposition_value == "claims_sufficient"
            and any(status != "fact_checkable" for status in proposition_statuses)
        )
        errors["insufficient_without_matching_proposition"] += int(
            disposition_value == "insufficient_external_evidence"
            and "insufficient_external_evidence" not in proposition_statuses
        )
        errors["not_fact_checkable_disposition_mismatch"] += int(
            disposition_value == "not_fact_checkable"
            and any(status != "not_fact_checkable" for status in proposition_statuses)
        )
        auditor_id = disposition.get("auditor_id")
        errors["review_auditor"] += int(
            not isinstance(auditor_id, str) or not auditor_id.strip()
        )
        if isinstance(auditor_id, str) and auditor_id.strip():
            auditor_ids.add(auditor_id)
        errors["review_rationale"] += int(
            not isinstance(disposition.get("audit_rationale"), str)
            or not disposition.get("audit_rationale").strip()
        )
        errors["review_audited_at"] += int(
            parse_aware_time(disposition.get("audited_at")) is None
        )

    errors["duplicate_review_dispositions"] += len(disposition_ids) - len(
        set(disposition_ids)
    )
    errors["duplicate_proposition_ids"] += len(proposition_ids) - len(
        set(proposition_ids)
    )
    missing_review_ids = source_review_ids - set(disposition_ids)
    extra_review_ids = set(disposition_ids) - source_review_ids
    errors["missing_review_dispositions"] += len(missing_review_ids)
    errors["extra_review_dispositions"] += len(extra_review_ids)
    errors = Counter({key: value for key, value in errors.items() if value})
    structurally_valid = not errors
    lock_result: dict[str, Any] = {"lock_present": False, "lock_valid": False}
    if structurally_valid:
        lock_result = lock_or_verify_fact_registry(
            sorted(auditor_ids), len(claims), len(dispositions)
        )
        if lock_result["lock_valid"]:
            os.chmod(path, 0o444)
            os.chmod(protocol.FACT_REGISTRY_LOCK_PATH, 0o444)
    return {
        "completed_file_present": True,
        "structurally_valid": structurally_valid,
        "submission_lock_valid": lock_result["lock_valid"],
        "submission_lock": lock_result,
        "registry_complete": bool(structurally_valid and lock_result["lock_valid"]),
        "object_count": len(records),
        "claim_count": len(claims),
        "review_disposition_count": len(dispositions),
        "proposition_count": proposition_count,
        "missing_review_disposition_count": len(missing_review_ids),
        "structural_error_count": sum(errors.values()),
        "structural_error_counts": dict(sorted(errors.items())),
        "auditor_ids": sorted(auditor_ids),
    }


def anchor_has_semantic_content(response: str, span: list[int] | None) -> bool:
    if span is None or len(span) != 2:
        return False
    start, end = span
    if not (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start < end <= len(response)
    ):
        return False
    return any(
        not character.isspace()
        and not unicodedata.category(character).startswith("P")
        for character in response[start:end]
    )


def validate_completed_structure(
    slot: str,
    manifest: dict[str, Any],
    packet_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = phase593.completed_response_path(slot)
    if not path.exists():
        return (
            {
                "reviewer_slot": slot,
                "completed_file_present": False,
                "structurally_valid": False,
                "reviewed_packet_item_count": 0,
                "structural_error_count": protocol.PACKET_ITEM_COUNT,
                "reviewer_id": None,
                "batch_time_policy_pass": False,
            },
            {},
        )
    try:
        rows = read_jsonl(path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return (
            {
                "reviewer_slot": slot,
                "completed_file_present": True,
                "structurally_valid": False,
                "reviewed_packet_item_count": 0,
                "structural_error_count": protocol.PACKET_ITEM_COUNT,
                "reviewer_id": None,
                "batch_time_policy_pass": False,
            },
            {},
        )

    ids = [row.get("submission_id") for row in rows]
    reviewer_ids = {row.get("reviewer_id") for row in rows}
    errors = 0
    accepted: dict[str, dict[str, Any]] = {}
    for row in rows:
        submission_id = row.get("submission_id")
        packet = packet_by_id.get(submission_id)
        confidence = row.get("confidence_1_to_5")
        errors += int(packet is None)
        errors += int(row.get("reviewer_slot") != slot)
        errors += int(row.get("packet_digest") != manifest["packet_digests"][slot])
        errors += int(row.get("semantic_polarity") not in protocol.POLARITY_LABELS)
        errors += int(row.get("factuality") not in protocol.FACTUALITY_LABELS)
        errors += int(
            row.get("factuality_source_tier") not in protocol.FACTUALITY_SOURCE_TIERS
        )
        errors += int(not str(row.get("factuality_evidence") or "").strip())
        errors += int(not isinstance(row.get("negation_scope_text"), list))
        errors += int(not isinstance(row.get("condition_scope_text"), list))
        errors += int(not phase593_analysis.valid_condition_types(row))
        errors += int(not isinstance(row.get("has_contrast"), bool))
        errors += int(not isinstance(row.get("response_complete"), bool))
        errors += int(
            not isinstance(row.get("later_text_changes_final_semantics"), bool)
        )
        errors += int(
            not isinstance(confidence, int)
            or isinstance(confidence, bool)
            or not protocol.CONFIDENCE_MIN <= confidence <= protocol.CONFIDENCE_MAX
        )
        errors += int(not str(row.get("rationale") or "").strip())
        errors += int(not str(row.get("reviewed_at") or "").strip())
        errors += int(row.get("attestation") != phase593.REVIEW_ATTESTATION)
        errors += int(parse_aware_time(row.get("reviewed_at")) is None)
        if packet is not None:
            errors += int(row.get("batch_id") != packet["batch_id"])
            errors += int(not phase593_analysis.valid_span(row, packet["response"]))
            for field in ("negation_scope_text", "condition_scope_text"):
                values = row.get(field)
                if isinstance(values, list):
                    errors += sum(
                        not isinstance(value, str) or value not in packet["response"]
                        for value in values
                    )
        if packet is not None and submission_id not in accepted:
            accepted[submission_id] = row

    expected_ids = set(packet_by_id)
    errors += len(expected_ids - set(ids))
    errors += len(set(ids) - expected_ids)
    errors += len(ids) - len(set(ids))

    batch_time_errors = 0
    previous_end: datetime | None = None
    for batch_index in range(1, protocol.BATCH_COUNT + 1):
        batch_id = f"batch_{batch_index:02d}"
        batch_rows = [row for row in rows if row.get("batch_id") == batch_id]
        batch_time_errors += abs(len(batch_rows) - protocol.PACKET_ITEMS_PER_BATCH)
        starts = {row.get("batch_started_at") for row in batch_rows}
        ends = {row.get("batch_completed_at") for row in batch_rows}
        if len(starts) != 1 or len(ends) != 1:
            batch_time_errors += 1
            continue
        start = parse_aware_time(next(iter(starts)))
        end = parse_aware_time(next(iter(ends)))
        if start is None or end is None:
            batch_time_errors += 1
            continue
        batch_time_errors += int(end <= start)
        if previous_end is not None:
            batch_time_errors += int(start < previous_end)
        previous_end = end
    errors += batch_time_errors

    reviewer_id = next(iter(reviewer_ids)) if len(reviewer_ids) == 1 else None
    structurally_valid = bool(
        len(rows) == protocol.PACKET_ITEM_COUNT
        and len(reviewer_ids) == 1
        and None not in reviewer_ids
        and errors == 0
    )
    return (
        {
            "reviewer_slot": slot,
            "completed_file_present": True,
            "structurally_valid": structurally_valid,
            "reviewed_packet_item_count": len(rows),
            "structural_error_count": errors,
            "reviewer_id": reviewer_id,
            "batch_time_policy_pass": batch_time_errors == 0,
        },
        accepted if structurally_valid else {},
    )


def normalized_condition_types(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(row["condition_types"]))


def split_repeat_ledgers(
    slot: str,
    accepted: dict[str, dict[str, Any]],
    private_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    slot_map = [row for row in private_rows if row["reviewer_slot"] == slot]
    main_rows: dict[str, dict[str, Any]] = {}
    repeat_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mapping in slot_map:
        completed = accepted[mapping["submission_id"]]
        if mapping["item_role"] == "main":
            main_rows[mapping["canonical_review_id"]] = completed
        else:
            repeat_rows[mapping["canonical_review_id"]].append(completed)

    substantive_mismatch_ids: list[str] = []
    anchor_mismatch_ids: list[str] = []
    substantive_dimension_counts: Counter[str] = Counter()
    anchor_dimension_counts: Counter[str] = Counter()
    pair_count = 0
    for canonical_review_id, repeats in repeat_rows.items():
        main = main_rows[canonical_review_id]
        for repeat in repeats:
            pair_count += 1
            substantive_mismatch = False
            for field in protocol.SUBSTANTIVE_REPEAT_FIELDS:
                left = (
                    normalized_condition_types(main)
                    if field == "condition_types"
                    else main[field]
                )
                right = (
                    normalized_condition_types(repeat)
                    if field == "condition_types"
                    else repeat[field]
                )
                if left != right:
                    substantive_mismatch = True
                    substantive_dimension_counts[field] += 1
            if substantive_mismatch:
                substantive_mismatch_ids.append(canonical_review_id)

            anchor_mismatch = False
            for field in protocol.ANCHOR_REPEAT_FIELDS:
                if main[field] != repeat[field]:
                    anchor_mismatch = True
                    anchor_dimension_counts[field] += 1
            if anchor_mismatch:
                anchor_mismatch_ids.append(canonical_review_id)

    substantive_mismatch_count = len(substantive_mismatch_ids)
    return main_rows, {
        "repeat_pair_count": pair_count,
        "substantive_repeat_agreement_count": pair_count - substantive_mismatch_count,
        "substantive_repeat_mismatch_count": substantive_mismatch_count,
        "substantive_repeat_mismatch_canonical_review_ids": sorted(
            set(substantive_mismatch_ids)
        ),
        "substantive_mismatch_dimension_counts": dict(
            sorted(substantive_dimension_counts.items())
        ),
        "anchor_repeat_agreement_count": pair_count - len(anchor_mismatch_ids),
        "anchor_repeat_mismatch_count": len(anchor_mismatch_ids),
        "anchor_repeat_mismatch_canonical_review_ids": sorted(
            set(anchor_mismatch_ids)
        ),
        "anchor_mismatch_dimension_counts": dict(
            sorted(anchor_dimension_counts.items())
        ),
        "reviewer_quality_pass": substantive_mismatch_count
        <= protocol.MAX_SUBSTANTIVE_REPEAT_MISMATCHES,
        "maximum_substantive_repeat_mismatches": (
            protocol.MAX_SUBSTANTIVE_REPEAT_MISMATCHES
        ),
        "repeat_controls_change_gold_denominator": False,
    }


def lock_quality_valid_submission(
    slot: str,
    result: dict[str, Any],
    repeat_audit: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if not result["structurally_valid"] or not repeat_audit["reviewer_quality_pass"]:
        return {
            "lock_present": phase593.submission_lock_path(slot).exists(),
            "lock_created_now": False,
            "lock_valid": False,
            "blocked_by_quality_gate": not repeat_audit["reviewer_quality_pass"],
        }
    reviewer_id = result["reviewer_id"]
    lock_result = phase593_analysis.lock_or_verify_submission(
        phase593.completed_response_path(slot),
        phase593.submission_lock_path(slot),
        reviewer_id,
        protocol.PACKET_ITEM_COUNT,
        manifest["packet_digests"][slot],
    )
    if lock_result["lock_valid"]:
        os.chmod(phase593.completed_response_path(slot), 0o444)
        os.chmod(phase593.submission_lock_path(slot), 0o444)
    return {**lock_result, "blocked_by_quality_gate": False}


def make_adjudication_packet(
    items: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    ordered = sorted(
        items,
        key=lambda row: io_helpers.sha256_text("phase594-adj|" + row["review_id"]),
    )
    instruction = {
        "phase_id": protocol.PHASE,
        "task": (
            "Independently resolve semantic polarity, conditional subtype, and factuality without "
            "seeing prior labels, repeat outcomes, model metadata, or the fact-registry draft."
        ),
        "semantic_labels": list(protocol.POLARITY_LABELS),
        "factuality_labels": list(protocol.FACTUALITY_LABELS),
        "condition_types": list(protocol.CONDITION_TYPES),
        "factuality_source_tiers": list(protocol.FACTUALITY_SOURCE_TIERS),
        "attestation": protocol.ADJUDICATOR_ATTESTATION,
    }
    packet_digest = io_helpers.sha256_text(
        io_helpers.canonical_json(instruction)
        + "\n"
        + "\n".join(io_helpers.canonical_json(row) for row in ordered)
    )
    packet = [
        {
            **row,
            "phase_id": protocol.PHASE,
            "packet_digest": packet_digest,
            "adjudication_instruction": instruction,
        }
        for row in ordered
    ]
    io_helpers.write_jsonl_gz(ADJUDICATION_PACKET_PATH, packet)
    if not protocol.adjudicator_completed_path().exists():
        io_helpers.write_jsonl(
            ADJUDICATION_TEMPLATE_PATH,
            [
                {
                    "review_id": row["review_id"],
                    "packet_digest": packet_digest,
                    "adjudicator_id": None,
                    "semantic_polarity": None,
                    "condition_types": None,
                    "factuality": None,
                    "factuality_source_tier": None,
                    "factuality_evidence": None,
                    "confidence_1_to_5": None,
                    "rationale": None,
                    "attestation": protocol.ADJUDICATOR_ATTESTATION,
                    "adjudicated_at": None,
                }
                for row in packet
            ],
        )
    manifest["adjudication_packet_digest"] = packet_digest


def valid_adjudication_condition_types(row: dict[str, Any]) -> bool:
    condition_types = row.get("condition_types")
    if not isinstance(condition_types, list) or not condition_types:
        return False
    if (
        any(value not in protocol.CONDITION_TYPES for value in condition_types)
        or len(condition_types) != len(set(condition_types))
    ):
        return False
    conditional = bool(
        row.get("semantic_polarity") == "conditional"
        or row.get("factuality") == "conditional"
    )
    if conditional:
        return "none" not in condition_types
    return condition_types == ["none"] or "none" not in condition_types


def validate_adjudicator(
    adjudication_items: list[dict[str, Any]], reviewer_ids: list[str]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not adjudication_items:
        return (
            {
                "required": False,
                "completed_file_present": False,
                "structurally_valid": False,
                "submission_lock_valid": False,
                "reviewed_item_count": 0,
                "structural_error_count": 0,
                "adjudicator_id": None,
            },
            {},
        )
    path = protocol.adjudicator_completed_path()
    if not path.exists() or not ADJUDICATION_PACKET_PATH.exists():
        return (
            {
                "required": True,
                "completed_file_present": path.exists(),
                "structurally_valid": False,
                "submission_lock_valid": False,
                "reviewed_item_count": 0,
                "structural_error_count": len(adjudication_items),
                "adjudicator_id": None,
            },
            {},
        )
    packet = read_jsonl_gz(ADJUDICATION_PACKET_PATH)
    packet_by_id = {row["review_id"]: row for row in packet}
    try:
        rows = read_jsonl(path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        rows = []
    ids = [row.get("review_id") for row in rows]
    adjudicator_ids = {row.get("adjudicator_id") for row in rows}
    errors = 0
    accepted: dict[str, dict[str, Any]] = {}
    for row in rows:
        review_id = row.get("review_id")
        source = packet_by_id.get(review_id)
        confidence = row.get("confidence_1_to_5")
        errors += int(source is None)
        errors += int(
            source is not None and row.get("packet_digest") != source["packet_digest"]
        )
        errors += int(row.get("semantic_polarity") not in protocol.POLARITY_LABELS)
        errors += int(row.get("factuality") not in protocol.FACTUALITY_LABELS)
        errors += int(not valid_adjudication_condition_types(row))
        errors += int(
            row.get("factuality_source_tier") not in protocol.FACTUALITY_SOURCE_TIERS
        )
        errors += int(not str(row.get("factuality_evidence") or "").strip())
        errors += int(
            not isinstance(confidence, int)
            or isinstance(confidence, bool)
            or not protocol.CONFIDENCE_MIN <= confidence <= protocol.CONFIDENCE_MAX
        )
        errors += int(not str(row.get("rationale") or "").strip())
        errors += int(row.get("attestation") != protocol.ADJUDICATOR_ATTESTATION)
        errors += int(parse_aware_time(row.get("adjudicated_at")) is None)
        if source is not None and review_id not in accepted:
            accepted[review_id] = row
    expected_ids = set(packet_by_id)
    errors += len(expected_ids - set(ids))
    errors += len(set(ids) - expected_ids)
    errors += len(ids) - len(set(ids))
    adjudicator_id = next(iter(adjudicator_ids)) if len(adjudicator_ids) == 1 else None
    errors += int(adjudicator_id is None or adjudicator_id in set(reviewer_ids))
    structurally_valid = bool(
        len(rows) == len(adjudication_items)
        and len(adjudicator_ids) == 1
        and None not in adjudicator_ids
        and errors == 0
    )
    lock_result: dict[str, Any] = {"lock_valid": False, "lock_present": False}
    if structurally_valid and adjudicator_id is not None and packet:
        lock_result = phase593_analysis.lock_or_verify_submission(
            path,
            protocol.adjudicator_lock_path(),
            adjudicator_id,
            len(rows),
            packet[0]["packet_digest"],
        )
        if lock_result["lock_valid"]:
            os.chmod(path, 0o444)
            os.chmod(protocol.adjudicator_lock_path(), 0o444)
    return (
        {
            "required": True,
            "completed_file_present": True,
            "structurally_valid": structurally_valid,
            "submission_lock_valid": lock_result["lock_valid"],
            "submission_lock": lock_result,
            "reviewed_item_count": len(rows),
            "structural_error_count": errors,
            "adjudicator_id": adjudicator_id,
        },
        accepted if structurally_valid and lock_result["lock_valid"] else {},
    )


def analyze() -> dict[str, Any]:
    amendment = read_json(protocol.PROTOCOL_PATH)
    phase593_manifest = read_json(phase593.PROTOCOL_PATH)
    private_rows = read_json(phase593.PRIVATE_MAP_PATH)["rows"]
    fact_registry_result = validate_fact_registry()
    packets: dict[str, dict[str, dict[str, Any]]] = {}
    reviewer_results: list[dict[str, Any]] = []
    main_rows_by_slot: dict[str, dict[str, dict[str, Any]]] = {}
    repeat_audits: dict[str, dict[str, Any]] = {}
    valid_locked_slots: list[str] = []
    for slot in protocol.REVIEWER_SLOTS:
        packet_rows = read_jsonl_gz(phase593.packet_path(slot))
        packets[slot] = {row["submission_id"]: row for row in packet_rows}
        result, rows = validate_completed_structure(
            slot, phase593_manifest, packets[slot]
        )
        if result["structurally_valid"]:
            main_rows, repeat_audit = split_repeat_ledgers(slot, rows, private_rows)
            repeat_audits[slot] = repeat_audit
            lock_result = lock_quality_valid_submission(
                slot, result, repeat_audit, phase593_manifest
            )
            result["reviewer_quality_pass"] = repeat_audit["reviewer_quality_pass"]
            result["submission_lock_valid"] = lock_result["lock_valid"]
            result["submission_lock"] = lock_result
            if repeat_audit["reviewer_quality_pass"] and lock_result["lock_valid"]:
                main_rows_by_slot[slot] = main_rows
                valid_locked_slots.append(slot)
        else:
            result["reviewer_quality_pass"] = False
            result["submission_lock_valid"] = False
        reviewer_results.append(result)

    reviewer_ids = [
        result["reviewer_id"]
        for result in reviewer_results
        if result["reviewer_slot"] in valid_locked_slots
    ]
    three_distinct_quality_valid_locked = bool(
        len(reviewer_ids) == 3 and len(set(reviewer_ids)) == 3
    )
    substantive_repeat_conflict_ids = {
        review_id
        for audit in repeat_audits.values()
        for review_id in audit["substantive_repeat_mismatch_canonical_review_ids"]
    }
    anchor_repeat_conflict_ids = {
        review_id
        for audit in repeat_audits.values()
        for review_id in audit["anchor_repeat_mismatch_canonical_review_ids"]
    }

    source_rows = io_helpers.read_jsonl_gz(phase593.SOURCE_QUEUE_PATH)
    source_by_id = {row["review_id"]: row for row in source_rows}
    direct_rows: list[dict[str, Any]] = []
    direct_resolved = 0
    anchor_qualified = 0
    anchor_punctuation_rejected = 0
    workflow_counts: Counter[str] = Counter()
    adjudication_items: list[dict[str, Any]] = []
    if three_distinct_quality_valid_locked and len(main_rows_by_slot) == 3:
        for review_id in sorted(source_by_id):
            rows = [
                main_rows_by_slot[slot][review_id]
                for slot in protocol.REVIEWER_SLOTS
            ]
            semantic, semantic_status = phase593_analysis.direct_consensus(
                [row["semantic_polarity"] for row in rows], "unresolved"
            )
            factuality, factuality_status = phase593_analysis.direct_consensus(
                [row["factuality"] for row in rows], "uncertain"
            )
            condition_signatures = [normalized_condition_types(row) for row in rows]
            conditional_structure_conflict = bool(
                (semantic == "conditional" or factuality == "conditional")
                and len(set(condition_signatures)) != 1
            )
            substantive_repeat_conflict = review_id in substantive_repeat_conflict_ids
            directly_resolved = bool(
                semantic is not None
                and factuality is not None
                and not conditional_structure_conflict
                and not substantive_repeat_conflict
            )
            if directly_resolved:
                direct_resolved += 1
                workflow_counts["directly_resolved"] += 1
                anchor_ok, anchor_span = phase593_analysis.majority_covered_anchor(
                    rows, semantic
                )
                response = source_by_id[review_id]["response"]
                semantic_content_ok = anchor_has_semantic_content(response, anchor_span)
                if anchor_ok and not semantic_content_ok:
                    anchor_punctuation_rejected += 1
                repeat_anchor_ok = review_id not in anchor_repeat_conflict_ids
                final_anchor_ok = bool(
                    anchor_ok
                    and semantic_content_ok
                    and repeat_anchor_ok
                    and semantic != "unresolved"
                )
                anchor_qualified += int(final_anchor_ok)
                direct_rows.append(
                    {
                        "schema_version": "phase594_resolution_provenance_item.v1",
                        "phase_id": protocol.PHASE,
                        "review_id": review_id,
                        "semantic_polarity": semantic,
                        "factuality": factuality,
                        "resolved_condition_types": (
                            list(condition_signatures[0])
                            if len(set(condition_signatures)) == 1
                            else None
                        ),
                        "semantic_resolution_mode": "direct_" + semantic_status,
                        "factuality_resolution_mode": "direct_" + factuality_status,
                        "event_anchor_qualified": final_anchor_ok,
                        "event_anchor_span": anchor_span if final_anchor_ok else None,
                        "event_anchor_semantic_content_pass": semantic_content_ok,
                        "event_anchor_repeat_stability_pass": repeat_anchor_ok,
                        "raw_first_pass_rows_preserved": True,
                    }
                )
            else:
                workflow_counts["requires_independent_adjudication"] += 1
                reasons = []
                if semantic is None:
                    reasons.append(semantic_status)
                if factuality is None:
                    reasons.append(factuality_status)
                if conditional_structure_conflict:
                    reasons.append("conditional_structure_conflict")
                if substantive_repeat_conflict:
                    reasons.append("substantive_repeat_control_conflict")
                source = source_by_id[review_id]
                adjudication_items.append(
                    {
                        "schema_version": "phase594_adjudication_item.v1",
                        "review_id": review_id,
                        "prompt": source["prompt"],
                        "response": source["response"],
                        "conflict_reasons": sorted(set(reasons)),
                        "prior_reviewer_labels_hidden": True,
                        "repeat_control_outcomes_hidden": True,
                        "hidden_model_metadata_absent": True,
                    }
                )
        if adjudication_items:
            make_adjudication_packet(adjudication_items, amendment)
    else:
        workflow_counts["pending_three_quality_valid_locked_reviewers"] = (
            protocol.MAIN_ITEM_COUNT
        )

    adjudicator_result, adjudicated_rows = validate_adjudicator(
        adjudication_items, reviewer_ids
    )
    adjudicator_valid = bool(
        adjudicator_result["structurally_valid"]
        and adjudicator_result["submission_lock_valid"]
    )
    adjudicated_resolved = len(adjudicated_rows) if adjudicator_valid else 0
    workflow_unresolved = (
        protocol.MAIN_ITEM_COUNT - direct_resolved - adjudicated_resolved
    )
    semantic_gold_complete = bool(
        three_distinct_quality_valid_locked
        and direct_resolved + adjudicated_resolved == protocol.MAIN_ITEM_COUNT
        and workflow_unresolved == 0
    )

    resolved_rows = list(direct_rows)
    if adjudicator_valid:
        resolved_rows.extend(
            {
                "schema_version": "phase594_resolution_provenance_item.v1",
                "phase_id": protocol.PHASE,
                "review_id": review_id,
                "semantic_polarity": row["semantic_polarity"],
                "factuality": row["factuality"],
                "resolved_condition_types": row["condition_types"],
                "semantic_resolution_mode": "independent_adjudication",
                "factuality_resolution_mode": "independent_adjudication",
                "event_anchor_qualified": False,
                "event_anchor_span": None,
                "event_anchor_semantic_content_pass": False,
                "event_anchor_repeat_stability_pass": False,
                "raw_first_pass_rows_preserved": True,
            }
            for review_id, row in sorted(adjudicated_rows.items())
        )
    if semantic_gold_complete:
        resolved_rows = sorted(resolved_rows, key=lambda row: row["review_id"])
        io_helpers.write_jsonl_gz(PROVENANCE_PATH, resolved_rows)
        io_helpers.write_jsonl_gz(
            SEMANTIC_GOLD_PATH,
            [
                {
                    "schema_version": "phase594_semantic_gold_item.v1",
                    "review_id": row["review_id"],
                    "semantic_polarity": row["semantic_polarity"],
                    "resolved_condition_types": row["resolved_condition_types"],
                    "resolution_mode": row["semantic_resolution_mode"],
                }
                for row in resolved_rows
            ],
        )
        io_helpers.write_jsonl_gz(
            FACTUALITY_GOLD_PATH,
            [
                {
                    "schema_version": "phase594_reviewed_factuality_gold_item.v1",
                    "review_id": row["review_id"],
                    "factuality": row["factuality"],
                    "resolved_condition_types": row["resolved_condition_types"],
                    "resolution_mode": row["factuality_resolution_mode"],
                    "authoritative_fact_registry_claim": False,
                }
                for row in resolved_rows
            ],
        )
        io_helpers.write_jsonl_gz(
            ANCHOR_GOLD_PATH,
            [
                {
                    "schema_version": "phase594_event_anchor_item.v1",
                    "review_id": row["review_id"],
                    "semantic_polarity": row["semantic_polarity"],
                    "event_anchor_span": row["event_anchor_span"],
                    "offset_unit": "zero_based_nfc_unicode_code_point_half_open_interval",
                    "tokenizer_mapping_complete": False,
                }
                for row in resolved_rows
                if row["event_anchor_qualified"]
            ],
        )

    status = {
        "schema_version": "phase594_external_review_status.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "reviewer_results": reviewer_results,
        "repeat_control_audits": repeat_audits,
        "quality_valid_locked_reviewer_count": len(valid_locked_slots),
        "three_distinct_quality_valid_locked_reviewer_identity_pass": (
            three_distinct_quality_valid_locked
        ),
        "completed_quality_valid_locked_main_label_count": len(valid_locked_slots)
        * protocol.MAIN_ITEM_COUNT,
        "completed_quality_valid_locked_repeat_label_count": len(valid_locked_slots)
        * protocol.REPEAT_ITEM_COUNT,
        "substantive_repeat_conflict_main_item_count": len(
            substantive_repeat_conflict_ids
        ),
        "anchor_repeat_conflict_main_item_count": len(anchor_repeat_conflict_ids),
        "workflow_status_counts": dict(sorted(workflow_counts.items())),
        "directly_resolved_item_count": direct_resolved,
        "adjudication_required_item_count": len(adjudication_items),
        "adjudicator_result": adjudicator_result,
        "independently_adjudicated_item_count": adjudicated_resolved,
        "workflow_unresolved_item_count": workflow_unresolved,
        "semantic_gold_complete": semantic_gold_complete,
        "reviewed_factuality_gold_complete": semantic_gold_complete,
        "authoritative_fact_registry_complete": fact_registry_result[
            "registry_complete"
        ],
        "authoritative_fact_registry_result": fact_registry_result,
        "external_semantic_and_fact_artifacts_ready": False,
        "evaluable_denominator_contract_frozen": False,
        "evaluable_denominator_gate_passed": False,
        "event_anchor_qualified_item_count": anchor_qualified,
        "event_anchor_punctuation_rejected_item_count": anchor_punctuation_rejected,
        "separate_gold_artifacts_written": semantic_gold_complete,
        "private_answer_key_read": False,
        "phase590_confirmation_heldout_or_sealed_read": False,
        "machine_or_agent_annotation_substituted": False,
        "model_case_count_consumed": 0,
        "internal_state_case_count_consumed": 0,
    }
    external_artifacts_ready = bool(
        semantic_gold_complete and fact_registry_result["registry_complete"]
    )
    evaluable_denominator_contract_frozen = False
    evaluable_denominator_gate_passed = False
    status["external_semantic_and_fact_artifacts_ready"] = external_artifacts_ready
    status["evaluable_denominator_contract_frozen"] = (
        evaluable_denominator_contract_frozen
    )
    status["evaluable_denominator_gate_passed"] = evaluable_denominator_gate_passed
    io_helpers.write_json(STATUS_PATH, status)

    behavior_truth_authorized = bool(
        external_artifacts_ready
        and evaluable_denominator_contract_frozen
        and evaluable_denominator_gate_passed
    )
    if external_artifacts_ready:
        stage_status = (
            "external_semantic_and_fact_artifacts_ready_pending_evaluable_"
            "denominator_contract"
        )
        next_external_action = (
            "freeze_evaluable_denominator_contract_before_behavior_truth_evaluation"
        )
    elif semantic_gold_complete:
        stage_status = "blocked_pending_external_fact_audit"
        next_external_action = "complete_and_lock_claim_driven_external_fact_registry"
    elif not three_distinct_quality_valid_locked:
        stage_status = "blocked_pending_three_external_human_reviewers"
        next_external_action = (
            "three_distinct_people_complete_existing_phase593_v3_packets"
        )
    elif adjudication_items and not adjudicator_valid:
        stage_status = "blocked_pending_independent_adjudicator"
        next_external_action = "independent_fourth_person_complete_adjudication_packet"
    else:
        stage_status = "blocked_pending_human_resolution"
        next_external_action = "inspect_unresolved_human_workflow_state"

    stage = {
        "schema_version": "phase594_stage_summary.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": stage_status,
        "denominators": {
            "required_total_human_entries": 900,
            "required_first_pass_main_labels": 864,
            "required_repeat_control_labels": 36,
            "completed_quality_valid_locked_main_labels": len(valid_locked_slots)
            * protocol.MAIN_ITEM_COUNT,
            "completed_quality_valid_locked_repeat_labels": len(valid_locked_slots)
            * protocol.REPEAT_ITEM_COUNT,
            "resolved_semantic_items": (
                direct_resolved + adjudicated_resolved if semantic_gold_complete else 0
            ),
            "completed_external_fact_claims": fact_registry_result["claim_count"],
            "completed_fact_review_dispositions": fact_registry_result[
                "review_disposition_count"
            ],
            "completed_fact_propositions": fact_registry_result.get(
                "proposition_count", 0
            ),
            "model_case_count_consumed": 0,
            "internal_state_case_count_consumed": 0,
            "sealed_case_count_consumed": 0,
        },
        "authorization": {
            "develop_observer_on_discovery_gold": semantic_gold_complete,
            "evaluate_behavior_truth": behavior_truth_authorized,
            "run_internal_atlas": False,
            "evaluate_confirmation_or_heldout": False,
            "run_long_generation": False,
            "run_qwen3": False,
            "run_glm4": False,
            "run_deepseek7b": False,
            "run_open_internal_trace": False,
            "run_causal_intervention": False,
            "read_phase590_sealed_set": False,
        },
        "readiness": {
            "semantic_gold_available": semantic_gold_complete,
            "fact_registry_locked": fact_registry_result["registry_complete"],
            "behavior_truth_artifacts_ready": external_artifacts_ready,
            "evaluable_denominator_contract_frozen": (
                evaluable_denominator_contract_frozen
            ),
            "evaluable_denominator_gate_passed": evaluable_denominator_gate_passed,
        },
        "automatic_model_execution_now": False,
        "next_required_external_action": next_external_action,
        "remaining_external_fact_action": (
            "none_fact_registry_complete"
            if fact_registry_result["registry_complete"]
            else "complete_separate_claim_driven_fact_registry_by_external_source_audit"
        ),
        "claim_boundary": (
            "final_pre_human_quality_amendment_only_no_new_model_behavior_internal_or_"
            "mechanism_evidence"
        ),
    }
    io_helpers.write_json(STAGE_PATH, stage)
    return stage


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
