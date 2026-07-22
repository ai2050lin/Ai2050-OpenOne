#!/usr/bin/env python3
"""Validate Phase593 human reviews without substituting machine annotations."""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase591_blind_semantic_gold_protocol as io_helpers  # noqa: E402
import phase593_human_semantic_review_integrity_protocol as protocol  # noqa: E402


STATUS_PATH = protocol.OUT_DIR / "phase593_external_review_status.json"
STAGE_PATH = protocol.OUT_DIR / "phase593_stage_summary.json"
ADJUDICATION_PACKET_PATH = protocol.REVIEW_DIR / "adjudicator_packet_v3.jsonl.gz"
ADJUDICATION_TEMPLATE_PATH = protocol.REVIEW_DIR / "adjudicator_response_template_v3.jsonl"
GOLD_PATH = protocol.OUT_DIR / "phase593_resolved_semantic_gold.jsonl.gz"


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


def valid_span(row: dict[str, Any], response: str) -> bool:
    start = row.get("decisive_span_start")
    end = row.get("decisive_span_end")
    if row.get("semantic_polarity") == "unresolved":
        return start is None and end is None
    return bool(
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start < end <= len(response)
    )


def valid_condition_types(row: dict[str, Any]) -> bool:
    condition_types = row.get("condition_types")
    condition_scope = row.get("condition_scope_text")
    if not isinstance(condition_types, list) or not condition_types:
        return False
    if (
        any(value not in protocol.CONDITION_TYPES for value in condition_types)
        or len(condition_types) != len(set(condition_types))
        or not isinstance(condition_scope, list)
    ):
        return False
    if not condition_scope:
        return bool(
            condition_types == ["none"]
            and row.get("semantic_polarity") != "conditional"
            and row.get("factuality") != "conditional"
        )
    if "none" in condition_types:
        return False
    if row.get("semantic_polarity") == "conditional" and not condition_types:
        return False
    return True


def lock_or_verify_submission(
    completed_path: Path,
    lock_path: Path,
    reviewer_id: str,
    item_count: int,
    packet_digest: str,
) -> dict[str, Any]:
    file_digest = io_helpers.sha256_file(completed_path)
    expected = {
        "schema_version": "phase593_submission_lock.v1",
        "phase_id": protocol.PHASE,
        "reviewer_id": reviewer_id,
        "item_count": item_count,
        "packet_digest": packet_digest,
        "completed_file_sha256": file_digest,
    }
    if lock_path.exists():
        existing = read_json(lock_path)
        comparisons = {
            key: existing.get(key) == value
            for key, value in expected.items()
            if key not in {"schema_version", "phase_id"}
        }
        return {
            "lock_present": True,
            "lock_created_now": False,
            "lock_valid": all(comparisons.values()),
            "completed_file_sha256": file_digest,
            "comparison_pass": comparisons,
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
        return lock_or_verify_submission(
            completed_path, lock_path, reviewer_id, item_count, packet_digest
        )
    return {
        "lock_present": True,
        "lock_created_now": True,
        "lock_valid": True,
        "completed_file_sha256": file_digest,
        "comparison_pass": {
            "reviewer_id": True,
            "item_count": True,
            "packet_digest": True,
            "completed_file_sha256": True,
        },
    }


def validate_completed(
    slot: str,
    manifest: dict[str, Any],
    packet_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = protocol.completed_response_path(slot)
    if not path.exists():
        return (
            {
                "reviewer_slot": slot,
                "completed_file_present": False,
                "structurally_valid": False,
                "submission_lock_valid": False,
                "reviewed_packet_item_count": 0,
                "structural_error_count": protocol.PACKET_ITEM_COUNT,
                "reviewer_id": None,
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
                "submission_lock_valid": False,
                "reviewed_packet_item_count": 0,
                "structural_error_count": protocol.PACKET_ITEM_COUNT,
                "reviewer_id": None,
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
        errors += int(not valid_condition_types(row))
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
        errors += int(not str(row.get("batch_started_at") or "").strip())
        errors += int(not str(row.get("batch_completed_at") or "").strip())
        errors += int(not str(row.get("reviewed_at") or "").strip())
        errors += int(row.get("attestation") != protocol.REVIEW_ATTESTATION)
        if packet is not None:
            errors += int(row.get("batch_id") != packet["batch_id"])
            errors += int(not valid_span(row, packet["response"]))
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
    if rows:
        for batch_index in range(1, protocol.BATCH_COUNT + 1):
            batch_id = f"batch_{batch_index:02d}"
            batch_rows = [row for row in rows if row.get("batch_id") == batch_id]
            errors += abs(len(batch_rows) - protocol.PACKET_ITEMS_PER_BATCH)
            starts = {row.get("batch_started_at") for row in batch_rows}
            ends = {row.get("batch_completed_at") for row in batch_rows}
            errors += int(len(starts) != 1 or None in starts or "" in starts)
            errors += int(len(ends) != 1 or None in ends or "" in ends)
            if (
                len(starts) == 1
                and len(ends) == 1
                and None not in starts | ends
                and "" not in starts | ends
            ):
                try:
                    start_time = datetime.fromisoformat(
                        str(next(iter(starts))).replace("Z", "+00:00")
                    )
                    end_time = datetime.fromisoformat(
                        str(next(iter(ends))).replace("Z", "+00:00")
                    )
                    errors += int(end_time < start_time)
                except ValueError:
                    errors += 1

    reviewer_id = next(iter(reviewer_ids)) if len(reviewer_ids) == 1 else None
    structurally_valid = bool(
        len(rows) == protocol.PACKET_ITEM_COUNT
        and len(reviewer_ids) == 1
        and None not in reviewer_ids
        and errors == 0
    )
    lock_result: dict[str, Any] = {
        "lock_present": protocol.submission_lock_path(slot).exists(),
        "lock_created_now": False,
        "lock_valid": False,
        "completed_file_sha256": io_helpers.sha256_file(path),
    }
    if structurally_valid and reviewer_id is not None:
        lock_result = lock_or_verify_submission(
            path,
            protocol.submission_lock_path(slot),
            reviewer_id,
            len(rows),
            manifest["packet_digests"][slot],
        )
    return (
        {
            "reviewer_slot": slot,
            "completed_file_present": True,
            "structurally_valid": structurally_valid,
            "submission_lock_valid": lock_result["lock_valid"],
            "submission_lock": lock_result,
            "reviewed_packet_item_count": len(rows),
            "structural_error_count": errors,
            "reviewer_id": reviewer_id,
        },
        accepted if structurally_valid and lock_result["lock_valid"] else {},
    )


def direct_consensus(
    labels: list[str], unresolved_label: str
) -> tuple[str | None, str]:
    counts = Counter(labels)
    label, count = counts.most_common(1)[0]
    if count == 3:
        return label, "unanimous"
    if count == 2:
        minority = next(value for value in labels if value != label)
        if label != unresolved_label and minority == unresolved_label:
            return label, "nonopposed_majority"
    return None, "requires_independent_adjudication"


def majority_covered_anchor(
    rows: list[dict[str, Any]], semantic_label: str
) -> tuple[bool, list[int] | None]:
    eligible = [
        row
        for row in rows
        if row["semantic_polarity"] == semantic_label
        and row["response_complete"]
        and not row["later_text_changes_final_semantics"]
        and isinstance(row["decisive_span_start"], int)
        and not isinstance(row["decisive_span_start"], bool)
        and isinstance(row["decisive_span_end"], int)
        and not isinstance(row["decisive_span_end"], bool)
    ]
    if len(eligible) < 2:
        return False, None
    lower = min(row["decisive_span_start"] for row in eligible)
    upper = max(row["decisive_span_end"] for row in eligible)
    covered = [
        position
        for position in range(lower, upper)
        if sum(
            row["decisive_span_start"] <= position < row["decisive_span_end"]
            for row in eligible
        )
        >= 2
    ]
    if not covered:
        return False, None
    start = covered[0]
    end = covered[-1] + 1
    if covered != list(range(start, end)):
        return False, None
    return True, [start, end]


def normalized_condition_types(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(row["condition_types"]))


def split_main_and_repeat_rows(
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

    comparison_fields = (
        "semantic_polarity",
        "factuality",
        "response_complete",
        "decisive_span_start",
        "decisive_span_end",
        "later_text_changes_final_semantics",
    )
    mismatch_ids: list[str] = []
    dimension_counts: Counter[str] = Counter()
    repeat_pair_count = 0
    for canonical_review_id, repeats in repeat_rows.items():
        main = main_rows[canonical_review_id]
        for repeat in repeats:
            repeat_pair_count += 1
            mismatch = False
            for field in comparison_fields:
                if main[field] != repeat[field]:
                    mismatch = True
                    dimension_counts[field] += 1
            if normalized_condition_types(main) != normalized_condition_types(repeat):
                mismatch = True
                dimension_counts["condition_types"] += 1
            if mismatch:
                mismatch_ids.append(canonical_review_id)
    return main_rows, {
        "repeat_pair_count": repeat_pair_count,
        "exact_repeat_agreement_count": repeat_pair_count - len(mismatch_ids),
        "repeat_mismatch_count": len(mismatch_ids),
        "repeat_mismatch_canonical_review_ids": sorted(set(mismatch_ids)),
        "mismatch_dimension_counts": dict(sorted(dimension_counts.items())),
        "repeat_controls_change_gold_denominator": False,
    }


def make_adjudication_packet(
    items: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    ordered = sorted(
        items,
        key=lambda row: io_helpers.sha256_text("phase593-adj|" + row["review_id"]),
    )
    instruction = {
        "phase_id": protocol.PHASE,
        "task": (
            "Independently resolve semantic polarity, conditional subtype, and factuality without "
            "seeing prior labels or repeat-control outcomes."
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
        adjudication_condition_row = {
            **row,
            "condition_scope_text": (
                ["adjudicated condition"]
                if row.get("condition_types") != ["none"]
                else []
            ),
        }
        errors += int(not valid_condition_types(adjudication_condition_row))
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
        errors += int(not str(row.get("adjudicated_at") or "").strip())
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
        lock_result = lock_or_verify_submission(
            path,
            protocol.adjudicator_lock_path(),
            adjudicator_id,
            len(rows),
            packet[0]["packet_digest"],
        )
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
    manifest = read_json(protocol.PROTOCOL_PATH)
    private_rows = read_json(protocol.PRIVATE_MAP_PATH)["rows"]
    packets: dict[str, dict[str, dict[str, Any]]] = {}
    valid_rows: dict[str, dict[str, dict[str, Any]]] = {}
    reviewer_results: list[dict[str, Any]] = []
    main_rows_by_slot: dict[str, dict[str, dict[str, Any]]] = {}
    repeat_audits: dict[str, dict[str, Any]] = {}
    for slot in protocol.REVIEWER_SLOTS:
        packet_rows = read_jsonl_gz(protocol.packet_path(slot))
        packets[slot] = {row["submission_id"]: row for row in packet_rows}
        result, rows = validate_completed(slot, manifest, packets[slot])
        reviewer_results.append(result)
        if result["structurally_valid"] and result["submission_lock_valid"]:
            valid_rows[slot] = rows
            main_rows, repeat_audit = split_main_and_repeat_rows(
                slot, rows, private_rows
            )
            main_rows_by_slot[slot] = main_rows
            repeat_audits[slot] = repeat_audit

    reviewer_ids = [
        result["reviewer_id"]
        for result in reviewer_results
        if result["structurally_valid"] and result["submission_lock_valid"]
    ]
    three_distinct_locked = bool(
        len(reviewer_ids) == 3 and len(set(reviewer_ids)) == 3
    )
    repeat_mismatch_ids = {
        review_id
        for audit in repeat_audits.values()
        for review_id in audit["repeat_mismatch_canonical_review_ids"]
    }

    direct_resolved = 0
    anchor_qualified = 0
    direct_gold_rows: list[dict[str, Any]] = []
    workflow_counts: Counter[str] = Counter()
    adjudication_items: list[dict[str, Any]] = []
    source_rows = io_helpers.read_jsonl_gz(protocol.SOURCE_QUEUE_PATH)
    source_by_id = {row["review_id"]: row for row in source_rows}
    if three_distinct_locked and len(main_rows_by_slot) == 3:
        for review_id in sorted(source_by_id):
            rows = [
                main_rows_by_slot[slot][review_id]
                for slot in protocol.REVIEWER_SLOTS
            ]
            semantic, semantic_status = direct_consensus(
                [row["semantic_polarity"] for row in rows], "unresolved"
            )
            factuality, factuality_status = direct_consensus(
                [row["factuality"] for row in rows], "uncertain"
            )
            condition_signatures = [normalized_condition_types(row) for row in rows]
            conditional_structure_conflict = bool(
                (semantic == "conditional" or factuality == "conditional")
                and len(set(condition_signatures)) != 1
            )
            repeat_control_conflict = review_id in repeat_mismatch_ids
            directly_resolved = bool(
                semantic is not None
                and factuality is not None
                and not conditional_structure_conflict
                and not repeat_control_conflict
            )
            if directly_resolved:
                direct_resolved += 1
                workflow_counts["directly_resolved"] += 1
                anchor_ok, anchor_span = majority_covered_anchor(rows, semantic)
                anchor_qualified += int(anchor_ok and semantic != "unresolved")
                direct_gold_rows.append(
                    {
                        "schema_version": "phase593_resolved_semantic_gold_item.v1",
                        "phase_id": protocol.PHASE,
                        "review_id": review_id,
                        "semantic_polarity": semantic,
                        "factuality": factuality,
                        "resolved_condition_types": (
                            list(condition_signatures[0])
                            if len(set(condition_signatures)) == 1
                            else None
                        ),
                        "resolution_mode": "direct_" + semantic_status,
                        "factuality_resolution_mode": "direct_" + factuality_status,
                        "event_anchor_qualified": bool(
                            anchor_ok and semantic != "unresolved"
                        ),
                        "event_anchor_majority_coverage_span": anchor_span,
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
                if repeat_control_conflict:
                    reasons.append("within_reviewer_repeat_control_conflict")
                source = source_by_id[review_id]
                adjudication_items.append(
                    {
                        "schema_version": "phase593_adjudication_item.v1",
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
            make_adjudication_packet(adjudication_items, manifest)
    else:
        workflow_counts["pending_three_distinct_locked_reviewers"] = protocol.MAIN_ITEM_COUNT

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
        three_distinct_locked
        and direct_resolved + adjudicated_resolved == protocol.MAIN_ITEM_COUNT
        and workflow_unresolved == 0
    )
    gold_rows = list(direct_gold_rows)
    if adjudicator_valid:
        gold_rows.extend(
            {
                "schema_version": "phase593_resolved_semantic_gold_item.v1",
                "phase_id": protocol.PHASE,
                "review_id": review_id,
                "semantic_polarity": row["semantic_polarity"],
                "factuality": row["factuality"],
                "resolved_condition_types": row["condition_types"],
                "resolution_mode": "independent_adjudication",
                "factuality_resolution_mode": "independent_adjudication",
                "event_anchor_qualified": False,
                "event_anchor_majority_coverage_span": None,
                "raw_first_pass_rows_preserved": True,
            }
            for review_id, row in sorted(adjudicated_rows.items())
        )
    if semantic_gold_complete:
        io_helpers.write_jsonl_gz(
            GOLD_PATH, sorted(gold_rows, key=lambda row: row["review_id"])
        )

    status = {
        "schema_version": "phase593_external_review_status.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "reviewer_results": reviewer_results,
        "three_distinct_locked_reviewer_identity_pass": three_distinct_locked,
        "valid_locked_first_pass_reviewer_count": len(valid_rows),
        "valid_locked_first_pass_main_label_count": len(valid_rows)
        * protocol.MAIN_ITEM_COUNT,
        "valid_locked_repeat_control_label_count": len(valid_rows)
        * protocol.REPEAT_ITEM_COUNT,
        "repeat_control_audits": repeat_audits,
        "repeat_control_conflict_main_item_count": len(repeat_mismatch_ids),
        "workflow_status_counts": dict(sorted(workflow_counts.items())),
        "directly_resolved_item_count": direct_resolved,
        "adjudication_required_item_count": len(adjudication_items),
        "adjudicator_result": adjudicator_result,
        "independently_adjudicated_item_count": adjudicated_resolved,
        "workflow_unresolved_item_count": workflow_unresolved,
        "semantic_gold_complete": semantic_gold_complete,
        "authoritative_factual_gold_complete": False,
        "authoritative_factual_gold_blocker": (
            "fixed_object_level_authoritative_registry_not_present"
        ),
        "resolved_gold_artifact_written": semantic_gold_complete,
        "resolved_gold_artifact_sha256": (
            io_helpers.sha256_file(GOLD_PATH) if semantic_gold_complete else None
        ),
        "event_anchor_qualified_item_count": anchor_qualified,
        "event_anchor_rule": "contiguous_majority_character_coverage",
        "private_answer_key_read": False,
        "phase590_sealed_cases_read": False,
        "machine_or_agent_annotation_substituted": False,
        "model_case_count_consumed": 0,
        "internal_state_case_count_consumed": 0,
    }
    io_helpers.write_json(STATUS_PATH, status)

    stage = {
        "schema_version": "phase593_stage_summary.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": (
            "semantic_gold_complete"
            if semantic_gold_complete
            else "blocked_pending_external_human_review"
        ),
        "denominators": {
            "main_gold_item_count": protocol.MAIN_ITEM_COUNT,
            "required_first_pass_human_main_labels": protocol.MAIN_ITEM_COUNT * 3,
            "required_repeat_control_labels": protocol.REPEAT_ITEM_COUNT * 3,
            "completed_valid_locked_first_pass_human_main_labels": len(valid_rows)
            * protocol.MAIN_ITEM_COUNT,
            "completed_valid_locked_repeat_control_labels": len(valid_rows)
            * protocol.REPEAT_ITEM_COUNT,
            "model_case_count_consumed": 0,
            "internal_state_case_count_consumed": 0,
            "sealed_case_count_consumed": 0,
        },
        "authorization": {
            "develop_observer_on_discovery_gold": semantic_gold_complete,
            "evaluate_confirmation_or_heldout": False,
            "run_long_generation": False,
            "run_qwen3": False,
            "run_glm4": False,
            "run_deepseek7b": False,
            "run_open_internal_trace": False,
            "run_causal_intervention": False,
            "read_phase590_sealed_set": False,
        },
        "automatic_model_execution_now": False,
        "next_required_external_action": (
            "none_semantic_gold_complete"
            if semantic_gold_complete
            else "three_distinct_people_complete_phase593_v3_packets_independently"
        ),
        "remaining_pre_review_limitation": (
            "no_fixed_object_level_authoritative_fact_source_registry"
        ),
        "claim_boundary": (
            "pre_review_integrity_freeze_only_no_new_model_behavior_internal_or_mechanism_evidence"
        ),
    }
    io_helpers.write_json(STAGE_PATH, stage)
    return stage


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
