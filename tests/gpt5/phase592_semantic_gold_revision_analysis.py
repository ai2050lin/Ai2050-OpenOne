#!/usr/bin/env python3
"""Validate Phase592 reviews, split semantic gold from event-anchor qualification."""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase591_blind_semantic_gold_protocol as io_helpers  # noqa: E402
import phase592_semantic_gold_revision_protocol as protocol  # noqa: E402


STATUS_PATH = protocol.OUT_DIR / "phase592_external_review_status.json"
STAGE_PATH = protocol.OUT_DIR / "phase592_stage_summary.json"
ADJUDICATION_PACKET_PATH = protocol.REVIEW_DIR / "adjudicator_packet_v2.jsonl.gz"
ADJUDICATION_TEMPLATE_PATH = protocol.REVIEW_DIR / "adjudicator_response_template_v2.jsonl"
GOLD_PATH = protocol.OUT_DIR / "phase592_resolved_semantic_gold.jsonl.gz"


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
                "reviewed_item_count": 0,
                "structural_error_count": protocol.EXPECTED_ITEM_COUNT,
                "reviewer_id": None,
            },
            {},
        )
    rows = read_jsonl(path)
    ids = [row.get("review_id") for row in rows]
    reviewer_ids = {row.get("reviewer_id") for row in rows}
    errors = 0
    accepted: dict[str, dict[str, Any]] = {}
    for row in rows:
        review_id = row.get("review_id")
        packet = packet_by_id.get(review_id)
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
        if packet is not None and review_id not in accepted:
            accepted[review_id] = row
    expected_ids = set(packet_by_id)
    errors += len(expected_ids - set(ids))
    errors += len(set(ids) - expected_ids)
    errors += len(ids) - len(set(ids))
    if rows:
        for batch_index in range(1, protocol.BATCH_COUNT + 1):
            batch_id = f"batch_{batch_index:02d}"
            batch_rows = [row for row in rows if row.get("batch_id") == batch_id]
            errors += abs(len(batch_rows) - protocol.ITEMS_PER_BATCH)
            starts = {row.get("batch_started_at") for row in batch_rows}
            ends = {row.get("batch_completed_at") for row in batch_rows}
            errors += int(len(starts) != 1 or None in starts or "" in starts)
            errors += int(len(ends) != 1 or None in ends or "" in ends)
            if len(starts) == 1 and len(ends) == 1 and None not in starts | ends and "" not in starts | ends:
                try:
                    start_time = datetime.fromisoformat(str(next(iter(starts))).replace("Z", "+00:00"))
                    end_time = datetime.fromisoformat(str(next(iter(ends))).replace("Z", "+00:00"))
                    errors += int(end_time < start_time)
                except ValueError:
                    errors += 1
    structurally_valid = bool(
        len(rows) == protocol.EXPECTED_ITEM_COUNT
        and len(reviewer_ids) == 1
        and None not in reviewer_ids
        and errors == 0
    )
    reviewer_id = next(iter(reviewer_ids)) if len(reviewer_ids) == 1 else None
    return (
        {
            "reviewer_slot": slot,
            "completed_file_present": True,
            "structurally_valid": structurally_valid,
            "reviewed_item_count": len(rows),
            "structural_error_count": errors,
            "reviewer_id": reviewer_id,
        },
        accepted if structurally_valid else {},
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


def overlapping_anchor(
    rows: list[dict[str, Any]], semantic_label: str
) -> tuple[bool, list[int] | None]:
    eligible = [
        row
        for row in rows
        if row["semantic_polarity"] == semantic_label
        and row["response_complete"]
        and not row["later_text_changes_final_semantics"]
        and row["decisive_span_start"] is not None
        and row["decisive_span_end"] is not None
    ]
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            start = max(left["decisive_span_start"], right["decisive_span_start"])
            end = min(left["decisive_span_end"], right["decisive_span_end"])
            if start < end:
                return True, [start, end]
    return False, None


def make_adjudication_packet(
    items: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    ordered = sorted(items, key=lambda row: io_helpers.sha256_text("phase592-adj|" + row["review_id"]))
    instruction = {
        "phase_id": protocol.PHASE,
        "task": "Independently resolve semantic polarity and factuality without seeing prior labels.",
        "semantic_labels": list(protocol.POLARITY_LABELS),
        "factuality_labels": list(protocol.FACTUALITY_LABELS),
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
                "completed_file_present": False,
                "structurally_valid": False,
                "reviewed_item_count": 0,
                "structural_error_count": 0,
                "adjudicator_id": None,
                "required": False,
            },
            {},
        )
    path = protocol.adjudicator_completed_path()
    if not path.exists() or not ADJUDICATION_PACKET_PATH.exists():
        return (
            {
                "completed_file_present": path.exists(),
                "structurally_valid": False,
                "reviewed_item_count": 0,
                "structural_error_count": len(adjudication_items),
                "adjudicator_id": None,
                "required": True,
            },
            {},
        )
    packet = read_jsonl_gz(ADJUDICATION_PACKET_PATH)
    packet_by_id = {row["review_id"]: row for row in packet}
    rows = read_jsonl(path)
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
    return (
        {
            "completed_file_present": True,
            "structurally_valid": structurally_valid,
            "reviewed_item_count": len(rows),
            "structural_error_count": errors,
            "adjudicator_id": adjudicator_id,
            "required": True,
        },
        accepted if structurally_valid else {},
    )


def analyze() -> dict[str, Any]:
    manifest = read_json(protocol.PROTOCOL_PATH)
    packets: dict[str, dict[str, dict[str, Any]]] = {}
    valid_rows: dict[str, dict[str, dict[str, Any]]] = {}
    reviewer_results: list[dict[str, Any]] = []
    for slot in protocol.REVIEWER_SLOTS:
        packet_rows = read_jsonl_gz(protocol.packet_path(slot))
        packets[slot] = {row["review_id"]: row for row in packet_rows}
        result, rows = validate_completed(slot, manifest, packets[slot])
        reviewer_results.append(result)
        if result["structurally_valid"]:
            valid_rows[slot] = rows

    reviewer_ids = [
        result["reviewer_id"]
        for result in reviewer_results
        if result["structurally_valid"]
    ]
    three_distinct = bool(
        len(reviewer_ids) == 3 and len(set(reviewer_ids)) == 3
    )
    direct_resolved = 0
    anchor_qualified = 0
    direct_gold_rows: list[dict[str, Any]] = []
    workflow_counts: Counter[str] = Counter()
    adjudication_items: list[dict[str, Any]] = []
    if three_distinct and len(valid_rows) == 3:
        review_ids = sorted(next(iter(packets.values())))
        for review_id in review_ids:
            rows = [valid_rows[slot][review_id] for slot in protocol.REVIEWER_SLOTS]
            semantic, semantic_status = direct_consensus(
                [row["semantic_polarity"] for row in rows], "unresolved"
            )
            factuality, factuality_status = direct_consensus(
                [row["factuality"] for row in rows], "uncertain"
            )
            if semantic is not None and factuality is not None:
                direct_resolved += 1
                workflow_counts["directly_resolved"] += 1
                anchor_ok, anchor_span = overlapping_anchor(rows, semantic)
                anchor_qualified += int(anchor_ok and semantic != "unresolved")
                direct_gold_rows.append(
                    {
                        "schema_version": "phase592_resolved_semantic_gold_item.v1",
                        "phase_id": protocol.PHASE,
                        "review_id": review_id,
                        "semantic_polarity": semantic,
                        "factuality": factuality,
                        "resolution_mode": "direct_" + semantic_status,
                        "event_anchor_qualified": bool(
                            anchor_ok and semantic != "unresolved"
                        ),
                        "event_anchor_overlap_span": anchor_span,
                        "initial_reviewer_labels_preserved_in_completed_files": True,
                    }
                )
            else:
                workflow_counts["requires_independent_adjudication"] += 1
                source = packets[protocol.REVIEWER_SLOTS[0]][review_id]
                adjudication_items.append(
                    {
                        "schema_version": "phase592_adjudication_item.v1",
                        "review_id": review_id,
                        "prompt": source["prompt"],
                        "response": source["response"],
                        "semantic_conflict": semantic_status,
                        "factuality_conflict": factuality_status,
                        "prior_reviewer_labels_hidden": True,
                        "hidden_model_metadata_absent": True,
                    }
                )
        if adjudication_items:
            make_adjudication_packet(adjudication_items, manifest)
    else:
        workflow_counts["pending_three_independent_reviewers"] = protocol.EXPECTED_ITEM_COUNT

    adjudicator_result, adjudicated_rows = validate_adjudicator(
        adjudication_items, reviewer_ids
    )
    adjudicator_valid = adjudicator_result["structurally_valid"]
    adjudicated_resolved = len(adjudicated_rows) if adjudicator_valid else 0
    workflow_unresolved = protocol.EXPECTED_ITEM_COUNT - direct_resolved - adjudicated_resolved
    semantic_gold_complete = bool(
        three_distinct
        and direct_resolved + adjudicated_resolved == protocol.EXPECTED_ITEM_COUNT
        and workflow_unresolved == 0
    )
    gold_rows = list(direct_gold_rows)
    if adjudicator_valid:
        gold_rows.extend(
            {
                "schema_version": "phase592_resolved_semantic_gold_item.v1",
                "phase_id": protocol.PHASE,
                "review_id": review_id,
                "semantic_polarity": row["semantic_polarity"],
                "factuality": row["factuality"],
                "resolution_mode": "independent_adjudication",
                "event_anchor_qualified": False,
                "event_anchor_overlap_span": None,
                "initial_reviewer_labels_preserved_in_completed_files": True,
            }
            for review_id, row in sorted(adjudicated_rows.items())
        )
    if semantic_gold_complete:
        io_helpers.write_jsonl_gz(GOLD_PATH, sorted(gold_rows, key=lambda row: row["review_id"]))
    status = {
        "schema_version": "phase592_external_review_status.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "reviewer_results": reviewer_results,
        "three_distinct_reviewer_identity_pass": three_distinct,
        "workflow_status_counts": dict(sorted(workflow_counts.items())),
        "directly_resolved_item_count": direct_resolved,
        "adjudication_required_item_count": len(adjudication_items),
        "adjudicator_structurally_valid": adjudicator_valid,
        "adjudicator_result": adjudicator_result,
        "independently_adjudicated_item_count": adjudicated_resolved,
        "workflow_unresolved_item_count": workflow_unresolved,
        "semantic_gold_complete": semantic_gold_complete,
        "resolved_gold_artifact_written": semantic_gold_complete,
        "resolved_gold_artifact_sha256": (
            io_helpers.sha256_file(GOLD_PATH) if semantic_gold_complete else None
        ),
        "event_anchor_qualified_item_count": anchor_qualified,
        "event_anchor_is_separate_from_semantic_gold": True,
        "private_answer_key_read": False,
        "phase590_sealed_cases_read": False,
        "machine_or_agent_annotation_substituted": False,
    }
    io_helpers.write_json(STATUS_PATH, status)

    stage = {
        "schema_version": "phase592_stage_summary.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": (
            "semantic_gold_complete"
            if semantic_gold_complete
            else "blocked_pending_external_human_review"
        ),
        "denominators": {
            "blind_review_item_count": protocol.EXPECTED_ITEM_COUNT,
            "required_first_pass_human_labels": protocol.EXPECTED_ITEM_COUNT * 3,
            "completed_structurally_valid_first_pass_human_labels": len(valid_rows)
            * protocol.EXPECTED_ITEM_COUNT,
            "model_case_count_consumed": 0,
            "internal_state_case_count_consumed": 0,
            "sealed_case_count_consumed": 0,
        },
        "authorization": {
            "read_private_answer_key": semantic_gold_complete,
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
        "automatic_execution_now": False,
        "next_required_external_action": (
            "none_semantic_gold_complete"
            if semantic_gold_complete
            else "three_distinct_people_complete_phase592_v2_packets_independently"
        ),
        "claim_boundary": (
            "pre_execution_review_contract_revision_only_no_new_model_behavior_internal_or_mechanism_evidence"
        ),
    }
    io_helpers.write_json(STAGE_PATH, stage)
    return stage


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
