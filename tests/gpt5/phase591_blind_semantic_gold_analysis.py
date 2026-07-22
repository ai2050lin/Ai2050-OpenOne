#!/usr/bin/env python3
"""Validate Phase591 human responses and keep all later gates closed until complete."""

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

import phase591_blind_semantic_gold_protocol as protocol  # noqa: E402


STATUS_PATH = protocol.OUT_DIR / "phase591_external_review_status.json"
ADJUDICATION_PATH = protocol.REVIEW_DIR / "phase591_adjudication_queue.jsonl.gz"
STAGE_PATH = protocol.OUT_DIR / "phase591_stage_summary.json"


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def valid_offset(value: Any, polarity: str, response: str) -> bool:
    if polarity == "unresolved":
        return value is None
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(response)


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
    expected_ids = set(packet_by_id)
    errors = 0
    accepted: dict[str, dict[str, Any]] = {}
    for row in rows:
        review_id = row.get("review_id")
        packet = packet_by_id.get(review_id)
        polarity = row.get("semantic_polarity")
        confidence = row.get("confidence_1_to_5")
        errors += int(packet is None)
        errors += int(row.get("reviewer_slot") != slot)
        errors += int(row.get("packet_digest") != manifest["packet_digests"][slot])
        errors += int(polarity not in protocol.POLARITY_LABELS)
        errors += int(row.get("factuality") not in protocol.FACTUALITY_LABELS)
        errors += int(row.get("factuality_basis") not in protocol.FACTUALITY_BASES)
        errors += int(not str(row.get("factuality_evidence") or "").strip())
        errors += int(not isinstance(row.get("negation_scope_text"), list))
        errors += int(not isinstance(row.get("condition_scope_text"), list))
        errors += int(not isinstance(row.get("has_contrast"), bool))
        errors += int(not isinstance(row.get("response_complete"), bool))
        errors += int(
            not isinstance(confidence, int)
            or isinstance(confidence, bool)
            or not protocol.CONFIDENCE_MIN <= confidence <= protocol.CONFIDENCE_MAX
        )
        errors += int(not str(row.get("rationale") or "").strip())
        errors += int(row.get("attestation") != protocol.REVIEW_ATTESTATION)
        errors += int(not str(row.get("reviewed_at") or "").strip())
        if packet is not None and polarity in protocol.POLARITY_LABELS:
            errors += int(
                not valid_offset(
                    row.get("first_decisive_character_offset"),
                    polarity,
                    packet["response"],
                )
            )
        if packet is not None:
            for field in ("negation_scope_text", "condition_scope_text"):
                values = row.get(field)
                if isinstance(values, list):
                    errors += sum(
                        not isinstance(value, str) or value not in packet["response"]
                        for value in values
                    )
        if packet is not None and review_id not in accepted:
            accepted[review_id] = row
    errors += len(expected_ids - set(ids))
    errors += len(set(ids) - expected_ids)
    errors += len(ids) - len(set(ids))
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


def majority_value(values: list[Any]) -> tuple[Any, int]:
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    return value, count


def label_consensus(rows: list[dict[str, Any]]) -> tuple[str | None, str]:
    labels = [row["semantic_polarity"] for row in rows]
    label, count = majority_value(labels)
    if count == 3:
        return str(label), "unanimous"
    if count == 2:
        minority = next(value for value in labels if value != label)
        if label != "unresolved" and minority == "unresolved":
            return str(label), "nonopposed_majority"
        return None, "semantic_label_conflict"
    return None, "semantic_label_conflict"


def analyze() -> dict[str, Any]:
    manifest = read_json(protocol.PROTOCOL_PATH)
    packets: dict[str, dict[str, dict[str, Any]]] = {}
    results: list[dict[str, Any]] = []
    completed: dict[str, dict[str, dict[str, Any]]] = {}
    for slot in protocol.REVIEWER_SLOTS:
        packet_rows = read_jsonl_gz(protocol.packet_path(slot))
        packets[slot] = {row["review_id"]: row for row in packet_rows}
        result, rows = validate_completed(slot, manifest, packets[slot])
        results.append(result)
        if result["structurally_valid"]:
            completed[slot] = rows

    valid_reviewer_ids = [
        result["reviewer_id"] for result in results if result["structurally_valid"]
    ]
    distinct_reviewers = bool(
        len(valid_reviewer_ids) == len(protocol.REVIEWER_SLOTS)
        and len(set(valid_reviewer_ids)) == len(protocol.REVIEWER_SLOTS)
    )
    consensus_counts: Counter[str] = Counter()
    adjudication_rows: list[dict[str, Any]] = []
    accepted_consensus_count = 0
    if distinct_reviewers and len(completed) == len(protocol.REVIEWER_SLOTS):
        reference_ids = sorted(next(iter(packets.values())))
        for review_id in reference_ids:
            rows = [completed[slot][review_id] for slot in protocol.REVIEWER_SLOTS]
            label, label_status = label_consensus(rows)
            if label is None:
                status = label_status
            else:
                label_rows = [row for row in rows if row["semantic_polarity"] == label]
                offsets = [row["first_decisive_character_offset"] for row in label_rows]
                _, offset_count = majority_value(offsets)
                factuality, factuality_count = majority_value(
                    [row["factuality"] for row in rows]
                )
                if offset_count < 2:
                    status = "decisive_boundary_conflict"
                elif factuality_count < 2:
                    status = "factuality_conflict"
                else:
                    status = "accepted_" + label_status
                    accepted_consensus_count += 1
            consensus_counts[status] += 1
            if not status.startswith("accepted_"):
                source = packets[protocol.REVIEWER_SLOTS[0]][review_id]
                adjudication_rows.append(
                    {
                        "schema_version": "phase591_semantic_adjudication_item.v1",
                        "phase_id": protocol.PHASE,
                        "review_id": review_id,
                        "prompt": source["prompt"],
                        "response": source["response"],
                        "adjudication_reason": status,
                        "prior_reviewer_answers_hidden": True,
                    }
                )
    else:
        consensus_counts["pending_three_independent_reviewers"] = protocol.EXPECTED_ITEM_COUNT

    write_jsonl_gz(ADJUDICATION_PATH, adjudication_rows)
    human_gold_complete = bool(
        distinct_reviewers
        and accepted_consensus_count == protocol.EXPECTED_ITEM_COUNT
        and not adjudication_rows
    )
    status = {
        "schema_version": "phase591_external_review_status.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "required_distinct_reviewer_count": len(protocol.REVIEWER_SLOTS),
        "completed_structurally_valid_reviewer_count": len(completed),
        "reviewer_results": results,
        "distinct_reviewer_identity_pass": distinct_reviewers,
        "consensus_status_counts": dict(sorted(consensus_counts.items())),
        "accepted_consensus_item_count": accepted_consensus_count,
        "adjudication_item_count": len(adjudication_rows),
        "independent_human_semantic_gold_complete": human_gold_complete,
        "private_answer_key_read": False,
        "phase590_sealed_cases_read": False,
        "machine_or_agent_annotation_substituted": False,
    }
    write_json(STATUS_PATH, status)

    stage = {
        "schema_version": "phase591_stage_summary.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "gold_gate_complete" if human_gold_complete else "blocked_pending_external_human_review",
        "denominators": {
            "blind_review_item_count": protocol.EXPECTED_ITEM_COUNT,
            "required_independent_reviewer_count": len(protocol.REVIEWER_SLOTS),
            "required_independent_human_labels": protocol.EXPECTED_ITEM_COUNT
            * len(protocol.REVIEWER_SLOTS),
            "completed_structurally_valid_human_labels": protocol.EXPECTED_ITEM_COUNT
            * len(completed),
            "model_case_count_consumed": 0,
            "internal_state_case_count_consumed": 0,
            "sealed_case_count_consumed": 0,
        },
        "authorization": {
            "develop_automatic_observer_from_discovery_gold": human_gold_complete,
            "evaluate_confirmation_or_heldout": False,
            "run_long_natural_generation": False,
            "run_qwen3": False,
            "run_glm4": False,
            "run_deepseek7b": False,
            "run_open_internal_trace": False,
            "run_causal_intervention": False,
            "read_private_answer_key": human_gold_complete,
            "read_phase590_sealed_set": False,
        },
        "automatic_execution_now": False,
        "next_required_external_action": (
            "none_gold_gate_complete"
            if human_gold_complete
            else "three_distinct_people_complete_their_288_item_response_templates_independently"
        ),
        "claim_boundary": (
            "review_workflow_and_blocking_gate_only_no_new_model_behavior_internal_or_mechanism_evidence"
        ),
    }
    write_json(STAGE_PATH, stage)
    return stage


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
