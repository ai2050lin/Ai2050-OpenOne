#!/usr/bin/env python3
"""Freeze three independent human-review packets for Phase591."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase591"
SOURCE_DIR = ROOT / "tests/gpt5/result/phase590_natural_semantic_event"
SOURCE_QUEUE_PATH = SOURCE_DIR / "phase590_blind_review_queue.jsonl.gz"

OUT_DIR = ROOT / "tests/gpt5/result/phase591_semantic_gold"
REVIEW_DIR = OUT_DIR / "external_review"
PROTOCOL_PATH = OUT_DIR / "phase591_blind_semantic_gold_protocol.json"
AUDIT_PATH = OUT_DIR / "phase591_static_audit.json"

REVIEWER_SLOTS = ("reviewer_a", "reviewer_b", "reviewer_c")
POLARITY_LABELS = ("positive", "negative", "conditional", "unresolved")
FACTUALITY_LABELS = ("supported", "contradicted", "conditional", "uncertain")
FACTUALITY_BASES = ("ordinary_knowledge", "external_reference", "uncertain")
CONFIDENCE_MIN = 1
CONFIDENCE_MAX = 5
EXPECTED_ITEM_COUNT = 288

REVIEW_ATTESTATION = (
    "I reviewed this packet independently without consulting another reviewer's "
    "answers, model identity, semantic group, expected polarity, frozen parser output, "
    "or the private answer key."
)

SLOT_SALTS = {
    "reviewer_a": "phase591-independent-review-a-v1",
    "reviewer_b": "phase591-independent-review-b-v1",
    "reviewer_c": "phase591-independent-review-c-v1",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def packet_path(slot: str) -> Path:
    return REVIEW_DIR / f"{slot}_packet.jsonl.gz"


def response_template_path(slot: str) -> Path:
    return REVIEW_DIR / f"{slot}_response_template.jsonl"


def completed_response_path(slot: str) -> Path:
    return REVIEW_DIR / f"{slot}_completed.jsonl"


def instruction(slot: str) -> dict[str, Any]:
    return {
        "phase_id": PHASE,
        "reviewer_slot": slot,
        "task": (
            "Judge the response's stated proposition about ordinary human food use. "
            "Label the response itself, not the frozen expected category."
        ),
        "polarity_labels": {
            "positive": "The response clearly states ordinary human food use.",
            "negative": "The response clearly rejects ordinary human food use.",
            "conditional": "Food use depends on part, processing, dose, culture, or another explicit condition.",
            "unresolved": "The response does not support one of the other labels or is too incomplete.",
        },
        "factuality_labels": {
            "supported": "The stated proposition is ordinary factual knowledge.",
            "contradicted": "The stated proposition is factually wrong.",
            "conditional": "Its factuality depends on an explicit condition or object part.",
            "uncertain": "The reviewer cannot determine factuality confidently.",
        },
        "factuality_basis_labels": {
            "ordinary_knowledge": "The judgment uses stable ordinary knowledge.",
            "external_reference": "The judgment was checked against an external reference named in factuality_evidence.",
            "uncertain": "No sufficiently reliable factual basis is available.",
        },
        "first_decisive_character_offset_rule": (
            "Use a zero-based character offset at the first character completing the earliest decisive "
            "surface expression; use null only for unresolved."
        ),
        "scope_rule": (
            "Copy exact response substrings governed by negation or a condition; use an empty list "
            "when no such scope exists."
        ),
        "required_independent_reviewer_count": 3,
        "attestation": REVIEW_ATTESTATION,
    }


def response_template(packet: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "review_id": row["review_id"],
            "reviewer_slot": row["reviewer_slot"],
            "packet_digest": row["packet_digest"],
            "reviewer_id": None,
            "semantic_polarity": None,
            "negation_scope_text": None,
            "condition_scope_text": None,
            "has_contrast": None,
            "response_complete": None,
            "first_decisive_character_offset": None,
            "factuality": None,
            "factuality_basis": None,
            "factuality_evidence": None,
            "confidence_1_to_5": None,
            "rationale": None,
            "attestation": REVIEW_ATTESTATION,
            "reviewed_at": None,
        }
        for row in packet
    ]


def validate_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row.get("review_id") for row in rows]
    forbidden_keys = {
        "model",
        "case_id",
        "split",
        "semantic_group",
        "expected_polarity",
        "frozen_parser_polarity",
    }
    audit = {
        "schema_version": "phase591_blind_semantic_gold_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "source_item_count": len(rows),
        "unique_review_id_count": len(set(ids)),
        "missing_prompt_count": sum(not str(row.get("prompt") or "").strip() for row in rows),
        "missing_response_count": sum(not str(row.get("response") or "").strip() for row in rows),
        "hidden_metadata_key_count": sum(
            len(forbidden_keys & set(row)) for row in rows
        ),
        "private_answer_key_read": False,
        "phase590_sealed_cases_read": False,
    }
    audit["valid"] = bool(
        len(rows) == EXPECTED_ITEM_COUNT
        and len(set(ids)) == EXPECTED_ITEM_COUNT
        and audit["missing_prompt_count"] == 0
        and audit["missing_response_count"] == 0
        and audit["hidden_metadata_key_count"] == 0
    )
    return audit


def register() -> dict[str, Any]:
    source_rows = read_jsonl_gz(SOURCE_QUEUE_PATH)
    audit = validate_source(source_rows)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))

    packet_digests: dict[str, str] = {}
    packet_orders: dict[str, list[str]] = {}
    for slot in REVIEWER_SLOTS:
        ordered = sorted(
            source_rows,
            key=lambda row: sha256_text(SLOT_SALTS[slot] + "|" + row["review_id"]),
        )
        review_instruction = instruction(slot)
        packet_digest = sha256_text(
            canonical_json(review_instruction)
            + "\n"
            + "\n".join(canonical_json(row) for row in ordered)
        )
        packet = [
            {
                **row,
                "phase_id": PHASE,
                "reviewer_slot": slot,
                "packet_digest": packet_digest,
                "review_instruction": review_instruction,
            }
            for row in ordered
        ]
        write_jsonl_gz(packet_path(slot), packet)
        if not completed_response_path(slot).exists():
            write_jsonl(response_template_path(slot), response_template(packet))
        packet_digests[slot] = packet_digest
        packet_orders[slot] = [row["review_id"] for row in ordered]

    audit.update(
        {
            "packet_item_count_by_slot": {
                slot: len(packet_orders[slot]) for slot in REVIEWER_SLOTS
            },
            "packet_order_pairwise_distinct": len(
                {tuple(packet_orders[slot]) for slot in REVIEWER_SLOTS}
            )
            == len(REVIEWER_SLOTS),
            "packet_digest_count": len(set(packet_digests.values())),
        }
    )
    audit["valid"] = bool(
        audit["valid"]
        and audit["packet_order_pairwise_distinct"]
        and audit["packet_digest_count"] == len(REVIEWER_SLOTS)
        and all(
            count == EXPECTED_ITEM_COUNT
            for count in audit["packet_item_count_by_slot"].values()
        )
    )
    write_json(AUDIT_PATH, audit)

    frozen = {
        "schema_version": "phase591_blind_semantic_gold_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Independent natural semantic gold review gate",
        "source_queue_path": str(SOURCE_QUEUE_PATH.relative_to(ROOT)),
        "source_queue_sha256": sha256_file(SOURCE_QUEUE_PATH),
        "source_item_count": len(source_rows),
        "reviewer_slots": list(REVIEWER_SLOTS),
        "required_distinct_reviewer_count": len(REVIEWER_SLOTS),
        "polarity_labels": list(POLARITY_LABELS),
        "factuality_labels": list(FACTUALITY_LABELS),
        "factuality_bases": list(FACTUALITY_BASES),
        "packet_digests": packet_digests,
        "completed_response_paths": {
            slot: str(completed_response_path(slot).relative_to(ROOT))
            for slot in REVIEWER_SLOTS
        },
        "consensus_rule": {
            "unanimous_label": "accept",
            "two_same_plus_unresolved": "accept_nonopposed_majority",
            "positive_negative_conflict": "adjudicate",
            "conditional_positive_or_negative_conflict": "adjudicate",
            "unresolved_majority_with_decisive_minority": "adjudicate",
            "decisive_boundary_requires_exact_majority": True,
            "factuality_requires_simple_majority": True,
        },
        "evidence_policy": {
            "three_distinct_external_people_required": True,
            "model_or_agent_self_annotation_cannot_substitute": True,
            "private_answer_key_must_remain_unread_until_consensus_complete": True,
            "phase590_sealed_set_must_remain_unread": True,
            "no_model_execution_before_gold_gate": True,
            "no_automatic_observer_development_before_gold_gate": True,
            "no_internal_trace_or_causal_intervention_before_gold_gate": True,
        },
        "static_audit_sha256": sha256_file(AUDIT_PATH),
    }
    write_json(PROTOCOL_PATH, frozen)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
