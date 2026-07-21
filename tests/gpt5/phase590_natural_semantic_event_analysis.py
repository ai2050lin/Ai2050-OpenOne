#!/usr/bin/env python3
"""Aggregate Phase590 model summaries without reading sealed rows."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase590_natural_semantic_event as runner  # noqa: E402
import phase590_natural_semantic_event_protocol as protocol  # noqa: E402


DECISION_PATH = protocol.OUT_DIR / "phase590_natural_semantic_event_decision.json"
ERROR_AUDIT_PATH = protocol.OUT_DIR / "phase590_observer_error_audit.json"
BLIND_REVIEW_PATH = protocol.OUT_DIR / "phase590_blind_review_queue.jsonl.gz"
BLIND_REVIEW_KEY_PATH = protocol.OUT_DIR / "protocol/private/phase590_blind_review_key.jsonl.gz"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_rows(model: str) -> list[dict[str, Any]]:
    return [
        row
        for row in iter_jsonl(runner.paths(model)["rows"])
        if row.get("row_kind") == "natural_generation"
    ]


def build_error_audit() -> dict[str, Any]:
    negative_lexeme = re.compile(
        r"\b(?:no|not|never|cannot|can't|unsafe|toxic|poisonous|dangerous|harmful)\b",
        re.I,
    )
    food_lexeme = re.compile(
        r"\b(?:food|eat|eaten|edible|consume|consumed|consumption|diet|meal|ingredient)\b",
        re.I,
    )
    model_audits: dict[str, Any] = {}
    for model in protocol.MODELS:
        rows = _generation_rows(model)
        by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            by_case[row["case_id"]][row["execution_repeat"]] = row
        first = [row for row in rows if row["execution_repeat"] == protocol.NOOP_REPEATS[0]]
        split_metrics: dict[str, Any] = {}
        for split in protocol.OPEN_SPLITS:
            split_rows = [row for row in first if row["split"] == split]
            exact_repeat = 0
            for row in split_rows:
                pair = by_case[row["case_id"]]
                exact_repeat += bool(
                    set(pair) == set(protocol.NOOP_REPEATS)
                    and pair[protocol.NOOP_REPEATS[0]]["normalized_generated"]
                    == pair[protocol.NOOP_REPEATS[1]]["normalized_generated"]
                )
            likely_negation_scope_miss = [
                row
                for row in split_rows
                if negative_lexeme.search(row["normalized_generated"])
                and food_lexeme.search(row["normalized_generated"])
                and row["semantic_polarity"] != "negative"
            ]
            lower_bound_clear_negative_group_contradiction = [
                row
                for row in split_rows
                if row["expected_polarity"] == "negative"
                and row["semantic_polarity"] == "positive"
                and not negative_lexeme.search(row["normalized_generated"])
            ]
            split_metrics[split] = {
                "case_count": len(split_rows),
                "exact_text_repeat_rate": exact_repeat / max(1, len(split_rows)),
                "max_new_token_boundary_rate": sum(
                    row["generated_token_count"] >= protocol.MAX_NEW_TOKENS
                    for row in split_rows
                )
                / max(1, len(split_rows)),
                "parser_event_counts": dict(Counter(row["semantic_event"] for row in split_rows)),
                "likely_negation_scope_miss_count": len(likely_negation_scope_miss),
                "likely_negation_scope_miss_rate": len(likely_negation_scope_miss)
                / max(1, len(split_rows)),
                "lower_bound_clear_negative_group_contradiction_count": len(
                    lower_bound_clear_negative_group_contradiction
                ),
                "lower_bound_clear_negative_group_contradiction_rate": len(
                    lower_bound_clear_negative_group_contradiction
                )
                / max(1, len(split_rows)),
            }
        model_audits[model] = {"split_metrics": split_metrics}
    return {
        "schema_version": "phase590_observer_error_audit.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "posthoc_diagnostic_only": True,
        "does_not_change_frozen_gate": True,
        "model_audits": model_audits,
        "interpretation_limits": {
            "negation_scope_miss_is_lexical_lower_bound_not_manual_gold": True,
            "clear_contradiction_is_lower_bound_not_total_model_error": True,
            "independent_human_gold_standard_available": False,
        },
        "sealed_split_read": False,
    }


def build_blind_review_queue() -> dict[str, Any]:
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for model in protocol.MODELS:
        rows = [
            row
            for row in _generation_rows(model)
            if row["execution_repeat"] == protocol.NOOP_REPEATS[0]
        ]
        for split in protocol.OPEN_SPLITS:
            for group in protocol.OBJECT_LABELS:
                pool = [
                    row
                    for row in rows
                    if row["split"] == split and row["semantic_group"] == group
                ]
                ranked = sorted(
                    pool,
                    key=lambda row: hashlib.sha256(
                        f"phase590-blind-v1|{model}|{row['case_id']}".encode("utf-8")
                    ).hexdigest(),
                )
                for row in ranked[:8]:
                    review_id = "p590r_" + hashlib.sha256(
                        f"{model}|{row['case_id']}".encode("utf-8")
                    ).hexdigest()[:16]
                    public_rows.append(
                        {
                            "schema_version": "phase590_blind_review_item.v1",
                            "review_id": review_id,
                            "prompt": row["raw_prompt"],
                            "response": row["generated_text"],
                            "review_fields": {
                                "semantic_polarity": None,
                                "first_decisive_character_offset": None,
                                "response_complete": None,
                                "notes": None,
                            },
                        }
                    )
                    private_rows.append(
                        {
                            "review_id": review_id,
                            "model": model,
                            "case_id": row["case_id"],
                            "split": split,
                            "semantic_group": group,
                            "expected_polarity": row["expected_polarity"],
                            "frozen_parser_polarity": row["semantic_polarity"],
                        }
                    )
    public_rows.sort(key=lambda row: row["review_id"])
    private_rows.sort(key=lambda row: row["review_id"])
    write_jsonl(BLIND_REVIEW_PATH, public_rows)
    write_jsonl(BLIND_REVIEW_KEY_PATH, private_rows)
    return {
        "schema_version": "phase590_blind_review_queue_manifest.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "review_item_count": len(public_rows),
        "reviewers_required": 3,
        "reviewers_completed": 0,
        "public_queue_sha256": sha256_file(BLIND_REVIEW_PATH),
        "private_key_sha256": sha256_file(BLIND_REVIEW_KEY_PATH),
        "model_group_split_metadata_hidden_from_review_queue": True,
        "independent_human_gold_standard_available": False,
    }


def analyze() -> dict[str, Any]:
    summaries = {model: read_json(runner.paths(model)["summary"]) for model in protocol.MODELS}
    qualified = [model for model, summary in summaries.items() if summary["automatic_observer_qualified"]]
    error_audit = build_error_audit()
    write_json(ERROR_AUDIT_PATH, error_audit)
    blind_review = build_blind_review_queue()
    decision = {
        "schema_version": "phase590_natural_semantic_event_decision.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model_execution_order": list(protocol.MODELS),
        "model_summaries": summaries,
        "qualified_model_count": len(qualified),
        "qualified_models": qualified,
        "next_stage": (
            "exploratory_open_hidden_semantic_event_trace"
            if qualified
            else "stop_internal_scan_and_repair_semantic_observer"
        ),
        "mechanism_claim_authorized": False,
        "causal_intervention_authorized": False,
        "sealed_split_read": False,
        "observer_error_audit_path": str(ERROR_AUDIT_PATH.relative_to(ROOT)),
        "blind_review_queue": blind_review,
        "primary_conclusion": (
            "At least one frozen automatic natural-generation observer passed all open gates; "
            "only an exploratory open hidden trace is authorized."
            if qualified
            else "No frozen automatic natural-generation observer passed all open gates; "
            "internal search remains blocked by the measurement layer."
        ),
    }
    write_json(DECISION_PATH, decision)
    return decision


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
