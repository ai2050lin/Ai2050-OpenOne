#!/usr/bin/env python3
"""Audit Phase1009 natural rollouts without changing frozen qualification."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import tokenizer_for
from phase1009_crossfamily_response_protocol import (
    FAMILIES,
    MODELS,
    OUT_ROOT,
    PHASE,
    read_jsonl,
    write_json,
    write_jsonl,
)


ANSWER_PATTERN = re.compile(
    r"Answer\s*:\s*([A-Za-z]+)\s+done",
    flags=re.IGNORECASE,
)
FLEXIBLE_FULL_PATTERN = re.compile(
    r"^\s*Answer\s*:\s*([A-Za-z]+)\s+done[\s.!?]*$",
    flags=re.IGNORECASE,
)


def audit_model(model_name: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(model_name)
    cases = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    behavior = read_jsonl(
        OUT_ROOT / "behavior" / model_name / "rows.jsonl"
    )
    case_by_id = {case["record_id"]: case for case in cases}
    rows = []
    for behavior_row in behavior:
        case = case_by_id[behavior_row["record_id"]]
        decoded = tokenizer.decode(
            behavior_row["generated_ids"],
            skip_special_tokens=True,
        )
        search = ANSWER_PATTERN.search(decoded)
        full = FLEXIBLE_FULL_PATTERN.fullmatch(decoded)
        extracted = None if search is None else search.group(1)
        semantic_hit = bool(
            extracted is not None
            and extracted.casefold() == case["gold"].casefold()
        )
        rows.append({
            "schema_version": "phase1009_rollout_surface_audit_row.v1",
            "phase": PHASE,
            "model": model_name,
            "family": case["family"],
            "split": case["split"],
            "template": int(case["template"]),
            "record_id": case["record_id"],
            "state": case["state"],
            "gold": case["gold"],
            "decoded_generation": decoded,
            "extracted_name": extracted,
            "name_case_insensitive_hit": semantic_hit,
            "flexible_full_protocol_hit": bool(
                full is not None
                and full.group(1).casefold() == case["gold"].casefold()
            ),
            "frozen_strict_exact": bool(behavior_row["natural_exact"]),
            "used_for_atlas_qualification": False,
        })
    family_rows = []
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        family_rows.append({
            "family": family,
            "n": len(selected),
            "name_case_insensitive_rate": float(np.mean([
                row["name_case_insensitive_hit"] for row in selected
            ])),
            "flexible_full_protocol_rate": float(np.mean([
                row["flexible_full_protocol_hit"] for row in selected
            ])),
            "frozen_strict_exact_rate": float(np.mean([
                row["frozen_strict_exact"] for row in selected
            ])),
        })
    summary = {
        "schema_version": "phase1009_rollout_surface_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(rows),
        "name_case_insensitive_rate": float(np.mean([
            row["name_case_insensitive_hit"] for row in rows
        ])),
        "flexible_full_protocol_rate": float(np.mean([
            row["flexible_full_protocol_hit"] for row in rows
        ])),
        "frozen_strict_exact_rate": float(np.mean([
            row["frozen_strict_exact"] for row in rows
        ])),
        "family_rates": family_rows,
        "posthoc_policy": (
            "This audit diagnoses output-surface mismatch only. It does not "
            "change frozen semantic qualification, candidate selection, or "
            "causal denominators."
        ),
    }
    root = OUT_ROOT / "behavior" / model_name
    write_jsonl(root / "rollout_surface_audit_rows.jsonl", rows)
    write_json(root / "rollout_surface_audit_summary.json", summary)
    return summary


def main() -> None:
    summaries = [audit_model(model_name) for model_name in MODELS]
    aggregate = {
        "schema_version": "phase1009_rollout_surface_audit_all.v1",
        "phase": PHASE,
        "models": summaries,
        "strict_exact_is_not_semantic_accuracy": True,
    }
    write_json(
        OUT_ROOT / "behavior" / "rollout_surface_audit_summary.json",
        aggregate,
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
