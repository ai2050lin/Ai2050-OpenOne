#!/usr/bin/env python3
"""Diagnose Phase1010 natural rollout formatting without changing gates."""
from __future__ import annotations

import json
import re
import sys
from typing import Any

import numpy as np


from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import tokenizer_for
from phase1010_output_type_protocol import (
    FAMILIES,
    MODELS,
    OUT_ROOT,
    OUTPUT_TYPES,
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
            "schema_version": "phase1010_rollout_surface_row.v1",
            "phase": PHASE,
            "model": model_name,
            "family": case["family"],
            "output_type": case["output_type"],
            "record_id": case["record_id"],
            "gold": case["gold"],
            "decoded_generation": decoded,
            "extracted_label": extracted,
            "label_case_insensitive_hit": semantic_hit,
            "flexible_full_protocol_hit": bool(
                full is not None
                and full.group(1).casefold() == case["gold"].casefold()
            ),
            "frozen_strict_exact": bool(behavior_row["natural_exact"]),
            "used_for_scan_or_causal_qualification": False,
        })
    panel_rates = []
    for family in FAMILIES:
        for output_type in OUTPUT_TYPES:
            selected = [
                row
                for row in rows
                if row["family"] == family
                and row["output_type"] == output_type
            ]
            panel_rates.append({
                "family": family,
                "output_type": output_type,
                "n": len(selected),
                "label_case_insensitive_rate": float(np.mean([
                    row["label_case_insensitive_hit"]
                    for row in selected
                ])),
                "flexible_full_protocol_rate": float(np.mean([
                    row["flexible_full_protocol_hit"]
                    for row in selected
                ])),
                "frozen_strict_exact_rate": float(np.mean([
                    row["frozen_strict_exact"] for row in selected
                ])),
            })
    summary = {
        "schema_version": "phase1010_rollout_surface_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(rows),
        "label_case_insensitive_rate": float(np.mean([
            row["label_case_insensitive_hit"] for row in rows
        ])),
        "flexible_full_protocol_rate": float(np.mean([
            row["flexible_full_protocol_hit"] for row in rows
        ])),
        "frozen_strict_exact_rate": float(np.mean([
            row["frozen_strict_exact"] for row in rows
        ])),
        "panel_rates": panel_rates,
        "posthoc_policy": (
            "diagnoses output-surface mismatch only; frozen behavior, "
            "scan, and causal qualification remain unchanged"
        ),
    }
    root = OUT_ROOT / "behavior" / model_name
    write_jsonl(root / "rollout_surface_rows.jsonl", rows)
    write_json(root / "rollout_surface_summary.json", summary)
    return summary


def main() -> None:
    summaries = [audit_model(model_name) for model_name in MODELS]
    result = {
        "schema_version": "phase1010_rollout_surface_all.v1",
        "phase": PHASE,
        "models": summaries,
        "strict_exact_is_not_semantic_accuracy": True,
    }
    write_json(
        OUT_ROOT / "behavior" / "rollout_surface_summary.json",
        result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
