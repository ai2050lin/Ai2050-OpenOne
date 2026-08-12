#!/usr/bin/env python3
"""Audit Phase 1001 shape rollouts without changing the frozen mechanism.

The original strict parser reads the first alphabetic word. Some intervened
generations start with an entity name and mention a shape later. This audit
rescans the already-recorded rollout for the first shape word anywhere in the
observed suffix. It performs no model run, no event selection, and no threshold
change.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1001_attention_physical_decomposition"
    / "structural_extrapolation"
    / "two_entity_shape"
)
INPUT_PATH = RESULT_ROOT / "natural_rows.jsonl"
OUTPUT_PATH = RESULT_ROOT / "rollout_rescore_audit.json"
VALUE_TO_LABEL = {
    "round": "red",
    "square": "blue",
    "oval": "green",
    "triangle": "yellow",
}
PATTERN = re.compile(r"\b(round|square|oval|triangle)\b", re.IGNORECASE)


def rate(rows: list[dict], predicate) -> float:
    return sum(bool(predicate(row)) for row in rows) / max(len(rows), 1)


def main() -> None:
    rows = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[str, list[dict]] = defaultdict(list)
    rescored_rows = []
    for row in rows:
        match = PATTERN.search(row["text"])
        any_candidate_prediction = (
            VALUE_TO_LABEL[match.group(1).lower()] if match else None
        )
        rescored = {
            **row,
            "any_candidate_prediction": any_candidate_prediction,
            "candidate_word_offset": match.start() if match else None,
        }
        groups[row["condition"]].append(rescored)
        rescored_rows.append(rescored)

    summaries = {}
    for condition, values in groups.items():
        summaries[condition] = {
            "n": len(values),
            "strict_first_word_source_rate": rate(
                values,
                lambda row: row["prediction"] == row["source_gold"],
            ),
            "strict_first_word_target_rate": rate(
                values,
                lambda row: row["prediction"] == row["target_gold"],
            ),
            "any_candidate_source_rate": rate(
                values,
                lambda row: (
                    row["any_candidate_prediction"] == row["source_gold"]
                ),
            ),
            "any_candidate_target_rate": rate(
                values,
                lambda row: (
                    row["any_candidate_prediction"] == row["target_gold"]
                ),
            ),
            "any_candidate_seen_rate": rate(
                values,
                lambda row: row["any_candidate_prediction"] is not None,
            ),
        }

    source_rate = summaries["source_do"]["any_candidate_source_rate"]
    target_rate = summaries["source_plus_frozen_head_restore"][
        "any_candidate_target_rate"
    ]
    payload = {
        "schema_version": "phase1001_structural_rollout_rescore_audit.v1",
        "phase": 1001,
        "model": "qwen3",
        "family": "two_entity_shape",
        "selection_changed": False,
        "model_rerun": False,
        "thresholds_changed": False,
        "observed_rollout_max_new_tokens": 4,
        "parser": "first recognized shape word anywhere in recorded suffix",
        "condition_summaries": summaries,
        "gate_thresholds": {
            "source_do_source_rate": 0.80,
            "restore_target_rate": 0.50,
        },
        "gate_checks": {
            "source_do_source_rate": source_rate >= 0.80,
            "restore_target_rate": target_rate >= 0.50,
        },
        "rescore_gate_pass": (
            source_rate >= 0.80 and target_rate >= 0.50
        ),
        "interpretation": (
            "Allowing a candidate after an entity-name prefix does not rescue "
            "the shape rollout. Candidate-level extrapolation remains positive, "
            "but natural autoregressive extrapolation remains a NO-GO."
        ),
        "remaining_limit": (
            "The stored rollout has only four generated tokens; longer variable-"
            "boundary generation requires a separate preregistered experiment."
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
