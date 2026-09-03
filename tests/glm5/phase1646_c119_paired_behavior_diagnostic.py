#!/usr/bin/env python3
"""C119-R: deterministic paired behavioral diagnosis of the frozen C118/C119 runs."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
C118 = RESULT / "phase1640_c118_identifiable_default_override_campaign"
C119 = RESULT / "phase1643_c119_identifiable_default_override_campaign"
OUT = C119 / "analysis/paired_behavior_diagnostic.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def unit_number(value: str) -> int:
    return int(value.rsplit("-", 1)[1])


def key(row: dict) -> tuple:
    return (
        unit_number(row["unit_id"]),
        row["default_factor"],
        row["hit_factor"],
        row["conflict_factor"],
        row["surface_factor"],
        row["output_format"],
    )


def summarize(pairs: list[tuple[dict, dict]]) -> dict:
    transitions = {
        "both_correct": 0,
        "c118_only_correct": 0,
        "c119_only_correct": 0,
        "both_wrong": 0,
    }
    for left, right in pairs:
        if left["correct"] and right["correct"]:
            transitions["both_correct"] += 1
        elif left["correct"]:
            transitions["c118_only_correct"] += 1
        elif right["correct"]:
            transitions["c119_only_correct"] += 1
        else:
            transitions["both_wrong"] += 1
    n = len(pairs)
    return {
        "n": n,
        "c118_accuracy": sum(left["correct"] for left, _ in pairs) / n,
        "c119_accuracy": sum(right["correct"] for _, right in pairs) / n,
        "accuracy_change": (
            sum(right["correct"] for _, right in pairs)
            - sum(left["correct"] for left, _ in pairs)
        ) / n,
        "paired_transitions": transitions,
    }


def main() -> None:
    left_rows = core.rows(C118 / "raw/qwen3_behavior_index.jsonl")
    right_rows = core.rows(C119 / "raw/qwen3_behavior_index.jsonl")
    left = {key(row): row for row in left_rows}
    right = {key(row): row for row in right_rows}
    if len(left) != 768 or set(left) != set(right):
        raise RuntimeError("C118/C119 factorial pairing is not exact")
    pairs = [(left[item], right[item]) for item in sorted(left)]

    scopes = {
        "all": pairs,
        "default_all_h_minus_1": [pair for pair in pairs if pair[0]["hit_factor"] == -1],
        "default_same_other_h_minus_1_k_plus_1": [
            pair for pair in pairs
            if pair[0]["hit_factor"] == -1 and pair[0]["conflict_factor"] == 1
        ],
        "default_conflicting_other_h_minus_1_k_minus_1": [
            pair for pair in pairs
            if pair[0]["hit_factor"] == -1 and pair[0]["conflict_factor"] == -1
        ],
        "hit_same_h_plus_1_k_plus_1": [
            pair for pair in pairs
            if pair[0]["hit_factor"] == 1 and pair[0]["conflict_factor"] == 1
        ],
        "hit_conflicting_h_plus_1_k_minus_1": [
            pair for pair in pairs
            if pair[0]["hit_factor"] == 1 and pair[0]["conflict_factor"] == -1
        ],
    }
    summaries = {name: summarize(value) for name, value in scopes.items()}

    factor_cells = []
    for partition in ("discovery", "confirmation", "lockbox"):
        for hit in (1, -1):
            for conflict in (1, -1):
                subset = [
                    pair for pair in pairs
                    if pair[0]["partition"] == partition
                    and pair[0]["hit_factor"] == hit
                    and pair[0]["conflict_factor"] == conflict
                ]
                factor_cells.append({
                    "partition": partition,
                    "hit_factor": hit,
                    "conflict_factor": conflict,
                    **summarize(subset),
                })

    default_polarity_cells = []
    for default in (1, -1):
        for conflict in (1, -1):
            for surface in (1, -1):
                for output_format in (1, -1):
                    subset = [
                        pair for pair in pairs
                        if pair[0]["hit_factor"] == -1
                        and pair[0]["default_factor"] == default
                        and pair[0]["conflict_factor"] == conflict
                        and pair[0]["surface_factor"] == surface
                        and pair[0]["output_format"] == output_format
                    ]
                    default_polarity_cells.append({
                        "default_factor": default,
                        "conflict_factor": conflict,
                        "surface_factor": surface,
                        "output_format": output_format,
                        **summarize(subset),
                    })

    by_unit: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for pair in scopes["default_all_h_minus_1"]:
        by_unit[unit_number(pair[0]["unit_id"])].append(pair)
    unit_rows = [
        {"unit_index": index, "partition": values[0][0]["partition"], **summarize(values)}
        for index, values in sorted(by_unit.items())
    ]

    default_same = summaries["default_same_other_h_minus_1_k_plus_1"]
    default_conflict = summaries["default_conflicting_other_h_minus_1_k_minus_1"]
    report = {
        "phase": 1646,
        "campaign": "C119-R",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_behavior_pairing_diagnosed",
        "inputs": {
            "c118_index_sha256": core.sha(C118 / "raw/qwen3_behavior_index.jsonl"),
            "c119_index_sha256": core.sha(C119 / "raw/qwen3_behavior_index.jsonl"),
            "c118_capture_sha256": core.sha(C118 / "analysis/capture_summary.json"),
            "c119_capture_sha256": core.sha(C119 / "analysis/capture_summary.json"),
        },
        "pairing": {
            "exact_factorial_keys": True,
            "pairs": len(pairs),
            "key_fields": [
                "unit_index", "default_factor", "hit_factor", "conflict_factor",
                "surface_factor", "output_format",
            ],
        },
        "scopes": summaries,
        "partition_hit_conflict_cells": factor_cells,
        "default_polarity_surface_format_cells": default_polarity_cells,
        "default_unit_rows": unit_rows,
        "strict_adjudication": {
            "default_error_localization": (
                "The conflict split does not support the proposed other-item-following diagnosis: the "
                "same-value other-item cell is worse than the conflicting other-item cell. The finer "
                "factor table localizes the dominant asymmetry to affirmative defaults (retains) versus "
                "negative defaults (lacks), with additional surface/output modulation. This is a behavior "
                "pattern only; it does not identify whether pragmatics, answer bias, or another process is causal."
            ),
            "paired_interface_claim": (
                "C119 has an aggregate negative paired change. The transition counts describe exactly "
                "which frozen cases changed; they do not establish a general harmful effect of explicit "
                "applicability wording beyond this reused material and prompt pair."
            ),
            "mechanism_boundary": (
                "No HiddenState archive is opened and no attention, MLP, weight, default-override field, "
                "or semantic module is inferred."
            ),
        },
        "authorization": (
            "execute_C120_matched_output_controlled_comparison_family_observation_campaign; "
            "do_not_tune_C118_C119_default_interface_in_this_major_stage"
        ),
    }
    core.save(OUT, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
