#!/usr/bin/env python3
"""Independent recomputation audit for the C119-R paired behavior diagnosis."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
C118 = RESULT / "phase1640_c118_identifiable_default_override_campaign"
C119 = RESULT / "phase1643_c119_identifiable_default_override_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def factor_key(row: dict) -> tuple:
    return (
        int(row["unit_id"].rsplit("-", 1)[1]), row["default_factor"],
        row["hit_factor"], row["conflict_factor"], row["surface_factor"],
        row["output_format"],
    )


def summary(pairs: list[tuple[dict, dict]]) -> tuple[int, int, int, int, int]:
    both = left = right = neither = 0
    for a, b in pairs:
        both += bool(a["correct"] and b["correct"])
        left += bool(a["correct"] and not b["correct"])
        right += bool(not a["correct"] and b["correct"])
        neither += bool(not a["correct"] and not b["correct"])
    return len(pairs), both, left, right, neither


if __name__ == "__main__":
    report = core.load(C119 / "analysis/paired_behavior_diagnostic.json")
    a = {factor_key(row): row for row in core.rows(C118 / "raw/qwen3_behavior_index.jsonl")}
    b = {factor_key(row): row for row in core.rows(C119 / "raw/qwen3_behavior_index.jsonl")}
    pairs = [(a[item], b[item]) for item in sorted(a)]
    default_pairs = [pair for pair in pairs if pair[0]["hit_factor"] == -1]
    conflict_pairs = [
        pair for pair in default_pairs if pair[0]["conflict_factor"] == -1
    ]
    same_pairs = [pair for pair in default_pairs if pair[0]["conflict_factor"] == 1]
    n, both, left, right, neither = summary(default_pairs)
    registered = report["scopes"]["default_all_h_minus_1"]
    checks = {
        "input_hashes": (
            report["inputs"]["c118_index_sha256"] == core.sha(C118 / "raw/qwen3_behavior_index.jsonl")
            and report["inputs"]["c119_index_sha256"] == core.sha(C119 / "raw/qwen3_behavior_index.jsonl")
        ),
        "exact_pairing": len(a) == len(b) == 768 and set(a) == set(b),
        "default_count": n == 384,
        "transitions": registered["paired_transitions"] == {
            "both_correct": both,
            "c118_only_correct": left,
            "c119_only_correct": right,
            "both_wrong": neither,
        },
        "accuracies": (
            abs(registered["c118_accuracy"] - sum(x[0]["correct"] for x in default_pairs) / n) < 1e-12
            and abs(registered["c119_accuracy"] - sum(x[1]["correct"] for x in default_pairs) / n) < 1e-12
        ),
        "split_counts": len(conflict_pairs) == len(same_pairs) == 192,
        "conflict_reversal": (
            report["scopes"]["default_conflicting_other_h_minus_1_k_minus_1"]["c118_accuracy"]
            > report["scopes"]["default_same_other_h_minus_1_k_plus_1"]["c118_accuracy"]
        ),
        "polarity_cells": len(report["default_polarity_surface_format_cells"]) == 16
            and all(row["n"] == 24 for row in report["default_polarity_surface_format_cells"]),
        "no_hidden_state_claim": "No HiddenState archive is opened" in report["strict_adjudication"]["mechanism_boundary"],
        "authorization": report["authorization"].startswith("execute_C120"),
    }
    audit = {
        "phase": 1646,
        "campaign": "C119-R",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": report["authorization"],
    }
    if not audit["all_checks_passed"]:
        raise RuntimeError(audit)
    core.save(C119 / "audit/paired_behavior_diagnostic_audit.json", audit)
    print(json.dumps(audit, indent=2))
