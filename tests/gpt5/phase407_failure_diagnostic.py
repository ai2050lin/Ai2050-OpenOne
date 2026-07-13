#!/usr/bin/env python3
"""Build auditable Phase407 failure, event, and numerical ledgers."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase407_event_horizon_protocol import MODELS, OUT  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def histogram(values: Iterable[int | None]) -> dict[str, int]:
    counts = Counter("right_censored_or_absent" if value is None else str(value) for value in values)
    return dict(
        sorted(
            counts.items(),
            key=lambda item: (
                item[0] == "right_censored_or_absent",
                int(item[0]) if item[0] != "right_censored_or_absent" else 0,
            ),
        )
    )


def failure_class(row: dict[str, Any]) -> str:
    if not row["all_generated_step_logits_valid"]:
        return "nonfinite_generation_path"
    if row["semantic_parse_ambiguous"]:
        return "ambiguous_or_revised_semantic_response"
    if row["normalized_semantic_state_private"] is None:
        return "no_registered_complete_semantic_state"
    if row["normalized_semantic_state_private"] != row["target_semantic_state_private"]:
        return "registered_wrong_semantic_state"
    if row["semantic_reversal"]:
        return "semantic_reversal"
    return "registered_semantic_success"


def first_nonfinite_step(row: dict[str, Any]) -> int | None:
    return next(
        (
            step["step"]
            for step in row["step_ledger_private"]
            if not step["logits_valid"]
        ),
        None,
    )


def axis_row(key: tuple[str, str, str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    model, family, interface, history = key
    return {
        "schema_version": "81.4.0",
        "phase_id": "Phase407-FailureDiagnostic",
        "model": model,
        "family_id": family,
        "interface": interface,
        "history_mode": history,
        "case_count": len(rows),
        "semantic_correct_count": sum(row["semantic_correct"] for row in rows),
        "complete_response_count": sum(row["complete_response"] for row in rows),
        "semantic_parse_count": sum(
            row["normalized_semantic_state_private"] is not None for row in rows
        ),
        "semantic_reversal_count": sum(row["semantic_reversal"] for row in rows),
        "eos_observed_count": sum(row["eos_observed"] for row in rows),
        "H48_right_edge_count": sum(row["H48_right_edge_reached"] for row in rows),
        "nonfinite_generation_path_count": sum(
            not row["all_generated_step_logits_valid"] for row in rows
        ),
        "canonical_target_preferred_count": sum(
            row["canonical_target_preferred_to_foil"] for row in rows
        ),
        "target_preferred_but_natural_semantic_wrong_count": sum(
            row["canonical_target_preferred_to_foil"] and not row["semantic_correct"]
            for row in rows
        ),
        "natural_semantic_correct_but_target_not_preferred_count": sum(
            row["semantic_correct"] and not row["canonical_target_preferred_to_foil"]
            for row in rows
        ),
        "failure_class_counts": dict(
            sorted(Counter(failure_class(row) for row in rows).items())
        ),
        "tau_semantic_histogram": histogram(
            row["tau_semantic_private"] for row in rows
        ),
        "tau_boundary_histogram": histogram(
            row["tau_boundary_private"] for row in rows
        ),
        "tau_stop_histogram": histogram(row["tau_stop_private"] for row in rows),
    }


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    for model in MODELS:
        all_rows.extend(
            read_jsonl(
                OUT
                / "analysis/discovery/private"
                / model
                / "semantic_rows.jsonl"
            )
        )

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[
            (
                row["model"],
                row["family_id"],
                row["interface_private"],
                row["history_mode_private"],
            )
        ].append(row)
    axes = [axis_row(key, selected) for key, selected in sorted(grouped.items())]

    group_rows = read_jsonl(
        OUT / "analysis/phase407_discovery_group_details.jsonl"
    )
    gate_pattern_counts = Counter()
    for row in group_rows:
        pattern = ",".join(
            f"{gate}={int(row[f'{gate}_group_pass'])}"
            for gate in ("surface", "interface", "history", "sequence", "completion")
        )
        gate_pattern_counts[pattern] += 1
    fully_semantic_gated_groups = [
        row
        for row in group_rows
        if all(
            row[f"{gate}_group_pass"]
            for gate in ("surface", "interface", "history", "sequence")
        )
    ]

    nonfinite = [
        row for row in all_rows if not row["all_generated_step_logits_valid"]
    ]
    nonfinite_breakdown = Counter(
        (
            row["model"],
            row["family_id"],
            row["interface_private"],
            row["history_mode_private"],
        )
        for row in nonfinite
    )
    nonfinite_rows = [
        {
            "model": key[0],
            "family_id": key[1],
            "interface": key[2],
            "history_mode": key[3],
            "case_count": count,
        }
        for key, count in sorted(nonfinite_breakdown.items())
    ]

    grammar = [
        row for row in all_rows if row["family_id"] == "grammar_constraint"
    ]
    perfect_or_progressive = re.compile(
        r"\b(?:has|have|had)\s+been\b|\b(?:is|are|was|were)\s+being\b",
        re.I,
    )
    registered_be = re.compile(r"\b(?:is|are|was|were)\b", re.I)
    grammar_unparsed = [
        row
        for row in grammar
        if row["normalized_semantic_state_private"] is None
    ]
    grammar_target_literal_anywhere = 0
    grammar_target_literal_only = 0
    for row in grammar_unparsed:
        target = row["target_semantic_state_private"]
        aliases = row["semantic_aliases_by_state_private"]
        text = row["generated_text_clean_private"]
        target_present = any(
            re.search(rf"\b{re.escape(alias)}\b", text, re.I)
            for alias in aliases[target]
        )
        other_present = any(
            re.search(rf"\b{re.escape(alias)}\b", text, re.I)
            for state, values in aliases.items()
            if state != target
            for alias in values
        )
        grammar_target_literal_anywhere += int(target_present)
        grammar_target_literal_only += int(target_present and not other_present)

    private_examples = []
    example_buckets: Counter[tuple[str, str]] = Counter()
    for row in all_rows:
        category = failure_class(row)
        if category == "registered_semantic_success":
            continue
        key = (row["model"], row["family_id"])
        if example_buckets[key] >= 12:
            continue
        example_buckets[key] += 1
        private_examples.append(
            {
                "model": row["model"],
                "family_id": row["family_id"],
                "interface_private": row["interface_private"],
                "history_mode_private": row["history_mode_private"],
                "state_id_private": row["state_id_private"],
                "failure_class": category,
                "generated_text_clean_private": row["generated_text_clean_private"],
                "first_nonfinite_step_private": first_nonfinite_step(row),
            }
        )

    event_summaries = []
    event_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        event_grouped[(row["model"], row["family_id"])].append(row)
    for (model, family), selected in sorted(event_grouped.items()):
        event_summaries.append(
            {
                "model": model,
                "family_id": family,
                "case_count": len(selected),
                "tau_semantic_histogram": histogram(
                    row["tau_semantic_private"] for row in selected
                ),
                "tau_boundary_minus_semantic_histogram": histogram(
                    (
                        row["tau_boundary_private"] - row["tau_semantic_private"]
                        if row["tau_boundary_private"] is not None
                        and row["tau_semantic_private"] is not None
                        else None
                    )
                    for row in selected
                ),
                "tau_stop_minus_boundary_histogram": histogram(
                    (
                        row["tau_stop_private"] - row["tau_boundary_private"]
                        if row["tau_stop_private"] is not None
                        and row["tau_boundary_private"] is not None
                        else None
                    )
                    for row in selected
                ),
            }
        )

    discovery = read_json(OUT / "phase407_discovery_analysis.json")
    payload = {
        "schema_version": "81.4.0",
        "phase_id": "Phase407-FailureDiagnostic",
        "created_at": now(),
        "case_count": len(all_rows),
        "formal_group_count": len(group_rows),
        "semantic_correct_count": sum(row["semantic_correct"] for row in all_rows),
        "complete_response_count": sum(row["complete_response"] for row in all_rows),
        "semantic_reversal_count": sum(
            row["semantic_reversal"] for row in all_rows
        ),
        "failure_class_counts": dict(
            sorted(Counter(failure_class(row) for row in all_rows).items())
        ),
        "fully_semantic_gated_group_count": len(fully_semantic_gated_groups),
        "fully_semantic_gated_groups": [
            {
                "model": row["model"],
                "family_id": row["family_id"],
                "anonymous_parallel_group_id": row[
                    "anonymous_parallel_group_id"
                ],
                "completion_group_pass": row["completion_group_pass"],
            }
            for row in fully_semantic_gated_groups
        ],
        "gate_pattern_counts": dict(sorted(gate_pattern_counts.items())),
        "canonical_target_preferred_count": sum(
            row["canonical_target_preferred_to_foil"] for row in all_rows
        ),
        "target_preferred_but_natural_semantic_wrong_count": sum(
            row["canonical_target_preferred_to_foil"] and not row["semantic_correct"]
            for row in all_rows
        ),
        "natural_semantic_correct_but_target_not_preferred_count": sum(
            row["semantic_correct"] and not row["canonical_target_preferred_to_foil"]
            for row in all_rows
        ),
        "nonfinite_generation_path": {
            "case_count": len(nonfinite),
            "all_nonfinite_cases_reached_H48": bool(nonfinite)
            and all(row["H48_right_edge_reached"] for row in nonfinite),
            "exclamation_run_case_count": sum(
                "!!!!!!!!" in row["generated_text_clean_private"]
                for row in nonfinite
            ),
            "first_nonfinite_step_histogram": histogram(
                first_nonfinite_step(row) for row in nonfinite
            ),
            "breakdown": nonfinite_rows,
        },
        "grammar_contract_diagnostic": {
            "case_count": len(grammar),
            "unparsed_case_count": len(grammar_unparsed),
            "unparsed_with_perfect_or_progressive_be_count": sum(
                bool(perfect_or_progressive.search(row["generated_text_clean_private"]))
                for row in grammar_unparsed
            ),
            "unparsed_with_any_registered_be_literal_count": sum(
                bool(registered_be.search(row["generated_text_clean_private"]))
                for row in grammar_unparsed
            ),
            "unparsed_with_target_literal_anywhere_count": (
                grammar_target_literal_anywhere
            ),
            "unparsed_with_target_literal_and_no_other_be_literal_count": (
                grammar_target_literal_only
            ),
            "interpretation": (
                "Unregistered but grammatical continuations are contract misses, "
                "not direct proof of missing grammar competence."
            ),
        },
        "event_summaries": event_summaries,
        "failure_axes_path": "analysis/phase407_failure_axes.jsonl",
        "private_failure_examples_path": (
            "diagnostics/private/phase407_failure_examples.jsonl"
        ),
        "authorization": discovery["authorization"],
        "claim_boundary": {
            "parser_no_match_is_model_semantic_failure": False,
            "registered_target_preference_is_natural_generation": False,
            "nonfinite_glm_path_is_language_state_failure": False,
            "grammar_contract_miss_is_grammar_incompetence": False,
            "zero_crossmodel_candidate_authorizes_physical_mapping": False,
        },
    }
    write_jsonl(OUT / "analysis/phase407_failure_axes.jsonl", axes)
    write_jsonl(
        OUT / "diagnostics/private/phase407_failure_examples.jsonl",
        private_examples,
    )
    write_json(OUT / "phase407_failure_diagnostic.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
