#!/usr/bin/env python3
"""Freeze complete cross-model Phase402 groups without split backfilling."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase402_behavior_protocol import (  # noqa: E402
    CANDIDATE_GROUPS_PER_SURFACE,
    CONDITIONS,
)
from phase402_multiparent_protocol import (  # noqa: E402
    MODELS,
    OUT,
    SPLIT_CANDIDATE_COUNTS,
    SPLIT_SELECTED_COUNTS,
    SURFACES,
)


SALT = "phase402-multiparent-freeze-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str, length: int = 64) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def read_json(path: Path) -> Any:
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


def main() -> None:
    cases = read_jsonl(OUT / "protocol/private/phase402_candidate_cases.jsonl")
    expected = (
        len(SURFACES)
        * CANDIDATE_GROUPS_PER_SURFACE
        * len(CONDITIONS)
        * len(MODELS)
    )
    case_by_id = {row["blind_case_id"]: row for row in cases}
    if len(case_by_id) != expected:
        raise RuntimeError(f"Phase402 expected {expected} cases, got {len(case_by_id)}")

    behavior: list[dict[str, Any]] = []
    model_complete: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        complete = read_json(OUT / "behavior" / model / "complete.json")
        if not complete["valid"] or not complete["unpadding_contract_pass"]:
            raise RuntimeError(f"Invalid Phase402 behavior execution for {model}")
        model_complete[model] = complete
        behavior.extend(read_jsonl(OUT / "behavior/private" / model / "rows.jsonl"))
    behavior_by_id = {row["blind_case_id"]: row for row in behavior}
    if len(behavior_by_id) != expected or set(behavior_by_id) != set(case_by_id):
        raise RuntimeError("Phase402 behavior/case identity mismatch")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavior:
        grouped[row["anonymous_parallel_group_id"]].append(row)
    metadata = {
        group: {
            "surface": rows[0]["task_surface_private"],
            "split": rows[0]["candidate_split_private"],
            "priority": rows[0]["selection_priority_private"],
        }
        for group, rows in grouped.items()
    }
    qualified = {
        group
        for group, rows in grouped.items()
        if len(rows) == len(MODELS) * len(CONDITIONS)
        and {row["model"] for row in rows} == set(MODELS)
        and {row["anonymous_condition_slot"] for row in rows} == set(CONDITIONS)
        and all(row["semantic_correct"] and row["semantic_span_resolved"] for row in rows)
    }
    split_counts = Counter(
        (metadata[group]["surface"], metadata[group]["split"])
        for group in qualified
    )
    eligible = [
        surface
        for surface in SURFACES
        if all(
            split_counts[(surface, split)] >= required
            for split, required in SPLIT_SELECTED_COUNTS.items()
        )
    ]

    selected: dict[str, dict[str, list[str]]] = {}
    reserve: dict[str, dict[str, list[str]]] = {}
    assignment: dict[str, tuple[str, str]] = {}
    for surface in eligible:
        selected[surface] = {}
        reserve[surface] = {}
        for split, count in SPLIT_SELECTED_COUNTS.items():
            ordered = sorted(
                (
                    group
                    for group in qualified
                    if metadata[group]["surface"] == surface
                    and metadata[group]["split"] == split
                ),
                key=lambda group: metadata[group]["priority"],
            )
            selected[surface][split] = ordered[:count]
            reserve[surface][split] = ordered[count:]
            for group in selected[surface][split]:
                assignment[group] = (
                    split,
                    "p402pg_" + digest(f"{SALT}:{group}", 24),
                )

    frozen: list[dict[str, Any]] = []
    for case in cases:
        group = case["anonymous_parallel_group_id"]
        if group not in assignment:
            continue
        result = behavior_by_id[case["blind_case_id"]]
        split, public_group = assignment[group]
        frozen.append(
            {
                **case,
                "schema_version": "76.3.0",
                "phase_id": "Phase402-FrozenDenominator",
                "phase402_split": split,
                "phase402_public_parallel_group_id": public_group,
                "generated_text_before_stop_private": result[
                    "generated_text_before_stop"
                ],
                "effective_generated_token_ids": result[
                    "effective_generated_token_ids"
                ],
                "generated_token_ids_before_stop": result[
                    "generated_token_ids_before_stop"
                ],
                "semantic_start_step": int(result["semantic_start_step"]),
                "semantic_completion_step": int(result["semantic_completion_step"]),
                "semantic_answer_token_ids": result["semantic_answer_token_ids"],
                "format_prefix_token_ids": result["format_prefix_token_ids"],
                "format_suffix_token_ids": result["format_suffix_token_ids"],
                "stop_step": result["stop_step"],
                "stop_kind": result["stop_kind"],
                "semantic_correct": True,
                "semantic_span_resolved": True,
                "execution_batch_size": 1,
            }
        )
    expected_frozen = (
        len(eligible)
        * sum(SPLIT_SELECTED_COUNTS.values())
        * len(CONDITIONS)
        * len(MODELS)
    )
    if len(frozen) != expected_frozen:
        raise RuntimeError(
            f"Phase402 frozen count {len(frozen)} != {expected_frozen}"
        )

    private = OUT / "protocol/private"
    write_jsonl(private / "phase402_frozen_execution_cases.jsonl", frozen)
    for split in SPLIT_SELECTED_COUNTS:
        write_jsonl(
            private / f"phase402_{split}_cases.jsonl",
            [row for row in frozen if row["phase402_split"] == split],
        )
    instrument_ids = {
        selected[surface]["discovery"][0] for surface in eligible
    }
    write_jsonl(
        private / "phase402_instrument_cases.jsonl",
        [
            row
            for row in frozen
            if row["anonymous_parallel_group_id"] in instrument_ids
        ],
    )

    failed = [row for row in behavior if not row["semantic_correct"]]
    payload = {
        "schema_version": "76.3.0",
        "phase_id": "Phase402-BehaviorFreeze",
        "created_at": now(),
        "denominator": {
            "candidate_case_count": len(behavior),
            "candidate_parallel_group_count": len(grouped),
            "qualified_parallel_group_count": len(qualified),
            "eligible_surface_count": len(eligible),
            "selected_parallel_group_count": len(assignment),
            "selected_case_count": len(frozen),
            "instrument_group_count": len(instrument_ids),
            "candidate_split_group_counts_per_surface": SPLIT_CANDIDATE_COUNTS,
            "selected_split_group_counts_per_surface": SPLIT_SELECTED_COUNTS,
        },
        "model_results": {
            model: {
                "case_count": model_complete[model]["case_count"],
                "semantic_correct_count": model_complete[model][
                    "semantic_correct_count"
                ],
                "semantic_span_resolved_count": model_complete[model][
                    "semantic_span_resolved_count"
                ],
                "exact_format_match_count": model_complete[model][
                    "exact_format_match_count"
                ],
            }
            for model in MODELS
        },
        "failure_ledger": {
            "semantic_failure_case_count": len(failed),
            "failures_by_model": dict(Counter(row["model"] for row in failed)),
            "failures_by_surface": dict(
                Counter(row["task_surface_private"] for row in failed)
            ),
            "failed_rows_retained": True,
        },
        "surface_gates": [
            {
                "task_surface": surface,
                "qualified_group_count": sum(
                    split_counts[(surface, split)]
                    for split in SPLIT_CANDIDATE_COUNTS
                ),
                "qualified_group_counts_by_split": {
                    split: split_counts[(surface, split)]
                    for split in SPLIT_CANDIDATE_COUNTS
                },
                "required_group_counts_by_split": SPLIT_SELECTED_COUNTS,
                "eligible": surface in eligible,
                "selected_group_count": (
                    sum(SPLIT_SELECTED_COUNTS.values())
                    if surface in eligible
                    else 0
                ),
            }
            for surface in SURFACES
        ],
        "eligible_surfaces": eligible,
        "selected_groups_private": selected,
        "reserve_groups_private": reserve,
        "selection_contract": {
            "split_frozen_before_behavior": True,
            "priority_frozen_before_behavior": True,
            "selection_does_not_cross_splits": True,
            "failed_groups_retained": True,
            "reserve_used_as_backfill": False,
        },
        "authorization": {
            "run_trace_and_instrument": bool(eligible),
            "run_discovery_before_instrument_pass": False,
            "run_calibration": False,
            "run_physical_holdout": False,
            "run_neuron_scan": False,
        },
        "claim_boundary": {
            "excluded_surface_has_no_internal_mechanism": False,
            "eligible_surface_is_a_closed_language_family": False,
        },
    }
    write_json(OUT / "phase402_behavior_freeze_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
