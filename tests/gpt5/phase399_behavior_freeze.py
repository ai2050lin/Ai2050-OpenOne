#!/usr/bin/env python3
"""Freeze Phase399 qualified groups and independent internal splits."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase379_global_layout_protocol import first_target_step  # noqa: E402
from phase399_dynamic_binding_protocol import (  # noqa: E402
    CANDIDATE_GROUPS_PER_SURFACE,
    CONDITIONS,
    MINIMUM_QUALIFIED_GROUPS,
    MODELS,
    SPLIT_COUNTS,
    SURFACES,
)


OUT = ROOT / "tests/gpt5/result/phase399_dynamic_binding"
SALT = "phase399-dynamic-binding-freeze-v1"


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
    cases = read_jsonl(OUT / "protocol/private/phase399_candidate_cases.jsonl")
    case_by_id = {row["blind_case_id"]: row for row in cases}
    expected_total = (
        len(SURFACES)
        * CANDIDATE_GROUPS_PER_SURFACE
        * len(CONDITIONS)
        * len(MODELS)
    )
    if len(case_by_id) != expected_total:
        raise RuntimeError(f"Expected {expected_total} Phase399 cases, got {len(case_by_id)}")
    behavior: list[dict[str, Any]] = []
    for model in MODELS:
        complete = read_json(OUT / "behavior" / model / "complete.json")
        if not complete["valid"] or complete["execution_batch_size"] != 1:
            raise RuntimeError(f"Invalid Phase399 behavior output for {model}")
        behavior.extend(read_jsonl(OUT / "behavior/private" / model / "rows.jsonl"))
    if len(behavior) != expected_total:
        raise RuntimeError(f"Expected {expected_total} behavior rows, got {len(behavior)}")

    tokenizers: dict[str, Any] = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
    enriched: list[dict[str, Any]] = []
    for row in behavior:
        case = case_by_id[row["blind_case_id"]]
        step = first_target_step(
            tokenizers[row["model"]], row["generated_token_ids"], case["target_aliases"]
        )
        has_next = step is not None and step + 1 < len(row["generated_token_ids"])
        enriched.append(
            {
                **row,
                "target_completion_step_private": step,
                "post_target_next_token_available": has_next,
                "dynamic_binding_behavior_qualified": bool(
                    row["strict_behavior_correct"] and has_next
                ),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[row["anonymous_parallel_group_id"]].append(row)
    surface_by_group = {
        group: rows[0]["task_surface_private"] for group, rows in grouped.items()
    }
    qualified = {
        group
        for group, rows in grouped.items()
        if len(rows) == len(MODELS) * len(CONDITIONS)
        and {row["model"] for row in rows} == set(MODELS)
        and {row["anonymous_condition_slot"] for row in rows} == set(CONDITIONS)
        and all(row["dynamic_binding_behavior_qualified"] for row in rows)
    }
    counts = Counter(surface_by_group[group] for group in qualified)
    eligible = [
        surface
        for surface in SURFACES
        if counts[surface] >= MINIMUM_QUALIFIED_GROUPS
    ]
    selected: dict[str, dict[str, list[str]]] = {}
    reserve: dict[str, list[str]] = {}
    assignment: dict[str, tuple[str, str]] = {}
    for surface in eligible:
        ordered = sorted(
            (group for group in qualified if surface_by_group[group] == surface),
            key=lambda group: digest(f"{SALT}:{surface}:{group}"),
        )
        selected[surface], cursor = {}, 0
        for split, count in SPLIT_COUNTS.items():
            split_groups = ordered[cursor : cursor + count]
            selected[surface][split] = split_groups
            for group in split_groups:
                assignment[group] = (
                    split,
                    "p399g_" + digest(f"{SALT}:{group}", 24),
                )
            cursor += count
        reserve[surface] = ordered[cursor:]

    behavior_by_case = {row["blind_case_id"]: row for row in enriched}
    frozen: list[dict[str, Any]] = []
    for case in cases:
        group = case["anonymous_parallel_group_id"]
        if group not in assignment:
            continue
        result = behavior_by_case[case["blind_case_id"]]
        split, public_group = assignment[group]
        frozen.append(
            {
                **case,
                "schema_version": "73.2.0",
                "phase_id": "Phase399-FrozenDenominator",
                "phase399_split": split,
                "phase399_public_parallel_group_id": public_group,
                "generated_text_private": result["generated_text"],
                "generated_token_ids": result["generated_token_ids"],
                "target_completion_step": int(result["target_completion_step_private"]),
                "strict_behavior_correct": True,
                "post_target_next_token_available": True,
                "semantic_labels_available_to_collection": True,
            }
        )
    expected_frozen = len(eligible) * sum(SPLIT_COUNTS.values()) * len(CONDITIONS) * len(MODELS)
    if len(frozen) != expected_frozen:
        raise RuntimeError(
            f"Phase399 selected case mismatch {len(frozen)} != {expected_frozen}"
        )
    private = OUT / "protocol/private"
    write_jsonl(private / "phase399_frozen_execution_cases.jsonl", frozen)
    for split in SPLIT_COUNTS:
        write_jsonl(
            private / f"phase399_{split}_cases.jsonl",
            [row for row in frozen if row["phase399_split"] == split],
        )
    instrument_ids = {
        selected[surface]["discovery"][0] for surface in eligible
    }
    instrument = [
        row
        for row in frozen
        if row["anonymous_parallel_group_id"] in instrument_ids
    ]
    write_jsonl(private / "phase399_instrument_cases.jsonl", instrument)
    summary = {
        "schema_version": "73.2.0",
        "phase_id": "Phase399-BehaviorFreeze",
        "created_at": now(),
        "denominator": {
            "candidate_case_count": len(behavior),
            "candidate_parallel_group_count": len(grouped),
            "qualified_parallel_group_count": len(qualified),
            "eligible_surface_count": len(eligible),
            "selected_parallel_group_count": len(assignment),
            "selected_case_count": len(frozen),
            "instrument_group_count": len(instrument_ids),
            "instrument_case_count": len(instrument),
            "minimum_qualified_groups_per_surface": MINIMUM_QUALIFIED_GROUPS,
            "split_group_counts_per_surface": SPLIT_COUNTS,
        },
        "model_results": {
            model: {
                "strict_correct_count": sum(
                    row["strict_behavior_correct"]
                    for row in enriched
                    if row["model"] == model
                ),
                "case_count": sum(row["model"] == model for row in enriched),
            }
            for model in MODELS
        },
        "surface_gates": [
            {
                "task_surface": surface,
                "candidate_group_count": CANDIDATE_GROUPS_PER_SURFACE,
                "qualified_group_count": counts[surface],
                "eligible": surface in eligible,
                "selected_group_count": (
                    sum(SPLIT_COUNTS.values()) if surface in eligible else 0
                ),
                "reserve_qualified_group_count": len(reserve.get(surface, [])),
                "exclusion_reason": (
                    None
                    if surface in eligible
                    else "fewer_than_20_three_model_sixteen_condition_groups_qualified"
                ),
            }
            for surface in SURFACES
        ],
        "eligible_surfaces": eligible,
        "selected_groups_private": selected,
        "reserve_groups_private": reserve,
        "selection_contract": {
            "minimum_20_groups_required": True,
            "all_failed_groups_retained_in_behavior_ledger": True,
            "reserve_used_as_backfill": False,
            "hash_split_before_internal_trace": True,
        },
        "authorization": {
            "run_instrument_trace": bool(eligible),
            "run_discovery_trace": False,
            "run_calibration": False,
            "open_physical_holdout": False,
            "run_joint_causal_intervention": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "behavior_factorial_is_dynamic_binding_mechanism": False,
            "excluded_surface_has_no_dynamic_process": False,
        },
    }
    write_json(OUT / "phase399_behavior_freeze_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
