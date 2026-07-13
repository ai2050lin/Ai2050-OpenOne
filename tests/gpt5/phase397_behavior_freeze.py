#!/usr/bin/env python3
"""Freeze Phase397 eligible surfaces and independent splits without backfill."""

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
from phase397_multitask_protocol import (  # noqa: E402
    CONDITIONS,
    MINIMUM_QUALIFIED_GROUPS,
    MODELS,
    SPLIT_COUNTS,
    SURFACES,
)


OUT = ROOT / "tests/gpt5/result/phase397_multitask_binding"
SALT = "phase397-multitask-binding-freeze-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str, length: int = 64) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    cases = read_jsonl(OUT / "protocol/private/phase397_candidate_cases.jsonl")
    case_by_id = {row["blind_case_id"]: row for row in cases}
    if len(case_by_id) != 4320:
        raise RuntimeError(f"Expected 4320 Phase397 cases, got {len(case_by_id)}")
    behavior: list[dict[str, Any]] = []
    for model in MODELS:
        complete = read_json(OUT / "behavior" / model / "complete.json")
        if not complete["valid"] or complete["execution_batch_size"] != 1:
            raise RuntimeError(f"Invalid Phase397 behavior output for {model}")
        behavior.extend(read_jsonl(OUT / "behavior/private" / model / "rows.jsonl"))
    if len(behavior) != 4320:
        raise RuntimeError(f"Expected 4320 Phase397 behavior rows, got {len(behavior)}")

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
        step = first_target_step(tokenizers[row["model"]], row["generated_token_ids"], case["target_aliases"])
        has_next = step is not None and step + 1 < len(row["generated_token_ids"])
        enriched.append(
            {
                **row,
                "target_decision_step_private": step,
                "post_decision_next_token_available": has_next,
                "multitask_behavior_qualified": bool(row["strict_behavior_correct"] and has_next),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[row["anonymous_parallel_group_id"]].append(row)
    surface_by_group = {group_id: rows[0]["task_surface_private"] for group_id, rows in grouped.items()}
    qualified_groups = {
        group_id
        for group_id, rows in grouped.items()
        if len(rows) == len(MODELS) * len(CONDITIONS)
        and {row["model"] for row in rows} == set(MODELS)
        and {row["contrast_condition_private"] for row in rows} == set(CONDITIONS)
        and all(row["multitask_behavior_qualified"] for row in rows)
    }
    qualified_counts = Counter(surface_by_group[group_id] for group_id in qualified_groups)
    eligible_surfaces = [surface for surface in SURFACES if qualified_counts[surface] >= MINIMUM_QUALIFIED_GROUPS]

    selected: dict[str, dict[str, list[str]]] = {}
    assignment: dict[str, tuple[str, str]] = {}
    reserve: dict[str, list[str]] = {}
    for surface in eligible_surfaces:
        ordered = sorted(
            (group_id for group_id in qualified_groups if surface_by_group[group_id] == surface),
            key=lambda group_id: digest(f"{SALT}:{surface}:{group_id}"),
        )
        selected[surface] = {}
        cursor = 0
        for split, count in SPLIT_COUNTS.items():
            split_groups = ordered[cursor : cursor + count]
            selected[surface][split] = split_groups
            for group_id in split_groups:
                assignment[group_id] = (split, "p397g_" + digest(f"{SALT}:{group_id}", 24))
            cursor += count
        reserve[surface] = ordered[cursor:]

    behavior_by_case = {row["blind_case_id"]: row for row in enriched}
    selected_cases: list[dict[str, Any]] = []
    for case in cases:
        source_group = case["anonymous_parallel_group_id"]
        if source_group not in assignment:
            continue
        result = behavior_by_case[case["blind_case_id"]]
        split, public_group = assignment[source_group]
        selected_cases.append(
            {
                **case,
                "schema_version": "71.2.0",
                "phase_id": "Phase397-FrozenDenominator",
                "phase397_split": split,
                "phase397_public_parallel_group_id": public_group,
                "generated_text_private": result["generated_text"],
                "generated_token_ids": result["generated_token_ids"],
                "target_decision_step": int(result["target_decision_step_private"]),
                "strict_behavior_correct": True,
                "post_decision_next_token_available": True,
                "semantic_labels_available_to_collection": False,
            }
        )
    expected_selected = len(eligible_surfaces) * sum(SPLIT_COUNTS.values()) * len(CONDITIONS) * len(MODELS)
    if len(selected_cases) != expected_selected:
        raise RuntimeError(f"Phase397 selected case mismatch: {len(selected_cases)} != {expected_selected}")

    private = OUT / "protocol/private"
    write_jsonl(private / "phase397_frozen_execution_cases.jsonl", selected_cases)
    for split in SPLIT_COUNTS:
        write_jsonl(private / f"phase397_{split}_cases.jsonl", [row for row in selected_cases if row["phase397_split"] == split])
    instrument_groups = {selected[surface]["discovery"][0] for surface in eligible_surfaces}
    instrument_cases = [
        {
            **row,
            "phase397_collection_stage": "instrument_audit",
            "instrument_reuse_scope": "engineering_conservation_only",
        }
        for row in selected_cases
        if row["anonymous_parallel_group_id"] in instrument_groups
    ]
    write_jsonl(private / "phase397_instrument_audit_cases.jsonl", instrument_cases)
    write_jsonl(
        OUT / "protocol/phase397_frozen_blind_groups.jsonl",
        [
            {
                "schema_version": "71.2.0",
                "phase_id": "Phase397-FrozenDenominator",
                "public_parallel_group_id": public_group,
                "split": split,
                "three_models_ten_conditions_and_next_token_qualified": True,
                "semantic_label_exported": False,
            }
            for _source_group, (split, public_group) in sorted(assignment.items())
        ],
    )
    summary = {
        "schema_version": "71.2.0",
        "phase_id": "Phase397-BehaviorFreeze",
        "created_at": now(),
        "denominator": {
            "candidate_case_count": len(behavior),
            "candidate_parallel_group_count": len(grouped),
            "qualified_parallel_group_count": len(qualified_groups),
            "eligible_surface_count": len(eligible_surfaces),
            "selected_parallel_group_count": len(assignment),
            "selected_case_count": len(selected_cases),
            "instrument_group_count": len(instrument_groups),
            "instrument_case_count": len(instrument_cases),
            "minimum_qualified_groups_per_surface": MINIMUM_QUALIFIED_GROUPS,
            "split_group_counts_per_surface": SPLIT_COUNTS,
        },
        "model_results": {
            model: {
                "strict_correct_count": sum(row["strict_behavior_correct"] for row in enriched if row["model"] == model),
                "case_count": sum(row["model"] == model for row in enriched),
            }
            for model in MODELS
        },
        "surface_gates": [
            {
                "task_surface": surface,
                "candidate_group_count": 24,
                "qualified_group_count": qualified_counts[surface],
                "eligible": surface in eligible_surfaces,
                "selected_group_count": sum(surface in selected and len(groups) for groups in selected.get(surface, {}).values()),
                "reserve_qualified_group_count": len(reserve.get(surface, [])),
                "exclusion_reason": None if surface in eligible_surfaces else "fewer_than_16_three_model_ten_condition_groups_qualified",
            }
            for surface in SURFACES
        ],
        "eligible_surfaces": eligible_surfaces,
        "selected_groups_private": selected,
        "reserve_groups_private": reserve,
        "selection_contract": {
            "minimum_16_groups_required": True,
            "failed_groups_replaced": False,
            "reserve_groups_used_as_backfill": False,
            "excluded_surfaces_backfilled": False,
            "hash_order_split_before_internal_collection": True,
        },
        "authorization": {
            "run_instrument_audit": bool(eligible_surfaces),
            "run_discovery_collection": False,
            "run_calibration": False,
            "open_physical_holdout": False,
            "run_single_neuron_scan": False,
        },
        "claim_boundary": {
            "behavior_qualified_surface_is_binding_mechanism": False,
            "excluded_surface_has_no_binding": False,
            "language_encoding_closed": False,
        },
    }
    write_json(OUT / "phase397_behavior_freeze_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
