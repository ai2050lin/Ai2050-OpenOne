#!/usr/bin/env python3
"""Freeze common three-model Phase381 groups before any internal trace."""

from __future__ import annotations

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


OUT = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = ("relation_binding", "entity_recency", "target_vs_wrong")
SELECTED_GROUPS_PER_MECHANISM = 8


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
    execution = {
        row["blind_case_id"]: row
        for row in read_jsonl(OUT / "private/phase381_execution_cases.jsonl")
    }
    behavior: list[dict[str, Any]] = []
    for model in MODELS:
        behavior.extend(
            read_jsonl(
                OUT
                / "behavior/private/models"
                / model
                / "phase381_behavior_rows.jsonl"
            )
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavior:
        groups[row["anonymous_parallel_group_id"]].append(row)
    qualified = {
        group_id
        for group_id, rows in groups.items()
        if len(rows) == 12
        and all(row["strict_behavior_correct"] for row in rows)
        and {row["model"] for row in rows} == set(MODELS)
        and all(
            sum(
                item["model"] == model
                and item["contrast_condition"].startswith(letter)
                for item in rows
            )
            == 1
            for model in MODELS
            for letter in "ABCD"
        )
    }
    mechanism_by_group = {
        group_id: rows[0]["mechanism_id"] for group_id, rows in groups.items()
    }
    qualified_counts = Counter(mechanism_by_group[group_id] for group_id in qualified)
    selected_groups = {
        mechanism: sorted(
            group_id
            for group_id in qualified
            if mechanism_by_group[group_id] == mechanism
        )[:SELECTED_GROUPS_PER_MECHANISM]
        for mechanism in MECHANISMS
    }
    gates = [
        {
            "mechanism_id": mechanism,
            "qualified_parallel_group_count": qualified_counts[mechanism],
            "selected_parallel_group_count": len(selected_groups[mechanism]),
            "minimum_required": SELECTED_GROUPS_PER_MECHANISM,
            "passed": len(selected_groups[mechanism]) == SELECTED_GROUPS_PER_MECHANISM,
        }
        for mechanism in MECHANISMS
    ]
    if not all(row["passed"] for row in gates):
        write_json(
            OUT / "phase381_behavior_analysis_summary.json",
            {
                "schema_version": "54.2.0",
                "phase_id": "Phase381-BehaviorAnalysis",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "gates": gates,
                "authorization": {"run_exact_trace": False},
            },
        )
        raise RuntimeError(f"Phase381 behavior gates failed: {gates}")
    selected_group_set = {
        group for mechanism_groups in selected_groups.values() for group in mechanism_groups
    }
    selected_behavior = [
        row for row in behavior if row["anonymous_parallel_group_id"] in selected_group_set
    ]
    tokenizers: dict[str, Any] = {}
    selected_cases: list[dict[str, Any]] = []
    try:
        for model in MODELS:
            spec = get_model_spec(model)
            tokenizers[model] = AutoTokenizer.from_pretrained(
                str(spec.local_dir),
                trust_remote_code=spec.trust_remote_code,
                local_files_only=True,
                use_fast=False,
            )
        for row in selected_behavior:
            base = execution[row["blind_case_id"]]
            step = first_target_step(
                tokenizers[row["model"]],
                row["generated_token_ids"],
                row["target_aliases"],
            )
            if step is None:
                raise RuntimeError(f"Missing Phase381 decision step: {row['blind_case_id']}")
            selected_cases.append(
                {
                    **base,
                    "generated_token_ids": row["generated_token_ids"],
                    "target_decision_step": step,
                    "strict_behavior_correct": True,
                    "semantic_labels_available_to_trace": False,
                    "target_specific_competition_available_to_trace": False,
                }
            )
    finally:
        tokenizers.clear()
    if len(selected_cases) != 288:
        raise RuntimeError(f"Expected 288 selected Phase381 cases, got {len(selected_cases)}")
    write_jsonl(OUT / "private/phase381_qualified_trace_cases.jsonl", selected_cases)
    write_jsonl(
        OUT / "phase381_selected_blind_groups.jsonl",
        [
            {
                "anonymous_parallel_group_id": group_id,
                "mechanism_id": mechanism,
                "all_three_models_all_four_conditions_correct": True,
                "selected_before_internal_trace": True,
            }
            for mechanism, mechanism_groups in selected_groups.items()
            for group_id in mechanism_groups
        ],
    )
    summary = {
        "schema_version": "54.2.0",
        "phase_id": "Phase381-BehaviorAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "behavior_case_count": len(behavior),
            "parallel_group_count": len(groups),
            "qualified_parallel_group_count": len(qualified),
            "selected_parallel_group_count": len(selected_group_set),
            "selected_trace_case_count": len(selected_cases),
        },
        "gates": gates,
        "selected_groups": selected_groups,
        "failed_groups_replaced": False,
        "threshold_retuned": False,
        "internal_trace_started_before_selection": False,
        "authorization": {
            "run_exact_trace": True,
            "run_joint_scan_before_trace_complete": False,
            "open_single_neuron_scan": False,
        },
    }
    write_json(OUT / "phase381_behavior_analysis_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
