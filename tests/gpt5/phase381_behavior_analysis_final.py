#!/usr/bin/env python3
"""Freeze the final Phase381 trace denominator after behavior-only expansion."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase379_global_layout_protocol import first_target_step  # noqa: E402
from phase381_joint_state_case_bank import read_jsonl, write_json, write_jsonl  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
EXP = OUT / "target_expansion"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = ("relation_binding", "entity_recency", "target_vs_wrong")
MINIMUM = 8


def qualified_groups(rows: list[dict[str, Any]]) -> set[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["anonymous_parallel_group_id"]].append(row)
    return {
        group_id
        for group_id, items in groups.items()
        if len(items) == 12
        and all(item["strict_behavior_correct"] for item in items)
        and {item["model"] for item in items} == set(MODELS)
        and all(
            sum(
                item["model"] == model
                and item["contrast_condition"].startswith(letter)
                for item in items
            )
            == 1
            for model in MODELS
            for letter in "ABCD"
        )
    }


def main() -> None:
    execution_rows = read_jsonl(OUT / "private/phase381_execution_cases.jsonl")
    expansion_execution = read_jsonl(EXP / "private/phase381x_execution_cases.jsonl")
    execution = {
        row["blind_case_id"]: row for row in [*execution_rows, *expansion_execution]
    }
    original_behavior: list[dict[str, Any]] = []
    expansion_behavior: list[dict[str, Any]] = []
    for model in MODELS:
        original_behavior.extend(
            read_jsonl(
                OUT
                / "behavior/private/models"
                / model
                / "phase381_behavior_rows.jsonl"
            )
        )
        expansion_behavior.extend(
            read_jsonl(EXP / "behavior/private/models" / model / "rows.jsonl")
        )
    original_qualified = qualified_groups(original_behavior)
    expansion_qualified = qualified_groups(expansion_behavior)
    behavior = [*original_behavior, *expansion_behavior]
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavior:
        group_rows[row["anonymous_parallel_group_id"]].append(row)
    mechanism_by_group = {
        group_id: rows[0]["mechanism_id"] for group_id, rows in group_rows.items()
    }
    all_qualified = original_qualified | expansion_qualified
    qualified_counts = Counter(mechanism_by_group[group] for group in all_qualified)
    selected_groups = {
        mechanism: sorted(
            group
            for group in all_qualified
            if mechanism_by_group[group] == mechanism
        )[:MINIMUM]
        for mechanism in MECHANISMS
    }
    gates = [
        {
            "mechanism_id": mechanism,
            "qualified_parallel_group_count": qualified_counts[mechanism],
            "selected_parallel_group_count": len(selected_groups[mechanism]),
            "minimum_required": MINIMUM,
            "passed": len(selected_groups[mechanism]) == MINIMUM,
        }
        for mechanism in MECHANISMS
    ]
    if not all(row["passed"] for row in gates):
        raise RuntimeError(f"Final Phase381 behavior gate failed: {gates}")
    selected_group_set = {group for values in selected_groups.values() for group in values}
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
                raise RuntimeError(f"Missing target decision: {row['blind_case_id']}")
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
        raise RuntimeError(f"Expected 288 final trace cases, got {len(selected_cases)}")
    write_jsonl(OUT / "private/phase381_qualified_trace_cases.jsonl", selected_cases)
    write_jsonl(
        OUT / "phase381_selected_blind_groups.jsonl",
        [
            {
                "anonymous_parallel_group_id": group,
                "mechanism_id": mechanism,
                "source_cohort": "target_expansion" if group.startswith("parallel381x_") else "original",
                "all_three_models_all_four_conditions_correct": True,
                "selected_before_internal_trace": True,
            }
            for mechanism, groups in selected_groups.items()
            for group in groups
        ],
    )
    summary = {
        "schema_version": "54.2.3",
        "phase_id": "Phase381-BehaviorAnalysisFinal",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "original_behavior_case_count": len(original_behavior),
            "expansion_behavior_case_count": len(expansion_behavior),
            "original_qualified_parallel_group_count": len(original_qualified),
            "expansion_qualified_parallel_group_count": len(expansion_qualified),
            "selected_parallel_group_count": len(selected_group_set),
            "selected_trace_case_count": len(selected_cases),
        },
        "gates": gates,
        "selected_groups": selected_groups,
        "original_failed_groups_replaced": False,
        "threshold_retuned": False,
        "internal_trace_started_before_final_gate": False,
        "authorization": {
            "run_exact_trace": True,
            "run_joint_scan_before_trace_complete": False,
            "open_single_neuron_scan": False,
        },
    }
    write_json(OUT / "phase381_behavior_analysis_final_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
