#!/usr/bin/env python3
"""Freeze common three-model Phase380 behavior-qualified validation groups."""

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


OUT = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    "relation_binding",
    "entity_recency",
    "number_agreement",
    "target_vs_wrong",
)


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
        for row in read_jsonl(OUT / "private/phase380_execution_cases.jsonl")
    }
    behavior = []
    for model in MODELS:
        behavior.extend(
            read_jsonl(
                OUT
                / "behavior/private/models"
                / model
                / "phase380_behavior_rows.jsonl"
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
    selected_behavior = [
        row for row in behavior if row["anonymous_parallel_group_id"] in qualified
    ]
    tokenizers = {}
    selected_cases = []
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
                raise RuntimeError(f"Missing Phase380 decision step: {row['blind_case_id']}")
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
    group_meta = {
        group_id: (
            rows[0]["mechanism_id"],
            rows[0]["semantic_group_id"],
        )
        for group_id, rows in groups.items()
    }
    mechanism_counts = Counter(group_meta[group_id][0] for group_id in qualified)
    gates = [
        {
            "mechanism_id": mechanism,
            "qualified_parallel_group_count": mechanism_counts[mechanism],
            "minimum_required": 8,
            "passed": mechanism_counts[mechanism] >= 8,
        }
        for mechanism in MECHANISMS
    ]
    all_passed = all(row["passed"] for row in gates)
    write_jsonl(OUT / "private/phase380_qualified_trace_cases.jsonl", selected_cases)
    write_jsonl(
        OUT / "phase380_qualified_blind_groups.jsonl",
        [
            {
                "anonymous_parallel_group_id": group_id,
                "all_three_models_all_four_conditions_correct": True,
            }
            for group_id in sorted(qualified)
        ],
    )
    summary = {
        "schema_version": "53.2.0",
        "phase_id": "Phase380-BehaviorAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "behavior_case_count": len(behavior),
            "parallel_group_count": len(groups),
            "qualified_parallel_group_count": len(qualified),
            "qualified_trace_case_count": len(selected_cases),
        },
        "gates": gates,
        "all_mechanism_gates_passed": all_passed,
        "failed_groups_replaced": False,
        "authorization": {
            "run_exact_trace": all_passed,
            "run_causal_scan": False,
            "open_prior_physical_holdout": False,
        },
    }
    write_json(OUT / "phase380_behavior_analysis_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
