#!/usr/bin/env python3
"""Finalize Phase380 behavior groups with the preregistered syntax expansion."""

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
EXP = OUT / "number_agreement_expansion"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = ("relation_binding", "entity_recency", "number_agreement", "target_vs_wrong")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def qualified_groups(rows: list[dict[str, Any]]) -> set[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["anonymous_parallel_group_id"]].append(row)
    return {
        group
        for group, values in groups.items()
        if len(values) == 12
        and all(row["strict_behavior_correct"] for row in values)
        and {row["model"] for row in values} == set(MODELS)
    }


def main() -> None:
    initial_cases = read_jsonl(OUT / "private/phase380_execution_cases.jsonl")
    expansion_cases = read_jsonl(EXP / "private/phase380x_execution_cases.jsonl")
    execution = {row["blind_case_id"]: row for row in initial_cases + expansion_cases}
    initial_behavior = []
    expansion_behavior = []
    for model in MODELS:
        initial_behavior.extend(
            read_jsonl(OUT / "behavior/private/models" / model / "phase380_behavior_rows.jsonl")
        )
        expansion_behavior.extend(
            read_jsonl(EXP / "behavior/private/models" / model / "rows.jsonl")
        )
    initial_behavior = [row for row in initial_behavior if row["mechanism_id"] != "number_agreement"]
    selected_group_ids = qualified_groups(initial_behavior) | qualified_groups(expansion_behavior)
    all_behavior = initial_behavior + expansion_behavior
    selected_behavior = [
        row for row in all_behavior if row["anonymous_parallel_group_id"] in selected_group_ids
    ]
    tokenizers = {}
    selected_cases = []
    try:
        for model in MODELS:
            spec = get_model_spec(model)
            tokenizers[model] = AutoTokenizer.from_pretrained(
                str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
                local_files_only=True, use_fast=False,
            )
        for row in selected_behavior:
            base = execution[row["blind_case_id"]]
            step = first_target_step(
                tokenizers[row["model"]], row["generated_token_ids"], row["target_aliases"]
            )
            if step is None:
                raise RuntimeError(f"Missing decision step {row['blind_case_id']}")
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
    group_mechanism = {}
    for row in selected_behavior:
        group_mechanism[row["anonymous_parallel_group_id"]] = row["mechanism_id"]
    counts = Counter(group_mechanism.values())
    gates = [
        {
            "mechanism_id": mechanism,
            "qualified_parallel_group_count": counts[mechanism],
            "minimum_required": 8,
            "passed": counts[mechanism] >= 8,
        }
        for mechanism in MECHANISMS
    ]
    passed = all(row["passed"] for row in gates)
    write_jsonl(OUT / "private/phase380_qualified_trace_cases.jsonl", selected_cases)
    summary = {
        "schema_version": "53.6.0",
        "phase_id": "Phase380-BehaviorAnalysisFinal",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "initial_behavior_case_count": 1152,
            "expansion_behavior_case_count": len(expansion_behavior),
            "qualified_parallel_group_count": len(selected_group_ids),
            "qualified_trace_case_count": len(selected_cases),
        },
        "gates": gates,
        "all_mechanism_gates_passed": passed,
        "original_number_agreement_groups_retired": True,
        "failed_groups_replaced": False,
        "internal_trace_started_before_final_behavior_gate": False,
        "authorization": {
            "run_exact_trace": passed,
            "run_causal_scan": False,
            "open_prior_physical_holdout": False,
        },
    }
    write_json(OUT / "phase380_behavior_analysis_final_summary.json", summary)
    # The exact tracer reads this stable path.
    write_json(OUT / "phase380_behavior_analysis_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
