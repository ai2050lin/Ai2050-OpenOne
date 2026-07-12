#!/usr/bin/env python3
"""Audit whether Phase371 fixed generation offsets align semantic answer decisions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402


PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
COLLECTOR = (
    PHASE371
    / "phase371c_behavior_analysis/private/phase371c_discovery_collector_cases.jsonl"
)
BEHAVIOR = PHASE371 / "phase371c_behavior_qualification/private/models"
OUT = ROOT / "tests/gpt5/result/phase376_decision_aligned_subgraphs"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def first_target_step(tokenizer: Any, token_ids: list[int], aliases: list[str]) -> int | None:
    lowered = [alias.strip().lower() for alias in aliases if alias.strip()]
    for index in range(len(token_ids)):
        text = tokenizer.decode(token_ids[: index + 1]).lower()
        if any(alias in text for alias in lowered):
            return index
    return None


def main() -> None:
    collector_ids = {row["blind_case_id"] for row in read_jsonl(COLLECTOR)}
    private_rows = []
    model_summaries = []
    cross: dict[tuple[str, str], dict[str, int | None]] = defaultdict(dict)
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
        rows = [
            row
            for row in read_jsonl(BEHAVIOR / model / "phase371c_behavior_rows.jsonl")
            if row["blind_case_id"] in collector_ids
        ]
        distribution: Counter[str] = Counter()
        mechanism_distribution: dict[str, Counter[str]] = defaultdict(Counter)
        within = 0
        missing = 0
        for row in rows:
            step = first_target_step(tokenizer, row["generated_token_ids"], row["target_aliases"])
            key = "missing" if step is None else str(step)
            distribution[key] += 1
            mechanism_distribution[row["mechanism_id"]][key] += 1
            within += int(step is not None and step <= 2)
            missing += int(step is None)
            cross[(row["semantic_group_id"], row["contrast_condition"])][model] = step
            private_rows.append(
                {
                    "schema_version": "49.0.0",
                    "phase_id": "Phase376-DecisionAlignmentAudit",
                    "model": model,
                    "blind_case_id": row["blind_case_id"],
                    "semantic_group_id": row["semantic_group_id"],
                    "mechanism_id": row["mechanism_id"],
                    "contrast_condition": row["contrast_condition"],
                    "target": row["target"],
                    "target_decision_step": step,
                    "within_phase371_fixed_window": step is not None and step <= 2,
                    "generated_token_count": len(row["generated_token_ids"]),
                    "generated_token_ids": row["generated_token_ids"],
                }
            )
        model_summaries.append(
            {
                "model": model,
                "case_count": len(rows),
                "decision_step_distribution": dict(
                    sorted(distribution.items(), key=lambda item: (item[0] == "missing", item[0]))
                ),
                "mechanism_distributions": {
                    mechanism: dict(values)
                    for mechanism, values in sorted(mechanism_distribution.items())
                },
                "within_fixed_t0_t2_count": within,
                "within_fixed_t0_t2_rate": within / len(rows),
                "missing_target_step_count": missing,
            }
        )
    comparable = [values for values in cross.values() if set(values) == set(MODELS)]
    exact_same = sum(
        len(set(values.values())) == 1 and None not in values.values() for values in comparable
    )
    all_within = sum(
        all(value is not None and value <= 2 for value in values.values()) for values in comparable
    )
    OUT.mkdir(parents=True, exist_ok=True)
    private_path = OUT / "private/phase376_decision_time_rows.jsonl"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    with private_path.open("w", encoding="utf-8") as handle:
        for row in private_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "49.0.0",
        "phase_id": "Phase376-DecisionAlignmentAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "test_whether_fixed_generation_offsets_are_crossmodel_semantic_events",
        "denominator": {
            "model_count": 3,
            "case_count": len(private_rows),
            "case_count_per_model": 88,
            "crossmodel_semantic_condition_count": len(comparable),
        },
        "models": model_summaries,
        "crossmodel": {
            "exact_same_decision_step_count": exact_same,
            "all_models_within_fixed_t0_t2_count": all_within,
            "semantic_condition_count": len(comparable),
        },
        "results": {
            "fixed_t0_t2_is_crossmodel_semantic_alignment": False,
            "phase374_375_fixed_time_results_are_early_prefix_diagnostics": True,
            "phase374_375_fixed_time_results_are_answer_decision_mechanisms": False,
            "decision_aligned_recollection_required": True,
        },
        "claim_boundary": {
            "invalidates_existing_exact_tensor_measurements": False,
            "invalidates_fixed_offset_semantic_comparison": True,
            "proves_decision_aligned_subgraph_causality": False,
        },
        "next_stage": {
            "events": ["answer_entry", "pre_target_decision", "target_decision"],
            "primary_readout": "direct_activation_swap_and_target_token_margin",
            "predictive_prefilter_required": False,
            "calibration_opened": False,
            "physical_opened": False,
        },
    }
    path = OUT / "phase376_decision_time_alignment_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
