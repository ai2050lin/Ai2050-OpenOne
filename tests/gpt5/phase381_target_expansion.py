#!/usr/bin/env python3
"""Freeze a behavior-only target-competition expansion before Phase381 tracing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from model_registry import get_model_spec
from phase333_dynamic_case_bank import interface_prompt
from phase381_joint_state_case_bank import (
    CONDITIONS,
    MODELS,
    TEMPLATES,
    digest,
    read_jsonl,
    task,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
EXP = OUT / "target_expansion"
GROUP_INDICES = range(24, 48)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    failed_gate = json.loads(
        (OUT / "phase381_behavior_analysis_summary.json").read_text(encoding="utf-8")
    )
    target_gate = next(
        row for row in failed_gate["gates"] if row["mechanism_id"] == "target_vs_wrong"
    )
    if target_gate["passed"] or target_gate["qualified_parallel_group_count"] != 6:
        raise RuntimeError("Expansion is authorized only by the frozen 6/8 target gate")
    prior_hashes = {
        digest(row["prompt"])
        for row in read_jsonl(OUT / "private/phase381_execution_cases.jsonl")
    }
    execution_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    prompt_hashes: set[str] = set()
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
        for group_index in GROUP_INDICES:
            semantic_group = f"phase381x_readout_competition_target_vs_wrong_{group_index:02d}"
            parallel_group = "parallel381x_" + digest(semantic_group)[:18]
            model_group = "group381x_" + digest(f"{model}:{semantic_group}")[:18]
            items = {
                "A": task("target_vs_wrong", group_index, "x", True),
                "B": task("target_vs_wrong", group_index, "x", False),
                "C": task("target_vs_wrong", group_index, "y", True),
                "D": task("target_vs_wrong", group_index, "y", False),
            }
            for condition in CONDITIONS:
                letter = condition[0]
                item = items[letter]
                raw_prompt = TEMPLATES[group_index % len(TEMPLATES)].format(**item)
                prompt, add_special, answer_phase = interface_prompt(
                    tokenizer, model, raw_prompt, "answer_aligned_chat"
                )
                prompt_hash = digest(prompt)
                if prompt_hash in prompt_hashes or prompt_hash in prior_hashes:
                    raise RuntimeError("Phase381 expansion prompt collision")
                prompt_hashes.add(prompt_hash)
                case_id = "p381x_" + digest(f"{model}:{semantic_group}:{condition}")[:22]
                common = {
                    "schema_version": "54.2.1",
                    "phase_id": "Phase381-TargetExpansion",
                    "created_at": now(),
                    "blind_case_id": case_id,
                    "anonymous_model_id": "am381_" + digest(model)[:11],
                    "anonymous_parallel_group_id": parallel_group,
                    "anonymous_group_id": model_group,
                    "anonymous_condition_slot": "slot381x_"
                    + digest(f"{model_group}:{condition}")[:9],
                    "phase381_split": "behavior_only_target_expansion",
                    "prompt": prompt,
                    "raw_prompt": raw_prompt,
                    "source_fragment": item["context"],
                    "query_fragment": item["question"],
                    "prompt_token_count": len(
                        tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    ),
                    "tokenization_add_special_tokens": add_special,
                    "interface": "answer_aligned_chat",
                    "answer_phase": answer_phase,
                }
                execution_rows.append(
                    {
                        **common,
                        "private_execution_model": model,
                        "family_id": "readout_competition",
                        "mechanism_id": "target_vs_wrong",
                        "semantic_group_id": semantic_group,
                        "contrast_condition": condition,
                        "operation_demanded": letter in {"A", "C"},
                        "target": item["target"],
                        "target_aliases": item["target_aliases"],
                        "distractors": item["distractors"],
                        "language": "en",
                    }
                )
                blind_rows.append(
                    {
                        **common,
                        "semantic_label_used_for_validation": False,
                        "target_or_distractor_exported": False,
                    }
                )
    if len(execution_rows) != 288 or len(prompt_hashes) != 288:
        raise RuntimeError("Invalid Phase381 target expansion denominator")
    write_jsonl(EXP / "private/phase381x_execution_cases.jsonl", execution_rows)
    write_jsonl(EXP / "phase381x_blind_case_registry.jsonl", blind_rows)
    summary = {
        "schema_version": "54.2.1",
        "phase_id": "Phase381-TargetExpansion",
        "created_at": now(),
        "authorization_reason": "original_target_vs_wrong_common_groups_6_below_frozen_minimum_8",
        "case_count": len(execution_rows),
        "parallel_group_count": 24,
        "model_group_count": 72,
        "prior_prompt_overlap_count": 0,
        "failed_original_groups_replaced": False,
        "threshold_lowered": False,
        "internal_trace_started": False,
    }
    write_json(EXP / "phase381x_protocol.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
