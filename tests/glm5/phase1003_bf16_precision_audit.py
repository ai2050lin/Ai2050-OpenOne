#!/usr/bin/env python3
"""BF16 audit of Phase1003 anchor and cache conclusions on Qwen3-4B."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1003_anchor_subset_exhaustive import (
    choose_donors as choose_attribute_donors,
)
from phase1003_crossparadigm_protocol import (
    ANCHOR_ROLES,
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    selected_directional_rows,
    write_json,
    write_jsonl,
)
from phase1003_structural_stress_causal import (
    cache_causal,
    choose_donors as choose_stress_donors,
    natural_confirmation,
    teacher_causal,
)
from phase1003_structural_stress_protocol import STRESS_ROOT


MODEL = "qwen3"
TASKS = ("color", "negation", "pronoun")
AUDIT_ROOT = OUT_ROOT / "precision_audit" / "qwen3_bf16"


def attribute_cases() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows = selected_directional_rows(
        MODEL, "color", "confirmation"
    )
    donors, donor_audit = choose_attribute_donors(
        rows, MODEL, "color", "confirmation"
    )
    cases = []
    prepared_donors = []
    for row, donor in zip(rows, donors):
        target = dict(row["target"])
        target["anchor_roles"] = list(ANCHOR_ROLES)
        target["candidate_labels"] = list(
            target["candidate_token_ids"]
        )
        donor = dict(donor)
        donor["anchor_roles"] = list(ANCHOR_ROLES)
        donor["candidate_labels"] = list(
            donor["candidate_token_ids"]
        )
        cases.append(target)
        prepared_donors.append(donor)
    return cases, prepared_donors, donor_audit


def structural_cases(
    task: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    all_cases = [
        case
        for case in read_jsonl(
            STRESS_ROOT
            / "protocol"
            / MODEL
            / "cases.jsonl"
        )
        if case["task"] == task
        and case["split"] == "confirmation"
    ]
    donors, audit = choose_stress_donors(
        all_cases, MODEL, task, "confirmation"
    )
    return all_cases, donors, audit


def previous_metrics(task: str) -> dict[str, Any]:
    if task == "color":
        anchor = read_json(
            OUT_ROOT
            / "anchor_subsets"
            / MODEL
            / "color"
            / "summary.json"
        )
        natural = read_json(
            OUT_ROOT
            / "anchor_natural"
            / MODEL
            / "color"
            / "summary.json"
        )
        cache = read_json(
            OUT_ROOT
            / "kv_replication"
            / MODEL
            / "color"
            / "summary.json"
        )
        selected_id = anchor["discovery_selection"][
            "selected_subset_ids"
        ][0]
        return {
            "teacher_full_donor_rate": anchor[
                "split_summary"
            ]["confirmation"][selected_id]["donor_rate"],
            "natural_full_donor_rate": next(
                item["donor_semantic_rate"]
                for name, item in natural[
                    "condition_summary"
                ].items()
                if name != "target_noop"
            ),
            "cache_target_instrument": cache["split_summary"][
                "confirmation"
            ]["target_cache"]["full_prediction_agreement"],
            "cache_source_instrument": cache["split_summary"][
                "confirmation"
            ]["all_source_cache"]["full_prediction_agreement"],
            "cache_key_donor_rate": cache["split_summary"][
                "confirmation"
            ]["source_keys_only"]["donor_rate"],
            "cache_value_donor_rate": cache["split_summary"][
                "confirmation"
            ]["source_values_only"]["donor_rate"],
        }
    previous = read_json(
        STRESS_ROOT
        / "causal"
        / MODEL
        / task
        / "summary.json"
    )
    return {
        "teacher_full_donor_rate": previous["teacher"][
            "confirmation"
        ]["conditions"]["full_source"]["donor_rate"],
        "natural_full_donor_rate": previous[
            "natural_confirmation"
        ]["conditions"]["full_source"]["donor_semantic_rate"],
        "cache_target_instrument": previous["cache"][
            "confirmation"
        ]["conditions"]["target_cache"][
            "full_prediction_agreement"
        ],
        "cache_source_instrument": previous["cache"][
            "confirmation"
        ]["conditions"]["all_source_cache"][
            "full_prediction_agreement"
        ],
        "cache_key_donor_rate": previous["cache"][
            "confirmation"
        ]["conditions"]["source_keys_only"]["donor_rate"],
        "cache_value_donor_rate": previous["cache"][
            "confirmation"
        ]["conditions"]["source_values_only"]["donor_rate"],
    }


def run() -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(prereg["source_depths"][MODEL])
    model = tokenizer = None
    started = time.time()
    summaries = {}
    try:
        model, tokenizer, device = load_model(
            MODEL, dtype=torch.bfloat16, use_8bit=False
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        effective_eos = eos_ids(model, tokenizer)
        for task in TASKS:
            if task == "color":
                cases, donors, donor_audit = attribute_cases()
            else:
                cases, donors, donor_audit = structural_cases(task)
            teacher_rows, teacher_summary = teacher_causal(
                model,
                layers,
                device,
                MODEL,
                task,
                source_depth,
                cases,
                donors,
                4,
            )
            natural_rows, natural_summary = natural_confirmation(
                model,
                tokenizer,
                layers,
                device,
                MODEL,
                task,
                source_depth,
                cases,
                donors,
                4,
                effective_eos,
            )
            cache_rows, cache_summary = cache_causal(
                model,
                layers,
                device,
                MODEL,
                task,
                source_depth,
                cases,
                donors,
                4,
            )
            previous = previous_metrics(task)
            current = {
                "teacher_full_donor_rate": teacher_summary[
                    "conditions"
                ]["full_source"]["donor_rate"],
                "natural_full_donor_rate": natural_summary[
                    "conditions"
                ]["full_source"]["donor_semantic_rate"],
                "cache_target_instrument": cache_summary[
                    "conditions"
                ]["target_cache"]["full_prediction_agreement"],
                "cache_source_instrument": cache_summary[
                    "conditions"
                ]["all_source_cache"]["full_prediction_agreement"],
                "cache_key_donor_rate": cache_summary[
                    "conditions"
                ]["source_keys_only"]["donor_rate"],
                "cache_value_donor_rate": cache_summary[
                    "conditions"
                ]["source_values_only"]["donor_rate"],
            }
            task_summary = {
                "schema_version": (
                    "phase1003_bf16_precision_task.v1"
                ),
                "phase": PHASE,
                "model": MODEL,
                "precision": "bfloat16",
                "task": task,
                "split": "confirmation",
                "n": len(cases),
                "source_depth": source_depth,
                "donor_audit": donor_audit,
                "teacher": teacher_summary,
                "natural": natural_summary,
                "cache": cache_summary,
                "formal_8bit_metrics": previous,
                "bf16_metrics": current,
                "absolute_metric_differences": {
                    key: abs(current[key] - previous[key])
                    for key in current
                },
                "instrument_disagreement_removed_in_bf16": (
                    (
                        previous["cache_target_instrument"] < 0.99
                        or previous["cache_source_instrument"] < 0.99
                    )
                    and current["cache_target_instrument"] >= 0.99
                    and current["cache_source_instrument"] >= 0.99
                ),
            }
            root = AUDIT_ROOT / task
            write_jsonl(root / "teacher_rows.jsonl", teacher_rows)
            write_jsonl(root / "natural_rows.jsonl", natural_rows)
            write_jsonl(root / "cache_rows.jsonl", cache_rows)
            write_json(root / "summary.json", task_summary)
            summaries[task] = task_summary
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": "phase1003_bf16_precision_audit.v1",
        "phase": PHASE,
        "model": MODEL,
        "precision": "bfloat16",
        "status": "complete",
        "tasks": summaries,
        "all_noop_instruments_pass": all(
            summary["teacher"]["conditions"]["target_noop"][
                "prediction_agreement"
            ] >= 0.99
            and summary["natural"]["conditions"]["target_noop"][
                "noop_sequence_agreement"
            ] >= 0.99
            for summary in summaries.values()
        ),
        "all_cache_instruments_pass": all(
            summary["cache"]["target_cache_instrument"]
            and summary["cache"]["source_cache_instrument"]
            for summary in summaries.values()
        ),
        "value_exceeds_key_in_all_tasks": all(
            summary["bf16_metrics"]["cache_value_donor_rate"]
            > summary["bf16_metrics"]["cache_key_donor_rate"]
            for summary in summaries.values()
        ),
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "This is a Qwen3 confirmation-set precision audit. It does "
            "not replace three-model 8-bit replication or establish "
            "precision invariance in GLM4/DeepSeek7B."
        ),
    }
    write_json(AUDIT_ROOT / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    run()
