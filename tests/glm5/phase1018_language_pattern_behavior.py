#!/usr/bin/env python3
"""Calibrate and record Phase1018 one-token behavior by pattern family."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch


ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import release_model
from phase1014_bf16_precision_confirmation import load_bf16
from phase1018_language_pattern_protocol import (
    FACTORIAL_STATES,
    FAMILIES,
    MODELS,
    OUT_ROOT,
    PHASE,
    PROMPT_MODES,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


BATCH_SIZE = 32


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def homogeneous_batches(
    rows: list[dict[str, Any]],
    size: int,
) -> Iterable[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[len(row["input_ids"])].append(row)
    for width in sorted(grouped):
        yield from chunks(grouped[width], size)


def load_cases(
    model_name: str,
    prompt_mode: str,
    *,
    discovery_only: bool,
) -> list[dict[str, Any]]:
    rows = read_jsonl(
        OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.{prompt_mode}.jsonl"
    )
    rows = [row for row in rows if row["state"] in FACTORIAL_STATES]
    if discovery_only:
        rows = [
            row
            for row in rows
            if row["split"] == "discovery" and int(row["world"]) < 2
        ]
    return rows


def evaluate(
    *,
    model,
    device,
    model_name: str,
    prompt_mode: str,
    scope: str,
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_entries = []
    case_by_id = {case["record_id"]: case for case in cases}
    for case in cases:
        for label, continuation in case["candidate_token_ids"].items():
            candidate_entries.append({
                "record_id": case["record_id"],
                "label": label,
                "prompt_length": len(case["input_ids"]),
                "continuation": [int(value) for value in continuation],
                "input_ids": (
                    list(case["input_ids"])
                    + [int(value) for value in continuation]
                ),
            })
    scores: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    started = time.time()
    for batch_index, batch in enumerate(
        homogeneous_batches(candidate_entries, BATCH_SIZE),
        1,
    ):
        input_ids = torch.tensor(
            [row["input_ids"] for row in batch],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = output.logits.float()
        for index, entry in enumerate(batch):
            prompt_length = int(entry["prompt_length"])
            continuation = entry["continuation"]
            token_scores = []
            for offset, token_id in enumerate(continuation):
                position_logits = logits[index, prompt_length + offset - 1]
                token_scores.append(float(
                    position_logits[int(token_id)].item()
                    - torch.logsumexp(position_logits, dim=-1).item()
                ))
            first_logits = logits[index, prompt_length - 1]
            scores[entry["record_id"]][entry["label"]] = {
                "mean_log_probability": float(np.mean(token_scores)),
                "sum_log_probability": float(np.sum(token_scores)),
                "token_count": len(token_scores),
                "first_token_prediction_id": int(
                    first_logits.argmax().item()
                ),
            }
        del output, logits, input_ids, attention_mask
        if batch_index % 20 == 0:
            print(
                f"[behavior] {model_name}/{prompt_mode}/{scope} "
                f"batch={batch_index}",
                flush=True,
            )

    rows = []
    for case in cases:
        candidate_scores = scores[case["record_id"]]
        gold_score = candidate_scores[case["gold"]]
        foil_score = candidate_scores[case["foil"]]
        top_id = int(gold_score["first_token_prediction_id"])
        gold_first = int(
            case["candidate_first_token_ids"][case["gold"]]
        )
        foil_first = int(
            case["candidate_first_token_ids"][case["foil"]]
        )
        rows.append({
                "schema_version": "phase1018_pattern_behavior_row.v1",
                "phase": PHASE,
                "protocol_revision": PROTOCOL_REVISION,
                "model": model_name,
                "prompt_mode": prompt_mode,
                "scope": scope,
                "record_id": case["record_id"],
                "unit_id": case["unit_id"],
                "family": case["family"],
                "subgroup": case["subgroup"],
                "item_id": case["item_id"],
                "split": case["split"],
                "template": int(case["template"]),
                "world": int(case["world"]),
                "state": case["state"],
                "gold": case["gold"],
                "foil": case["foil"],
                "candidate_margin": float(
                    gold_score["mean_log_probability"]
                    - foil_score["mean_log_probability"]
                ),
                "candidate_hit": bool(
                    gold_score["mean_log_probability"]
                    > foil_score["mean_log_probability"]
                ),
                "gold_mean_log_probability": (
                    gold_score["mean_log_probability"]
                ),
                "foil_mean_log_probability": (
                    foil_score["mean_log_probability"]
                ),
                "gold_token_count": int(gold_score["token_count"]),
                "foil_token_count": int(foil_score["token_count"]),
                "first_token_prediction_id": top_id,
                "first_token_hit": bool(
                    top_id == gold_first and gold_first != foil_first
                ),
            })
    summary = summarize(rows)
    summary.update({
        "schema_version": "phase1018_pattern_behavior_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "scope": scope,
        "elapsed_seconds": time.time() - started,
    })
    return rows, summary


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(subset),
            "candidate_accuracy": float(np.mean([
                row["candidate_hit"] for row in subset
            ])) if subset else None,
            "first_token_accuracy": float(np.mean([
                row["first_token_hit"] for row in subset
            ])) if subset else None,
            "median_candidate_margin": float(np.median([
                row["candidate_margin"] for row in subset
            ])) if subset else None,
        }

    return {
        "case_count": len(rows),
        **metrics(rows),
        "by_family": {
            family: metrics([
                row for row in rows if row["family"] == family
            ])
            for family in FAMILIES
        },
        "by_family_split": {
            f"{family}:{split}": metrics([
                row
                for row in rows
                if row["family"] == family and row["split"] == split
            ])
            for family in FAMILIES
            for split in ("discovery", "confirmation")
            if any(
                row["family"] == family and row["split"] == split
                for row in rows
            )
        },
        "by_subgroup": {
            subgroup: metrics([
                row for row in rows if row["subgroup"] == subgroup
            ])
            for subgroup in sorted({row["subgroup"] for row in rows})
        },
    }


def run_model(model_name: str) -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    output_root = OUT_ROOT / "behavior" / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    model = tokenizer = device = None
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        calibration = {}
        calibration_rows_by_mode = {}
        for prompt_mode in PROMPT_MODES:
            rows, summary = evaluate(
                model=model,
                device=device,
                model_name=model_name,
                prompt_mode=prompt_mode,
                scope="calibration",
                cases=load_cases(
                    model_name,
                    prompt_mode,
                    discovery_only=True,
                ),
            )
            summary["protocol_digest"] = prereg["protocol_digest"]
            summary["placement"] = placement
            calibration[prompt_mode] = summary
            calibration_rows_by_mode[prompt_mode] = rows
            write_jsonl(
                output_root / f"calibration.{prompt_mode}.jsonl",
                rows,
            )
            write_json(
                output_root / f"calibration.{prompt_mode}.summary.json",
                summary,
            )

        selected_by_family = {}
        for family in FAMILIES:
            ranked = sorted(
                PROMPT_MODES,
                key=lambda mode: (
                    calibration[mode]["by_family"][family][
                        "first_token_accuracy"
                    ],
                    calibration[mode]["by_family"][family][
                        "candidate_accuracy"
                    ],
                    mode == "native_chat",
                ),
                reverse=True,
            )
            selected_by_family[family] = ranked[0]
        selection = {
            "schema_version": "phase1018_prompt_mode_selection.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "model": model_name,
            "selection_used_discovery_only": True,
            "selected_by_family": selected_by_family,
            "selection_primary": "first_token_accuracy",
            "selection_secondary": "candidate_accuracy",
            "calibration": calibration,
        }
        write_json(output_root / "selection.json", selection)

        formal_cases = []
        for family, mode in selected_by_family.items():
            formal_cases.extend([
                row
                for row in load_cases(
                    model_name,
                    mode,
                    discovery_only=False,
                )
                if row["family"] == family
            ])
        formal_rows, formal_summary = evaluate(
            model=model,
            device=device,
            model_name=model_name,
            prompt_mode="family_selected",
            scope="formal",
            cases=formal_cases,
        )
        formal_summary["protocol_digest"] = prereg["protocol_digest"]
        formal_summary["selected_by_family"] = selected_by_family
        formal_summary["placement"] = placement
        write_jsonl(output_root / "formal.jsonl", formal_rows)
        write_json(output_root / "formal.summary.json", formal_summary)
        print(json.dumps({
            "model": model_name,
            "selected_by_family": selected_by_family,
            "formal": formal_summary["by_family"],
        }, ensure_ascii=False, indent=2))
        return formal_summary
    finally:
        if model is not None:
            release_model(model)
        del model, tokenizer, device
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    run_model(args.model)


if __name__ == "__main__":
    main()
