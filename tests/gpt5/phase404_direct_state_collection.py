#!/usr/bin/env python3
"""Collect Phase404 full next-token and finite-candidate responses."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase404_direct_state_protocol import (  # noqa: E402
    FAMILIES,
    FROZEN_DTYPES,
    MODELS,
    OUT,
)


SOURCE = OUT / "protocol/private/phase404_all_cases.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
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


def authorized_families(split: str) -> tuple[str, ...]:
    if split == "discovery":
        return FAMILIES
    if split == "calibration":
        return tuple(
            read_json(OUT / "phase404_discovery_analysis.json")[
                "crossmodel_candidate_families"
            ]
        )
    if split == "behavioral_holdout":
        return tuple(
            read_json(OUT / "phase404_calibration_analysis.json")[
                "crossmodel_candidate_families"
            ]
        )
    raise KeyError(split)


@torch.inference_mode()
def run(model: str, split: str) -> dict[str, Any]:
    families = authorized_families(split)
    all_cases = read_jsonl(SOURCE)
    cases = [
        row
        for row in all_cases
        if row["private_execution_model"] == model
        and row["candidate_split_private"] == split
        and row["family_id"] in families
    ]
    if not cases:
        payload = {
            "schema_version": "78.1.0",
            "phase_id": "Phase404-DirectStateCollection",
            "created_at": now(),
            "model": model,
            "split": split,
            "authorized_families": [],
            "case_count": 0,
            "valid": True,
            "stopped_by_prior_gate": True,
        }
        write_json(OUT / "collection" / split / model / "complete.json", payload)
        return payload

    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if runtime_dtype != FROZEN_DTYPES[model]:
            raise RuntimeError(
                f"Phase404 dtype mismatch for {model}: {runtime_dtype}"
            )
        tokenizer = loaded.tokenizer
        for index, case in enumerate(cases, 1):
            encoded = tokenizer(
                case["prompt"],
                add_special_tokens=case["tokenization_add_special_tokens"],
                return_tensors="pt",
            )
            if int(encoded["attention_mask"].sum()) != int(
                encoded["input_ids"].shape[1]
            ):
                raise RuntimeError("Phase404 formal batch=1 unexpectedly padded")
            encoded = {
                key: value.to(loaded.input_device) for key, value in encoded.items()
            }
            output = loaded.model(**encoded, use_cache=True, return_dict=True)
            logits = output.logits[0, -1].float()
            candidates = case["candidate_answers_private"]
            token_ids = [
                int(case["candidate_token_ids_private"][candidate])
                for candidate in candidates
            ]
            candidate_logits_tensor = logits[
                torch.tensor(token_ids, device=logits.device, dtype=torch.long)
            ]
            candidate_logits = candidate_logits_tensor.cpu()
            target = case["target_private"]
            target_position = candidates.index(target)
            candidate_logits_valid = bool(torch.isfinite(candidate_logits).all())
            if candidate_logits_valid:
                candidate_probabilities = torch.softmax(
                    candidate_logits, dim=0
                )
                order = sorted(
                    range(len(candidates)),
                    key=lambda position: float(candidate_logits[position]),
                    reverse=True,
                )
                predicted: str | None = candidates[order[0]]
                distractor_positions = [
                    position
                    for position in range(len(candidates))
                    if position != target_position
                ]
                target_margin: float | None = float(
                    candidate_logits[target_position]
                ) - max(
                    float(candidate_logits[position])
                    for position in distractor_positions
                )
                target_rank: int | None = order.index(target_position) + 1
                ranking = [candidates[position] for position in order]
            else:
                candidate_probabilities = None
                predicted = None
                target_margin = None
                target_rank = None
                ranking = []
            global_top_id = int(torch.argmax(logits).item())
            global_top_text = tokenizer.decode([global_top_id])
            rows.append(
                {
                    "schema_version": "78.1.0",
                    "phase_id": "Phase404-DirectStateCollection",
                    "created_at": now(),
                    "model": model,
                    "runtime_dtype": runtime_dtype,
                    "split": split,
                    "batch_size": 1,
                    "blind_case_id": case["blind_case_id"],
                    "family_id": case["family_id"],
                    "anonymous_parallel_group_id": case[
                        "anonymous_parallel_group_id"
                    ],
                    "group_priority": case["group_priority"],
                    "state_id_private": case["state_id_private"],
                    "abstract_state_private": case["abstract_state_private"],
                    "surface_id_private": case["surface_id_private"],
                    "surface_axes_private": case["surface_axes_private"],
                    "future_query_private": case["future_query_private"],
                    "target_private": target,
                    "candidate_answers_private": candidates,
                    "candidate_token_ids_private": case[
                        "candidate_token_ids_private"
                    ],
                    "candidate_logits_valid": candidate_logits_valid,
                    "candidate_logits_private": (
                        {
                            candidate: float(candidate_logits[position])
                            for position, candidate in enumerate(candidates)
                        }
                        if candidate_logits_valid
                        else {candidate: None for candidate in candidates}
                    ),
                    "candidate_probabilities_private": (
                        {
                            candidate: float(candidate_probabilities[position])
                            for position, candidate in enumerate(candidates)
                        }
                        if candidate_probabilities is not None
                        else {candidate: None for candidate in candidates}
                    ),
                    "candidate_ranking_private": ranking,
                    "predicted_candidate_private": predicted,
                    "finite_candidate_correct": candidate_logits_valid
                    and predicted == target,
                    "target_candidate_rank": target_rank,
                    "target_minus_best_distractor_logit": target_margin,
                    "global_top_token_id_private": global_top_id,
                    "global_top_token_text_private": global_top_text,
                    "global_top_is_target_token": global_top_id
                    == int(case["target_token_id_private"]),
                    "prompt_token_count": case["prompt_token_count"],
                    "formal_single_case_unpadded": True,
                }
            )
            if index % 128 == 0 or index == len(cases):
                print(
                    f"[{model}/phase404/{split}] {index}/{len(cases)} "
                    f"finite_correct={sum(row['finite_candidate_correct'] for row in rows)}",
                    flush=True,
                )
            if index % 256 == 0:
                del output, logits
                gc.collect()

        family_counts = Counter(row["family_id"] for row in rows)
        family_correct = Counter(
            row["family_id"]
            for row in rows
            if row["finite_candidate_correct"]
        )
        payload = {
            "schema_version": "78.1.0",
            "phase_id": "Phase404-DirectStateCollection",
            "created_at": now(),
            "model": model,
            "runtime_dtype": runtime_dtype,
            "split": split,
            "authorized_families": list(families),
            "batch_size": 1,
            "case_count": len(rows),
            "finite_candidate_correct_count": sum(
                row["finite_candidate_correct"] for row in rows
            ),
            "nonfinite_candidate_logit_case_count": sum(
                not row["candidate_logits_valid"] for row in rows
            ),
            "global_top_is_target_count": sum(
                row["global_top_is_target_token"] for row in rows
            ),
            "families": [
                {
                    "family_id": family,
                    "case_count": family_counts[family],
                    "finite_candidate_correct_count": family_correct[family],
                }
                for family in families
            ],
            "valid": all(row["formal_single_case_unpadded"] for row in rows),
            "claim_boundary": {
                "finite_candidate_distribution_is_full_vocabulary_distribution": False,
                "candidate_correct_is_internal_mechanism": False,
            },
        }
        write_jsonl(
            OUT / "collection" / split / "private" / model / "rows.jsonl",
            rows,
        )
        write_json(OUT / "collection" / split / model / "complete.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--split",
        choices=("discovery", "calibration", "behavioral_holdout"),
        required=True,
    )
    args = parser.parse_args()
    run(args.model, args.split)
