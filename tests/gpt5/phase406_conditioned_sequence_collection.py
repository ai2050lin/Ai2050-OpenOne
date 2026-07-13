#!/usr/bin/env python3
"""Collect deterministic short sequences for the Phase406 condition table."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase406_conditioned_sequence_protocol import (  # noqa: E402
    BATCH_SIZE_BY_MODEL,
    FAMILIES,
    FROZEN_DTYPES,
    MAX_NEW_TOKENS,
    MODELS,
    OUT,
    TOP_K,
)


SOURCE = OUT / "protocol/private/phase406_all_cases.jsonl"


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


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def authorized_families(split: str) -> tuple[str, ...]:
    if split == "discovery":
        return FAMILIES
    if split == "calibration":
        return tuple(
            read_json(OUT / "phase406_discovery_analysis.json")[
                "crossmodel_candidate_families"
            ]
        )
    if split == "behavioral_holdout":
        return tuple(
            read_json(OUT / "phase406_calibration_analysis.json")[
                "crossmodel_candidate_families"
            ]
        )
    raise KeyError(split)


def eos_ids(tokenizer: Any, model: Any) -> set[int]:
    result: set[int] = set()
    values = (
        getattr(tokenizer, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
    )
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(item) for item in value)
        else:
            result.add(int(value))
    return result


def finite_float(value: torch.Tensor) -> float | None:
    return float(value.item()) if bool(torch.isfinite(value)) else None


@torch.inference_mode()
def run(model: str, split: str) -> dict[str, Any]:
    families = authorized_families(split)
    cases = [
        row
        for row in read_jsonl(SOURCE)
        if row["private_execution_model"] == model
        and row["candidate_split_private"] == split
        and row["family_id"] in families
    ]
    if not cases:
        payload = {
            "schema_version": "80.1.0",
            "phase_id": "Phase406-ConditionedSequenceCollection",
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
    batch_count = 0
    try:
        loaded = load_probe_model(model)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if runtime_dtype != FROZEN_DTYPES[model]:
            raise RuntimeError(f"Phase406 dtype mismatch for {model}: {runtime_dtype}")
        tokenizer = loaded.tokenizer
        model_eos_ids = eos_ids(tokenizer, loaded.model)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise RuntimeError(f"Phase406 has no pad/eos token for {model}")

        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            by_length[int(case["prompt_token_count"])].append(case)

        max_batch = BATCH_SIZE_BY_MODEL[model]
        completed = 0
        for prompt_length in sorted(by_length):
            for batch_cases in chunks(by_length[prompt_length], max_batch):
                batch_count += 1
                input_ids = torch.tensor(
                    [case["prompt_token_ids_private"] for case in batch_cases],
                    dtype=torch.long,
                    device=loaded.input_device,
                )
                if input_ids.shape[1] != prompt_length:
                    raise RuntimeError("Phase406 exact-length bucket mismatch")
                attention_mask = torch.ones_like(input_ids)
                generated = loaded.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_logits=True,
                    pad_token_id=int(pad_token_id),
                )
                scores = list(generated.scores)
                raw_logits = list(generated.logits or ())
                new_token_matrix = generated.sequences[:, prompt_length:]
                if len(scores) != int(new_token_matrix.shape[1]):
                    raise RuntimeError("Phase406 generated score/token length mismatch")
                if len(raw_logits) != int(new_token_matrix.shape[1]):
                    raise RuntimeError("Phase406 raw logit/token length mismatch")

                for row_index, case in enumerate(batch_cases):
                    raw_ids = [
                        int(token_id)
                        for token_id in new_token_matrix[row_index].detach().cpu().tolist()
                    ]
                    trimmed_ids: list[int] = []
                    stop_step: int | None = None
                    for step_index, token_id in enumerate(raw_ids):
                        trimmed_ids.append(token_id)
                        if token_id in model_eos_ids:
                            stop_step = step_index
                            break

                    step_ledger = []
                    all_steps_valid = True
                    for step_index in range(len(trimmed_ids)):
                        step_scores = raw_logits[step_index][row_index].float()
                        generation_scores = scores[step_index][row_index].float()
                        logits_valid = bool(torch.isfinite(step_scores).all())
                        generation_score_nonfinite_count = int(
                            (~torch.isfinite(generation_scores)).sum().item()
                        )
                        all_steps_valid = all_steps_valid and logits_valid
                        top_values, top_ids = torch.topk(
                            step_scores, k=min(TOP_K, int(step_scores.numel()))
                        )
                        top_rows = []
                        for rank, (token_id, value) in enumerate(
                            zip(top_ids.tolist(), top_values), 1
                        ):
                            top_rows.append(
                                {
                                    "rank": rank,
                                    "token_id_private": int(token_id),
                                    "token_text_private": tokenizer.decode(
                                        [int(token_id)], skip_special_tokens=False
                                    ),
                                    "logit": finite_float(value),
                                }
                            )
                        selected_id = trimmed_ids[step_index]
                        selected_logit = step_scores[selected_id]
                        step_ledger.append(
                            {
                                "step": step_index,
                                "logits_valid": logits_valid,
                                "vocab_size": int(step_scores.numel()),
                                "selected_token_id_private": selected_id,
                                "selected_token_text_private": tokenizer.decode(
                                    [selected_id], skip_special_tokens=False
                                ),
                                "selected_logit": finite_float(selected_logit),
                                "selected_equals_raw_argmax": selected_id
                                == int(torch.argmax(step_scores).item()),
                                "generation_score_nonfinite_count": generation_score_nonfinite_count,
                                "top_k_private": top_rows,
                            }
                        )

                    first_scores = raw_logits[0][row_index].float()
                    candidates = case["semantic_candidate_labels_private"]
                    candidate_ids = [
                        int(case["semantic_candidate_token_ids_private"][candidate])
                        for candidate in candidates
                    ]
                    candidate_values = first_scores[
                        torch.tensor(
                            candidate_ids,
                            device=first_scores.device,
                            dtype=torch.long,
                        )
                    ]
                    candidate_valid = bool(torch.isfinite(candidate_values).all())
                    if candidate_valid:
                        ranking_positions = sorted(
                            range(len(candidates)),
                            key=lambda position: float(candidate_values[position]),
                            reverse=True,
                        )
                        candidate_ranking = [
                            candidates[position] for position in ranking_positions
                        ]
                        predicted_candidate: str | None = candidate_ranking[0]
                    else:
                        candidate_ranking = []
                        predicted_candidate = None

                    target = case["target_semantic_label_private"]
                    first_valid = bool(torch.isfinite(first_scores).all())
                    first_global_top = (
                        int(torch.argmax(first_scores).item()) if first_valid else None
                    )
                    generated_text_raw = tokenizer.decode(
                        trimmed_ids, skip_special_tokens=False
                    )
                    generated_text_clean = tokenizer.decode(
                        trimmed_ids, skip_special_tokens=True
                    )
                    rows.append(
                        {
                            "schema_version": "80.1.0",
                            "phase_id": "Phase406-ConditionedSequenceCollection",
                            "created_at": now(),
                            "model": model,
                            "runtime_dtype": runtime_dtype,
                            "split": split,
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
                            "interface_private": case["interface_private"],
                            "condition_id_private": case["condition_id_private"],
                            "target_semantic_label_private": target,
                            "semantic_candidate_labels_private": candidates,
                            "semantic_aliases_private": case[
                                "semantic_aliases_private"
                            ],
                            "semantic_candidate_token_ids_private": case[
                                "semantic_candidate_token_ids_private"
                            ],
                            "candidate_logits_valid": candidate_valid,
                            "candidate_logits_private": {
                                candidate: (
                                    finite_float(candidate_values[position])
                                    if candidate_valid
                                    else None
                                )
                                for position, candidate in enumerate(candidates)
                            },
                            "candidate_ranking_private": candidate_ranking,
                            "predicted_candidate_private": predicted_candidate,
                            "first_step_candidate_correct": candidate_valid
                            and predicted_candidate == target,
                            "first_step_global_logits_valid": first_valid,
                            "first_step_global_top_token_id_private": first_global_top,
                            "first_step_global_top_text_private": (
                                tokenizer.decode(
                                    [first_global_top], skip_special_tokens=False
                                )
                                if first_global_top is not None
                                else None
                            ),
                            "first_step_global_top_is_target": first_global_top
                            == int(case["target_token_id_private"]),
                            "generated_token_ids_private": trimmed_ids,
                            "generated_token_count": len(trimmed_ids),
                            "generated_text_raw_private": generated_text_raw,
                            "generated_text_clean_private": generated_text_clean,
                            "eos_observed": stop_step is not None,
                            "eos_step_private": stop_step,
                            "all_generated_step_logits_valid": all_steps_valid,
                            "step_ledger_private": step_ledger,
                            "prompt_token_count": prompt_length,
                            "batch_size": len(batch_cases),
                            "exact_length_batch_no_padding": True,
                        }
                    )

                completed += len(batch_cases)
                if completed % 128 < len(batch_cases) or completed == len(cases):
                    print(
                        f"[{model}/phase406/{split}] {completed}/{len(cases)} "
                        f"batches={batch_count}",
                        flush=True,
                    )
                del generated, scores, raw_logits, new_token_matrix, input_ids, attention_mask

        family_counts = Counter(row["family_id"] for row in rows)
        payload = {
            "schema_version": "80.1.0",
            "phase_id": "Phase406-ConditionedSequenceCollection",
            "created_at": now(),
            "model": model,
            "runtime_dtype": runtime_dtype,
            "split": split,
            "authorized_families": list(families),
            "case_count": len(rows),
            "batch_count": batch_count,
            "max_batch_size": max_batch,
            "batching": "same_prompt_token_length_only_no_padding",
            "max_new_tokens": MAX_NEW_TOKENS,
            "top_k_ledger_size": TOP_K,
            "first_step_candidate_correct_count": sum(
                row["first_step_candidate_correct"] for row in rows
            ),
            "first_step_global_top_is_target_count": sum(
                row["first_step_global_top_is_target"] for row in rows
            ),
            "eos_observed_count": sum(row["eos_observed"] for row in rows),
            "nonfinite_first_step_candidate_case_count": sum(
                not row["candidate_logits_valid"] for row in rows
            ),
            "nonfinite_first_step_global_case_count": sum(
                not row["first_step_global_logits_valid"] for row in rows
            ),
            "nonfinite_any_generated_step_case_count": sum(
                not row["all_generated_step_logits_valid"] for row in rows
            ),
            "generation_processor_masked_case_count": sum(
                any(
                    step["generation_score_nonfinite_count"] > 0
                    for step in row["step_ledger_private"]
                )
                for row in rows
            ),
            "families": [
                {
                    "family_id": family,
                    "case_count": family_counts[family],
                    "first_step_candidate_correct_count": sum(
                        row["first_step_candidate_correct"]
                        for row in rows
                        if row["family_id"] == family
                    ),
                    "first_step_global_top_is_target_count": sum(
                        row["first_step_global_top_is_target"]
                        for row in rows
                        if row["family_id"] == family
                    ),
                    "eos_observed_count": sum(
                        row["eos_observed"]
                        for row in rows
                        if row["family_id"] == family
                    ),
                }
                for family in families
            ],
            "valid": len(rows) == len(cases)
            and all(row["exact_length_batch_no_padding"] for row in rows),
            "claim_boundary": {
                "generated_sequence_is_semantic_state": False,
                "top_k_ledger_is_full_vocab_distribution": False,
                "deterministic_greedy_is_complete_response_kernel": False,
                "raw_model_logits_separated_from_generation_processor_scores": True,
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
