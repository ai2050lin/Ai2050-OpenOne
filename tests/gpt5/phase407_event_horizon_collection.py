#!/usr/bin/env python3
"""Collect Phase407 batch-one event-horizon response ledgers."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase407_event_horizon_protocol import (  # noqa: E402
    FAMILIES,
    FROZEN_DTYPES,
    MAX_NEW_TOKENS,
    MODELS,
    OUT,
    TOP_K,
)


SOURCE = OUT / "protocol/private/phase407_all_cases.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
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


def configure_determinism() -> None:
    torch.manual_seed(407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(407)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def eos_ids(tokenizer: Any, model: Any) -> set[int]:
    result: set[int] = set()
    for value in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
    ):
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(item) for item in value)
        else:
            result.add(int(value))
    return result


def finite_float(value: torch.Tensor | float, digits: int = 7) -> float | None:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    if not bool(torch.isfinite(tensor)):
        return None
    return round(float(tensor.item()), digits)


@torch.inference_mode()
def continuation_score(
    loaded: Any, prompt_ids: list[int], continuation_ids: list[int]
) -> dict[str, Any]:
    combined = prompt_ids + continuation_ids
    input_ids = torch.tensor(
        [combined], dtype=torch.long, device=loaded.input_device
    )
    attention_mask = torch.ones_like(input_ids)
    output = loaded.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = output.logits[0].float()
    positions = [len(prompt_ids) - 1 + index for index in range(len(continuation_ids))]
    selected = []
    valid = True
    for position, token_id in zip(positions, continuation_ids):
        row = logits[position]
        row_valid = bool(torch.isfinite(row).all())
        valid = valid and row_valid
        if row_valid:
            log_probability = torch.log_softmax(row, dim=-1)[token_id]
            selected.append(finite_float(log_probability))
        else:
            selected.append(None)
    values = [value for value in selected if value is not None]
    result = {
        "valid": valid and len(values) == len(continuation_ids),
        "token_count": len(continuation_ids),
        "token_logprob_private": selected,
        "sum_logprob": round(sum(values), 7) if len(values) == len(continuation_ids) else None,
        "mean_logprob": (
            round(sum(values) / len(values), 7)
            if len(values) == len(continuation_ids) and values
            else None
        ),
    }
    del output, logits, input_ids, attention_mask
    return result


@torch.inference_mode()
def generate_case(loaded: Any, case: dict[str, Any]) -> dict[str, Any]:
    tokenizer = loaded.tokenizer
    prompt_ids = [int(item) for item in case["prompt_token_ids_private"]]
    input_ids = torch.tensor(
        [prompt_ids], dtype=torch.long, device=loaded.input_device
    )
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError(f"Phase407 has no pad/eos token for {loaded.key}")
    model_eos_ids = eos_ids(tokenizer, loaded.model)

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
    new_ids = [
        int(item)
        for item in generated.sequences[0, len(prompt_ids) :].detach().cpu().tolist()
    ]
    if len(scores) != len(new_ids) or len(raw_logits) != len(new_ids):
        raise RuntimeError("Phase407 generated score/token length mismatch")

    trimmed_ids: list[int] = []
    eos_step: int | None = None
    for step, token_id in enumerate(new_ids, 1):
        trimmed_ids.append(token_id)
        if token_id in model_eos_ids:
            eos_step = step
            break

    candidate_ids = [
        int(item) for item in case["registered_candidate_first_token_ids_private"]
    ]
    target_first_id = int(case["target_completion_token_ids_private"][0])
    step_ledger: list[dict[str, Any]] = []
    all_steps_valid = True
    for index, selected_id in enumerate(trimmed_ids):
        raw = raw_logits[index][0].float()
        processor = scores[index][0].float()
        finite_mask = torch.isfinite(raw)
        logits_valid = bool(finite_mask.all())
        all_steps_valid = all_steps_valid and logits_valid
        safe = torch.where(finite_mask, raw, torch.full_like(raw, -torch.inf))
        top_values, top_ids = torch.topk(safe, min(TOP_K, int(raw.numel())))
        top_rows = []
        logsumexp = None
        entropy = None
        tail_mass = None
        candidate_mass = None
        target_first_mass = None
        generated_logprob = None
        if logits_valid:
            log_probs = torch.log_softmax(raw, dim=-1)
            probabilities = torch.exp(log_probs)
            logsumexp = finite_float(torch.logsumexp(raw, dim=-1))
            entropy = finite_float(-(probabilities * log_probs).sum())
            top_probabilities = probabilities[top_ids]
            tail_mass = finite_float(1.0 - top_probabilities.sum())
            candidate_tensor = torch.tensor(
                sorted(set(candidate_ids)),
                device=raw.device,
                dtype=torch.long,
            )
            candidate_mass = finite_float(probabilities[candidate_tensor].sum())
            target_first_mass = finite_float(probabilities[target_first_id])
            generated_logprob = finite_float(log_probs[selected_id])
            top_log_probs = log_probs[top_ids]
        else:
            top_log_probs = torch.full_like(top_values, torch.nan)
        for rank, (token_id, value, log_probability) in enumerate(
            zip(top_ids.tolist(), top_values, top_log_probs), 1
        ):
            top_rows.append(
                {
                    "rank": rank,
                    "token_id_private": int(token_id),
                    "token_text_private": tokenizer.decode(
                        [int(token_id)], skip_special_tokens=False
                    ),
                    "logit": finite_float(value),
                    "logprob": finite_float(log_probability),
                }
            )
        step_ledger.append(
            {
                "step": index + 1,
                "logits_valid": logits_valid,
                "nonfinite_logit_count": int((~finite_mask).sum().item()),
                "generation_score_nonfinite_count": int(
                    (~torch.isfinite(processor)).sum().item()
                ),
                "vocab_size": int(raw.numel()),
                "selected_token_id_private": selected_id,
                "selected_token_text_private": tokenizer.decode(
                    [selected_id], skip_special_tokens=False
                ),
                "decoded_prefix_private": tokenizer.decode(
                    trimmed_ids[: index + 1], skip_special_tokens=True
                ),
                "selected_equals_raw_argmax": selected_id
                == int(torch.argmax(safe).item()),
                "generated_token_logprob": generated_logprob,
                "full_vocab_logsumexp": logsumexp,
                "full_vocab_entropy": entropy,
                "top_k_tail_probability_mass": tail_mass,
                "registered_candidate_first_token_mass": candidate_mass,
                "target_first_token_mass": target_first_mass,
                "top_k_private": top_rows,
            }
        )
        del raw, processor, finite_mask, safe, top_values, top_ids

    generated_text_raw = tokenizer.decode(trimmed_ids, skip_special_tokens=False)
    generated_text_clean = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    del generated, scores, raw_logits, input_ids, attention_mask

    target_score = continuation_score(
        loaded,
        prompt_ids,
        [int(item) for item in case["target_completion_token_ids_private"]],
    )
    foil_score = continuation_score(
        loaded,
        prompt_ids,
        [int(item) for item in case["foil_completion_token_ids_private"]],
    )
    sequence_preference_valid = target_score["valid"] and foil_score["valid"]
    target_preferred = (
        sequence_preference_valid
        and target_score["mean_logprob"] > foil_score["mean_logprob"]
    )
    return {
        "schema_version": "81.1.0",
        "phase_id": "Phase407-EventHorizonCollection",
        "created_at": now(),
        "model": loaded.key,
        "runtime_dtype": str(next(loaded.model.parameters()).dtype).replace(
            "torch.", ""
        ),
        "split": case["candidate_split_private"],
        "blind_case_id": case["blind_case_id"],
        "family_id": case["family_id"],
        "anonymous_parallel_group_id": case["anonymous_parallel_group_id"],
        "group_priority": case["group_priority"],
        "state_id_private": case["state_id_private"],
        "abstract_state_private": case["abstract_state_private"],
        "surface_id_private": case["surface_id_private"],
        "surface_axes_private": case["surface_axes_private"],
        "interface_private": case["interface_private"],
        "history_mode_private": case["history_mode_private"],
        "condition_id_private": case["condition_id_private"],
        "target_semantic_state_private": case["target_semantic_state_private"],
        "foil_semantic_state_private": case["foil_semantic_state_private"],
        "semantic_state_ids_private": case["semantic_state_ids_private"],
        "semantic_aliases_by_state_private": case[
            "semantic_aliases_by_state_private"
        ],
        "generated_token_ids_private": trimmed_ids,
        "generated_token_count": len(trimmed_ids),
        "generated_text_raw_private": generated_text_raw,
        "generated_text_clean_private": generated_text_clean,
        "eos_observed": eos_step is not None,
        "eos_step_private": eos_step,
        "H48_right_edge_reached": eos_step is None
        and len(trimmed_ids) == MAX_NEW_TOKENS,
        "all_generated_step_logits_valid": all_steps_valid,
        "step_ledger_private": step_ledger,
        "target_completion_score_private": target_score,
        "foil_completion_score_private": foil_score,
        "sequence_preference_valid": sequence_preference_valid,
        "canonical_target_preferred_to_foil": target_preferred,
        "prompt_token_count": len(prompt_ids),
        "batch_size": 1,
        "padding_used": False,
        "deterministic_greedy": True,
    }


def authorized_families(split: str) -> tuple[str, ...]:
    if split == "discovery":
        return FAMILIES
    prior = {
        "calibration": "phase407_discovery_analysis.json",
        "behavioral_holdout": "phase407_calibration_analysis.json",
    }[split]
    return tuple(
        read_json(OUT / prior)["strict_crossmodel_semantic_candidate_families"]
    )


@torch.inference_mode()
def run(model_key: str, split: str) -> dict[str, Any]:
    qualification = OUT / "qualification" / f"{model_key}_complete.json"
    if not qualification.is_file() or not read_json(qualification).get("valid"):
        raise RuntimeError(
            f"Phase407 execution qualification missing or invalid: {model_key}"
        )
    families = authorized_families(split)
    cases = [
        row
        for row in read_jsonl(SOURCE)
        if row["private_execution_model"] == model_key
        and row["candidate_split_private"] == split
        and row["family_id"] in families
    ]
    output_path = OUT / "collection" / split / "private" / model_key / "rows.jsonl"
    complete_path = OUT / "collection" / split / model_key / "complete.json"
    if not cases:
        payload = {
            "schema_version": "81.1.0",
            "phase_id": "Phase407-EventHorizonCollection",
            "created_at": now(),
            "model": model_key,
            "split": split,
            "authorized_families": [],
            "case_count": 0,
            "valid": True,
            "stopped_by_prior_gate": True,
        }
        write_json(complete_path, payload)
        return payload

    existing = read_jsonl(output_path)
    existing_by_id = {row["blind_case_id"]: row for row in existing}
    if len(existing_by_id) != len(existing):
        raise RuntimeError(f"Phase407 duplicate persisted row: {model_key}/{split}")
    case_ids = {row["blind_case_id"] for row in cases}
    if not set(existing_by_id).issubset(case_ids):
        raise RuntimeError(f"Phase407 stale persisted rows: {model_key}/{split}")

    loaded = None
    try:
        configure_determinism()
        loaded = load_probe_model(model_key)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace(
            "torch.", ""
        )
        if runtime_dtype != FROZEN_DTYPES[model_key]:
            raise RuntimeError(
                f"Phase407 dtype mismatch {model_key}: {runtime_dtype}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if output_path.is_file() else "w"
        completed = len(existing_by_id)
        with output_path.open(mode, encoding="utf-8") as handle:
            for case in cases:
                if case["blind_case_id"] in existing_by_id:
                    continue
                row = generate_case(loaded, case)
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                completed += 1
                if completed % 8 == 0:
                    handle.flush()
                if completed % 32 == 0 or completed == len(cases):
                    print(
                        f"[{model_key}/phase407/{split}] {completed}/{len(cases)}",
                        flush=True,
                    )
        rows = read_jsonl(output_path)
        family_counts = Counter(row["family_id"] for row in rows)
        payload = {
            "schema_version": "81.1.0",
            "phase_id": "Phase407-EventHorizonCollection",
            "created_at": now(),
            "model": model_key,
            "runtime_dtype": runtime_dtype,
            "split": split,
            "authorized_families": list(families),
            "case_count": len(rows),
            "batch_count": len(rows),
            "batch_size": 1,
            "padding": "none",
            "max_new_tokens": MAX_NEW_TOKENS,
            "eos_observed_count": sum(row["eos_observed"] for row in rows),
            "H48_right_edge_count": sum(row["H48_right_edge_reached"] for row in rows),
            "nonfinite_any_step_case_count": sum(
                not row["all_generated_step_logits_valid"] for row in rows
            ),
            "canonical_target_preferred_count": sum(
                row["canonical_target_preferred_to_foil"] for row in rows
            ),
            "families": [
                {"family_id": family, "case_count": family_counts[family]}
                for family in families
            ],
            "valid": len(rows) == len(cases)
            and len({row["blind_case_id"] for row in rows}) == len(rows)
            and all(row["batch_size"] == 1 for row in rows)
            and all(not row["padding_used"] for row in rows),
            "claim_boundary": {
                "greedy_trace_is_full_stochastic_kernel": False,
                "canonical_sequence_preference_is_natural_generation": False,
                "H48_is_natural_language_boundary": False,
            },
        }
        write_json(complete_path, payload)
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
