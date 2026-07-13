#!/usr/bin/env python3
"""Collect Phase408 natural responses one model at a time on CUDA."""

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
from phase407_event_horizon_collection import (  # noqa: E402
    configure_determinism,
    eos_ids,
    finite_float,
)
from phase408_partition_interface_protocol import (  # noqa: E402
    FAMILIES,
    MAX_NEW_TOKENS,
    MODELS,
    OUT,
    TOP_K,
)


SOURCE = OUT / "protocol/private/phase408_all_cases.jsonl"


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


@torch.inference_mode()
def generate_case(loaded: Any, case: dict[str, Any]) -> dict[str, Any]:
    tokenizer = loaded.tokenizer
    prompt_ids = [int(item) for item in case["prompt_token_ids_private"]]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=loaded.input_device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError(f"Phase408 has no pad/eos token for {loaded.key}")
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
        raise RuntimeError("Phase408 generated score/token length mismatch")

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
    step_ledger: list[dict[str, Any]] = []
    all_raw_valid = True
    all_processed_valid = True
    for index, selected_id in enumerate(trimmed_ids):
        raw = raw_logits[index][0].float()
        processed = scores[index][0].float()
        raw_finite = torch.isfinite(raw)
        processed_finite = torch.isfinite(processed)
        raw_valid = bool(raw_finite.all())
        processed_valid = bool(processed_finite.all())
        all_raw_valid = all_raw_valid and raw_valid
        all_processed_valid = all_processed_valid and processed_valid
        safe = torch.where(raw_finite, raw, torch.full_like(raw, -torch.inf))
        top_values, top_ids = torch.topk(safe, min(TOP_K, int(raw.numel())))
        logsumexp = None
        entropy = None
        tail_mass = None
        candidate_mass = None
        generated_logprob = None
        top_log_probs = torch.full_like(top_values, torch.nan)
        if raw_valid:
            log_probs = torch.log_softmax(raw, dim=-1)
            probabilities = torch.exp(log_probs)
            logsumexp = finite_float(torch.logsumexp(raw, dim=-1))
            entropy = finite_float(-(probabilities * log_probs).sum())
            top_probabilities = probabilities[top_ids]
            tail_mass = finite_float(1.0 - top_probabilities.sum())
            candidate_tensor = torch.tensor(
                sorted(set(candidate_ids)), device=raw.device, dtype=torch.long
            )
            candidate_mass = finite_float(probabilities[candidate_tensor].sum())
            generated_logprob = finite_float(log_probs[selected_id])
            top_log_probs = log_probs[top_ids]
        top_rows = [
            {
                "rank": rank,
                "token_id_private": int(token_id),
                "token_text_private": tokenizer.decode(
                    [int(token_id)], skip_special_tokens=False
                ),
                "logit": finite_float(value),
                "logprob": finite_float(log_probability),
            }
            for rank, (token_id, value, log_probability) in enumerate(
                zip(top_ids.tolist(), top_values, top_log_probs), 1
            )
        ]
        step_ledger.append(
            {
                "step": index + 1,
                "raw_logits_valid": raw_valid,
                "processed_scores_valid": processed_valid,
                "nonfinite_raw_logit_count": int((~raw_finite).sum().item()),
                "nonfinite_processed_score_count": int(
                    (~processed_finite).sum().item()
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
                "top_k_private": top_rows,
            }
        )
        del raw, processed, raw_finite, processed_finite, safe, top_values, top_ids

    generated_text_raw = tokenizer.decode(trimmed_ids, skip_special_tokens=False)
    generated_text_clean = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    del generated, scores, raw_logits, input_ids, attention_mask
    return {
        "schema_version": "82.1.0",
        "phase_id": "Phase408-PartitionInterfaceCollection",
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
        "lexical_replica_private": case["lexical_replica_private"],
        "surface_id_private": case["surface_id_private"],
        "surface_axes_private": case["surface_axes_private"],
        "interface_private": case["interface_private"],
        "history_mode_private": "fixed_empty",
        "condition_id_private": case["condition_id_private"],
        "target_semantic_state_private": case["target_semantic_state_private"],
        "target_raw_response_class_private": case[
            "target_raw_response_class_private"
        ],
        "raw_response_aliases_private": case["raw_response_aliases_private"],
        "raw_class_to_semantic_state_private": case[
            "raw_class_to_semantic_state_private"
        ],
        "semantic_state_ids_private": case["semantic_state_ids_private"],
        "explicit_rejected_raw_classes_private": case[
            "explicit_rejected_raw_classes_private"
        ],
        "ambiguous_aliases_private": case["ambiguous_aliases_private"],
        "generated_token_ids_private": trimmed_ids,
        "generated_token_count": len(trimmed_ids),
        "generated_text_raw_private": generated_text_raw,
        "generated_text_clean_private": generated_text_clean,
        "eos_observed": eos_step is not None,
        "eos_step_private": eos_step,
        "H48_right_edge_reached": eos_step is None
        and len(trimmed_ids) == MAX_NEW_TOKENS,
        "all_generated_raw_logits_valid": all_raw_valid,
        "all_generated_processed_scores_valid": all_processed_valid,
        "step_ledger_private": step_ledger,
        "prompt_token_count": len(prompt_ids),
        "batch_size": 1,
        "padding_used": False,
        "deterministic_stepwise_greedy": True,
    }


def authorized_families(split: str) -> tuple[str, ...]:
    if split == "discovery":
        return FAMILIES
    prior = {
        "calibration": "phase408_discovery_analysis.json",
        "behavioral_holdout": "phase408_calibration_analysis.json",
    }[split]
    path = OUT / prior
    if not path.is_file():
        return ()
    return tuple(read_json(path).get("strict_crossmodel_partition_candidate_families", []))


@torch.inference_mode()
def run(model_key: str, split: str) -> dict[str, Any]:
    qualification_path = OUT / "qualification" / f"{model_key}_complete.json"
    if not qualification_path.is_file():
        raise RuntimeError(f"Phase408 qualification missing: {model_key}")
    qualification = read_json(qualification_path)
    if not qualification.get("valid"):
        raise RuntimeError(f"Phase408 qualification invalid: {model_key}")
    selected_dtype = qualification["selected_runtime_dtype"]
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
            "schema_version": "82.1.0",
            "phase_id": "Phase408-PartitionInterfaceCollection",
            "created_at": now(),
            "model": model_key,
            "split": split,
            "authorized_families": [],
            "case_count": 0,
            "valid": True,
            "stopped_by_prior_gate": True,
        }
        write_json(complete_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    existing = read_jsonl(output_path)
    existing_by_id = {row["blind_case_id"]: row for row in existing}
    case_ids = {row["blind_case_id"] for row in cases}
    if len(existing_by_id) != len(existing):
        raise RuntimeError(f"Phase408 duplicate persisted row: {model_key}/{split}")
    if not set(existing_by_id).issubset(case_ids):
        raise RuntimeError(f"Phase408 stale persisted rows: {model_key}/{split}")

    loaded = None
    prior_dtype = os.environ.get("PROBE_TORCH_DTYPE")
    try:
        os.environ["PROBE_TORCH_DTYPE"] = selected_dtype
        configure_determinism()
        loaded = load_probe_model(model_key)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace(
            "torch.", ""
        )
        if runtime_dtype != selected_dtype:
            raise RuntimeError(
                f"Phase408 dtype mismatch {model_key}: {runtime_dtype} != {selected_dtype}"
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
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                )
                completed += 1
                if completed % 8 == 0:
                    handle.flush()
                if completed % 32 == 0 or completed == len(cases):
                    print(
                        f"[{model_key}/phase408/{split}] {completed}/{len(cases)}",
                        flush=True,
                    )
        rows = read_jsonl(output_path)
        family_counts = Counter(row["family_id"] for row in rows)
        payload = {
            "schema_version": "82.1.0",
            "phase_id": "Phase408-PartitionInterfaceCollection",
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
            "nonfinite_raw_case_count": sum(
                not row["all_generated_raw_logits_valid"] for row in rows
            ),
            "nonfinite_processed_case_count": sum(
                not row["all_generated_processed_scores_valid"] for row in rows
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
                "stepwise_greedy_is_global_sequence_map": False,
                "candidate_first_token_mass_is_semantic_set_mass": False,
                "H48_is_natural_completion_boundary": False,
            },
        }
        write_json(complete_path, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    finally:
        release_loaded(loaded)
        if prior_dtype is None:
            os.environ.pop("PROBE_TORCH_DTYPE", None)
        else:
            os.environ["PROBE_TORCH_DTYPE"] = prior_dtype
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
