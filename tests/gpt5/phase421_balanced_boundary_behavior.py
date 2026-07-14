#!/usr/bin/env python3
"""Collect the broad Phase421 behavior boundary with batched CUDA scoring."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase416_real_collector_qualification import (  # noqa: E402
    eos_ids,
    exact_answer,
    neutral_generation_config,
    target_match,
)
from phase421_balanced_boundary_case_bank import (  # noqa: E402
    MODELS,
    OUT,
    SCHEMA_VERSION,
    continuation_ids,
    serialize_prompt,
)


PHASE_ID = "Phase421-BalancedBoundaryBehavior"
REGISTERED = OUT / "phase421_registered_conditions.jsonl"
SCORE_BATCH = {"qwen3": 16, "glm4": 8, "deepseek7b": 12}
GENERATION_BATCH = {"qwen3": 8, "glm4": 4, "deepseek7b": 6}
INITIAL_HORIZON = 12
EXTENDED_HORIZON = 24

COMPACT_KEYS = (
    "model",
    "phase421_condition_id",
    "group_id",
    "group_index",
    "split",
    "family_id",
    "mechanism_id",
    "template_id",
    "interface",
    "current_identity",
    "current_support_count",
    "history_reliability_score",
    "history_relation",
    "history_answer_role",
    "target",
    "opposite_identity_target",
    "registered_target_token_count",
    "behavior_generation_panel",
    "physical_development_panel",
    "physical_holdout_sealed",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase421 non-finite behavior scalar: {value}")
    return round(float(value), 10)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in COMPACT_KEYS}


def prompt_for(loaded: Any, row: dict[str, Any]) -> str:
    return serialize_prompt(
        loaded.tokenizer,
        row["raw_prompt"],
        row["source_fragment"],
        row["interface"],
        row["history_answer"],
        int(row["current_support_count"]),
        int(row["history_reliability_score"]),
    )


def branch_contract(tokenizer: Any, target: str, opposite: str) -> tuple[list[int], int, int]:
    target_ids = continuation_ids(tokenizer, target)
    opposite_ids = continuation_ids(tokenizer, opposite)
    if not target_ids or not opposite_ids or target_ids == opposite_ids:
        raise RuntimeError(f"Invalid branch candidates: {target!r}/{opposite!r}")
    common = 0
    while (
        common < len(target_ids)
        and common < len(opposite_ids)
        and target_ids[common] == opposite_ids[common]
    ):
        common += 1
    if common >= len(target_ids) or common >= len(opposite_ids):
        raise RuntimeError(f"One answer continuation prefixes the other: {target!r}/{opposite!r}")
    return target_ids[:common], int(target_ids[common]), int(opposite_ids[common])


def padded_batch(
    sequences: list[list[int]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[index, :length] = 1
    return input_ids.to(device), attention_mask.to(device)


@torch.inference_mode()
def final_logits_for_sequences(
    loaded: Any,
    sequences: list[list[int]],
    pad_id: int,
) -> torch.Tensor:
    input_ids, attention_mask = padded_batch(sequences, pad_id, loaded.input_device)
    result = loaded.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    final_positions = torch.tensor(
        [len(sequence) - 1 for sequence in sequences],
        dtype=torch.long,
        device=result.logits.device,
    )
    final_logits = result.logits[
        torch.arange(len(sequences), device=result.logits.device),
        final_positions,
    ].float()
    finite = bool(torch.isfinite(final_logits).all().item())
    if finite:
        output = final_logits.detach()
        del result, input_ids, attention_mask, final_positions, final_logits
        return output
    del result, input_ids, attention_mask, final_positions, final_logits
    if len(sequences) == 1:
        raise RuntimeError(
            f"Non-finite Phase421 singleton logits for {loaded.key}; "
            f"sequence_length={len(sequences[0])}"
        )
    middle = len(sequences) // 2
    print(
        f"[Phase421:{loaded.key}:margin] non-finite batch split "
        f"{len(sequences)}->{middle}+{len(sequences) - middle}",
        flush=True,
    )
    return torch.cat(
        [
            final_logits_for_sequences(loaded, sequences[:middle], pad_id),
            final_logits_for_sequences(loaded, sequences[middle:], pad_id),
        ],
        dim=0,
    )


@torch.inference_mode()
def collect_margin_scores(loaded: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batch_size = int(os.environ.get("PHASE421_SCORE_BATCH", SCORE_BATCH[loaded.key]))
    pad_id = int(loaded.tokenizer.pad_token_id)
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        sequences: list[list[int]] = []
        contracts: list[tuple[str, int, int, int, list[int]]] = []
        for row in batch_rows:
            prompt = prompt_for(loaded, row)
            prompt_ids = [
                int(value)
                for value in loaded.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            ]
            if len(prompt_ids) != int(row["registered_prompt_token_count"]):
                raise RuntimeError(f"Prompt contract changed: {row['phase421_condition_id']}")
            prefix, target_branch, opposite_branch = branch_contract(
                loaded.tokenizer,
                row["target"],
                row["opposite_identity_target"],
            )
            sequences.append(prompt_ids + prefix)
            contracts.append((prompt, len(prompt_ids), target_branch, opposite_branch, prefix))
        final_logits = final_logits_for_sequences(loaded, sequences, pad_id)
        for batch_index, row in enumerate(batch_rows):
            prompt, prompt_count, target_branch, opposite_branch, prefix = contracts[batch_index]
            target_logit = float(final_logits[batch_index, target_branch].item())
            opposite_logit = float(final_logits[batch_index, opposite_branch].item())
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase421-BehaviorMargin",
                    "created_at": now(),
                    **compact(row),
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_token_count": prompt_count,
                    "registered_prompt_token_count_pass": True,
                    "branch_common_prefix_token_count": len(prefix),
                    "target_branch_token_id": target_branch,
                    "opposite_branch_token_id": opposite_branch,
                    "target_branch_logit": clean(target_logit),
                    "opposite_branch_logit": clean(opposite_logit),
                    "target_branch_margin": clean(target_logit - opposite_logit),
                    "behavior_margin_pass": True,
                    "causal": False,
                }
            )
        del final_logits
        completed = min(start + batch_size, len(rows))
        if completed % (batch_size * 20) == 0 or completed == len(rows):
            print(
                f"[Phase421:{loaded.key}:margin] {completed}/{len(rows)}",
                flush=True,
            )
    return output


def generation_inputs(loaded: Any, rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, torch.Tensor]]:
    prompts = [prompt_for(loaded, row) for row in rows]
    loaded.tokenizer.padding_side = "left"
    encoded = loaded.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    return prompts, {key: value.to(loaded.input_device) for key, value in encoded.items()}


@torch.inference_mode()
def generate_rows(
    loaded: Any,
    rows: list[dict[str, Any]],
    horizon: int,
) -> list[dict[str, Any]]:
    prompts, encoded = generation_inputs(loaded, rows)
    input_width = int(encoded["input_ids"].shape[1])
    result = loaded.model.generate(
        **encoded,
        generation_config=neutral_generation_config(loaded),
        max_new_tokens=horizon,
        return_dict_in_generate=True,
        output_scores=False,
    )
    eos = eos_ids(loaded.tokenizer, loaded.model)
    output = []
    for index, row in enumerate(rows):
        generated = [int(value) for value in result.sequences[index, input_width:].tolist()]
        emitted_stop = any(token in eos for token in generated)
        text = loaded.tokenizer.decode(generated, skip_special_tokens=True)
        output.append(
            {
                "row": row,
                "prompt": prompts[index],
                "generated_token_ids": generated,
                "generated_text": text,
                "emitted_stop": emitted_stop,
                "right_censored": not emitted_stop and len(generated) >= horizon,
                "horizon": horizon,
            }
        )
    del result, encoded
    return output


@torch.inference_mode()
def collect_generation_panel(loaded: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    panel = [row for row in rows if row["behavior_generation_panel"]]
    batch_size = int(os.environ.get("PHASE421_GENERATION_BATCH", GENERATION_BATCH[loaded.key]))
    initial: list[dict[str, Any]] = []
    for start in range(0, len(panel), batch_size):
        initial.extend(generate_rows(loaded, panel[start : start + batch_size], INITIAL_HORIZON))
        completed = min(start + batch_size, len(panel))
        if completed % (batch_size * 12) == 0 or completed == len(panel):
            print(
                f"[Phase421:{loaded.key}:generation12] {completed}/{len(panel)}",
                flush=True,
            )
    extended_ids = {
        item["row"]["phase421_condition_id"]
        for item in initial
        if item["right_censored"]
    }
    extended_rows = [
        row for row in panel if row["phase421_condition_id"] in extended_ids
    ]
    extended_map: dict[str, dict[str, Any]] = {}
    for start in range(0, len(extended_rows), batch_size):
        batch = generate_rows(
            loaded,
            extended_rows[start : start + batch_size],
            EXTENDED_HORIZON,
        )
        extended_map.update(
            {item["row"]["phase421_condition_id"]: item for item in batch}
        )
    output = []
    for item in initial:
        row = item["row"]
        final = extended_map.get(row["phase421_condition_id"], item)
        text = final["generated_text"]
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase421-BehaviorGeneration",
                "created_at": now(),
                **compact(row),
                "prompt_sha256": sha256_text(final["prompt"]),
                "generated_token_ids": final["generated_token_ids"],
                "generated_text": text,
                "target_event_match": target_match(text, row["target_aliases"]),
                "opposite_event_match": target_match(text, [row["opposite_identity_target"]]),
                "exact_answer_match": exact_answer(text, row["target_aliases"]),
                "initial_right_censored": item["right_censored"],
                "behavior_horizon_extended": row["phase421_condition_id"] in extended_map,
                "behavior_horizon_used": final["horizon"],
                "right_censored": final["right_censored"],
                "emitted_stop": final["emitted_stop"],
                "behavior_generation_pass": True,
                "causal": False,
            }
        )
    return output


def run_model(model: str) -> dict[str, Any]:
    qualification = read_json(OUT / "phase421_denominator_qualification.json")
    if not qualification["valid"] or not qualification["behavior_collection_authorized"]:
        raise RuntimeError("Phase421 behavior denominator is not authorized")
    rows = [row for row in read_jsonl(REGISTERED) if row["model"] == model]
    if len(rows) != 10_368:
        raise RuntimeError(f"Expected 10368 Phase421 rows for {model}, found {len(rows)}")
    loaded = None
    started = time.monotonic()
    try:
        print(f"[Phase421] loading {model}; margin={len(rows)}", flush=True)
        loaded = load_probe_model(model)
        scores = collect_margin_scores(loaded, rows)
        generations = collect_generation_panel(loaded, rows)
        model_root = OUT / "models" / model
        write_jsonl(model_root / "phase421_behavior_margin_rows.jsonl", scores)
        write_jsonl(model_root / "phase421_behavior_generation_rows.jsonl", generations)
        valid = bool(
            len(scores) == 10_368
            and len(generations) == 576
            and all(row["behavior_margin_pass"] for row in scores)
            and all(row["behavior_generation_pass"] for row in generations)
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model,
            "margin_condition_count": len(scores),
            "margin_pass_count": sum(row["behavior_margin_pass"] for row in scores),
            "generation_panel_count": len(generations),
            "generation_target_event_count": sum(row["target_event_match"] for row in generations),
            "generation_exact_answer_count": sum(row["exact_answer_match"] for row in generations),
            "generation_initial_right_censored_count": sum(row["initial_right_censored"] for row in generations),
            "generation_final_right_censored_count": sum(row["right_censored"] for row in generations),
            "branch_common_prefix_positive_count": sum(
                row["branch_common_prefix_token_count"] > 0 for row in scores
            ),
            "all_behavior_rows_pass": valid,
            "margin_rows_sha256": hash_rows(scores),
            "generation_rows_sha256": hash_rows(generations),
            "elapsed_seconds": time.monotonic() - started,
            "vram_gb": vram_gb(),
            "physical_development_collection_authorized": False,
            "physical_holdout_collection_authorized": False,
            "causal_intervention_authorized": False,
            "single_neuron_scan_authorized": False,
        }
        write_json(model_root / "phase421_behavior_complete.json", summary)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()
    summary = run_model(args.model)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if not summary["all_behavior_rows_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
