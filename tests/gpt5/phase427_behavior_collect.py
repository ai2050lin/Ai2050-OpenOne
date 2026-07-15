#!/usr/bin/env python3
"""Collect Phase427 teacher-forced and natural behavior without physical hooks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase427_dual_route_protocol import (  # noqa: E402
    MODELS,
    OUT,
    SCHEMA_VERSION,
)


PHASE_ID = "Phase427-DualRouteBehaviorCollection"
BATCH_SIZE = {"qwen3": 8, "glm4": 2, "deepseek7b": 4}
CHECKPOINT_ROWS = 256


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
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


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase427 non-finite scalar: {value}")
    return round(float(value), 10)


def prompt_ids(loaded: Any, row: dict[str, Any]) -> list[int]:
    ids = [
        int(value)
        for value in loaded.tokenizer(
            row["rendered_prompt"], add_special_tokens=False
        )["input_ids"]
    ]
    digest = hashlib.sha256(
        ",".join(str(value) for value in ids).encode("ascii")
    ).hexdigest()
    if len(ids) != int(row["prompt_token_count"]):
        raise RuntimeError(f"Prompt token count changed: {row['condition_id']}")
    if digest != row["prompt_token_ids_sha256"]:
        raise RuntimeError(f"Prompt token hash changed: {row['condition_id']}")
    return ids


def padded_batch(
    sequences: list[list[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    width = max(len(row) for row in sequences)
    input_ids = torch.full(
        (len(sequences), width), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    pads = []
    for index, sequence in enumerate(sequences):
        pad = width - len(sequence)
        pads.append(pad)
        input_ids[index, pad:] = torch.tensor(
            sequence, dtype=torch.long, device=device
        )
        attention_mask[index, pad:] = 1
    return input_ids, attention_mask, pads


def sequence_scores(
    loaded: Any, rows: list[dict[str, Any]], variant: str
) -> list[tuple[float, float]]:
    token_key = (
        "target_sequence_token_ids"
        if variant == "target"
        else "opposite_sequence_token_ids"
    )
    prompts = [prompt_ids(loaded, row) for row in rows]
    continuations = [[int(value) for value in row[token_key]] for row in rows]
    sequences = [
        prompt + continuation
        for prompt, continuation in zip(prompts, continuations)
    ]
    pad_id = int(loaded.tokenizer.pad_token_id)
    input_ids, attention_mask, pads = padded_batch(
        sequences, pad_id, loaded.input_device
    )
    with torch.inference_mode():
        result = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    logits = result.logits.float()
    output = []
    for batch_index, (prompt, continuation, pad) in enumerate(
        zip(prompts, continuations, pads)
    ):
        values = []
        start = pad + len(prompt)
        for offset, token_id in enumerate(continuation):
            log_probs = torch.log_softmax(
                logits[batch_index, start + offset - 1], dim=-1
            )
            values.append(float(log_probs[token_id].item()))
        output.append((clean(sum(values)), clean(statistics.fmean(values))))
    del result, logits, input_ids, attention_mask
    return output


def parse_generation(
    text: str,
    generated_ids: list[int],
    row: dict[str, Any],
    eos_ids: set[int],
) -> dict[str, Any]:
    lowered = text.strip().lower()
    target = str(row["target"]).lower()
    opposite = str(row["opposite_target"]).lower()
    target_at = lowered.find(target)
    opposite_at = lowered.find(opposite)
    target_seen = target_at >= 0
    opposite_seen = opposite_at >= 0
    target_first = target_seen and (not opposite_seen or target_at < opposite_at)
    opposite_first = opposite_seen and (not target_seen or opposite_at < target_at)
    eos_seen = any(token_id in eos_ids for token_id in generated_ids)
    max_new_tokens = int(row["natural_generation_max_new_tokens"])
    boundary = bool(re.search(r"[.!?;\n]", text)) or eos_seen
    stop = eos_seen or len(generated_ids) < max_new_tokens
    stripped = text.strip().strip("`\"' ")
    if row["interface"] == "direct":
        exact_contract = stripped.lower() == target
    else:
        try:
            decoded = json.loads(text.strip().strip("` \n"))
            exact_contract = str(decoded.get("result", "")).lower() == target
        except (json.JSONDecodeError, AttributeError):
            exact_contract = False
    return {
        "natural_text": text,
        "natural_generated_token_ids": generated_ids,
        "natural_generated_token_count": len(generated_ids),
        "natural_target_seen": target_seen,
        "natural_opposite_seen": opposite_seen,
        "natural_target_first": target_first,
        "natural_opposite_first": opposite_first,
        "natural_other": not target_seen and not opposite_seen,
        "natural_revision": target_seen and opposite_seen,
        "natural_boundary": boundary,
        "natural_stop": stop,
        "natural_censoring": len(generated_ids) >= max_new_tokens and not eos_seen,
        "natural_exact_contract": exact_contract,
    }


def natural_scores(loaded: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_values = {int(row["natural_generation_max_new_tokens"]) for row in rows}
    if len(max_values) != 1:
        raise RuntimeError(f"Mixed generation horizons in batch: {max_values}")
    max_new_tokens = next(iter(max_values))
    prompts = [prompt_ids(loaded, row) for row in rows]
    pad_id = int(loaded.tokenizer.pad_token_id)
    input_ids, attention_mask, _ = padded_batch(
        prompts, pad_id, loaded.input_device
    )
    eos_value = loaded.tokenizer.eos_token_id
    eos_ids = (
        {int(value) for value in eos_value}
        if isinstance(eos_value, list)
        else ({int(eos_value)} if eos_value is not None else set())
    )
    with torch.inference_mode():
        generated = loaded.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_value,
            use_cache=True,
        )
    prefix_width = int(input_ids.shape[1])
    output = []
    for index, row in enumerate(rows):
        new_ids = [int(value) for value in generated[index, prefix_width:].tolist()]
        while new_ids and new_ids[-1] == pad_id and pad_id not in eos_ids:
            new_ids.pop()
        text = loaded.tokenizer.decode(new_ids, skip_special_tokens=True)
        output.append(parse_generation(text, new_ids, row, eos_ids))
    del generated, input_ids, attention_mask
    return output


def collect_batch(loaded: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_scores = sequence_scores(loaded, rows, "target")
    opposite_scores = sequence_scores(loaded, rows, "opposite")
    natural = natural_scores(loaded, rows)
    output = []
    for row, target_score, opposite_score, natural_row in zip(
        rows, target_scores, opposite_scores, natural
    ):
        target_sum, target_mean = target_score
        opposite_sum, opposite_mean = opposite_score
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": now(),
                "model": loaded.key,
                "condition_id": row["condition_id"],
                "semantic_group_id": row["semantic_group_id"],
                "block_id": row["block_id"],
                "family_id": row["family_id"],
                "mechanism_id": row["mechanism_id"],
                "candidate": row["candidate"],
                "matched_control_block_id": row["matched_control_block_id"],
                "split": row["split"],
                "interface": row["interface"],
                "history": row["history"],
                "role": row["role"],
                "route_mode": row["route_mode"],
                "source_role": row["source_role"],
                "query_role": row["query_role"],
                "source_route_target": row["source_route_target"],
                "query_route_target": row["query_route_target"],
                "target": row["target"],
                "opposite_target": row["opposite_target"],
                "normative_target": row["normative_target"],
                "descriptive_conflict_only": row["descriptive_conflict_only"],
                "descriptive_none_only": row["descriptive_none_only"],
                "target_sequence_logprob": target_sum,
                "opposite_sequence_logprob": opposite_sum,
                "teacher_sequence_logprob_margin": clean(target_sum - opposite_sum),
                "target_mean_token_logprob": target_mean,
                "opposite_mean_token_logprob": opposite_mean,
                "teacher_mean_token_logprob_margin": clean(
                    target_mean - opposite_mean
                ),
                "teacher_sequence_correct": target_sum > opposite_sum,
                "target_sequence_token_count": len(
                    row["target_sequence_token_ids"]
                ),
                "natural_generation_max_new_tokens": row[
                    "natural_generation_max_new_tokens"
                ],
                **natural_row,
                "physical": False,
                "observer": True,
                "predictive": False,
                "causal": False,
            }
        )
    return output


def stage_contract(stage: str, protocol: dict[str, Any]) -> tuple[Path, set[str]]:
    if stage == "instrument":
        return OUT / "phase427_instrument_conditions.jsonl", {
            block["block_id"] for block in protocol["blocks"]
        }
    if stage == "open":
        return OUT / "phase427_registered_conditions_open.jsonl", {
            block["block_id"] for block in protocol["blocks"]
        }
    gate_path = OUT / "phase427_open_gate_freeze.json"
    if not gate_path.exists():
        raise RuntimeError("Phase427 open gate freeze does not exist")
    gate = read_json(gate_path)
    if not gate.get("sealed_behavior_unlock"):
        raise RuntimeError("Phase427 sealed behavior split is not unlocked")
    authorized = set(gate["sealed_behavior_unlock_blocks"])
    contracts = {block["block_id"]: block for block in protocol["blocks"]}
    authorized.update(contracts[block]["matched_control_block_id"] for block in tuple(authorized))
    return OUT / "sealed" / "phase427_registered_conditions_sealed.jsonl", authorized


def run_model(model: str, stage: str) -> dict[str, Any]:
    protocol = read_json(OUT / "phase427_protocol.json")
    if not protocol["validation"]["valid"]:
        raise RuntimeError("Phase427 protocol is invalid")
    expected_hash = protocol["implementation_commitments"][Path(__file__).name]
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError("Phase427 collector changed after protocol freeze")
    condition_path, authorized_blocks = stage_contract(stage, protocol)
    conditions = [
        row
        for row in read_jsonl(condition_path)
        if row["model"] == model and row["block_id"] in authorized_blocks
    ]
    if stage == "open":
        conditions = [row for row in conditions if row["split"] in protocol["splits"][:3]]
    elif stage == "sealed":
        conditions = [row for row in conditions if row["split"] == "sealed_behavior_holdout"]
    model_root = OUT / "models" / model / stage
    output_path = model_root / "phase427_behavior_rows.jsonl"
    complete_path = model_root / "phase427_collection_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        complete = read_json(complete_path)
        print(json.dumps(complete, ensure_ascii=False, indent=2))
        return complete
    expected_dtype = protocol["execution_dtype_by_model"][model]
    checkpoint_root = model_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_parts = sorted(checkpoint_root.glob("phase427_behavior_part_*.jsonl"))
    existing_rows = [row for path in existing_parts for row in read_jsonl(path)]
    completed_ids = {row["condition_id"] for row in existing_rows}
    pending = [row for row in conditions if row["condition_id"] not in completed_ids]
    loaded = None
    started = time.monotonic()
    part_number = len(existing_parts)
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal part_number
        if not buffer:
            return
        write_jsonl(
            checkpoint_root / f"phase427_behavior_part_{part_number:05d}.jsonl",
            buffer,
        )
        part_number += 1
        buffer.clear()

    try:
        print(
            f"[Phase427] loading {model}; stage={stage}; conditions={len(conditions)}; "
            f"pending={len(pending)}",
            flush=True,
        )
        loaded = load_probe_model(model)
        actual_dtype = str(next(loaded.model.parameters()).dtype).removeprefix("torch.")
        if actual_dtype != expected_dtype:
            raise RuntimeError(
                f"Execution dtype mismatch: {actual_dtype} != {expected_dtype}"
            )
        batch_size = BATCH_SIZE[model]
        processed = len(completed_ids)
        by_horizon: dict[int, list[dict[str, Any]]] = {}
        for horizon in sorted(
            {int(row["natural_generation_max_new_tokens"]) for row in pending}
        ):
            by_horizon[horizon] = [
                row
                for row in pending
                if int(row["natural_generation_max_new_tokens"]) == horizon
            ]
        for horizon, horizon_rows in by_horizon.items():
            for start in range(0, len(horizon_rows), batch_size):
                batch = horizon_rows[start : start + batch_size]
                buffer.extend(collect_batch(loaded, batch))
                processed += len(batch)
                if len(buffer) >= CHECKPOINT_ROWS:
                    flush()
                if processed == len(batch) or processed % 512 < len(batch):
                    allocated, reserved = vram_gb()
                    print(
                        f"[Phase427] {model} {stage} {processed}/{len(conditions)}; "
                        f"horizon={horizon}; VRAM={allocated:.2f}/{reserved:.2f} GiB",
                        flush=True,
                    )
        flush()
        all_parts = sorted(checkpoint_root.glob("phase427_behavior_part_*.jsonl"))
        rows = [row for path in all_parts for row in read_jsonl(path)]
        unique = {row["condition_id"]: row for row in rows}
        rows = [unique[key] for key in sorted(unique)]
        if len(rows) != len(conditions):
            raise RuntimeError(
                f"Phase427 incomplete collection: {len(rows)} != {len(conditions)}"
            )
        write_jsonl(output_path, rows)
        finite = all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "target_sequence_logprob",
                "opposite_sequence_logprob",
                "teacher_sequence_logprob_margin",
            )
        )
        split_counts = Counter(row["split"] for row in rows)
        route_counts = Counter(row["route_mode"] for row in rows)
        complete = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model,
            "stage": stage,
            "execution_dtype": actual_dtype,
            "condition_count": len(rows),
            "independent_group_count": len(
                {row["semantic_group_id"] for row in rows}
            ),
            "condition_counts_by_split": dict(sorted(split_counts.items())),
            "condition_counts_by_route": dict(sorted(route_counts.items())),
            "finite_sequence_scores": finite,
            "natural_parser_complete": all(
                all(
                    key in row
                    for key in (
                        "natural_target_first",
                        "natural_opposite_first",
                        "natural_revision",
                        "natural_boundary",
                        "natural_stop",
                        "natural_censoring",
                    )
                )
                for row in rows
            ),
            "numerical_retry_count": 0,
            "elapsed_seconds": clean(time.monotonic() - started),
            "all_rows_complete": finite and len(rows) == len(conditions),
            "physical_hooks_installed": False,
            "sealed_read": stage == "sealed",
            "causal_tested": False,
        }
        write_json(complete_path, complete)
        print(json.dumps(complete, ensure_ascii=False, indent=2))
        return complete
    finally:
        if loaded is not None:
            release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=("instrument", "open", "sealed"), required=True)
    args = parser.parse_args()
    run_model(args.model, args.stage)


if __name__ == "__main__":
    main()
