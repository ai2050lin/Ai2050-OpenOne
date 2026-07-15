#!/usr/bin/env python3
"""Collect Phase431 execution identity, behavior and compact position-time traces."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase429_typed_route_collect import (  # noqa: E402
    event_position,
    natural_scores,
    sequence_pair_scores,
)
from phase429_typed_route_protocol import interface_payload, render_chat  # noqa: E402
from phase431_position_time_protocol import (  # noqa: E402
    ACTIVE_TAG,
    INTERFACE,
    LANGUAGE_MODEL,
    MODELS,
    NEUTRAL_TAG,
    OPEN_SPLITS,
    OUT,
    PHASE_ID as PROTOCOL_PHASE_ID,
    ROLES,
    ROUTE_MODES,
    SCHEMA_VERSION,
    SEALED_SPLIT,
    TRACE_SCHEMA_VERSION,
    freeze,
    read_json,
    route_tags,
    write_json,
    write_jsonl,
)


PHASE_ID = "Phase431-PositionTimeCollection"
BATCH_SIZE = {"qwen3": 8, "glm4": 2, "deepseek7b": 4}
PHYSICAL_BATCH_SIZE = 32
BEHAVIOR_CHECKPOINT_CONDITIONS = 256
PHYSICAL_CHECKPOINT_CONDITIONS = 16
GENERATION_STEPS = 4
QUERY_LINE = "Question: Which item is selected?"
INSTRUCTION_LINE = "Output exactly the selected item and then stop."


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase431 non-finite scalar: {value}")
    return round(float(value), 9)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") if path.suffix == ".gz" else opener(
        "r", encoding="utf-8"
    ) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def digest_ids(ids: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in ids).encode("ascii")
    ).hexdigest()


def module_attr(module: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise TypeError(f"Cannot locate any of {names} on {type(module).__name__}")


def component_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"No tensor component in {type(value).__name__}")


def padded_batch(
    sequences: list[list[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), width), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    pads: list[int] = []
    for index, sequence in enumerate(sequences):
        pad = width - len(sequence)
        pads.append(pad)
        input_ids[index, pad:] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, pad:] = 1
    return input_ids, attention_mask, pads


def prompt_ids(loaded: Any, row: dict[str, Any]) -> list[int]:
    ids = [
        int(value)
        for value in loaded.tokenizer(
            row["rendered_prompt"], add_special_tokens=False
        )["input_ids"]
    ]
    if len(ids) != int(row["prompt_token_count"]):
        raise RuntimeError(f"Prompt length changed: {row['condition_id']}")
    if digest_ids(ids) != row["prompt_token_ids_sha256"]:
        raise RuntimeError(f"Prompt ids changed: {row['condition_id']}")
    return ids


def group_rows(stage: str) -> list[dict[str, Any]]:
    if stage == "open":
        return read_jsonl(OUT / "phase431_groups_open.jsonl")
    if stage != "sealed":
        raise ValueError(stage)
    unlock_path = OUT / "phase431_open_gate.json"
    if not unlock_path.exists() or not read_json(unlock_path).get("sealed_unlock"):
        raise RuntimeError("Phase431 sealed groups are not authorized")
    return read_jsonl(OUT / "sealed" / "phase431_groups_sealed.jsonl")


def materialize_condition(
    group: dict[str, Any], role: str, route_mode: str, loaded: Any
) -> dict[str, Any]:
    before, after, source_role, query_role = route_tags(route_mode, role)
    if group["candidate"]:
        semantic_target = group["role_targets"][role]
        opposite_role = "b" if role == "a" else "a"
        semantic_opposite = group["role_targets"][opposite_role]
        query_semantic = semantic_opposite if route_mode == "conflict" else semantic_target
        normative = route_mode in {"source_only", "query_only", "consistent"}
    else:
        semantic_target = group["stable_target"]
        semantic_opposite = group["decoy"]
        query_semantic = semantic_target
        normative = True

    output = interface_payload(
        INTERFACE, group["first_item"], group["second_item"], semantic_target
    )
    before_line = f"Before-source selector slot: {before}."
    after_line = f"After-source selector slot: {after}."
    content = "\n".join(
        (
            "Follow this deterministic selector task.",
            group["task"],
            before_line,
            group["record"],
            after_line,
            QUERY_LINE,
            str(output["contract"]),
        )
    )
    rendered = render_chat(loaded.tokenizer, loaded.key, content)
    ids = [
        int(value)
        for value in loaded.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    ]
    target_ids = [
        int(value)
        for value in loaded.tokenizer(
            str(output["target"]), add_special_tokens=False
        )["input_ids"]
    ]
    opposite_ids = [
        int(value)
        for value in loaded.tokenizer(
            str(output["opposite"]), add_special_tokens=False
        )["input_ids"]
    ]
    source_1_ids = [
        int(value)
        for value in loaded.tokenizer(
            group["source_1"], add_special_tokens=False
        )["input_ids"]
    ]
    source_2_ids = [
        int(value)
        for value in loaded.tokenizer(
            group["source_2"], add_special_tokens=False
        )["input_ids"]
    ]
    if not target_ids or not opposite_ids or not source_1_ids or not source_2_ids:
        raise RuntimeError(f"Empty target ids for {group['semantic_group_id']}")
    condition_id = (
        f"{group['semantic_group_id']}__r{role}__route_{route_mode}__{loaded.key}"
    )
    return {
        **{
            key: value
            for key, value in group.items()
            if key not in {"task", "record", "history_text", "role_targets"}
        },
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "model": loaded.key,
        "condition_id": condition_id,
        "interface": INTERFACE,
        "role": role,
        "route_mode": route_mode,
        "source_role": source_role,
        "query_role": query_role,
        "source_route_target": semantic_target,
        "query_route_target": query_semantic,
        "content_prompt": content,
        "rendered_prompt": rendered,
        "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": digest_ids(ids),
        "prompt_token_count": len(ids),
        "semantic_target": semantic_target,
        "semantic_opposite": semantic_opposite,
        "target": str(output["target"]),
        "opposite_target": str(output["opposite"]),
        "target_sequence_token_ids": target_ids,
        "opposite_sequence_token_ids": opposite_ids,
        "source_1_first_token_id": source_1_ids[0],
        "source_2_first_token_id": source_2_ids[0],
        "natural_generation_max_new_tokens": int(output["max_new_tokens"]),
        "normative_target": normative,
        "descriptive_none_only": bool(group["candidate"] and route_mode == "none"),
        "descriptive_conflict_only": bool(
            group["candidate"] and route_mode == "conflict"
        ),
        "before_selector": before,
        "after_selector": after,
        "before_line": before_line,
        "after_line": after_line,
        "question_line": QUERY_LINE,
        "instruction_line": str(output["contract"]),
        "record_line": group["record"],
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
    }


def materialize_groups(loaded: Any, stage: str) -> list[dict[str, Any]]:
    rows = []
    for group in group_rows(stage):
        for role in ROLES:
            for route_mode in ROUTE_MODES:
                rows.append(materialize_condition(group, role, route_mode, loaded))
    rows.sort(key=lambda row: row["condition_id"])
    return rows


def actual_choice(text: str, row: dict[str, Any]) -> str:
    first_at = event_position(text, (row["source_1"],))
    second_at = event_position(text, (row["source_2"],))
    if first_at >= 0 and (second_at < 0 or first_at < second_at):
        return "source_1"
    if second_at >= 0 and (first_at < 0 or second_at < first_at):
        return "source_2"
    return "other"


def collect_behavior_batch(loaded: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired = sequence_pair_scores(loaded, rows)
    natural = natural_scores(loaded, rows)
    output = []
    for row, scores, generated in zip(rows, paired, natural):
        target_sum, target_mean, opposite_sum, opposite_mean = scores
        choice = actual_choice(str(generated["natural_text"]), row)
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": now(),
                **{
                    key: row.get(key)
                    for key in (
                        "model",
                        "condition_id",
                        "paired_group_id",
                        "semantic_group_id",
                        "block_id",
                        "family_id",
                        "mechanism_id",
                        "candidate",
                        "matched_control_block_id",
                        "split",
                        "contract_variant",
                        "role",
                        "route_mode",
                        "source_role",
                        "query_role",
                        "semantic_target",
                        "semantic_opposite",
                        "source_1",
                        "source_2",
                        "normative_target",
                        "descriptive_none_only",
                        "descriptive_conflict_only",
                    )
                },
                "target_sequence_logprob": clean(target_sum),
                "opposite_sequence_logprob": clean(opposite_sum),
                "teacher_sequence_logprob_margin": clean(target_sum - opposite_sum),
                "target_mean_token_logprob": clean(target_mean),
                "opposite_mean_token_logprob": clean(opposite_mean),
                "teacher_mean_token_logprob_margin": clean(target_mean - opposite_mean),
                "teacher_sequence_correct": target_sum > opposite_sum,
                "actual_choice": choice,
                "registered_source_choice": choice != "other",
                **generated,
                "physical": False,
                "observer": True,
                "predictive": False,
                "causal": False,
            }
        )
    return output


def collect_behavior(stage: str) -> dict[str, Any]:
    protocol = freeze()
    model = LANGUAGE_MODEL
    root = OUT / ("sealed" if stage == "sealed" else "open") / model / "behavior"
    complete_path = root / "phase431_behavior_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        rows = materialize_groups(loaded, stage)
        write_jsonl(root / "phase431_materialized_conditions.jsonl", rows)
        checkpoint_root = root / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        existing_parts = sorted(checkpoint_root.glob("phase431_behavior_part_*.jsonl"))
        existing = [row for path in existing_parts for row in read_jsonl(path)]
        completed_ids = {row["condition_id"] for row in existing}
        pending = [row for row in rows if row["condition_id"] not in completed_ids]
        part_number = len(existing_parts)
        buffer: list[dict[str, Any]] = []
        processed = len(completed_ids)
        print(
            f"[Phase431 behavior] {stage} {model}; conditions={len(rows)}; pending={len(pending)}",
            flush=True,
        )
        for start in range(0, len(pending), BATCH_SIZE[model]):
            batch = pending[start : start + BATCH_SIZE[model]]
            buffer.extend(collect_behavior_batch(loaded, batch))
            processed += len(batch)
            if (
                processed % BEHAVIOR_CHECKPOINT_CONDITIONS < len(batch)
                or processed == len(rows)
            ):
                write_jsonl(
                    checkpoint_root / f"phase431_behavior_part_{part_number:05d}.jsonl",
                    buffer,
                )
                part_number += 1
                buffer.clear()
            if processed == len(batch) or processed % 512 < len(batch):
                allocated, reserved = vram_gb()
                print(
                    f"[Phase431 behavior] {processed}/{len(rows)}; VRAM={allocated:.2f}/{reserved:.2f} GiB",
                    flush=True,
                )
        if buffer:
            write_jsonl(
                checkpoint_root / f"phase431_behavior_part_{part_number:05d}.jsonl",
                buffer,
            )
        collected = [
            row
            for path in sorted(checkpoint_root.glob("phase431_behavior_part_*.jsonl"))
            for row in read_jsonl(path)
        ]
        unique = {row["condition_id"]: row for row in collected}
        final_rows = [unique[key] for key in sorted(unique)]
        if len(final_rows) != len(rows):
            raise RuntimeError(f"Incomplete Phase431 behavior: {len(final_rows)} != {len(rows)}")
        write_jsonl(root / "phase431_behavior_rows.jsonl", final_rows)
        complete = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "stage": stage,
            "model": model,
            "condition_count": len(rows),
            "behavior_row_count": len(final_rows),
            "independent_group_count": len({row["semantic_group_id"] for row in rows}),
            "actual_choice_counts": dict(Counter(row["actual_choice"] for row in final_rows)),
            "elapsed_seconds": clean(time.monotonic() - started),
            "all_rows_complete": len(final_rows) == len(rows),
            "sealed_read": stage == "sealed",
            "protocol_open_condition_count": protocol["denominator_audit"]["open_condition_count"],
        }
        write_json(complete_path, complete)
        return complete
    finally:
        if loaded is not None:
            release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def final_norm_and_head(loaded: Any) -> tuple[Any, Any]:
    base = loaded.model.model if hasattr(loaded.model, "model") else loaded.model.transformer
    norm = module_attr(base, ("norm", "final_layernorm", "ln_f"))
    head = loaded.model.get_output_embeddings()
    if head is None:
        head = loaded.model.lm_head
    return norm, head


def identity_rows_for_model(loaded: Any) -> list[dict[str, Any]]:
    groups = [
        row
        for row in read_jsonl(OUT / "phase431_groups_open.jsonl")
        if row["split"] == "coordinate_calibration" and row["candidate"]
    ]
    return [materialize_condition(group, "a", "consistent", loaded) for group in groups]


@torch.inference_mode()
def collect_identity(model: str) -> dict[str, Any]:
    freeze()
    root = OUT / "models" / model / "identity"
    complete_path = root / "phase431_identity_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        rows = identity_rows_for_model(loaded)
        norm, head = final_norm_and_head(loaded)
        output_rows: list[dict[str, Any]] = []
        for start in range(0, len(rows), BATCH_SIZE[model]):
            batch = rows[start : start + BATCH_SIZE[model]]
            sequences = [prompt_ids(loaded, row) for row in batch]
            pad_id = int(loaded.tokenizer.pad_token_id)
            input_ids, attention_mask, _ = padded_batch(
                sequences, pad_id, loaded.input_device
            )
            captured: dict[str, torch.Tensor] = {}

            def norm_pre(_module: Any, args: tuple[Any, ...]) -> None:
                captured["pre"] = args[0].detach()

            def norm_post(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
                captured["post"] = component_tensor(output).detach()

            handles = [norm.register_forward_pre_hook(norm_pre), norm.register_forward_hook(norm_post)]
            try:
                with torch.inference_mode():
                    native = loaded.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
            finally:
                for handle in handles:
                    handle.remove()
            native_logits = native.logits[:, -1].float()
            reconstructed = head(captured["post"][:, -1]).float()
            reconstruction_max_abs = (
                native_logits - reconstructed
            ).abs().amax(dim=-1)
            native_top = native_logits.argmax(dim=-1)
            reconstruction_top = reconstructed.argmax(dim=-1)

            with torch.inference_mode():
                cached = loaded.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            cached_logits = cached.logits[:, -1].float()
            cached_top = cached_logits.argmax(dim=-1)
            cache_max_abs = (native_logits - cached_logits).abs().amax(dim=-1)

            next_ids = native_top[:, None]
            extended_mask = torch.cat(
                [attention_mask, torch.ones_like(next_ids)], dim=1
            )
            full_next_ids = torch.cat([input_ids, next_ids], dim=1)
            with torch.inference_mode():
                incremental = loaded.model(
                    input_ids=next_ids,
                    attention_mask=extended_mask,
                    past_key_values=cached.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                full_next = loaded.model(
                    input_ids=full_next_ids,
                    attention_mask=extended_mask,
                    use_cache=False,
                    return_dict=True,
                )
            incremental_logits = incremental.logits[:, -1].float()
            full_next_logits = full_next.logits[:, -1].float()
            batched_next_top_equal = incremental_logits.argmax(dim=-1).eq(
                full_next_logits.argmax(dim=-1)
            )
            batched_next_max_abs = (
                incremental_logits - full_next_logits
            ).abs().amax(dim=-1)

            single_logits: list[torch.Tensor] = []
            single_next_top_equal: list[bool] = []
            single_next_max_abs: list[float] = []
            for sequence in sequences:
                single_ids = torch.tensor(
                    [sequence], dtype=torch.long, device=loaded.input_device
                )
                with torch.inference_mode():
                    single = loaded.model(
                        input_ids=single_ids, use_cache=False, return_dict=True
                    )
                    single_cached = loaded.model(
                        input_ids=single_ids, use_cache=True, return_dict=True
                    )
                    single_next_id = single.logits[:, -1].argmax(dim=-1, keepdim=True)
                    single_mask = torch.ones(
                        (1, single_ids.shape[1] + 1),
                        dtype=torch.long,
                        device=loaded.input_device,
                    )
                    single_incremental = loaded.model(
                        input_ids=single_next_id,
                        attention_mask=single_mask,
                        past_key_values=single_cached.past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    single_full_next = loaded.model(
                        input_ids=torch.cat([single_ids, single_next_id], dim=1),
                        attention_mask=single_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                single_logits.append(single.logits[0, -1].float())
                single_incremental_logits = single_incremental.logits[0, -1].float()
                single_full_next_logits = single_full_next.logits[0, -1].float()
                single_next_top_equal.append(
                    bool(
                        single_incremental_logits.argmax()
                        .eq(single_full_next_logits.argmax())
                        .item()
                    )
                )
                single_next_max_abs.append(
                    clean(
                        (single_incremental_logits - single_full_next_logits)
                        .abs()
                        .amax()
                        .item()
                    )
                )
                del (
                    single,
                    single_cached,
                    single_incremental,
                    single_full_next,
                    single_ids,
                    single_next_id,
                    single_mask,
                    single_incremental_logits,
                    single_full_next_logits,
                )
            single_tensor = torch.stack(single_logits)
            batch_single_max_abs = (
                native_logits - single_tensor
            ).abs().amax(dim=-1)
            batch_single_top_equal = native_top.eq(single_tensor.argmax(dim=-1))

            for index, row in enumerate(batch):
                output_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "created_at": now(),
                        "model": model,
                        "condition_id": row["condition_id"],
                        "prompt_token_count": row["prompt_token_count"],
                        "prompt_terminal_index": row["prompt_token_count"] - 1,
                        "native_top_token_id": int(native_top[index].item()),
                        "reconstructed_top_token_id": int(
                            reconstruction_top[index].item()
                        ),
                        "terminal_reconstruction_top1_equal": bool(
                            native_top[index].eq(reconstruction_top[index]).item()
                        ),
                        "terminal_reconstruction_max_abs": clean(
                            reconstruction_max_abs[index].item()
                        ),
                        "cache_top1_equal": bool(
                            native_top[index].eq(cached_top[index]).item()
                        ),
                        "cache_max_abs": clean(cache_max_abs[index].item()),
                        "generation_step_one_cache_top1_equal": bool(
                            single_next_top_equal[index]
                        ),
                        "generation_step_one_cache_max_abs": clean(
                            single_next_max_abs[index]
                        ),
                        "batched_generation_step_one_cache_top1_equal": bool(
                            batched_next_top_equal[index].item()
                        ),
                        "batched_generation_step_one_cache_max_abs": clean(
                            batched_next_max_abs[index].item()
                        ),
                        "batch_single_top1_equal": bool(
                            batch_single_top_equal[index].item()
                        ),
                        "batch_single_max_abs": clean(
                            batch_single_max_abs[index].item()
                        ),
                        "finite": bool(
                            torch.isfinite(native_logits[index]).all().item()
                            and torch.isfinite(reconstructed[index]).all().item()
                        ),
                        "instrument_only": model != LANGUAGE_MODEL,
                        "causal": False,
                    }
                )
            del (
                native,
                cached,
                incremental,
                full_next,
                full_next_ids,
                input_ids,
                attention_mask,
                captured,
                native_logits,
                reconstructed,
                cached_logits,
                incremental_logits,
                full_next_logits,
                single_tensor,
            )
        write_jsonl(root / "phase431_identity_rows.jsonl", output_rows)
        all_pass = all(
            row["terminal_reconstruction_top1_equal"]
            and row["cache_top1_equal"]
            and row["generation_step_one_cache_top1_equal"]
            and row["batch_single_top1_equal"]
            and row["finite"]
            for row in output_rows
        )
        complete = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model,
            "condition_count": len(rows),
            "identity_row_count": len(output_rows),
            "execution_dtype": str(next(loaded.model.parameters()).dtype).removeprefix(
                "torch."
            ),
            "all_identity_top1_pass": all_pass,
            "max_terminal_reconstruction_error": max(
                row["terminal_reconstruction_max_abs"] for row in output_rows
            ),
            "max_cache_error": max(row["cache_max_abs"] for row in output_rows),
            "max_generation_step_one_cache_error": max(
                row["generation_step_one_cache_max_abs"] for row in output_rows
            ),
            "max_batched_generation_step_one_cache_error": max(
                row["batched_generation_step_one_cache_max_abs"]
                for row in output_rows
            ),
            "batched_generation_step_one_top1_pass_rate": clean(
                sum(
                    row["batched_generation_step_one_cache_top1_equal"]
                    for row in output_rows
                )
                / len(output_rows)
            ),
            "max_batch_single_error": max(
                row["batch_single_max_abs"] for row in output_rows
            ),
            "elapsed_seconds": clean(time.monotonic() - started),
            "all_rows_complete": len(output_rows) == len(rows),
            "language_interpretation_allowed": model == LANGUAGE_MODEL,
        }
        write_json(complete_path, complete)
        return complete
    finally:
        if loaded is not None:
            release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def token_positions(
    rendered: str,
    offsets: list[tuple[int, int]],
    value: str,
    start_at: int = 0,
) -> list[int]:
    left = rendered.find(value, start_at)
    if left < 0:
        raise RuntimeError(f"Registered span not found: {value!r}")
    right = left + len(value)
    return [
        index
        for index, (token_left, token_right) in enumerate(offsets)
        if token_right > token_left and token_left < right and token_right > left
    ]


def register_positions(fast_tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    rendered = row["rendered_prompt"]
    encoded = fast_tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    if digest_ids(ids) != row["prompt_token_ids_sha256"]:
        raise RuntimeError(f"Fast tokenizer disagreement: {row['condition_id']}")

    record_start = rendered.find(row["record_line"])
    before_start = rendered.find(row["before_line"])
    after_start = rendered.find(row["after_line"])
    question_start = rendered.find(row["question_line"])
    instruction_start = rendered.find(row["instruction_line"])
    if min(record_start, before_start, after_start, question_start, instruction_start) < 0:
        raise RuntimeError(f"Prompt role line missing: {row['condition_id']}")

    source_1 = token_positions(rendered, offsets, row["source_1"], record_start)
    source_2 = token_positions(rendered, offsets, row["source_2"], record_start)
    before_selector = token_positions(
        rendered, offsets, row["before_selector"], before_start
    )
    after_selector = token_positions(
        rendered, offsets, row["after_selector"], after_start
    )
    question = token_positions(rendered, offsets, row["question_line"], question_start)
    instruction = token_positions(
        rendered, offsets, row["instruction_line"], instruction_start
    )
    prompt_terminal = len(ids) - 1
    content_start = rendered.find(row["content_prompt"])
    content_end = content_start + len(row["content_prompt"])
    boundary = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left >= content_end
    ]
    assistant_boundary = boundary[-1] if boundary else prompt_terminal
    partitions = {
        "source_1": source_1,
        "source_2": source_2,
        "before_selector": before_selector,
        "after_selector": after_selector,
        "question": question,
        "instruction": instruction,
    }
    occupied = [value for values in partitions.values() for value in values]
    if len(occupied) != len(set(occupied)):
        raise RuntimeError(f"Overlapping source partition: {row['condition_id']}")
    return {
        "input_ids": ids,
        "position_roles": {
            "source_1_end": source_1[-1],
            "source_2_end": source_2[-1],
            "before_selector_end": before_selector[-1],
            "after_selector_end": after_selector[-1],
            "question_end": question[-1],
            "instruction_end": instruction[-1],
            "assistant_boundary": assistant_boundary,
            "prompt_terminal": prompt_terminal,
        },
        "source_partitions": partitions,
    }


def install_compact_hooks(
    layers: list[Any], captures: dict[tuple[str, int], torch.Tensor]
) -> list[Any]:
    handles = []
    for layer_index, layer in enumerate(layers):
        v_proj = module_attr(layer.self_attn, ("v_proj", "value"))

        def layer_pre(_module: Any, args: tuple[Any, ...], idx: int = layer_index) -> None:
            captures[("pre", idx)] = args[0].detach()

        def value_post(
            _module: Any, _args: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> None:
            captures[("value", idx)] = component_tensor(output).detach()

        def attention_post(
            _module: Any, _args: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> None:
            captures[("attention", idx)] = component_tensor(output).detach()
            if isinstance(output, (tuple, list)) and len(output) > 1 and torch.is_tensor(output[1]):
                captures[("probabilities", idx)] = output[1].detach()

        def mlp_post(
            _module: Any, _args: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> None:
            captures[("mlp", idx)] = component_tensor(output).detach()

        def layer_post(
            _module: Any, _args: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> None:
            captures[("post", idx)] = component_tensor(output).detach()

        handles.extend(
            [
                layer.register_forward_pre_hook(layer_pre),
                v_proj.register_forward_hook(value_post),
                layer.self_attn.register_forward_hook(attention_post),
                layer.mlp.register_forward_hook(mlp_post),
                layer.register_forward_hook(layer_post),
            ]
        )
    return handles


def vector_rms(vector: torch.Tensor) -> float:
    return clean(float(torch.sqrt(torch.mean(vector.float() ** 2).clamp_min(1e-20)).item()))


def vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= 1e-20:
        return 0.0
    return clean(float(torch.dot(left, right).item() / denominator.item()))


def relative_error(actual: torch.Tensor, reconstructed: torch.Tensor) -> float:
    error = torch.linalg.vector_norm(actual.float() - reconstructed.float())
    scale = torch.linalg.vector_norm(actual.float())
    return clean(float(error.item() / max(float(scale.item()), 1e-8)))


def random_projection_matrix(hidden_size: int, device: torch.device) -> torch.Tensor:
    rows = []
    for seed in (43101, 43102, 43103):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        signs = torch.randint(
            0, 2, (8, hidden_size), generator=generator, dtype=torch.int8
        ).float()
        rows.append((signs * 2.0 - 1.0) / math.sqrt(hidden_size))
    return torch.cat(rows, dim=0).to(device)


def prepare_physical_rows(stage: str) -> list[dict[str, Any]]:
    root = OUT / ("sealed" if stage == "sealed" else "open") / LANGUAGE_MODEL / "behavior"
    materialized = {
        row["condition_id"]: row
        for row in read_jsonl(root / "phase431_materialized_conditions.jsonl")
    }
    behavior = read_jsonl(root / "phase431_behavior_rows.jsonl")
    rows = []
    for result in behavior:
        row = dict(materialized[result["condition_id"]])
        row["actual_choice"] = result["actual_choice"]
        row["natural_generated_token_ids"] = result["natural_generated_token_ids"]
        row["natural_generated_token_count"] = result["natural_generated_token_count"]
        row["natural_target_first"] = result["natural_target_first"]
        row["natural_interface_valid"] = result["natural_interface_valid"]
        row["natural_stop"] = result["natural_stop"]
        rows.append(row)
    rows.sort(key=lambda row: row["condition_id"])
    return rows


def source_write_metrics(
    probabilities: torch.Tensor,
    repeated_value: torch.Tensor,
    output_blocks: torch.Tensor,
    positions: list[int],
    receiver: int,
    direction: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    if not positions:
        zero = torch.zeros(output_blocks.shape[0], device=output_blocks.device)
        return {
            "token_count": 0,
            "attention_mass_mean": 0.0,
            "write_norm": 0.0,
            "source_margin_write": 0.0,
        }, zero
    index = torch.tensor(positions, dtype=torch.long, device=probabilities.device)
    alpha = probabilities[:, receiver].index_select(-1, index).float()
    values = repeated_value.index_select(1, index).float()
    weighted = torch.einsum("hs,hsd->hd", alpha, values)
    head_writes = torch.einsum("hd,ohd->ho", weighted, output_blocks)
    write = head_writes.sum(dim=0)
    return {
        "token_count": len(positions),
        "attention_mass_mean": clean(float(alpha.sum(dim=-1).mean().item())),
        "write_norm": clean(float(torch.linalg.vector_norm(write).item())),
        "source_margin_write": clean(float(torch.dot(write.float(), direction.float()).item())),
    }, write


@torch.inference_mode()
def physical_batch_reference(
    loaded: Any,
    fast_tokenizer: Any,
    layers: list[Any],
    rows: list[dict[str, Any]],
    projection: torch.Tensor,
    selected_windows: set[tuple[int, str]] | None,
) -> list[dict[str, Any]]:
    registered = [register_positions(fast_tokenizer, row) for row in rows]
    slow_ids = [prompt_ids(loaded, row) for row in rows]
    if any(item["input_ids"] != ids for item, ids in zip(registered, slow_ids)):
        raise RuntimeError("Fast and execution tokenizer disagreement")
    extended_ids = []
    generation_counts = []
    for row, ids in zip(rows, slow_ids):
        generated = [int(value) for value in row["natural_generated_token_ids"]][
            :GENERATION_STEPS
        ]
        generation_counts.append(len(generated))
        extended_ids.append([*ids, *generated])
    pad_id = int(loaded.tokenizer.pad_token_id)
    input_ids, attention_mask, pads = padded_batch(
        extended_ids, pad_id, loaded.input_device
    )
    captures: dict[tuple[str, int], torch.Tensor] = {}
    handles = install_compact_hooks(layers, captures)
    try:
        with torch.inference_mode():
            result = loaded.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    del result

    final_norm, output_head = final_norm_and_head(loaded)
    output_weight = output_head.weight
    output_rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        expected = {"pre", "value", "attention", "probabilities", "mlp", "post"}
        actual = {name for name, index in captures if index == layer_index}
        if not expected.issubset(actual):
            raise RuntimeError(f"Missing Phase431 captures at layer {layer_index}: {actual}")
        probabilities = captures[("probabilities", layer_index)].float()
        value_raw = captures[("value", layer_index)]
        head_count = int(probabilities.shape[1])
        attention_module = layer.self_attn
        kv_heads = int(
            getattr(attention_module, "num_key_value_heads", 0)
            or getattr(attention_module.config, "num_key_value_heads", 0)
        )
        if kv_heads <= 0 or value_raw.shape[-1] % kv_heads != 0:
            raise RuntimeError(f"Cannot infer KV heads at layer {layer_index}")
        head_dim = int(value_raw.shape[-1] // kv_heads)
        values = value_raw.view(
            value_raw.shape[0], value_raw.shape[1], kv_heads, head_dim
        ).permute(0, 2, 1, 3)
        if kv_heads != head_count:
            if head_count % kv_heads != 0:
                raise RuntimeError("Query heads are not divisible by KV heads")
            values = values.repeat_interleave(head_count // kv_heads, dim=1)
        o_proj = module_attr(attention_module, ("o_proj", "dense"))
        output_blocks = o_proj.weight.float().view(
            o_proj.weight.shape[0], head_count, head_dim
        )
        bias = o_proj.bias.float() if o_proj.bias is not None else None

        for batch_index, row in enumerate(rows):
            pad = pads[batch_index]
            prompt_length = len(slow_ids[batch_index])
            position_roles = {
                role: pad + int(position)
                for role, position in registered[batch_index]["position_roles"].items()
            }
            position_roles["g0"] = position_roles["prompt_terminal"]
            for step in range(1, generation_counts[batch_index] + 1):
                position_roles[f"g{step}"] = pad + prompt_length + step - 1
            source_partitions = {
                role: [pad + int(position) for position in positions]
                for role, positions in registered[batch_index]["source_partitions"].items()
            }
            direction = (
                output_weight[int(row["source_1_first_token_id"])].float()
                - output_weight[int(row["source_2_first_token_id"])].float()
            )
            position_metrics: dict[str, Any] = {}
            for role, position in position_roles.items():
                if selected_windows is not None and (layer_index, role) not in selected_windows:
                    continue
                pre = captures[("pre", layer_index)][batch_index, position]
                attention = captures[("attention", layer_index)][batch_index, position]
                mlp = captures[("mlp", layer_index)][batch_index, position]
                post = captures[("post", layer_index)][batch_index, position]
                transition = post - pre
                reconstructed = attention + mlp
                normalized = final_norm(post)
                margin = torch.dot(normalized.float(), direction.float())
                payload = {
                    "absolute_token_index": int(position - pad),
                    "residual_pre_rms": vector_rms(pre),
                    "attention_write_rms": vector_rms(attention),
                    "mlp_write_rms": vector_rms(mlp),
                    "residual_post_rms": vector_rms(post),
                    "transition_rms": vector_rms(transition),
                    "block_reconstruction_relative_error": relative_error(
                        transition, reconstructed
                    ),
                    "attention_mlp_cosine": vector_cosine(attention, mlp),
                    "source_1_minus_source_2_margin": clean(margin.item()),
                    "attention_source_margin_write": clean(
                        torch.dot(attention.float(), direction.float()).item()
                    ),
                    "mlp_source_margin_write": clean(
                        torch.dot(mlp.float(), direction.float()).item()
                    ),
                }
                if role == "prompt_terminal":
                    payload["random_projection_sketch"] = [
                        clean(value)
                        for value in torch.mv(projection, post.float()).tolist()
                    ]
                position_metrics[role] = payload

            receiver_metrics: dict[str, Any] = {}
            receiver_roles = {
                role: position
                for role, position in position_roles.items()
                if role
                in {
                    "question_end",
                    "instruction_end",
                    "prompt_terminal",
                    "g1",
                    "g2",
                    "g3",
                    "g4",
                }
            }
            for role, receiver in receiver_roles.items():
                if selected_windows is not None and (layer_index, role) not in selected_windows:
                    continue
                valid_positions = set(range(pad, receiver + 1))
                partition_positions: dict[str, list[int]] = {}
                occupied: set[int] = set()
                for source_role, positions in source_partitions.items():
                    eligible = sorted(valid_positions.intersection(positions))
                    partition_positions[source_role] = eligible
                    occupied.update(eligible)
                partition_positions["other_positions"] = sorted(
                    valid_positions.difference(occupied)
                )
                role_payload: dict[str, Any] = {}
                replay = torch.zeros(
                    output_blocks.shape[0],
                    dtype=torch.float32,
                    device=output_blocks.device,
                )
                for source_role, positions in partition_positions.items():
                    metrics, write = source_write_metrics(
                        probabilities[batch_index],
                        values[batch_index],
                        output_blocks,
                        positions,
                        receiver,
                        direction,
                    )
                    role_payload[source_role] = metrics
                    replay = replay + write
                if bias is not None:
                    replay = replay + bias
                actual_attention = captures[("attention", layer_index)][
                    batch_index, receiver
                ]
                receiver_metrics[role] = {
                    "absolute_token_index": int(receiver - pad),
                    "source_partition": role_payload,
                    "attention_replay_relative_error": relative_error(
                        actual_attention, replay
                    ),
                    "partition_token_count": sum(
                        len(positions) for positions in partition_positions.values()
                    ),
                    "causal_prefix_token_count": receiver - pad + 1,
                }

            output_rows.append(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "created_at": now(),
                    "model": loaded.key,
                    "condition_id": row["condition_id"],
                    "paired_group_id": row["paired_group_id"],
                    "semantic_group_id": row["semantic_group_id"],
                    "split": row["split"],
                    "anonymous_context_id": hashlib.sha256(
                        f"{row['contract_variant']}|{row['interface']}|{row['candidate']}".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:16],
                    "block_id": row["block_id"],
                    "candidate": row["candidate"],
                    "role": row["role"],
                    "route_mode": row["route_mode"],
                    "normative_target": row["normative_target"],
                    "actual_choice": row["actual_choice"],
                    "layer": layer_index,
                    "relative_depth": clean(
                        layer_index / max(1, len(layers) - 1)
                    ),
                    "prompt_token_count": prompt_length,
                    "generation_token_count": generation_counts[batch_index],
                    "position_metrics": position_metrics,
                    "receiver_metrics": receiver_metrics,
                    "physical": True,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                    "pipeline_sealed": row["pipeline_sealed"],
                }
            )
        del probabilities, value_raw, values, output_blocks
    del input_ids, attention_mask, captures
    return output_rows


def clean_vector(values: torch.Tensor) -> list[float]:
    return [clean(value) for value in values.detach().float().cpu().tolist()]


@torch.inference_mode()
def physical_batch(
    loaded: Any,
    fast_tokenizer: Any,
    layers: list[Any],
    rows: list[dict[str, Any]],
    projection: torch.Tensor,
    selected_windows: set[tuple[int, str]] | None,
) -> list[dict[str, Any]]:
    """Collect the same compact ledger with one CUDA synchronization per metric."""
    registered = [register_positions(fast_tokenizer, row) for row in rows]
    slow_ids = [prompt_ids(loaded, row) for row in rows]
    if any(item["input_ids"] != ids for item, ids in zip(registered, slow_ids)):
        raise RuntimeError("Fast and execution tokenizer disagreement")
    extended_ids = []
    generation_counts = []
    for row, ids in zip(rows, slow_ids):
        generated = [int(value) for value in row["natural_generated_token_ids"]][
            :GENERATION_STEPS
        ]
        generation_counts.append(len(generated))
        extended_ids.append([*ids, *generated])
    pad_id = int(loaded.tokenizer.pad_token_id)
    input_ids, attention_mask, pads = padded_batch(
        extended_ids, pad_id, loaded.input_device
    )
    captures: dict[tuple[str, int], torch.Tensor] = {}
    handles = install_compact_hooks(layers, captures)
    try:
        loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    final_norm, output_head = final_norm_and_head(loaded)
    output_weight = output_head.weight
    directions = torch.stack(
        [
            output_weight[int(row["source_1_first_token_id"])].float()
            - output_weight[int(row["source_2_first_token_id"])].float()
            for row in rows
        ]
    )
    position_maps: list[dict[str, int]] = []
    partition_maps: list[dict[str, list[int]]] = []
    for batch_index, item in enumerate(registered):
        pad = pads[batch_index]
        prompt_length = len(slow_ids[batch_index])
        positions = {
            role: pad + int(position)
            for role, position in item["position_roles"].items()
        }
        positions["g0"] = positions["prompt_terminal"]
        for step in range(1, generation_counts[batch_index] + 1):
            positions[f"g{step}"] = pad + prompt_length + step - 1
        position_maps.append(positions)
        partition_maps.append(
            {
                role: [pad + int(position) for position in values]
                for role, values in item["source_partitions"].items()
            }
        )

    batch_size = len(rows)
    sequence_width = input_ids.shape[1]
    batch_axis = torch.arange(batch_size, device=loaded.input_device)
    sequence_axis = torch.arange(sequence_width, device=loaded.input_device)
    receiver_roles = {
        "question_end",
        "instruction_end",
        "prompt_terminal",
        "g1",
        "g2",
        "g3",
        "g4",
    }
    source_roles = (
        "source_1",
        "source_2",
        "before_selector",
        "after_selector",
        "question",
        "instruction",
    )
    output_rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        expected = {"pre", "value", "attention", "probabilities", "mlp", "post"}
        actual = {name for name, index in captures if index == layer_index}
        if not expected.issubset(actual):
            raise RuntimeError(f"Missing Phase431 captures at layer {layer_index}: {actual}")
        probabilities = captures[("probabilities", layer_index)].float()
        value_raw = captures[("value", layer_index)]
        head_count = int(probabilities.shape[1])
        attention_module = layer.self_attn
        kv_heads = int(
            getattr(attention_module, "num_key_value_heads", 0)
            or getattr(attention_module.config, "num_key_value_heads", 0)
        )
        if kv_heads <= 0 or value_raw.shape[-1] % kv_heads != 0:
            raise RuntimeError(f"Cannot infer KV heads at layer {layer_index}")
        head_dim = int(value_raw.shape[-1] // kv_heads)
        values = value_raw.view(
            value_raw.shape[0], value_raw.shape[1], kv_heads, head_dim
        ).permute(0, 2, 1, 3)
        if kv_heads != head_count:
            if head_count % kv_heads != 0:
                raise RuntimeError("Query heads are not divisible by KV heads")
            values = values.repeat_interleave(head_count // kv_heads, dim=1)
        o_proj = module_attr(attention_module, ("o_proj", "dense"))
        output_blocks = o_proj.weight.float().view(
            o_proj.weight.shape[0], head_count, head_dim
        )
        bias = o_proj.bias.float() if o_proj.bias is not None else None
        layer_positions: list[dict[str, Any]] = [{} for _ in rows]
        layer_receivers: list[dict[str, Any]] = [{} for _ in rows]

        all_roles = sorted({role for mapping in position_maps for role in mapping})
        for role in all_roles:
            if selected_windows is not None and (layer_index, role) not in selected_windows:
                continue
            indices = [index for index, mapping in enumerate(position_maps) if role in mapping]
            if not indices:
                continue
            index_tensor = torch.tensor(indices, dtype=torch.long, device=loaded.input_device)
            positions = torch.tensor(
                [position_maps[index][role] for index in indices],
                dtype=torch.long,
                device=loaded.input_device,
            )
            pre = captures[("pre", layer_index)][index_tensor, positions]
            attention = captures[("attention", layer_index)][index_tensor, positions]
            mlp = captures[("mlp", layer_index)][index_tensor, positions]
            post = captures[("post", layer_index)][index_tensor, positions]
            transition = post - pre
            reconstructed = attention + mlp
            normalized = final_norm(post)
            selected_directions = directions.index_select(0, index_tensor)
            attention_norm = torch.linalg.vector_norm(attention.float(), dim=-1)
            mlp_norm = torch.linalg.vector_norm(mlp.float(), dim=-1)
            cosine_denominator = attention_norm * mlp_norm
            cosine = (attention.float() * mlp.float()).sum(dim=-1) / cosine_denominator.clamp_min(1e-20)
            cosine = torch.where(cosine_denominator <= 1e-20, torch.zeros_like(cosine), cosine)
            metrics = {
                "residual_pre_rms": torch.sqrt(torch.mean(pre.float() ** 2, dim=-1).clamp_min(1e-20)),
                "attention_write_rms": torch.sqrt(torch.mean(attention.float() ** 2, dim=-1).clamp_min(1e-20)),
                "mlp_write_rms": torch.sqrt(torch.mean(mlp.float() ** 2, dim=-1).clamp_min(1e-20)),
                "residual_post_rms": torch.sqrt(torch.mean(post.float() ** 2, dim=-1).clamp_min(1e-20)),
                "transition_rms": torch.sqrt(torch.mean(transition.float() ** 2, dim=-1).clamp_min(1e-20)),
                "block_reconstruction_relative_error": torch.linalg.vector_norm(
                    transition.float() - reconstructed.float(), dim=-1
                )
                / torch.linalg.vector_norm(transition.float(), dim=-1).clamp_min(1e-8),
                "attention_mlp_cosine": cosine,
                "source_1_minus_source_2_margin": (
                    normalized.float() * selected_directions
                ).sum(dim=-1),
                "attention_source_margin_write": (
                    attention.float() * selected_directions
                ).sum(dim=-1),
                "mlp_source_margin_write": (mlp.float() * selected_directions).sum(dim=-1),
            }
            metric_names = tuple(metrics)
            metric_values = (
                torch.stack([metrics[key] for key in metric_names], dim=-1)
                .detach()
                .float()
                .cpu()
                .tolist()
            )
            position_values = positions.detach().cpu().tolist()
            sketches = None
            if role == "prompt_terminal":
                sketches = (post.float() @ projection.T).detach().cpu().tolist()
            for local_index, batch_index in enumerate(indices):
                payload = {
                    "absolute_token_index": int(position_values[local_index] - pads[batch_index]),
                    **{
                        key: clean(metric_values[local_index][metric_index])
                        for metric_index, key in enumerate(metric_names)
                    },
                }
                if sketches is not None:
                    payload["random_projection_sketch"] = [
                        clean(value) for value in sketches[local_index]
                    ]
                layer_positions[batch_index][role] = payload

        for role in sorted(receiver_roles):
            if selected_windows is not None and (layer_index, role) not in selected_windows:
                continue
            indices = [index for index, mapping in enumerate(position_maps) if role in mapping]
            if not indices:
                continue
            index_tensor = torch.tensor(indices, dtype=torch.long, device=loaded.input_device)
            local_axis = torch.arange(len(indices), device=loaded.input_device)
            receiver = torch.tensor(
                [position_maps[index][role] for index in indices],
                dtype=torch.long,
                device=loaded.input_device,
            )
            selected_probabilities = probabilities.index_select(0, index_tensor)
            receiver_probabilities = selected_probabilities[local_axis, :, receiver, :]
            selected_values = values.index_select(0, index_tensor).float()
            selected_directions = directions.index_select(0, index_tensor)
            selected_pads = torch.tensor(
                [pads[index] for index in indices],
                dtype=torch.long,
                device=loaded.input_device,
            )
            causal_mask = (
                (sequence_axis.unsqueeze(0) >= selected_pads.unsqueeze(1))
                & (sequence_axis.unsqueeze(0) <= receiver.unsqueeze(1))
            )
            masks: dict[str, torch.Tensor] = {}
            occupied = torch.zeros_like(causal_mask)
            receiver_values = receiver.detach().cpu().tolist()
            for source_role in source_roles:
                mask = torch.zeros_like(causal_mask)
                for local_index, batch_index in enumerate(indices):
                    positions = [
                        position
                        for position in partition_maps[batch_index][source_role]
                        if position <= int(receiver_values[local_index])
                    ]
                    if positions:
                        mask[local_index, positions] = True
                mask &= causal_mask
                masks[source_role] = mask
                occupied |= mask
            masks["other_positions"] = causal_mask & ~occupied
            replay = torch.zeros(
                (len(indices), output_blocks.shape[0]),
                dtype=torch.float32,
                device=loaded.input_device,
            )
            source_metric_names = (
                "token_count",
                "attention_mass_mean",
                "write_norm",
                "source_margin_write",
            )
            source_metric_tensors = []
            for source_role, mask in masks.items():
                alpha = receiver_probabilities * mask.unsqueeze(1)
                weighted = torch.einsum("nhs,nhsd->nhd", alpha, selected_values)
                head_writes = torch.einsum("nhd,ohd->nho", weighted, output_blocks)
                write = head_writes.sum(dim=1)
                replay += write
                source_metric_tensors.append(
                    torch.stack(
                        [
                            mask.sum(dim=-1).float(),
                            alpha.sum(dim=-1).mean(dim=-1),
                            torch.linalg.vector_norm(write, dim=-1),
                            (write * selected_directions).sum(dim=-1),
                        ],
                        dim=-1,
                    )
                )
            if bias is not None:
                replay += bias
            actual_attention = captures[("attention", layer_index)][
                index_tensor, receiver
            ].float()
            replay_error = (
                torch.linalg.vector_norm(actual_attention - replay, dim=-1)
                / torch.linalg.vector_norm(actual_attention, dim=-1).clamp_min(1e-8)
            )
            source_role_names = tuple(masks)
            source_payload_values = (
                torch.stack(source_metric_tensors, dim=1)
                .detach()
                .float()
                .cpu()
                .tolist()
            )
            receiver_summary = (
                torch.stack(
                    [replay_error, causal_mask.sum(dim=-1).float()], dim=-1
                )
                .detach()
                .float()
                .cpu()
                .tolist()
            )
            for local_index, batch_index in enumerate(indices):
                layer_receivers[batch_index][role] = {
                    "absolute_token_index": int(receiver_values[local_index] - pads[batch_index]),
                    "source_partition": {
                        source_role: {
                            key: (
                                int(source_payload_values[local_index][source_index][metric_index])
                                if key == "token_count"
                                else clean(
                                    source_payload_values[local_index][source_index][metric_index]
                                )
                            )
                            for metric_index, key in enumerate(source_metric_names)
                        }
                        for source_index, source_role in enumerate(source_role_names)
                    },
                    "attention_replay_relative_error": clean(receiver_summary[local_index][0]),
                    "partition_token_count": int(receiver_summary[local_index][1]),
                    "causal_prefix_token_count": int(
                        receiver_values[local_index] - pads[batch_index] + 1
                    ),
                }

        for batch_index, row in enumerate(rows):
            output_rows.append(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "created_at": now(),
                    "model": loaded.key,
                    "condition_id": row["condition_id"],
                    "paired_group_id": row["paired_group_id"],
                    "semantic_group_id": row["semantic_group_id"],
                    "split": row["split"],
                    "anonymous_context_id": hashlib.sha256(
                        f"{row['contract_variant']}|{row['interface']}|{row['candidate']}".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:16],
                    "block_id": row["block_id"],
                    "candidate": row["candidate"],
                    "role": row["role"],
                    "route_mode": row["route_mode"],
                    "normative_target": row["normative_target"],
                    "actual_choice": row["actual_choice"],
                    "layer": layer_index,
                    "relative_depth": clean(layer_index / max(1, len(layers) - 1)),
                    "prompt_token_count": len(slow_ids[batch_index]),
                    "generation_token_count": generation_counts[batch_index],
                    "position_metrics": layer_positions[batch_index],
                    "receiver_metrics": layer_receivers[batch_index],
                    "physical": True,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                    "pipeline_sealed": row["pipeline_sealed"],
                }
            )
        del probabilities, value_raw, values, output_blocks
    del input_ids, attention_mask, captures
    return output_rows


def collect_physical(stage: str) -> dict[str, Any]:
    freeze()
    if stage == "open":
        identity_gate = OUT / "phase431_identity_gate.json"
        if not identity_gate.exists() or not read_json(identity_gate).get(
            "language_model_identity_pass"
        ):
            raise RuntimeError("Phase431 Qwen3 identity did not authorize physical collection")
        behavior_gate = OUT / "phase431_open_behavior_gate.json"
        if not behavior_gate.exists() or not read_json(behavior_gate).get(
            "physical_collection_authorized"
        ):
            raise RuntimeError("Phase431 open behavior gate did not authorize physical collection")
        selected_windows = None
    else:
        open_gate = OUT / "phase431_open_gate.json"
        if not open_gate.exists() or not read_json(open_gate).get("sealed_unlock"):
            raise RuntimeError("Phase431 sealed physical collection is not authorized")
        freeze_payload = read_json(OUT / "phase431_blind_window_freeze.json")
        selected_windows = {
            (int(row["layer"]), str(row["position_role"]))
            for row in freeze_payload["windows"]
        }
    root = OUT / ("sealed" if stage == "sealed" else "open") / LANGUAGE_MODEL / "physical"
    complete_path = root / "phase431_physical_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    rows = prepare_physical_rows(stage)
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(LANGUAGE_MODEL)
        fast_tokenizer = AutoTokenizer.from_pretrained(
            str(loaded.spec.local_dir),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True,
        )
        layers = get_layers(loaded.model)
        if selected_windows is not None:
            # Keep blind upstream windows frozen; add only a terminal calibration anchor.
            selected_windows.add((len(layers) - 1, "prompt_terminal"))
        hidden_size = int(next(loaded.model.parameters()).shape[-1])
        if hidden_size < 128:
            hidden_size = int(loaded.model.config.hidden_size)
        projection = random_projection_matrix(hidden_size, loaded.input_device)
        checkpoint_root = root / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        existing_parts = sorted(checkpoint_root.glob("phase431_physical_part_*.jsonl.gz"))
        existing = [row for path in existing_parts for row in read_jsonl(path)]
        counts = Counter(row["condition_id"] for row in existing)
        expected_layers = (
            len({layer for layer, _ in selected_windows}) if selected_windows else len(layers)
        )
        completed_ids = {
            condition_id
            for condition_id, count in counts.items()
            if count == expected_layers
        }
        pending = [row for row in rows if row["condition_id"] not in completed_ids]
        part_number = len(existing_parts)
        buffer: list[dict[str, Any]] = []
        processed = len(completed_ids)
        print(
            f"[Phase431 physical] {stage}; conditions={len(rows)}; pending={len(pending)}; layers={len(layers)}",
            flush=True,
        )
        for start in range(0, len(pending), PHYSICAL_BATCH_SIZE):
            batch = pending[start : start + PHYSICAL_BATCH_SIZE]
            traced = physical_batch(
                loaded, fast_tokenizer, layers, batch, projection, selected_windows
            )
            if selected_windows is not None:
                selected_layers = {layer for layer, _ in selected_windows}
                traced = [row for row in traced if row["layer"] in selected_layers]
            buffer.extend(traced)
            processed += len(batch)
            if (
                processed % PHYSICAL_CHECKPOINT_CONDITIONS < len(batch)
                or processed == len(rows)
            ):
                write_jsonl_gz(
                    checkpoint_root / f"phase431_physical_part_{part_number:05d}.jsonl.gz",
                    buffer,
                )
                part_number += 1
                buffer.clear()
            if processed == len(batch) or processed % 128 < len(batch):
                allocated, reserved = vram_gb()
                print(
                    f"[Phase431 physical] {processed}/{len(rows)}; VRAM={allocated:.2f}/{reserved:.2f} GiB",
                    flush=True,
                )
        if buffer:
            write_jsonl_gz(
                checkpoint_root / f"phase431_physical_part_{part_number:05d}.jsonl.gz",
                buffer,
            )
        final_rows = [
            row
            for path in sorted(checkpoint_root.glob("phase431_physical_part_*.jsonl.gz"))
            for row in read_jsonl(path)
        ]
        unique = {
            (row["condition_id"], int(row["layer"])): row for row in final_rows
        }
        final_rows = [unique[key] for key in sorted(unique)]
        expected = len(rows) * expected_layers
        if len(final_rows) != expected:
            raise RuntimeError(f"Incomplete physical rows: {len(final_rows)} != {expected}")
        write_jsonl_gz(root / "phase431_compact_rows.jsonl.gz", final_rows)
        block_errors = [
            metric["block_reconstruction_relative_error"]
            for row in final_rows
            for metric in row["position_metrics"].values()
        ]
        attention_errors = [
            metric["attention_replay_relative_error"]
            for row in final_rows
            for metric in row["receiver_metrics"].values()
        ]
        complete = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "stage": stage,
            "model": LANGUAGE_MODEL,
            "condition_count": len(rows),
            "trace_row_count": len(final_rows),
            "layer_count": len(layers),
            "stored_layer_count": expected_layers,
            "block_reconstruction_relative_error_median": clean(
                statistics.median(block_errors)
            ),
            "attention_replay_relative_error_median": clean(
                statistics.median(attention_errors)
            ),
            "all_rows_complete": len(final_rows) == expected,
            "selected_window_only": selected_windows is not None,
            "sealed_read": stage == "sealed",
            "head_channel_neuron_scan": False,
            "intervention": False,
            "causal_tested": False,
            "elapsed_seconds": clean(time.monotonic() - started),
        }
        write_json(complete_path, complete)
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    identity = subparsers.add_parser("identity")
    identity.add_argument("--model", choices=MODELS, required=True)
    behavior = subparsers.add_parser("behavior")
    behavior.add_argument("--stage", choices=("open", "sealed"), default="open")
    physical = subparsers.add_parser("physical")
    physical.add_argument("--stage", choices=("open", "sealed"), default="open")
    args = parser.parse_args()
    if args.command == "identity":
        payload = collect_identity(args.model)
    elif args.command == "behavior":
        payload = collect_behavior(args.stage)
    else:
        payload = collect_physical(args.stage)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
