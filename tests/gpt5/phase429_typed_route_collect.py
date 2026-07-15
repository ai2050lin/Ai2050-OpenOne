#!/usr/bin/env python3
"""Collect Phase429 observer and typed route behavior without physical hooks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
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
from phase429_typed_route_protocol import (  # noqa: E402
    ACTIVE_TAG,
    BEHAVIOR_BLOCKS,
    MODELS,
    NEUTRAL_TAG,
    OPEN_BEHAVIOR_SPLITS,
    OUT,
    ROLES,
    ROUTE_MODES,
    SCHEMA_VERSION,
    interface_payload,
    render_chat,
    route_tags,
)


PHASE_ID = "Phase429-TypedRouteCollection"
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
        raise RuntimeError(f"Phase429 non-finite scalar: {value}")
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


def sequence_pair_scores(
    loaded: Any, rows: list[dict[str, Any]]
) -> list[tuple[float, float, float, float]]:
    prompts = [prompt_ids(loaded, row) for row in rows]
    targets = [[int(value) for value in row["target_sequence_token_ids"]] for row in rows]
    opposites = [[int(value) for value in row["opposite_sequence_token_ids"]] for row in rows]
    continuations = [*targets, *opposites]
    repeated_prompts = [*prompts, *prompts]
    sequences = [
        prompt + continuation
        for prompt, continuation in zip(repeated_prompts, continuations)
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
    values: list[tuple[float, float]] = []
    for batch_index, (prompt, continuation, pad) in enumerate(
        zip(repeated_prompts, continuations, pads)
    ):
        token_values = []
        start = pad + len(prompt)
        for offset, token_id in enumerate(continuation):
            logits = result.logits[batch_index, start + offset - 1].float()
            log_probability = logits[token_id] - torch.logsumexp(logits, dim=-1)
            token_values.append(float(log_probability.item()))
        total = clean(sum(token_values))
        values.append((total, clean(total / len(token_values))))
    size = len(rows)
    output = [
        (values[index][0], values[index][1], values[index + size][0], values[index + size][1])
        for index in range(size)
    ]
    del result, input_ids, attention_mask
    return output


def event_position(text: str, aliases: Iterable[str]) -> int:
    positions = []
    lowered = text.lower()
    for alias in aliases:
        value = str(alias).strip().lower()
        if not value:
            continue
        pattern = rf"(?<![a-z0-9_-]){re.escape(value)}(?![a-z0-9_-])"
        match = re.search(pattern, lowered)
        if match:
            positions.append(match.start())
    return min(positions) if positions else -1


def interface_parse(text: str, row: dict[str, Any]) -> dict[str, Any]:
    stripped = text.strip().strip("` \n\t")
    target = str(row["target"])
    opposite = str(row["opposite_target"])
    valid = False
    canonical = None
    if row["interface"] == "result_field":
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, dict) and set(decoded) == {"result"}:
                canonical = json.dumps(
                    {"result": str(decoded["result"])},
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                valid = canonical in {target, opposite}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    else:
        canonical = stripped.lower()
        valid = canonical in {target.lower(), opposite.lower()}
    return {
        "natural_interface_valid": valid,
        "natural_exact_target_contract": bool(valid and canonical.lower() == target.lower()),
        "natural_exact_opposite_contract": bool(valid and canonical.lower() == opposite.lower()),
    }


def parse_generation(
    text: str,
    generated_ids: list[int],
    row: dict[str, Any],
    eos_ids: set[int],
) -> dict[str, Any]:
    target_aliases = {str(row["target"]), str(row["semantic_target"])}
    opposite_aliases = {str(row["opposite_target"]), str(row["semantic_opposite"])}
    target_at = event_position(text, target_aliases)
    opposite_at = event_position(text, opposite_aliases)
    target_seen = target_at >= 0
    opposite_seen = opposite_at >= 0
    target_first = target_seen and (not opposite_seen or target_at < opposite_at)
    opposite_first = opposite_seen and (not target_seen or opposite_at < target_at)
    eos_seen = any(token_id in eos_ids for token_id in generated_ids)
    horizon = int(row["natural_generation_max_new_tokens"])
    boundary = bool(re.search(r"[.!?;\n}\]]", text)) or eos_seen
    stop = eos_seen or len(generated_ids) < horizon
    events = interface_parse(text, row)
    return {
        "natural_text": text,
        "natural_generated_token_ids": generated_ids,
        "natural_generated_token_count": len(generated_ids),
        "natural_target_seen": target_seen,
        "natural_opposite_seen": opposite_seen,
        "natural_target_first": target_first,
        "natural_opposite_first": opposite_first,
        "natural_event_coverage": target_seen or opposite_seen,
        "natural_other": not target_seen and not opposite_seen,
        "natural_revision": target_seen and opposite_seen,
        "natural_boundary": boundary,
        "natural_stop": stop,
        "natural_censoring": len(generated_ids) >= horizon and not eos_seen,
        **events,
    }


def natural_scores(loaded: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    horizons = {int(row["natural_generation_max_new_tokens"]) for row in rows}
    if len(horizons) != 1:
        raise RuntimeError(f"Mixed generation horizons in batch: {horizons}")
    horizon = next(iter(horizons))
    prompts = [prompt_ids(loaded, row) for row in rows]
    pad_id = int(loaded.tokenizer.pad_token_id)
    input_ids, attention_mask, _ = padded_batch(prompts, pad_id, loaded.input_device)
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
            max_new_tokens=horizon,
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


def materialize_behavior_condition(
    group: dict[str, Any], role: str, route_mode: str, interface: str, loaded: Any
) -> dict[str, Any]:
    before, after, source_role, query_role = route_tags(route_mode, role)
    if group["candidate"]:
        semantic_target = group["role_targets"][role]
        opposite_role = "b" if role == "a" else "a"
        semantic_opposite = group["role_targets"][opposite_role]
        query_semantic = (
            semantic_opposite if route_mode == "conflict" else semantic_target
        )
        normative = route_mode in {"source_only", "query_only", "consistent"}
    else:
        semantic_target = group["stable_target"]
        semantic_opposite = group["decoy"]
        query_semantic = semantic_target
        normative = True
    output = interface_payload(
        interface, group["first_item"], group["second_item"], semantic_target
    )
    before_line = f"Before-source selector slot: {before}."
    after_line = f"After-source selector slot: {after}."
    query_line = "Question: Which item is selected?"
    content = "\n".join(
        part
        for part in (
            "Follow this deterministic selector task.",
            group["task"],
            group["history_text"],
            before_line,
            group["record"],
            after_line,
            query_line,
            str(output["contract"]),
        )
        if part
    )
    rendered = render_chat(loaded.tokenizer, loaded.key, content)
    prompt_tokens = [
        int(value)
        for value in loaded.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    ]
    target_ids = [
        int(value)
        for value in loaded.tokenizer(str(output["target"]), add_special_tokens=False)[
            "input_ids"
        ]
    ]
    opposite_ids = [
        int(value)
        for value in loaded.tokenizer(str(output["opposite"]), add_special_tokens=False)[
            "input_ids"
        ]
    ]
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
        "stage_kind": "typed_route_behavior",
        "model": loaded.key,
        "condition_id": condition_id,
        "interface": interface,
        "role": role,
        "route_mode": route_mode,
        "source_role": source_role,
        "query_role": query_role,
        "source_route_target": semantic_target,
        "query_route_target": query_semantic,
        "content_prompt": content,
        "rendered_prompt": rendered,
        "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": hashlib.sha256(
            ",".join(str(value) for value in prompt_tokens).encode("ascii")
        ).hexdigest(),
        "prompt_token_count": len(prompt_tokens),
        "semantic_target": semantic_target,
        "semantic_opposite": semantic_opposite,
        "target": str(output["target"]),
        "opposite_target": str(output["opposite"]),
        "target_sequence_token_ids": target_ids,
        "opposite_sequence_token_ids": opposite_ids,
        "natural_generation_max_new_tokens": int(output["max_new_tokens"]),
        "normative_target": normative,
        "descriptive_none_only": bool(group["candidate"] and route_mode == "none"),
        "descriptive_conflict_only": bool(group["candidate"] and route_mode == "conflict"),
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
    }


def collect_batch(loaded: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired = sequence_pair_scores(loaded, rows)
    natural = natural_scores(loaded, rows)
    output = []
    for row, scores, natural_row in zip(rows, paired, natural):
        target_sum, target_mean, opposite_sum, opposite_mean = scores
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": now(),
                **{
                    key: row.get(key)
                    for key in (
                        "stage_kind",
                        "model",
                        "condition_id",
                        "semantic_group_id",
                        "block_id",
                        "family_id",
                        "mechanism_id",
                        "candidate",
                        "matched_control_block_id",
                        "split",
                        "interface",
                        "contract_variant",
                        "role",
                        "route_mode",
                        "source_role",
                        "query_role",
                        "source_route_target",
                        "query_route_target",
                        "semantic_target",
                        "semantic_opposite",
                        "target",
                        "opposite_target",
                        "normative_target",
                        "descriptive_none_only",
                        "descriptive_conflict_only",
                    )
                },
                "target_sequence_logprob": target_sum,
                "opposite_sequence_logprob": opposite_sum,
                "teacher_sequence_logprob_margin": clean(target_sum - opposite_sum),
                "target_mean_token_logprob": target_mean,
                "opposite_mean_token_logprob": opposite_mean,
                "teacher_mean_token_logprob_margin": clean(target_mean - opposite_mean),
                "teacher_sequence_correct": target_sum > opposite_sum,
                "target_sequence_token_count": len(row["target_sequence_token_ids"]),
                "opposite_sequence_token_count": len(row["opposite_sequence_token_ids"]),
                "natural_generation_max_new_tokens": row["natural_generation_max_new_tokens"],
                **natural_row,
                "physical": False,
                "observer": True,
                "predictive": False,
                "causal": False,
            }
        )
    return output


def stage_rows(model: str, stage: str, loaded: Any) -> list[dict[str, Any]]:
    if stage == "observer_instrument":
        return [
            row
            for row in read_jsonl(OUT / "phase429_observer_instrument_conditions.jsonl")
            if row["model"] == model
        ]
    if stage == "observer":
        return [
            row
            for row in read_jsonl(OUT / "phase429_observer_conditions.jsonl")
            if row["model"] == model
        ]
    selection_path = OUT / "phase429_interface_freeze.json"
    if not selection_path.exists():
        raise RuntimeError("Phase429 interface freeze does not exist")
    selection = read_json(selection_path)["models"][model]
    if not selection["behavior_authorized"]:
        raise RuntimeError(f"Phase429 behavior is closed for unqualified model {model}")
    interface = selection["selected_interface"]
    if stage == "behavior_instrument":
        groups = read_jsonl(OUT / "phase429_behavior_instrument_groups.jsonl")
    elif stage == "behavior":
        groups = read_jsonl(OUT / "phase429_behavior_groups_open.jsonl")
        groups = [row for row in groups if row["split"] in OPEN_BEHAVIOR_SPLITS]
    else:
        gate_path = OUT / "phase429_open_physical_gate.json"
        if not gate_path.exists() or not read_json(gate_path).get("sealed_unlock"):
            raise RuntimeError("Phase429 sealed behavior is not unlocked")
        authorized = {
            (row["model"], row["block_id"], row["contract_variant"])
            for row in read_json(gate_path)["sealed_authorized_candidates"]
        }
        block_ids = {
            block_id
            for gate_model, block_id, contract in authorized
            if gate_model == model
        }
        controls = {
            block["matched_control_block_id"]
            for block in BEHAVIOR_BLOCKS
            if block["block_id"] in block_ids
        }
        groups = read_jsonl(OUT / "sealed" / "phase429_behavior_groups_sealed.jsonl")
        groups = [
            row
            for row in groups
            if (model, row["block_id"], row["contract_variant"]) in authorized
            or row["block_id"] in controls
        ]
    return [
        materialize_behavior_condition(group, role, route, interface, loaded)
        for group in groups
        for role in ROLES
        for route in ROUTE_MODES
    ]


def run_model(model: str, stage: str) -> dict[str, Any]:
    protocol = read_json(OUT / "phase429_protocol.json")
    if not protocol["validation"]["valid"]:
        raise RuntimeError("Phase429 protocol is invalid")
    expected_hash = protocol["implementation_commitments"][Path(__file__).name]
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError("Phase429 collector changed after protocol freeze")
    model_root = OUT / "models" / model / stage
    output_path = model_root / "phase429_rows.jsonl"
    complete_path = model_root / "phase429_collection_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        complete = read_json(complete_path)
        print(json.dumps(complete, ensure_ascii=False, indent=2))
        return complete
    loaded = None
    started = time.monotonic()
    try:
        print(f"[Phase429] loading {model}; stage={stage}", flush=True)
        loaded = load_probe_model(model)
        actual_dtype = str(next(loaded.model.parameters()).dtype).removeprefix("torch.")
        expected_dtype = protocol["execution_dtype_by_model"][model]
        if actual_dtype != expected_dtype:
            raise RuntimeError(f"Execution dtype mismatch: {actual_dtype} != {expected_dtype}")
        conditions = stage_rows(model, stage, loaded)
        write_jsonl(model_root / "phase429_materialized_conditions.jsonl", conditions)
        checkpoint_root = model_root / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        existing_parts = sorted(checkpoint_root.glob("phase429_part_*.jsonl"))
        existing_rows = [row for path in existing_parts for row in read_jsonl(path)]
        completed_ids = {row["condition_id"] for row in existing_rows}
        pending = [row for row in conditions if row["condition_id"] not in completed_ids]
        part_number = len(existing_parts)
        buffer: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal part_number
            if not buffer:
                return
            write_jsonl(checkpoint_root / f"phase429_part_{part_number:05d}.jsonl", buffer)
            part_number += 1
            buffer.clear()

        print(
            f"[Phase429] {model} {stage}; conditions={len(conditions)}; pending={len(pending)}",
            flush=True,
        )
        processed = len(completed_ids)
        batch_size = BATCH_SIZE[model]
        for horizon in sorted({int(row["natural_generation_max_new_tokens"]) for row in pending}):
            horizon_rows = [
                row for row in pending if int(row["natural_generation_max_new_tokens"]) == horizon
            ]
            for start in range(0, len(horizon_rows), batch_size):
                batch = horizon_rows[start : start + batch_size]
                buffer.extend(collect_batch(loaded, batch))
                processed += len(batch)
                if len(buffer) >= CHECKPOINT_ROWS:
                    flush()
                if processed == len(batch) or processed % 512 < len(batch):
                    allocated, reserved = vram_gb()
                    print(
                        f"[Phase429] {model} {stage} {processed}/{len(conditions)}; "
                        f"horizon={horizon}; VRAM={allocated:.2f}/{reserved:.2f} GiB",
                        flush=True,
                    )
        flush()
        rows = [
            row
            for path in sorted(checkpoint_root.glob("phase429_part_*.jsonl"))
            for row in read_jsonl(path)
        ]
        unique = {row["condition_id"]: row for row in rows}
        rows = [unique[key] for key in sorted(unique)]
        if len(rows) != len(conditions):
            raise RuntimeError(f"Phase429 incomplete collection: {len(rows)} != {len(conditions)}")
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
        complete = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model,
            "stage": stage,
            "execution_dtype": actual_dtype,
            "condition_count": len(rows),
            "independent_group_count": len({row["semantic_group_id"] for row in rows}),
            "condition_counts_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
            "condition_counts_by_interface": dict(sorted(Counter(row["interface"] for row in rows).items())),
            "condition_counts_by_route": dict(sorted(Counter(row["route_mode"] for row in rows).items())),
            "finite_sequence_scores": finite,
            "natural_parser_complete": all("natural_interface_valid" in row for row in rows),
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
    parser.add_argument(
        "--stage",
        choices=("observer_instrument", "observer", "behavior_instrument", "behavior", "sealed"),
        required=True,
    )
    args = parser.parse_args()
    run_model(args.model, args.stage)


if __name__ == "__main__":
    main()
