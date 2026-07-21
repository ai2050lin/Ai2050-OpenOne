#!/usr/bin/env python3
"""Run one model on the frozen Phase577 natural-choice behavior gate."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
import phase577_natural_choice_protocol as protocol  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paths(model: str) -> dict[str, Path]:
    stem = protocol.OUT_DIR / f"phase577_{model}_behavior"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "registry": stem.with_name(stem.name + "_registry.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def candidate_position(text: str, candidate: str) -> int | None:
    match = re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", text, re.IGNORECASE)
    return match.start() if match else None


def classify(row: dict[str, Any], generated: str) -> dict[str, Any]:
    observed = sorted(
        (position, candidate)
        for candidate in row["all_candidates"]
        if (position := candidate_position(generated, candidate)) is not None
    )
    selected = observed[0][1] if observed else None
    normalized = " ".join(generated.strip().split())
    if selected == row["target"]:
        event = "target"
    elif selected == row["foil"]:
        event = "foil"
    else:
        event = "unrecoverable"
    return {
        "generated_text": generated,
        "normalized_generated": normalized,
        "selected_candidate": selected,
        "semantic_event": event,
        "semantic_correct": event == "target",
        "strict_sequence_correct": normalized.casefold() == row["target"].casefold(),
    }


def stable_expected(by_case_repeat: dict[tuple[str, str], dict[str, Any]], case_id: str) -> bool:
    first = by_case_repeat.get((case_id, "noop1"))
    second = by_case_repeat.get((case_id, "noop2"))
    return bool(
        first
        and second
        and first["semantic_correct"]
        and second["semantic_correct"]
        and first["selected_candidate"] == second["selected_candidate"]
        and first["normalized_generated"] == second["normalized_generated"]
    )


def generate_batch(
    loaded: Any,
    model: str,
    rows: list[dict[str, Any]],
    repeat: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    prompts = [render_chat(loaded.tokenizer, model, row["raw_prompt"]) for row in rows]
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    width = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = loaded.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    output = []
    for index, row in enumerate(rows):
        text = loaded.tokenizer.decode(generated[index, width:], skip_special_tokens=True)
        output.append({
            **row,
            **classify(row, text),
            "model": model,
            "execution_repeat": repeat,
            "observer_only": True,
            "causal": False,
        })
    del encoded, generated
    return output


def run(model: str, max_new_tokens: int, restart: bool) -> Path:
    output = paths(model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    frozen = read_json(protocol.PROTOCOL_PATH)
    audit = read_json(protocol.AUDIT_PATH)
    if not audit["valid"] or frozen["open_cases_sha256"] != sha256_file(
        protocol.OPEN_CASES_PATH
    ):
        raise RuntimeError("Phase577 protocol drift")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase577 behavior requires CUDA")
    cases = [
        row for row in iter_jsonl(protocol.OPEN_CASES_PATH)
        if row["split"] in protocol.STRUCTURE_SPLITS
    ]
    write_json(output["contract"], {
        "schema_version": "phase577_behavior_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": model,
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "splits_read": list(protocol.STRUCTURE_SPLITS),
        "causal_splits_read": False,
        "sealed_split_read": False,
        "fixed_batch_size": protocol.FIXED_BATCH_SIZE,
        "noop_repeats": list(protocol.NOOP_REPEATS),
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
    })
    loaded = None
    output_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase577 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "left"
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase577 requires BF16, got {dtype}")
        for split in protocol.STRUCTURE_SPLITS:
            split_rows = sorted(
                [row for row in cases if row["split"] == split],
                key=lambda row: row["case_id"],
            )
            for repeat in protocol.NOOP_REPEATS:
                for start in range(0, len(split_rows), protocol.FIXED_BATCH_SIZE):
                    output_rows.extend(generate_batch(
                        loaded,
                        model,
                        split_rows[start:start + protocol.FIXED_BATCH_SIZE],
                        repeat,
                        max_new_tokens,
                    ))
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase577 {split}/{repeat} "
                    f"{len(split_rows)}/{len(split_rows)}",
                    flush=True,
                )
        by_case_repeat = {
            (row["case_id"], row["execution_repeat"]): row for row in output_rows
        }
        by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cases:
            by_world[row["world_id"]].append(row)
        stable_worlds = {
            world_id
            for world_id, rows in by_world.items()
            if len(rows) == 2
            and all(stable_expected(by_case_repeat, row["case_id"]) for row in rows)
        }
        split_gates = {}
        selected_by_split = {}
        qualified_by_split = {}
        for split in protocol.STRUCTURE_SPLITS:
            split_cases = [row for row in cases if row["split"] == split]
            object_lookup = {
                row["object_id"]: row["is_fruit"] for row in split_cases
            }
            stable_counts: dict[tuple[str, str], int] = Counter(
                (by_world[world_id][0]["object_id"], by_world[world_id][0]["relation"])
                for world_id in stable_worlds
                if by_world[world_id][0]["split"] == split
            )
            qualified_objects = [
                object_id for object_id in sorted(object_lookup)
                if all(
                    stable_counts[(object_id, relation)]
                    >= protocol.MIN_STABLE_SURFACES_PER_RELATION
                    for relation in protocol.RELATIONS
                )
            ]
            qualified_set = set(qualified_objects)
            eligible_worlds = sorted(
                world_id for world_id in stable_worlds
                if by_world[world_id][0]["split"] == split
                and by_world[world_id][0]["object_id"] in qualified_set
            )
            banks: dict[tuple[bool, str, str], list[str]] = defaultdict(list)
            for world_id in eligible_worlds:
                row = by_world[world_id][0]
                banks[(row["is_fruit"], row["relation"], row["target"])].append(world_id)
            selected = []
            keys = sorted(banks, key=str)
            while keys and len(selected) < protocol.TRACE_WORLDS_PER_SPLIT:
                remaining = []
                for key in keys:
                    if len(selected) >= protocol.TRACE_WORLDS_PER_SPLIT:
                        break
                    if banks[key]:
                        selected.append(banks[key].pop(0))
                    if banks[key]:
                        remaining.append(key)
                keys = remaining
            fruit_count = sum(object_lookup[value] for value in qualified_objects)
            control_count = sum(not object_lookup[value] for value in qualified_objects)
            split_gates[split] = {
                "qualified_object_count": len(qualified_objects),
                "qualified_fruit_count": fruit_count,
                "qualified_control_count": control_count,
                "stable_option_swap_world_count": sum(
                    by_world[world_id][0]["split"] == split for world_id in stable_worlds
                ),
                "selected_trace_world_count": len(selected),
                "pass": bool(
                    fruit_count >= protocol.MIN_QUALIFIED_FRUITS_PER_SPLIT
                    and control_count >= protocol.MIN_QUALIFIED_CONTROLS_PER_SPLIT
                    and len(selected) >= protocol.TRACE_WORLDS_PER_SPLIT
                ),
            }
            qualified_by_split[split] = qualified_objects
            selected_by_split[split] = selected
        authorized = all(value["pass"] for value in split_gates.values())
        exact_repeat_mismatches = 0
        for case_id in sorted({row["case_id"] for row in output_rows}):
            first = by_case_repeat[(case_id, "noop1")]
            second = by_case_repeat[(case_id, "noop2")]
            exact_repeat_mismatches += first["normalized_generated"] != second["normalized_generated"]
        write_jsonl(output["rows"], output_rows)
        write_json(output["registry"], {
            "schema_version": "phase577_behavior_registry.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": model,
            "qualified_object_ids_by_split": qualified_by_split,
            "selected_trace_world_ids_by_split": selected_by_split,
            "natural_internal_trace_authorized": authorized,
            "causal_splits_read": False,
            "sealed_split_read": False,
        })
        summary = {
            "schema_version": "phase577_behavior_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "behavior_row_count": len(output_rows),
            "unique_case_count": len({row["case_id"] for row in output_rows}),
            "semantic_accuracy": sum(row["semantic_correct"] for row in output_rows) / len(output_rows),
            "strict_sequence_accuracy": sum(row["strict_sequence_correct"] for row in output_rows) / len(output_rows),
            "event_counts": dict(Counter(row["semantic_event"] for row in output_rows)),
            "exact_repeat_mismatch_count": exact_repeat_mismatches,
            "split_gates": split_gates,
            "natural_internal_trace_authorized": authorized,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
            "causal_splits_read": False,
            "sealed_split_read": False,
        }
        write_json(output["summary"], summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return output["summary"]
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.max_new_tokens, args.restart)


if __name__ == "__main__":
    main()
