#!/usr/bin/env python3
"""Run Phase578 world-level behavior on untouched Phase577 open rows."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
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
import phase577_natural_choice_behavior as behavior  # noqa: E402
import phase577_natural_choice_protocol as source  # noqa: E402
import phase578_choice_world_protocol as protocol  # noqa: E402


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
    stem = protocol.OUT_DIR / f"phase578_{model}_behavior"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "registry": stem.with_name(stem.name + "_registry.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def run(model: str, max_new_tokens: int, restart: bool) -> Path:
    output = paths(model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    frozen = read_json(protocol.PROTOCOL_PATH)
    audit = read_json(protocol.AUDIT_PATH)
    if not audit["valid"] or frozen["source_case_sha256"] != sha256_file(
        protocol.SOURCE_CASES_PATH
    ):
        raise RuntimeError("Phase578 protocol drift")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase578 behavior requires CUDA")
    cases = [
        row for row in iter_jsonl(protocol.SOURCE_CASES_PATH)
        if row["split"] in protocol.OPEN_SPLITS
    ]
    write_json(output["contract"], {
        "schema_version": "phase578_behavior_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": model,
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "source_case_sha256": sha256_file(protocol.SOURCE_CASES_PATH),
        "splits_read": list(protocol.OPEN_SPLITS),
        "natural_trace_executed": False,
        "causal_holdout_internal_state_read": False,
        "sealed_split_read": False,
        "torch_dtype_requested": "torch.bfloat16",
    })
    loaded = None
    output_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase578 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "left"
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase578 requires BF16, got {dtype}")
        for split in protocol.OPEN_SPLITS:
            split_rows = sorted(
                [row for row in cases if row["split"] == split],
                key=lambda row: row["case_id"],
            )
            for repeat in source.NOOP_REPEATS:
                for start in range(0, len(split_rows), source.FIXED_BATCH_SIZE):
                    output_rows.extend(behavior.generate_batch(
                        loaded,
                        model,
                        split_rows[start:start + source.FIXED_BATCH_SIZE],
                        repeat,
                        max_new_tokens,
                    ))
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase578 {split}/{repeat} "
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
            world_id for world_id, rows in by_world.items()
            if len(rows) == 2
            and all(behavior.stable_expected(by_case_repeat, row["case_id"]) for row in rows)
        }
        split_gates = {}
        natural_by_split = {}
        causal_by_split = {}
        selected_by_split = {}
        for split in protocol.OPEN_SPLITS:
            split_stable = sorted(
                world_id for world_id in stable_worlds
                if by_world[world_id][0]["split"] == split
            )
            relation_counts = Counter(by_world[world_id][0]["relation"] for world_id in split_stable)
            object_relations: dict[str, set[str]] = defaultdict(set)
            object_is_fruit = {}
            for world_id in split_stable:
                row = by_world[world_id][0]
                object_relations[row["object_id"]].add(row["relation"])
                object_is_fruit[row["object_id"]] = row["is_fruit"]
            diverse = [
                object_id for object_id, relations in object_relations.items()
                if relations == set(source.RELATIONS)
            ]
            fruit_diversity = sum(object_is_fruit[value] for value in diverse)
            control_diversity = sum(not object_is_fruit[value] for value in diverse)
            banks: dict[tuple[bool, str, str, str], list[str]] = defaultdict(list)
            for world_id in split_stable:
                row = by_world[world_id][0]
                banks[(row["is_fruit"], row["relation"], row["target"], row["surface_order"])].append(world_id)
            selected = []
            keys = sorted(banks, key=str)
            while keys and len(selected) < protocol.SELECTED_WORLDS_PER_SPLIT:
                remaining = []
                for key in keys:
                    if len(selected) >= protocol.SELECTED_WORLDS_PER_SPLIT:
                        break
                    if banks[key]:
                        selected.append(banks[key].pop(0))
                    if banks[key]:
                        remaining.append(key)
                keys = remaining
            natural = selected[0::2]
            causal = selected[1::2]
            passed = bool(
                len(split_stable) >= protocol.MIN_STABLE_WORLDS_PER_SPLIT
                and all(
                    relation_counts[relation] >= protocol.MIN_STABLE_WORLDS_PER_RELATION
                    for relation in source.RELATIONS
                )
                and fruit_diversity >= protocol.MIN_DIVERSE_FRUITS
                and control_diversity >= protocol.MIN_DIVERSE_CONTROLS
                and len(selected) == protocol.SELECTED_WORLDS_PER_SPLIT
                and len(natural) == protocol.NATURAL_TRACE_WORLDS_PER_SPLIT
                and len(causal) == protocol.CAUSAL_HOLDOUT_WORLDS_PER_SPLIT
            )
            split_gates[split] = {
                "stable_world_count": len(split_stable),
                "stable_world_rate": len(split_stable) / 224,
                "stable_world_count_by_relation": dict(relation_counts),
                "diverse_fruit_object_count": fruit_diversity,
                "diverse_control_object_count": control_diversity,
                "selected_world_count": len(selected),
                "natural_trace_world_count": len(natural),
                "causal_holdout_world_count": len(causal),
                "pass": passed,
            }
            selected_by_split[split] = selected
            natural_by_split[split] = natural
            causal_by_split[split] = causal
        authorized = all(value["pass"] for value in split_gates.values())
        write_jsonl(output["rows"], output_rows)
        write_json(output["registry"], {
            "schema_version": "phase578_behavior_registry.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": model,
            "selected_world_ids_by_split": selected_by_split,
            "natural_trace_world_ids_by_split": natural_by_split,
            "causal_holdout_world_ids_by_split": causal_by_split,
            "natural_trace_authorized": authorized,
            "causal_holdout_internal_state_read": False,
            "sealed_split_read": False,
        })
        summary = {
            "schema_version": "phase578_behavior_summary.v1",
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
            "split_gates": split_gates,
            "natural_trace_authorized": authorized,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
            "causal_holdout_internal_state_read": False,
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
