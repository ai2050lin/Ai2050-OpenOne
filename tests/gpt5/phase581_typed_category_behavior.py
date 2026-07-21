#!/usr/bin/env python3
"""Run one CUDA model on the frozen Phase581 typed-category behavior gate."""

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
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
import phase580_open_category_behavior as base  # noqa: E402
import phase581_typed_category_protocol as protocol  # noqa: E402


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
    stem = protocol.OUT_DIR / f"phase581_{model}_typed_category_behavior"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "registry": stem.with_name(stem.name + "_registry.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def generate_batch(
    loaded: Any,
    model: str,
    rows: list[dict[str, Any]],
    repeat: str,
) -> list[dict[str, Any]]:
    prompts = [render_chat(loaded.tokenizer, model, row["raw_prompt"]) for row in rows]
    encoded = loaded.tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=False
    )
    width = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = loaded.model.generate(
            **encoded,
            max_new_tokens=protocol.MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    output = []
    for index, row in enumerate(rows):
        text = loaded.tokenizer.decode(
            generated[index, width:], skip_special_tokens=True
        )
        output.append(
            {
                **row,
                **base.classify(row, text),
                "model": model,
                "execution_repeat": repeat,
                "observer_only": True,
                "causal": False,
            }
        )
    del encoded, generated
    return output


def summarize_relation_split(
    cases: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    by_case_repeat: dict[tuple[str, str], dict[str, Any]],
    split: str,
    relation: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    unit_cases = [
        row for row in cases if row["split"] == split and row["relation"] == relation
    ]
    unit_outputs = [
        row
        for row in output_rows
        if row["split"] == split and row["relation"] == relation
    ]
    stable_ids = [
        row["case_id"]
        for row in unit_cases
        if base.stable_case(by_case_repeat, row["case_id"])
    ]
    stable_set = set(stable_ids)
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unit_cases:
        by_object[row["object_id"]].append(row)
    qualified_objects = [
        object_id
        for object_id, rows in sorted(by_object.items())
        if sum(row["case_id"] in stable_set for row in rows)
        >= protocol.MIN_STABLE_SURFACES_PER_OBJECT
    ]
    category_by_object = {
        object_id: rows[0]["target_category"]
        for object_id, rows in by_object.items()
    }
    qualified_by_category = dict(
        Counter(category_by_object[object_id] for object_id in qualified_objects)
    )
    semantic_accuracy = sum(
        row["semantic_correct"] for row in unit_outputs
    ) / len(unit_outputs)
    stable_case_rate = len(stable_ids) / len(unit_cases)
    repeat_pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in unit_outputs:
        repeat_pairs[row["case_id"]][row["execution_repeat"]] = row
    repeat_exact_rate = sum(
        set(values) == set(protocol.NOOP_REPEATS)
        and values["noop1"]["normalized_generated"]
        == values["noop2"]["normalized_generated"]
        for values in repeat_pairs.values()
    ) / len(unit_cases)
    by_category = {}
    for category in protocol.RELATION_CATEGORIES[relation]:
        category_rows = [
            row for row in unit_outputs if row["target_category"] == category
        ]
        by_category[category] = {
            "row_count": len(category_rows),
            "semantic_accuracy": sum(
                row["semantic_correct"] for row in category_rows
            ) / len(category_rows),
            "event_counts": dict(Counter(row["semantic_event"] for row in category_rows)),
        }
    pass_gate = bool(
        semantic_accuracy >= protocol.MIN_SEMANTIC_ACCURACY
        and stable_case_rate >= protocol.MIN_STABLE_CASE_RATE
        and repeat_exact_rate >= protocol.MIN_REPEAT_EXACT_RATE
        and all(
            qualified_by_category.get(category, 0) >= minimum
            for category, minimum in protocol.MIN_QUALIFIED_BY_RELATION_CATEGORY[
                relation
            ].items()
        )
    )
    return (
        {
            "case_count": len(unit_cases),
            "output_row_count": len(unit_outputs),
            "semantic_accuracy": semantic_accuracy,
            "stable_case_count": len(stable_ids),
            "stable_case_rate": stable_case_rate,
            "repeat_exact_rate": repeat_exact_rate,
            "qualified_object_count": len(qualified_objects),
            "qualified_object_count_by_category": qualified_by_category,
            "by_category": by_category,
            "semantic_event_counts": dict(
                Counter(row["semantic_event"] for row in unit_outputs)
            ),
            "pass": pass_gate,
        },
        qualified_objects,
        stable_ids,
    )


def run(model: str, restart: bool) -> Path:
    output = paths(model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    frozen = read_json(protocol.PROTOCOL_PATH)
    audit = read_json(protocol.AUDIT_PATH)
    if not audit["valid"] or frozen["open_cases_sha256"] != sha256_file(
        protocol.OPEN_CASES_PATH
    ):
        raise RuntimeError("Phase581 protocol drift")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase581 behavior requires CUDA")
    cases = list(iter_jsonl(protocol.OPEN_CASES_PATH))
    if any(row["sealed"] for row in cases):
        raise RuntimeError("Phase581 behavior received sealed rows")
    write_json(
        output["contract"],
        {
            "schema_version": "phase581_typed_category_behavior_contract.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": model,
            "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
            "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
            "splits_read": list(protocol.OPEN_SPLITS),
            "relations": list(protocol.RELATIONS),
            "sealed_split_read": False,
            "fixed_batch_size": protocol.FIXED_BATCH_SIZE,
            "noop_repeats": list(protocol.NOOP_REPEATS),
            "max_new_tokens": protocol.MAX_NEW_TOKENS,
            "do_sample": False,
            "torch_dtype_requested": "torch.bfloat16",
        },
    )
    loaded = None
    output_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase581 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "left"
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase581 requires BF16, got {dtype}")
        for split in protocol.OPEN_SPLITS:
            split_rows = sorted(
                [row for row in cases if row["split"] == split],
                key=lambda row: row["case_id"],
            )
            for repeat in protocol.NOOP_REPEATS:
                for start in range(0, len(split_rows), protocol.FIXED_BATCH_SIZE):
                    output_rows.extend(
                        generate_batch(
                            loaded,
                            model,
                            split_rows[start : start + protocol.FIXED_BATCH_SIZE],
                            repeat,
                        )
                    )
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase581 "
                    f"{split}/{repeat} {len(split_rows)}/{len(split_rows)}",
                    flush=True,
                )
        by_case_repeat = {
            (row["case_id"], row["execution_repeat"]): row for row in output_rows
        }
        unit_metrics = {}
        qualified_registry = {}
        stable_registry = {}
        authorized_relations = []
        for relation in protocol.RELATIONS:
            relation_passes = []
            for split in protocol.OPEN_SPLITS:
                metrics, qualified, stable_ids = summarize_relation_split(
                    cases, output_rows, by_case_repeat, split, relation
                )
                key = f"{split}:{relation}"
                unit_metrics[key] = metrics
                qualified_registry[key] = qualified
                stable_registry[key] = stable_ids
                relation_passes.append(metrics["pass"])
            if all(relation_passes):
                authorized_relations.append(relation)
        write_jsonl(output["rows"], output_rows)
        summary = {
            "schema_version": "phase581_typed_category_behavior_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "case_count": len(cases),
            "row_count": len(output_rows),
            "unit_metrics": unit_metrics,
            "internal_trace_authorized_relations": authorized_relations,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
            "sealed_split_read": False,
        }
        write_json(output["summary"], summary)
        write_json(
            output["registry"],
            {
                "schema_version": "phase581_typed_category_behavior_registry.v1",
                "phase_id": protocol.PHASE,
                "created_at": now(),
                "model": model,
                "qualified_objects_by_split_relation": qualified_registry,
                "stable_case_ids_by_split_relation": stable_registry,
                "internal_trace_authorized_relations": authorized_relations,
                "causal_intervention_authorized": False,
                "sealed_split_read": False,
            },
        )
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
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.restart)


if __name__ == "__main__":
    main()
