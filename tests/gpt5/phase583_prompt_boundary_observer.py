#!/usr/bin/env python3
"""Read category first-token logits from candidate-free prompt boundaries."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import statistics
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
import phase581_typed_category_protocol as source  # noqa: E402
import phase582_external_continuation_observer as p582  # noqa: E402
import phase583_prompt_boundary_protocol as protocol  # noqa: E402


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
    stem = protocol.OUT_DIR / f"phase583_{model}_prompt_boundary"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "registry": stem.with_name(stem.name + "_registry.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def label_token_ids(tokenizer: Any) -> dict[str, dict[str, Any]]:
    result = {}
    for label in source.CATEGORY_ALIASES:
        tokens = [
            int(value) for value in tokenizer.encode(label, add_special_tokens=False)
        ]
        if not tokens:
            raise RuntimeError(f"Phase583 empty label tokens: {label}")
        result[label] = {"all_token_ids": tokens, "first_token_id": tokens[0]}
    for relation, categories in source.RELATION_CATEGORIES.items():
        first_ids = {result[value]["first_token_id"] for value in categories}
        if len(first_ids) != len(categories):
            raise RuntimeError(f"Phase583 first-token collision: {relation}")
    return result


def observe_batch(
    loaded: Any,
    model: str,
    rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    repeat: str,
) -> list[dict[str, Any]]:
    prompts = [render_chat(loaded.tokenizer, model, row["raw_prompt"]) for row in rows]
    encoded = loaded.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    with torch.inference_mode():
        result = loaded.model(
            **encoded,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    boundary_logits = result.logits[:, -1, :].float()
    output = []
    for index, row in enumerate(rows):
        target = row["target_category"]
        foil = p582.foil_for(row)
        target_logit = float(
            boundary_logits[index, labels[target]["first_token_id"]].item()
        )
        foil_logit = float(
            boundary_logits[index, labels[foil]["first_token_id"]].item()
        )
        margin = target_logit - foil_logit
        if not all(math.isfinite(value) for value in (target_logit, foil_logit, margin)):
            raise RuntimeError("Phase583 non-finite boundary logit")
        output.append(
            {
                **row,
                "model": model,
                "execution_repeat": repeat,
                "target_observer_label": target,
                "foil_observer_label": foil,
                "target_first_token_id": labels[target]["first_token_id"],
                "foil_first_token_id": labels[foil]["first_token_id"],
                "target_boundary_logit": target_logit,
                "foil_boundary_logit": foil_logit,
                "target_minus_foil_margin": margin,
                "target_wins": margin > 0.0,
                "candidate_words_inserted_into_model_input": False,
                "teacher_forced_continuation_used": False,
                "observer_only": True,
                "causal": False,
            }
        )
    del result, boundary_logits, encoded
    return output


def summarize_unit(
    cases: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
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
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in unit_outputs:
        by_case[row["case_id"]][row["execution_repeat"]] = row
    stable_ids = []
    repeat_deltas = []
    for case in unit_cases:
        values = by_case[case["case_id"]]
        if set(values) != set(protocol.NOOP_REPEATS):
            raise RuntimeError("Phase583 incomplete repeat pair")
        first = values["forward1"]
        second = values["forward2"]
        delta = max(
            abs(first["target_boundary_logit"] - second["target_boundary_logit"]),
            abs(first["foil_boundary_logit"] - second["foil_boundary_logit"]),
        )
        repeat_deltas.append(delta)
        if (
            first["target_wins"]
            and second["target_wins"]
            and delta <= protocol.MAX_REPEAT_LOGIT_DELTA
        ):
            stable_ids.append(case["case_id"])
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
    win_rate = sum(row["target_wins"] for row in unit_outputs) / len(unit_outputs)
    mean_margin = statistics.fmean(
        row["target_minus_foil_margin"] for row in unit_outputs
    )
    by_category = {}
    for category in source.RELATION_CATEGORIES[relation]:
        category_rows = [
            row for row in unit_outputs if row["target_category"] == category
        ]
        by_category[category] = {
            "row_count": len(category_rows),
            "target_win_rate": sum(row["target_wins"] for row in category_rows)
            / len(category_rows),
            "mean_margin": statistics.fmean(
                row["target_minus_foil_margin"] for row in category_rows
            ),
        }
    maximum_repeat_delta = max(repeat_deltas)
    pass_gate = bool(
        win_rate >= protocol.MIN_TARGET_WIN_RATE
        and mean_margin >= protocol.MIN_MEAN_MARGIN
        and maximum_repeat_delta <= protocol.MAX_REPEAT_LOGIT_DELTA
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
            "target_win_rate": win_rate,
            "mean_margin": mean_margin,
            "minimum_margin": min(
                row["target_minus_foil_margin"] for row in unit_outputs
            ),
            "maximum_margin": max(
                row["target_minus_foil_margin"] for row in unit_outputs
            ),
            "maximum_repeat_logit_delta": maximum_repeat_delta,
            "stable_case_count": len(stable_ids),
            "stable_case_rate": len(stable_ids) / len(unit_cases),
            "qualified_object_count": len(qualified_objects),
            "qualified_object_count_by_category": qualified_by_category,
            "by_category": by_category,
            "pass": pass_gate,
        },
        qualified_objects,
        stable_ids,
    )


def run(model: str, restart: bool) -> Path:
    frozen = read_json(protocol.PROTOCOL_PATH)
    if frozen["source_cases_sha256"] != sha256_file(source.OPEN_CASES_PATH):
        raise RuntimeError("Phase583 source hash drift")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase583 observer requires CUDA")
    output = paths(model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    cases = list(iter_jsonl(source.OPEN_CASES_PATH))
    if any(row["sealed"] or row["answer_word_present_in_raw_prompt"] for row in cases):
        raise RuntimeError("Phase583 source violates candidate-free open contract")
    write_json(
        output["contract"],
        {
            "schema_version": "phase583_prompt_boundary_contract.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": model,
            "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
            "source_cases_sha256": sha256_file(source.OPEN_CASES_PATH),
            "splits_read": list(protocol.OPEN_SPLITS),
            "sealed_split_read": False,
            "candidate_words_inserted_into_model_input": False,
            "teacher_forced_continuation_used": False,
            "fixed_batch_size": protocol.FIXED_BATCH_SIZE,
            "noop_repeats": list(protocol.NOOP_REPEATS),
            "torch_dtype_requested": "torch.bfloat16",
        },
    )
    loaded = None
    output_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase583 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "left"
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase583 requires BF16, got {dtype}")
        labels = label_token_ids(loaded.tokenizer)
        for split in protocol.OPEN_SPLITS:
            split_rows = sorted(
                [row for row in cases if row["split"] == split],
                key=lambda row: row["case_id"],
            )
            for repeat in protocol.NOOP_REPEATS:
                for start in range(0, len(split_rows), protocol.FIXED_BATCH_SIZE):
                    output_rows.extend(
                        observe_batch(
                            loaded,
                            model,
                            split_rows[start : start + protocol.FIXED_BATCH_SIZE],
                            labels,
                            repeat,
                        )
                    )
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase583 "
                    f"{split}/{repeat} {len(split_rows)}/{len(split_rows)}",
                    flush=True,
                )
        unit_metrics = {}
        qualified_registry = {}
        stable_registry = {}
        authorized_relations = []
        for relation in protocol.RELATIONS:
            relation_passes = []
            for split in protocol.OPEN_SPLITS:
                metrics, qualified, stable_ids = summarize_unit(
                    cases, output_rows, split, relation
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
            "schema_version": "phase583_prompt_boundary_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "case_count": len(cases),
            "row_count": len(output_rows),
            "label_token_ids": labels,
            "unit_metrics": unit_metrics,
            "prompt_trace_authorized_relations": authorized_relations,
            "natural_generation_qualified": False,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
            "sealed_split_read": False,
        }
        write_json(output["summary"], summary)
        write_json(
            output["registry"],
            {
                "schema_version": "phase583_prompt_boundary_registry.v1",
                "phase_id": protocol.PHASE,
                "created_at": now(),
                "model": model,
                "qualified_objects_by_split_relation": qualified_registry,
                "stable_case_ids_by_split_relation": stable_registry,
                "prompt_trace_authorized_relations": authorized_relations,
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
