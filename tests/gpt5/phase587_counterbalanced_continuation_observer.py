#!/usr/bin/env python3
"""Score Phase587 counterbalanced continuations with one local CUDA model."""

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
import phase587_counterbalanced_continuation_protocol as protocol  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    stem = protocol.OUT_DIR / f"phase587_{model}_counterbalanced_continuation"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "registry": stem.with_name(stem.name + "_registry.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def score_batch(
    loaded: Any,
    model: str,
    rows: list[dict[str, Any]],
    repeat: str,
) -> list[dict[str, Any]]:
    prompt_ids = [
        [
            int(value)
            for value in loaded.tokenizer.encode(
                render_chat(loaded.tokenizer, model, row["raw_prompt"]),
                add_special_tokens=True,
            )
        ]
        for row in rows
    ]
    sequences = []
    metadata = []
    for row_index, (row, prompt) in enumerate(zip(rows, prompt_ids, strict=True)):
        for candidate_class, continuation in row["continuations"].items():
            token_ids = [
                int(value)
                for value in loaded.tokenizer.encode(
                    continuation, add_special_tokens=False
                )
            ]
            if token_ids != row["candidate_token_ids_by_model"][model][candidate_class]:
                raise RuntimeError("Phase587 continuation tokenization drift")
            sequences.append(prompt + token_ids)
            metadata.append(
                {
                    "row_index": row_index,
                    "candidate_class": candidate_class,
                    "prompt_length": len(prompt),
                    "continuation_length": len(token_ids),
                }
            )
    pad_id = loaded.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = loaded.tokenizer.eos_token_id
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    pads = []
    for index, sequence in enumerate(sequences):
        pad = width - len(sequence)
        pads.append(pad)
        input_ids[index, pad:] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[index, pad:] = 1
    input_ids = input_ids.to(loaded.input_device)
    attention_mask = attention_mask.to(loaded.input_device)
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
    mean_scores: dict[int, dict[str, float]] = defaultdict(dict)
    sum_scores: dict[int, dict[str, float]] = defaultdict(dict)
    for sequence_index, item in enumerate(metadata):
        prompt_length = int(item["prompt_length"])
        continuation_length = int(item["continuation_length"])
        first_target = pads[sequence_index] + prompt_length
        token_ids = input_ids[
            sequence_index, first_target : first_target + continuation_length
        ]
        positions = torch.arange(
            first_target - 1,
            first_target + continuation_length - 1,
            device=loaded.input_device,
        )
        token_logits = logits[sequence_index, positions]
        selected = token_logits.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        log_probs = selected - torch.logsumexp(token_logits, dim=1)
        row_index = int(item["row_index"])
        candidate_class = str(item["candidate_class"])
        mean_scores[row_index][candidate_class] = float(log_probs.mean().item())
        sum_scores[row_index][candidate_class] = float(log_probs.sum().item())
    output = []
    for row_index, row in enumerate(rows):
        scores = mean_scores[row_index]
        target_class = row["target_continuation_class"]
        target_score = scores[target_class]
        foil_class, foil_score = max(
            ((key, value) for key, value in scores.items() if key != target_class),
            key=lambda item: item[1],
        )
        margin = target_score - foil_score
        if not all(math.isfinite(value) for value in (*scores.values(), margin)):
            raise RuntimeError("Phase587 non-finite score")
        output.append(
            {
                **row,
                "model": model,
                "execution_repeat": repeat,
                "candidate_mean_logprobs": scores,
                "candidate_sum_logprobs": sum_scores[row_index],
                "strongest_foil_class": foil_class,
                "target_minus_strongest_foil_margin": margin,
                "target_wins": margin > 0.0,
                "candidate_continuations_inserted_into_model_input": False,
                "observer_only": True,
                "causal": False,
            }
        )
    del result, logits, input_ids, attention_mask
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
            raise RuntimeError("Phase587 incomplete repeat pair")
        first = values[protocol.NOOP_REPEATS[0]]
        second = values[protocol.NOOP_REPEATS[1]]
        candidate_classes = set(first["candidate_mean_logprobs"])
        delta = max(
            abs(
                first["candidate_mean_logprobs"][candidate]
                - second["candidate_mean_logprobs"][candidate]
            )
            for candidate in candidate_classes
        )
        repeat_deltas.append(delta)
        if first["target_wins"] and second["target_wins"] and delta <= protocol.MAX_REPEAT_SCORE_DELTA:
            stable_ids.append(case["case_id"])
    stable_set = set(stable_ids)
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unit_cases:
        by_object[row["object_id"]].append(row)
    qualified_objects = [
        object_id
        for object_id, object_rows in sorted(by_object.items())
        if sum(row["case_id"] in stable_set for row in object_rows)
        >= protocol.MIN_STABLE_SURFACES_PER_OBJECT
    ]
    group_by_object = {
        object_id: object_rows[0]["semantic_group"]
        for object_id, object_rows in by_object.items()
    }
    qualified_by_group = dict(
        Counter(group_by_object[object_id] for object_id in qualified_objects)
    )
    first_repeat_rows = [
        row for row in unit_outputs if row["execution_repeat"] == protocol.NOOP_REPEATS[0]
    ]
    by_group = {}
    for group in ("fruit", "near_food_plant", "tool", "vehicle"):
        group_rows = [row for row in first_repeat_rows if row["semantic_group"] == group]
        by_group[group] = {
            "case_count": len(group_rows),
            "target_win_rate": sum(row["target_wins"] for row in group_rows)
            / max(1, len(group_rows)),
            "mean_margin": statistics.fmean(
                row["target_minus_strongest_foil_margin"] for row in group_rows
            ),
        }
    target_win_rate = sum(row["target_wins"] for row in first_repeat_rows) / len(
        first_repeat_rows
    )
    minimums = protocol.MIN_QUALIFIED_BY_SPLIT_GROUP[split]
    max_repeat_delta = max(repeat_deltas)
    pass_gate = bool(
        target_win_rate >= protocol.MIN_TARGET_WIN_RATE
        and all(
            by_group[group]["target_win_rate"] >= protocol.MIN_GROUP_TARGET_WIN_RATE
            for group in by_group
        )
        and max_repeat_delta <= protocol.MAX_REPEAT_SCORE_DELTA
        and all(
            qualified_by_group.get(group, 0) >= minimum
            for group, minimum in minimums.items()
        )
    )
    return (
        {
            "case_count": len(unit_cases),
            "output_row_count": len(unit_outputs),
            "target_win_rate": target_win_rate,
            "mean_margin": statistics.fmean(
                row["target_minus_strongest_foil_margin"] for row in first_repeat_rows
            ),
            "minimum_margin": min(
                row["target_minus_strongest_foil_margin"] for row in first_repeat_rows
            ),
            "maximum_margin": max(
                row["target_minus_strongest_foil_margin"] for row in first_repeat_rows
            ),
            "maximum_repeat_score_delta": max_repeat_delta,
            "stable_case_count": len(stable_ids),
            "stable_case_rate": len(stable_ids) / len(unit_cases),
            "qualified_object_count": len(qualified_objects),
            "qualified_object_count_by_group": qualified_by_group,
            "minimum_qualified_object_count_by_group": minimums,
            "by_group": by_group,
            "strongest_foil_counts": dict(
                Counter(row["strongest_foil_class"] for row in first_repeat_rows)
            ),
            "pass": pass_gate,
        },
        qualified_objects,
        stable_ids,
    )


def run(model: str, restart: bool) -> Path:
    frozen = json.loads(protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
    if frozen["open_cases_sha256"] != sha256_file(protocol.OPEN_CASES_PATH):
        raise RuntimeError("Phase587 protocol drift")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase587 observer requires CUDA")
    output = paths(model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    cases = list(iter_jsonl(protocol.OPEN_CASES_PATH))
    if any(row["sealed"] for row in cases):
        raise RuntimeError("Phase587 observer received sealed rows")
    write_json(
        output["contract"],
        {
            "schema_version": "phase587_counterbalanced_contract.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": model,
            "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
            "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
            "splits_read": list(protocol.OPEN_SPLITS),
            "sealed_split_read": False,
            "candidate_continuations_inserted_into_model_input": False,
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
            raise RuntimeError(f"Phase587 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "left"
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase587 requires BF16, got {dtype}")
        for split in protocol.OPEN_SPLITS:
            split_rows = sorted(
                [row for row in cases if row["split"] == split], key=lambda row: row["case_id"]
            )
            for repeat in protocol.NOOP_REPEATS:
                for start in range(0, len(split_rows), protocol.FIXED_BATCH_SIZE):
                    output_rows.extend(
                        score_batch(
                            loaded,
                            model,
                            split_rows[start : start + protocol.FIXED_BATCH_SIZE],
                            repeat,
                        )
                    )
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase587 "
                    f"{split}/{repeat} {len(split_rows)}/{len(split_rows)}",
                    flush=True,
                )
        unit_metrics = {}
        qualified_registry = {}
        stable_registry = {}
        authorized_relations = []
        for relation in protocol.RELATIONS:
            relation_pass = True
            for split in protocol.OPEN_SPLITS:
                metrics, qualified, stable_ids = summarize_unit(
                    cases, output_rows, split, relation
                )
                key = f"{split}:{relation}"
                unit_metrics[key] = metrics
                qualified_registry[key] = qualified
                stable_registry[key] = stable_ids
                relation_pass = relation_pass and metrics["pass"]
            if relation_pass:
                authorized_relations.append(relation)
        write_jsonl(output["rows"], output_rows)
        summary = {
            "schema_version": "phase587_counterbalanced_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "case_count": len(cases),
            "row_count": len(output_rows),
            "unit_metrics": unit_metrics,
            "open_hidden_capture_authorized_relations": authorized_relations,
            "natural_generation_qualified": False,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
            "sealed_split_read": False,
        }
        write_json(output["summary"], summary)
        write_json(
            output["registry"],
            {
                "schema_version": "phase587_counterbalanced_registry.v1",
                "phase_id": protocol.PHASE,
                "created_at": now(),
                "model": model,
                "qualified_objects_by_unit": qualified_registry,
                "stable_case_ids_by_unit": stable_registry,
                "open_hidden_capture_authorized_relations": authorized_relations,
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
