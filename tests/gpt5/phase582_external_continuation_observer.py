#!/usr/bin/env python3
"""Score external category continuations for candidate-free Phase581 prompts."""

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
import phase582_external_continuation_protocol as protocol  # noqa: E402


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
    stem = protocol.OUT_DIR / f"phase582_{model}_external_continuation"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "registry": stem.with_name(stem.name + "_registry.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def foil_for(row: dict[str, Any]) -> str:
    categories = source.RELATION_CATEGORIES[row["relation"]]
    values = [value for value in categories if value != row["target_category"]]
    if len(values) != 1:
        raise RuntimeError("Phase582 relation does not define one foil")
    return values[0]


def candidate_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    result = {
        category: [
            int(value)
            for value in tokenizer.encode(category, add_special_tokens=False)
        ]
        for category in source.CATEGORY_ALIASES
    }
    if any(not values for values in result.values()):
        raise RuntimeError("Phase582 empty continuation token sequence")
    return result


def score_batch(
    loaded: Any,
    model: str,
    rows: list[dict[str, Any]],
    candidates: dict[str, list[int]],
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
        target = row["target_category"]
        foil = foil_for(row)
        for role, category in (("target", target), ("foil", foil)):
            continuation = candidates[category]
            sequences.append(prompt + continuation)
            metadata.append(
                {
                    "row_index": row_index,
                    "role": role,
                    "category": category,
                    "prompt_length": len(prompt),
                    "continuation_length": len(continuation),
                }
            )
    pad_id = loaded.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = loaded.tokenizer.eos_token_id
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), width), int(pad_id), dtype=torch.long
    )
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
    scores: dict[int, dict[str, float]] = defaultdict(dict)
    token_sums: dict[int, dict[str, float]] = defaultdict(dict)
    for sequence_index, item in enumerate(metadata):
        prompt_length = int(item["prompt_length"])
        continuation_length = int(item["continuation_length"])
        first_target = pads[sequence_index] + prompt_length
        token_ids = input_ids[
            sequence_index, first_target : first_target + continuation_length
        ]
        prediction_positions = torch.arange(
            first_target - 1,
            first_target + continuation_length - 1,
            device=loaded.input_device,
        )
        token_logits = logits[sequence_index, prediction_positions]
        selected = token_logits.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        log_probs = selected - torch.logsumexp(token_logits, dim=1)
        row_index = int(item["row_index"])
        role = str(item["role"])
        scores[row_index][role] = float(log_probs.mean().item())
        token_sums[row_index][role] = float(log_probs.sum().item())
    output = []
    for row_index, row in enumerate(rows):
        target_score = scores[row_index]["target"]
        foil_score = scores[row_index]["foil"]
        margin = target_score - foil_score
        if not all(math.isfinite(value) for value in (target_score, foil_score, margin)):
            raise RuntimeError("Phase582 non-finite continuation score")
        output.append(
            {
                **row,
                "model": model,
                "execution_repeat": repeat,
                "target_continuation": row["target_category"],
                "foil_continuation": foil_for(row),
                "target_mean_logprob": target_score,
                "foil_mean_logprob": foil_score,
                "target_sum_logprob": token_sums[row_index]["target"],
                "foil_sum_logprob": token_sums[row_index]["foil"],
                "target_minus_foil_margin": margin,
                "target_wins": margin > 0.0,
                "candidate_words_inserted_into_prompt": False,
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
            raise RuntimeError("Phase582 incomplete repeat pair")
        first = values["score1"]
        second = values["score2"]
        delta = max(
            abs(first["target_mean_logprob"] - second["target_mean_logprob"]),
            abs(first["foil_mean_logprob"] - second["foil_mean_logprob"]),
        )
        repeat_deltas.append(delta)
        if (
            first["target_wins"]
            and second["target_wins"]
            and delta <= protocol.MAX_REPEAT_SCORE_DELTA
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
        and maximum_repeat_delta <= protocol.MAX_REPEAT_SCORE_DELTA
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
            "maximum_repeat_score_delta": maximum_repeat_delta,
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
        raise RuntimeError("Phase582 source hash drift")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase582 observer requires CUDA")
    output = paths(model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    cases = list(iter_jsonl(source.OPEN_CASES_PATH))
    if any(row["sealed"] or row["answer_word_present_in_raw_prompt"] for row in cases):
        raise RuntimeError("Phase582 source violates candidate-free open contract")
    write_json(
        output["contract"],
        {
            "schema_version": "phase582_external_continuation_contract.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": model,
            "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
            "source_cases_sha256": sha256_file(source.OPEN_CASES_PATH),
            "splits_read": list(protocol.OPEN_SPLITS),
            "sealed_split_read": False,
            "candidate_words_inserted_into_prompt": False,
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
            raise RuntimeError(f"Phase582 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "left"
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase582 requires BF16, got {dtype}")
        candidates = candidate_token_ids(loaded.tokenizer)
        for split in protocol.OPEN_SPLITS:
            split_rows = sorted(
                [row for row in cases if row["split"] == split],
                key=lambda row: row["case_id"],
            )
            for repeat in protocol.NOOP_REPEATS:
                for start in range(0, len(split_rows), protocol.FIXED_BATCH_SIZE):
                    output_rows.extend(
                        score_batch(
                            loaded,
                            model,
                            split_rows[start : start + protocol.FIXED_BATCH_SIZE],
                            candidates,
                            repeat,
                        )
                    )
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase582 "
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
            "schema_version": "phase582_external_continuation_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "case_count": len(cases),
            "row_count": len(output_rows),
            "candidate_token_ids": candidates,
            "unit_metrics": unit_metrics,
            "observer_trace_authorized_relations": authorized_relations,
            "natural_generation_qualified": False,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
            "sealed_split_read": False,
        }
        write_json(output["summary"], summary)
        write_json(
            output["registry"],
            {
                "schema_version": "phase582_external_continuation_registry.v1",
                "phase_id": protocol.PHASE,
                "created_at": now(),
                "model": model,
                "qualified_objects_by_split_relation": qualified_registry,
                "stable_case_ids_by_split_relation": stable_registry,
                "observer_trace_authorized_relations": authorized_relations,
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
