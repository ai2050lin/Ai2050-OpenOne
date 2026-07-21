#!/usr/bin/env python3
"""Run the prospective Phase589 food-attribute observer on one CUDA model."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
import phase589_food_attribute_protocol as protocol  # noqa: E402


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paths(model: str) -> dict[str, Path]:
    stem = protocol.OUT_DIR / f"phase589_{model}_food_attribute"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def pairwise_auc(positive: list[float], negative: list[float]) -> float:
    wins = 0.0
    for left in positive:
        for right in negative:
            if left > right:
                wins += 1.0
            elif left == right:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def score_batch(loaded: Any, model: str, rows: list[dict[str, Any]], repeat: str) -> list[dict[str, Any]]:
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
    sequences: list[list[int]] = []
    metadata: list[dict[str, Any]] = []
    for row_index, (row, prompt) in enumerate(zip(rows, prompt_ids, strict=True)):
        for variant, continuation in row["continuations"].items():
            token_ids = [
                int(value)
                for value in loaded.tokenizer.encode(continuation, add_special_tokens=False)
            ]
            if token_ids != row["candidate_token_ids_by_model"][model][variant]:
                raise RuntimeError("Phase589 continuation tokenization drift")
            sequences.append(prompt + token_ids)
            metadata.append(
                {
                    "row_index": row_index,
                    "variant": variant,
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
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
    logits = result.logits.float()
    mean_scores: dict[int, dict[str, float]] = defaultdict(dict)
    sum_scores: dict[int, dict[str, float]] = defaultdict(dict)
    for sequence_index, item in enumerate(metadata):
        prompt_length = int(item["prompt_length"])
        continuation_length = int(item["continuation_length"])
        first_target = pads[sequence_index] + prompt_length
        target_ids = input_ids[
            sequence_index, first_target : first_target + continuation_length
        ]
        positions = torch.arange(
            first_target - 1,
            first_target + continuation_length - 1,
            device=loaded.input_device,
        )
        token_logits = logits[sequence_index, positions]
        selected = token_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        log_probs = selected - torch.logsumexp(token_logits, dim=1)
        row_index = int(item["row_index"])
        variant = str(item["variant"])
        mean_scores[row_index][variant] = float(log_probs.mean().item())
        sum_scores[row_index][variant] = float(log_probs.sum().item())
    output = []
    for row_index, row in enumerate(rows):
        scores = mean_scores[row_index]
        if not all(math.isfinite(value) for value in scores.values()):
            raise RuntimeError("Phase589 non-finite score")
        output.append(
            {
                **row,
                "model": model,
                "execution_repeat": repeat,
                "candidate_mean_logprobs": scores,
                "candidate_sum_logprobs": sum_scores[row_index],
                "candidate_continuations_inserted_into_model_input": False,
                "observer_only": True,
                "causal": False,
            }
        )
    del result, logits, input_ids, attention_mask
    return output


def comparison_auc(surface_rows: list[dict[str, Any]], variant: str, positive: set[str], negative: set[str]) -> float:
    pos = [
        float(row["candidate_mean_logprobs"][variant])
        for row in surface_rows
        if row["semantic_group"] in positive
    ]
    neg = [
        float(row["candidate_mean_logprobs"][variant])
        for row in surface_rows
        if row["semantic_group"] in negative
    ]
    return pairwise_auc(pos, neg)


def summarize_split(cases: list[dict[str, Any]], outputs: list[dict[str, Any]], split: str) -> dict[str, Any]:
    split_cases = [row for row in cases if row["split"] == split]
    split_outputs = [row for row in outputs if row["split"] == split]
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in split_outputs:
        by_case[row["case_id"]][row["execution_repeat"]] = row
    repeat_deltas = []
    for case in split_cases:
        pair = by_case[case["case_id"]]
        if set(pair) != set(protocol.NOOP_REPEATS):
            raise RuntimeError("Phase589 incomplete repeat pair")
        repeat_deltas.extend(
            abs(
                pair["score1"]["candidate_mean_logprobs"][variant]
                - pair["score2"]["candidate_mean_logprobs"][variant]
            )
            for variant in protocol.FOOD_CONTINUATIONS
        )
    first = [row for row in split_outputs if row["execution_repeat"] == "score1"]
    surfaces = sorted({int(row["surface_id"]) for row in first})
    variant_metrics: dict[str, Any] = {}
    for variant in protocol.FOOD_CONTINUATIONS:
        by_surface: dict[str, Any] = {}
        for surface in surfaces:
            surface_rows = [row for row in first if int(row["surface_id"]) == surface]
            by_surface[str(surface)] = {
                "edible_vs_nonedible_auc": comparison_auc(
                    surface_rows,
                    variant,
                    {"fruit", "edible_nonfruit"},
                    {"nonfood_plant", "tool", "vehicle"},
                ),
                "edible_vs_nonfood_plant_auc": comparison_auc(
                    surface_rows,
                    variant,
                    {"fruit", "edible_nonfruit"},
                    {"nonfood_plant"},
                ),
                "fruit_vs_artifact_auc": comparison_auc(
                    surface_rows, variant, {"fruit"}, {"tool", "vehicle"}
                ),
                "edible_nonfruit_vs_artifact_auc": comparison_auc(
                    surface_rows,
                    variant,
                    {"edible_nonfruit"},
                    {"tool", "vehicle"},
                ),
            }
        means = {
            key: mean(value[key] for value in by_surface.values())
            for key in next(iter(by_surface.values()))
        }
        qualified = sum(
            value["edible_vs_nonedible_auc"] >= protocol.MIN_SURFACE_AUC
            for value in by_surface.values()
        )
        passed = bool(
            means["edible_vs_nonedible_auc"] >= protocol.MIN_EDIBLE_VS_NONEDIBLE_AUC
            and means["edible_vs_nonfood_plant_auc"]
            >= protocol.MIN_EDIBLE_VS_NONFOOD_PLANT_AUC
            and means["fruit_vs_artifact_auc"]
            >= protocol.MIN_EDIBLE_GROUP_VS_ARTIFACT_AUC
            and means["edible_nonfruit_vs_artifact_auc"]
            >= protocol.MIN_EDIBLE_GROUP_VS_ARTIFACT_AUC
            and qualified >= protocol.MIN_QUALIFIED_SURFACES
        )
        variant_metrics[variant] = {
            "mean_auc": means,
            "qualified_surface_count": qualified,
            "surface_metrics": by_surface,
            "pass": passed,
        }
    max_repeat_delta = max(repeat_deltas, default=0.0)
    return {
        "case_count": len(split_cases),
        "output_row_count": len(split_outputs),
        "surface_count": len(surfaces),
        "maximum_repeat_score_delta": max_repeat_delta,
        "variant_metrics": variant_metrics,
        "pass": bool(
            max_repeat_delta <= protocol.MAX_REPEAT_SCORE_DELTA
            and all(value["pass"] for value in variant_metrics.values())
        ),
    }


def run(model: str, restart: bool) -> Path:
    frozen = json.loads(protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
    if frozen["open_cases_sha256"] != sha256_file(protocol.OPEN_CASES_PATH):
        raise RuntimeError("Phase589 protocol drift")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase589 observer requires CUDA")
    output = paths(model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    cases = list(iter_jsonl(protocol.OPEN_CASES_PATH))
    if any(row["sealed"] for row in cases):
        raise RuntimeError("Phase589 observer received sealed rows")
    write_json(
        output["contract"],
        {
            "schema_version": "phase589_food_attribute_contract.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": model,
            "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
            "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
            "splits_read": list(protocol.OPEN_SPLITS),
            "sealed_split_read": False,
            "candidate_continuations_inserted_into_model_input": False,
            "torch_dtype_requested": "torch.bfloat16",
        },
    )
    loaded = None
    output_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase589 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "left"
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase589 requires BF16, got {dtype}")
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
                            repeat,
                        )
                    )
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase589 "
                    f"{split}/{repeat} {len(split_rows)}/{len(split_rows)}",
                    flush=True,
                )
        split_metrics = {
            split: summarize_split(cases, output_rows, split)
            for split in protocol.OPEN_SPLITS
        }
        authorized = all(value["pass"] for value in split_metrics.values())
        write_jsonl(output["rows"], output_rows)
        summary = {
            "schema_version": "phase589_food_attribute_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "case_count": len(cases),
            "row_count": len(output_rows),
            "split_metrics": split_metrics,
            "open_hidden_capture_authorized": authorized,
            "natural_generation_qualified": False,
            "causal_intervention_authorized": False,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
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
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.restart)


if __name__ == "__main__":
    main()
