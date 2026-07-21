#!/usr/bin/env python3
"""Run Phase590 natural generation and tokenwise diagnostics on one CUDA model."""

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
import phase590_natural_semantic_event_protocol as protocol  # noqa: E402


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paths(model: str) -> dict[str, Path]:
    stem = protocol.OUT_DIR / f"phase590_{model}_natural_semantic_event"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def first_semantic_prefix(tokenizer: Any, generated_ids: list[int]) -> dict[str, Any]:
    for index in range(1, len(generated_ids) + 1):
        prefix = tokenizer.decode(generated_ids[:index], skip_special_tokens=True)
        parsed = protocol.classify_semantic_text(prefix)
        if parsed["semantic_event_observed"]:
            return {
                "semantic_prefix_token_index": index,
                "semantic_prefix_text": parsed["normalized_generated"],
                "semantic_prefix_polarity": parsed["semantic_polarity"],
            }
    return {
        "semantic_prefix_token_index": None,
        "semantic_prefix_text": None,
        "semantic_prefix_polarity": None,
    }


def generate_batch(loaded: Any, model: str, rows: list[dict[str, Any]], repeat: str) -> list[dict[str, Any]]:
    prompts = [render_chat(loaded.tokenizer, model, row["raw_prompt"]) for row in rows]
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
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
        generated_ids = [int(value) for value in generated[index, width:].tolist()]
        if loaded.tokenizer.eos_token_id in generated_ids:
            generated_ids = generated_ids[: generated_ids.index(loaded.tokenizer.eos_token_id)]
        text = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = protocol.classify_semantic_text(text)
        output.append(
            {
                **row,
                "model": model,
                "execution_repeat": repeat,
                "generated_token_ids": generated_ids,
                "generated_token_count": len(generated_ids),
                "generated_text": text,
                **parsed,
                **first_semantic_prefix(loaded.tokenizer, generated_ids),
                "semantic_correct": parsed["semantic_polarity"] == row["expected_polarity"],
                "observer_only": True,
                "causal": False,
            }
        )
    del encoded, generated
    return output


def score_diagnostic_batch(loaded: Any, model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        for polarity, continuation in row["diagnostic_continuations"].items():
            token_ids = [
                int(value)
                for value in loaded.tokenizer.encode(continuation, add_special_tokens=False)
            ]
            if token_ids != row["diagnostic_token_ids_by_model"][model][polarity]:
                raise RuntimeError("Phase590 diagnostic tokenization drift")
            sequences.append(prompt + token_ids)
            metadata.append(
                {
                    "row_index": row_index,
                    "polarity": polarity,
                    "prompt_length": len(prompt),
                    "continuation_ids": token_ids,
                }
            )
    pad_id = loaded.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = loaded.tokenizer.eos_token_id
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    pads: list[int] = []
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
    scored: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for sequence_index, item in enumerate(metadata):
        prompt_length = int(item["prompt_length"])
        continuation_ids = item["continuation_ids"]
        first_target = pads[sequence_index] + prompt_length
        target_ids = input_ids[
            sequence_index, first_target : first_target + len(continuation_ids)
        ]
        positions = torch.arange(
            first_target - 1,
            first_target + len(continuation_ids) - 1,
            device=loaded.input_device,
        )
        token_logits = logits[sequence_index, positions]
        selected = token_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        log_probs = selected - torch.logsumexp(token_logits, dim=1)
        values = [float(value) for value in log_probs.detach().cpu().tolist()]
        if not values or not all(math.isfinite(value) for value in values):
            raise RuntimeError("Phase590 non-finite diagnostic token score")
        scored[int(item["row_index"])][str(item["polarity"])] = {
            "token_ids": continuation_ids,
            "token_pieces": loaded.tokenizer.convert_ids_to_tokens(continuation_ids),
            "token_logprobs": values,
            "first_token_logprob": values[0],
            "mean_logprob": sum(values) / len(values),
            "sum_logprob": sum(values),
        }
    output = []
    for row_index, row in enumerate(rows):
        candidates = scored[row_index]
        target = row["expected_polarity"]
        foil = "negative" if target == "positive" else "positive"
        first_margin = candidates[target]["first_token_logprob"] - candidates[foil]["first_token_logprob"]
        mean_margin = candidates[target]["mean_logprob"] - candidates[foil]["mean_logprob"]
        output.append(
            {
                "case_id": row["case_id"],
                "split": row["split"],
                "semantic_group": row["semantic_group"],
                "object_id": row["object_id"],
                "surface_id": row["surface_id"],
                "expected_polarity": target,
                "candidate_token_ledger": candidates,
                "first_token_target_margin": first_margin,
                "mean_target_margin": mean_margin,
                "first_token_target_win": first_margin > 0.0,
                "full_mean_target_win": mean_margin > 0.0,
                "first_mean_direction_agree": (first_margin > 0.0) == (mean_margin > 0.0),
                "teacher_forced_auxiliary_only": True,
            }
        )
    del result, logits, input_ids, attention_mask
    return output


def stable_case(by_case: dict[str, dict[str, dict[str, Any]]], case_id: str) -> bool:
    pair = by_case.get(case_id, {})
    if set(pair) != set(protocol.NOOP_REPEATS):
        return False
    first = pair[protocol.NOOP_REPEATS[0]]
    second = pair[protocol.NOOP_REPEATS[1]]
    return bool(
        first["semantic_correct"]
        and second["semantic_correct"]
        and first["semantic_polarity"] == second["semantic_polarity"]
    )


def summarize_split(
    cases: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    split_cases = [row for row in cases if row["split"] == split]
    split_outputs = [row for row in generations if row["split"] == split]
    split_diagnostics = [row for row in diagnostics if row["split"] == split]
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in split_outputs:
        by_case[row["case_id"]][row["execution_repeat"]] = row
    stable_ids = {row["case_id"] for row in split_cases if stable_case(by_case, row["case_id"])}
    clear_count = sum(row["semantic_event_observed"] for row in split_outputs)
    correct_count = sum(row["semantic_correct"] for row in split_outputs)
    repeat_polarity_count = sum(
        set(pair) == set(protocol.NOOP_REPEATS)
        and pair[protocol.NOOP_REPEATS[0]]["semantic_polarity"]
        == pair[protocol.NOOP_REPEATS[1]]["semantic_polarity"]
        for pair in by_case.values()
    )
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_cases:
        by_object[row["object_id"]].append(row)
    qualified_objects = [
        object_id
        for object_id, object_rows in by_object.items()
        if sum(row["case_id"] in stable_ids for row in object_rows)
        >= protocol.MIN_STABLE_SURFACES_PER_OBJECT
    ]
    group_by_object = {
        object_id: object_rows[0]["semantic_group"] for object_id, object_rows in by_object.items()
    }
    qualified_by_group = dict(Counter(group_by_object[object_id] for object_id in qualified_objects))
    group_metrics = {}
    for group in protocol.OBJECT_LABELS:
        group_rows = [row for row in split_outputs if row["semantic_group"] == group]
        group_metrics[group] = {
            "row_count": len(group_rows),
            "coverage": sum(row["semantic_event_observed"] for row in group_rows) / max(1, len(group_rows)),
            "semantic_accuracy": sum(row["semantic_correct"] for row in group_rows) / max(1, len(group_rows)),
            "event_counts": dict(Counter(row["semantic_event"] for row in group_rows)),
        }
    coverage = clear_count / max(1, len(split_outputs))
    semantic_accuracy = correct_count / max(1, len(split_outputs))
    repeat_polarity_rate = repeat_polarity_count / max(1, len(split_cases))
    minimum_qualified = protocol.MIN_QUALIFIED_BY_SPLIT_GROUP[split]
    pass_gate = bool(
        coverage >= protocol.MIN_COVERAGE
        and semantic_accuracy >= protocol.MIN_SEMANTIC_ACCURACY
        and repeat_polarity_rate >= protocol.MIN_REPEAT_POLARITY_RATE
        and all(
            value["semantic_accuracy"] >= protocol.MIN_GROUP_ACCURACY
            for value in group_metrics.values()
        )
        and all(
            qualified_by_group.get(group, 0) >= minimum_qualified
            for group in protocol.OBJECT_LABELS
        )
    )
    prefix_positions = [
        int(row["semantic_prefix_token_index"])
        for row in split_outputs
        if row["semantic_prefix_token_index"] is not None
    ]
    return {
        "case_count": len(split_cases),
        "generation_row_count": len(split_outputs),
        "coverage": coverage,
        "semantic_accuracy": semantic_accuracy,
        "repeat_polarity_rate": repeat_polarity_rate,
        "stable_case_rate": len(stable_ids) / max(1, len(split_cases)),
        "qualified_object_count": len(qualified_objects),
        "qualified_object_count_by_group": qualified_by_group,
        "group_metrics": group_metrics,
        "semantic_prefix_token_index_mean": sum(prefix_positions) / max(1, len(prefix_positions)),
        "semantic_prefix_token_index_max": max(prefix_positions, default=None),
        "diagnostic": {
            "case_count": len(split_diagnostics),
            "first_token_target_win_rate": sum(row["first_token_target_win"] for row in split_diagnostics)
            / max(1, len(split_diagnostics)),
            "full_mean_target_win_rate": sum(row["full_mean_target_win"] for row in split_diagnostics)
            / max(1, len(split_diagnostics)),
            "first_mean_direction_agreement_rate": sum(
                row["first_mean_direction_agree"] for row in split_diagnostics
            )
            / max(1, len(split_diagnostics)),
        },
        "pass": pass_gate,
    }


def run(model: str, restart: bool) -> Path:
    frozen = json.loads(protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
    if frozen["open_cases_sha256"] != sha256_file(protocol.OPEN_CASES_PATH):
        raise RuntimeError("Phase590 protocol drift")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase590 requires CUDA")
    output = paths(model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    cases = list(iter_jsonl(protocol.OPEN_CASES_PATH))
    if any(row["sealed"] for row in cases):
        raise RuntimeError("Phase590 received sealed rows")
    write_json(
        output["contract"],
        {
            "schema_version": "phase590_natural_semantic_event_contract.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": model,
            "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
            "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
            "splits_read": list(protocol.OPEN_SPLITS),
            "sealed_split_read": False,
            "independent_human_gold_standard_available": False,
            "torch_dtype_requested": "torch.bfloat16",
        },
    )
    loaded = None
    generations: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase590 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "left"
        if loaded.tokenizer.pad_token_id is None:
            loaded.tokenizer.pad_token_id = loaded.tokenizer.eos_token_id
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase590 requires BF16, got {dtype}")
        for split in protocol.OPEN_SPLITS:
            split_rows = sorted(
                [row for row in cases if row["split"] == split], key=lambda row: row["case_id"]
            )
            for repeat in protocol.NOOP_REPEATS:
                for start in range(0, len(split_rows), protocol.FIXED_BATCH_SIZE):
                    generations.extend(
                        generate_batch(
                            loaded,
                            model,
                            split_rows[start : start + protocol.FIXED_BATCH_SIZE],
                            repeat,
                        )
                    )
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase590 {split}/{repeat} "
                    f"{len(split_rows)}/{len(split_rows)}",
                    flush=True,
                )
            for start in range(0, len(split_rows), protocol.FIXED_BATCH_SIZE):
                diagnostics.extend(
                    score_diagnostic_batch(
                        loaded,
                        model,
                        split_rows[start : start + protocol.FIXED_BATCH_SIZE],
                    )
                )
            print(
                f"[{time.strftime('%H:%M:%S')}] {model} Phase590 {split}/token-ledger "
                f"{len(split_rows)}/{len(split_rows)}",
                flush=True,
            )
        split_metrics = {
            split: summarize_split(cases, generations, diagnostics, split)
            for split in protocol.OPEN_SPLITS
        }
        authorized = all(metrics["pass"] for metrics in split_metrics.values())
        row_payload = [
            {"row_kind": "natural_generation", **row} for row in generations
        ] + [{"row_kind": "teacher_forced_token_ledger", **row} for row in diagnostics]
        write_jsonl(output["rows"], row_payload)
        summary = {
            "schema_version": "phase590_natural_semantic_event_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "case_count": len(cases),
            "generation_row_count": len(generations),
            "diagnostic_row_count": len(diagnostics),
            "split_metrics": split_metrics,
            "automatic_observer_qualified": authorized,
            "exploratory_open_hidden_capture_authorized": authorized,
            "mechanism_grade_trace_authorized": False,
            "causal_intervention_authorized": False,
            "independent_human_gold_standard_available": False,
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
