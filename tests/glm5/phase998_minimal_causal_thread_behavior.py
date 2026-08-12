#!/usr/bin/env python3
"""Phase 998 Qwen3 behavior admission for the causal-thread test."""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model
from phase998_minimal_causal_thread_protocol import (
    COLORS,
    MODEL,
    OUT_ROOT,
    PHASE,
    canonical,
    write_json,
)


COLOR_RE = re.compile(r"(?<![A-Za-z])(red|blue|green|yellow)(?![A-Za-z])", re.I)
STRICT_RE = re.compile(r"^\s*(red|blue|green|yellow)\s*[.!]?\s*$", re.I)
THRESHOLDS = {
    "candidate_accuracy": 0.95,
    "natural_accuracy": 0.95,
    "min_template_order_query_accuracy": 0.90,
    "repeat_stability": 0.98,
    "eos_rate": 0.95,
    "exact_short_rate": 0.90,
    "max_order_gap": 0.05,
    "max_query_gap": 0.05,
    "pair_closure": 0.90,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def eos_ids(model, tokenizer) -> list[int]:
    values = getattr(model.generation_config, "eos_token_id", None)
    if values is None:
        values = tokenizer.eos_token_id
    if isinstance(values, int):
        values = [values]
    result = sorted({int(value) for value in values if value is not None})
    if not result:
        raise RuntimeError("no EOS token IDs")
    return result


def strip_at_eos(ids: list[int], eos: set[int]) -> tuple[list[int], int | None]:
    position = next((index for index, value in enumerate(ids) if value in eos), None)
    return (ids if position is None else ids[:position], position)


def parse_generated(text: str) -> dict[str, Any]:
    matches = [match.group(1).lower() for match in COLOR_RE.finditer(text)]
    first = matches[0] if matches else None
    return {
        "first_color": first,
        "all_colors": matches,
        "exact_short": bool(STRICT_RE.fullmatch(text)),
    }


def batch_rows(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["template"],
            row["world"],
            row["order"],
            row["query_role"],
            row["arm"],
        ),
    )
    for template in range(4):
        group = [row for row in ordered if row["template"] == template]
        lengths = {row["input_token_count"] for row in group}
        if len(lengths) != 1:
            raise RuntimeError(f"template {template} length drift: {lengths}")
        for start in range(0, len(group), batch_size):
            yield group[start : start + batch_size]


def generation_once(model, input_ids, attention_mask, eos: list[int], pad: int, budget: int):
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
            max_new_tokens=budget,
            eos_token_id=eos,
            pad_token_id=pad,
            return_dict_in_generate=True,
            output_scores=False,
            output_hidden_states=False,
            output_attentions=False,
        )
    return generated.sequences[:, input_ids.shape[1] :].detach().cpu().tolist()


def accuracy(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([bool(row[field]) for row in rows])) if rows else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for template in range(4):
        for order in (0, 1):
            for query in (0, 1):
                subset = [
                    row
                    for row in rows
                    if row["template"] == template
                    and row["order"] == order
                    and row["query_role"] == query
                ]
                key = f"t{template}.o{order}.q{query}"
                groups[key] = {
                    "n": len(subset),
                    "candidate_accuracy": accuracy(subset, "candidate_correct"),
                    "natural_accuracy": accuracy(subset, "natural_correct_both"),
                    "repeat_stability": accuracy(subset, "repeat_stable"),
                    "eos_rate": accuracy(subset, "eos_both"),
                    "exact_short_rate": accuracy(subset, "exact_short_both"),
                }

    order_accuracy = {
        str(order): accuracy(
            [row for row in rows if row["order"] == order], "natural_correct_both"
        )
        for order in (0, 1)
    }
    query_accuracy = {
        str(query): accuracy(
            [row for row in rows if row["query_role"] == query],
            "natural_correct_both",
        )
        for query in (0, 1)
    }
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_groups[row["pair_id"]].append(row)
    pair_closed = {
        pair_id: len(items) == 2
        and all(item["candidate_correct"] and item["natural_correct_both"] for item in items)
        for pair_id, items in pair_groups.items()
    }
    world_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        world_groups[row["world_id"]].append(row)
    world_closed = {
        world_id: len(items) == 32
        and all(item["candidate_correct"] and item["natural_correct_both"] for item in items)
        for world_id, items in world_groups.items()
    }

    metrics = {
        "candidate_accuracy": accuracy(rows, "candidate_correct"),
        "natural_accuracy": accuracy(rows, "natural_correct_both"),
        "repeat_stability": accuracy(rows, "repeat_stable"),
        "eos_rate": accuracy(rows, "eos_both"),
        "exact_short_rate": accuracy(rows, "exact_short_both"),
        "min_template_order_query_accuracy": min(
            value["natural_accuracy"] for value in groups.values()
        ),
        "order_gap": abs(order_accuracy["0"] - order_accuracy["1"]),
        "query_gap": abs(query_accuracy["0"] - query_accuracy["1"]),
        "pair_closure": float(np.mean(list(pair_closed.values()))),
        "world_closure": float(np.mean(list(world_closed.values()))),
    }
    checks = {
        "candidate_accuracy": metrics["candidate_accuracy"]
        >= THRESHOLDS["candidate_accuracy"],
        "natural_accuracy": metrics["natural_accuracy"]
        >= THRESHOLDS["natural_accuracy"],
        "min_template_order_query_accuracy": metrics[
            "min_template_order_query_accuracy"
        ]
        >= THRESHOLDS["min_template_order_query_accuracy"],
        "repeat_stability": metrics["repeat_stability"]
        >= THRESHOLDS["repeat_stability"],
        "eos_rate": metrics["eos_rate"] >= THRESHOLDS["eos_rate"],
        "exact_short_rate": metrics["exact_short_rate"]
        >= THRESHOLDS["exact_short_rate"],
        "order_gap": metrics["order_gap"] <= THRESHOLDS["max_order_gap"],
        "query_gap": metrics["query_gap"] <= THRESHOLDS["max_query_gap"],
        "pair_closure": metrics["pair_closure"] >= THRESHOLDS["pair_closure"],
    }
    return {
        "schema_version": "phase998_behavior_summary.v1",
        "phase": PHASE,
        "model": MODEL,
        "row_count": len(rows),
        "pair_count": len(pair_groups),
        "world_count": len(world_groups),
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "checks": checks,
        "behavior_gate_pass": all(checks.values()),
        "order_accuracy": order_accuracy,
        "query_accuracy": query_accuracy,
        "subgroups": groups,
        "candidate_prediction_counts": dict(
            Counter(row["candidate_prediction"] for row in rows)
        ),
        "natural_prediction_counts": dict(
            Counter(str(row["natural_prediction_1"]) for row in rows)
        ),
        "pair_closed_count": sum(pair_closed.values()),
        "world_closed_count": sum(world_closed.values()),
    }


def run(scope: str, batch_size: int, budget: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 998 requires CUDA")
    protocol_root = OUT_ROOT / ("smoke" if scope == "smoke" else "protocol")
    cases = read_jsonl(protocol_root / "cases.jsonl")
    protocol = json.loads((protocol_root / "protocol.json").read_text(encoding="utf-8"))
    if not protocol.get("cpu_protocol_pass"):
        raise RuntimeError("CPU protocol gate is not open")
    output_root = OUT_ROOT / ("smoke_behavior" if scope == "smoke" else "behavior")
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "behavior_rows.jsonl"
    temp_path = output_path.with_suffix(".jsonl.tmp")

    model = tokenizer = None
    started = time.time()
    results: list[dict[str, Any]] = []
    try:
        model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16, use_8bit=False)
        tokenizer.padding_side = "left"
        pad = int(tokenizer.pad_token_id)
        effective_eos = eos_ids(model, tokenizer)
        eos_set = set(effective_eos)
        candidate_ids = {
            color: int(protocol["candidate_token_ids"][color]) for color in COLORS
        }
        batches = list(batch_rows(cases, batch_size))
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for batch_index, batch in enumerate(batches):
                input_ids = torch.tensor(
                    [row["input_ids"] for row in batch],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.ones_like(input_ids)
                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    last_logits = outputs.logits[:, -1, :].float()
                generated_1 = generation_once(
                    model, input_ids, attention_mask, effective_eos, pad, budget
                )
                generated_2 = generation_once(
                    model, input_ids, attention_mask, effective_eos, pad, budget
                )
                for index, row in enumerate(batch):
                    logits = {
                        color: float(last_logits[index, token_id].detach().cpu())
                        for color, token_id in candidate_ids.items()
                    }
                    ranked = sorted(COLORS, key=lambda color: logits[color], reverse=True)
                    suffix_1 = [int(value) for value in generated_1[index]]
                    suffix_2 = [int(value) for value in generated_2[index]]
                    before_1, eos_position_1 = strip_at_eos(suffix_1, eos_set)
                    before_2, eos_position_2 = strip_at_eos(suffix_2, eos_set)
                    text_1 = tokenizer.decode(
                        before_1,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    text_2 = tokenizer.decode(
                        before_2,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    parsed_1 = parse_generated(text_1)
                    parsed_2 = parse_generated(text_2)
                    item = {
                        "schema_version": "phase998_behavior_row.v1",
                        "phase": PHASE,
                        "model": MODEL,
                        "record_id": row["record_id"],
                        "pair_id": row["pair_id"],
                        "world_id": row["world_id"],
                        "split": row["split"],
                        "template": row["template"],
                        "order": row["order"],
                        "query_role": row["query_role"],
                        "arm": row["arm"],
                        "gold": row["gold"],
                        "foil": row["foil"],
                        "candidate_logits": logits,
                        "candidate_prediction": ranked[0],
                        "candidate_rank": ranked.index(row["gold"]) + 1,
                        "candidate_margin": logits[row["gold"]] - logits[row["foil"]],
                        "candidate_correct": ranked[0] == row["gold"],
                        "generated_suffix_1": suffix_1,
                        "generated_suffix_2": suffix_2,
                        "generated_before_eos_1": before_1,
                        "generated_before_eos_2": before_2,
                        "generated_text_1": text_1,
                        "generated_text_2": text_2,
                        "natural_prediction_1": parsed_1["first_color"],
                        "natural_prediction_2": parsed_2["first_color"],
                        "natural_correct_1": parsed_1["first_color"] == row["gold"],
                        "natural_correct_2": parsed_2["first_color"] == row["gold"],
                        "natural_correct_both": parsed_1["first_color"] == row["gold"]
                        and parsed_2["first_color"] == row["gold"],
                        "repeat_stable": before_1 == before_2 and parsed_1 == parsed_2,
                        "eos_position_1": eos_position_1,
                        "eos_position_2": eos_position_2,
                        "eos_both": eos_position_1 is not None
                        and eos_position_2 is not None,
                        "exact_short_1": parsed_1["exact_short"],
                        "exact_short_2": parsed_2["exact_short"],
                        "exact_short_both": parsed_1["exact_short"]
                        and parsed_2["exact_short"],
                    }
                    handle.write(canonical(item) + "\n")
                    results.append(item)
                handle.flush()
                del input_ids, attention_mask, outputs, last_logits
                if (batch_index + 1) % 8 == 0 or batch_index + 1 == len(batches):
                    print(
                        f"[behavior] {batch_index + 1}/{len(batches)} batches, "
                        f"{len(results)}/{len(cases)} rows",
                        flush=True,
                    )
        temp_path.replace(output_path)
        summary = summarize(results)
        summary.update(
            {
                "scope": scope,
                "batch_size": batch_size,
                "max_new_tokens": budget,
                "effective_eos_token_ids": effective_eos,
                "elapsed_seconds": time.time() - started,
                "cuda_device": torch.cuda.get_device_name(0),
                "input_protocol_manifest_sha256": protocol["case_manifest_sha256"],
            }
        )
        write_json(output_root / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    summary = run(args.scope, args.batch_size, args.max_new_tokens)
    print(
        json.dumps(
            {
                "passed": summary["behavior_gate_pass"],
                "scope": args.scope,
                "metrics": summary["metrics"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
