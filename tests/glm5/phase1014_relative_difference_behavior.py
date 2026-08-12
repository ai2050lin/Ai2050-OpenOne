#!/usr/bin/env python3
"""Measure broad behavior for the Phase1014 relative-difference protocol."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1006_blind_source_and_behavior import (
    eos_token_ids,
    natural_generate,
    sequence_metrics,
)
from phase1014_relative_difference_protocol import (
    FAMILIES,
    MODELS,
    OUT_ROOT,
    OUTPUT_MODES,
    PAIR_OPERATIONS,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


BATCH_SIZE = 16


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def case_tensors(cases: list[dict[str, Any]], device):
    widths = {len(case["input_ids"]) for case in cases}
    if len(widths) != 1:
        raise RuntimeError(f"input width drift: {widths}")
    input_ids = torch.tensor(
        [case["input_ids"] for case in cases],
        dtype=torch.long,
        device=device,
    )
    return input_ids, torch.ones_like(input_ids)


def behavior_batch(
    *,
    model,
    layers,
    tokenizer,
    device,
    model_name: str,
    cases: list[dict[str, Any]],
    effective_eos: set[int],
) -> list[dict[str, Any]]:
    input_ids, attention = case_tensors(cases, device)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        logits = output.logits[:, -1, :]
        full_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        panel_ids = []
        for index, case in enumerate(cases):
            candidates = [
                int(value)
                for value in case["candidate_token_ids"].values()
            ]
            candidate_tensor = torch.tensor(
                candidates,
                dtype=torch.long,
                device=logits.device,
            )
            panel_ids.append(int(candidate_tensor[
                logits[index].index_select(0, candidate_tensor).argmax()
            ].item()))
        del output, logits
    finally:
        del input_ids, attention

    generated = natural_generate(
        model,
        layers,
        tokenizer,
        device,
        cases,
        effective_eos_ids=effective_eos,
    )
    sequence_rows, _ = sequence_metrics(generated, cases, effective_eos)
    sequence_by_id = {
        row["record_id"]: row for row in sequence_rows
    }
    rows = []
    for index, case in enumerate(cases):
        expected = int(case["answer_token_ids"][0])
        rollout = sequence_by_id[case["record_id"]]
        content = rollout["content_ids"]
        decoded = tokenizer.decode(content, skip_special_tokens=True)
        match = re.search(r"[A-Za-z]+", decoded)
        first_word = None if match is None else match.group(0)
        gold_text = str(case["natural_gold_text"])
        first_hit = bool(
            first_word is not None
            and first_word.casefold() == gold_text.casefold()
        )
        decoded_exact = decoded.strip().casefold() == gold_text.casefold()
        terminated = rollout["eos_position"] is not None
        rows.append({
            "schema_version": "phase1014_relative_behavior_row.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "model": model_name,
            "family": case["family"],
            "output_mode": case["output_mode"],
            "split": case["split"],
            "template": int(case["template"]),
            "name_pool": int(case["name_pool"]),
            "world_index": int(case["world_index"]),
            "unit_id": case["unit_id"],
            "record_id": case["record_id"],
            "state": case["state"],
            "gold": case["gold"],
            "expected_semantic_id": expected,
            "candidate_panel_prediction_id": int(panel_ids[index]),
            "full_vocabulary_prediction_id": int(full_ids[index]),
            "semantic_panel_hit": int(panel_ids[index]) == expected,
            "semantic_full_vocab_hit": int(full_ids[index]) == expected,
            "generated_ids": rollout["generated_ids"],
            "generated_content_ids": content,
            "generated_text": decoded,
            "generated_first_word": first_word,
            "rollout_first_word_hit": first_hit,
            "raw_token_exact": bool(rollout["exact"]),
            "natural_exact": bool(decoded_exact),
            "terminated_within_budget": bool(terminated),
            "eos_position": rollout["eos_position"],
            "rollout_gate": first_hit,
            "strict_rollout_gate": bool(decoded_exact and terminated),
            "batch_size": len(cases),
            "batch_behavior_is_not_used_for_singleton_panel_selection": True,
        })
    return rows


def pair_rows(
    units: list[dict[str, Any]],
    behavior_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for unit in units:
        for operation in PAIR_OPERATIONS:
            pair = unit["operation_pairs"][operation]
            base = behavior_by_id[pair["base"]]
            variant = behavior_by_id[pair["variant"]]
            rows.append({
                "schema_version": (
                    "phase1014_relative_pair_qualification.v1"
                ),
                "phase": PHASE,
                "protocol_revision": PROTOCOL_REVISION,
                "model": unit["model"],
                "family": unit["family"],
                "output_mode": unit["output_mode"],
                "split": unit["split"],
                "template": int(unit["template"]),
                "name_pool": int(unit["name_pool"]),
                "world_index": int(unit["world_index"]),
                "unit_id": unit["unit_id"],
                "operation": operation,
                "base_record_id": base["record_id"],
                "variant_record_id": variant["record_id"],
                "batched_semantic_pair_qualified": bool(
                    base["semantic_panel_hit"]
                    and variant["semantic_panel_hit"]
                ),
                "rollout_pair_qualified": bool(
                    base["rollout_gate"] and variant["rollout_gate"]
                ),
                "strict_rollout_pair_qualified": bool(
                    base["strict_rollout_gate"]
                    and variant["strict_rollout_gate"]
                ),
            })
    return rows


def aggregate(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for values, group in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, values)}
        item.update({
            "n": len(group),
            "semantic_panel_rate": float(np.mean([
                row["semantic_panel_hit"] for row in group
            ])),
            "semantic_full_vocab_rate": float(np.mean([
                row["semantic_full_vocab_hit"] for row in group
            ])),
            "rollout_first_word_rate": float(np.mean([
                row["rollout_first_word_hit"] for row in group
            ])),
            "natural_exact_rate": float(np.mean([
                row["natural_exact"] for row in group
            ])),
            "strict_rollout_rate": float(np.mean([
                row["strict_rollout_gate"] for row in group
            ])),
        })
        result.append(item)
    return result


def run_model(model_name: str) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    if int(protocol["protocol_revision"]) != PROTOCOL_REVISION:
        raise RuntimeError("Phase1014 protocol revision drift")
    cases = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    units = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "units.jsonl"
    )
    started = time.time()
    model = tokenizer = device = None
    rows = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        info = get_model_info(model, model_name)
        layers = get_layers(model)
        effective_eos = eos_token_ids(model, tokenizer, model_name)
        grouped = defaultdict(list)
        for case in cases:
            grouped[(
                case["family"],
                case["output_mode"],
                case["split"],
                int(case["template"]),
                case["state"],
                len(case["input_ids"]),
            )].append(case)
        for key, group in sorted(grouped.items()):
            panel = []
            for batch in chunks(group, BATCH_SIZE):
                panel.extend(behavior_batch(
                    model=model,
                    layers=layers,
                    tokenizer=tokenizer,
                    device=device,
                    model_name=model_name,
                    cases=batch,
                    effective_eos=effective_eos,
                ))
            rows.extend(panel)
            print(
                f"[behavior] {model_name}/"
                f"{'/'.join(str(value) for value in key[:5])} "
                f"n={len(panel)} "
                f"panel={np.mean([r['semantic_panel_hit'] for r in panel]):.3f} "
                f"rollout={np.mean([r['rollout_gate'] for r in panel]):.3f}",
                flush=True,
            )
        if len(rows) != len(cases):
            raise RuntimeError("behavior case coverage drift")
        behavior_by_id = {row["record_id"]: row for row in rows}
        if len(behavior_by_id) != len(rows):
            raise RuntimeError("duplicate behavior record")
        pairs = pair_rows(units, behavior_by_id)
        summary = {
            "schema_version": "phase1014_relative_behavior_summary.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "model": model_name,
            "protocol_digest": protocol["preregistration_digest"],
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "model_class": info.model_class,
                "loaded_8bit": True,
            },
            "case_count": len(rows),
            "pair_count": len(pairs),
            "effective_eos_ids": sorted(effective_eos),
            "panel_rates": aggregate(
                rows, ("family", "output_mode")
            ),
            "output_mode_rates": aggregate(rows, ("output_mode",)),
            "family_rates": aggregate(rows, ("family",)),
            "overall_semantic_panel_rate": float(np.mean([
                row["semantic_panel_hit"] for row in rows
            ])),
            "overall_full_vocab_rate": float(np.mean([
                row["semantic_full_vocab_hit"] for row in rows
            ])),
            "overall_rollout_first_word_rate": float(np.mean([
                row["rollout_gate"] for row in rows
            ])),
            "overall_strict_rollout_rate": float(np.mean([
                row["strict_rollout_gate"] for row in rows
            ])),
            "elapsed_seconds": time.time() - started,
            "claim_limits": [
                "batched behavior is descriptive and never selects "
                "singleton internal events",
                "natural first-word success is not a causal result",
            ],
        }
        output_root = OUT_ROOT / "behavior" / model_name
        write_jsonl(output_root / "rows.jsonl", rows)
        write_jsonl(
            output_root / "pair_qualification.jsonl",
            pairs,
        )
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    run_model(args.model)


if __name__ == "__main__":
    main()
