#!/usr/bin/env python3
"""Audit cached Phase1058 generation against full recomputation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1054_joint_kv_rollout_scan as eos_tools
import phase1058_multitoken_translation_protocol as protocol
import phase1058_multitoken_translation_scan as scan


def evenly_spaced(rows: list[Any], count: int) -> list[Any]:
    if len(rows) <= count:
        return list(rows)
    return [
        rows[(index * len(rows)) // count] for index in range(count)
    ]


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summary = protocol.read_json(
        protocol.OUT_ROOT / "atlas" / model_name / "summary.json"
    )
    case_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    target_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"targets.{model_name}.jsonl"
    )
    cases = {
        int(row["semantic_case_index"]): row for row in case_rows
    }
    targets = {
        int(row["target_index"]): row for row in target_rows
    }
    selected_ids = [
        int(value)
        for value in summary["selected_target_indices"][
            "phrase_post_kv"
        ]
    ]
    selected = evenly_spaced(
        [targets[value] for value in selected_ids],
        int(prereg["cache_parity_pair_limit"]),
    )
    records = {
        int(row["target_index"]): row
        for row in summary["condition_records"]["phrase_post_kv"]
    }
    plan = prereg["model_plans"][model_name]
    condition = {
        "site": "source_phrase",
        "groups": [int(value) for value in plan["all_groups"]],
        "depths": [
            int(value) for value in plan["postsource_depths"]
        ],
    }
    started = time.time()
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        width = bridge.projection_width(layers[0].self_attn.k_proj)
        n_kv_heads = int(plan["n_kv_heads"])
        if width % n_kv_heads:
            raise RuntimeError("KV projection geometry drift")
        head_dim = width // n_kv_heads
        raw, _ = bridge.rollout_pairs(
            model,
            tokenizer,
            device,
            layers,
            selected,
            cases,
            condition,
            head_dim=head_dim,
            steps=int(prereg["generation_steps"]),
            pair_limit=len(selected),
            pair_batch_size=bridge.PAIR_BATCH_SIZE[model_name],
        )
        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        arm_checks = []
        detailed = []
        for row in raw:
            target_index = int(row["target_index"])
            cached = records[target_index]
            comparisons = {}
            for execution, side, cached_key in (
                ("clean", "target", "clean_target"),
                ("clean", "cross", "clean_cross"),
                ("patched", "target", "patched_target"),
                ("patched", "cross", "patched_cross"),
            ):
                recomputed = scan.censored_tokens(
                    [
                        int(value)
                        for value in row[execution][side]
                    ],
                    eos_ids,
                )
                cached_values = scan.censored_tokens(
                    [
                        int(value)
                        for value in cached[cached_key]
                    ],
                    eos_ids,
                )
                matches = recomputed == cached_values
                arm_checks.append(matches)
                comparisons[f"{execution}_{side}"] = {
                    "matches": matches,
                    "cached": cached_values,
                    "full_recompute": recomputed,
                }
            detailed.append({
                "target_index": target_index,
                "comparisons": comparisons,
            })
        match_rate = (
            sum(arm_checks) / len(arm_checks) if arm_checks else 0.0
        )
        output = {
            "schema_version": "phase1058_cache_parity_audit.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "pair_count": len(selected),
            "sequence_count": len(arm_checks),
            "eos_censored_exact_sequence_match_count": sum(arm_checks),
            "eos_censored_exact_sequence_match_rate": match_rate,
            "passed": (
                match_rate >= prereg["gates"][
                    "cache_parity_rate_min"
                ]
            ),
            "records": detailed,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(
            protocol.OUT_ROOT
            / "atlas"
            / model_name
            / "cache_parity_audit.json",
            output,
        )
        print(json.dumps({
            "model": model_name,
            "pairs": len(selected),
            "sequences": len(arm_checks),
            "exact_sequence_match_rate": match_rate,
            "passed": output["passed"],
            "elapsed_seconds": output["elapsed_seconds"],
        }), flush=True)
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=protocol.MODELS
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
