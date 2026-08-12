#!/usr/bin/env python3
"""Refine broad prompt support inside the frozen cache-value carrier layers."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
import phase1002_kv_value_position_localization as position_test
from phase1002_multitoken_frozen_topology import read_json
from phase1002_multitoken_protocol import (
    MODELS,
    OUT_ROOT,
    write_json,
    write_jsonl,
)


PHASE = 1002
REFINED_GROUPS = (
    "slot0_entity",
    "slot1_entity",
    "slot0_color",
    "slot1_color",
    "query_name",
    "prompt_boundary",
    "generated_prefix",
    "user_header",
    "fact_scaffold",
    "query_scaffold",
    "instruction",
    "assistant_protocol",
)


def make_group_builder(tokenizer):
    boundary_cache: dict[str, dict[str, int]] = {}

    def prefix_boundary(
        text: str, char_index: int, ids: list[int]
    ) -> int:
        prefix = tokenizer.encode(
            text[:char_index], add_special_tokens=False
        )
        if list(ids[:len(prefix)]) != list(prefix):
            raise RuntimeError(
                f"prefix tokenization drift at char {char_index}"
            )
        return len(prefix)

    def boundaries(case: dict[str, Any]) -> dict[str, int]:
        record_id = case["record_id"]
        if record_id in boundary_cache:
            return boundary_cache[record_id]
        rendered = case["rendered_prompt"]
        raw = case["raw_prompt"]
        raw_start = rendered.index(raw)
        lines = raw.splitlines()
        if len(lines) != 3:
            raise RuntimeError(f"line drift: {lines}")
        query_char = raw_start + len(lines[0]) + 1
        instruction_char = query_char + len(lines[1]) + 1
        assistant_char = raw_start + len(raw)
        values = {
            "query_start": prefix_boundary(
                rendered, query_char, case["input_ids"]
            ),
            "instruction_start": prefix_boundary(
                rendered, instruction_char, case["input_ids"]
            ),
            "assistant_start": prefix_boundary(
                rendered, assistant_char, case["input_ids"]
            ),
        }
        boundary_cache[record_id] = values
        return values

    def builder(
        rows: list[dict[str, Any]],
        prefix_length: int,
    ) -> dict[str, list[list[int]]]:
        result = {name: [] for name in REFINED_GROUPS}
        for row in rows:
            target = row["target"]
            roles = target["role_positions"]
            prompt_length = int(target["input_token_count"])
            marks = boundaries(target)
            first_fact = min(
                int(roles[name])
                for name in (
                    "slot0_entity",
                    "slot1_entity",
                    "slot0_color",
                    "slot1_color",
                )
            )
            atomic = {
                "slot0_entity": [int(roles["slot0_entity"])],
                "slot1_entity": [int(roles["slot1_entity"])],
                "slot0_color": [int(roles["slot0_color"])],
                "slot1_color": [int(roles["slot1_color"])],
                "query_name": [int(roles["query_name"])],
                "prompt_boundary": [prompt_length - 1],
                "generated_prefix": list(
                    range(prompt_length, prefix_length)
                ),
            }
            excluded = {
                position
                for values in atomic.values()
                for position in values
            }
            ranges = {
                "user_header": (0, first_fact),
                "fact_scaffold": (
                    first_fact, marks["query_start"]
                ),
                "query_scaffold": (
                    marks["query_start"],
                    marks["instruction_start"],
                ),
                "instruction": (
                    marks["instruction_start"],
                    marks["assistant_start"],
                ),
                "assistant_protocol": (
                    marks["assistant_start"],
                    prompt_length,
                ),
            }
            for name, (start, end) in ranges.items():
                atomic[name] = [
                    position
                    for position in range(start, end)
                    if position not in excluded
                ]
            flattened = [
                position
                for name in REFINED_GROUPS
                for position in atomic[name]
            ]
            if (
                len(flattened) != prefix_length
                or len(set(flattened)) != prefix_length
            ):
                raise RuntimeError(
                    f"refined partition drift "
                    f"{len(flattened)}/{len(set(flattened))}/"
                    f"{prefix_length}"
                )
            for name in REFINED_GROUPS:
                result[name].append(atomic[name])
        return result

    return builder


def run_model(model_name: str, batch_size: int) -> dict[str, Any]:
    layer_summary = read_json(
        OUT_ROOT
        / "kv_value_layer_localization"
        / model_name
        / "summary.json"
    )
    broad_summary = read_json(
        OUT_ROOT
        / "kv_value_position_localization"
        / model_name
        / "summary.json"
    )
    if (
        not layer_summary["value_layer_localization_pass"]
        or not broad_summary["value_position_localization_pass"]
    ):
        raise RuntimeError(f"{model_name}: prerequisite gate failed")
    selected_layer_numbers = layer_summary["selection"][
        "selected_layer_numbers"
    ]
    selected_layers = {
        int(layer_number) - 1
        for layer_number in selected_layer_numbers
    }
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(
        prereg["frozen_phase1001_topology"][model_name]["source_depth"]
    )
    source_threshold = prereg["primary_thresholds"][
        "source_do_semantic_flip_rate"
    ]
    restore_threshold = prereg["primary_thresholds"][
        "frozen_topology_semantic_restore_rate"
    ]
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        layers = get_layers(model)
        group_builder = make_group_builder(tokenizer)
        screen_rows, ranking = position_test.screen_groups(
            model,
            layers,
            device,
            model_name,
            source_depth,
            selected_layers,
            batch_size,
            group_names=REFINED_GROUPS,
            group_builder=group_builder,
        )
        joint_rows = position_test.evaluate_group_sizes(
            model,
            layers,
            device,
            model_name,
            source_depth,
            selected_layers,
            ranking,
            batch_size,
            group_names=REFINED_GROUPS,
            group_builder=group_builder,
        )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()

    model_root = OUT_ROOT / "kv_value_background_refinement" / model_name
    write_jsonl(model_root / "discovery_screen_rows.jsonl", screen_rows)
    write_json(model_root / "frozen_region_ranking.json", ranking)
    write_jsonl(model_root / "joint_rows.jsonl", joint_rows)
    joint_summary = position_test.summarize_joint(
        joint_rows, group_count=len(REFINED_GROUPS)
    )
    selected_size, discovery_gate = position_test.choose_size(
        joint_summary["discovery"],
        source_threshold,
        restore_threshold,
    )
    confirmation = joint_summary["confirmation"][str(selected_size)]
    confirmation_gate = (
        confirmation["sufficiency_source_rate"] >= source_threshold
        and confirmation["restore_target_rate"] >= restore_threshold
        and confirmation["median_mediation"] >= 0.30
    )
    summary = {
        "schema_version": (
            "phase1002_kv_value_background_refinement_summary.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "source_depth": source_depth,
        "selected_layer_numbers": selected_layer_numbers,
        "screen_direction_count": 64,
        "formal_direction_count_per_split": 256,
        "region_ranking": ranking,
        "joint_summary": joint_summary,
        "selection": {
            "selected_group_count": selected_size,
            "selected_groups": [
                item["position_group"]
                for item in ranking[:selected_size]
            ],
            "discovery_gate": discovery_gate,
            "confirmation_gate": confirmation_gate,
        },
        "background_refinement_pass": (
            discovery_gate and confirmation_gate
        ),
        "thresholds": {
            "sufficiency_source_rate": source_threshold,
            "restore_target_rate": restore_threshold,
            "median_mediation": 0.30,
        },
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "These are token regions inside already localized value layers. "
            "They are not individual KV heads, channels, or neurons."
        ),
    }
    write_json(model_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def aggregate() -> dict[str, Any]:
    summaries = {
        model_name: read_json(
            OUT_ROOT
            / "kv_value_background_refinement"
            / model_name
            / "summary.json"
        )
        for model_name in MODELS
        if (
            OUT_ROOT
            / "kv_value_background_refinement"
            / model_name
            / "summary.json"
        ).exists()
    }
    payload = {
        "schema_version": (
            "phase1002_kv_value_background_refinement_cross_model.v1"
        ),
        "phase": PHASE,
        "models": summaries,
        "pass_count": sum(
            summary["background_refinement_pass"]
            for summary in summaries.values()
        ),
        "cross_model_pass": (
            len(summaries) == len(MODELS)
            and sum(
                summary["background_refinement_pass"]
                for summary in summaries.values()
            ) >= 2
        ),
    }
    write_json(
        OUT_ROOT / "kv_value_background_refinement" / "summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.model:
        run_model(args.model, args.batch_size)
    else:
        raise SystemExit("provide --model or --aggregate")


if __name__ == "__main__":
    main()
