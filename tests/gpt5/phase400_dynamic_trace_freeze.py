#!/usr/bin/env python3
"""Freeze semantic roles, generation times, and answer competitors for Phase400."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase390_role_mapping import prompt_token_ids  # noqa: E402
from phase399_dynamic_trace_freeze import (  # noqa: E402
    ATTENTION_SOURCE_ROLES,
    STATE_ROLES,
    partition_for_receiver,
    role_positions,
)
from phase400_dynamic_protocol import MODELS, OUT  # noqa: E402
from phase400_partial_order_protocol import SPLIT_SELECTED_COUNTS  # noqa: E402


SOURCE = OUT / "protocol/private/phase400_frozen_execution_cases.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def contains_subsequence(values: list[int], candidate: list[int]) -> bool:
    return any(
        values[index : index + len(candidate)] == candidate
        for index in range(len(values) - len(candidate) + 1)
    )


def semantic_first_token(
    tokenizer: Any, generated: list[int], word: str
) -> tuple[int, str]:
    for prefix in ("", " "):
        ids = [
            int(value)
            for value in tokenizer(prefix + word, add_special_tokens=False)["input_ids"]
        ]
        if ids and contains_subsequence(generated, ids):
            return ids[0], prefix
    raise RuntimeError(f"Phase400 target token sequence absent from generation: {word}")


def first_word_token_id(tokenizer: Any, word: str, prefix: str) -> int:
    ids = tokenizer(prefix + word, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError(f"Phase400 empty answer tokenization: {word}")
    return int(ids[0])


def main() -> None:
    freeze = read_json(OUT / "phase400_behavior_freeze_summary.json")
    if not freeze["authorization"]["run_instrument_trace"]:
        raise RuntimeError("Phase400 dynamic trace is not authorized")
    tokenizers: dict[str, Any] = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(SOURCE):
        model = row["private_execution_model"]
        tokenizer = tokenizers[model]
        ids = prompt_token_ids(tokenizer, row)
        mapped = role_positions(tokenizer, row, ids)
        step = int(row["target_completion_step"])
        generated = [int(value) for value in row["generated_token_ids"]]
        if step < 0 or step + 1 >= len(generated):
            raise RuntimeError(f"Phase400 target transition invalid: {row['blind_case_id']}")
        target_first, lexical_prefix = semantic_first_token(
            tokenizer, generated, row["target"]
        )
        distractor_ids = sorted(
            {
                first_word_token_id(tokenizer, value, lexical_prefix)
                for value in row["distractors"]
            }
        )
        if target_first in distractor_ids or not distractor_ids:
            raise RuntimeError(f"Phase400 invalid target competition: {row['blind_case_id']}")
        partitions = {
            receiver_name: partition_for_receiver(mapped, mapped[receiver_name][0])
            for receiver_name in ("query_end", "answer_anchor")
        }
        rows.append(
            {
                **row,
                "schema_version": "74.4.0",
                "phase_id": "Phase400-DynamicTraceFrozen",
                "prompt_token_ids_private": ids,
                "state_role_positions_private": {key: mapped[key] for key in STATE_ROLES},
                "attention_source_partitions_private": partitions,
                "first_answer_token_id_private": generated[0],
                "target_first_token_id_private": target_first,
                "distractor_first_token_ids_private": distractor_ids,
                "semantic_answer_token_prefix_private": lexical_prefix,
                "target_completion_prefix_token_ids_private": generated[:step],
                "target_completion_token_id_private": generated[step],
                "post_target_prefix_token_ids_private": generated[: step + 1],
                "post_target_next_token_id_private": generated[step + 1],
                "target_completion_is_first_answer_decision": step == 0,
            }
        )
    expected = (
        len(freeze["eligible_surfaces"])
        * sum(SPLIT_SELECTED_COUNTS.values())
        * 16
        * len(MODELS)
    )
    if len(rows) != expected:
        raise RuntimeError(f"Invalid Phase400 trace case count {len(rows)} != {expected}")
    private = OUT / "dynamic_trace/protocol/private"
    write_jsonl(private / "phase400_dynamic_trace_cases.jsonl", rows)
    for split in SPLIT_SELECTED_COUNTS:
        write_jsonl(
            private / f"phase400_{split}_dynamic_trace_cases.jsonl",
            [row for row in rows if row["phase400_split"] == split],
        )
    instrument_ids = {
        groups["discovery"][0]
        for groups in freeze["selected_groups_private"].values()
    }
    instrument = [
        row for row in rows if row["anonymous_parallel_group_id"] in instrument_ids
    ]
    write_jsonl(private / "phase400_instrument_dynamic_trace_cases.jsonl", instrument)
    anchor_groups = {
        surface: groups["discovery"][0]
        for surface, groups in freeze["selected_groups_private"].items()
    }
    payload = {
        "schema_version": "74.4.0",
        "phase_id": "Phase400-DynamicTraceProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "capture_interval_direction_readout_and_raw_anchor_data_for_partial_order_graphs",
        "denominator": {
            "eligible_surfaces": freeze["eligible_surfaces"],
            "selected_case_count": len(rows),
            "instrument_case_count": len(instrument),
            "split_case_counts": {
                split: sum(row["phase400_split"] == split for row in rows)
                for split in SPLIT_SELECTED_COUNTS
            },
        },
        "raw_anchor_group_ids_private": anchor_groups,
        "capture_contract": {
            "state_roles": list(STATE_ROLES),
            "attention_receivers": ["query_end", "answer_anchor"],
            "attention_source_roles": list(ATTENTION_SOURCE_ROLES),
            "parent_components": [
                "layer_input",
                "attention_output",
                "mlp_output",
                "layer_output",
            ],
            "all_layers": True,
            "roq_effect_trajectories": True,
            "roq_cross_layer_direction_cosines": True,
            "query_and_terminal_target_competitor_logit_lens": True,
            "raw_signed_roq_anchor_private": True,
            "head_channel_neuron_identity": False,
        },
        "quality_gates": {
            "block_relative_error_max": 0.01,
            "attention_role_replay_relative_error_max": 0.01,
            "attention_probability_sum_absolute_error_max": 0.01,
            "exact_first_answer_replay_required": True,
            "exact_target_completion_replay_required": True,
            "exact_post_target_replay_required": True,
        },
        "authorization": {
            "instrument_trace": True,
            "discovery_trace": False,
            "calibration_trace": False,
            "physical_holdout_trace": False,
            "joint_causal_intervention": False,
            "head_channel_neuron_scan": False,
        },
        "claim_boundary": {
            "logit_lens_prediction_is_model_generation": False,
            "role_partition_is_a_causal_graph": False,
            "raw_anchor_is_a_neuron_map": False,
        },
    }
    write_json(OUT / "phase400_dynamic_trace_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
