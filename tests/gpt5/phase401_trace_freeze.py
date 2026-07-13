#!/usr/bin/env python3
"""Freeze Phase401 semantic roles, generated spans, and trace inputs."""

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
from phase401_local_edge_protocol import (  # noqa: E402
    MODELS,
    OUT,
    SPLIT_SELECTED_COUNTS,
)


SOURCE = OUT / "protocol/private/phase401_frozen_execution_cases.jsonl"


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


def first_word_token_id(tokenizer: Any, word: str, prefix: str) -> int:
    ids = tokenizer(prefix + word, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError(f"Phase401 empty answer tokenization: {word}")
    return int(ids[0])


def lexical_prefix(tokenizer: Any, semantic_ids: list[int], target: str) -> str:
    observed = tokenizer.decode(semantic_ids, skip_special_tokens=True)
    for prefix in ("", " "):
        target_ids = tokenizer(prefix + target, add_special_tokens=False)["input_ids"]
        if target_ids and semantic_ids[: len(target_ids)] == [int(v) for v in target_ids]:
            return prefix
    if observed.strip().casefold().startswith(target.casefold()):
        return " " if observed.startswith(" ") else ""
    raise RuntimeError(f"Phase401 semantic span does not begin with target: {observed!r}")


def main() -> None:
    freeze = read_json(OUT / "phase401_behavior_freeze_summary.json")
    if not freeze["authorization"]["run_instrument_ledger"]:
        raise RuntimeError("Phase401 trace freeze is not authorized")
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
        generated = [int(value) for value in row["effective_generated_token_ids"]]
        pre_stop = [int(value) for value in row["generated_token_ids_before_stop"]]
        start = int(row["semantic_start_step"])
        completion = int(row["semantic_completion_step"])
        if not (0 <= start <= completion < len(pre_stop)):
            raise RuntimeError(f"Phase401 semantic transition invalid: {row['blind_case_id']}")
        semantic_ids = pre_stop[start : completion + 1]
        prefix = lexical_prefix(tokenizer, semantic_ids, row["target"])
        target_first = int(semantic_ids[0])
        distractor_ids = sorted(
            {
                first_word_token_id(tokenizer, value, prefix)
                for value in row["distractors"]
            }
        )
        if target_first in distractor_ids or not distractor_ids:
            raise RuntimeError(f"Phase401 invalid target competition: {row['blind_case_id']}")
        partitions = {
            receiver_name: partition_for_receiver(mapped, mapped[receiver_name][0])
            for receiver_name in ("query_end", "answer_anchor")
        }
        rows.append(
            {
                **row,
                "schema_version": "75.5.0",
                "phase_id": "Phase401-TraceFrozen",
                "prompt_token_ids_private": ids,
                "state_role_positions_private": {key: mapped[key] for key in STATE_ROLES},
                "attention_source_partitions_private": partitions,
                "first_generated_token_id_private": generated[0],
                "semantic_first_token_id_private": target_first,
                "distractor_first_token_ids_private": distractor_ids,
                "semantic_answer_token_prefix_private": prefix,
                "semantic_prefix_token_ids_private": generated[:start],
                "semantic_completion_prefix_token_ids_private": generated[:completion],
                "semantic_completion_token_id_private": generated[completion],
                "post_semantic_prefix_token_ids_private": generated[: completion + 1],
                "post_semantic_next_token_id_private": (
                    generated[completion + 1] if completion + 1 < len(generated) else None
                ),
                "semantic_completion_is_first_generation_decision": completion == 0,
                "exact_generated_replay_token_ids_private": generated,
            }
        )
    expected = (
        len(freeze["eligible_surfaces"])
        * sum(SPLIT_SELECTED_COUNTS.values())
        * 16
        * len(MODELS)
    )
    if len(rows) != expected:
        raise RuntimeError(f"Invalid Phase401 trace case count {len(rows)} != {expected}")
    private = OUT / "trace/protocol/private"
    write_jsonl(private / "phase401_trace_cases.jsonl", rows)
    for split in SPLIT_SELECTED_COUNTS:
        write_jsonl(
            private / f"phase401_{split}_trace_cases.jsonl",
            [row for row in rows if row["phase401_split"] == split],
        )
    instrument_ids = {
        groups["discovery"][0]
        for groups in freeze["selected_groups_private"].values()
    }
    instrument = [
        row for row in rows if row["anonymous_parallel_group_id"] in instrument_ids
    ]
    write_jsonl(private / "phase401_instrument_trace_cases.jsonl", instrument)
    payload = {
        "schema_version": "75.5.0",
        "phase_id": "Phase401-TraceProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "freeze_same_shape_exact_replay_and_real_compute_graph_ledgers",
        "denominator": {
            "eligible_surfaces": freeze["eligible_surfaces"],
            "selected_case_count": len(rows),
            "instrument_case_count": len(instrument),
            "split_case_counts": {
                split: sum(row["phase401_split"] == split for row in rows)
                for split in SPLIT_SELECTED_COUNTS
            },
        },
        "capture_contract": {
            "state_roles": list(STATE_ROLES),
            "attention_receivers": ["query_end", "answer_anchor"],
            "attention_source_roles": list(ATTENTION_SOURCE_ROLES),
            "all_layers": True,
            "actual_qkv_after_rotary": True,
            "attention_probabilities": True,
            "mlp_down_projection_input": True,
            "exact_full_generated_replay": True,
            "head_channel_neuron_selection": False,
        },
        "authorization": {
            "instrument_ledger": True,
            "discovery_local_edges": False,
            "calibration_local_edges": False,
            "physical_holdout": False,
            "head_channel_neuron_scan": False,
        },
        "claim_boundary": {
            "role_partition_is_a_causal_language_graph": False,
            "ledger_conservation_is_a_language_mechanism": False,
        },
    }
    write_json(OUT / "phase401_trace_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
