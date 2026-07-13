#!/usr/bin/env python3
"""Freeze Phase402 token roles and four disjoint attention parent partitions."""

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
    STATE_ROLES,
    partition_for_receiver,
    role_positions,
)
from phase401_trace_freeze import lexical_prefix  # noqa: E402
from phase402_multiparent_protocol import (  # noqa: E402
    MODELS,
    OUT,
    PARENT_CATEGORIES,
    SPLIT_SELECTED_COUNTS,
)


SOURCE = OUT / "protocol/private/phase402_frozen_execution_cases.jsonl"
RECEIVER_ROLES = ("query_end", "query_entity", "answer_anchor")


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


def parent_partition(
    mapped: dict[str, list[int]], receiver: int
) -> dict[str, list[int]]:
    base = partition_for_receiver(mapped, receiver)
    result = {
        "source_content": sorted(
            base["source_entity_a"]
            + base["source_entity_b"]
            + base["source_value_a"]
            + base["source_value_b"]
        ),
        "source_structure": base["source_structure"],
        "query_local": sorted(
            base["query_entity"] + base["query_context"] + base["receiver_self"]
        ),
        "remaining_prefix": base["other_prior_context"],
    }
    if tuple(result) != PARENT_CATEGORIES:
        raise RuntimeError("Phase402 parent category order changed")
    flattened = [position for positions in result.values() for position in positions]
    if sorted(flattened) != list(range(receiver + 1)):
        raise RuntimeError("Phase402 parent partition does not conserve prefix")
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Phase402 parent partition overlaps")
    if any(not positions for positions in result.values()):
        raise RuntimeError("Phase402 parent partition contains an empty category")
    return result


def main() -> None:
    freeze = read_json(OUT / "phase402_behavior_freeze_summary.json")
    if not freeze["authorization"]["run_trace_and_instrument"]:
        raise RuntimeError("Phase402 trace freeze is not authorized")
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
    category_lengths: dict[tuple[str, str, str, str], tuple[int, ...]] = {}
    for row in read_jsonl(SOURCE):
        model = row["private_execution_model"]
        tokenizer = tokenizers[model]
        ids = prompt_token_ids(tokenizer, row)
        mapped = role_positions(tokenizer, row, ids)
        partitions = {
            receiver_role: parent_partition(
                mapped, int(mapped[receiver_role][0])
            )
            for receiver_role in RECEIVER_ROLES
        }
        group_key = (
            model,
            row["anonymous_parallel_group_id"],
            row["anonymous_condition_slot"].split("_")[0],
            row["anonymous_condition_slot"].split("_")[2],
        )
        lengths = tuple(
            len(partitions["query_end"][category])
            for category in PARENT_CATEGORIES
        )
        prior_lengths = category_lengths.setdefault(group_key, lengths)
        if prior_lengths != lengths:
            raise RuntimeError(
                f"Phase402 role length mismatch within donor controls: {group_key}"
            )

        generated = [int(value) for value in row["effective_generated_token_ids"]]
        pre_stop = [int(value) for value in row["generated_token_ids_before_stop"]]
        start = int(row["semantic_start_step"])
        completion = int(row["semantic_completion_step"])
        if not (0 <= start <= completion < len(pre_stop)):
            raise RuntimeError(f"Phase402 invalid semantic span: {row['blind_case_id']}")
        semantic_ids = pre_stop[start : completion + 1]
        prefix = lexical_prefix(tokenizer, semantic_ids, row["target"])
        target_first = int(semantic_ids[0])
        distractor_ids = sorted(
            {
                int(
                    tokenizer(prefix + value, add_special_tokens=False)["input_ids"][
                        0
                    ]
                )
                for value in row["distractors"]
            }
        )
        if target_first in distractor_ids or not distractor_ids:
            raise RuntimeError(
                f"Phase402 invalid target competition: {row['blind_case_id']}"
            )
        rows.append(
            {
                **row,
                "schema_version": "76.4.0",
                "phase_id": "Phase402-TraceFrozen",
                "prompt_token_ids_private": ids,
                "state_role_positions_private": {
                    key: mapped[key] for key in STATE_ROLES
                },
                "parent_partitions_private": partitions,
                "first_generated_token_id_private": generated[0],
                "semantic_first_token_id_private": target_first,
                "distractor_first_token_ids_private": distractor_ids,
                "semantic_answer_token_prefix_private": prefix,
                "semantic_prefix_token_ids_private": generated[:start],
                "semantic_completion_prefix_token_ids_private": generated[:completion],
                "exact_generated_replay_token_ids_private": generated,
                "remaining_prefix_is_generated_history": False,
            }
        )

    expected = freeze["denominator"]["selected_case_count"]
    if len(rows) != expected:
        raise RuntimeError(f"Phase402 trace count {len(rows)} != {expected}")
    private = OUT / "trace/protocol/private"
    write_jsonl(private / "phase402_trace_cases.jsonl", rows)
    for split in SPLIT_SELECTED_COUNTS:
        write_jsonl(
            private / f"phase402_{split}_trace_cases.jsonl",
            [row for row in rows if row["phase402_split"] == split],
        )
    instrument_groups = {
        groups["discovery"][0]
        for groups in freeze["selected_groups_private"].values()
    }
    instrument = [
        row
        for row in rows
        if row["anonymous_parallel_group_id"] in instrument_groups
    ]
    write_jsonl(private / "phase402_instrument_trace_cases.jsonl", instrument)
    payload = {
        "schema_version": "76.4.0",
        "phase_id": "Phase402-TraceProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "eligible_surfaces": freeze["eligible_surfaces"],
            "selected_case_count": len(rows),
            "instrument_case_count": len(instrument),
            "discovery_case_count": sum(
                row["phase402_split"] == "discovery" for row in rows
            ),
        },
        "parent_partition": {
            "categories": list(PARENT_CATEGORIES),
            "disjoint_and_prefix_conserving": True,
            "receiver_roles": list(RECEIVER_ROLES),
            "remaining_prefix_is_generated_history": False,
        },
        "authorization": {
            "instrument": True,
            "discovery_before_instrument_pass": False,
            "calibration": False,
            "physical_holdout": False,
        },
    }
    write_json(OUT / "phase402_trace_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
