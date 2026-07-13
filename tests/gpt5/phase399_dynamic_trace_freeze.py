#!/usr/bin/env python3
"""Freeze semantic roles and exact generation times for Phase399 traces."""

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
from phase371c_blind_vector_contrast import static_roles  # noqa: E402
from phase390_role_mapping import fragment_positions, prompt_token_ids  # noqa: E402
from phase399_dynamic_binding_protocol import MODELS  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase399_dynamic_binding"
SOURCE = OUT / "protocol/private/phase399_frozen_execution_cases.jsonl"
STATE_ROLES = (
    "source_entity_a",
    "source_entity_b",
    "source_value_a",
    "source_value_b",
    "clause_end_0",
    "clause_end_1",
    "query_entity",
    "query_end",
    "answer_anchor",
)
ATTENTION_SOURCE_ROLES = (
    "source_entity_a",
    "source_entity_b",
    "source_value_a",
    "source_value_b",
    "source_structure",
    "query_entity",
    "query_context",
    "other_prior_context",
    "receiver_self",
)


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


def within(tokenizer: Any, ids: list[int], fragment: str, container: str) -> list[int]:
    allowed = set(fragment_positions(tokenizer, ids, container))
    return [
        position
        for position in fragment_positions(tokenizer, ids, fragment)
        if position in allowed
    ]


def role_positions(tokenizer: Any, row: dict[str, Any], ids: list[int]) -> dict[str, list[int]]:
    slots = row["semantic_slot_fragments_private"]
    source = set(fragment_positions(tokenizer, ids, row["source_fragment"]))
    query = set(fragment_positions(tokenizer, ids, row["query_fragment"]))
    mapped = {
        "source_entity_a": within(tokenizer, ids, slots["entity_a"], row["source_fragment"]),
        "source_entity_b": within(tokenizer, ids, slots["entity_b"], row["source_fragment"]),
        "source_value_a": within(tokenizer, ids, slots["value_a"], row["source_fragment"]),
        "source_value_b": within(tokenizer, ids, slots["value_b"], row["source_fragment"]),
        "query_entity": within(tokenizer, ids, slots["query_entity"], row["query_fragment"]),
    }
    for key, positions in mapped.items():
        if not positions:
            raise RuntimeError(f"Phase399 missing role {key}: {row['blind_case_id']}")
    clauses = row["clause_fragments_private"]
    for index, clause in enumerate(clauses):
        positions = [
            position
            for position in fragment_positions(tokenizer, ids, clause)
            if position in source
        ]
        if not positions:
            raise RuntimeError(
                f"Phase399 missing clause {index}: {row['blind_case_id']}"
            )
        mapped[f"clause_end_{index}"] = [max(positions)]
    static, base_length = static_roles(tokenizer, row)
    if base_length != len(ids):
        raise RuntimeError(f"Phase399 base length mismatch: {row['blind_case_id']}")
    mapped["query_end"] = [int(static[1])]
    mapped["answer_anchor"] = [len(ids) - 1]
    claimed_source = set().union(
        *(set(mapped[key]) for key in (
            "source_entity_a",
            "source_entity_b",
            "source_value_a",
            "source_value_b",
        ))
    )
    mapped["source_structure"] = sorted(source - claimed_source)
    mapped["query_context"] = sorted(query - set(mapped["query_entity"]))
    if not mapped["source_structure"] or not mapped["query_context"]:
        raise RuntimeError(f"Phase399 empty structural role: {row['blind_case_id']}")
    return {key: sorted(set(value)) for key, value in mapped.items()}


def partition_for_receiver(
    mapped: dict[str, list[int]], receiver: int
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    claimed: set[int] = set()
    for key in (
        "source_entity_a",
        "source_entity_b",
        "source_value_a",
        "source_value_b",
        "source_structure",
        "query_entity",
        "query_context",
    ):
        positions = [position for position in mapped[key] if position <= receiver]
        positions = [position for position in positions if position != receiver]
        if claimed.intersection(positions):
            raise RuntimeError(f"Phase399 overlapping attention role {key}")
        result[key] = positions
        claimed.update(positions)
    result["receiver_self"] = [receiver]
    claimed.add(receiver)
    result["other_prior_context"] = [
        position for position in range(receiver + 1) if position not in claimed
    ]
    flattened = [position for positions in result.values() for position in positions]
    if sorted(flattened) != list(range(receiver + 1)) or len(flattened) != len(
        set(flattened)
    ):
        raise RuntimeError("Phase399 attention source partition does not conserve prefix")
    return result


def main() -> None:
    freeze = read_json(OUT / "phase399_behavior_freeze_summary.json")
    if not freeze["authorization"]["run_instrument_trace"]:
        raise RuntimeError("Phase399 dynamic trace is not authorized")
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
        tokenizer = tokenizers[row["private_execution_model"]]
        ids = prompt_token_ids(tokenizer, row)
        mapped = role_positions(tokenizer, row, ids)
        step = int(row["target_completion_step"])
        generated = [int(value) for value in row["generated_token_ids"]]
        if step < 0 or step + 1 >= len(generated):
            raise RuntimeError(f"Phase399 target transition invalid: {row['blind_case_id']}")
        partitions = {
            receiver_name: partition_for_receiver(mapped, mapped[receiver_name][0])
            for receiver_name in ("query_end", "answer_anchor")
        }
        rows.append(
            {
                **row,
                "schema_version": "73.3.0",
                "phase_id": "Phase399-DynamicTraceFrozen",
                "prompt_token_ids_private": ids,
                "state_role_positions_private": {
                    key: mapped[key] for key in STATE_ROLES
                },
                "attention_source_partitions_private": partitions,
                "first_answer_token_id_private": generated[0],
                "target_completion_prefix_token_ids_private": generated[:step],
                "target_completion_token_id_private": generated[step],
                "post_target_prefix_token_ids_private": generated[: step + 1],
                "post_target_next_token_id_private": generated[step + 1],
                "target_completion_is_first_answer_decision": step == 0,
            }
        )
    expected = (
        len(freeze["eligible_surfaces"])
        * sum(freeze["denominator"]["split_group_counts_per_surface"].values())
        * 16
        * len(MODELS)
    )
    if len(rows) != expected:
        raise RuntimeError(f"Invalid Phase399 trace case count {len(rows)} != {expected}")
    private = OUT / "dynamic_trace/protocol/private"
    write_jsonl(private / "phase399_dynamic_trace_cases.jsonl", rows)
    for split in ("discovery", "calibration", "physical_holdout"):
        write_jsonl(
            private / f"phase399_{split}_dynamic_trace_cases.jsonl",
            [row for row in rows if row["phase399_split"] == split],
        )
    instrument_ids = {
        groups["discovery"][0]
        for groups in freeze["selected_groups_private"].values()
    }
    instrument = [
        row
        for row in rows
        if row["anonymous_parallel_group_id"] in instrument_ids
    ]
    write_jsonl(private / "phase399_instrument_dynamic_trace_cases.jsonl", instrument)
    protocol = {
        "schema_version": "73.3.0",
        "phase_id": "Phase399-DynamicTraceProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "trace_multi_position_parent_events_and_role_partitioned_attention_writes",
        "denominator": {
            "eligible_surfaces": freeze["eligible_surfaces"],
            "selected_case_count": len(rows),
            "instrument_case_count": len(instrument),
            "discovery_case_count": sum(
                row["phase399_split"] == "discovery" for row in rows
            ),
            "calibration_case_count": sum(
                row["phase399_split"] == "calibration" for row in rows
            ),
            "physical_holdout_case_count": sum(
                row["phase399_split"] == "physical_holdout" for row in rows
            ),
        },
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
            "role_attention_write_vectors": True,
            "role_attention_probability_mass": True,
            "attention_bias_is_separate_event": True,
            "depths": "all_layers",
            "raw_vectors_persisted": False,
            "group_factorial_event_metrics_persisted": True,
            "head_identity_persisted": False,
            "mlp_channel_identity_persisted": False,
        },
        "quality_gates": {
            "block_relative_error_max": 0.01,
            "attention_role_replay_relative_error_max": 0.01,
            "attention_probability_sum_absolute_error_max": 0.01,
            "exact_first_answer_replay_required": True,
            "exact_target_completion_replay_required": True,
        },
        "authorization": {
            "instrument_trace": True,
            "discovery_trace": False,
            "calibration_trace": False,
            "physical_holdout_trace": False,
            "joint_causal_intervention": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "role_attention_write_is_a_causal_binding_edge": False,
            "factorial_event_chain_is_a_dynamic_binding_algorithm": False,
            "trace_is_a_complete_language_path": False,
        },
    }
    write_json(OUT / "phase399_dynamic_trace_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
