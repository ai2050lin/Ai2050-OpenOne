#!/usr/bin/env python3
"""Freeze exact query/answer coordinates for Phase398 compact full-depth traces."""

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
from phase398_joint_factorial_protocol import MODELS  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"
SOURCE = OUT / "protocol/private/phase398_frozen_execution_cases.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    freeze = read_json(OUT / "phase398_behavior_freeze_summary.json")
    if not freeze["authorization"]["run_instrument_trace"]:
        raise RuntimeError("Phase398 query trace is not authorized")
    tokenizers = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
    rows = []
    for row in read_jsonl(SOURCE):
        model = row["private_execution_model"]
        tokenizer = tokenizers[model]
        ids = prompt_token_ids(tokenizer, row)
        static, base_length = static_roles(tokenizer, row)
        if base_length != len(ids):
            raise RuntimeError(f"Phase398 base length mismatch: {row['blind_case_id']}")
        query_end = int(static[1])
        answer_anchor = len(ids) - 1
        query_entity = row["semantic_slot_fragments_private"]["query_entity"]
        query_set = set(fragment_positions(tokenizer, ids, row["query_fragment"]))
        query_entity_positions = [position for position in fragment_positions(tokenizer, ids, query_entity) if position in query_set]
        if not query_entity_positions or max(query_entity_positions) > query_end:
            raise RuntimeError(f"Phase398 query entity mapping failed: {row['blind_case_id']}")
        step = int(row["target_decision_step"])
        generated = [int(value) for value in row["generated_token_ids"]]
        if step < 0 or step + 1 >= len(generated):
            raise RuntimeError(f"Phase398 target transition invalid: {row['blind_case_id']}")
        rows.append({
            **row,
            "schema_version": "72.3.0", "phase_id": "Phase398-QueryTraceFrozen",
            "prompt_token_ids_private": ids,
            "target_decision_prefix_token_ids_private": generated[:step],
            "target_first_token_id_private": generated[step],
            "query_end_position_private": query_end,
            "query_entity_positions_private": query_entity_positions,
            "answer_anchor_position_private": answer_anchor,
            # first_target_step historically marks the token where the target
            # string first becomes complete, which can be later than its first
            # divergent token for multi-token values.
            "target_encoded_completion_token_id_private": generated[step],
        })
    expected = len(freeze["eligible_surfaces"]) * 16 * 16 * len(MODELS)
    if len(rows) != expected:
        raise RuntimeError(f"Invalid Phase398 trace case count {len(rows)} != {expected}")
    private = OUT / "query_trace/protocol/private"
    write_jsonl(private / "phase398_query_trace_cases.jsonl", rows)
    for split in ("discovery", "calibration", "physical_holdout"):
        write_jsonl(private / f"phase398_{split}_query_trace_cases.jsonl", [row for row in rows if row["phase398_split"] == split])
    instrument_ids = {groups["discovery"][0] for groups in freeze["selected_groups_private"].values()}
    instrument = [row for row in rows if row["anonymous_parallel_group_id"] in instrument_ids]
    write_jsonl(private / "phase398_instrument_query_trace_cases.jsonl", instrument)
    protocol = {
        "schema_version": "72.3.0", "phase_id": "Phase398-QueryTraceProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "map_full_depth_relation_order_query_factorial_effects_at_causally_legal_query_and_answer_coordinates",
        "denominator": {
            "eligible_surfaces": freeze["eligible_surfaces"], "selected_case_count": len(rows),
            "instrument_case_count": len(instrument),
            "discovery_case_count": sum(row["phase398_split"] == "discovery" for row in rows),
            "calibration_case_count": sum(row["phase398_split"] == "calibration" for row in rows),
            "physical_holdout_case_count": sum(row["phase398_split"] == "physical_holdout" for row in rows),
        },
        "capture_contract": {
            "coordinates": ["query_end", "answer_anchor"],
            "components": ["layer_input", "attention_output", "mlp_output", "layer_output"],
            "depths": "all_layers",
            "target_completion_logits": True,
            "common_value_a_minus_value_b_margin": False,
            "full_vectors_persisted": False,
            "group_level_factorial_effects_persisted": True,
            "top_k_reduction": False,
            "single_neuron_claim": False,
        },
        "authorization": {"instrument_trace": True, "discovery_trace": False, "calibration_trace": False, "physical_holdout_trace": False, "causal_intervention": False, "single_neuron_scan": False},
        "claim_boundary": {
            "factorial_interaction_is_causal_binding": False,
            "target_completion_step_is_first_value_divergence": False,
            "full_depth_query_trace_is_complete_language_path": False,
        },
    }
    write_json(OUT / "phase398_query_trace_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
