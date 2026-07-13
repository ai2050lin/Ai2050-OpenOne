#!/usr/bin/env python3
"""Materialize exact Phase397 positions and factor-trace execution cases."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase390_role_mapping import fragment_positions, prompt_token_ids  # noqa: E402
from phase397_multitask_protocol import CONDITIONS, MODELS  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase397_multitask_binding"
SOURCE = OUT / "protocol/private/phase397_frozen_execution_cases.jsonl"
CANDIDATE_LAYERS = {"qwen3": 20, "glm4": 22, "deepseek7b": 15}
WRONG_LAYERS = {"qwen3": 5, "glm4": 6, "deepseek7b": 4}


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


def require_positions(tokenizer: Any, ids: list[int], fragment: str, label: str) -> list[int]:
    positions = fragment_positions(tokenizer, ids, fragment)
    if not positions:
        raise RuntimeError(f"Missing Phase397 {label} fragment: {fragment}")
    return positions


def main() -> None:
    freeze = read_json(OUT / "phase397_behavior_freeze_summary.json")
    if not freeze["authorization"]["run_instrument_audit"]:
        raise RuntimeError("Phase397 factor trace is not authorized")
    tokenizers: dict[str, Any] = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(SOURCE):
        model = row["private_execution_model"]
        tokenizer = tokenizers[model]
        ids = prompt_token_ids(tokenizer, row)
        source_positions = require_positions(tokenizer, ids, row["source_fragment"], "source")
        query_positions = require_positions(tokenizer, ids, row["query_fragment"], "query")
        source_set = set(source_positions)
        slots = row["semantic_slot_fragments_private"]
        entities = {
            key: [position for position in require_positions(tokenizer, ids, slots[key], key) if position in source_set]
            for key in ("entity_a", "entity_b")
        }
        literals = {
            key: [position for position in require_positions(tokenizer, ids, slots[key], key) if position in source_set]
            for key in ("value_a", "value_b")
        }
        if any(not positions for positions in (*entities.values(), *literals.values())):
            raise RuntimeError(f"Phase397 source slot mapping failed: {row['blind_case_id']}")
        patch_count = sum(len(value) for value in literals.values())
        excluded = source_set | set(query_positions)
        random_positions = [position for position in range(max(query_positions) + 1) if position not in excluded][:patch_count]
        if len(random_positions) != patch_count:
            raise RuntimeError(f"Insufficient Phase397 random controls: {row['blind_case_id']}")
        step = int(row["target_decision_step"])
        generated = [int(value) for value in row["generated_token_ids"]]
        if step < 0 or step + 1 >= len(generated):
            raise RuntimeError(f"Invalid target transition: {row['blind_case_id']}")
        rows.append(
            {
                **row,
                "schema_version": "71.3.0",
                "phase_id": "Phase397-FactorTraceFrozen",
                "condition_code_private": row["contrast_condition"][0],
                "prompt_token_ids_private": ids,
                "target_decision_prefix_token_ids_private": generated[:step],
                "target_first_token_id_private": generated[step],
                "literal_value_positions_private": literals,
                "source_entity_positions_private": entities,
                "source_positions_private": source_positions,
                "query_positions_private": query_positions,
                "random_control_positions_private": random_positions,
                "candidate_layer": CANDIDATE_LAYERS[model],
                "wrong_depth_layer": WRONG_LAYERS[model],
            }
        )
    expected = len(freeze["eligible_surfaces"]) * 16 * len(CONDITIONS) * len(MODELS)
    if len(rows) != expected:
        raise RuntimeError(f"Invalid Phase397 trace denominator {len(rows)} != {expected}")
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["private_execution_model"], row["phase397_public_parallel_group_id"])][row["condition_code_private"]] = row
    if any(set(conditions) != set("ABCDEFGHIJ") for conditions in grouped.values()):
        raise RuntimeError("Incomplete Phase397 ten-condition groups")

    private = OUT / "factor_trace/protocol/private"
    write_jsonl(private / "phase397_factor_trace_cases.jsonl", rows)
    for split in ("discovery", "calibration", "physical_holdout"):
        write_jsonl(private / f"phase397_{split}_trace_cases.jsonl", [row for row in rows if row["phase397_split"] == split])
    instrument_ids = {groups["discovery"][0] for groups in freeze["selected_groups_private"].values()}
    instrument = [row for row in rows if row["anonymous_parallel_group_id"] in instrument_ids]
    write_jsonl(private / "phase397_instrument_trace_cases.jsonl", instrument)
    protocol = {
        "schema_version": "71.3.0",
        "phase_id": "Phase397-FactorTraceProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "trace_relation_content_order_syntax_and_query_factors_at_exact_source_value_states",
        "denominator": {
            "eligible_surfaces": freeze["eligible_surfaces"],
            "selected_group_count": freeze["denominator"]["selected_parallel_group_count"],
            "selected_case_count": len(rows),
            "instrument_case_count": len(instrument),
            "discovery_case_count": sum(row["phase397_split"] == "discovery" for row in rows),
            "calibration_case_count": sum(row["phase397_split"] == "calibration" for row in rows),
            "physical_holdout_case_count": sum(row["phase397_split"] == "physical_holdout" for row in rows),
        },
        "capture_contract": {
            "capture_object": "full_prompt_layer_input_at_two_externally_frozen_depths",
            "candidate_layers": CANDIDATE_LAYERS,
            "wrong_depth_layers": WRONG_LAYERS,
            "literal_positions_recomputed_per_case": True,
            "generation_transition_replayed": True,
            "top_k_reduction": False,
            "single_neuron_claim": False,
        },
        "factor_pairs": {
            "relation_x": "A_vs_B_same_literals_same_value_positions",
            "relation_y": "F_vs_G_same_literals_same_value_positions",
            "content": "A_vs_F_same_positions_different_literals",
            "order": "A_vs_C_same_binding_same_literals_clause_order_changed",
            "syntax": "A_vs_D_same_binding_same_literals_paraphrased",
            "query_source_invariance": "A_vs_E_identical_source_later_query_changed",
        },
        "authorization": {
            "instrument_trace": True,
            "discovery_trace": False,
            "calibration_trace": False,
            "physical_holdout_trace": False,
            "causal_intervention": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "trace_contrast_is_causal_binding": False,
            "wrong_depth_control_maps_full_depth_curve": False,
            "aggregate_token_state_is_neuron": False,
        },
    }
    write_json(OUT / "phase397_factor_trace_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
