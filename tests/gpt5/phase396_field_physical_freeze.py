#!/usr/bin/env python3
"""Freeze an independent field-extraction physical test after Phase395 calibration."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase390_role_mapping import fragment_positions, prompt_token_ids  # noqa: E402
from phase395_causal_freeze import CONDITION_MAP, SCENARIOS  # noqa: E402


SOURCE_OUT = ROOT / "tests/gpt5/result/phase395_natural_binding"
OUT = ROOT / "tests/gpt5/result/phase396_field_binding_physical"
SOURCE_CASES = SOURCE_OUT / "protocol/private/phase395_physical_holdout_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACE = "field_extraction"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def require_positions(tokenizer: Any, ids: list[int], fragment: str, label: str) -> list[int]:
    positions = fragment_positions(tokenizer, ids, fragment)
    if not positions:
        raise RuntimeError(f"Missing Phase396 {label} fragment: {fragment}")
    return positions


def main() -> None:
    phase395 = read_json(SOURCE_OUT / "phase395_causal_calibration_analysis.json")
    if not phase395["authorization"]["phase396_field_specific_physical_protocol"]:
        raise RuntimeError("Phase396 field physical protocol is not authorized")
    source_bytes = SOURCE_CASES.read_bytes()
    source_rows = [
        row for row in read_jsonl(SOURCE_CASES)
        if row["task_surface_private"] == SURFACE
    ]
    candidate = read_json(SOURCE_OUT / "phase395_discovery_candidate_freeze.json")["frozen_candidate"]
    calibration_groups = {
        row["phase395_public_parallel_group_id"]
        for row in read_jsonl(SOURCE_OUT / "protocol/private/phase395_calibration_cases.jsonl")
    }
    tokenizers: dict[str, Any] = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
    frozen: list[dict[str, Any]] = []
    for row in source_rows:
        model = row["private_execution_model"]
        tokenizer = tokenizers[model]
        ids = prompt_token_ids(tokenizer, row)
        step = int(row["target_decision_step"])
        generated = [int(value) for value in row["generated_token_ids"]]
        source_positions = require_positions(tokenizer, ids, row["source_fragment"], "source")
        query_positions = require_positions(tokenizer, ids, row["query_fragment"], "query")
        source_set = set(source_positions)
        slots = row["semantic_slot_fragments_private"]
        entities = {
            key: [position for position in require_positions(tokenizer, ids, slots[key], key) if position in source_set]
            for key in ("entity_a", "entity_b")
        }
        literals = {
            key: [int(value) for value in row["literal_value_positions_private"][key]]
            for key in ("value_a", "value_b")
        }
        count = sum(len(value) for value in literals.values())
        excluded = source_set | set(query_positions)
        random_positions = [
            position for position in range(max(query_positions) + 1)
            if position not in excluded
        ][:count]
        if step < 0 or step + 1 >= len(generated) or any(not value for value in entities.values()):
            raise RuntimeError(f"Invalid Phase396 case {row['blind_case_id']}")
        if len(random_positions) != count:
            raise RuntimeError(f"Insufficient Phase396 random controls {row['blind_case_id']}")
        frozen.append({
            **row,
            "schema_version": "70.0.0",
            "phase_id": "Phase396-FrozenFieldPhysical",
            "condition_code_private": CONDITION_MAP[row["contrast_condition"]],
            "prompt_token_ids_private": ids,
            "target_decision_prefix_token_ids_private": generated[:step],
            "target_first_token_id_private": generated[step],
            "literal_value_positions_private": literals,
            "source_entity_positions_private": entities,
            "source_positions_private": source_positions,
            "query_positions_private": query_positions,
            "random_control_positions_private": random_positions,
            "candidate_layer": candidate["model_layers"][model]["candidate_layer"],
            "wrong_depth_layer": candidate["model_layers"][model]["wrong_depth_layer"],
        })
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in frozen:
        group_id = row["phase395_public_parallel_group_id"]
        if group_id in calibration_groups:
            raise RuntimeError("Phase396 physical/calibration group overlap")
        groups[(row["private_execution_model"], group_id)][row["condition_code_private"]] = row
    if len(frozen) != 72 or len(groups) != 18 or any(set(items) != {"A", "B", "C", "D"} for items in groups.values()):
        raise RuntimeError(f"Invalid Phase396 denominator cases={len(frozen)} groups={len(groups)}")
    write_jsonl(OUT / "protocol/private/phase396_physical_cases.jsonl", frozen)
    inherited_gates = read_json(
        SOURCE_OUT / "phase395_causal_calibration_protocol.json"
    )["frozen_gates"]
    protocol = {
        "schema_version": "70.0.0",
        "phase_id": "Phase396-FieldPhysicalProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis_source": "Phase395 calibration field_extraction passed on all three models while crosssurface gate failed",
        "source_holdout_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "denominator": {
            "models": list(MODELS),
            "task_surface": SURFACE,
            "groups_per_model": 6,
            "conditions_per_group": 4,
            "case_count": len(frozen),
            "directions_per_model": 24,
            "total_direction_count": 72,
        },
        "scenarios": list(SCENARIOS),
        "frozen_gates_inherited_without_change": inherited_gates,
        "phase396_field_gate": {
            "all_three_models_required": True,
            "numeric_thresholds_inherited_from_phase395": True,
        },
        "independence": {
            "group_overlap_with_phase395_calibration": 0,
            "field_physical_groups_opened_after_protocol_freeze": True,
            "entity_recency_rows_loaded_for_file_filtering": True,
            "entity_recency_physical_internal_evaluation_run": False,
            "failed_physical_groups_replaceable": False,
        },
        "authorization": {
            "run_three_models_sequentially": True,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "field_specific_replication_is_crosssurface_rule": False,
            "context_transport_is_abstract_binding_algorithm": False,
            "physical_replication_is_natural_necessity": False,
        },
    }
    write_json(OUT / "phase396_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
