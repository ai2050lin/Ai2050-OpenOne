#!/usr/bin/env python3
"""Freeze Phase395 calibration interventions before any causal result is observed."""

from __future__ import annotations

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


OUT = ROOT / "tests/gpt5/result/phase395_natural_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("field_extraction", "entity_recency")
CONDITION_MAP = {
    "A_direct_lex_x": "A",
    "B_swapped_lex_x": "B",
    "C_direct_lex_y": "C",
    "D_swapped_lex_y": "D",
}
SCENARIOS = (
    "no_intervention",
    "identity_same_literal_candidate",
    "donor_same_literal_candidate",
    "donor_same_position_candidate",
    "donor_source_entities_candidate",
    "donor_same_count_random_candidate",
    "donor_query_candidate",
    "donor_full_source_candidate",
    "donor_same_literal_wrong_depth",
)


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


def positions_for_fragment(tokenizer: Any, ids: list[int], fragment: str, label: str) -> list[int]:
    positions = fragment_positions(tokenizer, ids, fragment)
    if not positions:
        raise RuntimeError(f"Missing {label} fragment: {fragment}")
    return positions


def main() -> None:
    replication = read_json(
        OUT / "calibration_analysis/phase395_calibration_replication.json"
    )
    if not replication["authorization"]["causal_intervention"]:
        raise RuntimeError("Phase395 calibration causal intervention is not authorized")
    candidate = read_json(OUT / "phase395_discovery_candidate_freeze.json")["frozen_candidate"]
    source_rows = read_jsonl(OUT / "protocol/private/phase395_calibration_cases.jsonl")
    tokenizers: dict[str, Any] = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )

    frozen: list[dict[str, Any]] = []
    for row in source_rows:
        model = row["private_execution_model"]
        tokenizer = tokenizers[model]
        ids = prompt_token_ids(tokenizer, row)
        step = int(row["target_decision_step"])
        generated = [int(value) for value in row["generated_token_ids"]]
        if step < 0 or step + 1 >= len(generated):
            raise RuntimeError(f"Invalid target step for {row['blind_case_id']}")
        slots = row["semantic_slot_fragments_private"]
        source_positions = positions_for_fragment(
            tokenizer, ids, row["source_fragment"], "source"
        )
        query_positions = positions_for_fragment(
            tokenizer, ids, row["query_fragment"], "query"
        )
        source_set = set(source_positions)
        source_entities = {
            key: [
                position for position in positions_for_fragment(
                    tokenizer, ids, slots[key], key
                )
                if position in source_set
            ]
            for key in ("entity_a", "entity_b")
        }
        if any(not positions for positions in source_entities.values()):
            raise RuntimeError(f"Missing source entity position for {row['blind_case_id']}")
        literal_positions = {
            key: [int(value) for value in row["literal_value_positions_private"][key]]
            for key in ("value_a", "value_b")
        }
        literal_count = sum(len(value) for value in literal_positions.values())
        excluded = source_set | set(query_positions)
        random_positions = [
            position for position in range(max(query_positions) + 1)
            if position not in excluded
        ][:literal_count]
        if len(random_positions) != literal_count:
            raise RuntimeError(f"Insufficient random controls for {row['blind_case_id']}")
        frozen.append({
            **row,
            "schema_version": "69.7.0",
            "phase_id": "Phase395-FrozenCausalCalibration",
            "condition_code_private": CONDITION_MAP[row["contrast_condition"]],
            "prompt_token_ids_private": ids,
            "target_decision_prefix_token_ids_private": generated[:step],
            "target_first_token_id_private": generated[step],
            "literal_value_positions_private": literal_positions,
            "source_entity_positions_private": source_entities,
            "source_positions_private": source_positions,
            "query_positions_private": query_positions,
            "random_control_positions_private": random_positions,
            "candidate_layer": candidate["model_layers"][model]["candidate_layer"],
            "wrong_depth_layer": candidate["model_layers"][model]["wrong_depth_layer"],
        })

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in frozen:
        grouped[(row["private_execution_model"], row["phase395_public_parallel_group_id"])][
            row["condition_code_private"]
        ] = row
    if len(frozen) != 144 or len(grouped) != 36:
        raise RuntimeError(f"Invalid frozen denominator: cases={len(frozen)} groups={len(grouped)}")
    for key, conditions in grouped.items():
        if set(conditions) != {"A", "B", "C", "D"}:
            raise RuntimeError(f"Incomplete causal group {key}")
        for left_name, right_name in (("A", "B"), ("C", "D")):
            left, right = conditions[left_name], conditions[right_name]
            if len(left["prompt_token_ids_private"]) != len(right["prompt_token_ids_private"]):
                raise RuntimeError(f"Prompt length mismatch in {key}/{left_name}{right_name}")
            if left["target_first_token_id_private"] == right["target_first_token_id_private"]:
                raise RuntimeError(f"Target token collision in {key}/{left_name}{right_name}")
            for field in (
                "source_positions_private",
                "query_positions_private",
                "random_control_positions_private",
            ):
                if len(left[field]) != len(right[field]):
                    raise RuntimeError(f"Position count mismatch for {field} in {key}")

    write_jsonl(OUT / "protocol/private/phase395_calibration_causal_cases.jsonl", frozen)
    protocol = {
        "schema_version": "69.7.0",
        "phase_id": "Phase395-CausalCalibrationProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "task_surfaces": list(SURFACES),
            "groups_per_surface_model": 6,
            "conditions_per_group": 4,
            "case_count": len(frozen),
            "directions_per_model": 48,
            "total_direction_count": 144,
        },
        "scenarios": list(SCENARIOS),
        "frozen_gates": {
            "minimum_median_same_literal_normalized_margin_mediation": 0.10,
            "minimum_same_literal_advantage_over_entity": 0.05,
            "minimum_same_literal_advantage_over_random": 0.05,
            "minimum_positive_same_literal_direction_rate": 0.75,
            "minimum_same_literal_answer_switch_rate": 0.50,
            "minimum_candidate_advantage_over_wrong_depth_for_depth_specificity": 0.05,
            "all_three_models_and_both_surfaces_required_for_shared_state": True,
        },
        "control_meaning": {
            "same_literal": "literal token identity is fixed while contextual relation membership changes",
            "same_position": "physical source position is fixed and literal content changes",
            "query": "integrated endpoint-state upper bound, not an upstream binding proof",
            "full_source": "broad source-state upper bound, not a minimal mechanism",
        },
        "authorization": {
            "instrument_audit": True,
            "full_calibration_after_instrument_audit": False,
            "physical_holdout": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "same_literal_failure_means_no_binding": False,
            "query_or_full_source_success_is_binding_closure": False,
            "same_position_success_is_abstract_binding": False,
            "calibration_success_is_physical_holdout": False,
        },
    }
    write_json(OUT / "phase395_causal_calibration_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
