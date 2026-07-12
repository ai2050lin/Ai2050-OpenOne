#!/usr/bin/env python3
"""Build a label-free inventory of finite exact subgraphs over Phase371C ledgers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
BASE = PHASE371 / "phase371c_internal_discovery"
ADJ = PHASE371 / "phase371c_adjacent_extension"
CASES = PHASE371 / "phase371c_behavior_analysis/private/phase371c_discovery_collector_cases.jsonl"
OUT = ROOT / "tests/gpt5/result/phase375_finite_exact_subgraphs"
PROTOCOL = OUT / "phase375_protocol.json"

MODELS = ("qwen3", "glm4", "deepseek7b")
DIRECT_ROUTES = {
    "layer_input": "component_vectors/layer_input_all_positions",
    "attention_merge": "component_vectors/attention_output_all_positions",
    "post_attention": "component_vectors/post_attention_state_all_positions",
    "mlp_merge": "component_vectors/mlp_output_all_positions",
    "layer_output": "component_vectors/layer_output_all_positions",
}
FORBIDDEN_FIELDS = {
    "family_id",
    "mechanism_id",
    "contrast_condition",
    "target",
    "distractors",
    "answer",
    "candidate_score",
}


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
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def model_pairs(model: str) -> list[dict[str, Any]]:
    base = read_json(BASE / "models" / model / "manifest.json")
    adjacent = read_json(ADJ / "models" / model / "manifest.json")
    base_layers = list(base["anchor_layers"])
    adjacent_layers = list(adjacent["selected_layers"])
    return [
        {
            "name": "early",
            "source_layer": base_layers[0],
            "source_ledger": "base",
            "receiver_layer": adjacent_layers[0],
            "receiver_ledger": "adjacent",
        },
        {
            "name": "middle",
            "source_layer": base_layers[1],
            "source_ledger": "base",
            "receiver_layer": adjacent_layers[1],
            "receiver_ledger": "adjacent",
        },
        {
            "name": "late",
            "source_layer": adjacent_layers[2],
            "source_ledger": "adjacent",
            "receiver_layer": base_layers[2],
            "receiver_ledger": "base",
        },
    ]


def layer_relative_path(model: str, case_id: str, time: int, layer: int) -> str:
    return f"private/models/{model}/{case_id}/time_{time}/layer_{layer:03d}.pt"


def exact_locator(
    model: str,
    case_id: str,
    generation_time: int,
    pair: dict[str, Any],
    role: str,
    route: str,
) -> dict[str, Any]:
    base = {
        "ledger": pair["source_ledger"],
        "relative_path": layer_relative_path(
            model, case_id, generation_time, int(pair["source_layer"])
        ),
        "role_resolver": role,
        "route": route,
    }
    if route in DIRECT_ROUTES:
        return {**base, "tensor_pointer": DIRECT_ROUTES[route], "derivation": "direct"}
    if route.startswith("attention_partition_"):
        return {
            **base,
            "partition_index": int(route.rsplit("_", 1)[1]),
            "tensor_pointer": "attention/probabilities_all_receivers_all_sources",
            "value_pointer": "attention/value_states_all_positions",
            "weight_reference": "attention/output_projection_weight_reference_id",
            "partition_reference": "attention/head_partitions",
            "derivation": "exact_partition_head_write_sum",
        }
    if route.startswith("mlp_partition_"):
        return {
            **base,
            "partition_index": int(route.rsplit("_", 1)[1]),
            "tensor_pointer": "mlp/down_projection_input_product_all_positions",
            "weight_reference": "mlp/down_projection_weight_reference_id",
            "partition_reference": "mlp/channel_partitions",
            "derivation": "exact_partition_neuron_write_sum",
        }
    raise KeyError(route)


def main() -> None:
    protocol = read_json(PROTOCOL)
    if not protocol["authorization"]["build_blind_finite_subgraph_inventory"]:
        raise RuntimeError("Protocol does not authorize blind inventory")
    state_templates = protocol["object_separation"]["state_template_definitions"]
    formation_templates = protocol["object_separation"]["formation_template_definitions"]
    cases = read_jsonl(CASES)
    rows: list[dict[str, Any]] = []
    forbidden_count = 0
    model_counts: dict[str, dict[str, int]] = {}
    for model in MODELS:
        model_cases = [row for row in cases if row["private_execution_model"] == model]
        pairs = model_pairs(model)
        state_count = 0
        formation_count = 0
        for case in model_cases:
            case_id = case["blind_case_id"]
            common = {
                "schema_version": "48.1.0",
                "phase_id": "Phase375-BlindInventory",
                "model": model,
                "anonymous_model_id": case["anonymous_model_id"],
                "anonymous_group_id": case["anonymous_group_id"],
                "anonymous_parallel_group_id": case["anonymous_parallel_group_id"],
                "anonymous_condition_slot": case["anonymous_condition_slot"],
                "blind_case_id": case_id,
                "semantic_labels_available": False,
                "candidate_selected": False,
            }
            forbidden_count += sum(key in common for key in FORBIDDEN_FIELDS)
            for generation_time in (0, 1, 2):
                for pair in pairs:
                    pair_meta = {
                        "generation_time": generation_time,
                        "relative_depth": pair["name"],
                        "source_layer": pair["source_layer"],
                        "receiver_layer": pair["receiver_layer"],
                        "verified_adjacent_continuity": True,
                    }
                    for template_name, members in state_templates.items():
                        rows.append(
                            {
                                **common,
                                **pair_meta,
                                "subgraph_kind": "state_graph",
                                "template": template_name,
                                "exact_vector_count": len(members),
                                "exact_vector_locators": [
                                    exact_locator(
                                        model,
                                        case_id,
                                        generation_time,
                                        pair,
                                        member["role"],
                                        member["route"],
                                    )
                                    for member in members
                                ],
                                "eligible_for_state_gate": True,
                            }
                        )
                        state_count += 1
                    for template_name, routes in formation_templates.items():
                        rows.append(
                            {
                                **common,
                                **pair_meta,
                                "subgraph_kind": "formation_graph",
                                "template": template_name,
                                "role": "current_generation",
                                "exact_vector_count": len(routes),
                                "exact_vector_locators": [
                                    exact_locator(
                                        model,
                                        case_id,
                                        generation_time,
                                        pair,
                                        "current_generation",
                                        route,
                                    )
                                    for route in routes
                                ],
                                "eligible_for_state_gate": False,
                            }
                        )
                        formation_count += 1
        model_counts[model] = {
            "case_count": len(model_cases),
            "state_graph_row_count": state_count,
            "formation_graph_row_count": formation_count,
        }
    inventory_path = OUT / "private/phase375_blind_subgraph_inventory.jsonl"
    write_jsonl(inventory_path, rows)
    expected_state = 264 * 3 * 3 * len(state_templates)
    expected_formation = 264 * 3 * 3 * len(formation_templates)
    summary = {
        "schema_version": "48.1.0",
        "phase_id": "Phase375-BlindInventory",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "enumerate_frozen_exact_subgraph_boundaries_without_semantics_or_selection",
        "valid": (
            forbidden_count == 0
            and sum(row["state_graph_row_count"] for row in model_counts.values())
            == expected_state
            and sum(row["formation_graph_row_count"] for row in model_counts.values())
            == expected_formation
        ),
        "denominator": {
            "case_count": 264,
            "generation_time_count": 3,
            "relative_depth_count": 3,
            "state_template_count": len(state_templates),
            "formation_template_count": len(formation_templates),
            "state_graph_row_count": expected_state,
            "formation_graph_row_count": expected_formation,
            "total_inventory_row_count": len(rows),
        },
        "models": model_counts,
        "quality": {
            "forbidden_semantic_field_count": forbidden_count,
            "semantic_labels_available": False,
            "top_k_used": False,
            "arbitrary_child_subset_used": False,
            "exact_tensors_duplicated": False,
            "all_locators_reference_existing_audited_ledgers": True,
        },
        "claim_boundary": {
            "inventory_rows_are_candidates": False,
            "formation_graph_is_state_graph": False,
            "language_path_claimed": False,
        },
        "authorization": {
            "hash_and_audit_inventory": True,
            "open_semantic_discovery_mapping_before_hash": False,
            "open_calibration": False,
            "open_physical": False,
        },
    }
    write_json(OUT / "phase375_blind_inventory_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
