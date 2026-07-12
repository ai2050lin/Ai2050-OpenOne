#!/usr/bin/env python3
"""Extract frozen multi-coordinate relations from Phase386 discovery events."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
COLLECTION = PHASE_ROOT / "collection/discovery"
CASES = PHASE_ROOT / "protocol/private/phase386_discovery_cases.jsonl"
OUT = PHASE_ROOT / "discovery_relations"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = ("relation_binding", "entity_recency", "field_extraction")
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
COORDINATES = (
    "source_encoded",
    "query_integrated",
    "pre_decision",
    "target_encoded",
    "post_decision_next_token",
)
TRANSITIONS = tuple(zip(COORDINATES[:-1], COORDINATES[1:]))
VECTOR_FAMILIES = (
    "layer_input",
    "attention_output",
    "mlp_output",
    "layer_output",
    "attention_head_state",
    "mlp_channel_product",
)
DEPTH_BIN_COUNT = 6
EPSILON = 1e-8


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


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


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    denominator = float(
        torch.linalg.vector_norm(left).item()
        * torch.linalg.vector_norm(right).item()
    )
    if denominator <= EPSILON:
        return 0.0
    return float(torch.dot(left, right).item() / denominator)


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).item())


def depth_bin(layer_index: int, layer_count: int) -> int:
    return min(
        DEPTH_BIN_COUNT - 1,
        int(math.floor(layer_index * DEPTH_BIN_COUNT / layer_count)),
    )


def repeat_key_value(values: torch.Tensor, head_count: int) -> torch.Tensor:
    if values.shape[1] == head_count:
        return values
    if head_count % values.shape[1]:
        raise RuntimeError("Attention head count is not divisible by KV head count")
    return values.repeat_interleave(head_count // values.shape[1], dim=1)


def exact_vectors(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    vectors = {
        name: payload["component_vectors"][name][0].float()
        for name in VECTOR_FAMILIES[:4]
    }
    vectors["mlp_channel_product"] = payload["mlp"][
        "down_projection_input_product_at_coordinates"
    ][0].float()
    attention_by_coordinate: dict[str, torch.Tensor] = {}
    for frame in payload["attention"]["frames"]:
        probabilities = frame["probabilities_receivers_all_sources"].float()
        values = frame["value_states_all_sources"].float()
        values = repeat_key_value(values, int(frame["head_count"]))
        head_states = torch.einsum(
            "bhqs,bhsd->bqhd", probabilities, values
        )[0]
        for index, coordinate in enumerate(frame["coordinate_names"]):
            attention_by_coordinate[coordinate] = head_states[index].flatten()
    if set(attention_by_coordinate) != set(COORDINATES):
        raise RuntimeError("Incomplete attention coordinate ledger")
    vectors["attention_head_state"] = torch.stack(
        [attention_by_coordinate[name] for name in COORDINATES]
    )
    if any(value.shape[0] != len(COORDINATES) for value in vectors.values()):
        raise RuntimeError("Phase386 exact vector coordinate count mismatch")
    return vectors


def condition_effects(
    vectors: dict[str, dict[str, torch.Tensor]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        family: (
            vectors[CONDITIONS[0]][family] - vectors[CONDITIONS[1]][family],
            vectors[CONDITIONS[2]][family] - vectors[CONDITIONS[3]][family],
        )
        for family in VECTOR_FAMILIES
    }


def median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def aggregate_cell(
    key: tuple[str, str, str, str, str, int, int],
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    model, mechanism, family, source, target, bin_index, layer_index = key
    nonzero_rate = sum(row["nonzero_effect"] for row in rows) / len(rows)
    positive_x = sum(row["relation_cosine_lex_x"] > 0 for row in rows) / len(rows)
    positive_y = sum(row["relation_cosine_lex_y"] > 0 for row in rows) / len(rows)
    result = {
        "schema_version": "60.8.0",
        "phase_id": "Phase386-DiscoveryRelations",
        "model": model,
        "mechanism_id": mechanism,
        "vector_family": family,
        "source_coordinate": source,
        "target_coordinate": target,
        "depth_bin": bin_index,
        "layer_index": layer_index,
        "group_count": len(rows),
        "nonzero_effect_group_rate": nonzero_rate,
        "median_relation_cosine_lex_x": median(
            row["relation_cosine_lex_x"] for row in rows
        ),
        "median_relation_cosine_lex_y": median(
            row["relation_cosine_lex_y"] for row in rows
        ),
        "positive_relation_group_rate_lex_x": positive_x,
        "positive_relation_group_rate_lex_y": positive_y,
        "median_source_lexical_replication": median(
            row["source_lexical_replication"] for row in rows
        ),
        "median_target_lexical_replication": median(
            row["target_lexical_replication"] for row in rows
        ),
    }
    result["gates"] = {
        "complete_group_count": result["group_count"]
        == gate["complete_group_count"],
        "nonzero_effect_group_rate": nonzero_rate
        >= gate["nonzero_effect_group_rate_min"],
        "median_relation_cosine_lex_x": result[
            "median_relation_cosine_lex_x"
        ]
        >= gate["median_relation_cosine_lex_x_min"],
        "median_relation_cosine_lex_y": result[
            "median_relation_cosine_lex_y"
        ]
        >= gate["median_relation_cosine_lex_y_min"],
        "positive_relation_group_rate_lex_x": positive_x
        >= gate["positive_relation_group_rate_lex_x_min"],
        "positive_relation_group_rate_lex_y": positive_y
        >= gate["positive_relation_group_rate_lex_y_min"],
        "median_source_lexical_replication": result[
            "median_source_lexical_replication"
        ]
        >= gate["median_source_lexical_replication_min"],
        "median_target_lexical_replication": result[
            "median_target_lexical_replication"
        ]
        >= gate["median_target_lexical_replication_min"],
    }
    result["all_discovery_gates_pass"] = all(result["gates"].values())
    return result


def main() -> None:
    contract = read_json(PHASE_ROOT / "phase386_relation_contract.json")
    if not contract["authorization"]["run_discovery_relation_extraction"]:
        raise RuntimeError("Phase386 discovery relation extraction is not authorized")
    cases = read_jsonl(CASES)
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        key = (
            case["private_execution_model"],
            case["mechanism_id"],
            case["phase386_public_parallel_group_id"],
        )
        condition = case["contrast_condition"]
        if condition in grouped[key]:
            raise RuntimeError(f"Duplicate Phase386 condition for {key}/{condition}")
        grouped[key][condition] = case
    if len(grouped) != 72 or any(set(rows) != set(CONDITIONS) for rows in grouped.values()):
        raise RuntimeError("Phase386 discovery four-condition groups are incomplete")

    node_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    relation_cells: dict[
        tuple[str, str, str, str, str, int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    completed = Counter()
    for (model, mechanism, group_id), condition_cases in sorted(grouped.items()):
        manifest = read_json(COLLECTION / "models" / model / "manifest.json")
        layer_count = int(manifest["layer_count"])
        for layer_index in range(layer_count):
            by_condition: dict[str, dict[str, torch.Tensor]] = {}
            for condition in CONDITIONS:
                case = condition_cases[condition]
                path = (
                    COLLECTION
                    / "private/models"
                    / model
                    / case["blind_case_id"]
                    / f"layer_{layer_index:03d}.pt"
                )
                by_condition[condition] = exact_vectors(
                    torch.load(path, map_location="cpu", weights_only=False)
                )
            effects = condition_effects(by_condition)
            bin_index = depth_bin(layer_index, layer_count)
            for family, (delta_x, delta_y) in effects.items():
                lexical = [cosine(delta_x[i], delta_y[i]) for i in range(5)]
                norms_x = [norm(delta_x[i]) for i in range(5)]
                norms_y = [norm(delta_y[i]) for i in range(5)]
                for coordinate_index, coordinate in enumerate(COORDINATES):
                    node_rows.append(
                        {
                            "schema_version": "60.8.0",
                            "phase_id": "Phase386-DiscoveryRelations",
                            "model": model,
                            "mechanism_id": mechanism,
                            "public_parallel_group_id": group_id,
                            "layer_index": layer_index,
                            "depth_bin": bin_index,
                            "coordinate": coordinate,
                            "vector_family": family,
                            "effect_norm_lex_x": norms_x[coordinate_index],
                            "effect_norm_lex_y": norms_y[coordinate_index],
                            "lexical_replication_cosine": lexical[coordinate_index],
                            "nonzero_effect": bool(
                                norms_x[coordinate_index] > EPSILON
                                and norms_y[coordinate_index] > EPSILON
                            ),
                        }
                    )
                for source_index, (source, target) in enumerate(TRANSITIONS):
                    target_index = source_index + 1
                    row = {
                        "schema_version": "60.8.0",
                        "phase_id": "Phase386-DiscoveryRelations",
                        "model": model,
                        "mechanism_id": mechanism,
                        "public_parallel_group_id": group_id,
                        "layer_index": layer_index,
                        "layer_count": layer_count,
                        "depth_bin": bin_index,
                        "source_coordinate": source,
                        "target_coordinate": target,
                        "vector_family": family,
                        "relation_cosine_lex_x": cosine(
                            delta_x[source_index], delta_x[target_index]
                        ),
                        "relation_cosine_lex_y": cosine(
                            delta_y[source_index], delta_y[target_index]
                        ),
                        "source_lexical_replication": lexical[source_index],
                        "target_lexical_replication": lexical[target_index],
                        "source_norm_lex_x": norms_x[source_index],
                        "source_norm_lex_y": norms_y[source_index],
                        "target_norm_lex_x": norms_x[target_index],
                        "target_norm_lex_y": norms_y[target_index],
                        "nonzero_effect": bool(
                            min(
                                norms_x[source_index],
                                norms_y[source_index],
                                norms_x[target_index],
                                norms_y[target_index],
                            )
                            > EPSILON
                        ),
                    }
                    relation_rows.append(row)
                    cell_key = (
                        model,
                        mechanism,
                        family,
                        source,
                        target,
                        bin_index,
                        layer_index,
                    )
                    relation_cells[cell_key].append(row)
            del by_condition, effects
        completed[(model, mechanism)] += 1
        print(
            f"[Phase386 relations] {model}/{mechanism} "
            f"groups={completed[(model, mechanism)]}/8",
            flush=True,
        )

    gate = contract["discovery_gate_per_model_exact_layer"]
    exact_cells = [
        aggregate_cell(key, rows, gate)
        for key, rows in sorted(relation_cells.items())
    ]
    passing_by_crosscell: dict[
        tuple[str, str, str, str, int], dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in exact_cells:
        if not row["all_discovery_gates_pass"]:
            continue
        key = (
            row["mechanism_id"],
            row["vector_family"],
            row["source_coordinate"],
            row["target_coordinate"],
            row["depth_bin"],
        )
        passing_by_crosscell[key][row["model"]].append(row)

    candidates: list[dict[str, Any]] = []
    for key, by_model in sorted(passing_by_crosscell.items()):
        if set(by_model) != set(MODELS):
            continue
        mechanism, family, source, target, bin_index = key
        selected = {
            model: min(by_model[model], key=lambda row: row["layer_index"])
            for model in MODELS
        }
        candidate_id = "p386r_" + digest(
            ":".join(map(str, [*key, *(selected[m]["layer_index"] for m in MODELS)]))
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "mechanism_id": mechanism,
                "vector_family": family,
                "source_coordinate": source,
                "target_coordinate": target,
                "depth_bin": bin_index,
                "model_layers": {
                    model: selected[model]["layer_index"] for model in MODELS
                },
                "model_discovery_metrics": {
                    model: {
                        key: value
                        for key, value in selected[model].items()
                        if key
                        in {
                            "group_count",
                            "nonzero_effect_group_rate",
                            "median_relation_cosine_lex_x",
                            "median_relation_cosine_lex_y",
                            "positive_relation_group_rate_lex_x",
                            "positive_relation_group_rate_lex_y",
                            "median_source_lexical_replication",
                            "median_target_lexical_replication",
                        }
                    }
                    for model in MODELS
                },
                "all_three_models_pass_discovery_gates": True,
                "model_layer_selection": "earliest_passing_exact_layer_in_depth_bin",
                "composite_score_used": False,
                "causal_claim": False,
            }
        )

    write_jsonl(OUT / "private/phase386_discovery_node_rows.jsonl", node_rows)
    write_jsonl(OUT / "private/phase386_discovery_relation_rows.jsonl", relation_rows)
    write_jsonl(OUT / "phase386_discovery_exact_layer_cells.jsonl", exact_cells)
    write_jsonl(OUT / "phase386_frozen_relation_candidates.jsonl", candidates)
    counts_mechanism = Counter(row["mechanism_id"] for row in candidates)
    counts_family = Counter(row["vector_family"] for row in candidates)
    counts_transition = Counter(
        f"{row['source_coordinate']}->{row['target_coordinate']}"
        for row in candidates
    )
    summary = {
        "schema_version": "60.8.0",
        "phase_id": "Phase386-DiscoveryRelations",
        "created_at": now(),
        "denominator": {
            "parallel_group_count": len(grouped),
            "node_row_count": len(node_rows),
            "relation_row_count": len(relation_rows),
            "exact_layer_cell_count": len(exact_cells),
            "passing_exact_layer_cell_count": sum(
                row["all_discovery_gates_pass"] for row in exact_cells
            ),
            "crossmodel_frozen_candidate_count": len(candidates),
        },
        "candidate_counts": {
            "by_mechanism": dict(sorted(counts_mechanism.items())),
            "by_vector_family": dict(sorted(counts_family.items())),
            "by_transition": dict(sorted(counts_transition.items())),
            "neuron_channel_relation_candidate_count": counts_family[
                "mlp_channel_product"
            ],
            "attention_head_relation_candidate_count": counts_family[
                "attention_head_state"
            ],
        },
        "results": {
            "all_raw_vectors_retained": True,
            "top_k_used": False,
            "pairwise_gram_materialized": False,
            "composite_relation_score_used": False,
            "descriptive_crossmodel_relation_candidates_found": bool(candidates),
            "causal_relation_established": False,
            "language_encoding_closed": False,
        },
        "authorization": {
            "calibration_collection": bool(candidates),
            "physical_holdout_collection": False,
            "causal_intervention": False,
        },
    }
    write_json(PHASE_ROOT / "phase386_discovery_relation_summary.json", summary)
    freeze = {
        "schema_version": "60.8.0",
        "phase_id": "Phase386-DiscoveryRelationFreeze",
        "created_at": now(),
        "candidate_count": len(candidates),
        "candidate_file": "discovery_relations/phase386_frozen_relation_candidates.jsonl",
        "candidate_file_sha256": hashlib.sha256(
            (OUT / "phase386_frozen_relation_candidates.jsonl").read_bytes()
        ).hexdigest(),
        "calibration_data_read_before_freeze": False,
        "physical_holdout_opened": False,
        "authorization": {
            "calibration_collection": bool(candidates),
            "physical_holdout_collection": False,
            "causal_intervention": False,
        },
    }
    write_json(PHASE_ROOT / "phase386_discovery_relation_freeze.json", freeze)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
