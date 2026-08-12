#!/usr/bin/env python3
"""Summarize repeated real-head and real-neuron structure for Phase1008."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1008_global_response_atlas_protocol import (
    OUT_ROOT,
    PHASE,
    canonical,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


MODELS = ("qwen3", "glm4")
OPERATIONS = ("B", "Q", "BQ", "X")
OP_INDEX = {operation: index for index, operation in enumerate(OPERATIONS)}
ANALYSIS_CONTRACT = {
    "head": {
        "top_fraction": 0.25,
        "minimum_qualified_per_split": 8,
        "minimum_split_recurrence": 0.50,
        "group_rate": 0.40,
        "minimum_group_n": 2,
        "minimum_templates_per_split": 1,
        "minimum_pools_per_split": 2,
    },
    "neuron": {
        "top_fraction": 0.01,
        "minimum_qualified_per_split": 8,
        "minimum_split_recurrence": 0.15,
        "group_rate": 0.10,
        "minimum_group_n": 2,
        "minimum_templates_per_split": 1,
        "minimum_pools_per_split": 2,
    },
    "causal_selection": {
        "operations": ["B", "Q"],
        "selection_split": "discovery",
        "evaluation_split": "confirmation",
        "heads_per_operation": 3,
        "minimum_discovery_recurrence": 0.50,
        "selection_layer": "frozen_parent_atlas_peak",
    },
}


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def membership_matrix(
    values: np.ndarray,
    qualified: np.ndarray,
    top_fraction: float,
) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.bool_)
    component_count = values.shape[1]
    top_count = max(1, int(np.ceil(component_count * top_fraction)))
    rows = np.flatnonzero(qualified)
    if rows.size:
        indices = np.argpartition(
            values[rows], -top_count, axis=1
        )[:, -top_count:]
        result[
            np.repeat(rows, top_count),
            indices.reshape(-1),
        ] = True
    return result


def subset_stats(
    membership: np.ndarray,
    qualified: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    selected = qualified & mask
    count = int(selected.sum())
    if count == 0:
        return np.zeros(membership.shape[1], dtype=np.float64), 0
    return membership[selected].mean(axis=0), count


def group_support(
    membership: np.ndarray,
    qualified: np.ndarray,
    units: list[dict[str, Any]],
    *,
    split: str,
    field: str,
    minimum_n: int,
    minimum_rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = sorted({
        unit[field]
        for unit in units
        if unit["split"] == split
    })
    support = np.zeros(membership.shape[1], dtype=np.int32)
    detail: dict[str, Any] = {}
    for value in values:
        mask = np.array([
            unit["split"] == split and unit[field] == value
            for unit in units
        ], dtype=np.bool_)
        rates, count = subset_stats(membership, qualified, mask)
        eligible = count >= minimum_n
        if eligible:
            support += rates >= minimum_rate
        detail[str(value)] = {
            "qualified_n": count,
            "eligible": bool(eligible),
            "maximum_rate": safe_float(rates.max(initial=0.0)),
        }
    return support, detail


def population_metrics(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "participation_fraction_median": 0.0,
            "top_1pct_mass_median": 0.0,
            "top_5pct_mass_median": 0.0,
            "top_10pct_mass_median": 0.0,
        }
    total = values.sum(axis=1)
    squared = np.square(values).sum(axis=1)
    participation = np.square(total) / np.maximum(
        squared * values.shape[1], 1e-12
    )
    ordered = np.sort(values, axis=1)

    def mass_share(fraction: float) -> float:
        count = max(1, int(np.ceil(values.shape[1] * fraction)))
        shares = ordered[:, -count:].sum(axis=1) / np.maximum(total, 1e-12)
        return safe_float(np.median(shares))

    return {
        "participation_fraction_median": safe_float(
            np.median(participation)
        ),
        "top_1pct_mass_median": mass_share(0.01),
        "top_5pct_mass_median": mass_share(0.05),
        "top_10pct_mass_median": mass_share(0.10),
    }


def candidate_rows(
    *,
    model_name: str,
    component: str,
    target: dict[str, Any],
    operation: str,
    values: np.ndarray,
    qualified: np.ndarray,
    units: list[dict[str, Any]],
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    membership = membership_matrix(
        values, qualified, contract["top_fraction"]
    )
    split_rates: dict[str, np.ndarray] = {}
    split_counts: dict[str, int] = {}
    template_support: dict[str, np.ndarray] = {}
    pool_support: dict[str, np.ndarray] = {}
    group_detail: dict[str, Any] = {}
    for split in ("discovery", "confirmation"):
        split_mask = np.array([
            unit["split"] == split for unit in units
        ], dtype=np.bool_)
        rates, count = subset_stats(
            membership, qualified, split_mask
        )
        split_rates[split] = rates
        split_counts[split] = count
        template_support[split], template_detail = group_support(
            membership,
            qualified,
            units,
            split=split,
            field="template",
            minimum_n=contract["minimum_group_n"],
            minimum_rate=contract["group_rate"],
        )
        pool_support[split], pool_detail = group_support(
            membership,
            qualified,
            units,
            split=split,
            field="name_pool",
            minimum_n=contract["minimum_group_n"],
            minimum_rate=contract["group_rate"],
        )
        group_detail[split] = {
            "templates": template_detail,
            "name_pools": pool_detail,
        }
    eligible_splits = all(
        split_counts[split] >= contract["minimum_qualified_per_split"]
        for split in ("discovery", "confirmation")
    )
    candidate_mask = np.ones(values.shape[1], dtype=np.bool_)
    if not eligible_splits:
        candidate_mask[:] = False
    for split in ("discovery", "confirmation"):
        candidate_mask &= (
            split_rates[split] >= contract["minimum_split_recurrence"]
        )
        candidate_mask &= (
            template_support[split]
            >= contract["minimum_templates_per_split"]
        )
        candidate_mask &= (
            pool_support[split] >= contract["minimum_pools_per_split"]
        )
    candidate_indices = np.flatnonzero(candidate_mask)
    rows = []
    for component_index in candidate_indices:
        observed = qualified & membership[:, component_index]
        rows.append({
            "schema_version": "phase1008_repeated_component.v1",
            "phase": PHASE,
            "model": model_name,
            "component": component,
            "operation": operation,
            "stage": target["stage"],
            "role": target["role"],
            "layer": int(target["layer"]),
            "relative_depth": float(target["relative_depth"]),
            (
                "head_index"
                if component == "attention_head"
                else "neuron_index"
            ): int(component_index),
            "top_fraction": float(contract["top_fraction"]),
            "qualified_n": int(qualified.sum()),
            "discovery_qualified_n": split_counts["discovery"],
            "confirmation_qualified_n": split_counts["confirmation"],
            "discovery_recurrence": safe_float(
                split_rates["discovery"][component_index]
            ),
            "confirmation_recurrence": safe_float(
                split_rates["confirmation"][component_index]
            ),
            "discovery_template_support": int(
                template_support["discovery"][component_index]
            ),
            "confirmation_template_support": int(
                template_support["confirmation"][component_index]
            ),
            "discovery_pool_support": int(
                pool_support["discovery"][component_index]
            ),
            "confirmation_pool_support": int(
                pool_support["confirmation"][component_index]
            ),
            "median_write_when_observed": safe_float(
                np.median(values[observed, component_index])
                if observed.any()
                else 0.0
            ),
            "evidence_meaning": (
                "repeated descriptive contributor; neither transport nor "
                "causal necessity has been established"
            ),
        })
    population = population_metrics(values[qualified])
    population.update({
        "schema_version": "phase1008_population_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "component": component,
        "operation": operation,
        "stage": target["stage"],
        "role": target["role"],
        "layer": int(target["layer"]),
        "relative_depth": float(target["relative_depth"]),
        "population_size": int(values.shape[1]),
        "qualified_n": int(qualified.sum()),
        "candidate_count": int(candidate_indices.size),
        "candidate_fraction": safe_float(
            candidate_indices.size / values.shape[1]
        ),
        "split_qualified_n": split_counts,
        "group_eligibility": group_detail,
    })
    return rows, population, candidate_mask


def operation_overlaps(
    rows: list[dict[str, Any]],
    component: str,
) -> list[dict[str, Any]]:
    index_name = (
        "head_index" if component == "attention_head" else "neuron_index"
    )
    grouped: dict[tuple[Any, ...], dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows:
        key = (
            row["model"],
            row["stage"],
            row["role"],
            row["layer"],
        )
        grouped[key][row["operation"]].add(row[index_name])
    output = []
    for key, operation_sets in sorted(grouped.items()):
        for left_index, left in enumerate(OPERATIONS):
            for right in OPERATIONS[left_index + 1:]:
                left_set = operation_sets[left]
                right_set = operation_sets[right]
                union = left_set | right_set
                output.append({
                    "schema_version": "phase1008_operation_overlap.v1",
                    "phase": PHASE,
                    "model": key[0],
                    "component": component,
                    "stage": key[1],
                    "role": key[2],
                    "layer": int(key[3]),
                    "operation_left": left,
                    "operation_right": right,
                    "left_count": len(left_set),
                    "right_count": len(right_set),
                    "intersection_count": len(left_set & right_set),
                    "jaccard": safe_float(
                        len(left_set & right_set) / max(len(union), 1)
                    ),
                })
    return output


def discovery_head_selection(
    *,
    model_name: str,
    protocol: dict[str, Any],
    units: list[dict[str, Any]],
    values: np.ndarray,
    semantic_qualified: np.ndarray,
    attention_targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    peak_layer = int(
        protocol["model_targets"][model_name]["attention"]["peak_layers"][0]
    )
    target_index = next(
        index for index, target in enumerate(attention_targets)
        if int(target["layer"]) == peak_layer
    )
    peak_relative_depth = float(
        attention_targets[target_index]["relative_depth"]
    )
    discovery_mask = np.array([
        unit["split"] == "discovery" for unit in units
    ], dtype=np.bool_)
    output = []
    for operation in ANALYSIS_CONTRACT["causal_selection"]["operations"]:
        operation_index = OP_INDEX[operation]
        qualified = semantic_qualified[:, operation_index] & discovery_mask
        target_values = values[:, operation_index, target_index, :]
        membership = membership_matrix(target_values, qualified, 0.25)
        rates, qualified_n = subset_stats(
            membership, qualified, np.ones(len(units), dtype=np.bool_)
        )
        medians = np.array([
            np.median(target_values[qualified, index])
            if qualified.any()
            else 0.0
            for index in range(target_values.shape[1])
        ])
        order = sorted(
            range(target_values.shape[1]),
            key=lambda index: (rates[index], medians[index], -index),
            reverse=True,
        )
        selected = order[
            :ANALYSIS_CONTRACT["causal_selection"][
                "heads_per_operation"
            ]
        ]
        controls = sorted(
            [
                index for index in range(target_values.shape[1])
                if index not in selected
            ],
            key=lambda index: (rates[index], medians[index], index),
        )[:len(selected)]
        output.append({
            "schema_version": "phase1008_causal_head_selection.v1",
            "phase": PHASE,
            "model": model_name,
            "operation": operation,
            "selection_split": "discovery",
            "evaluation_split": "confirmation",
            "stage": "semantic0",
            "role": "decision_boundary",
            "layer": peak_layer,
            "relative_depth": peak_relative_depth,
            "qualified_n": int(qualified_n),
            "selected_heads": [int(value) for value in selected],
            "selected_discovery_recurrence": [
                safe_float(rates[index]) for index in selected
            ],
            "control_heads": [int(value) for value in controls],
            "control_discovery_recurrence": [
                safe_float(rates[index]) for index in controls
            ],
            "confirmation_data_used_for_selection": False,
            "selection_pass": bool(
                qualified_n >= 8
                and min(rates[index] for index in selected) >= 0.50
            ),
        })
    return output


def finalize_model(
    model_name: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    source = OUT_ROOT / "refinement_scan" / model_name
    destination = OUT_ROOT / "refinement_final" / model_name
    summary = read_json(source / "summary.json")
    if (
        summary["refinement_protocol_digest"]
        != protocol["preregistration_digest"]
    ):
        raise RuntimeError(f"{model_name}: stale refinement scan")
    if not summary["weight_reconstruction_audit"]["all_pass"]:
        raise RuntimeError(f"{model_name}: reconstruction audit failed")
    if not summary["dual_weight_rank_audit"]["all_pass"]:
        raise RuntimeError(f"{model_name}: dual-weight audit failed")
    units = read_jsonl(source / "units.jsonl")
    attention_targets = read_jsonl(source / "attention_targets.jsonl")
    mlp_targets = read_jsonl(source / "mlp_targets.jsonl")
    head_data = np.load(source / "head_observations.npz")
    neuron_data = np.load(source / "neuron_observations.npz")
    head_values = head_data["write_magnitude"]
    neuron_values = neuron_data["write_magnitude"]
    semantic_qualified = head_data["semantic_qualified"]
    if not np.array_equal(
        semantic_qualified, neuron_data["semantic_qualified"]
    ):
        raise RuntimeError(f"{model_name}: qualification drift")
    layer_count = int(read_json(
        OUT_ROOT / "scan" / model_name / "summary.json"
    )["model_info"]["n_layers"])
    for target in (*attention_targets, *mlp_targets):
        target["relative_depth"] = int(target["layer"]) / layer_count

    head_rows: list[dict[str, Any]] = []
    neuron_rows: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(attention_targets):
        for operation in OPERATIONS:
            operation_index = OP_INDEX[operation]
            rows, population, _ = candidate_rows(
                model_name=model_name,
                component="attention_head",
                target=target,
                operation=operation,
                values=head_values[
                    :, operation_index, target_index, :
                ],
                qualified=semantic_qualified[:, operation_index],
                units=units,
                contract=ANALYSIS_CONTRACT["head"],
            )
            head_rows.extend(rows)
            population_rows.append(population)
    for target_index, target in enumerate(mlp_targets):
        for operation in OPERATIONS:
            operation_index = OP_INDEX[operation]
            rows, population, _ = candidate_rows(
                model_name=model_name,
                component="mlp_neuron",
                target=target,
                operation=operation,
                values=neuron_values[
                    target_index, :, operation_index, :
                ],
                qualified=semantic_qualified[:, operation_index],
                units=units,
                contract=ANALYSIS_CONTRACT["neuron"],
            )
            neuron_rows.extend(rows)
            population_rows.append(population)
    overlap_rows = (
        operation_overlaps(head_rows, "attention_head")
        + operation_overlaps(neuron_rows, "mlp_neuron")
    )
    causal_selection = discovery_head_selection(
        model_name=model_name,
        protocol=protocol,
        units=units,
        values=head_values,
        semantic_qualified=semantic_qualified,
        attention_targets=attention_targets,
    )
    write_jsonl(destination / "head_candidates.jsonl", head_rows)
    write_jsonl(destination / "neuron_candidates.jsonl", neuron_rows)
    write_jsonl(destination / "population_summaries.jsonl", population_rows)
    write_jsonl(destination / "operation_overlaps.jsonl", overlap_rows)
    write_json(destination / "causal_selection.json", {
        "schema_version": "phase1008_causal_selection_bundle.v1",
        "phase": PHASE,
        "model": model_name,
        "selections": causal_selection,
    })
    result = {
        "schema_version": "phase1008_refinement_final_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "protocol_digest": protocol["preregistration_digest"],
        "analysis_contract_digest": digest(ANALYSIS_CONTRACT),
        "head_candidate_count": len(head_rows),
        "neuron_candidate_count": len(neuron_rows),
        "candidate_counts_by_operation": {
            component: {
                operation: sum(
                    row["operation"] == operation
                    for row in (
                        head_rows
                        if component == "attention_head"
                        else neuron_rows
                    )
                )
                for operation in OPERATIONS
            }
            for component in ("attention_head", "mlp_neuron")
        },
        "population_summary_count": len(population_rows),
        "operation_overlap_count": len(overlap_rows),
        "causal_selection": causal_selection,
        "causal_claims_from_observation": 0,
    }
    write_json(destination / "summary.json", result)
    return result


def main() -> None:
    protocol = read_json(OUT_ROOT / "refinement" / "protocol.json")
    results = {
        model_name: finalize_model(model_name, protocol)
        for model_name in MODELS
    }
    common_operations = [
        operation
        for operation in ("B", "Q")
        if all(
            results[model_name]["candidate_counts_by_operation"][
                "attention_head"
            ][operation] > 0
            for model_name in MODELS
        )
    ]
    causal_ready = all(
        selection["selection_pass"]
        for model_name in MODELS
        for selection in results[model_name]["causal_selection"]
    )
    summary = {
        "schema_version": "phase1008_refinement_cross_model_summary.v1",
        "phase": PHASE,
        "protocol_digest": protocol["preregistration_digest"],
        "analysis_contract": ANALYSIS_CONTRACT,
        "analysis_contract_digest": digest(ANALYSIS_CONTRACT),
        "models": results,
        "coordinate_alignment_forbidden": True,
        "common_attention_operations": common_operations,
        "automatic_next_action": (
            "heldout_head_causal_sampling_warranted"
            if common_operations and causal_ready
            else "continue_descriptive_atlas_without_local_causal_sampling"
        ),
        "interpretation": (
            "Repeated heads and neurons are descriptive contributors. "
            "Cross-model agreement is functional and relative-depth based; "
            "physical IDs are never aligned."
        ),
    }
    write_json(OUT_ROOT / "refinement_final" / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
