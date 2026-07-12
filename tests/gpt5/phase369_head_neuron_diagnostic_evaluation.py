#!/usr/bin/env python3
"""Evaluate exploratory head/neuron topology without reopening Phase369 calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from phase369_blind_future_and_crossmodel import (
    DEPTH_POINTS, MODELS, PAIR_SPECS, nearest_indices, resample_case, rms_cdist,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
RAW_FEATURES = BASE / "raw_relation_features/private/cases"
TOPOLOGY = BASE / "head_neuron_topology_diagnostic"
OUT = BASE / "head_neuron_topology_diagnostic_evaluation"
SHARD_COUNTS = (8, 32, 128)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resample_records(records: list[dict[str, Any]], values: torch.Tensor, include_source: bool) -> tuple[list[tuple], torch.Tensor]:
    if include_source:
        keys = sorted({
            (int(row["generation_time"]), row["source_role"], row["receiver_role"])
            for row in records
        })
        key_for = lambda row: (int(row["generation_time"]), row["source_role"], row["receiver_role"])
    else:
        keys = sorted({(int(row["generation_time"]), row["receiver_role"]) for row in records})
        key_for = lambda row: (int(row["generation_time"]), row["receiver_role"])
    indices = {key: [] for key in keys}
    for index, row in enumerate(records):
        indices[key_for(row)].append(index)
    output = []
    for key in keys:
        selected = values[indices[key]].float().transpose(0, 1).unsqueeze(0)
        output.append(F.interpolate(selected, size=DEPTH_POINTS, mode="linear", align_corners=True)[0].transpose(0, 1))
    return keys, torch.stack(output)


def minimax_neighbor(distances: list[torch.Tensor], groups: list[str]) -> torch.Tensor:
    size = distances[0].shape[0]
    ranks = []
    for distance in distances:
        masked = distance.clone()
        for left, group in enumerate(groups):
            for right, other in enumerate(groups):
                if group == other:
                    masked[left, right] = float("inf")
        order = masked.argsort(dim=1)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(size).unsqueeze(0).expand(size, -1))
        ranks.append(rank)
    stacked = torch.stack(ranks)
    worst = stacked.max(dim=0).values
    rank_sum = stacked.sum(dim=0)
    score = worst * (len(distances) * size + 1) + rank_sum
    for left, group in enumerate(groups):
        for right, other in enumerate(groups):
            if group == other:
                score[left, right] = torch.iinfo(score.dtype).max
    return score.argmin(dim=1)


def retrieval_from_components(
    source_components: list[torch.Tensor],
    target_components: list[torch.Tensor],
    source_hashes: list[str],
    target_hashes: list[str],
) -> dict[str, float]:
    distances = []
    for left, right in zip(source_components, target_components, strict=True):
        left = left - left.mean(dim=0, keepdim=True)
        right = right - right.mean(dim=0, keepdim=True)
        distances.append(rms_cdist(left, right))
    size = len(source_hashes)
    ranks_by_component = []
    for distance in distances:
        order = distance.argsort(dim=1)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(size).unsqueeze(0).expand(size, -1))
        ranks_by_component.append(rank)
    stacked = torch.stack(ranks_by_component)
    score = stacked.max(dim=0).values * (len(distances) * size + 1) + stacked.sum(dim=0)
    order = score.argsort(dim=1)
    target_by_hash = {value: index for index, value in enumerate(target_hashes)}
    correct = torch.tensor([target_by_hash[value] for value in source_hashes])
    ranks = torch.empty(size, dtype=torch.long)
    for index in range(size):
        ranks[index] = int((order[index] == correct[index]).nonzero(as_tuple=False)[0, 0]) + 1
    matched_distances = [distance[torch.arange(size), correct] for distance in distances]
    ratios = []
    for distance, matched in zip(distances, matched_distances, strict=True):
        mask = torch.ones_like(distance, dtype=torch.bool)
        mask[torch.arange(size), correct] = False
        ratios.append(float(matched.mean() / distance[mask].mean()))
    return {
        "top1_retrieval_rate": round(float((ranks <= 1).float().mean()), 8),
        "top5_retrieval_rate": round(float((ranks <= 5).float().mean()), 8),
        "mean_rank": round(float(ranks.float().mean()), 8),
        "worst_component_matched_separation_ratio": round(max(ratios), 8),
        "component_matched_separation_ratios": [round(value, 8) for value in ratios],
    }


def main() -> None:
    gate = read_json(TOPOLOGY / "phase369_head_neuron_diagnostic_gate.json")
    if not gate["frozen_before_diagnostic_evaluation"]:
        raise RuntimeError("Diagnostic gate is not frozen")
    raw_summary = read_json(BASE / "blind_future_and_crossmodel/phase369_blind_future_and_crossmodel_summary.json")
    raw_model_summary = {row["model"]: row for row in raw_summary["future_prediction"]}
    raw_pair_summary = {
        (row["left_model"], row["right_model"]): row for row in raw_summary["cross_model"]
    }
    cases_by_model: dict[str, list[dict[str, Any]]] = {}
    topology_by_model: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    model_rows = []
    for model in MODELS:
        raw_paths = sorted((RAW_FEATURES / model).glob("*.pt"))
        cases = [resample_case(torch.load(path, map_location="cpu", weights_only=False)) for path in raw_paths]
        cases_by_model[model] = cases
        topology_by_case = {}
        for path in sorted((TOPOLOGY / "private/cases" / model).glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            per_resolution = {}
            for shard_count in SHARD_COUNTS:
                start, end = payload["shard_slices"][str(shard_count)]
                head_keys, head = resample_records(
                    payload["head_records"], payload["head_hash_topology"][:, start:end], True,
                )
                neuron_keys, neuron = resample_records(
                    payload["neuron_records"], payload["neuron_hash_topology"][:, start:end], False,
                )
                per_resolution[str(shard_count)] = {
                    "head_keys": head_keys, "head": head,
                    "neuron_keys": neuron_keys, "neuron": neuron,
                }
            topology_by_case[payload["anonymous_case_id"]] = per_resolution
        topology_by_model[model] = topology_by_case
        groups = [case["group_id"] for case in cases]
        keys = cases[0]["keys"]
        raw_t0 = torch.tensor([index for index, key in enumerate(keys) if key[0] == 0])
        raw_t1 = torch.tensor([index for index, key in enumerate(keys) if key[0] == 1])
        raw_prefix = torch.stack([case["raw"].index_select(0, raw_t0).flatten() for case in cases])
        raw_future = torch.stack([case["raw"].index_select(0, raw_t1).flatten() for case in cases])
        vocab_future = torch.stack([case["vocab"][1] for case in cases])
        raw_neighbor = nearest_indices(rms_cdist(raw_prefix, raw_prefix), groups)
        future_distance = rms_cdist(raw_future, raw_future)
        vocab_distance = rms_cdist(vocab_future, vocab_future)
        raw_flow_error = future_distance[torch.arange(len(cases)), raw_neighbor]
        raw_vocab_error = vocab_distance[torch.arange(len(cases)), raw_neighbor]
        resolutions = []
        for shard_count in SHARD_COUNTS:
            key = str(shard_count)
            head_prefix_rows = []
            neuron_prefix_rows = []
            for case in cases:
                topo = topology_by_case[case["case_id"]][key]
                head_indices = torch.tensor([i for i, item in enumerate(topo["head_keys"]) if item[0] == 0])
                neuron_indices = torch.tensor([i for i, item in enumerate(topo["neuron_keys"]) if item[0] == 0])
                head_prefix_rows.append(topo["head"].index_select(0, head_indices).flatten())
                neuron_prefix_rows.append(topo["neuron"].index_select(0, neuron_indices).flatten())
            head_prefix = torch.stack(head_prefix_rows)
            neuron_prefix = torch.stack(neuron_prefix_rows)
            composite_neighbor = minimax_neighbor(
                [
                    rms_cdist(raw_prefix, raw_prefix),
                    rms_cdist(head_prefix, head_prefix),
                    rms_cdist(neuron_prefix, neuron_prefix),
                ],
                groups,
            )
            flow_error = future_distance[torch.arange(len(cases)), composite_neighbor]
            vocab_error = vocab_distance[torch.arange(len(cases)), composite_neighbor]
            components = {
                "composite_mean_future_flow_error_below_raw_relation": float(flow_error.mean()) < float(raw_flow_error.mean()),
                "composite_case_win_fraction_over_raw_above_half": float((flow_error < raw_flow_error).float().mean()) > 0.5,
                "composite_mean_vocab_error_not_above_raw_relation": float(vocab_error.mean()) <= float(raw_vocab_error.mean()),
            }
            resolutions.append({
                "shard_count": shard_count,
                "composite_mean_future_flow_error": round(float(flow_error.mean()), 8),
                "raw_relation_mean_future_flow_error": round(float(raw_flow_error.mean()), 8),
                "composite_case_win_fraction_over_raw": round(float((flow_error < raw_flow_error).float().mean()), 8),
                "composite_mean_vocab_error": round(float(vocab_error.mean()), 8),
                "raw_relation_mean_vocab_error": round(float(raw_vocab_error.mean()), 8),
                "component_gates": components,
                "all_future_components_pass": all(components.values()),
            })
        model_rows.append({"model": model, "case_count": len(cases), "resolutions": resolutions})
        print(f"[{model}] topology future diagnostic complete", flush=True)

    pair_rows = []
    for left_model, right_model, architecture_relation in PAIR_SPECS:
        left_cases = cases_by_model[left_model]
        right_cases = cases_by_model[right_model]
        resolution_rows = []
        for shard_count in SHARD_COUNTS:
            key = str(shard_count)

            def components(model: str, cases: list[dict[str, Any]]) -> list[torch.Tensor]:
                raw = torch.stack([case["raw"].flatten() for case in cases])
                heads = torch.stack([
                    topology_by_model[model][case["case_id"]][key]["head"].flatten()
                    for case in cases
                ])
                neurons = torch.stack([
                    topology_by_model[model][case["case_id"]][key]["neuron"].flatten()
                    for case in cases
                ])
                return [raw, heads, neurons]

            metrics = retrieval_from_components(
                components(left_model, left_cases),
                components(right_model, right_cases),
                [case["prompt_hash"] for case in left_cases],
                [case["prompt_hash"] for case in right_cases],
            )
            raw_metrics = raw_pair_summary[(left_model, right_model)]["raw_residual"]
            components_gate = {
                "composite_worst_separation_ratio_below_raw_relation": metrics["worst_component_matched_separation_ratio"] < raw_metrics["matched_separation_ratio"],
                "composite_top5_rate_above_raw_relation": metrics["top5_retrieval_rate"] > raw_metrics["top5_retrieval_rate"],
                "composite_top5_rate_above_random": metrics["top5_retrieval_rate"] > 5 / len(right_cases),
            }
            resolution_rows.append({
                "shard_count": shard_count,
                "composite": metrics,
                "raw_relation": raw_metrics,
                "component_gates": components_gate,
                "all_cross_model_components_pass": all(components_gate.values()),
            })
        pair_rows.append({
            "left_model": left_model,
            "right_model": right_model,
            "architecture_relation": architecture_relation,
            "resolutions": resolution_rows,
        })
        print(f"[{left_model}->{right_model}] topology cross-model diagnostic complete", flush=True)

    model_pass = {
        (row["model"], item["shard_count"]): item["all_future_components_pass"]
        for row in model_rows for item in row["resolutions"]
    }
    eligible_pairs = []
    for pair in pair_rows:
        if pair["architecture_relation"] != "heterogeneous":
            continue
        passing_resolutions = [
            item["shard_count"] for item in pair["resolutions"]
            if item["all_cross_model_components_pass"]
            and model_pass[(pair["left_model"], item["shard_count"])]
            and model_pass[(pair["right_model"], item["shard_count"])]
        ]
        if len(passing_resolutions) >= 2:
            eligible_pairs.append({
                "pair": f"{pair['left_model']}->{pair['right_model']}",
                "passing_resolutions": passing_resolutions,
            })
    summary = {
        "schema_version": "46.4.0",
        "phase_id": "Phase369-Diagnostic",
        "objective": "test_whether_fixed_hash_head_and_neuron_topology_is_a_promising_missing_state_for_a_new_independent_cycle",
        "denominator": {
            "model_count": 3,
            "case_count": 336,
            "hash_resolution_count": 3,
            "hash_seed_count": 3,
            "hash_seeds_are_independent_replications": False,
        },
        "future_prediction": model_rows,
        "cross_model": pair_rows,
        "new_cycle_evidence": {
            "eligible_heterogeneous_pair_count": len(eligible_pairs),
            "eligible_heterogeneous_pairs": eligible_pairs,
        },
        "claim_boundary": {
            "exploratory_same_discovery_cycle": True,
            "rescues_phase369": False,
            "calibration_executed": False,
            "physical_holdout_opened": False,
            "semantic_labels_used": False,
            "single_neuron_causal_mechanism_found": False,
        },
        "authorization": {
            "new_independent_topology_cycle": len(eligible_pairs) > 0,
            "phase369_calibration": False,
            "physical_holdout": False,
        },
        "next_decision": (
            "freeze_new_independent_topology_cycle"
            if eligible_pairs else
            "do_not_expand_hash_topology_without_a_new_path_object"
        ),
    }
    write_json(OUT / "phase369_head_neuron_diagnostic_evaluation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
