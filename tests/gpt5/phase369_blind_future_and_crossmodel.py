#!/usr/bin/env python3
"""Evaluate Phase369 raw relations against low-resolution and strong blind controls."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
FEATURES = BASE / "raw_relation_features"
GATE_PATH = BASE / "discovery_gate_freeze/phase369_discovery_gate_freeze.json"
OUT = BASE / "blind_future_and_crossmodel"
MODELS = ("qwen3", "glm4", "deepseek7b")
PAIR_SPECS = (
    ("qwen3", "glm4", "heterogeneous"),
    ("glm4", "deepseek7b", "heterogeneous"),
    ("qwen3", "deepseek7b", "shared_qwen_architecture_family"),
)
DEPTH_POINTS = 32
LOW_LOG_INDICES = (0, 2, 4, 5, 8)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def bounded_low(values: torch.Tensor) -> torch.Tensor:
    result = values.clone()
    for index in LOW_LOG_INDICES:
        value = result[..., index]
        result[..., index] = value.sign() * torch.log1p(value.abs())
    return result


def resample_case(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload["records"]
    keys = sorted({
        (int(row["generation_time"]), row["source_role"], row["receiver_role"])
        for row in records
    })
    expected_key_counts = {0: 12, 1: 20, 2: 20}
    if {time: sum(key[0] == time for key in keys) for time in range(3)} != expected_key_counts:
        raise RuntimeError(f"Unexpected route topology for {payload['anonymous_case_id']}")
    index_by_key: dict[tuple[int, str, str], list[int]] = {key: [] for key in keys}
    for index, row in enumerate(records):
        index_by_key[(int(row["generation_time"]), row["source_role"], row["receiver_role"])].append(index)

    def build(feature_name: str) -> torch.Tensor:
        feature = payload[feature_name].float()
        if feature_name == "low_descriptor_features":
            feature = bounded_low(feature)
        rows = []
        for key in keys:
            selected = feature[index_by_key[key]]
            selected = selected.transpose(0, 1).unsqueeze(0)
            interpolated = F.interpolate(selected, size=DEPTH_POINTS, mode="linear", align_corners=True)
            rows.append(interpolated[0].transpose(0, 1))
        return torch.stack(rows)

    return {
        "case_id": payload["anonymous_case_id"],
        "group_id": payload["anonymous_group_id"],
        "parallel_group_id": payload["anonymous_parallel_group_id"],
        "prompt_hash": payload["parallel_prompt_hash"],
        "keys": keys,
        "low": build("low_descriptor_features"),
        "raw": build("raw_relation_features"),
        "vocab": payload["vocab_state_features"].float(),
    }


def rms_cdist(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.cdist(left, right) / math.sqrt(left.shape[1])


def deterministic_permutation(size: int, seed_text: str) -> torch.Tensor:
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16) % (2**63 - 1)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(size, generator=generator)


def nearest_indices(distance: torch.Tensor, group_ids: list[str]) -> torch.Tensor:
    masked = distance.clone()
    for left, group in enumerate(group_ids):
        for right, other in enumerate(group_ids):
            if group == other:
                masked[left, right] = float("inf")
    return masked.argmin(dim=1)


def deterministic_random_indices(group_ids: list[str], case_ids: list[str]) -> torch.Tensor:
    values = []
    for index, (group, case_id) in enumerate(zip(group_ids, case_ids, strict=True)):
        allowed = [item for item, other in enumerate(group_ids) if other != group]
        position = int(hashlib.sha256(f"random:{case_id}".encode()).hexdigest()[:16], 16) % len(allowed)
        values.append(allowed[position])
    return torch.tensor(values, dtype=torch.long)


def selected_errors(distance: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return distance[torch.arange(distance.shape[0]), indices]


def model_evaluation(model: str, cases: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_tensors = torch.stack([case["raw"] for case in cases])
    low_tensors = torch.stack([case["low"] for case in cases])
    keys = cases[0]["keys"]
    if any(case["keys"] != keys for case in cases):
        raise RuntimeError(f"Route keys differ within {model}")
    t0 = torch.tensor([index for index, key in enumerate(keys) if key[0] == 0])
    t1 = torch.tensor([index for index, key in enumerate(keys) if key[0] == 1])
    raw_prefix_tensor = raw_tensors.index_select(1, t0)
    low_prefix_tensor = low_tensors.index_select(1, t0)
    raw_prefix = raw_prefix_tensor.flatten(1)
    low_prefix = low_prefix_tensor.flatten(1)
    raw_future = raw_tensors.index_select(1, t1).flatten(1)
    vocab_future = torch.stack([case["vocab"][1] for case in cases])
    groups = [case["group_id"] for case in cases]
    case_ids = [case["case_id"] for case in cases]

    raw_neighbor = nearest_indices(rms_cdist(raw_prefix, raw_prefix), groups)
    low_neighbor = nearest_indices(rms_cdist(low_prefix, low_prefix), groups)
    random_neighbor = deterministic_random_indices(groups, case_ids)

    shuffled = raw_prefix_tensor.clone()
    role_permuted = raw_prefix_tensor.clone()
    for index, case_id in enumerate(case_ids):
        depth_perm = deterministic_permutation(DEPTH_POINTS, f"time:{case_id}")
        role_perm = deterministic_permutation(raw_prefix_tensor.shape[1], f"role:{case_id}")
        shuffled[index] = shuffled[index, :, depth_perm, :]
        role_permuted[index] = role_permuted[index, role_perm]
    time_neighbor = nearest_indices(rms_cdist(shuffled.flatten(1), shuffled.flatten(1)), groups)
    role_neighbor = nearest_indices(rms_cdist(role_permuted.flatten(1), role_permuted.flatten(1)), groups)
    energy_prefix = raw_prefix_tensor[..., -6:].flatten(1)
    energy_neighbor = nearest_indices(rms_cdist(energy_prefix, energy_prefix), groups)

    future_distance = rms_cdist(raw_future, raw_future)
    vocab_distance = rms_cdist(vocab_future, vocab_future)
    methods = {
        "raw_relation": raw_neighbor,
        "low_ten_descriptor": low_neighbor,
        "random_flow": random_neighbor,
        "time_shuffle": time_neighbor,
        "role_permutation": role_neighbor,
        "equal_energy_wrong_flow": energy_neighbor,
    }
    future_errors = {name: selected_errors(future_distance, indices) for name, indices in methods.items()}
    vocab_errors = {name: selected_errors(vocab_distance, indices) for name, indices in methods.items()}
    backbone = raw_future.mean(dim=0, keepdim=True)
    backbone_flow = torch.linalg.vector_norm(raw_future - backbone, dim=1) / math.sqrt(raw_future.shape[1])
    backbone_vocab_value = vocab_future.mean(dim=0, keepdim=True)
    backbone_vocab = torch.linalg.vector_norm(vocab_future - backbone_vocab_value, dim=1) / math.sqrt(vocab_future.shape[1])
    low_flow = future_errors["low_ten_descriptor"]
    raw_flow = future_errors["raw_relation"]
    low_vocab = vocab_errors["low_ten_descriptor"]
    raw_vocab = vocab_errors["raw_relation"]
    component_gates = {
        "raw_mean_future_flow_error_strictly_below_low_ten_descriptor": float(raw_flow.mean()) < float(low_flow.mean()),
        "raw_case_win_fraction_over_low_strictly_above_half": float((raw_flow < low_flow).float().mean()) > 0.5,
        "raw_mean_vocab_error_not_above_low": float(raw_vocab.mean()) <= float(low_vocab.mean()),
        "raw_mean_future_flow_error_below_random": float(raw_flow.mean()) < float(future_errors["random_flow"].mean()),
        "raw_mean_future_flow_error_below_time_shuffle": float(raw_flow.mean()) < float(future_errors["time_shuffle"].mean()),
        "raw_mean_future_flow_error_below_role_permutation": float(raw_flow.mean()) < float(future_errors["role_permutation"].mean()),
        "raw_mean_future_flow_error_below_equal_energy_wrong_flow": float(raw_flow.mean()) < float(future_errors["equal_energy_wrong_flow"].mean()),
        "raw_mean_future_flow_error_below_public_backbone": float(raw_flow.mean()) < float(backbone_flow.mean()),
    }
    rows = []
    for index, case in enumerate(cases):
        rows.append({
            "anonymous_case_id": case["case_id"],
            "anonymous_group_id": case["group_id"],
            "model": model,
            "future_flow_errors": {
                **{name: round(float(values[index]), 8) for name, values in future_errors.items()},
                "public_backbone": round(float(backbone_flow[index]), 8),
            },
            "vocab_errors": {
                **{name: round(float(values[index]), 8) for name, values in vocab_errors.items()},
                "public_backbone": round(float(backbone_vocab[index]), 8),
            },
            "raw_neighbor_case_id": cases[int(raw_neighbor[index])]["case_id"],
            "low_neighbor_case_id": cases[int(low_neighbor[index])]["case_id"],
            "semantic_labels_used": False,
        })
    summary = {
        "model": model,
        "case_count": len(cases),
        "independent_group_count": len(set(groups)),
        "mean_future_flow_errors": {
            **{name: round(float(values.mean()), 8) for name, values in future_errors.items()},
            "public_backbone": round(float(backbone_flow.mean()), 8),
        },
        "mean_vocab_errors": {
            **{name: round(float(values.mean()), 8) for name, values in vocab_errors.items()},
            "public_backbone": round(float(backbone_vocab.mean()), 8),
        },
        "raw_case_win_fraction_over_low": round(float((raw_flow < low_flow).float().mean()), 8),
        "component_gates": component_gates,
        "all_future_prediction_components_pass": all(component_gates.values()),
    }
    return summary, rows


def retrieval_metrics(source: list[dict[str, Any]], target: list[dict[str, Any]], feature: str) -> dict[str, float]:
    source_tensor = torch.stack([case[feature] for case in source]).flatten(1)
    target_tensor = torch.stack([case[feature] for case in target]).flatten(1)
    source_tensor = source_tensor - source_tensor.mean(dim=0, keepdim=True)
    target_tensor = target_tensor - target_tensor.mean(dim=0, keepdim=True)
    distance = rms_cdist(source_tensor, target_tensor)
    target_by_hash = {case["prompt_hash"]: index for index, case in enumerate(target)}
    correct = torch.tensor([target_by_hash[case["prompt_hash"]] for case in source], dtype=torch.long)
    matched = distance[torch.arange(len(source)), correct]
    wrong_mask = torch.ones_like(distance, dtype=torch.bool)
    wrong_mask[torch.arange(len(source)), correct] = False
    wrong_mean = distance[wrong_mask].mean()
    order = distance.argsort(dim=1)
    ranks = torch.empty(len(source), dtype=torch.long)
    for index in range(len(source)):
        ranks[index] = int((order[index] == correct[index]).nonzero(as_tuple=False)[0, 0]) + 1
    return {
        "matched_mean_distance": round(float(matched.mean()), 8),
        "wrong_mean_distance": round(float(wrong_mean), 8),
        "matched_separation_ratio": round(float(matched.mean() / wrong_mean), 8),
        "top1_retrieval_rate": round(float((ranks <= 1).float().mean()), 8),
        "top5_retrieval_rate": round(float((ranks <= 5).float().mean()), 8),
        "mean_rank": round(float(ranks.float().mean()), 8),
    }


def main() -> None:
    gate = read_json(GATE_PATH)
    if not gate["frozen_before_discovery_evaluation"]:
        raise RuntimeError("Discovery gate was not frozen")
    cases_by_model: dict[str, list[dict[str, Any]]] = {}
    model_summaries = []
    all_case_rows = []
    for model in MODELS:
        paths = sorted((FEATURES / "private/cases" / model).glob("*.pt"))
        cases = [resample_case(torch.load(path, map_location="cpu", weights_only=False)) for path in paths]
        cases_by_model[model] = cases
        summary, rows = model_evaluation(model, cases)
        model_summaries.append(summary)
        all_case_rows.extend(rows)
        print(f"[{model}] blind future evaluation complete", flush=True)
    write_jsonl(OUT / "private/phase369_blind_future_case_rows.jsonl", all_case_rows)

    pair_rows = []
    for left, right, architecture_relation in PAIR_SPECS:
        raw_metrics = retrieval_metrics(cases_by_model[left], cases_by_model[right], "raw")
        low_metrics = retrieval_metrics(cases_by_model[left], cases_by_model[right], "low")
        random_rate = 5 / len(cases_by_model[right])
        components = {
            "raw_residual_matched_separation_ratio_below_one": raw_metrics["matched_separation_ratio"] < 1.0,
            "raw_residual_matched_separation_ratio_below_low_residual_ratio": raw_metrics["matched_separation_ratio"] < low_metrics["matched_separation_ratio"],
            "raw_residual_top5_retrieval_rate_above_low_residual": raw_metrics["top5_retrieval_rate"] > low_metrics["top5_retrieval_rate"],
            "raw_residual_top5_retrieval_rate_above_random_rate": raw_metrics["top5_retrieval_rate"] > random_rate,
        }
        pair_rows.append({
            "left_model": left,
            "right_model": right,
            "architecture_relation": architecture_relation,
            "case_count_each": len(cases_by_model[left]),
            "raw_residual": raw_metrics,
            "low_residual": low_metrics,
            "random_top5_rate": round(random_rate, 8),
            "component_gates": components,
            "all_cross_model_components_pass": all(components.values()),
            "unrestricted_coordinate_rotation_fitted": False,
        })
        print(f"[{left}->{right}] cross-model retrieval complete", flush=True)

    model_pass = {row["model"]: row["all_future_prediction_components_pass"] for row in model_summaries}
    level_2_pairs = [
        row for row in pair_rows
        if row["architecture_relation"] == "heterogeneous"
        and row["all_cross_model_components_pass"]
        and model_pass[row["left_model"]]
        and model_pass[row["right_model"]]
    ]
    architecture_family_only = any(
        row["architecture_relation"] == "shared_qwen_architecture_family"
        and row["all_cross_model_components_pass"]
        and model_pass[row["left_model"]]
        and model_pass[row["right_model"]]
        for row in pair_rows
    )
    level_3 = all(model_pass.values()) and all(row["all_cross_model_components_pass"] for row in pair_rows)
    summary = {
        "schema_version": "46.2.0",
        "phase_id": "Phase369",
        "created_at": now(),
        "objective": "test_whether_coordinate_invariant_raw_relations_add_blind_future_and_cross_model_information_beyond_the_old_ten_descriptors",
        "denominator": {
            "model_count": 3,
            "case_count": sum(len(cases) for cases in cases_by_model.values()),
            "case_count_per_model": 112,
            "independent_group_count_per_model": 28,
            "cross_model_pair_count": 3,
            "depth_resample_points": DEPTH_POINTS,
        },
        "future_prediction": model_summaries,
        "cross_model": pair_rows,
        "evidence": {
            "level_1_model_count": sum(model_pass.values()),
            "level_1_models": [model for model, passed in model_pass.items() if passed],
            "level_2_heterogeneous_pair_count": len(level_2_pairs),
            "level_2_heterogeneous_pairs": [f"{row['left_model']}->{row['right_model']}" for row in level_2_pairs],
            "qwen_deepseek_architecture_family_only_pass": architecture_family_only,
            "level_3_all_three_models": level_3,
        },
        "controls": {
            "low_ten_descriptor": True,
            "random_flow": True,
            "time_shuffle": True,
            "role_permutation": True,
            "equal_energy_wrong_flow": True,
            "public_backbone": True,
        },
        "claim_boundary": {
            "semantic_labels_used": False,
            "target_rank_or_margin_used": False,
            "head_or_neuron_topology_tested": False,
            "calibration_executed": False,
            "physical_holdout_opened": False,
            "language_mechanism_discovered": False,
        },
        "authorization": {
            "head_and_neuron_topology_refinement": len(level_2_pairs) > 0,
            "fresh_calibration_raw_collection": False,
            "physical_holdout": False,
        },
        "next_decision": (
            "derive_fixed_hash_head_and_neuron_topology_then_freeze_candidates"
            if len(level_2_pairs) > 0 else
            "stop_phase369_before_head_neuron_calibration_and_physical_holdout"
        ),
    }
    write_json(OUT / "phase369_blind_future_and_crossmodel_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
