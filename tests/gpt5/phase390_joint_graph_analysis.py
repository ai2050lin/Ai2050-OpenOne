#!/usr/bin/env python3
"""Analyze Phase390 graph-legal joint writes without a learned predictor or basis."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase365_dynamic_bundle_extraction import load_weight  # noqa: E402
from phase390_role_mapping import REGISTERED_ROLES, semantic_role_indices  # noqa: E402


PHASE_ROOT = ROOT / "tests/gpt5/result/phase390_joint_formation_graph"
COLLECTION = PHASE_ROOT / "collection"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "calibration", "physical_holdout")
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
RECEIVERS = ("query_integrated", "pre_decision")
SEMANTIC_ROLES = REGISTERED_ROLES[:-1]
OTHER_ROLE = "other_causal_prefix"
ANCHOR_COUNT = 8
WINDOW_LENGTHS = (1, 2, 4)
MODEL_LAYER_COUNTS = {"qwen3": 36, "glm4": 40, "deepseek7b": 28}
SUPPORT_REQUIRED = {"discovery": 8, "calibration": 4, "physical_holdout": 4}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(left, right).item()) / denominator


def minimum_alignment(
    left_x: torch.Tensor,
    left_y: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
) -> float:
    return min(cosine(left_x, target_x), cosine(left_y, target_y))


def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(left.float() - right.float()).item()) / max(
        float(torch.linalg.vector_norm(left.float()).item()), 1e-12
    )


def repeat_kv(values: torch.Tensor, head_count: int) -> torch.Tensor:
    if values.shape[1] == head_count:
        return values
    if head_count % values.shape[1]:
        raise RuntimeError("Attention head count is not divisible by K/V head count")
    return values.repeat_interleave(head_count // values.shape[1], dim=1)


def frame_for(payload: dict[str, Any], coordinate: str) -> tuple[dict[str, Any], int]:
    for frame in payload["attention"]["frames"]:
        if coordinate in frame["coordinate_names"]:
            return frame, frame["coordinate_names"].index(coordinate)
    raise RuntimeError(f"Missing coordinate {coordinate}")


def layer_path(split: str, model: str, case_id: str, layer: int) -> Path:
    return (
        COLLECTION
        / split
        / "private/models"
        / model
        / case_id
        / f"layer_{layer:03d}.pt"
    )


def contrast(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left.float() - right.float()


def window_lattice(layer_count: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for anchor_index in range(ANCHOR_COUNT):
        fraction = anchor_index / (ANCHOR_COUNT - 1)
        for width in WINDOW_LENGTHS:
            start = round(fraction * (layer_count - width))
            key = (start, width)
            if key in seen:
                continue
            seen.add(key)
            windows.append(
                {
                    "anchor_index": anchor_index,
                    "anchor_fraction": fraction,
                    "window_length": width,
                    "start_layer": start,
                    "end_layer": start + width - 1,
                }
            )
    return windows


def case_groups(split: str, model: str) -> dict[str, dict[str, dict[str, Any]]]:
    cases = [
        row
        for row in read_jsonl(
            PHASE_ROOT / f"protocol/private/phase390_{split}_cases.jsonl"
        )
        if row["private_execution_model"] == model
    ]
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        grouped[case["phase390_public_parallel_group_id"]][
            case["contrast_condition"]
        ] = case
    expected_groups = 12 if split == "discovery" else 6
    if len(grouped) != expected_groups:
        raise RuntimeError(
            f"Expected {expected_groups} Phase390 {split}/{model} groups, got {len(grouped)}"
        )
    if any(set(conditions) != set(CONDITIONS) for conditions in grouped.values()):
        raise RuntimeError(f"Incomplete Phase390 four-condition group for {model}/{split}")
    return dict(sorted(grouped.items()))


def terminal_targets(
    split: str,
    model: str,
    groups: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, torch.Tensor]]:
    final_layer = MODEL_LAYER_COUNTS[model] - 1
    targets: dict[str, dict[str, torch.Tensor]] = {}
    for group_id, conditions in groups.items():
        states: dict[str, dict[str, torch.Tensor]] = {}
        for condition, case in conditions.items():
            payload = torch.load(
                layer_path(split, model, case["blind_case_id"], final_layer),
                map_location="cpu",
                weights_only=False,
            )
            states[condition] = {
                coordinate: payload["component_vectors"]["layer_output"][
                    0, payload["coordinate_names"].index(coordinate)
                ].float()
                for coordinate in ("target_encoded", "post_decision_next_token")
            }
        targets[group_id] = {
            "terminal_x": contrast(
                states["A_operation_lex_x"]["target_encoded"],
                states["B_control_lex_x"]["target_encoded"],
            ),
            "terminal_y": contrast(
                states["C_operation_lex_y"]["target_encoded"],
                states["D_control_lex_y"]["target_encoded"],
            ),
            "wrong_time_x": contrast(
                states["A_operation_lex_x"]["post_decision_next_token"],
                states["B_control_lex_x"]["post_decision_next_token"],
            ),
            "wrong_time_y": contrast(
                states["C_operation_lex_y"]["post_decision_next_token"],
                states["D_control_lex_y"]["post_decision_next_token"],
            ),
        }
    return targets


def event_vectors(
    payload: dict[str, Any],
    coordinate: str,
    case: dict[str, Any],
    tokenizer: Any,
    o_weight: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float, dict[str, Any]]:
    coordinate_index = payload["coordinate_names"].index(coordinate)
    frame, receiver_index = frame_for(payload, coordinate)
    probabilities = frame["probabilities_receivers_all_sources"].float()[
        0, :, receiver_index
    ].to(device)
    values = repeat_kv(
        frame["value_states_all_sources"].float().to(device),
        int(frame["head_count"]),
    )[0]
    receiver_position = int(frame["global_positions"][receiver_index])
    sequence_length = int(values.shape[1])
    partition, role_audit = semantic_role_indices(
        tokenizer,
        case,
        receiver_position,
        total_sequence_length=sequence_length,
    )
    if role_audit["missing_fragments"] or not role_audit["partition_conserved"]:
        raise RuntimeError(
            f"Role mapping failed for {case['blind_case_id']}/{coordinate}: {role_audit}"
        )
    weighted = probabilities[:, :, None] * values
    head_count = int(frame["head_count"])
    head_dim = int(frame["head_dim"])
    blocks = o_weight.view(o_weight.shape[0], head_count, head_dim)
    role_head_space = []
    for role in REGISTERED_ROLES:
        indices = partition[role]
        if indices:
            index = torch.tensor(indices, dtype=torch.long, device=device)
            role_head_space.append(weighted.index_select(1, index).sum(dim=1))
        else:
            role_head_space.append(torch.zeros_like(weighted[:, 0]))
    role_head_space_tensor = torch.stack(role_head_space)
    projected = torch.einsum("rhd,ohd->rho", role_head_space_tensor, blocks)
    role_vectors = projected.sum(dim=1)
    semantic_head_vectors = projected[: len(SEMANTIC_ROLES)].sum(dim=0)
    semantic_joint = semantic_head_vectors.sum(dim=0)
    other = role_vectors[-1]
    full_replay = role_vectors.sum(dim=0)
    recorded_attention = payload["component_vectors"]["attention_output"][
        0, coordinate_index
    ].float().to(device)
    replay_error = relative_error(recorded_attention, full_replay)
    result = {
        "semantic_joint": semantic_joint.detach().cpu(),
        "other_prefix": other.detach().cpu(),
        "roles": role_vectors[: len(SEMANTIC_ROLES)].detach().cpu(),
        "heads": semantic_head_vectors.detach().cpu(),
        "block": (
            recorded_attention
            + payload["component_vectors"]["mlp_output"][0, coordinate_index]
            .float()
            .to(device)
        )
        .detach()
        .cpu(),
    }
    del weighted, role_head_space_tensor, projected, role_vectors, blocks
    return result, replay_error, role_audit


def collect_layer_contrasts(
    split: str,
    model: str,
    groups: dict[str, dict[str, dict[str, Any]]],
    tokenizer: Any,
    device: torch.device,
) -> tuple[dict[str, list[dict[str, dict[str, torch.Tensor]]]], dict[str, Any]]:
    layer_count = MODEL_LAYER_COUNTS[model]
    by_receiver: dict[str, list[dict[str, dict[str, torch.Tensor]]]] = {
        receiver: [] for receiver in RECEIVERS
    }
    max_replay_error = 0.0
    max_missing_fragments = 0
    for layer in range(layer_count):
        o_weight = load_weight(
            model, f"model.layers.{layer}.self_attn.o_proj.weight"
        ).to(device=device, dtype=torch.float32)
        receiver_groups: dict[str, dict[str, dict[str, torch.Tensor]]] = {
            receiver: {} for receiver in RECEIVERS
        }
        for group_id, conditions in groups.items():
            condition_events: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
            for condition, case in conditions.items():
                payload = torch.load(
                    layer_path(split, model, case["blind_case_id"], layer),
                    map_location="cpu",
                    weights_only=False,
                )
                condition_events[condition] = {}
                for receiver in RECEIVERS:
                    vectors, replay_error, role_audit = event_vectors(
                        payload,
                        receiver,
                        case,
                        tokenizer,
                        o_weight,
                        device,
                    )
                    condition_events[condition][receiver] = vectors
                    max_replay_error = max(max_replay_error, replay_error)
                    max_missing_fragments = max(
                        max_missing_fragments,
                        sum(len(value) for value in role_audit["missing_fragments"].values()),
                    )
                del payload
            for receiver in RECEIVERS:
                result: dict[str, torch.Tensor] = {}
                for key in ("semantic_joint", "other_prefix", "roles", "heads", "block"):
                    result[f"{key}_x"] = contrast(
                        condition_events["A_operation_lex_x"][receiver][key],
                        condition_events["B_control_lex_x"][receiver][key],
                    ).to(torch.float16)
                    result[f"{key}_y"] = contrast(
                        condition_events["C_operation_lex_y"][receiver][key],
                        condition_events["D_control_lex_y"][receiver][key],
                    ).to(torch.float16)
                receiver_groups[receiver][group_id] = result
        for receiver in RECEIVERS:
            by_receiver[receiver].append(receiver_groups[receiver])
        print(
            f"[{model}/{split}] Phase390 derived layer {layer + 1}/{layer_count}",
            flush=True,
        )
        del o_weight, receiver_groups
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return by_receiver, {
        "max_attention_role_replay_relative_error": max_replay_error,
        "max_missing_semantic_fragment_count": max_missing_fragments,
    }


def sum_window(
    layers: list[dict[str, dict[str, torch.Tensor]]],
    group_id: str,
    key: str,
    start: int,
    width: int,
) -> torch.Tensor:
    return sum(
        (layers[layer][group_id][key].float() for layer in range(start, start + width)),
        torch.zeros_like(layers[start][group_id][key].float()),
    )


def group_metric(
    layers: list[dict[str, dict[str, torch.Tensor]]],
    targets: dict[str, dict[str, torch.Tensor]],
    ordered_groups: list[str],
    group_index: int,
    start: int,
    width: int,
) -> dict[str, float]:
    group_id = ordered_groups[group_index]
    wrong_group = ordered_groups[(group_index + 1) % len(ordered_groups)]
    target = targets[group_id]
    joint_x = sum_window(layers, group_id, "semantic_joint_x", start, width)
    joint_y = sum_window(layers, group_id, "semantic_joint_y", start, width)
    other_x = sum_window(layers, group_id, "other_prefix_x", start, width)
    other_y = sum_window(layers, group_id, "other_prefix_y", start, width)
    role_x = sum_window(layers, group_id, "roles_x", start, width)
    role_y = sum_window(layers, group_id, "roles_y", start, width)
    head_x = sum_window(layers, group_id, "heads_x", start, width)
    head_y = sum_window(layers, group_id, "heads_y", start, width)
    block_x = sum_window(layers, group_id, "block_x", start, width)
    block_y = sum_window(layers, group_id, "block_y", start, width)
    correct = minimum_alignment(
        joint_x, joint_y, target["terminal_x"], target["terminal_y"]
    )
    wrong_group_alignment = minimum_alignment(
        joint_x,
        joint_y,
        targets[wrong_group]["terminal_x"],
        targets[wrong_group]["terminal_y"],
    )
    wrong_time_alignment = minimum_alignment(
        joint_x, joint_y, target["wrong_time_x"], target["wrong_time_y"]
    )
    role_alignments = [
        minimum_alignment(
            role_x[index],
            role_y[index],
            target["terminal_x"],
            target["terminal_y"],
        )
        for index in range(role_x.shape[0])
    ]
    head_alignments = [
        minimum_alignment(
            head_x[index],
            head_y[index],
            target["terminal_x"],
            target["terminal_y"],
        )
        for index in range(head_x.shape[0])
    ]
    other_alignment = minimum_alignment(
        other_x, other_y, target["terminal_x"], target["terminal_y"]
    )
    block_alignment = minimum_alignment(
        block_x, block_y, target["terminal_x"], target["terminal_y"]
    )
    single_layer_block = max(
        minimum_alignment(
            layers[layer][group_id]["block_x"],
            layers[layer][group_id]["block_y"],
            target["terminal_x"],
            target["terminal_y"],
        )
        for layer in range(start, start + width)
    )
    lexical_replication = min(
        cosine(joint_x, joint_y),
        cosine(block_x, block_y),
    )
    return {
        "correct_alignment": correct,
        "lexical_replication": lexical_replication,
        "correct_minus_wrong_group": correct - wrong_group_alignment,
        "correct_minus_wrong_time": correct - wrong_time_alignment,
        "multi_source_advantage": correct - max(role_alignments),
        "other_prefix_advantage": correct - other_alignment,
        "multi_head_advantage": correct - max(head_alignments),
        "cross_layer_advantage": block_alignment - single_layer_block,
        "block_alignment": block_alignment,
    }


def aggregate_candidate(
    split: str,
    model: str,
    receiver: str,
    window: dict[str, Any],
    group_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    thresholds = {
        "correct_alignment": 0.10,
        "lexical_replication": 0.10,
        "correct_minus_wrong_group": 0.05,
        "correct_minus_wrong_time": 0.05,
        "multi_source_advantage": 0.05,
        "other_prefix_advantage": 0.05,
        "multi_head_advantage": 0.05,
        "cross_layer_advantage": 0.05,
    }
    medians = {
        f"median_{name}": median(row[name] for row in group_metrics)
        for name in thresholds
    }
    support_count = sum(
        all(row[name] >= threshold for name, threshold in thresholds.items())
        for row in group_metrics
    )
    width = int(window["window_length"])
    gate = (
        width > 1
        and all(medians[f"median_{name}"] >= threshold for name, threshold in thresholds.items())
        and support_count >= SUPPORT_REQUIRED[split]
    )
    return {
        "schema_version": "64.6.0",
        "phase_id": f"Phase390-{split.title()}JointAnalysis",
        "split": split,
        "model": model,
        "receiver_coordinate": receiver,
        **window,
        "group_count": len(group_metrics),
        "support_count": support_count,
        "support_required": SUPPORT_REQUIRED[split],
        **medians,
        "thresholds": thresholds,
        "model_candidate_gate_pass": gate,
        "causal_path_claim": False,
    }


def analyze_model(split: str, model: str, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(COLLECTION / split / "models" / model / "manifest.json")
    expected_cases = 48 if split == "discovery" else 24
    if (
        not manifest["valid"]
        or manifest["case_count"] != expected_cases
        or not manifest["all_case_gates_pass"]
        or manifest["required_transition_pass_count"] != expected_cases
    ):
        raise RuntimeError(f"Invalid Phase390 collection manifest for {model}/{split}")
    groups = case_groups(split, model)
    targets = terminal_targets(split, model, groups)
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=False,
    )
    layers_by_receiver, quality = collect_layer_contrasts(
        split, model, groups, tokenizer, device
    )
    if quality["max_attention_role_replay_relative_error"] > 0.01:
        raise RuntimeError(f"Role replay conservation failed for {model}/{split}: {quality}")
    windows = window_lattice(MODEL_LAYER_COUNTS[model])
    ordered_groups = sorted(groups)
    rows: list[dict[str, Any]] = []
    for receiver in RECEIVERS:
        layers = layers_by_receiver[receiver]
        for window in windows:
            metrics = [
                group_metric(
                    layers,
                    targets,
                    ordered_groups,
                    group_index,
                    int(window["start_layer"]),
                    int(window["window_length"]),
                )
                for group_index in range(len(ordered_groups))
            ]
            rows.append(
                aggregate_candidate(split, model, receiver, window, metrics)
            )
    quality.update(
        {
            "model": model,
            "split": split,
            "candidate_count": len(rows),
            "model_gate_pass_count": sum(row["model_candidate_gate_pass"] for row in rows),
        }
    )
    del layers_by_receiver, targets
    gc.collect()
    return rows, quality


def crossmodel_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["receiver_coordinate"],
        row["anchor_index"],
        row["window_length"],
    )


def allowed_keys(split: str) -> set[tuple[Any, ...]] | None:
    if split == "discovery":
        return None
    name = (
        "phase390_discovery_candidate_freeze.json"
        if split == "calibration"
        else "phase390_calibration_summary.json"
    )
    payload = read_json(PHASE_ROOT / name)
    field = (
        "frozen_crossmodel_candidates"
        if split == "calibration"
        else "physical_candidates"
    )
    return {
        (
            row["receiver_coordinate"],
            row["anchor_index"],
            row["window_length"],
        )
        for row in payload[field]
    }


def main(split: str, device_name: str) -> None:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    allowed = allowed_keys(split)
    all_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for model in MODELS:
        rows, quality = analyze_model(split, model, device)
        if allowed is not None:
            rows = [row for row in rows if crossmodel_key(row) in allowed]
            quality["candidate_count_after_frozen_filter"] = len(rows)
            quality["model_gate_pass_count_after_frozen_filter"] = sum(
                row["model_candidate_gate_pass"] for row in rows
            )
        all_rows.extend(rows)
        quality_rows.append(quality)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in all_rows:
        by_key[crossmodel_key(row)][row["model"]] = row
    crossmodel_rows: list[dict[str, Any]] = []
    for key, models in sorted(by_key.items()):
        if set(models) != set(MODELS):
            continue
        receiver, anchor, width = key
        passed = all(models[model]["model_candidate_gate_pass"] for model in MODELS)
        crossmodel_rows.append(
            {
                "schema_version": "64.6.0",
                "phase_id": f"Phase390-{split.title()}JointAnalysis",
                "split": split,
                "receiver_coordinate": receiver,
                "anchor_index": anchor,
                "anchor_fraction": models[MODELS[0]]["anchor_fraction"],
                "window_length": width,
                "model_layers": {
                    model: [models[model]["start_layer"], models[model]["end_layer"]]
                    for model in MODELS
                },
                "models_passing": [
                    model for model in MODELS if models[model]["model_candidate_gate_pass"]
                ],
                "crossmodel_candidate_gate_pass": passed,
                "causal_path_claim": False,
            }
        )
    passing = [row for row in crossmodel_rows if row["crossmodel_candidate_gate_pass"]]
    out_dir = PHASE_ROOT / f"{split}_joint_analysis"
    write_jsonl(out_dir / "phase390_model_candidate_rows.jsonl", all_rows)
    write_jsonl(out_dir / "phase390_crossmodel_candidate_rows.jsonl", crossmodel_rows)
    if split == "discovery":
        freeze = {
            "schema_version": "64.6.0",
            "phase_id": "Phase390-DiscoveryCandidateFreeze",
            "created_at": now(),
            "frozen_crossmodel_candidates": passing,
            "denominator": {
                "model_candidate_count": len(all_rows),
                "crossmodel_candidate_count": len(crossmodel_rows),
                "passing_crossmodel_candidate_count": len(passing),
            },
            "quality": quality_rows,
            "authorization": {
                "calibration_collection": bool(passing),
                "physical_holdout_collection": False,
                "causal_replay": False,
                "single_neuron_scan": False,
            },
            "claim_boundary": {
                "discovery_candidate_is_causal_path": False,
                "no_candidate_means_no_language_structure": False,
            },
        }
        write_json(PHASE_ROOT / "phase390_discovery_candidate_freeze.json", freeze)
        summary = freeze
    elif split == "calibration":
        summary = {
            "schema_version": "64.7.0",
            "phase_id": "Phase390-CalibrationSummary",
            "created_at": now(),
            "calibrated_crossmodel_candidates": passing,
            "physical_candidates": passing,
            "denominator": {
                "frozen_candidate_count": len(allowed or set()),
                "calibration_crossmodel_candidate_count": len(crossmodel_rows),
                "passing_calibration_candidate_count": len(passing),
            },
            "quality": quality_rows,
            "authorization": {
                "physical_holdout_collection": bool(passing),
                "causal_replay": False,
                "single_neuron_scan": False,
            },
        }
        write_json(PHASE_ROOT / "phase390_calibration_summary.json", summary)
    else:
        summary = {
            "schema_version": "64.8.0",
            "phase_id": "Phase390-PhysicalSummary",
            "created_at": now(),
            "physical_crossmodel_candidates": passing,
            "denominator": {
                "frozen_physical_candidate_count": len(allowed or set()),
                "physical_crossmodel_candidate_count": len(crossmodel_rows),
                "passing_physical_candidate_count": len(passing),
            },
            "quality": quality_rows,
            "authorization": {
                "graph_consistent_parent_boundary_replay": bool(passing),
                "single_neuron_scan": False,
            },
            "claim_boundary": {
                "physical_prediction_is_causal_path": False,
                "language_encoding_closed": False,
            },
        }
        write_json(PHASE_ROOT / "phase390_physical_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.split, args.device)
