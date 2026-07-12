#!/usr/bin/env python3
"""Build Phase391 replicated local direct-parent physical layout candidates."""

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

import phase390_joint_graph_analysis as joint  # noqa: E402
from model_registry import get_model_spec  # noqa: E402


P390 = ROOT / "tests/gpt5/result/phase390_joint_formation_graph"
OUT = ROOT / "tests/gpt5/result/phase391_local_parent_graph"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "calibration", "physical_holdout")
RECEIVERS = ("query_integrated", "pre_decision")
ROLE_NAMES = tuple(joint.SEMANTIC_ROLES)
ANCHOR_COUNT = 8
LAYER_COUNTS = joint.MODEL_LAYER_COUNTS
SUPPORT_REQUIRED = {"discovery": 8, "calibration": 4, "physical_holdout": 4}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def inner_share(child: torch.Tensor, parent: torch.Tensor) -> float:
    parent = parent.float()
    denominator = float(torch.dot(parent, parent).item())
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(child.float(), parent).item()) / denominator


def cancellation(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = float(torch.linalg.vector_norm(left.float() + right.float()).item())
    denominator = float(
        torch.linalg.vector_norm(left.float()).item()
        + torch.linalg.vector_norm(right.float()).item()
    )
    return numerator / max(denominator, 1e-12)


def relative_layer(model: str, anchor_index: int) -> int:
    fraction = anchor_index / (ANCHOR_COUNT - 1)
    return round(fraction * (LAYER_COUNTS[model] - 1))


def group_local_metrics(layer_row: dict[str, torch.Tensor]) -> dict[str, Any]:
    semantic_x = layer_row["semantic_joint_x"].float()
    semantic_y = layer_row["semantic_joint_y"].float()
    other_x = layer_row["other_prefix_x"].float()
    other_y = layer_row["other_prefix_y"].float()
    attention_x = semantic_x + other_x
    attention_y = semantic_y + other_y
    mlp_x = layer_row["block_x"].float() - attention_x
    mlp_y = layer_row["block_y"].float() - attention_y
    semantic_shares = (
        inner_share(semantic_x, attention_x),
        inner_share(semantic_y, attention_y),
    )
    other_shares = (
        inner_share(other_x, attention_x),
        inner_share(other_y, attention_y),
    )
    roles_x = layer_row["roles_x"].float()
    roles_y = layer_row["roles_y"].float()
    heads_x = layer_row["heads_x"].float()
    heads_y = layer_row["heads_y"].float()
    return {
        "semantic_share": min(semantic_shares),
        "semantic_minus_other_share": min(
            semantic_shares[0] - other_shares[0],
            semantic_shares[1] - other_shares[1],
        ),
        "lexical_replication": joint.cosine(semantic_x, semantic_y),
        "role_shares": [
            min(
                inner_share(roles_x[index], attention_x),
                inner_share(roles_y[index], attention_y),
            )
            for index in range(roles_x.shape[0])
        ],
        "head_shares": [
            min(
                inner_share(heads_x[index], attention_x),
                inner_share(heads_y[index], attention_y),
            )
            for index in range(heads_x.shape[0])
        ],
        "attention_mlp_compensation": max(
            joint.cosine(attention_x, mlp_x),
            joint.cosine(attention_y, mlp_y),
        ),
        "event_cancellation": min(
            cancellation(attention_x, mlp_x),
            cancellation(attention_y, mlp_y),
        ),
    }


def choose_fixed_index(rows: list[dict[str, Any]], key: str) -> tuple[int, list[float]]:
    width = len(rows[0][key])
    scores = [median(row[key][index] for row in rows) for index in range(width)]
    return max(range(width), key=lambda index: scores[index]), scores


def aggregate_model_cell(
    split: str,
    model: str,
    receiver: str,
    anchor_index: int,
    layer: int,
    rows: list[dict[str, Any]],
    frozen: dict[str, Any] | None,
) -> dict[str, Any]:
    if frozen is None:
        best_role, role_scores = choose_fixed_index(rows, "role_shares")
        best_head, head_scores = choose_fixed_index(rows, "head_shares")
        participating_roles = [
            index for index, score in enumerate(role_scores) if score >= 0.05
        ]
        participating_heads = [
            index for index, score in enumerate(head_scores) if score >= 0.02
        ]
    else:
        best_role = int(frozen["best_role_index"])
        best_head = int(frozen["best_head_index"])
        participating_roles = [int(value) for value in frozen["participating_role_indices"]]
        participating_heads = [int(value) for value in frozen["participating_head_indices"]]
        role_scores = [median(row["role_shares"][index] for row in rows) for index in range(len(ROLE_NAMES))]
        head_scores = [median(row["head_shares"][index] for row in rows) for index in range(len(rows[0]["head_shares"]))]
    semantic = median(row["semantic_share"] for row in rows)
    specificity = median(row["semantic_minus_other_share"] for row in rows)
    replication = median(row["lexical_replication"] for row in rows)
    role_advantages = [
        row["semantic_share"] - row["role_shares"][best_role] for row in rows
    ]
    head_advantages = [
        row["semantic_share"] - row["head_shares"][best_head] for row in rows
    ]
    support_count = sum(
        row["semantic_share"] >= 0.10
        and row["semantic_minus_other_share"] >= 0.05
        and row["lexical_replication"] >= 0.10
        and row["semantic_share"] - row["role_shares"][best_role] >= 0.05
        and row["semantic_share"] - row["head_shares"][best_head] >= 0.05
        for row in rows
    )
    metrics = {
        "median_semantic_share": semantic,
        "median_semantic_minus_other_share": specificity,
        "median_lexical_replication": replication,
        "median_joint_advantage_over_fixed_role": median(role_advantages),
        "median_joint_advantage_over_fixed_head": median(head_advantages),
        "participating_role_count": len(participating_roles),
        "participating_head_count": len(participating_heads),
        "median_attention_mlp_compensation": median(
            row["attention_mlp_compensation"] for row in rows
        ),
        "median_event_cancellation": median(row["event_cancellation"] for row in rows),
        "support_count": support_count,
    }
    gate = (
        semantic >= 0.10
        and specificity >= 0.05
        and replication >= 0.10
        and metrics["median_joint_advantage_over_fixed_role"] >= 0.05
        and metrics["median_joint_advantage_over_fixed_head"] >= 0.05
        and len(participating_roles) >= 2
        and len(participating_heads) >= 2
        and support_count >= SUPPORT_REQUIRED[split]
    )
    return {
        "schema_version": "65.1.0",
        "phase_id": f"Phase391-{split.title()}LocalParentAnalysis",
        "split": split,
        "model": model,
        "receiver_coordinate": receiver,
        "anchor_index": anchor_index,
        "anchor_fraction": anchor_index / (ANCHOR_COUNT - 1),
        "layer_index": layer,
        "group_count": len(rows),
        "best_role_index": best_role,
        "best_role_name": ROLE_NAMES[best_role],
        "best_head_index": best_head,
        "participating_role_indices": participating_roles,
        "participating_role_names": [ROLE_NAMES[index] for index in participating_roles],
        "participating_head_indices": participating_heads,
        "role_median_shares": role_scores,
        "head_median_shares": head_scores,
        **metrics,
        "support_required": SUPPORT_REQUIRED[split],
        "model_local_parent_gate_pass": gate,
        "terminal_prediction_claim": False,
        "causal_language_path_claim": False,
    }


def collection_root(split: str) -> Path:
    return P390 / "collection" if split == "discovery" else OUT / "collection"


def frozen_model_cells(split: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    if split == "discovery":
        return {}
    source = (
        OUT / "phase391_discovery_candidate_freeze.json"
        if split == "calibration"
        else OUT / "phase391_calibration_summary.json"
    )
    payload = read_json(source)
    field = (
        "frozen_crossmodel_candidates"
        if split == "calibration"
        else "physical_candidates"
    )
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for candidate in payload[field]:
        for model, cell in candidate["model_cells"].items():
            result[(model, candidate["receiver_coordinate"], candidate["anchor_index"])] = cell
    return result


def analyze_model(
    split: str,
    model: str,
    device: torch.device,
    frozen_cells: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = collection_root(split)
    manifest = read_json(root / split / "models" / model / "manifest.json")
    expected = 48 if split == "discovery" else 24
    if not manifest["valid"] or manifest["case_count"] != expected:
        raise RuntimeError(f"Invalid Phase391 source manifest for {model}/{split}")
    groups = joint.case_groups(split, model)
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=False,
    )
    original_collection = joint.COLLECTION
    try:
        joint.COLLECTION = root
        layers_by_receiver, quality = joint.collect_layer_contrasts(
            split, model, groups, tokenizer, device
        )
    finally:
        joint.COLLECTION = original_collection
    if quality["max_attention_role_replay_relative_error"] > 0.01:
        raise RuntimeError(f"Phase391 role replay failed for {model}/{split}")
    rows: list[dict[str, Any]] = []
    for receiver in RECEIVERS:
        for anchor_index in range(ANCHOR_COUNT):
            key = (model, receiver, anchor_index)
            if split != "discovery" and key not in frozen_cells:
                continue
            layer = relative_layer(model, anchor_index)
            group_rows = [
                group_local_metrics(layer_group)
                for layer_group in layers_by_receiver[receiver][layer].values()
            ]
            rows.append(
                aggregate_model_cell(
                    split,
                    model,
                    receiver,
                    anchor_index,
                    layer,
                    group_rows,
                    frozen_cells.get(key),
                )
            )
    quality.update(
        {
            "model": model,
            "split": split,
            "evaluated_model_cell_count": len(rows),
            "passing_model_cell_count": sum(
                row["model_local_parent_gate_pass"] for row in rows
            ),
        }
    )
    del layers_by_receiver
    gc.collect()
    return rows, quality


def crossmodel_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["receiver_coordinate"], row["anchor_index"])][row["model"]] = row
    result: list[dict[str, Any]] = []
    for (receiver, anchor), models in sorted(grouped.items()):
        if set(models) != set(MODELS):
            continue
        gate = all(models[model]["model_local_parent_gate_pass"] for model in MODELS)
        result.append(
            {
                "schema_version": "65.1.0",
                "phase_id": f"Phase391-{rows[0]['split'].title()}LocalParentAnalysis",
                "split": rows[0]["split"],
                "receiver_coordinate": receiver,
                "anchor_index": anchor,
                "anchor_fraction": anchor / (ANCHOR_COUNT - 1),
                "model_cells": {
                    model: {
                        "layer_index": models[model]["layer_index"],
                        "best_role_index": models[model]["best_role_index"],
                        "best_role_name": models[model]["best_role_name"],
                        "best_head_index": models[model]["best_head_index"],
                        "participating_role_indices": models[model][
                            "participating_role_indices"
                        ],
                        "participating_head_indices": models[model][
                            "participating_head_indices"
                        ],
                        "model_gate_pass": models[model][
                            "model_local_parent_gate_pass"
                        ],
                    }
                    for model in MODELS
                },
                "models_passing": [
                    model
                    for model in MODELS
                    if models[model]["model_local_parent_gate_pass"]
                ],
                "crossmodel_local_parent_gate_pass": gate,
                "terminal_prediction_claim": False,
                "causal_language_path_claim": False,
            }
        )
    return result


def main(split: str, device_name: str) -> None:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    frozen_cells = frozen_model_cells(split)
    model_rows: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for model in MODELS:
        rows, model_quality = analyze_model(
            split, model, device, frozen_cells
        )
        model_rows.extend(rows)
        quality.append(model_quality)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    crossmodel = crossmodel_candidates(model_rows)
    passing = [row for row in crossmodel if row["crossmodel_local_parent_gate_pass"]]
    split_root = OUT / f"{split}_analysis"
    write_jsonl(split_root / "phase391_model_cells.jsonl", model_rows)
    write_jsonl(split_root / "phase391_crossmodel_cells.jsonl", crossmodel)
    if split == "discovery":
        summary = {
            "schema_version": "65.1.0",
            "phase_id": "Phase391-DiscoveryCandidateFreeze",
            "created_at": now(),
            "frozen_crossmodel_candidates": passing,
            "denominator": {
                "model_cell_count": len(model_rows),
                "crossmodel_cell_count": len(crossmodel),
                "passing_crossmodel_candidate_count": len(passing),
            },
            "quality": quality,
            "authorization": {
                "calibration_collection": bool(passing),
                "physical_holdout_collection": False,
                "causal_replay": False,
                "single_neuron_scan": False,
            },
            "claim_boundary": {
                "local_parent_candidate_is_terminal_prediction": False,
                "local_parent_candidate_is_causal_language_path": False,
            },
        }
        write_json(OUT / "phase391_discovery_candidate_freeze.json", summary)
    elif split == "calibration":
        summary = {
            "schema_version": "65.2.0",
            "phase_id": "Phase391-CalibrationSummary",
            "created_at": now(),
            "calibrated_crossmodel_candidates": passing,
            "physical_candidates": passing,
            "denominator": {
                "frozen_candidate_count": len(frozen_cells) // len(MODELS),
                "passing_calibration_candidate_count": len(passing),
            },
            "quality": quality,
            "authorization": {
                "physical_holdout_collection": bool(passing),
                "causal_replay": False,
                "single_neuron_scan": False,
            },
        }
        write_json(OUT / "phase391_calibration_summary.json", summary)
    else:
        summary = {
            "schema_version": "65.3.0",
            "phase_id": "Phase391-PhysicalSummary",
            "created_at": now(),
            "physical_crossmodel_candidates": passing,
            "denominator": {
                "frozen_physical_candidate_count": len(frozen_cells) // len(MODELS),
                "passing_physical_candidate_count": len(passing),
            },
            "quality": quality,
            "authorization": {
                "graph_consistent_parent_boundary_replay": bool(passing),
                "single_neuron_scan": False,
            },
            "claim_boundary": {
                "physical_local_parent_layout_is_causal_language_path": False,
                "language_encoding_closed": False,
            },
        }
        write_json(OUT / "phase391_physical_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.split, args.device)
