#!/usr/bin/env python3
"""Separate node-role polarity from edge binding in the Phase532 discovery data."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from phase532_glm4_role_normalized_world_geometry import (
    build_examples,
    canonical_endpoints,
    fit_direction,
    group_folds,
    pair_feature,
    rate,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "tests/gpt5/result/phase532_glm4_role_normalized_world_geometry"
OUT_DIR = ROOT / "tests/gpt5/result/phase533_world_geometry_role_binding_audit"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def oof_pair_scores(
    examples: dict[str, Any],
    metadata: list[dict[str, Any]],
    fold_count: int,
) -> np.ndarray:
    features = examples["features"]
    labels = examples["labels"]
    orientation = examples["kinds"] != "disconnected"
    world_indices = examples["world_indices"]
    row_folds = group_folds(metadata, fold_count)
    scores = np.zeros((len(labels), features.shape[1], features.shape[2]), dtype=np.float32)
    for layer in range(features.shape[1]):
        for projection in range(features.shape[2]):
            local = features[:, layer, projection]
            for fold in range(fold_count):
                test = row_folds[world_indices] == fold
                train = orientation & ~test
                direction, threshold = fit_direction(local[train], labels[train])
                scores[test, layer, projection] = local[test] @ direction - threshold
    return scores


def node_role_oof(
    endpoints: np.ndarray,
    metadata: list[dict[str, Any]],
    fold_count: int,
) -> np.ndarray:
    features = []
    labels = []
    world_indices = []
    for world_index, row in enumerate(metadata):
        sources = {int(edge[0]) for edge in row["edges"]}
        for entity in range(4):
            features.append(endpoints[world_index, :, :, entity])
            labels.append(entity in sources)
            world_indices.append(world_index)
    values = np.stack(features, axis=0).transpose(0, 2, 1, 3)
    labels_array = np.asarray(labels, dtype=bool)
    world_indices_array = np.asarray(world_indices, dtype=np.int32)
    row_folds = group_folds(metadata, fold_count)
    correct = np.zeros((values.shape[1], values.shape[2]), dtype=np.float32)
    for layer in range(values.shape[1]):
        for projection in range(values.shape[2]):
            predictions = np.zeros(len(labels_array), dtype=bool)
            local = values[:, layer, projection]
            for fold in range(fold_count):
                test = row_folds[world_indices_array] == fold
                direction, threshold = fit_direction(local[~test], labels_array[~test])
                predictions[test] = local[test] @ direction > threshold
            correct[layer, projection] = float((predictions == labels_array).mean())
    return correct


def disconnected_categories(metadata: list[dict[str, Any]]) -> list[str]:
    categories = []
    disconnected_pairs = ((0, 2), (2, 0), (0, 3), (3, 0), (1, 2), (2, 1), (1, 3), (3, 1))
    for row in metadata:
        sources = {int(edge[0]) for edge in row["edges"]}
        # Four orientation examples precede the eight disconnected examples.
        categories.extend(["orientation"] * 4)
        for source, target in disconnected_pairs:
            left = "source" if source in sources else "target"
            right = "source" if target in sources else "target"
            categories.append(f"{left}_to_{right}")
    return categories


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    array = np.load(SOURCE_DIR / "phase532_glm4_discovery_projection.npz")["projected"]
    metadata = read_jsonl(SOURCE_DIR / "phase532_glm4_discovery_metadata.jsonl")
    endpoints = canonical_endpoints(array, metadata)
    examples = build_examples(endpoints, metadata)
    scores = oof_pair_scores(examples, metadata, 4)
    node_rates = node_role_oof(endpoints, metadata, 4)
    categories = np.asarray(disconnected_categories(metadata))
    if len(categories) != len(examples["labels"]):
        raise RuntimeError("example category alignment failed")

    orientation = examples["kinds"] != "disconnected"
    orientation_rates = np.zeros((scores.shape[1], scores.shape[2]), dtype=np.float32)
    for layer in range(scores.shape[1]):
        for projection in range(scores.shape[2]):
            orientation_rates[layer, projection] = float(
                np.mean((scores[orientation, layer, projection] > 0) == examples["labels"][orientation])
            )

    selected_cells = []
    for projection in range(scores.shape[2]):
        best_rate = float(orientation_rates[:, projection].max())
        candidate_layers = np.where(orientation_rates[:, projection] == best_rate)[0]
        layer = int(candidate_layers[len(candidate_layers) // 2])
        category_reports = {}
        for category in sorted(set(categories) - {"orientation"}):
            mask = categories == category
            category_reports[category] = {
                "predicted_edge": rate(scores[mask, layer, projection] > 0),
                "expected_edge": False,
            }
        selected_cells.append({
            "projection_index": projection,
            "layer_with_embedding": layer,
            "orientation_accuracy": best_rate,
            "node_source_target_role_accuracy": float(node_rates[layer, projection]),
            "disconnected_by_node_role": category_reports,
        })

    source_target_false_positive = []
    for cell in selected_cells:
        source_target_false_positive.append(
            cell["disconnected_by_node_role"]["source_to_target"]["predicted_edge"]["rate"]
        )
    payload = {
        "schema_version": "phase533_world_geometry_role_binding_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "model": "glm4",
        "discovery_row_count": len(metadata),
        "selected_cells": selected_cells,
        "mean_source_to_target_disconnected_false_positive": float(np.mean(source_target_false_positive)),
        "conclusion": (
            "The observer recovers source/target node-role polarity but does not recover pair-specific edge binding."
            if np.mean(source_target_false_positive) >= 0.75
            else "Node-role polarity alone does not fully explain the disconnected failures."
        ),
        "evidence_boundary": {
            "offline_reanalysis_only": True,
            "new_model_run": False,
            "edge_binding_confirmed": False,
            "causal": False,
            "sealed_split_read": False,
        },
    }
    output = OUT_DIR / "phase533_world_geometry_role_binding_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
