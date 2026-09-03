#!/usr/bin/env python3
"""Shared helpers for the C097 observation-first relation-field campaign."""
from __future__ import annotations

from collections import Counter

import numpy as np


FAMILIES = ("similarity", "class_inclusion", "whole_part")
FAMILY_PAIRS = (("similarity", "class_inclusion"), ("similarity", "whole_part"), ("class_inclusion", "whole_part"))
SURFACES = ("prequery", "postquery")
ROLES = ("source_word", "target_word", "relation_anchor", "boundary")
FOCUS_STATES = (31, 32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def top_indices(vector: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(np.abs(vector), kind="stable")[-k:]


def balanced_accuracy(rows: list[dict]) -> float:
    recalls = []
    for truth in (True, False):
        subset = [row for row in rows if row["truth"] is truth]
        recalls.append(sum(row["correct"] for row in subset) / len(subset))
    return float(np.mean(recalls))


def majority_predictions(rows: list[dict], key: str) -> list[str]:
    lookup = {}
    for value in {row[key] for row in rows}:
        labels = [row["gold_label"] for row in rows if row[key] == value]
        lookup[value] = Counter(labels).most_common(1)[0][0]
    return [lookup[row[key]] for row in rows]


def decompose_contrasts(vectors: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Identify the mean contrast G and zero-sum residuals without a raw-cell intercept collision."""
    stack = np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors])
    common = stack.mean(axis=0)
    residuals = [stack[index] - common for index in range(stack.shape[0])]
    total = float(np.sum(stack * stack))
    common_energy = float(stack.shape[0] * np.sum(common * common))
    residual_energy = float(sum(np.sum(value * value) for value in residuals))
    return common, residuals, {
        "total_contrast_energy": total,
        "common_energy": common_energy,
        "residual_energy": residual_energy,
        "common_fraction": common_energy / total if total > 0 else 0.0,
        "energy_identity_error": abs(total - common_energy - residual_energy),
        "residual_sum_max_abs": float(np.max(np.abs(np.sum(residuals, axis=0)))),
    }
