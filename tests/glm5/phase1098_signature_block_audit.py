#!/usr/bin/env python3
"""Decompose the frozen Phase1097 trajectory signature into its three blocks."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1097_conditional_transition_protocol as p1097


OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1098_relative_relation_geometry"
EPSILON = 1e-12
BLOCKS = ("amplitude", "depth_gram", "local_margin", "combined")


def safe_mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    result = np.full(total.shape, np.nan, dtype=np.float64)
    valid = count > 0
    result[valid] = total[valid] / count[valid]
    return result


def unit(value: np.ndarray) -> np.ndarray:
    clean = np.where(np.isfinite(value), value, 0.0).astype(np.float64, copy=False).reshape(-1)
    norm = float(np.linalg.norm(clean))
    return clean / norm if norm > EPSILON else np.zeros_like(clean)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = unit(left)
    right_unit = unit(right)
    if np.linalg.norm(left_unit) <= EPSILON or np.linalg.norm(right_unit) <= EPSILON:
        return 0.0
    return float(left_unit @ right_unit)


def load_model(model_name: str) -> dict[str, np.ndarray]:
    root = p1097.OUT_ROOT / "atlas" / model_name
    with np.load(root / "transition_aggregates.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return {
        "amplitude": safe_mean(arrays["amplitude_sum"], arrays["amplitude_count"]),
        "depth_gram": safe_mean(arrays["gram_sum"], arrays["gram_count"]),
        "local_margin": safe_mean(arrays["local_margin_sum"], arrays["local_margin_count"]),
    }


def block_vector(data: dict[str, np.ndarray], block: str, indices: tuple[int, int, int, int, int]) -> np.ndarray:
    relation, surface, split, field, role = indices
    amplitude = unit(data["amplitude"][relation, surface, split, field, role])
    gram = data["depth_gram"][relation, surface, split, field, role]
    upper = np.triu_indices(gram.shape[-1], k=1)
    depth_gram = unit(gram[upper])
    local_margin = unit(data["local_margin"][relation, surface, split, field, role])
    if block == "amplitude":
        return amplitude
    if block == "depth_gram":
        return depth_gram
    if block == "local_margin":
        return local_margin
    if block == "combined":
        return np.concatenate((amplitude, depth_gram, local_margin)) / math.sqrt(3.0)
    raise KeyError(block)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def split_repeat(data: dict[str, np.ndarray], block: str) -> dict[str, Any]:
    fields = {name: index for index, name in enumerate(p1097.FIELDS)}
    roles = {name: index for index, name in enumerate(p1097.CAPTURE_ROLES)}
    records = []
    for relation_index, relation in enumerate(p1097.RELATIONS):
        for surface_index, surface in enumerate(p1097.SURFACES):
            for role_name in ("query_end", "answer_boundary"):
                role = roles[role_name]
                content = cosine(
                    block_vector(data, block, (relation_index, surface_index, 0, fields["relational_execution"], role)),
                    block_vector(data, block, (relation_index, surface_index, 1, fields["relational_execution"], role)),
                )
                carrier = cosine(
                    block_vector(data, block, (relation_index, surface_index, 0, fields["relational_carrier"], role)),
                    block_vector(data, block, (relation_index, surface_index, 1, fields["relational_carrier"], role)),
                )
                records.append({
                    "relation": relation,
                    "surface": surface,
                    "role": role_name,
                    "content_cosine": content,
                    "carrier_cosine": carrier,
                    "content_minus_carrier": content - carrier,
                })
    return {
        "content": summarize([row["content_cosine"] for row in records]),
        "carrier": summarize([row["carrier_cosine"] for row in records]),
        "advantage": summarize([row["content_minus_carrier"] for row in records]),
        "content_at_least_0_80": sum(row["content_cosine"] >= 0.80 for row in records),
        "content_advantage_at_least_0_05": sum(row["content_minus_carrier"] >= 0.05 for row in records),
        "record_count": len(records),
        "records": records,
    }


def heldout_commonality(data: dict[str, np.ndarray], block: str) -> dict[str, Any]:
    fields = {name: index for index, name in enumerate(p1097.FIELDS)}
    role = p1097.CAPTURE_ROLES.index("answer_boundary")
    records = []
    for surface_index, surface in enumerate(p1097.SURFACES):
        for split_index, split in enumerate(p1097.SPLITS):
            for heldout_index, relation in enumerate(p1097.RELATIONS):
                train = [index for index in range(len(p1097.RELATIONS)) if index != heldout_index]
                content_center = unit(np.mean([
                    block_vector(data, block, (index, surface_index, split_index, fields["relational_execution"], role))
                    for index in train
                ], axis=0))
                carrier_center = unit(np.mean([
                    block_vector(data, block, (index, surface_index, split_index, fields["relational_carrier"], role))
                    for index in train
                ], axis=0))
                content = cosine(content_center, block_vector(
                    data, block, (heldout_index, surface_index, split_index, fields["relational_execution"], role)
                ))
                carrier = cosine(carrier_center, block_vector(
                    data, block, (heldout_index, surface_index, split_index, fields["relational_carrier"], role)
                ))
                records.append({
                    "surface": surface,
                    "split": split,
                    "heldout_relation": relation,
                    "content_cosine": content,
                    "carrier_cosine": carrier,
                    "content_minus_carrier": content - carrier,
                })
    return {
        "content": summarize([row["content_cosine"] for row in records]),
        "carrier": summarize([row["carrier_cosine"] for row in records]),
        "advantage": summarize([row["content_minus_carrier"] for row in records]),
        "strict_passes": sum(
            row["content_cosine"] >= 0.80 and row["content_minus_carrier"] >= 0.05
            for row in records
        ),
        "record_count": len(records),
        "records": records,
    }


def cross_language_identity(data: dict[str, np.ndarray], block: str) -> dict[str, Any]:
    fields = {name: index for index, name in enumerate(p1097.FIELDS)}
    role = p1097.CAPTURE_ROLES.index("answer_boundary")
    records = []
    for source_surface, target_surface in ((0, 1), (1, 0)):
        for split_index, split in enumerate(p1097.SPLITS):
            source_content = [
                block_vector(data, block, (relation, source_surface, split_index, fields["relational_execution"], role))
                for relation in range(len(p1097.RELATIONS))
            ]
            target_content = [
                block_vector(data, block, (relation, target_surface, split_index, fields["relational_execution"], role))
                for relation in range(len(p1097.RELATIONS))
            ]
            source_carrier = [
                block_vector(data, block, (relation, source_surface, split_index, fields["relational_carrier"], role))
                for relation in range(len(p1097.RELATIONS))
            ]
            target_carrier = [
                block_vector(data, block, (relation, target_surface, split_index, fields["relational_carrier"], role))
                for relation in range(len(p1097.RELATIONS))
            ]
            content_matrix = np.asarray([[cosine(left, right) for right in target_content] for left in source_content])
            carrier_diagonal = [cosine(source_carrier[index], target_carrier[index]) for index in range(len(p1097.RELATIONS))]
            for relation_index, relation in enumerate(p1097.RELATIONS):
                predicted = int(np.argmax(content_matrix[relation_index]))
                content = float(content_matrix[relation_index, relation_index])
                carrier = float(carrier_diagonal[relation_index])
                records.append({
                    "source_surface": p1097.SURFACES[source_surface],
                    "target_surface": p1097.SURFACES[target_surface],
                    "split": split,
                    "relation": relation,
                    "predicted_relation": p1097.RELATIONS[predicted],
                    "identity_correct": predicted == relation_index,
                    "content_cosine": content,
                    "carrier_cosine": carrier,
                    "content_minus_carrier": content - carrier,
                })
    return {
        "identity_correct": sum(row["identity_correct"] for row in records),
        "strict_passes": sum(
            row["identity_correct"] and row["content_minus_carrier"] >= 0.05
            for row in records
        ),
        "record_count": len(records),
        "content": summarize([row["content_cosine"] for row in records]),
        "advantage": summarize([row["content_minus_carrier"] for row in records]),
        "records": records,
    }


def model_audit(model_name: str) -> dict[str, Any]:
    data = load_model(model_name)
    return {
        block: {
            "split_repeat": split_repeat(data, block),
            "heldout_commonality": heldout_commonality(data, block),
            "cross_language_identity": cross_language_identity(data, block),
        }
        for block in BLOCKS
    }


def main() -> None:
    model_results = {model: model_audit(model) for model in p1097.MODELS}
    result = {
        "phase": 1098,
        "source_phase": 1097,
        "analysis_kind": "frozen_signature_block_decomposition",
        "models": model_results,
        "interpretation_guardrails": [
            "This audit decomposes a frozen descriptive signature; it does not identify a neural mechanism.",
            "Held-out commonality tests a shared shape, not held-out relation identity.",
            "A stable block can still be driven by task shell, lexical surface, or answer preparation.",
        ],
    }
    output = OUT_ROOT / "analysis" / "signature_block_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = {
        model: {
            block: {
                "split_content_median": values[block]["split_repeat"]["content"]["median"],
                "split_advantage_median": values[block]["split_repeat"]["advantage"]["median"],
                "heldout_content_median": values[block]["heldout_commonality"]["content"]["median"],
                "heldout_strict_passes": values[block]["heldout_commonality"]["strict_passes"],
                "cross_language_identity": values[block]["cross_language_identity"]["identity_correct"],
                "cross_language_strict_passes": values[block]["cross_language_identity"]["strict_passes"],
            }
            for block in BLOCKS
        }
        for model, values in model_results.items()
    }
    print(json.dumps(digest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
