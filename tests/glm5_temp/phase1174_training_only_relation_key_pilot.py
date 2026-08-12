"""Exclusion-only pilot for the Phase1174 training-only relation key.

Formal task exponents and formal seeds are intentionally absent. This pilot
only checks support, abstention, and split mechanics before preregistration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5_temp/phase1174_training_only_relation_key_pilot.json"
P = 61
TRANSFORMS = (1, 2, 3)
THRESHOLDS = {
    "key_edge_min": 200,
    "source_support_min": 2,
    "source_coverage_min": 0.70,
    "key_consistency_min": 0.98,
    "mapping_injectivity_min": 0.95,
    "validation_edge_min": 200,
    "validation_consistency_min": 0.98,
}


def task_table(inner_power: int, outer_power: int) -> np.ndarray:
    if math.gcd(outer_power, P - 1) != 1:
        raise ValueError("outer power must permute nonzero residues")
    return np.asarray(
        [
            [pow((a + pow(b, inner_power, P)) % P, outer_power, P) for b in range(P)]
            for a in range(P)
        ],
        dtype=np.int64,
    )


def split_material(table: np.ndarray, seed: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    flat_order = rng.permutation(P * P)
    train = np.zeros(P * P, dtype=bool)
    train[flat_order[: (P * P) // 2]] = True
    train = train.reshape(P, P)
    backgrounds = rng.permutation(P)
    return train, (backgrounds[:20], backgrounds[20:40], backgrounds[40:])


def edges(train: np.ndarray, contexts: np.ndarray, shift: int) -> list[tuple[int, int, int]]:
    rows = []
    for b in map(int, contexts):
        for a in range(P):
            target_a = (a + shift) % P
            if train[a, b] and train[target_a, b]:
                rows.append((a, b, target_a))
    return rows


def infer_relation(table: np.ndarray, train: np.ndarray, key_contexts: np.ndarray, fit_contexts: np.ndarray, shift: int) -> dict:
    key_edges = edges(train, key_contexts, shift)
    counts = np.zeros((P, P), dtype=np.int64)
    for a, b, target_a in key_edges:
        counts[int(table[a, b]), int(table[target_a, b])] += 1
    totals = counts.sum(axis=1)
    supported = np.flatnonzero(totals >= THRESHOLDS["source_support_min"])
    mapping = np.full(P, -1, dtype=np.int64)
    mapping[supported] = np.argmax(counts[supported], axis=1)
    key_correct = int(sum(int(counts[source, mapping[source]]) for source in supported))
    key_total = int(sum(int(totals[source]) for source in supported))
    key_consistency = key_correct / max(key_total, 1)
    coverage = len(supported) / P
    injectivity = len(set(map(int, mapping[supported]))) / max(len(supported), 1)

    validation_edges = edges(train, fit_contexts, shift)
    validation_total = 0
    validation_correct = 0
    for a, b, target_a in validation_edges:
        source = int(table[a, b])
        if mapping[source] < 0:
            continue
        validation_total += 1
        validation_correct += int(mapping[source] == int(table[target_a, b]))
    validation_consistency = validation_correct / max(validation_total, 1)
    checks = {
        "key_edges": len(key_edges) >= THRESHOLDS["key_edge_min"],
        "coverage": coverage >= THRESHOLDS["source_coverage_min"],
        "key_consistency": key_consistency >= THRESHOLDS["key_consistency_min"],
        "injectivity": injectivity >= THRESHOLDS["mapping_injectivity_min"],
        "validation_edges": validation_total >= THRESHOLDS["validation_edge_min"],
        "validation_consistency": validation_consistency >= THRESHOLDS["validation_consistency_min"],
    }
    return {
        "shift": shift,
        "key_edge_count": len(key_edges),
        "supported_source_count": len(supported),
        "source_coverage": coverage,
        "key_consistency": key_consistency,
        "mapping_injectivity": injectivity,
        "validation_edge_count": validation_total,
        "validation_consistency": validation_consistency,
        "eligible": all(checks.values()),
        "checks": checks,
    }


def evaluate(table: np.ndarray, seed: int) -> dict:
    train, (key_contexts, fit_contexts, test_contexts) = split_material(table, seed)
    relations = [infer_relation(table, train, key_contexts, fit_contexts, shift) for shift in TRANSFORMS]
    return {
        "relations": relations,
        "eligible_count": sum(row["eligible"] for row in relations),
        "key_context_count": len(key_contexts),
        "fit_context_count": len(fit_contexts),
        "test_context_count": len(test_contexts),
        "test_labels_used": False,
        "holdout_used": False,
    }


def main() -> None:
    systems = []
    for index, (inner, outer) in enumerate(((8, 47), (9, 49), (14, 53), (18, 59))):
        systems.append({
            "name": f"excluded_structured_{index}",
            "result": evaluate(task_table(inner, outer), 1_174_000 + index * 101),
        })
    rng = np.random.default_rng(1_174_999)
    systems.append({
        "name": "excluded_random_table",
        "result": evaluate(rng.integers(0, P, size=(P, P), dtype=np.int64), 1_174_999),
    })
    payload = {
        "scope": "exclusion_only_non_evidential_pilot",
        "thresholds": THRESHOLDS,
        "systems": systems,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
