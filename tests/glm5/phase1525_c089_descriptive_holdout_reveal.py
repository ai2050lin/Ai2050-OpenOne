#!/usr/bin/env python3
"""Phase1525: reveal C089 holdouts for descriptive, not semantic, replication."""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1521_c089_natural_relation_observation_contract"
ATLAS = RESULT / "phase1523_c089_truth_contrast_atlas"
FREEZE = RESULT / "phase1524_c089_discovery_observation_freeze"
OUT = RESULT / "phase1525_c089_descriptive_holdout_reveal"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

TOPK = 26


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else 0.0


def top(vector: np.ndarray) -> set[int]:
    return set(np.argpartition(np.abs(vector), -TOPK)[-TOPK:].tolist())


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1525 exists")
    parent = core.load(FREEZE / "analysis/final.json")
    parent_audit = core.load(FREEZE / "audit/independent_final_audit.json")
    freeze = core.load(FREEZE / "protocol/frozen_descriptive_predictions.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1525_c089_descriptive_holdout_reveal" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1524 authorization missing")
    if parent["freeze_sha256"] != freeze["freeze_sha256"]:
        raise RuntimeError("freeze mismatch")
    mean = np.load(ATLAS / "atlas/partition_family_truth_contrast_mean.float32.npy", mmap_mode="r")
    frozen = np.load(FREEZE / "protocol/frozen_discovery_centroids.float32.npy")
    families, roles = protocol["families"], protocol["roles"]
    thresholds = freeze["thresholds"]
    results = {}
    for fi, family in enumerate(families):
        candidate = freeze["family_candidates"][family]
        state, ri = candidate["state"], roles.index(candidate["role"])
        discovery = np.asarray(frozen[candidate["vector_index"]], dtype=np.float64)
        partitions = {}
        centroids = {}
        for pi, partition in ((1, "confirmation"), (2, "lockbox")):
            vectors = [np.asarray(mean[pi, fi, ui, state, ri], dtype=np.float64) for ui in range(2)]
            centroid = np.mean(vectors, axis=0)
            centroids[partition] = centroid
            partitions[partition] = {
                "mean_discovery_cosine": float(np.mean([cosine(vector, discovery) for vector in vectors])),
                "surface_cosine": cosine(vectors[0], vectors[1]),
                "mean_top26_jaccard": float(np.mean([jaccard(top(vector), top(discovery)) for vector in vectors])),
                "mean_norm": float(np.mean([np.linalg.norm(vector) for vector in vectors])),
            }
        confirmation_lockbox = cosine(centroids["confirmation"], centroids["lockbox"])
        components = {
            "confirmation_discovery": partitions["confirmation"]["mean_discovery_cosine"] >= thresholds["discovery_centroid_cosine"],
            "lockbox_discovery": partitions["lockbox"]["mean_discovery_cosine"] >= thresholds["discovery_centroid_cosine"],
            "confirmation_lockbox": confirmation_lockbox >= thresholds["confirmation_lockbox_cosine"],
            "confirmation_surface": partitions["confirmation"]["surface_cosine"] >= thresholds["within_partition_surface_cosine"],
            "lockbox_surface": partitions["lockbox"]["surface_cosine"] >= thresholds["within_partition_surface_cosine"],
            "confirmation_coordinates": partitions["confirmation"]["mean_top26_jaccard"] >= thresholds["top26_jaccard"],
            "lockbox_coordinates": partitions["lockbox"]["mean_top26_jaccard"] >= thresholds["top26_jaccard"],
        }
        results[family] = {
            "candidate": {"state": state, "role": candidate["role"]}, "partitions": partitions,
            "confirmation_lockbox_cosine": confirmation_lockbox, "components": components,
            "descriptive_replication_all_components": all(components.values()),
            "semantic_replication": "not authorized because behavior qualification failed before Hidden-State interpretation",
        }
    shared = freeze["shared_candidate"]
    shared_discovery = np.asarray(frozen[shared["vector_index"]], dtype=np.float64)
    shared_results = {}
    for pi, partition in ((1, "confirmation"), (2, "lockbox")):
        family_vectors = [np.asarray(mean[pi, fi, :, shared["state"], roles.index(shared["role"])], dtype=np.float64).mean(axis=0) for fi in range(3)]
        pairwise = [cosine(family_vectors[i], family_vectors[j]) for i, j in combinations(range(3), 2)]
        centroid = np.mean(family_vectors, axis=0)
        shared_results[partition] = {
            "discovery_cosine": cosine(centroid, shared_discovery),
            "cross_family_cosine": float(np.mean(pairwise)),
            "mean_family_discovery_cosine": float(np.mean([cosine(vector, shared_discovery) for vector in family_vectors])),
            "centroid_norm": float(np.linalg.norm(centroid)),
        }
    shared_components = {
        "confirmation_discovery": shared_results["confirmation"]["discovery_cosine"] >= thresholds["discovery_centroid_cosine"],
        "lockbox_discovery": shared_results["lockbox"]["discovery_cosine"] >= thresholds["discovery_centroid_cosine"],
        "confirmation_cross_family": shared_results["confirmation"]["cross_family_cosine"] >= thresholds["within_partition_surface_cosine"],
        "lockbox_cross_family": shared_results["lockbox"]["cross_family_cosine"] >= thresholds["within_partition_surface_cosine"],
    }
    summary = {
        "phase": 1525, "campaign": "C089", "freeze_sha256": freeze["freeze_sha256"],
        "family_results": results,
        "shared_result": {"candidate": {"state": shared["state"], "role": shared["role"]}, "partitions": shared_results, "components": shared_components, "descriptive_replication_all_components": all(shared_components.values())},
        "semantic_validation_authorized": False,
        "strict_conclusion": "holdout repetition can establish a stable task-associated response geometry, but cannot establish natural relation semantics because the frozen behavior gate failed for every family",
    }
    checks = {
        "freeze": summary["freeze_sha256"] == parent["freeze_sha256"],
        "families": set(results) == set(families),
        "finite": all(np.isfinite(value) for result in results.values() for partition in result["partitions"].values() for value in partition.values()),
        "components": all(len(result["components"]) == 7 for result in results.values()),
        "shared": len(shared_components) == 4 and all(np.isfinite(value) for partition in shared_results.values() for value in partition.values()),
        "semantic_block": not summary["semantic_validation_authorized"],
        "thresholds_unchanged": thresholds == protocol["validation_components"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary["checks"] = checks
    core.save(OUT / "analysis/descriptive_holdout_reveal_summary.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1525, "campaign": "C089", "status": "descriptive_holdouts_revealed_semantic_claim_blocked", "authorization": "run_phase1526_c089_full_dimensional_diagnostics"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
