#!/usr/bin/env python3
"""Phase1533: canonical holdout reveal and old-camera artifact adjudication."""
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
OLD_ATLAS = RESULT / "phase1523_c089_truth_contrast_atlas"
OLD_FREEZE = RESULT / "phase1524_c089_discovery_observation_freeze"
ATLAS = RESULT / "phase1531_c090_canonical_truth_contrast_atlas"
FREEZE = RESULT / "phase1532_c090_discovery_observation_freeze"
PARENT = FREEZE
OUT = RESULT / "phase1533_c090_holdout_and_artifact_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1524_c089_discovery_observation_freeze import cosine, jaccard, top_indices


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1533 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    freeze = core.load(FREEZE / "protocol/frozen_canonical_descriptive_predictions.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1533_c090_holdout_and_artifact_adjudication" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1532 authorization missing")
    mean = np.load(ATLAS / "atlas/canonical_partition_family_truth_mean.float32.npy", mmap_mode="r")
    old_mean = np.load(OLD_ATLAS / "atlas/partition_family_truth_contrast_mean.float32.npy", mmap_mode="r")
    vectors = np.load(FREEZE / "protocol/frozen_canonical_discovery_centroids.float32.npy")
    old_freeze = core.load(OLD_FREEZE / "protocol/frozen_descriptive_predictions.json")
    families, partitions, roles = protocol["families"], protocol["partitions"], protocol["roles"]
    thresholds = freeze["thresholds"]
    results, execution_comparison = {}, {}
    for fi, family in enumerate(families):
        candidate = freeze["family_candidates"][family]
        state, ri = candidate["state"], roles.index(candidate["role"])
        discovery = np.asarray(vectors[candidate["vector_index"]], dtype=np.float64)
        cells, centroids = {}, {}
        for pi, partition in ((1, "confirmation"), (2, "lockbox")):
            surface_vectors = [np.asarray(mean[pi, fi, ui, state, ri], dtype=np.float64) for ui in range(2)]
            centroid = np.mean(surface_vectors, axis=0)
            centroids[partition] = centroid
            cells[partition] = {
                "mean_discovery_cosine": float(np.mean([cosine(vector, discovery) for vector in surface_vectors])),
                "surface_cosine": cosine(*surface_vectors),
                "mean_top26_jaccard": float(np.mean([jaccard(top_indices(vector), top_indices(discovery)) for vector in surface_vectors])),
                "mean_norm": float(np.mean([np.linalg.norm(vector) for vector in surface_vectors])),
            }
        conf_lock = cosine(centroids["confirmation"], centroids["lockbox"])
        components = {
            "confirmation_discovery": cells["confirmation"]["mean_discovery_cosine"] >= thresholds["discovery_centroid_cosine"],
            "lockbox_discovery": cells["lockbox"]["mean_discovery_cosine"] >= thresholds["discovery_centroid_cosine"],
            "confirmation_lockbox": conf_lock >= thresholds["confirmation_lockbox_cosine"],
            "confirmation_surface": cells["confirmation"]["surface_cosine"] >= thresholds["within_partition_surface_cosine"],
            "lockbox_surface": cells["lockbox"]["surface_cosine"] >= thresholds["within_partition_surface_cosine"],
            "confirmation_coordinates": cells["confirmation"]["mean_top26_jaccard"] >= thresholds["top26_jaccard"],
            "lockbox_coordinates": cells["lockbox"]["mean_top26_jaccard"] >= thresholds["top26_jaccard"],
        }
        results[family] = {"candidate": {"state": state, "role": candidate["role"]}, "partitions": cells, "confirmation_lockbox_cosine": conf_lock, "components": components, "descriptive_replication_all_components": all(components.values()), "semantic_replication": False}
        execution_comparison[family] = {
            "old_candidate_same_location": old_freeze["family_candidates"][family]["state"] == state and old_freeze["family_candidates"][family]["role"] == candidate["role"],
            "partition_mean_old_new_cosine": {partition: cosine(np.asarray(old_mean[pi, fi, :, state, ri], dtype=np.float64).mean(axis=0), np.asarray(mean[pi, fi, :, state, ri], dtype=np.float64).mean(axis=0)) for pi, partition in enumerate(partitions)},
        }
    shared = freeze["shared_candidate"]
    shared_discovery = np.asarray(vectors[shared["vector_index"]], dtype=np.float64)
    shared_results = {}
    shared_centroids = []
    for pi, partition in ((1, "confirmation"), (2, "lockbox")):
        family_vectors = [np.asarray(mean[pi, fi, :, shared["state"], roles.index(shared["role"])], dtype=np.float64).mean(axis=0) for fi in range(3)]
        centroid = np.mean(family_vectors, axis=0)
        shared_centroids.append(centroid)
        shared_results[partition] = {
            "discovery_cosine": cosine(centroid, shared_discovery),
            "cross_family_cosine": float(np.mean([cosine(family_vectors[i], family_vectors[j]) for i, j in combinations(range(3), 2)])),
            "mean_family_discovery_cosine": float(np.mean([cosine(vector, shared_discovery) for vector in family_vectors])),
        }
    shared_components = {
        "confirmation_discovery": shared_results["confirmation"]["discovery_cosine"] >= thresholds["discovery_centroid_cosine"],
        "lockbox_discovery": shared_results["lockbox"]["discovery_cosine"] >= thresholds["discovery_centroid_cosine"],
        "confirmation_cross_family": shared_results["confirmation"]["cross_family_cosine"] >= thresholds["within_partition_surface_cosine"],
        "lockbox_cross_family": shared_results["lockbox"]["cross_family_cosine"] >= thresholds["within_partition_surface_cosine"],
    }
    source_old = float(np.max(np.abs(np.asarray(np.load(OLD_ATLAS / "atlas/group_truth_contrast.float16.npy", mmap_mode="r")[:, :, :, 0], dtype=np.float32))))
    source_new = float(np.max(np.abs(np.asarray(np.load(ATLAS / "atlas/canonical_group_truth_contrast.float16.npy", mmap_mode="r")[:, :, :, 0], dtype=np.float32))))
    summary = {
        "phase": 1533, "campaign": "C090", "freeze_sha256": freeze["freeze_sha256"],
        "family_results": results,
        "shared_result": {"candidate": {"state": shared["state"], "role": shared["role"]}, "partitions": shared_results, "confirmation_lockbox_cosine": cosine(*shared_centroids), "components": shared_components, "descriptive_replication_all_components": all(shared_components.values())},
        "execution_adjudication": {
            "source_truth_contrast_old_max_abs": source_old, "source_truth_contrast_canonical_max_abs": source_new,
            "family_candidate_comparison": execution_comparison,
            "finding": "the old engine introduced a causal-prefix artifact, but the selected target and boundary response candidates survive the canonical engine at nearly the same locations; survival does not grant semantic qualification",
        },
        "semantic_validation_authorized": False,
    }
    checks = {
        "freeze": summary["freeze_sha256"] == parent["freeze_sha256"],
        "families": set(results) == set(families), "family_components": all(len(row["components"]) == 7 for row in results.values()),
        "shared": len(shared_components) == 4, "source_repair": source_old > 1e-2 and source_new == 0.0,
        "candidate_locations": all(row["old_candidate_same_location"] for row in execution_comparison.values()),
        "semantic_block": not summary["semantic_validation_authorized"],
        "thresholds": thresholds == protocol["validation_components"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary["checks"] = checks
    core.save(OUT / "analysis/holdout_and_artifact_adjudication.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1533, "campaign": "C090", "status": "canonical_descriptive_replication_complete_semantic_block_retained", "authorization": "run_phase1534_c089_c090_major_stage_closure"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
