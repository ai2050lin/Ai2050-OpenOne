#!/usr/bin/env python3
"""Phase1532: canonical discovery-only observation and descriptive freeze."""
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
CAPTURE = RESULT / "phase1530_c090_canonical_full_recapture"
ATLAS = RESULT / "phase1531_c090_canonical_truth_contrast_atlas"
OUT = RESULT / "phase1532_c090_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1524_c089_discovery_observation_freeze import cosine, jaccard, select_candidate, top_indices


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1532 exists")
    parent = core.load(ATLAS / "analysis/final.json")
    parent_audit = core.load(ATLAS / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    capture = core.load(CAPTURE / "analysis/canonical_behavior_and_capture_summary.json")
    if parent["authorization"] != "run_phase1532_c090_discovery_observation_freeze" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1531 authorization missing")
    mean = np.load(ATLAS / "atlas/canonical_partition_family_truth_mean.float32.npy", mmap_mode="r")
    group = np.load(ATLAS / "atlas/canonical_group_truth_contrast.float16.npy", mmap_mode="r")
    group_index = core.rows(ATLAS / "atlas/canonical_group_truth_contrast_index.jsonl")
    families, roles = protocol["families"], protocol["roles"]
    observations, candidates, vectors = [], {}, []
    for fi, family in enumerate(families):
        indices = [row["group_index"] for row in group_index if row["partition"] == "response_discovery" and row["family"] == family]
        family_rows = []
        for state in range(37):
            for ri, role in enumerate(roles):
                a = np.asarray(mean[0, fi, 0, state, ri], dtype=np.float64)
                b = np.asarray(mean[0, fi, 1, state, ri], dtype=np.float64)
                alignment = [cosine(np.asarray(group[gi, ui, state, ri], dtype=np.float64), (a, b)[ui]) for gi in indices for ui in range(2)]
                row = {"family": family, "state": state, "role": role, "surface_cosine": cosine(a, b), "mean_norm": float((np.linalg.norm(a) + np.linalg.norm(b)) / 2), "group_consistency": float(np.mean(alignment)), "top26_jaccard": jaccard(top_indices(a), top_indices(b))}
                observations.append(row)
                family_rows.append(row)
        candidate = select_candidate(family_rows)
        state, ri = candidate["state"], roles.index(candidate["role"])
        vector = np.asarray(mean[0, fi, :, state, ri], dtype=np.float32).mean(axis=0)
        candidate.update({"vector_index": len(vectors), "behavior_qualified": family in capture["behavior_qualified_families"], "semantic_interpretation_authorized": False})
        candidates[family] = candidate
        vectors.append(vector)
    shared_rows = []
    for state in range(1, 37):
        for ri, role in enumerate(roles):
            family_vectors = [np.asarray(mean[0, fi, :, state, ri], dtype=np.float64).mean(axis=0) for fi in range(3)]
            shared_rows.append({"state": state, "role": role, "cross_family_cosine": float(np.mean([cosine(family_vectors[i], family_vectors[j]) for i, j in combinations(range(3), 2)])), "mean_norm": float(np.mean([np.linalg.norm(vector) for vector in family_vectors]))})
    peak = max(row["mean_norm"] for row in shared_rows)
    shared = max((row for row in shared_rows if row["mean_norm"] >= 0.25 * peak), key=lambda row: (row["cross_family_cosine"], row["mean_norm"], -row["state"], row["role"]))
    shared_vector = np.asarray(mean[0, :, :, shared["state"], roles.index(shared["role"])], dtype=np.float32).mean(axis=(0, 1))
    shared.update({"vector_index": len(vectors), "semantic_interpretation_authorized": False})
    vectors.append(shared_vector)
    vector_path = OUT / "protocol/frozen_canonical_discovery_centroids.float32.npy"
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vector_path, np.stack(vectors))
    observation_path = OUT / "analysis/canonical_discovery_state_role_observations.jsonl"
    core.write_rows(observation_path, observations)
    freeze = {
        "phase": 1532, "campaign": "C090", "source_partition": "response_discovery",
        "engine": capture["engine"], "behavior_qualified_families": capture["behavior_qualified_families"],
        "semantic_validation_authorized": False, "descriptive_holdout_validation_authorized": True,
        "family_candidates": candidates, "shared_candidate": shared, "thresholds": protocol["validation_components"],
        "centroid_sha256": core.sha(vector_path), "observation_sha256": core.sha(observation_path), "holdout_hidden_accessed": False,
    }
    freeze["freeze_sha256"] = core.digest(freeze)
    core.save(OUT / "protocol/frozen_canonical_descriptive_predictions.json", freeze)
    checks = {
        "rows": len(observations) == 444, "state0": all(row["mean_norm"] == 0.0 for row in observations if row["state"] == 0),
        "source_zero": all(row["mean_norm"] == 0.0 for row in observations if row["role"] == "source_word"),
        "candidates": set(candidates) == set(families), "vectors": list(np.load(vector_path).shape) == [4, 2560],
        "unqualified": capture["behavior_qualified_families"] == [] and not freeze["semantic_validation_authorized"],
        "blinded": not freeze["holdout_hidden_accessed"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary = {"phase": 1532, "campaign": "C090", "family_candidates": candidates, "shared_candidate": shared, "freeze_sha256": freeze["freeze_sha256"], "checks": checks}
    core.save(OUT / "analysis/canonical_discovery_observation_summary.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1532, "campaign": "C090", "status": "canonical_discovery_frozen_without_semantic_qualification", "freeze_sha256": freeze["freeze_sha256"], "authorization": "run_phase1533_c090_holdout_and_artifact_adjudication"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
