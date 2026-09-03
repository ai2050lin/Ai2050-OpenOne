#!/usr/bin/env python3
"""Phase1524: discovery-only observation and descriptive validation freeze for C089."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1521_c089_natural_relation_observation_contract"
CAPTURE = RESULT / "phase1522_c089_unified_forward_capture"
ATLAS = RESULT / "phase1523_c089_truth_contrast_atlas"
OUT = RESULT / "phase1524_c089_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

TOPK = 26


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else 0.0


def top_indices(vector: np.ndarray, k: int = TOPK) -> set[int]:
    return set(np.argpartition(np.abs(vector), -k)[-k:].tolist())


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right)


def select_candidate(rows: list[dict]) -> dict:
    peak = max(row["mean_norm"] for row in rows)
    eligible = [row for row in rows if row["state"] > 0 and row["mean_norm"] >= 0.25 * peak]
    winner = max(eligible, key=lambda row: (row["surface_cosine"] + row["group_consistency"] + row["top26_jaccard"], row["mean_norm"], -row["state"], row["role"]))
    return {**winner, "selection_score": winner["surface_cosine"] + winner["group_consistency"] + winner["top26_jaccard"], "peak_norm": peak}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1524 exists")
    parent = core.load(ATLAS / "analysis/final.json")
    parent_audit = core.load(ATLAS / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    capture = core.load(CAPTURE / "analysis/unified_behavior_and_capture_summary.json")
    if parent["authorization"] != "run_phase1524_c089_discovery_observation_freeze" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1523 authorization missing")
    mean = np.load(ATLAS / "atlas/partition_family_truth_contrast_mean.float32.npy", mmap_mode="r")
    group = np.load(ATLAS / "atlas/group_truth_contrast.float16.npy", mmap_mode="r")
    group_index = core.rows(ATLAS / "atlas/group_truth_contrast_index.jsonl")
    families, roles = protocol["families"], protocol["roles"]
    observations, candidates, frozen_vectors = [], {}, []
    for fi, family in enumerate(families):
        indices = [row["group_index"] for row in group_index if row["partition"] == "response_discovery" and row["family"] == family]
        family_rows = []
        for state in range(37):
            for ri, role in enumerate(roles):
                a = np.asarray(mean[0, fi, 0, state, ri], dtype=np.float64)
                b = np.asarray(mean[0, fi, 1, state, ri], dtype=np.float64)
                alignments = []
                for gi in indices:
                    alignments.append(cosine(np.asarray(group[gi, 0, state, ri], dtype=np.float64), a))
                    alignments.append(cosine(np.asarray(group[gi, 1, state, ri], dtype=np.float64), b))
                row = {
                    "family": family, "state": state, "role": role,
                    "surface_cosine": cosine(a, b), "mean_norm": float((np.linalg.norm(a) + np.linalg.norm(b)) / 2),
                    "group_consistency": float(np.mean(alignments)), "top26_jaccard": jaccard(top_indices(a), top_indices(b)),
                }
                observations.append(row)
                family_rows.append(row)
        candidate = select_candidate(family_rows)
        state, ri = candidate["state"], roles.index(candidate["role"])
        vector = (np.asarray(mean[0, fi, 0, state, ri], dtype=np.float32) + np.asarray(mean[0, fi, 1, state, ri], dtype=np.float32)) / 2
        candidate["vector_index"] = len(frozen_vectors)
        candidate["behavior_qualified"] = family in capture["behavior_qualified_families"]
        candidate["semantic_interpretation_authorized"] = False
        candidates[family] = candidate
        frozen_vectors.append(vector)
    shared_rows = []
    for state in range(1, 37):
        for ri, role in enumerate(roles):
            vectors = [np.asarray(mean[0, fi, :, state, ri], dtype=np.float64).mean(axis=0) for fi in range(len(families))]
            pairwise = [cosine(vectors[i], vectors[j]) for i, j in combinations(range(len(vectors)), 2)]
            shared_rows.append({
                "state": state, "role": role, "cross_family_cosine": float(np.mean(pairwise)),
                "mean_norm": float(np.mean([np.linalg.norm(vector) for vector in vectors])),
            })
    peak = max(row["mean_norm"] for row in shared_rows)
    eligible = [row for row in shared_rows if row["mean_norm"] >= 0.25 * peak]
    shared = max(eligible, key=lambda row: (row["cross_family_cosine"], row["mean_norm"], -row["state"], row["role"]))
    shared_vector = np.asarray(mean[0, :, :, shared["state"], roles.index(shared["role"])], dtype=np.float32).mean(axis=(0, 1))
    shared["vector_index"] = len(frozen_vectors)
    shared["semantic_interpretation_authorized"] = False
    frozen_vectors.append(shared_vector)
    vector_path = OUT / "protocol/frozen_discovery_centroids.float32.npy"
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vector_path, np.stack(frozen_vectors))
    observation_path = OUT / "analysis/discovery_state_role_observations.jsonl"
    core.write_rows(observation_path, observations)
    freeze = {
        "phase": 1524, "campaign": "C089", "source_partition": "response_discovery",
        "behavior_qualified_families": capture["behavior_qualified_families"],
        "semantic_validation_authorized": False,
        "descriptive_holdout_validation_authorized": True,
        "family_candidates": candidates, "shared_candidate": shared,
        "thresholds": protocol["validation_components"],
        "scope": "selected structures are task-input-associated descriptive response candidates because no family passed the frozen behavior qualification",
        "centroid_sha256": core.sha(vector_path), "observation_sha256": core.sha(observation_path),
        "holdout_hidden_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    freeze["freeze_sha256"] = core.digest(freeze)
    core.save(OUT / "protocol/frozen_descriptive_predictions.json", freeze)
    checks = {
        "observation_rows": len(observations) == 3 * 37 * 4,
        "discovery_only": freeze["source_partition"] == "response_discovery" and not freeze["holdout_hidden_accessed"],
        "candidates": set(candidates) == set(families),
        "finite": all(np.isfinite(value) for row in observations for key, value in row.items() if key not in {"family", "role"}),
        "state0_zero": all(row["mean_norm"] == 0.0 for row in observations if row["state"] == 0),
        "unqualified": capture["behavior_qualified_families"] == [] and not freeze["semantic_validation_authorized"],
        "vectors": list(np.load(vector_path).shape) == [4, 2560],
        "scope": "descriptive" in freeze["scope"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary = {
        "phase": 1524, "campaign": "C089", "family_candidates": candidates, "shared_candidate": shared,
        "checks": checks, "freeze_sha256": freeze["freeze_sha256"],
        "result": "exploratory full-state structures frozen for descriptive replication only; semantic validation remains unauthorized",
    }
    core.save(OUT / "analysis/discovery_observation_summary.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1524, "campaign": "C089", "status": "discovery_observation_frozen_without_semantic_qualification", "freeze_sha256": freeze["freeze_sha256"], "authorization": "run_phase1525_c089_descriptive_holdout_reveal"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
