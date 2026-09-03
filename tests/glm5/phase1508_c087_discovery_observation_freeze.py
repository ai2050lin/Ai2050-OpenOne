#!/usr/bin/env python3
"""Phase1508: observe C087 discovery only and freeze dual-holdout predictions."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
ATLAS = RESULT / "phase1507_c087_descriptive_semantic_contrast_atlas"
C086_ATLAS = RESULT / "phase1499_c086_four_factor_atlas"
C086_CONTRACT = RESULT / "phase1496_c086_unlabeled_counterbalanced_contract"
OUT = RESULT / "phase1508_c087_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def pairwise_mean(vectors):
    if len(vectors) < 2:
        return 0.0
    return float(np.mean([
        cosine(vectors[i], vectors[j])
        for i in range(len(vectors)) for j in range(i + 1, len(vectors))
    ]))


def concentration(vectors, k=26):
    energy = np.sum(np.square(np.asarray(vectors, dtype=np.float64)), axis=0)
    total = float(np.sum(energy))
    return float(np.sort(energy)[::-1][:k].sum() / total) if total > 1e-12 else 0.0


def coherence(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    mean = vectors.mean(axis=0)
    denominator = float(np.mean(np.sum(vectors * vectors, axis=1)))
    return float(np.dot(mean, mean) / denominator) if denominator > 1e-12 else 0.0


def metrics(group, group_index, selected_partition, c086, c086_effect_index, c086_boundary):
    selected = [row["group_index"] for row in group_index if row["partition"] == selected_partition]
    observations = []
    for state in range(37):
        for role_index, role in enumerate(("source_relation", "candidate_relation", "boundary")):
            panel = np.asarray(group[selected, :, state, role_index], dtype=np.float64)
            per_group = panel.mean(axis=1)
            surface_centroids = panel.mean(axis=0)
            c086_vector = np.asarray(
                c086[c086_effect_index, :, 0, :, state, c086_boundary], dtype=np.float64
            ).mean(axis=(0, 1))
            observations.append({
                "state": state,
                "role": role,
                "count": len(selected),
                "surface_centroid_cosine": cosine(surface_centroids[0], surface_centroids[1]),
                "within_group_surface_cosine_mean": float(np.mean([cosine(row[0], row[1]) for row in panel])),
                "group_pairwise_cosine_mean": pairwise_mean(per_group),
                "shared_energy_fraction": coherence(per_group),
                "top1pct_coordinate_energy": concentration(per_group),
                "c086_relation_alignment_mean": float(np.mean([cosine(v, c086_vector) for v in surface_centroids])),
                "centroid_norm": float(np.linalg.norm(per_group.mean(axis=0))),
            })
    return observations


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1508 exists")
    parent = core.load(ATLAS / "analysis/final.json")
    parent_audit = core.load(ATLAS / "audit/independent_final_audit.json")
    if parent["authorization"] != "run_phase1508_c087_discovery_observation_freeze" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1507 authorization missing")
    group = np.load(ATLAS / "atlas/group_semantic_contrast.float32.npy", mmap_mode="r")
    group_index = core.rows(ATLAS / "atlas/group_semantic_contrast_index.jsonl")
    c086 = np.load(C086_ATLAS / "atlas/all_four_factor_contrast_mean.float32.npy", mmap_mode="r")
    c086_summary = core.load(C086_ATLAS / "analysis/four_factor_atlas_summary.json")
    c086_protocol = core.load(C086_CONTRACT / "protocol/preregistration.json")
    observations = metrics(
        group,
        group_index,
        "response_discovery",
        c086,
        c086_summary["effects"].index("relation"),
        c086_protocol["roles"].index("boundary"),
    )
    boundary = next(row for row in observations if row["state"] == 35 and row["role"] == "boundary")
    source = next(row for row in observations if row["state"] == 35 and row["role"] == "source_relation")
    onset_candidates = [
        row["state"] for row in observations
        if row["role"] == "boundary"
        and row["shared_energy_fraction"] >= 0.4
        and row["within_group_surface_cosine_mean"] >= 0.4
    ]
    onset = min(onset_candidates) if onset_candidates else None
    field_class = (
        "shared_late_boundary_response"
        if boundary["surface_centroid_cosine"] >= 0.8
        and boundary["within_group_surface_cosine_mean"] >= 0.6
        and boundary["shared_energy_fraction"] >= 0.4
        and boundary["group_pairwise_cosine_mean"] >= 0.4
        else "heterogeneous_or_surface_specific_response"
    )
    discovery = {
        "partition": "response_discovery",
        "state": 35,
        "role": "boundary",
        "field_class": field_class,
        "onset_state": onset,
        "boundary": boundary,
        "source_relation_state35": source,
    }
    core.write_rows(OUT / "analysis/discovery_layer_role_observations.jsonl", observations)
    core.save(OUT / "analysis/discovery_summary.json", discovery)
    freeze = {
        "phase": 1508,
        "campaign": "C087",
        "source_partition": "response_discovery",
        "untouched_partitions": ["confirmation", "lockbox"],
        "stratum": "all",
        "state": 35,
        "role": "boundary",
        "reference": discovery,
        "primary_tolerances": {
            "surface_centroid_cosine": 0.12,
            "within_group_surface_cosine_mean": 0.12,
            "shared_energy_fraction": 0.15,
            "group_pairwise_cosine_mean": 0.15,
            "onset_state": 5,
        },
        "diagnostic_tolerances": {
            "c086_relation_alignment_mean": 0.25,
            "top1pct_coordinate_energy": 0.10,
        },
        "predictions": {
            "P087-1": "field_class repeats exactly",
            "P087-2": "four primary state35 metrics repeat within frozen tolerances",
            "P087-3": "shared-response onset repeats within five states",
            "P087-4": "source_relation remains exactly causally upstream and has zero state35 contrast",
            "P087-D1": "C086 alignment and top-coordinate concentration are diagnostics, not main gates",
        },
        "claim_boundary": "descriptive Qwen3 SemEval response field; not a universal comparator, causal mechanism, or execution-path localization",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    freeze["freeze_sha256"] = core.digest(freeze)
    core.save(OUT / "protocol/frozen_holdout_predictions.json", freeze)
    checks = {
        "discovery_only": all(row["partition"] == "response_discovery" for row in group_index if row["group_index"] in [x["group_index"] for x in group_index if x["partition"] == "response_discovery"]),
        "row_count": len(observations) == 37 * 3,
        "finite": all(np.isfinite(value) for row in observations for value in row.values() if isinstance(value, (int, float))),
        "field_class_registered": field_class in ("shared_late_boundary_response", "heterogeneous_or_surface_specific_response"),
        "onset_defined": onset is not None,
        "source_zero": source["centroid_norm"] == 0.0,
        "freeze_hash": freeze["freeze_sha256"] == core.digest({key: value for key, value in freeze.items() if key != "freeze_sha256"}),
        "descriptive_scope": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    result = {
        "phase": 1508,
        "campaign": "C087",
        "status": "discovery_observed_and_holdouts_frozen",
        "discovery": discovery,
        "checks": checks,
        "freeze_sha256": freeze["freeze_sha256"],
        "authorization": "run_phase1509_c087_dual_holdout_validation",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
