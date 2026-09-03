#!/usr/bin/env python3
"""Phase1477: synthesize C082 and freeze future fresh-material predictions."""
from __future__ import annotations

import itertools
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
ATLAS = RESULT / "phase1476_c082_coordinate_atlas"
OUT = RESULT / "phase1477_c082_atlas_synthesis"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def scalar_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def relation_vectors(means: np.ndarray, state: int, role_index: int) -> list[np.ndarray]:
    return [np.asarray(means[index, :, :, state, role_index], dtype=np.float32).mean(axis=(0, 1)) for index in range(means.shape[0])]


def pairwise_summary(vectors: list[np.ndarray]) -> dict:
    values = [scalar_cosine(vectors[left], vectors[right]) for left, right in itertools.combinations(range(len(vectors)), 2)]
    return {"minimum": min(values), "mean": statistics.mean(values), "maximum": max(values), "all_values": values}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1477 exists")
    parent = core.load(ATLAS / "analysis/final.json")
    parent_audit = core.load(ATLAS / "audit/independent_final_audit.json")
    metadata = core.load(ATLAS / "analysis/atlas_metadata.json")
    if parent["authorization"] != "run_phase1477_c082_atlas_audit_and_synthesis" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1476 did not authorize Phase1477")
    means = np.load(ATLAS / "atlas/mean_effect.float32.npy", mmap_mode="r")
    layer_rows = core.rows(ATLAS / "analysis/layer_role_metrics.jsonl")
    onset_rows = core.rows(ATLAS / "analysis/onset_metrics.jsonl")
    panel_rows = core.rows(ATLAS / "analysis/panel_stability.jsonl")
    roles = metadata["roles"]
    relations = metadata["relations"]
    ql = roles.index("query_label")
    qr = roles.index("query_relation")
    boundary = roles.index("boundary")
    trajectory = {}
    for name, state, role_index in (
        ("query_label_state0", 0, ql),
        ("query_label_state5", 5, ql),
        ("query_relation_state0", 0, qr),
        ("query_relation_state5", 5, qr),
        ("boundary_state24", 24, boundary),
        ("boundary_state32", 32, boundary),
        ("boundary_state35", 35, boundary),
        ("boundary_state36", 36, boundary),
    ):
        trajectory[name] = pairwise_summary(relation_vectors(means, state, role_index))
    boundary_peaks = {}
    for relation in relations:
        values = []
        for state in range(37):
            cells = [row["mean_l2_norm"] for row in layer_rows if row["relation"] == relation and row["role"] == "boundary" and row["state"] == state]
            values.append(statistics.mean(cells))
        state = max(range(37), key=values.__getitem__)
        stability = next(row for row in panel_rows if row["relation"] == relation and row["role"] == "boundary" and row["state"] == state)
        cells = [row for row in layer_rows if row["relation"] == relation and row["role"] == "boundary" and row["state"] == state]
        boundary_peaks[relation] = {
            "state": state,
            "mean_norm": values[state],
            "minimum_panel_cosine": stability["minimum_panel_cosine"],
            "median_k90": statistics.median(row["k90"] for row in cells),
            "median_maximum_coordinate_energy_share": statistics.median(row["maximum_coordinate_energy_share"] for row in cells),
        }
    boundary35_cells = [row for row in layer_rows if row["role"] == "boundary" and row["state"] == 35]
    boundary35_vectors = relation_vectors(means, 35, boundary)
    common = np.mean(np.stack(boundary35_vectors, axis=0), axis=0, dtype=np.float32)
    top_sets = [set(np.argpartition(vector * vector, -26)[-26:].tolist()) for vector in boundary35_vectors]
    intersection = sorted(set.intersection(*top_sets))
    union = sorted(set.union(*top_sets))
    intersection_energy = {relation: float(np.sum(vector[intersection] ** 2) / np.sum(vector ** 2)) for relation, vector in zip(relations, boundary35_vectors)}
    union_energy = {relation: float(np.sum(vector[union] ** 2) / np.sum(vector ** 2)) for relation, vector in zip(relations, boundary35_vectors)}
    common_cosines = {relation: scalar_cosine(vector, common) for relation, vector in zip(relations, boundary35_vectors)}
    onset = {}
    for role in roles:
        cells = [row for row in onset_rows if row["role"] == role]
        onset[role] = {
            key: (statistics.median([row[key] for row in cells if row[key] is not None]) if any(row[key] is not None for row in cells) else None)
            for key in ("first_state_10pct", "first_state_50pct", "first_state_90pct", "peak_state")
        }
    top_pairwise_jaccard = []
    for left, right in itertools.combinations(range(6), 2):
        top_pairwise_jaccard.append(len(top_sets[left] & top_sets[right]) / len(top_sets[left] | top_sets[right]))
    OUT.joinpath("frozen").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "frozen/common_boundary_state35_vector.float32.npy", common.astype(np.float32))
    synthesis = {
        "phase": 1477,
        "campaign": "C082",
        "trajectory_pairwise_cosine": trajectory,
        "boundary_relation_peaks": boundary_peaks,
        "boundary_state35": {
            "median_k50": statistics.median(row["k50"] for row in boundary35_cells),
            "median_k80": statistics.median(row["k80"] for row in boundary35_cells),
            "median_k90": statistics.median(row["k90"] for row in boundary35_cells),
            "median_maximum_coordinate_energy_share": statistics.median(row["maximum_coordinate_energy_share"] for row in boundary35_cells),
            "mean_sample_to_mean_cosine": statistics.mean(row["mean_sample_to_mean_cosine"] for row in boundary35_cells),
            "top_1pct_pairwise_jaccard": {"minimum": min(top_pairwise_jaccard), "mean": statistics.mean(top_pairwise_jaccard), "maximum": max(top_pairwise_jaccard)},
            "top_1pct_all_relation_intersection": intersection,
            "top_1pct_all_relation_union": union,
            "intersection_energy_fraction": intersection_energy,
            "union_energy_fraction": union_energy,
            "cosine_to_common_vector": common_cosines,
            "common_vector_norm": float(np.linalg.norm(common)),
        },
        "median_onset_by_role": onset,
        "interpretation": "relation-specific lexical difference directions at query tokens converge to a highly shared, distributed late-boundary decision response",
        "claim_boundary": "retrospective C079 observation; the shared late direction is a candidate task/decision carrier, not a relation-semantic vector",
    }
    core.save(OUT / "analysis/synthesis.json", synthesis)
    manifest = {
        "phase": 1477,
        "campaign": "C082",
        "source_atlas_sha256": metadata["files"]["mean_effect.float32.npy"]["sha256"],
        "common_vector_sha256": core.sha(OUT / "frozen/common_boundary_state35_vector.float32.npy"),
        "frozen_coordinates": {"boundary_state35_top_1pct_intersection": intersection, "boundary_state35_top_1pct_union": union},
        "future_fresh_material_predictions": [
            {"id": "P082-1", "prediction": "query-label state0 relation-pair mean vectors remain non-common", "gate": "maximum pairwise relation cosine <= 0.30"},
            {"id": "P082-2", "prediction": "late boundary relation effects converge to one common full-vector direction", "gate": "state35 minimum pairwise relation cosine >= 0.90 and each relation cosine to frozen common vector >= 0.90"},
            {"id": "P082-3", "prediction": "the late boundary response remains distributed", "gate": "state35 median k90 between 900 and 1400 and median maximum-coordinate energy share <= 0.06"},
            {"id": "P082-4", "prediction": "a small recurrent coordinate scaffold is reused without carrying most energy", "gate": "top-1-percent all-relation intersection has at least 12 coordinates and carries 0.05 to 0.20 energy in every relation"},
            {"id": "P082-5", "prediction": "boundary growth is late", "gate": "median first state reaching 50 percent of boundary peak lies in states 28 through 32 and all six relation peaks lie in states 34 through 36"},
        ],
        "required_future_design": "fresh behavior-qualified material, frozen interfaces, raw full-coordinate capture, discovery before unopened confirmation/lockbox",
        "not_confirmed_here": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest["freeze_sha256"] = core.digest(manifest)
    core.save(OUT / "frozen/future_prediction_manifest.json", manifest)
    checks = {
        "source_audit": parent_audit["all_checks_passed"],
        "early_non_common": trajectory["query_label_state0"]["maximum"] <= 0.30,
        "late_common": trajectory["boundary_state35"]["minimum"] >= 0.90 and min(common_cosines.values()) >= 0.90,
        "distributed": 900 <= synthesis["boundary_state35"]["median_k90"] <= 1400 and synthesis["boundary_state35"]["median_maximum_coordinate_energy_share"] <= 0.06,
        "recurrent_scaffold": len(intersection) >= 12 and all(0.05 <= value <= 0.20 for value in intersection_energy.values()),
        "late_growth": 28 <= onset["boundary"]["first_state_50pct"] <= 32 and all(34 <= value["state"] <= 36 for value in boundary_peaks.values()),
        "retrospective_only": manifest["not_confirmed_here"],
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "analysis/final.json", {"phase": 1477, "campaign": "C082", "synthesis_complete": True, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "freeze_sha256": manifest["freeze_sha256"], "authorization": "run_phase1478_c082_campaign_closure"})
    print(json.dumps({"checks": checks, "trajectory": trajectory, "boundary_peaks": boundary_peaks, "boundary_state35": synthesis["boundary_state35"], "freeze_sha256": manifest["freeze_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
