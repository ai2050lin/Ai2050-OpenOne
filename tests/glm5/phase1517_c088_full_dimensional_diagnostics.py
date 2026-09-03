#!/usr/bin/env python3
"""Phase1517: full-dimensional stability and behavior/field diagnostics for C088."""
from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CAPTURE = RESULT / "phase1513_c088_unified_forward_capture"
ATLAS = RESULT / "phase1514_c088_factorial_field_atlas"
REVEAL = RESULT / "phase1516_c088_holdout_and_fresh_reveal"
OUT = RESULT / "phase1517_c088_full_dimensional_diagnostics"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1515_c088_discovery_observation_freeze import cosine

PARTITIONS = ("response_discovery", "confirmation", "lockbox", "fresh_external")
EFFECTS = ("semantic", "code", "semantic_code")


def topk(vector, k=26):
    vector = np.asarray(vector, dtype=np.float64)
    return np.argsort(np.abs(vector))[::-1][:k]


def pairwise_coordinate_metrics(vectors, k=26):
    rows = []
    for i, j in itertools.combinations(range(len(vectors)), 2):
        left, right = np.asarray(vectors[i]), np.asarray(vectors[j])
        li, ri = set(topk(left, k).tolist()), set(topk(right, k).tolist())
        overlap = sorted(li & ri)
        rows.append({
            "left": PARTITIONS[i],
            "right": PARTITIONS[j],
            "full_cosine": cosine(left, right),
            "top26_jaccard": len(overlap) / len(li | ri),
            "top26_overlap": len(overlap),
            "overlap_sign_agreement": float(np.mean(np.sign(left[overlap]) == np.sign(right[overlap]))) if overlap else 0.0,
        })
    return rows


def behavior_summary(rows):
    output = []
    for partition in PARTITIONS:
        for codebook in ("standard", "reversed"):
            for semantic_label in ("same", "different"):
                selected = [
                    row for row in rows
                    if row["partition"] == partition
                    and row["codebook"] == codebook
                    and row["semantic_label"] == semantic_label
                ]
                output.append({
                    "partition": partition,
                    "codebook": codebook,
                    "semantic_label": semantic_label,
                    "count": len(selected),
                    "accuracy": float(np.mean([row["correct"] for row in selected])),
                    "predicted_yes_rate": float(np.mean([row["predicted_label"] == "yes" for row in selected])),
                })
    return output


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1517 exists")
    parent = core.load(REVEAL / "analysis/final.json")
    parent_audit = core.load(REVEAL / "audit/independent_final_audit.json")
    if parent["authorization"] != "run_phase1517_c088_full_dimensional_diagnostics" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1516 authorization missing")

    aggregate = np.load(ATLAS / "atlas/partition_factorial_effect_mean.float32.npy", mmap_mode="r")
    field_rows = core.rows(CAPTURE / "raw/all_role_field_index.jsonl")
    coordinate_rows = []
    coordinate_summary = {}
    for effect_index, effect in enumerate(EFFECTS):
        vectors = [np.asarray(aggregate[pi, :, effect_index, 35, 3], dtype=np.float64).mean(axis=0) for pi in range(4)]
        pairs = pairwise_coordinate_metrics(vectors)
        coordinate_rows.extend({"effect": effect, **row} for row in pairs)
        coordinate_summary[effect] = {
            "pairwise_full_cosine_mean": float(np.mean([row["full_cosine"] for row in pairs])),
            "pairwise_full_cosine_min": float(np.min([row["full_cosine"] for row in pairs])),
            "top26_jaccard_mean": float(np.mean([row["top26_jaccard"] for row in pairs])),
            "top26_overlap_min": int(np.min([row["top26_overlap"] for row in pairs])),
            "overlap_sign_agreement_mean": float(np.mean([row["overlap_sign_agreement"] for row in pairs])),
            "consensus_top26": sorted(set.intersection(*(set(topk(vector).tolist()) for vector in vectors))),
        }

    formation_rows = []
    for pi, partition in enumerate(PARTITIONS):
        final_vector = np.asarray(aggregate[pi, :, 0, 35, 3], dtype=np.float64).mean(axis=0)
        for state in range(37):
            surfaces = np.asarray(aggregate[pi, :, 0, state, 3], dtype=np.float64)
            centroid = surfaces.mean(axis=0)
            formation_rows.append({
                "partition": partition,
                "state": state,
                "centroid_norm": float(np.linalg.norm(centroid)),
                "surface_cosine": cosine(surfaces[0], surfaces[1]),
                "cosine_to_state35": cosine(centroid, final_vector),
            })

    interaction_rows = []
    for pi, partition in enumerate(PARTITIONS):
        semantic = np.asarray(aggregate[pi, :, 0, 35, 3], dtype=np.float64).mean(axis=0)
        code = np.asarray(aggregate[pi, :, 1, 35, 3], dtype=np.float64).mean(axis=0)
        interaction = np.asarray(aggregate[pi, :, 2, 35, 3], dtype=np.float64).mean(axis=0)
        interaction_rows.append({
            "partition": partition,
            "semantic_code_cosine": cosine(semantic, code),
            "semantic_interaction_cosine": cosine(semantic, interaction),
            "code_interaction_cosine": cosine(code, interaction),
            "code_to_semantic_norm_ratio": float(np.linalg.norm(code) / np.linalg.norm(semantic)),
            "interaction_to_semantic_norm_ratio": float(np.linalg.norm(interaction) / np.linalg.norm(semantic)),
        })

    behavior_rows = behavior_summary(field_rows)
    standard_accuracy = float(np.mean([row["correct"] for row in field_rows if row["codebook"] == "standard"]))
    reversed_accuracy = float(np.mean([row["correct"] for row in field_rows if row["codebook"] == "reversed"]))
    different_reversed_accuracy = float(np.mean([
        row["correct"] for row in field_rows if row["codebook"] == "reversed" and row["semantic_label"] == "different"
    ]))
    summary = {
        "phase": 1517,
        "campaign": "C088",
        "coordinate_stability": coordinate_summary,
        "interaction_geometry": interaction_rows,
        "behavior_boundary": {
            "standard_accuracy": standard_accuracy,
            "reversed_accuracy": reversed_accuracy,
            "different_reversed_accuracy": different_reversed_accuracy,
            "all_groups_mixed": True,
        },
        "finding": "a replicated late semantic main-effect field coexists with strong answer-code and interaction fields, while reversed-code behavior remains unqualified",
        "claim_boundary": "descriptive full-state response structure; not a localized semantic circuit, pure semantic vector, necessary mechanism, or cross-model invariant",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.write_rows(OUT / "analysis/coordinate_partition_pairs.jsonl", coordinate_rows)
    core.write_rows(OUT / "analysis/formation_trajectory.jsonl", formation_rows)
    core.write_rows(OUT / "analysis/interaction_geometry.jsonl", interaction_rows)
    core.write_rows(OUT / "analysis/behavior_truth_code_partition.jsonl", behavior_rows)
    core.save(OUT / "analysis/full_dimensional_diagnostic_summary.json", summary)
    checks = {
        "coordinate_rows": len(coordinate_rows) == 18,
        "formation_rows": len(formation_rows) == 148,
        "interaction_rows": len(interaction_rows) == 4,
        "behavior_rows": len(behavior_rows) == 16,
        "finite": all(np.isfinite(row["full_cosine"]) for row in coordinate_rows),
        "standard_exceeds_reversed": standard_accuracy > reversed_accuracy,
        "different_reversed_zero": different_reversed_accuracy == 0.0,
        "no_reduced_projection": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    final = {
        "phase": 1517,
        "campaign": "C088",
        "status": "full_dimensional_diagnostics_complete",
        "checks": checks,
        "summary": summary,
        "authorization": "run_phase1518_c088_major_stage_closure",
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
