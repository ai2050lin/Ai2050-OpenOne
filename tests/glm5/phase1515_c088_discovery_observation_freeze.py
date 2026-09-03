#!/usr/bin/env python3
"""Phase1515: observe C088 discovery only and freeze factorial-field predictions."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
ATLAS = RESULT / "phase1514_c088_factorial_field_atlas"
C087_ATLAS = RESULT / "phase1507_c087_descriptive_semantic_contrast_atlas"
C087_CAPTURE = RESULT / "phase1506_c087_all_case_field_capture"
OUT = RESULT / "phase1515_c088_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

EFFECTS = ("semantic", "code", "semantic_code")
ROLES = ("source_relation", "candidate_relation", "code_rule", "boundary")


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def pairwise_mean(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-12
    unit = vectors[valid] / norms[valid, None]
    n = len(unit)
    if n < 2:
        return 0.0
    total = float(np.dot(unit.sum(axis=0), unit.sum(axis=0)) - n)
    return total / (n * (n - 1))


def coherence(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    denominator = float(np.mean(np.sum(vectors * vectors, axis=1)))
    mean = vectors.mean(axis=0)
    return float(np.dot(mean, mean) / denominator) if denominator > 1e-12 else 0.0


def concentration(vectors, k=26):
    energy = np.sum(np.square(np.asarray(vectors, dtype=np.float64)), axis=0)
    total = float(energy.sum())
    return float(np.sort(energy)[::-1][:k].sum() / total) if total > 1e-12 else 0.0


def effect_metrics(panel):
    per_group = panel.mean(axis=1)
    surface_centroids = panel.mean(axis=0)
    return {
        "surface_centroid_cosine": cosine(surface_centroids[0], surface_centroids[1]),
        "within_group_surface_cosine_mean": float(np.mean([cosine(row[0], row[1]) for row in panel])),
        "group_pairwise_cosine_mean": pairwise_mean(per_group),
        "shared_energy_fraction": coherence(per_group),
        "top1pct_coordinate_energy": concentration(per_group),
        "centroid_norm": float(np.linalg.norm(per_group.mean(axis=0))),
        "mean_group_norm": float(np.mean(np.linalg.norm(per_group, axis=1))),
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1515 exists")
    parent = core.load(ATLAS / "analysis/final.json")
    parent_audit = core.load(ATLAS / "audit/independent_final_audit.json")
    atlas_summary = core.load(ATLAS / "analysis/factorial_field_atlas_summary.json")
    if parent["authorization"] != "run_phase1515_c088_discovery_observation_freeze" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1514 authorization missing")
    if core.sha(ATLAS / "atlas/group_factorial_effect.float16.npy") != atlas_summary["files"]["group"]["sha256"]:
        raise RuntimeError("C088 atlas hash mismatch")

    group = np.load(ATLAS / "atlas/group_factorial_effect.float16.npy", mmap_mode="r")
    group_index = core.rows(ATLAS / "atlas/group_factorial_effect_index.jsonl")
    selected = [row["group_index"] for row in group_index if row["partition"] == "response_discovery"]
    if selected != list(range(72)):
        raise RuntimeError("discovery partition is not the frozen first 72 groups")

    c087 = np.load(C087_ATLAS / "atlas/group_semantic_contrast.float32.npy", mmap_mode="r")
    c087_index = core.rows(C087_ATLAS / "atlas/group_semantic_contrast_index.jsonl")
    c087_selected = [row["group_index"] for row in c087_index if row["partition"] == "response_discovery"]
    c087_capture = core.load(C087_CAPTURE / "analysis/capture_metadata.json")
    if len(c087_selected) != 72 or c087_capture["shape"] != [864, 37, 3, 2560]:
        raise RuntimeError("C087 discovery reference invalid")

    observations = []
    for state in range(37):
        for role_index, role in enumerate(ROLES):
            effects = {}
            panels = {}
            for effect_index, effect in enumerate(EFFECTS):
                panel = np.asarray(group[selected, :, effect_index, state, role_index], dtype=np.float64)
                panels[effect] = panel
                effects[effect] = effect_metrics(panel)
            semantic = panels["semantic"]
            interaction = panels["semantic_code"]
            standard = semantic + interaction
            reversed_code = semantic - interaction
            semantic_energy = float(np.sum(semantic * semantic))
            interaction_energy = float(np.sum(interaction * interaction))
            total = semantic_energy + interaction_energy
            conditional = {
                "semantic_energy_fraction": semantic_energy / total if total > 1e-12 else 0.0,
                "conditional_centroid_cosine": cosine(standard.mean(axis=(0, 1)), reversed_code.mean(axis=(0, 1))),
                "conditional_group_cosine_mean": float(np.mean([
                    cosine(standard[i].mean(axis=0), reversed_code[i].mean(axis=0))
                    for i in range(len(selected))
                ])),
            }
            natural_alignment = None
            if role != "code_rule":
                c087_role = ("source_relation", "candidate_relation", "boundary").index(role)
                natural = np.asarray(c087[c087_selected, :, state, c087_role], dtype=np.float64).mean(axis=1)
                factorial = semantic.mean(axis=1)
                natural_alignment = {
                    "centroid_cosine": cosine(factorial.mean(axis=0), natural.mean(axis=0)),
                    "group_cosine_mean": float(np.mean([cosine(factorial[i], natural[i]) for i in range(72)])),
                }
            observations.append({
                "state": state,
                "role": role,
                "count": len(selected),
                "effects": effects,
                "conditional_semantic": conditional,
                "c087_natural_alignment": natural_alignment,
            })

    target = next(row for row in observations if row["state"] == 35 and row["role"] == "boundary")
    semantic_target = target["effects"]["semantic"]
    structural_presence = (
        semantic_target["surface_centroid_cosine"] >= 0.70
        and semantic_target["within_group_surface_cosine_mean"] >= 0.40
        and semantic_target["group_pairwise_cosine_mean"] >= 0.35
        and semantic_target["shared_energy_fraction"] >= 0.35
    )
    onset_candidates = [
        row["state"] for row in observations
        if row["role"] == "boundary"
        and row["effects"]["semantic"]["within_group_surface_cosine_mean"] >= 0.40
        and row["effects"]["semantic"]["shared_energy_fraction"] >= 0.35
    ]
    onset = min(onset_candidates) if onset_candidates else None

    logit_rows = core.rows(ATLAS / "analysis/group_logit_factorial_effects.jsonl")
    discovery_logits = [row for row in logit_rows if row["partition"] == "response_discovery"]
    logit_summary = {
        effect: {
            "mean": float(np.mean([row[effect] for row in discovery_logits])),
            "mean_abs": float(np.mean([abs(row[effect]) for row in discovery_logits])),
        }
        for effect in EFFECTS
    }
    discovery = {
        "partition": "response_discovery",
        "groups": 72,
        "target_state": 35,
        "target_role": "boundary",
        "structural_presence": structural_presence,
        "onset_state": onset,
        "target": target,
        "logit_factorial_effects": logit_summary,
        "interpretation": "semantic structure presence is separated from exact effect-size equality and from behavioral code compliance",
    }
    core.write_rows(OUT / "analysis/discovery_full_layer_role_observations.jsonl", observations)
    core.save(OUT / "analysis/discovery_summary.json", discovery)

    freeze = {
        "phase": 1515,
        "campaign": "C088",
        "source_partition": "response_discovery",
        "untouched_partitions": ["confirmation", "lockbox", "fresh_external"],
        "target_state": 35,
        "target_role": "boundary",
        "reference": discovery,
        "structure_presence_gates": {
            "surface_centroid_cosine_min": 0.70,
            "within_group_surface_cosine_mean_min": 0.40,
            "group_pairwise_cosine_mean_min": 0.35,
            "shared_energy_fraction_min": 0.35,
        },
        "paired_effect_size_tolerances": {
            "surface_centroid_cosine": 0.15,
            "within_group_surface_cosine_mean": 0.15,
            "group_pairwise_cosine_mean": 0.15,
            "shared_energy_fraction": 0.15,
            "semantic_energy_fraction": 0.15,
            "conditional_centroid_cosine": 0.20,
            "onset_state": 5,
        },
        "fresh_external_scope": "directional structure-presence test only; no exact effect-size equality gate",
        "predictions": {
            "P088-1": "semantic main-effect structure presence repeats in confirmation and lockbox",
            "P088-2": "paired holdout effect sizes remain within frozen tolerances",
            "P088-3": "fresh roots satisfy the same structural presence gate without an equality claim",
            "P088-4": "C087 natural-query and C088 semantic main-effect centroids remain directionally aligned",
            "P088-D1": "code and semantic-code effects are diagnostics and cannot by themselves establish semantic execution",
        },
        "claim_boundary": "Qwen3 cross-root full-state factorial response field; no universal semantic vector, natural code following, or component-level mechanism claim",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    freeze["freeze_sha256"] = core.digest(freeze)
    core.save(OUT / "protocol/frozen_factorial_predictions.json", freeze)
    checks = {
        "discovery_only": len(selected) == 72 and selected == list(range(72)),
        "observation_rows": len(observations) == 37 * 4,
        "finite": all(
            np.isfinite(value)
            for row in observations
            for effect in row["effects"].values()
            for value in effect.values()
        ),
        "source_causal_zero": all(row["effects"]["semantic"]["centroid_norm"] == 0.0 for row in observations if row["role"] == "source_relation"),
        "onset_defined": onset is not None,
        "freeze_hash": freeze["freeze_sha256"] == core.digest({key: value for key, value in freeze.items() if key != "freeze_sha256"}),
        "separate_gate_types": "structure_presence_gates" in freeze and "paired_effect_size_tolerances" in freeze,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    result = {
        "phase": 1515,
        "campaign": "C088",
        "status": "discovery_observed_and_factorial_predictions_frozen",
        "discovery": discovery,
        "checks": checks,
        "freeze_sha256": freeze["freeze_sha256"],
        "authorization": "run_phase1516_c088_holdout_and_fresh_reveal",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
