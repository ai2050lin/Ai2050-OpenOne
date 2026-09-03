#!/usr/bin/env python3
"""Phase1516: reveal paired holdouts and fresh-root external panel for C088."""
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
FREEZE = RESULT / "phase1515_c088_discovery_observation_freeze"
C087_ATLAS = RESULT / "phase1507_c087_descriptive_semantic_contrast_atlas"
OUT = RESULT / "phase1516_c088_holdout_and_fresh_reveal"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1515_c088_discovery_observation_freeze import cosine, effect_metrics

PARTITIONS = ("confirmation", "lockbox", "fresh_external")


def partition_metrics(group, group_index, partition, c087, c087_index):
    selected = [row["group_index"] for row in group_index if row["partition"] == partition]
    target_panel = np.asarray(group[selected, :, :, 35, 3], dtype=np.float64)
    effects = {
        effect: effect_metrics(target_panel[:, :, effect_index])
        for effect_index, effect in enumerate(("semantic", "code", "semantic_code"))
    }
    semantic = target_panel[:, :, 0]
    interaction = target_panel[:, :, 2]
    standard, reversed_code = semantic + interaction, semantic - interaction
    semantic_energy = float(np.sum(semantic * semantic))
    interaction_energy = float(np.sum(interaction * interaction))
    conditional = {
        "semantic_energy_fraction": semantic_energy / (semantic_energy + interaction_energy),
        "conditional_centroid_cosine": cosine(standard.mean(axis=(0, 1)), reversed_code.mean(axis=(0, 1))),
        "conditional_group_cosine_mean": float(np.mean([
            cosine(standard[i].mean(axis=0), reversed_code[i].mean(axis=0)) for i in range(len(selected))
        ])),
    }
    onset_candidates = []
    for state in range(37):
        panel = np.asarray(group[selected, :, 0, state, 3], dtype=np.float64)
        metrics = effect_metrics(panel)
        if metrics["within_group_surface_cosine_mean"] >= 0.40 and metrics["shared_energy_fraction"] >= 0.35:
            onset_candidates.append(state)
    natural_alignment = None
    if partition != "fresh_external":
        c087_selected = [row["group_index"] for row in c087_index if row["partition"] == partition]
        natural = np.asarray(c087[c087_selected, :, 35, 2], dtype=np.float64).mean(axis=1)
        factorial = semantic.mean(axis=1)
        natural_alignment = {
            "centroid_cosine": cosine(factorial.mean(axis=0), natural.mean(axis=0)),
            "group_cosine_mean": float(np.mean([cosine(factorial[i], natural[i]) for i in range(len(selected))])),
        }
    return {
        "partition": partition,
        "groups": len(selected),
        "target_state": 35,
        "target_role": "boundary",
        "effects": effects,
        "conditional_semantic": conditional,
        "onset_state": min(onset_candidates) if onset_candidates else None,
        "c087_natural_alignment": natural_alignment,
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1516 exists")
    parent = core.load(FREEZE / "analysis/final.json")
    parent_audit = core.load(FREEZE / "audit/independent_final_audit.json")
    freeze = core.load(FREEZE / "protocol/frozen_factorial_predictions.json")
    if parent["authorization"] != "run_phase1516_c088_holdout_and_fresh_reveal" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1515 authorization missing")
    if freeze["freeze_sha256"] != parent["freeze_sha256"]:
        raise RuntimeError("freeze hash mismatch")

    group = np.load(ATLAS / "atlas/group_factorial_effect.float16.npy", mmap_mode="r")
    group_index = core.rows(ATLAS / "atlas/group_factorial_effect_index.jsonl")
    c087 = np.load(C087_ATLAS / "atlas/group_semantic_contrast.float32.npy", mmap_mode="r")
    c087_index = core.rows(C087_ATLAS / "atlas/group_semantic_contrast_index.jsonl")
    results = [partition_metrics(group, group_index, partition, c087, c087_index) for partition in PARTITIONS]

    reference = freeze["reference"]
    ref_semantic = reference["target"]["effects"]["semantic"]
    ref_conditional = reference["target"]["conditional_semantic"]
    presence_gates = freeze["structure_presence_gates"]
    tolerances = freeze["paired_effect_size_tolerances"]
    metric_map = {
        "surface_centroid_cosine": "surface_centroid_cosine",
        "within_group_surface_cosine_mean": "within_group_surface_cosine_mean",
        "group_pairwise_cosine_mean": "group_pairwise_cosine_mean",
        "shared_energy_fraction": "shared_energy_fraction",
    }
    verdict_rows = []
    for result in results:
        semantic = result["effects"]["semantic"]
        presence_checks = {
            "surface_centroid_cosine": semantic["surface_centroid_cosine"] >= presence_gates["surface_centroid_cosine_min"],
            "within_group_surface_cosine_mean": semantic["within_group_surface_cosine_mean"] >= presence_gates["within_group_surface_cosine_mean_min"],
            "group_pairwise_cosine_mean": semantic["group_pairwise_cosine_mean"] >= presence_gates["group_pairwise_cosine_mean_min"],
            "shared_energy_fraction": semantic["shared_energy_fraction"] >= presence_gates["shared_energy_fraction_min"],
        }
        equality_checks = None
        if result["partition"] != "fresh_external":
            equality_checks = {
                key: abs(semantic[source] - ref_semantic[source]) <= tolerances[key]
                for key, source in metric_map.items()
            }
            equality_checks.update({
                "semantic_energy_fraction": abs(result["conditional_semantic"]["semantic_energy_fraction"] - ref_conditional["semantic_energy_fraction"]) <= tolerances["semantic_energy_fraction"],
                "conditional_centroid_cosine": abs(result["conditional_semantic"]["conditional_centroid_cosine"] - ref_conditional["conditional_centroid_cosine"]) <= tolerances["conditional_centroid_cosine"],
                "onset_state": result["onset_state"] is not None and abs(result["onset_state"] - reference["onset_state"]) <= tolerances["onset_state"],
            })
        verdict_rows.append({
            **result,
            "structure_presence_checks": presence_checks,
            "structure_presence_passed": all(presence_checks.values()),
            "effect_size_equality_checks": equality_checks,
            "effect_size_equality_passed": None if equality_checks is None else all(equality_checks.values()),
            "natural_alignment_directional": None if result["c087_natural_alignment"] is None else result["c087_natural_alignment"]["centroid_cosine"] > 0.0,
        })

    paired = [row for row in verdict_rows if row["partition"] != "fresh_external"]
    fresh = next(row for row in verdict_rows if row["partition"] == "fresh_external")
    verdict = {
        "structure_presence_paired_holdouts": all(row["structure_presence_passed"] for row in paired),
        "paired_effect_size_equality": all(row["effect_size_equality_passed"] for row in paired),
        "fresh_root_directional_presence": fresh["structure_presence_passed"],
        "c087_natural_alignment_directional": all(row["natural_alignment_directional"] for row in paired),
    }
    verdict["all_pre_registered_predictions_passed"] = all(verdict.values())
    core.write_rows(OUT / "analysis/partition_reveal_metrics.jsonl", verdict_rows)
    summary = {
        "phase": 1516,
        "campaign": "C088",
        "freeze_sha256": freeze["freeze_sha256"],
        "verdict": verdict,
        "partitions": verdict_rows,
        "evidence_rule": "structure presence, paired effect-size equality, fresh-root direction, and C087 alignment are separate claims",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/holdout_and_fresh_reveal_summary.json", summary)
    checks = {
        "counts": [row["groups"] for row in results] == [72, 72, 32],
        "finite": all(
            np.isfinite(value)
            for row in results
            for effect in row["effects"].values()
            for value in effect.values()
        ),
        "fresh_no_equality_gate": fresh["effect_size_equality_passed"] is None,
        "separate_verdicts": len(verdict) == 5,
        "freeze_unchanged": freeze["freeze_sha256"] == core.digest({key: value for key, value in freeze.items() if key != "freeze_sha256"}),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    final = {
        "phase": 1516,
        "campaign": "C088",
        "status": "holdout_and_fresh_reveal_complete",
        "verdict": verdict,
        "checks": checks,
        "authorization": "run_phase1517_c088_full_dimensional_diagnostics",
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
