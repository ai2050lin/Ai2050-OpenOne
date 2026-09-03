#!/usr/bin/env python3
"""Phase1555: behavior-stratified and raw-coordinate diagnostics for C095."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1553_c095_existing_field_batch_mining_contract"
PARENT = RESULT / "phase1554_c095_triadic_interaction_and_field_atlas"
FIELD_SOURCE = RESULT / "phase1539_c091_canonical_all_state_capture"
BEHAVIOR_SOURCE = RESULT / "phase1537_c091_behavior_only_qualification"
OUT = RESULT / "phase1555_c095_behavior_stratified_and_raw_coordinate_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PARTITIONS = ("response_discovery", "confirmation", "lockbox")
SURFACES = ("prequery", "postquery")
CONCRETENESS = ("concrete", "abstract")
FAMILIES = ("similarity", "class_inclusion", "whole_part")
FAMILY_PAIRS = (("similarity", "class_inclusion"), ("similarity", "whole_part"), ("class_inclusion", "whole_part"))
ROLES = ("source_word", "target_word", "relation_anchor", "boundary")
FOCUS_STATES = (31, 32)
SUPPORT_COUNTS = (16, 64, 256, 1024)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def top_indices(vector: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(np.abs(vector), kind="stable")[-k:]


def energy_fraction(vector: np.ndarray, indices: np.ndarray) -> float:
    denominator = float(np.dot(vector, vector))
    return float(np.dot(vector[indices], vector[indices]) / denominator) if denominator > 0 else 0.0


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1555 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    contract = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1555_c095_behavior_stratified_and_raw_coordinate_atlas" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1554 authorization missing")

    field = np.load(FIELD_SOURCE / "raw/canonical_all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(FIELD_SOURCE / "raw/canonical_all_role_field_index.jsonl")
    behavior = core.rows(BEHAVIOR_SOURCE / "raw/behavior_logits.jsonl")
    behavior_by_case = {row["case_id"]: row for row in behavior}
    if set(behavior_by_case) != {row["case_id"] for row in index}:
        raise RuntimeError("behavior/index case mismatch")

    raw_dir = OUT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    behavior_diff_path = raw_dir / "behavior_correct_minus_incorrect.float16.npy"
    behavior_diff = np.lib.format.open_memmap(behavior_diff_path, mode="w+", dtype=np.float16, shape=(18, 37, 4, 2560))
    behavior_index: list[dict] = []
    behavior_atlas: list[dict] = []
    behavior_vectors: dict[tuple[str, str, str], np.ndarray] = {}
    missing_cells = 0
    cursor = 0
    for partition in PARTITIONS:
        for surface in SURFACES:
            for query_family in FAMILIES:
                rows = [row for row in index if row["partition"] == partition and row["surface"] == surface and row["query_family"] == query_family]
                correct_rows = [int(row["row_index"]) for row in rows if behavior_by_case[row["case_id"]]["correct"]]
                incorrect_rows = [int(row["row_index"]) for row in rows if not behavior_by_case[row["case_id"]]["correct"]]
                missing = not correct_rows or not incorrect_rows
                if missing:
                    vector = np.full((37, 4, 2560), np.nan, dtype=np.float32)
                    missing_cells += 1
                else:
                    vector = np.asarray(field[correct_rows], dtype=np.float32).mean(axis=0) - np.asarray(field[incorrect_rows], dtype=np.float32).mean(axis=0)
                behavior_diff[cursor] = vector.astype(np.float16)
                behavior_vectors[(partition, surface, query_family)] = vector
                behavior_index.append({
                    "row_index": cursor,
                    "partition": partition,
                    "surface": surface,
                    "query_family": query_family,
                    "n_correct": len(correct_rows),
                    "n_incorrect": len(incorrect_rows),
                    "missingness": "M_CELL" if missing else None,
                })
                dynamic = vector - vector[0:1]
                for state in range(37):
                    for ri, role in enumerate(ROLES):
                        behavior_atlas.append({
                            "partition": partition,
                            "surface": surface,
                            "query_family": query_family,
                            "state": state,
                            "role": role,
                            "n_correct": len(correct_rows),
                            "n_incorrect": len(incorrect_rows),
                            "missingness": "M_CELL" if missing else None,
                            "raw_norm": None if missing else float(np.linalg.norm(vector[state, ri].astype(np.float64))),
                            "dynamic_state0_subtracted_norm": None if missing else float(np.linalg.norm(dynamic[state, ri].astype(np.float64))),
                        })
                cursor += 1
    behavior_diff.flush()
    behavior_index_path = raw_dir / "behavior_correct_minus_incorrect_index.jsonl"
    behavior_atlas_path = OUT / "analysis/behavior_stratified_atlas.jsonl"
    core.write_rows(behavior_index_path, behavior_index)
    core.write_rows(behavior_atlas_path, behavior_atlas)

    centroids = np.load(PARENT / "raw/triadic_interaction_centroids.float32.npy", mmap_mode="r")
    concentration: list[dict] = []
    coordinate_stability: list[dict] = []
    behavior_to_triadic: list[dict] = []
    for pi, partition in enumerate(PARTITIONS):
        for si, surface in enumerate(SURFACES):
            for ci, concreteness in enumerate(CONCRETENESS):
                for fi, (family_a, family_b) in enumerate(FAMILY_PAIRS):
                    for state in FOCUS_STATES:
                        vector = np.asarray(centroids[pi, si, ci, fi, state, 3], dtype=np.float64)
                        for k in SUPPORT_COUNTS:
                            indices = top_indices(vector, k)
                            concentration.append({
                                "partition": partition,
                                "surface": surface,
                                "concreteness": concreteness,
                                "family_a": family_a,
                                "family_b": family_b,
                                "state": state,
                                "role": "boundary",
                                "k": k,
                                "energy_fraction": energy_fraction(vector, indices),
                                "mean_abs_on_support": float(np.mean(np.abs(vector[indices]))),
                                "maximum_abs": float(np.max(np.abs(vector))),
                            })

    for si, surface in enumerate(SURFACES):
        for ci, concreteness in enumerate(CONCRETENESS):
            for fi, (family_a, family_b) in enumerate(FAMILY_PAIRS):
                for state in FOCUS_STATES:
                    reference = np.asarray(centroids[0, si, ci, fi, state, 3], dtype=np.float64)
                    for target_pi in (1, 2):
                        target = np.asarray(centroids[target_pi, si, ci, fi, state, 3], dtype=np.float64)
                        for k in SUPPORT_COUNTS:
                            reference_indices = top_indices(reference, k)
                            target_indices = top_indices(target, k)
                            reference_set = set(reference_indices.tolist())
                            target_set = set(target_indices.tolist())
                            complement = np.setdiff1d(np.arange(2560), reference_indices, assume_unique=False)
                            nonzero = (reference[reference_indices] != 0) & (target[reference_indices] != 0)
                            sign_agreement = float(np.mean(np.sign(reference[reference_indices][nonzero]) == np.sign(target[reference_indices][nonzero]))) if np.any(nonzero) else 0.0
                            coordinate_stability.append({
                                "reference_partition": PARTITIONS[0],
                                "target_partition": PARTITIONS[target_pi],
                                "surface": surface,
                                "concreteness": concreteness,
                                "family_a": family_a,
                                "family_b": family_b,
                                "state": state,
                                "role": "boundary",
                                "k": k,
                                "reference_energy_fraction": energy_fraction(reference, reference_indices),
                                "target_energy_on_reference_support": energy_fraction(target, reference_indices),
                                "sign_agreement": sign_agreement,
                                "support_jaccard": float(len(reference_set & target_set) / len(reference_set | target_set)),
                                "restricted_cosine": cosine(reference[reference_indices], target[reference_indices]),
                                "complement_cosine": cosine(reference[complement], target[complement]),
                                "full_cosine": cosine(reference, target),
                            })

    for pi, partition in enumerate(PARTITIONS):
        for si, surface in enumerate(SURFACES):
            for state in FOCUS_STATES:
                generic = np.asarray(centroids[pi, si, :, :, state, 3], dtype=np.float64).mean(axis=(0, 1))
                for query_family in FAMILIES:
                    vector = behavior_vectors[(partition, surface, query_family)]
                    missing = not np.all(np.isfinite(vector))
                    behavior_to_triadic.append({
                        "partition": partition,
                        "surface": surface,
                        "query_family": query_family,
                        "state": state,
                        "role": "boundary",
                        "missingness": "M_CELL" if missing else None,
                        "cosine_to_generic_triadic_centroid": None if missing else cosine(vector[state, 3].astype(np.float64), generic),
                    })

    concentration_path = OUT / "analysis/raw_coordinate_concentration_atlas.jsonl"
    stability_path = OUT / "analysis/cross_partition_raw_coordinate_stability.jsonl"
    behavior_to_triadic_path = OUT / "analysis/behavior_to_triadic_focus_cosines.jsonl"
    core.write_rows(concentration_path, concentration)
    core.write_rows(stability_path, coordinate_stability)
    core.write_rows(behavior_to_triadic_path, behavior_to_triadic)

    decomposition = core.rows(PARENT / "analysis/three_by_three_decomposition_atlas.jsonl")
    focus_decomposition = []
    for row in decomposition:
        if row["role"] == "boundary" and row["state"] in FOCUS_STATES:
            pair = row["pair_main_mean_energy"]
            query = row["query_main_mean_energy"]
            interaction = row["interaction_mean_energy"]
            focus_decomposition.append({
                **row,
                "interaction_to_pair_main_energy_ratio": float(interaction / pair) if pair > 0 else None,
                "interaction_to_query_main_energy_ratio": float(interaction / query) if query > 0 else None,
            })
    focus_decomposition_path = OUT / "analysis/focus_decomposition_energy.jsonl"
    core.write_rows(focus_decomposition_path, focus_decomposition)

    k64 = [row for row in concentration if row["k"] == 64]
    stability64 = [row for row in coordinate_stability if row["k"] == 64]
    valid_behavior_cosines = [row["cosine_to_generic_triadic_centroid"] for row in behavior_to_triadic if row["cosine_to_generic_triadic_centroid"] is not None]
    summary = {
        "phase": 1555,
        "campaign": "C095",
        "status": "behavior_stratified_and_raw_coordinate_atlas_complete",
        "model_run": False,
        "behavior_stratification": {
            "groups": len(behavior_index),
            "missing_cells": missing_cells,
            "minimum_correct_count": min(row["n_correct"] for row in behavior_index),
            "minimum_incorrect_count": min(row["n_incorrect"] for row in behavior_index),
            "scope": "diagnostic only; correctness is confounded with item difficulty, lexical identity, and output identity",
        },
        "coordinate_concentration_k64": {
            "minimum_energy_fraction": float(min(row["energy_fraction"] for row in k64)),
            "median_energy_fraction": float(np.median([row["energy_fraction"] for row in k64])),
            "maximum_energy_fraction": float(max(row["energy_fraction"] for row in k64)),
        },
        "cross_partition_k64": {
            "minimum_sign_agreement": float(min(row["sign_agreement"] for row in stability64)),
            "median_sign_agreement": float(np.median([row["sign_agreement"] for row in stability64])),
            "minimum_support_jaccard": float(min(row["support_jaccard"] for row in stability64)),
            "median_support_jaccard": float(np.median([row["support_jaccard"] for row in stability64])),
            "minimum_restricted_cosine": float(min(row["restricted_cosine"] for row in stability64)),
            "median_restricted_cosine": float(np.median([row["restricted_cosine"] for row in stability64])),
            "minimum_complement_cosine": float(min(row["complement_cosine"] for row in stability64)),
            "median_complement_cosine": float(np.median([row["complement_cosine"] for row in stability64])),
        },
        "behavior_to_generic_triadic": {
            "minimum_cosine": float(min(valid_behavior_cosines)),
            "median_cosine": float(np.median(valid_behavior_cosines)),
            "maximum_cosine": float(max(valid_behavior_cosines)),
            "n": len(valid_behavior_cosines),
        },
        "coverage": {
            "behavior_groups": len(behavior_index),
            "behavior_atlas_rows": len(behavior_atlas),
            "coordinate_concentration_rows": len(concentration),
            "coordinate_stability_rows": len(coordinate_stability),
            "behavior_to_triadic_rows": len(behavior_to_triadic),
            "focus_decomposition_rows": len(focus_decomposition),
        },
        "files": {
            "behavior_diff": {"path": str(behavior_diff_path.relative_to(ROOT)), "sha256": core.sha(behavior_diff_path), "shape": [18, 37, 4, 2560]},
            "behavior_index": {"path": str(behavior_index_path.relative_to(ROOT)), "sha256": core.sha(behavior_index_path), "rows": len(behavior_index)},
            "behavior_atlas": {"path": str(behavior_atlas_path.relative_to(ROOT)), "sha256": core.sha(behavior_atlas_path), "rows": len(behavior_atlas)},
            "concentration": {"path": str(concentration_path.relative_to(ROOT)), "sha256": core.sha(concentration_path), "rows": len(concentration)},
            "stability": {"path": str(stability_path.relative_to(ROOT)), "sha256": core.sha(stability_path), "rows": len(coordinate_stability)},
            "behavior_to_triadic": {"path": str(behavior_to_triadic_path.relative_to(ROOT)), "sha256": core.sha(behavior_to_triadic_path), "rows": len(behavior_to_triadic)},
            "focus_decomposition": {"path": str(focus_decomposition_path.relative_to(ROOT)), "sha256": core.sha(focus_decomposition_path), "rows": len(focus_decomposition)},
        },
        "claim_boundary": contract["claim_boundary"],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "run_phase1556_c095_joint_synthesis_and_major_stage_closure",
    }
    core.save(OUT / "analysis/behavior_and_coordinate_summary.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1555, "campaign": "C095", "status": summary["status"], "authorization": summary["authorization"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
