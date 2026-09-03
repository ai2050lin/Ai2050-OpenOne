#!/usr/bin/env python3
"""Independent audit for Phase1555."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1554_c095_triadic_interaction_and_field_atlas"
FIELD_SOURCE = RESULT / "phase1539_c091_canonical_all_state_capture"
BEHAVIOR_SOURCE = RESULT / "phase1537_c091_behavior_only_qualification"
OUT = RESULT / "phase1555_c095_behavior_stratified_and_raw_coordinate_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def main() -> None:
    py_compile.compile(str(TESTS / "phase1555_c095_behavior_stratified_and_raw_coordinate_atlas.py"), doraise=True)
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    summary = core.load(OUT / "analysis/behavior_and_coordinate_summary.json")
    final = core.load(OUT / "analysis/final.json")
    behavior_diff = np.load(OUT / "raw/behavior_correct_minus_incorrect.float16.npy", mmap_mode="r")
    behavior_index = core.rows(OUT / "raw/behavior_correct_minus_incorrect_index.jsonl")
    behavior_atlas = core.rows(OUT / "analysis/behavior_stratified_atlas.jsonl")
    concentration = core.rows(OUT / "analysis/raw_coordinate_concentration_atlas.jsonl")
    stability = core.rows(OUT / "analysis/cross_partition_raw_coordinate_stability.jsonl")
    behavior_to_triadic = core.rows(OUT / "analysis/behavior_to_triadic_focus_cosines.jsonl")
    focus_decomposition = core.rows(OUT / "analysis/focus_decomposition_energy.jsonl")

    field = np.load(FIELD_SOURCE / "raw/canonical_all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(FIELD_SOURCE / "raw/canonical_all_role_field_index.jsonl")
    behavior = {row["case_id"]: row for row in core.rows(BEHAVIOR_SOURCE / "raw/behavior_logits.jsonl")}
    first = behavior_index[0]
    rows = [row for row in index if row["partition"] == first["partition"] and row["surface"] == first["surface"] and row["query_family"] == first["query_family"]]
    correct_rows = [int(row["row_index"]) for row in rows if behavior[row["case_id"]]["correct"]]
    incorrect_rows = [int(row["row_index"]) for row in rows if not behavior[row["case_id"]]["correct"]]
    expected = np.asarray(field[correct_rows], dtype=np.float32).mean(axis=0) - np.asarray(field[incorrect_rows], dtype=np.float32).mean(axis=0)
    behavior_exact = np.array_equal(expected.astype(np.float16), np.asarray(behavior_diff[0]))

    centroids = np.load(PARENT / "raw/triadic_interaction_centroids.float32.npy", mmap_mode="r")
    first_coordinate = concentration[0]
    family_pairs = (("similarity", "class_inclusion"), ("similarity", "whole_part"), ("class_inclusion", "whole_part"))
    fi = family_pairs.index((first_coordinate["family_a"], first_coordinate["family_b"]))
    vector = np.asarray(centroids[0, 0, 0, fi, first_coordinate["state"], 3], dtype=np.float64)
    k = first_coordinate["k"]
    indices = np.argsort(np.abs(vector), kind="stable")[-k:]
    expected_energy = float(np.dot(vector[indices], vector[indices]) / np.dot(vector, vector))
    concentration_recomputed = abs(expected_energy - first_coordinate["energy_fraction"]) <= 1e-12

    first_stability = stability[0]
    reference = np.asarray(centroids[0, 0, 0, 0, first_stability["state"], 3], dtype=np.float64)
    target = np.asarray(centroids[1, 0, 0, 0, first_stability["state"], 3], dtype=np.float64)
    reference_indices = np.argsort(np.abs(reference), kind="stable")[-first_stability["k"]:]
    restricted_recomputed = abs(cosine(reference[reference_indices], target[reference_indices]) - first_stability["restricted_cosine"]) <= 1e-12

    file_hashes = all(core.sha(ROOT / item["path"]) == item["sha256"] for item in summary["files"].values())
    checks = {
        "parent_audited": parent_audit["all_checks_passed"],
        "behavior_shape": list(behavior_diff.shape) == [18, 37, 4, 2560] and len(behavior_index) == 18,
        "coverage": len(behavior_atlas) == 2664 and len(concentration) == 288 and len(stability) == 192 and len(behavior_to_triadic) == 36 and len(focus_decomposition) == 48,
        "behavior_source_recomputed": behavior_exact,
        "concentration_recomputed": concentration_recomputed,
        "restricted_cosine_recomputed": restricted_recomputed,
        "missingness_consistent": summary["behavior_stratification"]["missing_cells"] == sum(row["missingness"] == "M_CELL" for row in behavior_index),
        "finite_nonmissing": all(row["raw_norm"] is None or np.isfinite(row["raw_norm"]) for row in behavior_atlas) and all(np.isfinite(row["energy_fraction"]) for row in concentration),
        "full_coordinate_only": {row["k"] for row in concentration} == {16, 64, 256, 1024},
        "file_hashes": file_hashes,
        "scope": summary["claim_boundary"]["allowed"].startswith("retrospective descriptive"),
        "authorization": final["authorization"] == "run_phase1556_c095_joint_synthesis_and_major_stage_closure",
    }
    result = {"phase": 1555, "campaign": "C095", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
