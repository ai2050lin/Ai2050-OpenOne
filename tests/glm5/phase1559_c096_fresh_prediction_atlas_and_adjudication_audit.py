#!/usr/bin/env python3
"""Independent source and prediction audit for Phase1559."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1557_c096_fresh_human_relation_field_contract"
FIELD_SOURCE = RESULT / "phase1558_c096_unified_behavior_and_all_state_capture"
OUT = RESULT / "phase1559_c096_fresh_prediction_atlas_and_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    py_compile.compile(str(TESTS / "phase1559_c096_fresh_prediction_atlas_and_adjudication.py"), doraise=True)
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/c096_prediction_adjudication.json")
    final = core.load(OUT / "analysis/final.json")
    individual = np.load(OUT / "raw/c096_triadic_individual_interactions.float16.npy", mmap_mode="r")
    individual_index = core.rows(OUT / "raw/c096_triadic_individual_interactions_index.jsonl")
    centroids = np.load(OUT / "raw/c096_triadic_interaction_centroids.float32.npy", mmap_mode="r")
    triadic = core.rows(OUT / "analysis/c096_triadic_family_pair_cosines.jsonl")
    cross_partition = core.rows(OUT / "analysis/c096_cross_partition_cosines.jsonl")
    concentration = core.rows(OUT / "analysis/c096_raw_coordinate_concentration.jsonl")
    stability = core.rows(OUT / "analysis/c096_raw_coordinate_stability.jsonl")
    cross_material = core.rows(OUT / "analysis/c091_to_c096_cross_material_coordinate_cosines.jsonl")

    field = np.load(FIELD_SOURCE / "raw/c096_all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(FIELD_SOURCE / "raw/c096_all_role_field_index.jsonl")
    lookup = {(row["pair_id"], row["surface"], row["query_family"]): int(row["row_index"]) for row in index}
    meta = individual_index[0]
    expected = 0.5 * (
        field[lookup[(meta["pair_a"], meta["surface"], meta["family_a"])]].astype(np.float32)
        + field[lookup[(meta["pair_b"], meta["surface"], meta["family_b"])]].astype(np.float32)
        - field[lookup[(meta["pair_a"], meta["surface"], meta["family_b"])]].astype(np.float32)
        - field[lookup[(meta["pair_b"], meta["surface"], meta["family_a"])]].astype(np.float32)
    )
    source_formula = np.array_equal(expected.astype(np.float16), np.asarray(individual[0]))
    centroid_formula = np.array_equal(np.asarray(individual[0:5], dtype=np.float32).mean(axis=0), np.asarray(centroids[0, 0, 0, 0]))

    pre = [row for row in triadic if row["surface"] == "prequery" and row["state"] in {31, 32} and row["role"] == "boundary"]
    post = [row for row in triadic if row["surface"] == "postquery" and row["state"] in {31, 32} and row["role"] == "boundary"]
    cross = [row for row in cross_partition if row["state"] in {31, 32} and row["role"] == "boundary"]
    k64 = [row for row in concentration if row["k"] == 64]
    stable64 = [row for row in stability if row["k"] == 64]
    independently_passed = {
        "P096_1_prequery_common_field": min(row["minimum_pairwise_cosine"] for row in pre) >= 0.75,
        "P096_2_cross_partition": min(row["minimum_partition_cosine"] for row in cross) >= 0.75,
        "P096_3_top64_energy": 0.15 <= float(np.median([row["energy_fraction"] for row in k64])) <= 0.25,
        "P096_4_coordinate_stability": min(row["sign_agreement"] for row in stable64) >= 0.90 and min(row["restricted_cosine"] for row in stable64) >= 0.85,
        "P096_5_order_conditioning": float(np.median([row["minimum_pairwise_cosine"] for row in pre])) - float(np.median([row["minimum_pairwise_cosine"] for row in post])) >= 0.05,
    }
    file_hashes = all(core.sha(ROOT / item["path"]) == item["sha256"] for item in report["files"].values())
    checks = {
        "capture_audited": core.load(FIELD_SOURCE / "audit/independent_final_audit.json")["all_checks_passed"],
        "source_formula": source_formula,
        "centroid_formula": centroid_formula,
        "shapes": list(individual.shape) == [180, 37, 4, 2560] and list(centroids.shape) == [3, 2, 2, 3, 37, 4, 2560],
        "coverage": report["coverage"] == {"individual_interactions": 180, "interaction_rows": 5328, "triadic_rows": 1776, "cross_partition_rows": 1776, "concentration_rows": 288, "stability_rows": 192, "cross_material_rows": 72},
        "prediction_names": set(report["predictions"]) == set(protocol["frozen_predictions"]),
        "prediction_recompute": all(report["predictions"][key]["passed"] == value for key, value in independently_passed.items()),
        "individual_adjudication": report["passed_predictions"] == sum(independently_passed.values()),
        "file_hashes": file_hashes,
        "finite": all(np.isfinite(row["minimum_pairwise_cosine"]) for row in triadic) and all(np.isfinite(row["full_cosine"]) for row in cross_material),
        "scope": report["evidence_typing"]["similarity"] == "M_BEHAVIOR diagnostic" and report["evidence_typing"]["causal"] == "M_CAUSAL",
        "authorization": final["authorization"] == "run_phase1560_c096_major_stage_closure",
    }
    result = {"phase": 1559, "campaign": "C096", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
