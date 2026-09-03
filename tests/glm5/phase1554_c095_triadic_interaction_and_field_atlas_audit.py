#!/usr/bin/env python3
"""Independent source-to-atlas audit for Phase1554."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1553_c095_existing_field_batch_mining_contract"
FIELD_SOURCE = RESULT / "phase1539_c091_canonical_all_state_capture"
OUT = RESULT / "phase1554_c095_triadic_interaction_and_field_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    py_compile.compile(str(TESTS / "phase1554_c095_triadic_interaction_and_field_atlas.py"), doraise=True)
    contract = core.load(CONTRACT / "protocol/preregistration.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    summary = core.load(OUT / "analysis/triadic_and_field_summary.json")
    final = core.load(OUT / "analysis/final.json")
    individual = np.load(OUT / "raw/triadic_individual_interactions.float16.npy", mmap_mode="r")
    individual_index = core.rows(OUT / "raw/triadic_individual_interactions_index.jsonl")
    centroids = np.load(OUT / "raw/triadic_interaction_centroids.float32.npy", mmap_mode="r")
    components = np.load(OUT / "raw/three_by_three_raw_components.float16.npy", mmap_mode="r")
    interaction_atlas = core.rows(OUT / "analysis/triadic_interaction_atlas.jsonl")
    triadic = core.rows(OUT / "analysis/triadic_family_pair_cosines.jsonl")
    cross_partition = core.rows(OUT / "analysis/cross_partition_interaction_cosines.jsonl")
    decomposition = core.rows(OUT / "analysis/three_by_three_decomposition_atlas.jsonl")

    field = np.load(FIELD_SOURCE / "raw/canonical_all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(FIELD_SOURCE / "raw/canonical_all_role_field_index.jsonl")
    lookup = {(row["pair_id"], row["surface"], row["query_family"]): int(row["row_index"]) for row in index}
    sample_meta = individual_index[0]
    pair_a = sample_meta["pair_a"]
    pair_b = sample_meta["pair_b"]
    surface = sample_meta["surface"]
    family_a = sample_meta["family_a"]
    family_b = sample_meta["family_b"]
    expected = 0.5 * (
        field[lookup[(pair_a, surface, family_a)]].astype(np.float32)
        + field[lookup[(pair_b, surface, family_b)]].astype(np.float32)
        - field[lookup[(pair_a, surface, family_b)]].astype(np.float32)
        - field[lookup[(pair_b, surface, family_a)]].astype(np.float32)
    )
    source_formula_exact = np.array_equal(expected.astype(np.float16), np.asarray(individual[0]))
    first_group_mean = np.asarray(individual[0:5], dtype=np.float32).mean(axis=0)
    centroid_exact = np.array_equal(first_group_mean, np.asarray(centroids[0, 0, 0, 0]))

    file_hashes = all(core.sha(ROOT / item["path"]) == item["sha256"] for item in summary["files"].values() if "path" in item)
    checks = {
        "parent_audited": parent_audit["all_checks_passed"],
        "source_hashes": core.sha(ROOT / contract["source_assets"]["field"]["path"]) == contract["source_assets"]["field"]["sha256"] and core.sha(ROOT / contract["source_assets"]["index"]["path"]) == contract["source_assets"]["index"]["sha256"],
        "individual_shape": list(individual.shape) == [180, 37, 4, 2560] and len(individual_index) == 180,
        "centroid_shape": list(centroids.shape) == [3, 2, 2, 3, 37, 4, 2560],
        "component_shape": list(components.shape) == [12, 16, 37, 4, 2560],
        "coverage": len(interaction_atlas) == 5328 and len(triadic) == 1776 and len(cross_partition) == 1776 and len(decomposition) == 3552,
        "source_formula_exact_float16": source_formula_exact,
        "centroid_recomputed": centroid_exact,
        "reconstruction": summary["max_reconstruction_error"] <= 1e-4 and max(row["reconstruction_max_abs"] for row in decomposition) <= 1e-4,
        "file_hashes": file_hashes,
        "finite": all(np.isfinite(row["minimum_pairwise_cosine"]) for row in triadic) and all(np.isfinite(row["minimum_partition_cosine"]) for row in cross_partition),
        "scope": "retrospective descriptive" in summary["claim_boundary"]["allowed"] and "causal intervention" in summary["claim_boundary"]["missing"],
        "authorization": final["authorization"] == "run_phase1555_c095_behavior_stratified_and_raw_coordinate_atlas",
    }
    result = {"phase": 1554, "campaign": "C095", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
