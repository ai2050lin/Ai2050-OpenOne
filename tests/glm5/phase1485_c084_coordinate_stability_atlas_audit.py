#!/usr/bin/env python3
"""Independent audit for Phase1485."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
SOURCE = RESULT / "phase1476_c082_coordinate_atlas/atlas/mean_effect.float32.npy"
OUT = RESULT / "phase1485_c084_coordinate_stability_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left.astype(np.float64), right.astype(np.float64)) / denominator) if denominator > 1e-12 else 0.0


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(OUT / "analysis/coordinate_atlas_summary.json")
    contract = core.load(RESULT / "phase1484_c084_batch_deep_mining_contract/protocol/preregistration.json")
    manifest = core.load(RESULT / "phase1477_c082_atlas_synthesis/frozen/future_prediction_manifest.json")
    py_compile.compile(str(TESTS / "phase1485_c084_coordinate_stability_atlas.py"), doraise=True)
    means = np.load(SOURCE, mmap_mode="r")
    relation_mean = np.mean(means, axis=(1, 2), dtype=np.float64).astype(np.float32)
    signs = np.load(OUT / "atlas/relation_mean_sign.int8.npy", mmap_mode="r")
    support = np.load(OUT / "atlas/support_membership.uint8.npy", mmap_mode="r")
    boundary_role = contract["axes"]["roles"].index("boundary")
    masks = np.asarray(support[1, :, 35, boundary_role], dtype=bool)
    intersection = np.flatnonzero(np.all(masks, axis=0)).tolist()
    early_vectors = relation_mean[:, 0, contract["axes"]["roles"].index("query_label")]
    early_cos = [cosine(early_vectors[i], early_vectors[j]) for i in range(6) for j in range(i + 1, 6)]
    fixed_rows = core.rows(OUT / "analysis/frozen17_sign_audit.jsonl")
    checks = {
        "status": final["status"] == "coordinate_stability_atlas_complete" and all(final["output_checks"].values()),
        "signs": np.array_equal(signs, np.sign(relation_mean).astype(np.int8)),
        "support_counts": all(np.all(np.sum(support[index], axis=-1) == count) for index, count in enumerate(contract["coordinate_branch"]["support_counts"])),
        "intersection": intersection == manifest["frozen_coordinates"]["boundary_state35_top_1pct_intersection"],
        "early_mean": abs(float(np.mean(early_cos)) - summary["state0_cyclic_geometry"]["global|query_label"]["pairwise_cosine"]["mean"]) < 1e-10,
        "fixed17": len(fixed_rows) == 17 and [row["coordinate"] for row in fixed_rows] == intersection,
        "hashes": core.sha(OUT / "atlas/support_membership.uint8.npy") == summary["files"]["support_membership.uint8.npy"]["sha256"] and core.sha(OUT / "atlas/relation_mean_sign.int8.npy") == summary["files"]["relation_mean_sign.int8.npy"]["sha256"],
        "no_model": not final["model_run"],
    }
    result = {"phase": 1485, "campaign": "C084", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
