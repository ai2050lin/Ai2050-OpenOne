#!/usr/bin/env python3
"""Independent audit for Phase1477."""
from __future__ import annotations

import itertools
import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
ATLAS = RESULT / "phase1476_c082_coordinate_atlas"
OUT = RESULT / "phase1477_c082_atlas_synthesis"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1477_c082_atlas_synthesis as phase


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    synthesis = core.load(OUT / "analysis/synthesis.json")
    manifest = core.load(OUT / "frozen/future_prediction_manifest.json")
    metadata = core.load(ATLAS / "analysis/atlas_metadata.json")
    means = np.load(ATLAS / "atlas/mean_effect.float32.npy", mmap_mode="r")
    common_saved = np.load(OUT / "frozen/common_boundary_state35_vector.float32.npy")
    role = metadata["roles"].index("boundary")
    vectors = phase.relation_vectors(means, 35, role)
    common = np.mean(np.stack(vectors, axis=0), axis=0, dtype=np.float32)
    sets = [set(np.argpartition(vector * vector, -26)[-26:].tolist()) for vector in vectors]
    intersection = sorted(set.intersection(*sets))
    union = sorted(set.union(*sets))
    early = phase.pairwise_summary(phase.relation_vectors(means, 0, metadata["roles"].index("query_label")))
    late = phase.pairwise_summary(vectors)
    py_compile.compile(str(TESTS / "phase1477_c082_atlas_synthesis.py"), doraise=True)
    checks = {
        "final": final["synthesis_complete"] and all(final["checks"].values()),
        "common_vector": np.array_equal(common, common_saved),
        "common_hash": manifest["common_vector_sha256"] == core.sha(OUT / "frozen/common_boundary_state35_vector.float32.npy"),
        "coordinates": intersection == manifest["frozen_coordinates"]["boundary_state35_top_1pct_intersection"] and union == manifest["frozen_coordinates"]["boundary_state35_top_1pct_union"],
        "early": abs(early["maximum"] - synthesis["trajectory_pairwise_cosine"]["query_label_state0"]["maximum"]) < 1e-12,
        "late": abs(late["minimum"] - synthesis["trajectory_pairwise_cosine"]["boundary_state35"]["minimum"]) < 1e-12,
        "freeze": manifest["freeze_sha256"] == core.digest({key: value for key, value in manifest.items() if key != "freeze_sha256"}) == final["freeze_sha256"],
        "predictions": [row["id"] for row in manifest["future_fresh_material_predictions"]] == [f"P082-{index}" for index in range(1, 6)],
        "scope": manifest["not_confirmed_here"] and synthesis["claim_boundary"].startswith("retrospective C079 observation"),
        "authorization": final["authorization"] == "run_phase1478_c082_campaign_closure",
    }
    result = {"phase": 1477, "campaign": "C082", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
