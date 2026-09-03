#!/usr/bin/env python3
"""Independent audit for Phase1467 C079 holdout validation."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

DISCOVERY = TESTS / "result/phase1466_c079_discovery_basic_observation_and_freeze"
OUT = TESTS / "result/phase1467_c079_holdout_capture_and_validation"


def main() -> None:
    manifest = core.load(DISCOVERY / "frozen/candidate_manifest.json")
    summary = core.load(OUT / "analysis/holdout_summary.json")
    final = core.load(OUT / "analysis/final.json")
    validation = core.rows(OUT / "analysis/candidate_holdout_validation.jsonl")
    index = core.rows(OUT / "raw/holdout_role_field_index.jsonl")
    field = np.load(OUT / "raw/holdout_role_field.float16.npy", mmap_mode="r")
    thresholds = manifest["validation_thresholds"]
    recalculated = []
    for candidate in manifest["candidates"]:
        rows = [row for row in validation if row["candidate_id"] == candidate["candidate_id"]]
        split_pass = []
        for row in rows:
            surface_pass = all(value["cosine_to_discovery"] >= thresholds["cosine_to_discovery_each_surface_min"] and value["direction_to_discovery"] >= thresholds["direction_to_discovery_min"] and value["selectivity_ratio"] >= thresholds["selectivity_ratio_min"] for value in row["surface"].values())
            split_pass.append(surface_pass and row["holdout_cross_surface_cosine"] >= thresholds["holdout_cross_surface_cosine_min"] and row["split_passed"])
        if len(rows) == 2 and all(split_pass):
            recalculated.append(candidate["candidate_id"])
    finite = True
    for start in range(0, len(field), 64):
        finite = finite and bool(np.isfinite(field[start:start + 64]).all())
    checks = {
        "capture": all(summary["capture_checks"].values()),
        "shape": list(field.shape) == summary["shape"] == [2208, 37, 9, 2560] and field.dtype == np.float16,
        "index": len(index) == 2208 and [row["row_index"] for row in index] == list(range(2208)),
        "splits": Counter(row["partition"] for row in index) == {"confirmation": 1088, "lockbox": 1120},
        "finite": finite,
        "raw_hash": core.sha(OUT / "raw/holdout_role_field.float16.npy") == summary["raw_sha256"],
        "index_hash": core.sha(OUT / "raw/holdout_role_field_index.jsonl") == summary["index_sha256"],
        "validation_rows": len(validation) == 36 and Counter(row["split"] for row in validation) == {"confirmation": 18, "lockbox": 18},
        "decision": recalculated == summary["robust_candidates"] == final["robust_candidates"],
        "freeze": summary["discovery_freeze_sha256"] == manifest["freeze_sha256"],
        "authorization": final["authorization"] == "run_phase1468_c079_campaign_closure",
    }
    result = {"phase": 1467, "campaign": "C079", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
