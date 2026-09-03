#!/usr/bin/env python3
"""Independent audit for Phase1466 C079 discovery freeze."""
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
import phase1466_c079_discovery_basic_observation_and_freeze as phase

CAPTURE = TESTS / "result/phase1465_c079_discovery_full_field_capture"
OUT = TESTS / "result/phase1466_c079_discovery_basic_observation_and_freeze"


def main() -> None:
    metadata = core.load(CAPTURE / "analysis/capture_metadata.json")
    index = core.rows(CAPTURE / "raw/discovery_role_field_index.jsonl")
    manifest = core.load(OUT / "frozen/candidate_manifest.json")
    summary = core.load(OUT / "analysis/observation_summary.json")
    final = core.load(OUT / "analysis/final.json")
    pool = core.rows(OUT / "analysis/relation_candidate_pool.jsonl")
    relation_order = list(dict.fromkeys(row["relation"] for row in manifest["candidates"]))
    expected = phase.select_candidates(pool, relation_order)
    actual = [{key: row[key] for key in expected[0]} for row in manifest["candidates"]]
    vectors = np.load(OUT / "frozen/discovery_candidate_mean_vectors.npz")
    checks = {
        "raw_hash": core.sha(CAPTURE / "raw/discovery_role_field.float16.npy") == metadata["raw_sha256"] == manifest["source_raw_sha256"],
        "discovery": all(row["partition"] == "response_discovery" for row in index) and not manifest["holdout_accessed"],
        "metrics": summary["surface_metric_rows"] == 11988 and summary["cross_metric_rows"] == 5994 and summary["candidate_pool_rows"] == 1998,
        "selection": expected == actual,
        "candidate_count": len(manifest["candidates"]) == summary["candidate_count"] == 18,
        "relations": Counter(row["relation"] for row in manifest["candidates"]) == {relation: 3 for relation in relation_order},
        "distinct_roles": all(len({row["role"] for row in manifest["candidates"] if row["relation"] == relation}) == 3 for relation in relation_order),
        "vectors": len(vectors.files) == 36 and core.sha(OUT / "frozen/discovery_candidate_mean_vectors.npz") == manifest["vector_npz_sha256"],
        "freeze": core.digest({key: value for key, value in manifest.items() if key != "freeze_sha256"}) == manifest["freeze_sha256"] == final["freeze_sha256"],
        "thresholds": manifest["validation_thresholds"] == {"cosine_to_discovery_each_surface_min": 0.7, "holdout_cross_surface_cosine_min": 0.7, "direction_to_discovery_min": 0.5, "selectivity_ratio_min": 1.0, "both_confirmation_and_lockbox_required": True},
        "authorization": final["authorization"] == "run_phase1467_c079_holdout_capture_and_validation",
    }
    result = {"phase": 1466, "campaign": "C079", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
