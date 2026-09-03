#!/usr/bin/env python3
"""Independent audit for Phase1498."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1498_c086_all_case_field_capture"
BEHAVIOR = TESTS / "result/phase1497_c086_behavior_stratification"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    meta = core.load(OUT / "analysis/capture_metadata.json")
    index = core.rows(OUT / "raw/all_role_field_index.jsonl")
    arr = np.load(OUT / "raw/all_role_field.float16.npy", mmap_mode="r")
    behavior = {r["case_id"]: r for r in core.rows(BEHAVIOR / "raw/behavior.jsonl")}
    py_compile.compile(str(TESTS / "phase1498_c086_all_case_field_capture.py"), doraise=True)
    sample = np.asarray(arr[[0, 3455, 6911]], dtype=np.float32)
    group_rows = core.rows(
        BEHAVIOR / "material/stratified_composition_sets.jsonl"
    )
    by_capture = {r["case_id"]: r for r in index}
    protocol = core.load(
        TESTS / "result/phase1496_c086_unlabeled_counterbalanced_contract/protocol/preregistration.json"
    )
    keys = tuple(
        f"{surface}_{codebook}_{cell}"
        for surface in protocol["surfaces"]
        for codebook in protocol["codebooks"]
        for cell in protocol["cells"]
    )
    stratum_identity = True
    for group in group_rows:
        n = sum(
            by_capture[group[key]]["capture_prediction"]
            == by_capture[group[key]]["gold_position"]
            for key in keys
        )
        observed = "success" if n == 32 else "failed" if n == 0 else "mixed"
        stratum_identity = stratum_identity and observed == group["stratum"]
    agreement_count = sum(
        r["capture_prediction"] == behavior[r["case_id"]]["prediction"] for r in index
    )
    checks = {
        "shape": list(arr.shape) == meta["shape"] == [6912, 37, 7, 2560],
        "index": len(index) == 6912
        and all(r["row_index"] == i for i, r in enumerate(index)),
        "hashes": core.sha(OUT / "raw/all_role_field.float16.npy") == meta["raw_sha256"]
        and core.sha(OUT / "raw/all_role_field_index.jsonl") == meta["index_sha256"],
        "finite_sample": bool(np.isfinite(sample).all()),
        "behavior_stratum_identity": stratum_identity,
        "prediction_disagreement_reported": agreement_count
        == meta["behavior_prediction_agreement_count"],
        "counterbalance": all(
            r["output_yes"] == (r["relation_match"] == (r["code_sign"] == 1))
            for r in index
        ),
        "runtime": all(meta["checks"].values()),
    }
    result = {
        "phase": 1498,
        "campaign": "C086",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
