#!/usr/bin/env python3
"""Independent artifact, contract and numeric audit for Phase1571 / C098."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1571_c098_observation_first_graph_campaign"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1571_c098_observation_first_graph_campaign.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    summary = core.load(OUT / "analysis/c098_graph_field_summary.json")
    final = core.load(OUT / "analysis/final.json")
    raw = np.load(OUT / "raw/all_token_all_state_field.float16.npy", mmap_mode="r")
    walsh = np.load(OUT / "raw/focus_role_walsh_coefficients.float32.npy", mmap_mode="r")
    index = core.rows(OUT / "raw/all_token_field_index.jsonl")
    validation = core.rows(OUT / "analysis/dual_holdout_xy_validation.jsonl")
    design_null = core.rows(OUT / "analysis/c097_shared_cell_design_null.jsonl")
    visualization = core.load(OUT / "analysis/visualization_decision.json")
    checks = {
        "producer_compiles": True,
        "producer_frozen": protocol["producer_sha256"] == core.sha(producer),
        "premodel": pre["all_checks_passed"] and pre["passed"] == pre["total"],
        "material_hashes": all(
            core.sha(ROOT / path) == protocol["material"][key]
            for key, path in (
                ("unit_sha256", "tests/glm5/result/phase1571_c098_observation_first_graph_campaign/material/frozen_graph_units.jsonl"),
                ("case_sha256", "tests/glm5/result/phase1571_c098_observation_first_graph_campaign/material/frozen_cases.jsonl"),
                ("compiled_sha256", "tests/glm5/result/phase1571_c098_observation_first_graph_campaign/compiled/qwen3_active.jsonl"),
            )
        ),
        "raw_shape": list(raw.shape) == capture["shape"] and raw.dtype == np.float16,
        "raw_hash": core.sha(OUT / "raw/all_token_all_state_field.float16.npy") == capture["raw_sha256"],
        "walsh_shape": list(walsh.shape) == summary["walsh"]["shape"] and walsh.dtype == np.float32,
        "walsh_hash": core.sha(OUT / "raw/focus_role_walsh_coefficients.float32.npy") == summary["walsh"]["sha256"],
        "coverage": len(index) == 1152 and index[-1]["token_end"] == raw.shape[1],
        "finite_sample": bool(np.isfinite(np.asarray(raw[:, :: max(1, raw.shape[1] // 257), :64])).all()) and bool(np.isfinite(np.asarray(walsh[:, :, :, :, :64])).all()),
        "numeric": all(capture["checks"].values()),
        "holdout_count": len(validation) == 3 * 4 * 37 * 4 * 2,
        "discovery_only_support": all(row["partition"] in ("confirmation", "lockbox") for row in validation),
        "design_null": len(design_null) == 12 and all(row["null_q99"] >= row["null_q95"] >= row["null_median"] for row in design_null),
        "visualization": (not visualization["important"]) or (
            visualization["client_identity"]
            and core.sha(ROOT / visualization["asset"]) == visualization["sha256"]
            and core.sha(ROOT / visualization["client"]) == visualization["sha256"]
        ),
        "scope": "new mathematics" in protocol["claim_boundary"]["forbidden"] and "attention" in protocol["forbidden"] and "MLP" in protocol["forbidden"],
        "final": final["all_checks_passed"] and final["status"] == "major_observation_stage_complete",
    }
    result = {
        "phase": 1571,
        "campaign": "C098",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
