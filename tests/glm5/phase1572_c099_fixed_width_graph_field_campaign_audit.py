#!/usr/bin/env python3
"""Independent audit for Phase1572 / C099."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C098 = TESTS / "result/phase1571_c098_observation_first_graph_campaign"
OUT = TESTS / "result/phase1572_c099_fixed_width_graph_field_campaign"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1572_c099_fixed_width_graph_field_campaign.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    pre = core.load(OUT / "audit/pre_model_correction_audit.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    summary = core.load(OUT / "analysis/c098_graph_field_summary.json")
    final = core.load(OUT / "analysis/final.json")
    raw = np.load(OUT / "raw/all_token_all_state_field.float16.npy", mmap_mode="r")
    walsh = np.load(OUT / "raw/focus_role_walsh_coefficients.float32.npy", mmap_mode="r")
    visualization = core.load(OUT / "analysis/visualization_decision.json")
    checks = {
        "producer_compiles": True,
        "C098_failure_preserved": core.load(C098 / "analysis/final.json")["status"] == "closed_at_preregistered_numeric_execution_gate",
        "correction_frozen": pre["all_checks_passed"] and protocol["producer_sha256"] == core.sha(producer),
        "only_execution_changed": protocol["single_changed_variable"] == "all batches right-padded to frozen global width 210",
        "material_identity": protocol["material"]["identity_to_C098"],
        "capture_numeric": all(capture["checks"].values()) and capture["numeric"]["causal_prefix_max_abs"] <= 1e-6,
        "raw_shape": list(raw.shape) == capture["shape"] and raw.dtype == np.float16,
        "raw_hash": core.sha(OUT / "raw/all_token_all_state_field.float16.npy") == capture["raw_sha256"],
        "walsh_shape": list(walsh.shape) == summary["walsh"]["shape"] and walsh.dtype == np.float32,
        "walsh_hash": core.sha(OUT / "raw/focus_role_walsh_coefficients.float32.npy") == summary["walsh"]["sha256"],
        "finite_sample": bool(np.isfinite(np.asarray(raw[:, :: max(1, raw.shape[1] // 257), :64])).all()) and bool(np.isfinite(np.asarray(walsh[:, :, :, :, :64])).all()),
        "holdout": len(core.rows(OUT / "analysis/dual_holdout_xy_validation.jsonl")) == 3 * 4 * 37 * 4 * 2,
        "design_null": len(core.rows(OUT / "analysis/c097_shared_cell_design_null.jsonl")) == 12,
        "visualization": (not visualization["important"]) or (
            visualization["client_identity"]
            and core.sha(ROOT / visualization["asset"]) == visualization["sha256"]
            and core.sha(ROOT / visualization["client"]) == visualization["sha256"]
        ),
        "scope": "attention" in protocol["forbidden"] and "MLP" in protocol["forbidden"] and "new mathematics" in protocol["claim_boundary"]["forbidden"],
        "final": final["all_checks_passed"] and final["status"] == "fixed_width_graph_observation_major_stage_complete",
    }
    result = {"phase": 1572, "campaign": "C099", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
