#!/usr/bin/env python3
"""Independent audit for Phase1573 / C100."""
from __future__ import annotations

import json
import os
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
C099 = RESULT / "phase1572_c099_fixed_width_graph_field_campaign"
OUT = RESULT / "phase1573_c100_graph_field_analysis_adapter"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1573_c100_graph_field_analysis_adapter.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    pre = core.load(OUT / "audit/pre_analysis_adapter_audit.json")
    summary = core.load(OUT / "analysis/c098_graph_field_summary.json")
    final = core.load(OUT / "analysis/final.json")
    raw = np.load(OUT / "raw/all_token_all_state_field.float16.npy", mmap_mode="r")
    walsh = np.load(OUT / "raw/focus_role_walsh_coefficients.float32.npy", mmap_mode="r")
    visualization = core.load(OUT / "analysis/visualization_decision.json")
    checks = {
        "producer_compiles": True,
        "producer_frozen": protocol["producer_sha256"] == core.sha(producer),
        "adapter_audited": pre["all_checks_passed"],
        "hardlink": os.path.samefile(C099 / "raw/all_token_all_state_field.float16.npy", OUT / "raw/all_token_all_state_field.float16.npy"),
        "raw_hash": core.sha(OUT / "raw/all_token_all_state_field.float16.npy") == protocol["source_raw_sha256"],
        "raw_shape": list(raw.shape) == [37, 191600, 2560] and raw.dtype == np.float16,
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
        "final": final["all_checks_passed"] and final["status"] == "graph_field_observation_major_stage_complete",
    }
    result = {"phase": 1573, "campaign": "C100", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
