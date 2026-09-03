#!/usr/bin/env python3
"""Independent audit for Phase1679/C145."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
OUT = RESULT / "phase1679_c145_correct_error_depth_trajectory_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
freeze = core.load(OUT / "protocol/frozen_error_nominee.json"); report = core.load(OUT / "analysis/confirmation.json")
d = np.load(OUT / "analysis/discovery_matched_error_residuals.float32.npy", mmap_mode="r")
c = np.load(OUT / "analysis/confirmation_exact_support_error_residuals.float32.npy", mmap_mode="r")
e = np.load(OUT / "analysis/confirmation_exploratory_all_error_residuals.float32.npy", mmap_mode="r")
checks = {
    "internal_discovery": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"],
    "internal_confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_checks_passed"],
    "internal_closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
    "shapes": list(d.shape) == [8,6,38,2560] and list(c.shape) == [7,6,38,2560] and list(e.shape) == [11,6,38,2560],
    "freeze": report["nominee"]["role"] == freeze["role"] and report["nominee"]["checkpoint"] == freeze["checkpoint"],
    "behavior_cells": len(core.load(OUT / "analysis/behavior_depth_table.json")) == 8,
    "depth_rows": len(report["depth_effect_rows"]) == 228,
    "typed_missingness": report["missing_exact_support_count"] == 4 and not report["eligibility_passed"],
    "boundary": "not a cause" in report["claim_boundary"],
}
audit = {"phase":1679,"campaign":"C145","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_error_replication_passed":report["matched_error_replication_passed"],"authorization":"start_C146"}
core.save(OUT / "audit/independent_closure_audit.json", audit); print(json.dumps(audit, indent=2))
