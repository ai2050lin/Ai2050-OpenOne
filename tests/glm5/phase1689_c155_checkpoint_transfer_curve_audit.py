#!/usr/bin/env python3
"""Independent audit for Phase1689/C155."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1689_c155_checkpoint_transfer_curve"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

r = core.load(OUT / "analysis/transfer_curve.json")
scores = np.load(OUT / "raw/checkpoint_candidate_logits.float32.npy", mmap_mode="r")
payload = core.load(PUBLIC)
best = max(r["checkpoint_rows"], key=lambda row: row["mean_gain"])["state"]
broad = sum(row["mean_gain"] > 0 and row["donor_choice_increase"] >= 0.10 for row in r["checkpoint_rows"])
checks = {
    "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
    "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"],
    "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
    "shape": list(scores.shape) == [12, 128, 2],
    "rows": len(r["checkpoint_rows"]) == 11,
    "best": best == r["best_state"],
    "broad": broad == r["broad_checkpoint_count"],
    "asset": payload["phase"] == 1689 and "c155_checkpoint_transfer" in payload,
    "hash": core.sha(PUBLIC) == core.load(OUT / "audit/internal_closure_audit.json")["asset_sha256"],
}
audit = {"phase": 1689, "campaign": "C155", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "memo_and_big_stage_synthesis"}
core.save(OUT / "audit/independent_closure_audit.json", audit)
print(json.dumps(audit, indent=2))
