#!/usr/bin/env python3
"""Independent audit for Phase1688/C154."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1688_c154_type_graph_hiddenstate_causal_adjudication"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

r = core.load(OUT / "analysis/causal.json")
scores = np.load(OUT / "raw/intervention_candidate_logits.float32.npy", mmap_mode="r")
payload = core.load(PUBLIC)
checks = {
    "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
    "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"],
    "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
    "score_shape": list(scores.shape) == [8, 128, 2],
    "gate_recomputed": r["causal_gate_passed"] == all(r["gates"].values()),
    "controls": set(r["paired_win_rates"]) == {"reverse", "wrong_role", "wrong_coordinate", "wrong_checkpoint"},
    "asset": payload["phase"] == 1688 and "c154_type_graph_causal" in payload,
    "hash": core.sha(PUBLIC) == core.load(OUT / "audit/internal_closure_audit.json")["asset_sha256"],
}
audit = {"phase": 1688, "campaign": "C154", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": r["causal_gate_passed"], "authorization": "memo_and_campaign_synthesis"}
core.save(OUT / "audit/independent_closure_audit.json", audit)
print(json.dumps(audit, indent=2))
