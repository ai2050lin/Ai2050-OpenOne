#!/usr/bin/env python3
"""Independent audit for the C112 synthesis and current major-stage closure."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1615_c112_value_identity_role_lattice"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1617_c112_synthesis_heatmap_and_closure.py"
    py_compile.compile(str(producer), doraise=True)
    report = core.load(OUT / "audit/internal_closure_audit.json")
    closure = core.load(OUT / "analysis/closure.json")
    payload = core.load(PUBLIC)
    checks = {
        "producer": core.sha(producer) == report["producer_sha256"],
        "internal": report["all_checks_passed"],
        "schema": payload["schema"] == "c109_role_state_field_atlas.v1" and payload["phase"] == 1617,
        "coordinates": payload["dimensions"] == list(range(2560)),
        "summaries": len(payload["c112_batch"]["summaries"]) == 8,
        "l2": payload["c112_batch"]["max_permutation_l2_relative_error"] < 0.001,
        "hash": core.sha(PUBLIC) == report["asset_sha256"] == closure["heatmap"]["sha256"],
        "puzzles": set(closure["new_puzzles"]) == {"K292-R1", "K293-R1", "K294-R1"},
        "claims": "minimal or necessary circuit" in closure["claim_boundary"] and "semantic neurons" in closure["claim_boundary"],
        "authorization": report["authorization"] == "audit_frontend_append_c112_memo_and_close_current_major_stage",
    }
    result = {"phase": 1617, "campaign": "C112", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_closure_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
