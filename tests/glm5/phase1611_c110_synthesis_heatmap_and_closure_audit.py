#!/usr/bin/env python3
"""Independent audit for the C110 synthesis, heatmap update, and major-stage closure."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1611_c110_synthesis_heatmap_and_closure.py"
    py_compile.compile(str(producer), doraise=True)
    report = core.load(OUT / "audit/internal_closure_audit.json")
    closure = core.load(OUT / "analysis/closure.json")
    payload = core.load(PUBLIC)
    contrasts = core.rows(OUT / "analysis/transport_pair_contrast_summary.jsonl")
    c110_effects = [row for row in payload["effect_rows"] if row.get("dataset") == "C110_fresh"]
    c110_raw = [row for row in payload["raw_rows"] if row.get("dataset") == "C110_fresh"]
    checks = {
        "producer": core.sha(producer) == report["producer_sha256"],
        "internal": report["all_checks_passed"],
        "schema": payload["schema"] == "c109_role_state_field_atlas.v1" and payload["phase"] == 1611,
        "coordinates": payload["dimensions"] == list(range(2560)),
        "fresh_effects": len(c110_effects) == 4 and all(len(row["values"]) == 2560 for row in c110_effects),
        "fresh_raw": len(c110_raw) == 12 and {row["state"] for row in c110_raw} == {0, 19, 36},
        "contrasts": len(contrasts) == 8 and sum(row["query_plus_record_additional_truth_flips"] for row in contrasts) == 0,
        "hash": core.sha(PUBLIC) == report["asset_sha256"] == closure["heatmap"]["sha256"],
        "puzzles": set(closure["new_puzzles"]) == {"K286-R1", "K287-R1", "K288-BOUND"},
        "claims": "one-to-one semantic coordinate values" in closure["claim_boundary"] and "minimal role coalition" in closure["claim_boundary"],
        "authorization": report["authorization"] == "audit_frontend_build_append_c110_memo_and_close_major_stage",
    }
    result = {"phase": 1611, "campaign": "C110", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_closure_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
