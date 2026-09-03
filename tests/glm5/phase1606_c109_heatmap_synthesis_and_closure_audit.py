#!/usr/bin/env python3
"""Independent audit for the C109 synthesis, heatmap, and closure."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1606_c109_heatmap_synthesis_and_closure.py"
    py_compile.compile(str(producer), doraise=True)
    report = core.load(OUT / "audit/internal_closure_audit.json")
    closure = core.load(OUT / "analysis/closure.json")
    payload = core.load(PUBLIC)
    leverage = core.rows(OUT / "analysis/c108_pair_coordinate_leverage.jsonl")
    checks = {
        "producer": core.sha(producer) == report["producer_sha256"],
        "internal": report["all_checks_passed"],
        "schema": payload["schema"] == "c109_role_state_field_atlas.v1" and payload["result_type"] == "role_state_field_atlas_heatmap",
        "coordinates": payload["dimensions"] == list(range(2560)),
        "raw_states": {row["state"] for row in payload["raw_rows"]} == set(range(37)),
        "leverage": len(leverage) == 192,
        "hash": core.sha(PUBLIC) == report["asset_sha256"] == closure["heatmap"]["sha256"],
        "claims": set(closure["new_puzzles"]) == {"K283-R1", "K284-OBS", "K285-CONTROL"},
        "corrections": "task closure" in closure["claim_corrections"]["K281-R1"] and "were not refuted" in closure["claim_corrections"]["K282-R1"],
        "authorization": report["authorization"] == "audit_frontend_build_append_memo_and_close_c109",
    }
    result = {"phase": 1606, "campaign": "C109", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_closure_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
