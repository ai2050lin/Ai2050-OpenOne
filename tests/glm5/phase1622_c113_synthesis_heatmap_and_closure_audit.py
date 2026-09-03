#!/usr/bin/env python3
"""Independent closure audit for Phase1622 / C113."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1618_c113_fourth_lexicon_role_lattice_replication"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    report = core.load(OUT / "audit/internal_closure_audit.json")
    closure = core.load(OUT / "analysis/closure.json")
    asset = core.load(PUBLIC)
    canonical = OUT / "visualization/c109_c113_coordinate_multi_position_atlas.json"
    c113_effects = [row for row in asset["effect_rows"] if row.get("dataset") == "C113"]
    c113_raw = [row for row in asset["raw_rows"] if row.get("dataset") == "C113"]
    checks = {
        "internal": report["all_checks_passed"],
        "producer": report["producer_sha256"] == core.sha(TESTS / "phase1622_c113_synthesis_heatmap_and_closure.py"),
        "historical_identity": core.sha(canonical) == report["asset_sha256"] == closure["heatmap"]["sha256"],
        "current_public_superset": "c113_batch" in asset and asset["dimensions"] == list(range(2560)),
        "schema": asset["schema"] == "c109_role_state_field_atlas.v1" and asset["phase"] >= 1622 and "c113_batch" in asset,
        "coordinates": asset["dimensions"] == list(range(2560)) and len(c113_effects) == 280 and all(len(row["values"]) == 2560 for row in c113_effects),
        "raw": len(c113_raw) == 28 and {row["state_kind"] for row in c113_raw} == {"embedding", "hidden_state"},
        "puzzles": set(closure["new_puzzles"]) == {"K295", "K296", "K297"},
        "boundary": "not weights" in closure["claim_boundary"] and "task-aligned rescue is typed missingness" in " ".join(closure["problems"]),
        "authorization": closure["next_authorization"].startswith("C114 existing-data structural atlas"),
    }
    audit = {"phase": 1622, "campaign": "C113", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": "build_client_append_c113_memo_and_execute_c114"}
    if not audit["all_checks_passed"]:
        raise RuntimeError(audit)
    core.save(OUT / "audit/independent_closure_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
