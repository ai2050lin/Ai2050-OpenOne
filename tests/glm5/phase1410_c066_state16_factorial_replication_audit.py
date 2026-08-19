#!/usr/bin/env python3
"""Independent audit for Phase1410."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1408_c066_midstate_breadth_contract"
OUT = TESTS / "result/phase1410_c066_state16_factorial_replication"
ARMS = {"self", "surface_same", "member_same", "family_same_polarity", "polarity_same_family", "family_and_polarity"}


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/state16_replication_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/state16_factorial_swaps.jsonl")
    checks = {
        "rows": len(rows) == 1440,
        "sets": len({r["set_id"] for r in rows}) == 120,
        "splits": {r["partition"] for r in rows} == {"confirmation", "lockbox"},
        "state16": {r["state_index"] for r in rows} == {16},
        "candidates": len({r["candidate_id"] for r in rows}) == 6,
        "arms": {r["arm"] for r in rows} == ARMS,
        "finite": all(math.isfinite(r[k]) for r in rows for k in ("baseline_margin", "swap_margin", "signed_damage", "loss_fraction")),
        "self": max(abs(r["signed_damage"]) for r in rows if r["arm"] == "self") <= protocol["mechanism"]["self_max_abs_diff"],
        "summary": summary["all_checks_passed"] and all(summary["checks"].values()),
        "contract": summary["contract_sha256"] == protocol["contract_sha256"],
        "authorization": final["authorization"] == "run_phase1411_c066_campaign_closure",
    }
    result = {"phase": 1410, "campaign": "C066", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
