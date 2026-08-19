#!/usr/bin/env python3
"""Independent file-level audit for Phase1406."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1406_c065_holdout_factorial_swaps"
FIELD = TESTS / "result/phase1405_c065_natural_discovery_field"
CONTRACT = TESTS / "result/phase1403_c065_active_only_natural_state_contract"
ARMS = {"self", "surface_same", "member_same", "family_same_polarity", "polarity_same_family", "family_and_polarity"}


def main() -> None:
    summary = core.load(OUT / "analysis/factorial_swap_summary.json")
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    candidates = core.load(FIELD / "protocol/frozen_natural_event_candidates.json")["candidates"]
    rows = core.rows(OUT / "raw/factorial_swaps.jsonl")
    checks = {
        "rows": len(rows) == 1296,
        "sets": len({r["set_id"] for r in rows}) == 36,
        "partitions": {r["partition"] for r in rows} == {"confirmation", "lockbox"},
        "candidates": {r["candidate_id"] for r in rows} == {r["candidate_id"] for r in candidates},
        "arms": {r["arm"] for r in rows} == ARMS,
        "six_candidates_per_set": all(sum(r["set_id"] == sid for r in rows) == 36 for sid in {r["set_id"] for r in rows}),
        "finite": all(math.isfinite(r[k]) for r in rows for k in ("baseline_margin", "swap_margin", "signed_damage", "loss_fraction")),
        "self": max(abs(r["signed_damage"]) for r in rows if r["arm"] == "self") <= protocol["factorial_swap"]["self_max_abs_diff"],
        "summary": summary["all_checks_passed"] and all(summary["checks"].values()),
        "candidate_hash": summary["candidate_sha256"] == core.sha(FIELD / "protocol/frozen_natural_event_candidates.json"),
        "decision": final["authorization"] == "run_phase1407_c065_campaign_closure",
    }
    result = {
        "phase": 1406,
        "campaign": "C065",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
