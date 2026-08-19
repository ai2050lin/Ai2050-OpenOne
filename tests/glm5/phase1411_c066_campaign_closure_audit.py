#!/usr/bin/env python3
"""Independent audit for Phase1411."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1411_c066_campaign_closure"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "status": final["status"] == "closed_after_partial_state16_breadth_replication",
        "record": final["confirmed"]["record_family_state16"] == ["catalog:family_identity:s16"],
        "query": len(final["confirmed"]["query_family_state16"]) == 3,
        "rejected_surfaces": set(final["rejected"]["record_family_state16"]) == {"ordinary:family_identity:s16", "statement:family_identity:s16"},
        "no_layer_search": final["next_question"]["no_layer_search"],
        "authorization": final["authorization"] == "preregister_c067_paired_state_relational_composition",
        "checks": final["all_checks_passed"] and all(final["checks"].values()),
    }
    result = {"phase": 1411, "campaign": "C066", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
