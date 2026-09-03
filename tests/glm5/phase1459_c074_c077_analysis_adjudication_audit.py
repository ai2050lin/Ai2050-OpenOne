#!/usr/bin/env python3
"""Independent audit for Phase1459 adjudication."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1459_c074_c077_analysis_adjudication"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "main": all(final["checks"].values()),
        "accepted": len(final["accepted"]) == 5,
        "corrections": len(final["corrections"]) == 5,
        "rejected": len(final["rejected"]) == 5,
        "behavior_first": final["c078_constraints"]["behavior_first"],
        "scope": set(final["c078_constraints"]["observables"]) == {"embeddings", "all Hidden States", "yes/no logits"},
        "forbidden": all(name in final["c078_constraints"]["forbidden"] for name in ("attention", "MLP", "PCA", "TDA")),
        "authorization": final["authorization"] == "run_phase1460_c078_contract",
    }
    result = {"phase": 1459, "campaign": "C074-C077", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
