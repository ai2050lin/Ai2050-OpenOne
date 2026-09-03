#!/usr/bin/env python3
"""Independent audit for Phase1534."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1534_c089_c090_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    closure = core.load(OUT / "analysis/major_stage_closure.json")
    py_compile.compile(str(TESTS / "phase1534_c089_c090_major_stage_closure.py"), doraise=True)
    checks = {
        "status": final["status"] == "major_stage_closed_with_k266_descriptive",
        "ledger": len(closure["audit_ledger"]) == 14 and all(row["all_checks_passed"] for row in closure["audit_ledger"].values()),
        "camera": closure["canonical_results"]["causal_prefix_max_relative_l2"] == 0.0,
        "behavior": closure["canonical_results"]["behavior_qualified_families"] == [],
        "replication": all(closure["canonical_results"]["family_descriptive_replication"].values()) and closure["canonical_results"]["shared_descriptive_replication"],
        "scope": closure["puzzle_update"]["evidence"] == "E3-HS-descriptive; semantic qualification absent",
        "theory": not closure["theory"]["new_mathematics_required"],
        "checks": all(closure["checks"].values()),
        "authorization": final["authorization"] == "preregister_c091_behavior_grounded_natural_relation_latent_use_bridge",
    }
    result = {"phase": 1534, "campaign": "C089-C090", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
