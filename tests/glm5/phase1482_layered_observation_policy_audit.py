#!/usr/bin/env python3
"""Independent audit for Phase1482."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1482_layered_observation_policy"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    policy = core.load(OUT / "protocol/layered_observation_policy.json")
    parent = core.load(RESULT / "phase1481_c080_c083_major_stage_closure/analysis/final.json")
    py_compile.compile(str(TESTS / "phase1482_layered_observation_policy.py"), doraise=True)
    unsigned = {key: value for key, value in policy.items() if key not in {"policy_sha256", "authorization"}}
    checks = {
        "digest": policy["policy_sha256"] == core.digest(unsigned),
        "layers": [row["id"] for row in policy["evidence_layers"]] == ["L0", "L1", "L2", "L3", "L4", "L5"],
        "missingness": set(policy["missingness_codes"]) == {"M0", "M1", "M2", "M3", "M4"},
        "prospective": policy["effective_scope"].startswith("prospective campaigns"),
        "historical_lock": any("C080, C081, and C083" in rule for rule in policy["route_rules"]),
        "behavior_scope": any("Behavior qualification remains mandatory" in rule for rule in policy["route_rules"]),
        "parent_restart": parent["authorization"].endswith("project_level_gate_policy_is_preregistered"),
        "final": final["status"] == "project_level_gate_policy_preregistered" and not final["model_run"] and not final["hidden_access"],
    }
    result = {"phase": 1482, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
