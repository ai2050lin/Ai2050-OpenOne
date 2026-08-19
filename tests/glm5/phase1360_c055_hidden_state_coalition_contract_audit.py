#!/usr/bin/env python3
"""Independent audit for Phase1360/C055."""
from __future__ import annotations

import json
import py_compile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1360_c055_hidden_state_coalition_contract"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    protocol = load(OUT / "protocol/preregistration.json")
    preaudit = load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    final = load(OUT / "analysis/final.json")
    replay = rows(OUT / "material/causal_replay_manifest.jsonl")
    cell_counts = Counter((x["partition"], x["surface"], x["recipient_tested_family"]) for x in replay)
    checks = {
        "preaudit": preaudit["all_checks_passed"],
        "finite_coalitions": len(protocol["coalitions"]) == 7,
        "roles": {role for roles in protocol["coalitions"].values() for role in roles}
                 == {"target", "family", "boundary"},
        "balanced_cells": len(cell_counts) == 18 and len(set(cell_counts.values())) == 1,
        "case_count": len(replay) == protocol["material"]["causal_case_count"],
        "minimum_cases": len(replay) >= 162,
        "hidden_state_only": all(term in protocol["forbidden"] for term in
                                 ("attention weights", "MLP states or weights", "learned probe")),
        "finite_branching": set(protocol["branching"]) == {"phase1361", "phase1362", "phase1363", "phase1364", "finish"},
        "contract_link": final["contract_sha256"] == protocol["contract_sha256"],
        "authorization": final["authorization"] == "run_phase1361_c055_hidden_state_observation",
        "script_compiles": True,
    }
    try:
        py_compile.compile(str(TESTS / "phase1360_c055_hidden_state_coalition_contract.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    result = {"phase": 1360, "campaign": "C055", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
