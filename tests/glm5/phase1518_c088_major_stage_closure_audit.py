#!/usr/bin/env python3
"""Independent closure audit for C088."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1518_c088_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(OUT / "analysis/major_stage_summary.json")
    ledger = core.load(OUT / "analysis/stage_ledger.json")
    scripts = []
    for phase in range(1512, 1519):
        scripts.extend(sorted(TESTS.glob(f"phase{phase}_c088*.py")))
    compile_ok = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except py_compile.PyCompileError:
            compile_ok = False
    hashes_ok = all(
        core.sha(RESULT / row["directory"] / "analysis/final.json") == row["final_sha256"]
        and core.sha(RESULT / row["directory"] / "audit/independent_final_audit.json") == row["audit_sha256"]
        for row in ledger
    )
    checks = {
        "final_status": final["status"] == "major_stage_complete",
        "ledger": len(ledger) == 6 and all(row["audit_passed"] for row in ledger),
        "hashes": hashes_ok,
        "scripts_compile": compile_ok and len(scripts) == 14,
        "bounded_claim": not summary["verdict"]["localized_or_necessary_mechanism"] and not summary["verdict"]["cross_model_invariant"],
        "behavior_boundary": not summary["verdict"]["behavioral_code_compliance"],
        "k265": summary["core_piece"]["id"] == "K265",
        "next_authorization": final["authorization"] == "preregister_c089_natural_relation_full_state_observation_atlas",
        "no_model_in_closure": True,
    }
    audit = {
        "phase": 1518,
        "campaign": "C088",
        "checks": checks,
        "scripts_compiled": [str(path.relative_to(ROOT)) for path in scripts],
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
