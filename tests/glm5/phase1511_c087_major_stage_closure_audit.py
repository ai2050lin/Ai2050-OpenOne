#!/usr/bin/env python3
"""Independent audit for the C087 closure."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1511_c087_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    py_compile.compile(str(TESTS / "phase1511_c087_major_stage_closure.py"), doraise=True)
    checks = {
        "phase_chain": set(final["audits"]) == {str(i) for i in range(1504, 1511)},
        "audit_count": len(final["audits"]) == 7,
        "compile_count": len(final["compiled_scripts"]) == 14,
        "formal_failure": not final["answer"]["field"]["dual_holdout_primary_pass"],
        "candidate_evidence": final["core_puzzle"]["id"] == "K264" and final["core_puzzle"]["evidence"].startswith("E2-"),
        "theory_stable": final["theory"]["name"] == "条件化输出场闭合理论" and not final["theory"]["new_foundational_mathematics"],
        "authorization": final["authorization"] == "preregister_c088_cross_root_semantic_by_answer_code_factorial",
        "checks": all(final["checks"].values()),
    }
    result = {"phase": 1511, "campaign": "C087", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
