#!/usr/bin/env python3
"""Independent audit for Phase1551."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1551_c094_and_output_orthogonalization_final_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    report = core.load(OUT / "analysis/macro_stage_final_adjudication.json")
    recommendation = core.load(OUT / "protocol/next_stage_requirements.json")
    checks = {
        "nine_audits": set(report["audits"]) == {str(value) for value in range(1542, 1551)},
        "audits_passed": all(left == right for value in report["audits"].values() for left, right in [value.split("/")]),
        "c094_branch": not report["c094"]["native_qualified"] and report["c094"]["reversed_qualified"] and not report["c094"]["hidden_states_accessed"],
        "k267_preserved": "K267 remains valid" in report["macro_answer"]["interpretation"],
        "no_new_puzzle": report["core_puzzle_update"] == "none_after_K267",
        "no_terms_identified": report["theory"]["identified_terms"] == [],
        "math_gate": report["theory"]["new_mathematics_gate"] == "closed",
        "recommendation_identical": recommendation == report["recommendation"],
        "automatic_stop_is_scoped": recommendation["automatic_model_run_authorized"] is False and recommendation["next_campaign"] == "C095",
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "no_automatic_model_run_until_C095_object_contract",
    }
    result = {"phase": 1551, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
