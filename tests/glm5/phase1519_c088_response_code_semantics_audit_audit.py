#!/usr/bin/env python3
"""Independent audit for the C088 response-code semantic correction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1512_c088_cross_root_semantic_code_factorial_contract"
OUT = RESULT / "phase1519_c088_response_code_semantics_audit"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1519_c088_response_code_semantics_audit import mapping_defined


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    report = core.load(OUT / "analysis/response_code_semantics_audit.json")
    cases = core.rows(CONTRACT / "material/active_cases.jsonl")
    explicit = sum(mapping_defined(row["prompt"], row["codebook"]) for row in cases)
    checks = {
        "authorization": final["authorization"] == "preregister_c089_natural_relation_full_state_observation_atlas",
        "case_count": len(cases) == report["case_count"] == 1984,
        "mapping_recompute": explicit == report["explicit_mapping_definition_count"] == 0,
        "labels_balanced": report["standard_count"] == report["reversed_count"] == 992,
        "withdrawals": len(report["claim_correction"]["withdraw"]) == 3,
        "revised_k265": "same/different-associated" in report["claim_correction"]["k265_revised_title"],
        "no_model": final["checks"]["no_model_run"],
    }
    audit = {
        "phase": 1519,
        "campaign": "C088",
        "checks": checks,
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
