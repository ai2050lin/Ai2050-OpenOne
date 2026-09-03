#!/usr/bin/env python3
"""Independent audit for Phase1520 full-input correction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1520_c088_full_input_code_semantics_correction"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    report = core.load(OUT / "analysis/full_input_code_semantics_correction.json")
    sample = report["decoded_full_input_sample"]
    checks = {
        "status": final["status"] == "phase1519_superseded_full_input_semantics_restored",
        "counts": report["case_count"] == report["compiled_count"] == 1984,
        "exact_recompile": report["exact_chat_recompile_count"] == 1984,
        "mapping_all": report["full_input_mapping_definition_count"] == 1984,
        "sample_standard": "standard code means same -> yes and different -> no" in sample,
        "sample_reversed": "reversed code means same -> no and different -> yes" in sample,
        "supersession": report["phase1519_status"] == "superseded_due_to_incomplete_input_scope",
        "bounded_restoration": len(report["claim_restoration"]["retain_boundaries"]) == 3,
        "authorization": final["authorization"] == "preregister_c089_natural_relation_full_state_observation_atlas",
        "no_model": final["checks"]["no_model_run"],
    }
    audit = {
        "phase": 1520,
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
