#!/usr/bin/env python3
"""Independent audit for Phase1455 C076 closure."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
PHASE, CAMPAIGN = 1455, "C076"; OUT = TESTS / "result/phase1455_c076_behavior_gate_closure"; P1454 = TESTS / "result/phase1454_c076_behavior"
def main() -> None:
    result = core.load(OUT / "analysis/final.json"); behavior = core.load(P1454 / "analysis/behavior_summary.json")
    checks = {"closure": result["all_checks_passed"] and all(result["checks"].values()), "status": result["status"] == "closed_at_behavior_gate_after_morphology_nuisance_failure", "behavior": not behavior["behavior_qualified"] and behavior["qualified_relations"] == [], "eligible": behavior["eligible_count"] == 57, "hidden": behavior["hidden_state_accessed"] is False, "authorization": result["authorization"] == "preregister_c077_labeled_relation_full_field_calibration"}
    audit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}; core.save(OUT / "audit/independent_final_audit.json", audit); print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]: raise SystemExit(1)
if __name__ == "__main__": main()
