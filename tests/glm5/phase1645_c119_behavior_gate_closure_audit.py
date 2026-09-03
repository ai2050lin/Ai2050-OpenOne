#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1643_c119_identifiable_default_override_campaign"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

if __name__ == "__main__":
    closure = core.load(OUT / "analysis/closure.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    internal = core.load(OUT / "audit/internal_closure_audit.json")
    checks = {"internal": internal["all_checks_passed"], "status": closure["status"] == "explicit_applicability_repair_failed_default_inheritance_gate", "behavior": closure["headline"] == capture["behavior"], "gate": closure["gate_checks"] == capture["behavior_gate_checks"] and not capture["behavior_gate_passed"], "paired": abs(closure["paired_change"] - (closure["headline"]["default_inheritance"] - closure["paired_c118_default_inheritance"])) < 1e-12, "sealed": "sealed" in closure["raw_archive_status"], "authorization": closure["next_authorization"].startswith("C120")}
    report = {"phase": 1645, "campaign": "C119", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": closure["next_authorization"]}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/independent_closure_audit.json", report)
    print(json.dumps(report, indent=2))
