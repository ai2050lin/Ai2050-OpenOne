#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1650_c121_structured_comparison_qualification"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
if __name__ == "__main__":
    closure = core.load(OUT / "analysis/closure.json"); behavior = core.load(OUT / "analysis/behavior_qualification.json"); internal = core.load(OUT / "audit/internal_closure_audit.json")
    checks = {"internal": internal["all_checks_passed"], "headline": closure["headline"] == behavior["behavior"], "failed": not behavior["gate_passed"], "no_hidden": not (OUT / "raw/qwen3_role_subtoken_all_states.uint16.npy").exists(), "authorization": closure["next_authorization"].startswith("execute_C122")}
    report = {"phase": 1652, "campaign": "C121", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": closure["next_authorization"]}
    if not report["all_checks_passed"]: raise RuntimeError(report)
    core.save(OUT / "audit/independent_closure_audit.json", report); print(json.dumps(report, indent=2))
