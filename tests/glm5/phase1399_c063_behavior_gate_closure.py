#!/usr/bin/env python3
"""Phase1399: close C063 after the frozen behavior failure."""
from __future__ import annotations

import json
import py_compile
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1399, "C063"
CONTRACT = TESTS / "result/phase1397_c063_identity_polarity_campaign_contract"
BEHAVIOR = TESTS / "result/phase1398_c063_factorized_behavior"
OUT = TESTS / "result/phase1399_c063_behavior_gate_closure"


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1399 already exists")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    behavior = core.load(BEHAVIOR / "analysis/qwen3_behavior_summary.json")
    final = core.load(BEHAVIOR / "analysis/final.json")
    audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    for phase in (1397, 1398, 1399):
        for path in TESTS.glob(f"phase{phase}_c063*.py"):
            py_compile.compile(str(path), doraise=True)
    hidden_outputs = [path for path in (TESTS / "result").glob("phase1399_c063*/raw/*hidden*")]
    checks = {
        "contract_audited": core.load(CONTRACT / "audit/independent_final_audit.json")["all_checks_passed"],
        "behavior_audited": audit["all_checks_passed"],
        "behavior_failed": not behavior["behavior_qualified"] and not behavior["qualified_families"],
        "authorization_close": final["authorization"] == "close_c063_at_behavior_gate",
        "numeric_healthy": behavior["breadth_checks"]["numeric"] and behavior["breadth_checks"]["finite"],
        "key_failure_visible": max(v["metrics"]["key"]["beta"] for v in behavior["family_results"].values()) < 0.52,
        "hidden_not_accessed": not hidden_outputs,
        "scripts_compile": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed_at_behavior_gate",
        "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "formal_results": {"global_active_accuracy": behavior["global"]["active_accuracy"],
                           "global_status_accuracy": behavior["global"]["status_accuracy"],
                           "qualified_families": behavior["qualified_families"]},
        "claim_boundary": {
            "supported": ["the frozen four-answer response-key behavior object failed despite healthy BF16 execution"],
            "not_supported": ["absence of family identity state", "absence of answer-polarity state", "hidden-state mechanism claim"],
        },
        "authorization": "preregister_c064_fixed_natural_answer_factorial_campaign",
        "next_required_action": "retain family/truth donor orthogonality but use the already qualified fixed yes/no interface; do not repair C063",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not result["all_checks_passed"]:
        raise RuntimeError({k: v for k, v in checks.items() if not v})
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
