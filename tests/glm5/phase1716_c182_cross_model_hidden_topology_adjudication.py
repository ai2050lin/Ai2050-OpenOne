#!/usr/bin/env python3
"""C182: typed-not-tested adjudication for cross-model HiddenState topology."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1716_c182_cross_model_hidden_topology_adjudication"
C181 = TESTS / "result/phase1715_c181_cross_model_functional_eligibility"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C181 / "audit/independent_final_audit.json")
    summary = core.load(C181 / "analysis/summary.json")
    checks = {
        "parent_closed": parent["all_checks_passed"],
        "authorized_branch": "C182_typed_not_tested" in parent["authorization"],
        "fewer_than_four_common_families": len(summary["common_two_model_families"]) < 4,
        "hidden_not_eligible": not summary["cross_model_hidden_eligible"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1716,
        "campaign": "C182",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "typed_not_tested_contract_frozen",
        "object": "cross-model relative HiddenState role topology",
        "required_gate": "at least four relation families behaviorally qualified on at least two models",
        "observed_common_families": summary["common_two_model_families"],
        "forbidden_inference": [
            "cross-model topology absent",
            "GLM4 or DeepSeek lacks relation encoding",
            "same physical coordinate index should align across models",
        ],
        "producer_sha256": core.sha(Path(__file__)),
    }
    final = {
        "phase": 1716,
        "campaign": "C182",
        "status": "closed_typed_not_tested",
        "tests_run": 0,
        "hidden_states_loaded": False,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "conclusion": "cross-model HiddenState topology was not tested because no family qualified on two models",
        "next_authorization": "run_C183_qwen_response_field_synthesis_and_parameter_heatmap",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
