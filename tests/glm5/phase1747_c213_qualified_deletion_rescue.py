#!/usr/bin/env python3
"""C213: adjudicate whether a prospectively qualified object exists for deletion/rescue."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C213
PHASE, CAMPAIGN = 1747, "C213"


def run() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.C212 / "audit/independent_final_audit.json")
    c208 = core.load(common.C208 / "analysis/orthogonal_prediction.json")
    c212 = core.load(common.C212 / "analysis/factorial_composition.json")
    checks = {"authorization": parent["all_checks_passed"], "c208_typed": isinstance(c208["predictive_gate_passed"], bool), "c212_typed": isinstance(c212["factorial_composition_gate_passed"], bool)}
    if not all(checks.values()):
        raise RuntimeError(checks)
    qualified = c208["predictive_gate_passed"] or c212["factorial_composition_gate_passed"]
    status = "causal_object_authorized" if qualified else "typed_not_tested"
    reason = None if qualified else "C208 did not qualify an unseen-direction response operator and C212 did not qualify both frozen factorial arms. The path arm alone passed descriptively, but no prospectively frozen coordinate/path object exists to delete; selecting one now would be post hoc."
    OUT.mkdir(parents=True)
    protocol = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": status, "qualification_rule": "C208 complete response operator OR C212 complete two-arm factorial object", "inputs": {"C208_predictive_gate": c208["predictive_gate_passed"], "C212_complete_factorial_gate": c212["factorial_composition_gate_passed"], "C212_path_arm_only": c212["arm_summaries"]["path_factorial"]["fresh"]["passed"]}, "qualified": qualified, "reason": reason, "causal_tests_run": False, "claim_boundary": "typed non-test is not evidence against causal structure", "producer_sha256": core.sha(Path(__file__)), "authorization": "C214_cross_model_functional_isomorphism_continues_independently"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": protocol, "next_authorization": protocol["authorization"]})
    core.save(OUT / "audit/internal_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(protocol, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()

