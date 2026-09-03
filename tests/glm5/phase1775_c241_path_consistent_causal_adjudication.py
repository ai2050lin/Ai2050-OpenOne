#!/usr/bin/env python3
"""C241: adjudicate eligibility for path-consistent causal intervention."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C241"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C240"] / "audit/independent_final_audit.json")
    unseen = core.load(common.OUTS["C238"] / "analysis/summary.json")
    composition = core.load(common.OUTS["C240"] / "analysis/summary.json")
    checks = {"authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C241"), "unseen_loaded": unseen["phase"] == 1772, "composition_loaded": composition["phase"] == 1774}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    eligible = unseen["campaign_passed"] and composition["campaign_passed"]
    protocol = {"phase": 1775, "campaign": "C241", "created_at_utc": datetime.now(timezone.utc).isoformat(), "eligibility_formula": "C238 campaign_passed AND C240 campaign_passed", "required_intervention_chain": ["selective deletion", "correct ordered replay", "wrong relation/order/surface replay", "downstream trajectory recovery", "non-target preservation"], "producer_sha256": core.sha(Path(__file__))}
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {
        "phase": 1775, "campaign": "C241", "status": "typed_not_tested" if not eligible else "eligible_not_implemented",
        "C238_unseen_event_passed": unseen["campaign_passed"], "C238_families_passed": unseen["families_passed"],
        "C240_composition_passed": composition["campaign_passed"], "C240_families_passed": composition["families_passed"],
        "causal_eligible": eligible, "model_loaded": False, "patch_executed": False,
        "strict_conclusion": "The frozen observational chain did not qualify a path-consistent intervention. No arbitrary coordinate patch was run, so causality is untested rather than disproved.",
        "next_authorization": "C242_cross_model_abstract_event_graph_continues_despite_local_causal_ineligibility",
    }
    core.save(OUT / "analysis/summary.json", report)
    audit_checks = {"contract": all(checks.values()), "typed_state": report["status"] == "typed_not_tested", "no_model": not report["model_loaded"], "no_patch": not report["patch_executed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1775, "campaign": "C241", "status": "closed", "checks": audit_checks, "all_checks_passed": all(audit_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_final_audit.json", {"checks": audit_checks, "all_checks_passed": all(audit_checks.values())})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
