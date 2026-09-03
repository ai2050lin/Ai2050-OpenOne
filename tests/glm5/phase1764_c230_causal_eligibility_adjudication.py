#!/usr/bin/env python3
"""C230: type-check causal eligibility without turning failed prediction into patch evidence."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C230"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    transport = core.load(common.OUTS["C227"] / "analysis/lockbox_summary.json")
    composition = core.load(common.OUTS["C229"] / "analysis/lockbox_summary.json")
    parent = core.load(common.OUTS["C229"] / "audit/independent_final_audit.json")
    OUT.mkdir(parents=True)
    eligible = transport["confirmation_gate_passed"] and transport["lockbox_gate_passed"] and composition["campaign_gate_passed"]
    status = "eligible" if eligible else "typed_not_tested"
    protocol = {"phase": 1764, "campaign": "C230", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "causal_eligibility_adjudicated", "required": ["C226 confirmation transport pass", "C227 lockbox transport pass", "C229 at least three family composition pass"], "forbidden_if_ineligible": ["deletion", "rescue", "wrong-relation rescue", "mechanism claim from arbitrary patch"], "producer_sha256": core.sha(Path(__file__)), "authorization": "C231_cross_model_functional_topology_observation_independent_of_causal_gate"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {
        "phase": 1764, "campaign": "C230", "status": status, "causal_eligible": eligible,
        "ledger": {"transport_confirmation": transport["confirmation_gate_passed"], "transport_lockbox": transport["lockbox_gate_passed"], "composition_campaign": composition["campaign_gate_passed"], "composition_families_passed": composition["families_passed"]},
        "tests_executed": [] if not eligible else ["not_reached_in_this_result"],
        "interpretation": "No causal experiment is evidence-neutral when its predicted target field has not passed. This is a type boundary, not a negative result about natural causal mechanisms.",
        "next_authorization": protocol["authorization"],
    }
    core.save(OUT / "analysis/eligibility.json", report)
    checks = {"authorization": parent["all_checks_passed"], "typed_status": status in ("eligible", "typed_not_tested"), "ledger": set(report["ledger"]) == {"transport_confirmation", "transport_lockbox", "composition_campaign", "composition_families_passed"}, "no_illegal_test": eligible or report["tests_executed"] == []}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    final = {"phase": 1764, "campaign": "C230", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

