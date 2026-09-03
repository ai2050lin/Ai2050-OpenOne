#!/usr/bin/env python3
"""C211: consolidate five flagship language routes without inventing new mechanism claims."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C211
PHASE, CAMPAIGN = 1745, "C211"
ROUTES = {
    "attitude_event_binding": "attitude_event",
    "type_graph_chain": "type_chain",
    "contrast_noncommutativity": "contrast",
    "translation_commutation": "translation",
    "comparison_relations": "comparison",
}


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.C210 / "audit/independent_final_audit.json")
    checks = {"authorization": parent["all_checks_passed"], "routes": len(ROUTES) == 5, "distinct_programs": len(set(ROUTES.values())) == 5}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "flagship_route_ledger_frozen",
        "routes": ROUTES,
        "inputs": ["C198 behavior", "C206 dose regime", "C208 unseen-direction response", "C210 natural paraphrase trajectory"],
        "policy": "this is an evidence ledger after upstream reveals; it may rank observational candidates but cannot retroactively create a confirmatory gate",
        "candidate_thresholds": {"behavior_partition_min": 0.65, "natural_edit_nrmse_max": 0.75, "natural_edit_sign_min": 0.75, "unseen_direction_nrmse_max": 0.80, "unseen_direction_sign_min": 0.75},
        "claim_boundary": "five controlled-English route slices, not the full natural-language family or a language algebra",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "freeze_and_run_C212_new_factorial_composition_material",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "routes": ROUTES}, indent=2))


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    thresholds = protocol["candidate_thresholds"]
    c198 = core.load(common.C198 / "analysis/natural_trajectory.json")["behavior"]
    c208 = core.load(common.C208 / "analysis/orthogonal_prediction.json")
    c210 = core.load(common.C210 / "analysis/natural_edit_trajectory.json")
    c208_by = {(row["program"], row["unit"]): row for row in c208["anchor_rows"]}
    rows = []
    for route, program in ROUTES.items():
        behavior = c198["by_program_partition"][program]
        natural = c210["per_program_fresh"][program]
        direction = c208_by[(program, 6)]["odd"]
        checks = {
            "behavior": min(behavior.values()) >= thresholds["behavior_partition_min"],
            "natural_edit": natural["nrmse"] <= thresholds["natural_edit_nrmse_max"] and natural["weighted_sign"] >= thresholds["natural_edit_sign_min"],
            "unseen_direction": direction["nrmse"] <= thresholds["unseen_direction_nrmse_max"] and direction["weighted_sign"] >= thresholds["unseen_direction_sign_min"],
        }
        rows.append({"route": route, "program": program, "behavior": behavior, "natural_edit": natural, "unseen_direction": direction, "candidate_checks": checks, "all_candidate_checks": all(checks.values())})
    candidates = [row["route"] for row in rows if row["all_candidate_checks"]]
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "five_flagship_routes_consolidated", "route_rows": rows, "observational_candidates": candidates, "all_five_observational_candidates": len(candidates) == 5, "interpretation": "All labels are observational candidate labels assembled from already revealed evidence. They authorize new factorial material, not causal or algebraic claims.", "next_authorization": "C212_voice_order_and_path_factorial_composition"}
    core.save(OUT / "analysis/flagship_route_ledger.json", report)
    checks = {"five_rows": len(rows) == 5, "accounting": set(candidates) <= set(ROUTES), "finite": bool(np.isfinite([value for row in rows for value in (*row["behavior"].values(), row["natural_edit"]["nrmse"], row["natural_edit"]["weighted_sign"], row["unseen_direction"]["nrmse"], row["unseen_direction"]["weighted_sign"])]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/flagship_route_ledger.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()

