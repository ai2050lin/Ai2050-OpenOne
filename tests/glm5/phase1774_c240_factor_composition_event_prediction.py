#!/usr/bin/env python3
"""C240: test whether frozen atomic event rules predict interaction events."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common
from phase1772_c238_unseen_surface_event_prediction import signed_jaccard

core = common.core
OUT = common.OUTS["C240"]
C236 = common.OUTS["C236"]
C237 = common.OUTS["C237"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C239"] / "audit/independent_final_audit.json")
    gate = core.load(common.OUTS["C234"] / "protocol/preregistration.json")["composition_gate"]
    checks = {"authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C240"), "gate": gate == {"family_signed_jaccard_min": 0.15, "families_min": 3, "must_beat_atomic_controls_by": 0.02}}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1774, "campaign": "C240", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "atomic_to_interaction_event_prediction_frozen",
        "prediction": "sign(rule_A + rule_B), with exact opposite signs cancelling to zero",
        "truth": "role-aligned interaction event on lockbox and fresh groups",
        "controls": ["factor_a_only", "factor_b_only", "best_wrong_family_interaction", "zero"],
        "gate": gate,
        "caveat": "C239 already summarized all partitions, so this is a frozen-formula validation on previously observed material, not an untouched independent lockbox.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "evaluate_once_then_adjudicate_C241_eligibility",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    rules = np.load(C237 / "analysis/rule_codes.int8.npy", mmap_mode="r")
    events = np.load(C237 / "raw/role_events.int8.npy", mmap_mode="r")
    groups = core.rows(C236 / "protocol/effect_groups.jsonl")
    rows = []
    for group in [row for row in groups if row["partition"] in ("lockbox", "fresh")]:
        family_i = common.FAMILIES.index(group["family"])
        truth = np.asarray(events[group["effect_index"], 2])
        a = np.asarray(rules[family_i, 0])
        b = np.asarray(rules[family_i, 1])
        pred = np.sign(a.astype(np.int16) + b.astype(np.int16)).astype(np.int8)
        wrong = max(signed_jaccard(np.asarray(rules[other_i, 2]), truth) for other_i in range(5) if other_i != family_i)
        rows.append({
            "family": group["family"], "partition": group["partition"], "surface": group["surface"], "unit": group["unit"], "order": group["order"],
            "atomic_composition_signed_jaccard": signed_jaccard(pred, truth),
            "factor_a_only_signed_jaccard": signed_jaccard(a, truth),
            "factor_b_only_signed_jaccard": signed_jaccard(b, truth),
            "best_wrong_family_interaction_signed_jaccard": wrong,
            "zero_signed_jaccard": signed_jaccard(np.zeros_like(pred), truth),
        })
    core.write_rows(OUT / "analysis/prediction_rows.jsonl", rows)
    controls = ("factor_a_only_signed_jaccard", "factor_b_only_signed_jaccard", "best_wrong_family_interaction_signed_jaccard", "zero_signed_jaccard")
    family_results = []
    for family in common.FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        primary = float(np.median([row["atomic_composition_signed_jaccard"] for row in selected]))
        baselines = {name: float(np.median([row[name] for row in selected])) for name in controls}
        margin = primary - max(baselines.values())
        family_results.append({"family": family, "support": len(selected), "atomic_composition_signed_jaccard": primary, "controls": baselines, "all_control_margin": margin, "passed": bool(primary >= 0.15 and margin >= 0.02)})
    passed_families = [row["family"] for row in family_results if row["passed"]]
    campaign_passed = len(passed_families) >= 3
    report = {
        "phase": 1774, "campaign": "C240", "status": "factor_composition_event_prediction_adjudicated",
        "family_results": family_results, "families_passed": passed_families, "campaign_passed": campaign_passed,
        "strict_interpretation": "The test asks whether two frozen atomic sign rules predict a difference-of-differences event. Failure does not refute nonlinear or state-dependent composition.",
        "next_authorization": "C241_joint_C238_C240_causal_eligibility_adjudication_then_continue_C242",
    }
    core.save(OUT / "analysis/summary.json", report)
    audit_checks = {"contract": all(checks.values()), "rows": len(rows) == 50, "families": len(family_results) == 5, "finite": bool(np.isfinite([row[key] for row in rows for key in ("atomic_composition_signed_jaccard",) + controls]).all()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1774, "campaign": "C240", "status": "closed", "checks": audit_checks, "all_checks_passed": all(audit_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_final_audit.json", {"checks": audit_checks, "all_checks_passed": all(audit_checks.values())})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
