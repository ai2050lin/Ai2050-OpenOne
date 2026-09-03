#!/usr/bin/env python3
"""C239: observation-first account of all five flagship event routes."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C239"]
C235 = common.OUTS["C235"]
C236 = common.OUTS["C236"]
C237 = common.OUTS["C237"]
C238 = common.OUTS["C238"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C238 / "audit/independent_final_audit.json")
    checks = {"authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C239"), "five_routes": len(common.FAMILIES) == 5}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1773, "campaign": "C239", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "five_flagship_observation_frozen",
        "routes": {
            "attitude_event": {"factor_a": "attitude polarity", "factor_b": "active/passive event rendering"},
            "type_graph": {"factor_a": "direct/two-hop path", "factor_b": "direct shortcut"},
            "contrast": {"factor_a": "but/although connective", "factor_b": "clause order"},
            "translation": {"factor_a": "direct/two-hop code path", "factor_b": "direct gloss shortcut"},
            "comparison": {"factor_a": "height/weight dimension", "factor_b": "direct/inverse syntax"},
        },
        "metrics": ["full-token activity", "role-aligned activity", "embedding share", "first formation", "interaction burden", "candidate-order agreement", "unseen prediction"],
        "no_new_model_run": True, "no_route_stop": True,
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "observe_all_five_routes_then_C240_frozen_composition_test",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    full_events = np.load(C236 / "raw/events.int8.npy", mmap_mode="r")
    role_events = np.load(C237 / "raw/role_events.int8.npy", mmap_mode="r")
    rules = np.load(C237 / "analysis/rule_codes.int8.npy", mmap_mode="r")
    groups = core.rows(C236 / "protocol/effect_groups.jsonl")
    order_rows = core.rows(C236 / "analysis/candidate_order_agreement.jsonl")
    behavior = core.load(C235 / "analysis/behavior_capture_summary.json")
    unseen = {row["family"]: row for row in core.load(C238 / "analysis/summary.json")["family_final_results"]}
    route_rows = []
    for family_i, family in enumerate(common.FAMILIES):
        selected = [row["effect_index"] for row in groups if row["family"] == family]
        route_effects = []
        for effect_i, effect in enumerate(common.EFFECTS):
            full = np.asarray(full_events[selected, effect_i])
            role = np.asarray(role_events[selected, effect_i])
            rule = np.asarray(rules[family_i, effect_i])
            density_q = np.mean(role != 0, axis=(0, 2, 3))
            rule_q = np.mean(rule != 0, axis=(1, 2))
            route_effects.append({
                "effect": effect,
                "full_token_active_density": float(np.mean(full != 0)),
                "role_active_density": float(np.mean(role != 0)),
                "embedding_role_active_density": float(density_q[0]),
                "final_role_active_density": float(density_q[-1]),
                "stable_rule_events": int(np.count_nonzero(rule)),
                "stable_rule_embedding_share": float(np.count_nonzero(rule[0]) / max(np.count_nonzero(rule), 1)),
                "first_checkpoint_with_rule": int(np.flatnonzero(rule_q)[0]) if np.any(rule_q) else -1,
                "checkpoint_of_max_rule_density": int(np.argmax(rule_q)),
            })
        ab = route_effects[2]["stable_rule_events"]
        atomic = route_effects[0]["stable_rule_events"] + route_effects[1]["stable_rule_events"]
        order = [row["signed_agreement_on_union"] for row in order_rows if row["family"] == family]
        route_rows.append({
            "family": family,
            "behavior_accuracy": behavior["by_family_accuracy"][family],
            "effects": route_effects,
            "interaction_to_atomic_event_ratio": float(ab / max(atomic, 1)),
            "candidate_order_signed_agreement_median": float(np.median(order)),
            "unseen_signed_jaccard": unseen[family]["correct_signed_jaccard"],
            "unseen_control_margin": unseen[family]["all_control_margin"],
            "unseen_passed": unseen[family]["passed"],
        })
    core.save(OUT / "analysis/route_profiles.json", route_rows)
    overlap = []
    for left_i in range(5):
        for right_i in range(left_i + 1, 5):
            left = np.asarray(rules[left_i])
            right = np.asarray(rules[right_i])
            union = (left != 0) | (right != 0)
            overlap.append({"families": [common.FAMILIES[left_i], common.FAMILIES[right_i]], "signed_rule_jaccard": float(np.mean(left[union] == right[union])) if union.any() else 1.0, "active_overlap_fraction_of_union": float(np.mean((left[union] != 0) & (right[union] != 0))) if union.any() else 0.0})
    core.write_rows(OUT / "analysis/cross_family_rule_overlap.jsonl", overlap)
    passed_profiles = [row for row in route_rows if row["unseen_passed"]]
    failed_profiles = [row for row in route_rows if not row["unseen_passed"]]
    report = {
        "phase": 1773, "campaign": "C239", "status": "five_flagship_routes_observed",
        "routes": route_rows, "cross_family_overlaps": overlap,
        "passed_routes": [row["family"] for row in passed_profiles], "failed_routes": [row["family"] for row in failed_profiles],
        "median_cross_family_signed_rule_jaccard": float(np.median([row["signed_rule_jaccard"] for row in overlap])),
        "strict_interpretation": "The profiles describe where repeatable events occur. They do not establish a common operator or a unique coordinate gear.",
        "next_authorization": "C240_atomic_to_interaction_event_prediction",
    }
    core.save(OUT / "analysis/summary.json", report)
    audit_checks = {"contract": all(checks.values()), "routes": len(route_rows) == 5, "effects": all(len(row["effects"]) == 3 for row in route_rows), "overlap": len(overlap) == 10, "finite": bool(np.isfinite([row["median_cross_family_signed_rule_jaccard"] for row in [report]]).all()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1773, "campaign": "C239", "status": "closed", "checks": audit_checks, "all_checks_passed": all(audit_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_final_audit.json", {"checks": audit_checks, "all_checks_passed": all(audit_checks.values())})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
