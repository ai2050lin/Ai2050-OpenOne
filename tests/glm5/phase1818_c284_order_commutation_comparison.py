#!/usr/bin/env python3
"""C284: observe contrast order, translation shortcut, and comparison direction."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1811_c277_c289_joint_response_common as common
import phase1816_c282_natural_attitude_composition as composition

core, OUT = common.core, common.OUTS["C284"]
C278 = common.OUTS["C278"]
FAMILIES = ("contrast", "translation", "comparison")


def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C283"] / "analysis/final.json")
    checks = {"parent": parent["all_checks_passed"], "families": True, "all_coordinates": True, "no_operator_name_overclaim": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"): (OUT / subdir).mkdir()
    protocol = {
        "phase": 1818, "campaign": "C284", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "order_panel_frozen",
        "panels": {
            "contrast": "connective form x clause order",
            "translation": "direct/two-hop map x direct shortcut",
            "comparison": "comparison dimension x surface direction",
        },
        "test": "full-coordinate factorial residual and cross-lexicon-half signed event agreement",
        "claim_boundary": "These panels are finite graph edits. A residual does not by itself prove noncommutativity, a commuting square, or a shared comparison operator.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C285_eligibility_adjudication",
    }
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    states = np.load(C278 / "raw/role_states.float16.npy", mmap_mode="r"); index = core.rows(C278 / "raw/hidden_index.jsonl")
    rows = []
    for family in FAMILIES:
        metrics, atlas = composition.factorial_observation(states, index, family); rows.append(metrics); np.save(OUT / f"analysis/{family}_interaction_atlas.float16.npy", atlas)
        print(f"[C284] {family}: active={metrics['interaction_active_fraction']:.4f}, ratio={metrics['interaction_to_main_l1_ratio']:.4f}, halfJ={metrics['cross_lexicon_half_signed_jaccard']:.4f}", flush=True)
    core.write_rows(OUT / "analysis/family_results.jsonl", rows)
    report = {"phase": 1818, "campaign": "C284", "status": "order_panels_observed", "families": rows, "strict_interpretation": protocol["claim_boundary"], "next_authorization": "C285_eligibility_adjudication"}
    core.save(OUT / "analysis/summary.json", report)
    ach = {"families": len(rows) == 3, "groups": all(r["complete_factorial_groups"] >= 16 for r in rows), "finite": bool(np.isfinite([r["interaction_to_main_l1_ratio"] for r in rows]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())})
    fch = {"contract": all(checks.values()), "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1818, "campaign": "C284", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

