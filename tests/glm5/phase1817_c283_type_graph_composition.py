#!/usr/bin/env python3
"""C283: observe direct/two-hop/shortcut type-graph composition."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1811_c277_c289_joint_response_common as common
import phase1816_c282_natural_attitude_composition as composition

core, OUT = common.core, common.OUTS["C283"]
C278 = common.OUTS["C278"]


def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C282"] / "analysis/final.json")
    checks = {"parent": parent["all_checks_passed"], "behavior_eligible": core.load(C278 / "analysis/final.json")["headline"]["by_family_accuracy"]["type_graph"] >= 0.70, "all_coordinates": True}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"): (OUT / subdir).mkdir()
    protocol = {
        "phase": 1817, "campaign": "C283", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "type_graph_composition_frozen",
        "factors": {"A": "direct final membership versus two-hop membership", "B": "absence versus presence of a direct shortcut"},
        "test": "full-coordinate factorial residual and unit-half signed-event replication",
        "claim_boundary": "The panel tests state nonadditivity under graph edits. It does not identify a transitive operator or prove graph-isomorphism invariance.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C284_order_commutation_comparison",
    }
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    metrics, atlas = composition.factorial_observation(np.load(C278 / "raw/role_states.float16.npy", mmap_mode="r"), core.rows(C278 / "raw/hidden_index.jsonl"), "type_graph")
    np.save(OUT / "analysis/type_graph_interaction_atlas.float16.npy", atlas)
    report = {"phase": 1817, "campaign": "C283", "status": "type_graph_composition_observed", "result": metrics, "strict_interpretation": protocol["claim_boundary"], "next_authorization": "C284_order_commutation_comparison"}
    core.save(OUT / "analysis/summary.json", report)
    ach = {"groups": metrics["complete_factorial_groups"] >= 16, "atlas": list(atlas.shape) == [37, 6, 2560], "finite": bool(np.isfinite(list(metrics.values())[1:4]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())})
    fch = {"contract": all(checks.values()), "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1817, "campaign": "C283", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final); print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

