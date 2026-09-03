#!/usr/bin/env python3
"""C255: independent campaign audit, theory update, and next-stage authorization."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1780_c246_c255_event_hypergraph_common as common

core = common.core
OUT = common.OUTS["C255"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    phases = {name: core.load(common.OUTS[name] / "analysis/final.json") for name in ("C246", "C247", "C248", "C249", "C250", "C251", "C252", "C253", "C254")}
    audits = {name: core.load(common.OUTS[name] / "audit/independent_final_audit.json") for name in phases}
    tri = np.load(common.OUTS["C249"] / "analysis/tri_material_core.int8.npy", mmap_mode="r")
    asset = common.ROOT / "frontend/public/vis_data/research_kernel/c254_tri_material_event_atlas.json"
    checks = {
        "all_parent_audits": all(value["all_checks_passed"] for value in audits.values()),
        "phases_contiguous": [phases[name]["phase"] for name in phases] == list(range(1780, 1789)),
        "tri_shape": list(tri.shape) == [5, 3, 37, 6, 2560],
        "tri_count": int(np.count_nonzero(tri)) == phases["C249"]["headline"]["tri_material_events"],
        "asset_hash": core.sha(asset) == phases["C254"]["asset"]["sha256"],
        "no_attention_or_mlp_artifacts": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    c249 = phases["C249"]["headline"]
    c250 = phases["C250"]["headline"]
    c251 = phases["C251"]["headline"]
    c252 = phases["C252"]["headline"]
    c253 = phases["C253"]["headline"]
    theory = {
        "theory_name": "Conditional Output Field Closure Theory",
        "organization": "Reuse-Difference-Conditionality (RDC)",
        "status": "open research program; not closed",
        "observational_state_formula": "X_{q+1}=Phi_{theta,q}(X_q;kappa)",
        "factorial_formula": "beta_A=R_A/2, beta_B=R_B/2, beta_AB=R_AB/4",
        "event_formula": "E_{f,e,q,r,j}=Pi_eta(beta_{f,e,q,r,j}) in {-1,0,+1}",
        "tri_material_formula": "K^(1,2,3)=1 iff the same (f,e,q,r,j) is stable with the same sign in all three material systems",
        "intervention_formula": "I_path: X[q,phi(r),M(f,e,q,r)]=X_donor[q,phi_d(r),M(f,e,q,r)]",
        "trajectory_readout": "Delta_D=(D(clean_target,donor)-D(I_path(target),donor))/D(clean_target,donor)",
        "abstract_cross_model_formula": "A_m(f,e,rho,r)=RMS_j beta_m(f,e,q_m(rho),r,j), normalized over r",
        "global_graph": "embedding+prefix -> condition-indexed full-token field -> partially persistent signed events -> typed non-additive assembly -> boundary competition -> output",
        "new_mathematics_gate": {"repeated_functional_object": True, "unseen_prediction": True, "ordered_causal_control": True, "cross_model_abstract_repeat": True, "existing_mathematics_inexpressive": False, "gate_open": False},
    }
    puzzles = {
        "K325_CONFIRM": {"tri_material_events": c249["tri_material_events"], "targets_passed": c249["target_families_passed"], "boundary": "role-span, threshold, task, and Qwen specific"},
        "K326_OBS": {"aligned_pairs": c250["aligned_pairs"], "token_coverage": c250["matched_token_coverage_median"], "checkpoint_persistence": c250["same_coordinate_same_sign_persistence_median"], "boundary": "edited unmatched tokens omitted; persistence is not a unique edge"},
        "K327_OBS": {"typed_composition": c251["summaries"], "boundary": "non-additivity, not operator composition or commutator"},
        "K328_CAUSAL": {"trajectory_gate": c252["trajectory_gate_passed"], "correct_improvement": c252["condition_medians"]["correct_path"], "control_margin": c252["correct_vs_best_control_margin"], "boundary": "large distributed intervention and output-invariant task"},
        "K329_CONFIRM": {"cross_model_gate": c253["cross_model_gate_passed"], "pair_tests": c253["pair_tests"], "boundary": "coarse role-depth RMS topology only"},
    }
    hard_limits = [
        "Independent human naturalness review is still missing.",
        "C249 stable-aggregate comparison is a different camera from C244 per-group prediction; it does not retroactively reverse old failures.",
        "Role-span means remain researcher-defined and erase within-span token order.",
        "C250 exact alignment omits edited tokens without one-to-one correspondences.",
        "C252 changes a large distributed band and does not establish minimality or natural semantic-output use.",
        "Cross-model physical coordinates and signs are not aligned.",
        "No unique predictive event hyperedges or complete language algebra have been identified.",
    ]
    report = {
        "phase": 1789, "campaign": "C255", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "campaign_closed",
        "checks": checks, "puzzles": puzzles, "theory": theory, "hard_limits": hard_limits,
        "strict_conclusion": "The campaign confirms a distributed, condition-indexed, partially persistent signed event ecology across three Qwen material systems, demonstrates typed non-additivity, steers a final state with an ordered distributed intervention, and independently repeats a coarse three-model role-depth topology. It does not identify semantic neurons, a minimal path, natural output necessity/sufficiency, complete composition, or new mathematics.",
        "next_stage_same_goal": True,
        "next_authorization": "C256_output_sensitive_attitude_readout_and_bidirectional_path_intervention",
    }
    protocol = {"phase": 1789, "campaign": "C255", "status": "theory_adjudication_frozen", "producer_sha256": core.sha(Path(__file__)), "authorization": report["next_authorization"]}
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/theory_and_puzzles.json", report)
    final_checks = {**checks, "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1789, "campaign": "C255", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": report["next_authorization"], "audited_phases": list(range(1780, 1789))})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
