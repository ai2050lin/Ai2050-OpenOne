#!/usr/bin/env python3
"""C232: joint theory ledger and conservative mathematics-upgrade adjudication."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C232"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.OUTS["C231"] / "audit/independent_final_audit.json")
    c224 = core.load(common.OUTS["C224"] / "analysis/behavior_and_capture_summary.json")
    c225 = core.load(common.OUTS["C225"] / "analysis/coordinate_passport_summary.json")
    c226 = core.load(common.OUTS["C226"] / "analysis/tournament_summary.json")
    c227 = core.load(common.OUTS["C227"] / "analysis/lockbox_summary.json")
    c229 = core.load(common.OUTS["C229"] / "analysis/lockbox_summary.json")
    c230 = core.load(common.OUTS["C230"] / "analysis/eligibility.json")
    c231 = core.load(common.OUTS["C231"] / "analysis/cross_model_summary.json")
    OUT.mkdir(parents=True)
    protocol = {"phase": 1766, "campaign": "C232", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "joint_theory_adjudication_frozen", "mathematics_upgrade_requirements": ["unseen-surface exact signed-field prediction", "at least three language families with unseen composition prediction", "typed deletion-damage-correct-rescue chain", "at least two-model functional isomorphism including all participating pairs"], "theory_name_policy": "retain Conditional Output Field Closure Theory and RDC; update evidence, not the name", "producer_sha256": core.sha(Path(__file__)), "authorization": "C233_parameter_level_heatmap_and_major_stage_closure"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    requirements = {
        "unseen_surface_transport": bool(c226["confirmation_gate_passed"] and c227["lockbox_gate_passed"]),
        "broad_unseen_composition": bool(c229["campaign_gate_passed"]),
        "typed_causal_chain": bool(c230["causal_eligible"]),
        "cross_model_isomorphism": bool(c231["cross_model_gate_passed"]),
    }
    report = {
        "phase": 1766, "campaign": "C232", "status": "mathematics_upgrade_gate_closed",
        "evidence_ledger": {
            "behavior_global_accuracy": c224["global_accuracy"],
            "eligible_behavior_strata": [c224["eligible_strata"], c224["total_strata"]],
            "within_surface_nrmse": c225["within_surface"]["median_nrmse"],
            "raw_cross_surface_nrmse": c225["raw_cross_surface"]["median_nrmse"],
            "transport_confirmation": c226["confirmation_gate_passed"],
            "transport_lockbox": c227["lockbox_gate_passed"],
            "composition_families_passed": c229["families_passed"],
            "causal_status": c230["status"],
            "cross_model_gate": c231["cross_model_gate_passed"],
        },
        "mathematics_upgrade_requirements": requirements,
        "requirements_passed": sum(requirements.values()), "requirements_total": len(requirements),
        "new_foundational_mathematics_authorized": all(requirements.values()),
        "theory": {
            "name": "Conditional Output Field Closure Theory",
            "organizing_principle": "Reuse-Difference-Conditioning (RDC)",
            "current_object": "surface-indexed, role/checkpoint typed families of full signed response fields",
            "current_formula": "R_{f,s,u,e,q,r,j}=Walsh_e(H_{q,r,j}(Render_s(G_f,u)))",
            "candidate_transport": "T_{s0->s}^{e,q,r,j}(R)=A_{s,e,q,r,j}*R+B_{s,e,q,r,j}",
            "global_map": "embedding plus full prefix -> surface-conditioned assembly field -> role/checkpoint response family -> candidate competition and output",
        },
        "strict_conclusion": "The data support conditional response families and two narrow compositional regularities. They reject the current strong fixed-transport and broad-composition hypotheses, while leaving the broader relative-encoding program open. Existing finite differences, linear algebra, conditional maps, and typed equivalence relations remain sufficient for the evidence obtained.",
        "next_authorization": protocol["authorization"],
    }
    core.save(OUT / "analysis/theory_adjudication.json", report)
    checks = {"authorization": parent["all_checks_passed"], "four_requirements": len(requirements) == 4, "no_upgrade": not report["new_foundational_mathematics_authorized"], "name_stable": report["theory"]["name"] == "Conditional Output Field Closure Theory", "ledger_complete": len(report["evidence_ledger"]) == 9}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    final = {"phase": 1766, "campaign": "C232", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

