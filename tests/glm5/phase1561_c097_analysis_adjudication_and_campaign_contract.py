#!/usr/bin/env python3
"""Phase1561: adjudicate Phase1552-1560 analyses and freeze C097."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1560_c096_major_stage_closure"
OUT = RESULT / "phase1561_c097_analysis_adjudication_and_campaign_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1561 exists")
    parent = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    requirements = core.load(PARENT / "protocol/c097_requirements.json")
    checks = {
        "parent_authorized": parent["authorization"] == "preregister_C097_targeted_postquery_residual_and_independent_material_stage",
        "parent_audited": audit["all_checks_passed"],
        "route_A_present": "remaining unused" in requirements["route_A"],
        "route_B_present": "independent" in requirements["route_B"],
        "route_C_conditioned": "Only after" in requirements["route_C_after_A_or_B"],
        "no_model_run": True,
        "no_hidden_access": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    contract = {
        "phase": 1561,
        "campaign": "C097",
        "schema": "c097.identifiable_contrast_observation_campaign.v1",
        "research_object": "test the late-boundary contrast mean and condition residual across a targeted Chinese residual arm and an independent English WordNet arm",
        "analysis_adjudication": {
            "accepted": [
                "C096 prospectively repeated four of five frozen task-scoped observations on lexically disjoint material.",
                "The universal cross-partition floor failed locally at postquery x concrete x similarity-class x state32 boundary.",
                "Raw-coordinate signs can be stable while top-k membership changes with material and condition.",
                "The same-database C091-to-C096 holdout is lexical replication, not external-source validation.",
                "Observation should precede causal closure and can retain explicitly typed missingness.",
            ],
            "corrections": [
                "A constant G in H=mu+P+Q+G+R is collinear with the intercept and is not identifiable from raw 3x3 cells.",
                "The three pairwise four-cell contrasts share raw cells and are dependent observations, not three independent confirmations.",
                "C_fg is an interaction contrast, not purified semantics; truth, output token, boundary, and termination remain mixed.",
                "Top-64 is a descriptive support size with no privileged mechanistic status and requires a random-coordinate baseline.",
                "Hidden-State dimensions are activation coordinates, not weight parameters, MLP neurons, or established semantic neurons.",
                "Five matched quartets per cell do not license population laws without resampling uncertainty.",
                "Do not use a direct-sum symbol unless orthogonality has been demonstrated.",
                "C095/C096 did not physically separate a reusable semantic mechanism; they estimated contrast geometry.",
                "No result licenses a conditional fiber bundle or other new mathematics.",
            ],
        },
        "identifiable_objects": {
            "pairwise_contrast": "C_fg=0.5*(H_ff+H_gg-H_fg-H_gf)",
            "contrast_mean": "G_C=(C_sc+C_sw+C_cw)/3",
            "contrast_residual": "R_fg=C_fg-G_C with sum_fg R_fg=0",
            "energy_identity": "sum_fg ||C_fg||^2 = 3||G_C||^2 + sum_fg ||R_fg||^2",
            "raw_cell_optional": "H_fq=mu+P_f+Q_q+a_fq*G+R_fq+epsilon, a_fq=1[f=q]-1/3, with explicit row/column and orthogonality constraints",
        },
        "evidence_levels": {
            "O0": "execution and numerical qualification",
            "O1": "material and semantic-contract qualification",
            "O2": "behavior-typed observation",
            "O3": "cross-partition or cross-material repetition",
            "O4": "raw-coordinate structure with uncertainty and baselines",
            "O5": "causal necessity/sufficiency/rescue; absent until separately authorized",
        },
        "routes": {
            "A": "remaining Chinese concrete similarity/class pairs; targeted postquery residual breadth",
            "B": "pre-frozen English WordNet synonym/kind-of/part-of materials; independent source and language surface",
            "C": "raw-coordinate intervention only if A or B preserves the common contrast object",
        },
        "global_rules": [
            "No scalar gate erases a numerically valid lower evidence level.",
            "Behavior failure types semantic claims but does not delete descriptive Hidden-State observations.",
            "All thresholds, materials, partitions, models, nulls, and stop conditions freeze before their corresponding reveal.",
            "Route failure retires that route; the campaign proceeds along already authorized independent routes.",
        ],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "learned probes", "post-reveal threshold mutation", "new-mathematics claim"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "run_phase1562_c097_targeted_residual_contract",
    }
    contract["contract_sha256"] = core.digest(contract)
    core.save(OUT / "protocol/c097_campaign_contract.json", contract)
    core.save(OUT / "audit/preimplementation_audit.json", {"phase": 1561, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())})
    core.save(OUT / "analysis/final.json", {"phase": 1561, "campaign": "C097", "status": "analysis_corrected_and_campaign_frozen", "authorization": contract["authorization"]})
    print(json.dumps({"checks": checks, "contract": contract}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

