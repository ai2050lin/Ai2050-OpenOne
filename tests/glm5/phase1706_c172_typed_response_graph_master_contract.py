#!/usr/bin/env python3
"""C172: freeze the typed full-coordinate response-graph campaign."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1706_c172_typed_response_graph_master_contract"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1706, "C172"
PARENTS = {
    "C159": RESULT / "phase1693_c159_natural_isomorphic_dual_graph_atlas",
    "C162": RESULT / "phase1696_c162_linguistic_program_field",
    "C164": RESULT / "phase1698_c164_three_model_free_interface",
    "C168": RESULT / "phase1702_c168_fresh_relation_residual_confirmation",
    "C170": RESULT / "phase1704_c170_role_checkpoint_relation_transport_atlas",
    "C171": RESULT / "phase1705_c171_role_checkpoint_coordinate_heatmap",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    audits = {name: core.load(path / "audit/independent_final_audit.json") for name, path in PARENTS.items()}
    c168 = core.load(PARENTS["C168"] / "analysis/final.json")
    c170 = core.load(PARENTS["C170"] / "analysis/final.json")
    checks = {
        "parent_audits": all(a["all_checks_passed"] for a in audits.values()),
        "fresh_relation_replication": bool(c168["headline"]["passed"]),
        "role_heterogeneity_observed": c170["headline"]["label_counts"] == {"stable": 3, "partial": 3, "absent": 3},
        "next_phase_is_contiguous": PHASE == 1706,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "typed_response_graph_campaign_frozen",
        "evidence_audit": {
            "retained": [
                "C168 prospectively replicated a q24 relation-role local response signature on fresh lexical material.",
                "C170 showed that the same relation-selected coordinate alliance has heterogeneous responses across source roles.",
                "Full physical coordinates and typed token roles are more informative objects than one global vector cosine.",
            ],
            "corrected_overclaims": [
                "C170 ranks one fixed alliance; it does not estimate operator norms or total role capacity.",
                "Finite perturbation responses are local effective response interfaces, not unique natural causal circuits.",
                "Wrong-relation margins from four-way centered residuals are partly structural; signed fresh prediction and source permutation are primary controls.",
                "The existing evidence does not establish that new mathematics is necessary.",
                "Coordinate-axis perturbations are physically orthogonal but not guaranteed to be semantically orthogonal.",
            ],
        },
        "frozen_object": "typed local response graph over checkpoint, semantic token role, physical source coordinate, target role, and physical target coordinate",
        "response": "R[q,r,i -> q+1,s,j | x] = (H_plus-H_minus)/(2 epsilon)",
        "campaign_arms": [
            {"campaign": "C173", "objective": "discover primary/query role-specific full-coordinate source alliances and prospectively validate them"},
            {"campaign": "C174", "objective": "compress validated response fields into signed target-edge sets by held-out reconstruction error"},
            {"campaign": "C175", "objective": "measure pairwise non-additive hyperedges for locked role-specific coordinates"},
            {"campaign": "C176", "objective": "test whether the typed-response object organizes many language-program factors in existing broad data"},
            {"campaign": "C177", "objective": "test natural knowledge-ecology external validity on qualified behavior"},
            {"campaign": "C178", "objective": "audit cross-model functional-interface eligibility without coordinate-number matching"},
            {"campaign": "C179", "objective": "synthesize results and export parameter-level heatmap data"},
        ],
        "partitions": {"discovery": "may rank coordinates and edges", "confirmation": "locked evaluation", "fresh": "locked lexical evaluation"},
        "models": {"primary": "Qwen3-4B BF16 CUDA", "cross_model": ["GLM4", "DeepSeek-7B"], "execution": "strictly sequential"},
        "materials": {
            "core": "C159 balanced natural-lexical/isomorphic-nonce directed graphs",
            "breadth": "C162 eight-factor compositional language program",
            "natural": "new explicit natural knowledge-ecology panel",
        },
        "material_audit": {
            "semantic_uniqueness": "exact directed graph truth and answer key required",
            "surface_balance": "natural and isomorphic nonce panels are balanced by relation and partition",
            "naturalness_boundary": "controlled prompts are readable but are not evidence for unrestricted natural language",
        },
        "primary_metrics": {
            "signed_nrmse": "||A-P||_2 / max(||A||_2, tiny)",
            "signed_explained_energy": "1 - ||A-P||_2^2/max(||A||_2^2,tiny)",
            "active_coordinate_sign_agreement": "sign agreement on discovery-frozen active target edges",
            "rollout_error": "held-out error after composing consecutive local response maps",
            "retained_energy": "energy preserved by an edge set selected without holdout access",
        },
        "secondary_metrics": ["cosine", "relation-centered margin", "median norm"],
        "zero_models": ["source-coordinate permutation", "wrong-role alliance", "wrong-relation template", "sign reversal", "no perturbation", "late-only output convergence"],
        "branch_policy": "a failed arm limits only that arm; all preauthorized observational arms continue",
        "causal_policy": "natural necessity/rescue is authorized only for a response object that predicts fresh HiddenState fields; failure does not terminate observation arms",
        "cross_model_policy": "compare relative layer depth, role topology, and response composition; never compare physical coordinate numbers",
        "forbidden": ["attention attribution", "MLP attribution", "weights", "PCA as primary evidence", "claiming unique circuit", "claiming new mathematics from fit alone"],
        "next_authorization": "run_C173_role_specific_full_coordinate_campaign",
        "source_hashes": {name: core.sha(path / "analysis/final.json") for name, path in PARENTS.items()},
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "closed",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "headline": {"object": protocol["frozen_object"], "arms": len(protocol["campaign_arms"]), "overclaims_corrected": len(protocol["evidence_audit"]["corrected_overclaims"])},
        "next_authorization": protocol["next_authorization"],
    }
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_final_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
