#!/usr/bin/env python3
"""Phase1518: close C088 and issue the bounded next-campaign authorization."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1518_c088_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE_DIRS = {
    1512: "phase1512_c088_cross_root_semantic_code_factorial_contract",
    1513: "phase1513_c088_unified_forward_capture",
    1514: "phase1514_c088_factorial_field_atlas",
    1515: "phase1515_c088_discovery_observation_freeze",
    1516: "phase1516_c088_holdout_and_fresh_reveal",
    1517: "phase1517_c088_full_dimensional_diagnostics",
}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1518 exists")
    ledger = []
    for phase, directory in PHASE_DIRS.items():
        root = RESULT / directory
        final = core.load(root / "analysis/final.json")
        audit = core.load(root / "audit/independent_final_audit.json")
        ledger.append({
            "phase": phase,
            "directory": directory,
            "status": final["status"],
            "authorization": final.get("authorization"),
            "audit_passed": audit["all_checks_passed"],
            "audit_score": f"{audit['passed']}/{audit['total']}",
            "final_sha256": core.sha(root / "analysis/final.json"),
            "audit_sha256": core.sha(root / "audit/independent_final_audit.json"),
        })
    reveal = core.load(RESULT / PHASE_DIRS[1516] / "analysis/final.json")
    diagnostics = core.load(RESULT / PHASE_DIRS[1517] / "analysis/full_dimensional_diagnostic_summary.json")
    capture = core.load(RESULT / PHASE_DIRS[1513] / "analysis/unified_behavior_and_capture_summary.json")
    all_audits = all(row["audit_passed"] for row in ledger)
    verdict = {
        "contract_and_material_gate": True,
        "single_pass_execution_identity": True,
        "factorial_algebra_and_causal_zeros": True,
        "paired_holdout_structure_presence": reveal["verdict"]["structure_presence_paired_holdouts"],
        "paired_holdout_effect_size_equality": reveal["verdict"]["paired_effect_size_equality"],
        "fresh_root_directional_presence": reveal["verdict"]["fresh_root_directional_presence"],
        "natural_query_directional_alignment": reveal["verdict"]["c087_natural_alignment_directional"],
        "behavioral_code_compliance": False,
        "localized_or_necessary_mechanism": False,
        "cross_model_invariant": False,
    }
    core_piece = {
        "id": "K265",
        "evidence": "E3-HS-descriptive",
        "title": "cross-root code-marginalized late semantic response field",
        "statement": (
            "In Qwen3 under C088, the state35 boundary semantic main effect repeats across discovery, "
            "confirmation, lockbox, and eight fresh lexical roots; physical coordinates are highly stable, "
            "but the field is descriptive and entangled with stronger answer-code and interaction responses."
        ),
        "does_not_imply": [
            "a pure or universal semantic vector",
            "natural compliance with a reversed answer code",
            "a localized, necessary, or sufficient semantic circuit",
            "cross-model invariance",
        ],
    }
    next_program = {
        "authorization": "preregister_c089_natural_relation_full_state_observation_atlas",
        "objective": "replace answer-code manipulation with multiple naturally answerable relation tasks and map repeated full-state laws before intervention",
        "required_branches": [
            "large natural-behavior qualification with lexical-root holdouts",
            "single-pass all-hidden-state capture for qualified Qwen3 cases",
            "full-dimensional layer-role observation atlas without PCA, probes, attention, or MLP analysis",
            "discovery of repeated formation, transport, and coordinate laws",
            "pre-registered confirmation only after discovery observations are frozen",
        ],
        "stop_rule": "route-level failures retire only that branch; continue through the pre-authorized observation atlas",
    }
    summary = {
        "phase": 1518,
        "campaign": "C088",
        "status": "major_stage_complete",
        "ledger": ledger,
        "all_audits_passed": all_audits,
        "verdict": verdict,
        "core_piece": core_piece,
        "behavior": {
            "global_accuracy": capture["global_accuracy"],
            "partition": capture["partition"],
            "surface": capture["surface"],
            "codebook": capture["codebook"],
            "semantic": capture["semantic"],
            "truth_code": capture["truth_code"],
        },
        "coordinate_stability": diagnostics["coordinate_stability"],
        "next_program": next_program,
        "theory_update": (
            "RDC keeps the same name. Its empirical object is narrowed to a conditional full-state response field "
            "whose semantic, protocol-code, and interaction terms can be separately observed but are not yet causally closed."
        ),
        "mathematics": "existing factorial contrasts, conditional dynamical systems, causal ordering, and full-dimensional vector comparison suffice for C088",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checks = {
        "six_prior_phases": len(ledger) == 6,
        "all_audits": all_audits,
        "all_predictions_passed": reveal["verdict"]["all_pre_registered_predictions_passed"],
        "behavior_boundary_retained": diagnostics["behavior_boundary"]["different_reversed_accuracy"] == 0.0,
        "claim_is_bounded": not verdict["localized_or_necessary_mechanism"] and not verdict["cross_model_invariant"],
        "one_core_piece": core_piece["id"] == "K265",
        "next_route_observation_first": "observation_atlas" in next_program["authorization"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "analysis/stage_ledger.json", ledger)
    core.save(OUT / "analysis/major_stage_summary.json", summary)
    final = {
        "phase": 1518,
        "campaign": "C088",
        "status": "major_stage_complete",
        "checks": checks,
        "authorization": next_program["authorization"],
        "auto_continue": False,
        "reason": "C088 is complete; C089 requires a new frozen natural-task material contract",
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
