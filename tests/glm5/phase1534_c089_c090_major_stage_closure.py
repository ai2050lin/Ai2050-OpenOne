#!/usr/bin/env python3
"""Phase1534: close the C089-C090 natural-relation observation and camera-repair stage."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1533_c090_holdout_and_artifact_adjudication"
OUT = RESULT / "phase1534_c089_c090_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE_DIRS = {
    1520: "phase1520_c088_full_input_code_semantics_correction",
    1521: "phase1521_c089_natural_relation_observation_contract",
    1522: "phase1522_c089_unified_forward_capture",
    1523: "phase1523_c089_truth_contrast_atlas",
    1524: "phase1524_c089_discovery_observation_freeze",
    1525: "phase1525_c089_descriptive_holdout_reveal",
    1526: "phase1526_c089_full_dimensional_diagnostics",
    1527: "phase1527_c090_singleton_numeric_calibration",
    1528: "phase1528_c090_right_padded_group_calibration_contract",
    1529: "phase1529_c090_right_padded_group_calibration",
    1530: "phase1530_c090_canonical_full_recapture",
    1531: "phase1531_c090_canonical_truth_contrast_atlas",
    1532: "phase1532_c090_discovery_observation_freeze",
    1533: "phase1533_c090_holdout_and_artifact_adjudication",
}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1534 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "run_phase1534_c089_c090_major_stage_closure" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1533 authorization missing")
    audits = {}
    for phase, dirname in PHASE_DIRS.items():
        audit = core.load(RESULT / dirname / "audit/independent_final_audit.json")
        audits[str(phase)] = {"passed": audit["passed"], "total": audit["total"], "all_checks_passed": audit["all_checks_passed"]}
    behavior = core.load(RESULT / PHASE_DIRS[1530] / "analysis/canonical_behavior_and_capture_summary.json")
    calibration = core.load(RESULT / PHASE_DIRS[1529] / "analysis/right_padded_group_calibration.json")
    reveal = core.load(RESULT / PHASE_DIRS[1533] / "analysis/holdout_and_artifact_adjudication.json")
    correction = core.load(RESULT / PHASE_DIRS[1520] / "analysis/full_input_code_semantics_correction.json")
    closure = {
        "phase": 1534, "campaign": "C089-C090", "status": "major_stage_complete",
        "audit_ledger": audits,
        "uploaded_analysis_adjudication": {
            "retained": [
                "C088 used a defined answer-code mapping because the system message supplied all four mapping clauses",
                "factorial contrasts and full-dimensional observation are useful when execution identity and causal-prefix identity are calibrated",
                "natural language object structure should enter the contract before Hidden-State interpretation",
                "observation must precede hypothesis freezing and holdout reveal",
            ],
            "corrected": [
                "Phase1519 is superseded because it omitted the system message",
                "C089 Phase1522-1525 hidden results are superseded by a causal-prefix camera failure",
                "the canonical target and boundary patterns are descriptive task-response geometry, not qualified semantic codes",
                "the additive shared/residual description is not an orthogonal direct sum and does not establish a circuit",
                "existing mathematics remains sufficient; new-mathematics claims are not triggered by these data",
            ],
        },
        "canonical_results": {
            "behavior_accuracy": behavior["global_accuracy"],
            "behavior_qualified_families": behavior["behavior_qualified_families"],
            "numeric_repeat_hidden_max_abs": calibration["repeat_hidden_max_abs"],
            "numeric_repeat_logit_max_abs": calibration["repeat_logit_max_abs"],
            "causal_prefix_max_relative_l2": calibration["causal_prefix_max_relative_l2"],
            "family_candidates": {family: result["candidate"] for family, result in reveal["family_results"].items()},
            "family_descriptive_replication": {family: result["descriptive_replication_all_components"] for family, result in reveal["family_results"].items()},
            "shared_candidate": reveal["shared_result"]["candidate"],
            "shared_descriptive_replication": reveal["shared_result"]["descriptive_replication_all_components"],
            "old_source_artifact_max_abs": reveal["execution_adjudication"]["source_truth_contrast_old_max_abs"],
            "canonical_source_artifact_max_abs": reveal["execution_adjudication"]["source_truth_contrast_canonical_max_abs"],
        },
        "puzzle_update": {
            "id": "K266", "evidence": "E3-HS-descriptive; semantic qualification absent",
            "title": "numerically canonical target-state and late-boundary natural-relation task-response field",
            "statement": "In frozen Qwen3 natural noun-relation prompts, exact source-target counterbalancing and a causally calibrated right-padded engine reveal reproducible family-associated target-state responses at states 22-27 and a shared state35 boundary response across discovery, confirmation, and lockbox. No relation family passed the frozen behavior gate, so this is a hidden-behavior dissociation and not evidence of a semantic relation code.",
        },
        "theory": {
            "name": "conditional output field closure theory",
            "organizing_principle": "reuse-difference-conditioning (RDC)",
            "update": "add an execution-identity operator and explicitly allow stable task-conditioned hidden responses that fail to close into qualified behavior",
            "contrast_formula": "C_f=(H(s1,t1)+H(s2,t2)-H(s1,t2)-H(s2,t1))/2",
            "descriptive_decomposition": "C_f=S+R_f+epsilon, where S=mean_f C_f; this is additive bookkeeping, not orthogonality",
            "closure_gate": "semantic interpretation requires behavior qualification AND numeric identity AND hidden replication; C090 has the latter two but not the first",
            "new_mathematics_required": False,
        },
        "hard_limits": [
            "WordNet machine naturalness does not replace independent human ratings",
            "all 45 composition sets were behavior-mixed and no family passed the two-surface discovery gate",
            "stable target and boundary geometry can reflect task comparison and answer preparation without correct semantic use",
            "one Qwen3 model and one English noun-relation contract do not establish cross-model or general-language invariants",
            "no attention, MLP, parameter, gradient, necessity, sufficiency, intervention, or natural generation mechanism was tested",
        ],
        "next_campaign": {
            "authorization": "preregister_c091_behavior_grounded_natural_relation_latent_use_bridge",
            "objective": "use fresh, independently natural-rated common-noun panels and relation-specific interfaces to obtain qualified behavior before testing whether the canonical target response predicts and transports to the late boundary",
            "routes": [
                "fresh common-word synonym/kind/part panels with human naturalness and ambiguity audit",
                "relation-specific behavior interfaces frozen independently rather than one universal template",
                "right-padded same-shape execution as a mandatory camera invariant",
                "all-state observation first; freeze target-to-boundary transport hypotheses only after discovery",
                "route-level retirement rather than project-level stop if one family fails",
            ],
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checks = {
        "audits": all(row["all_checks_passed"] for row in audits.values()),
        "correction": correction["phase1519_status"] == "superseded_due_to_incomplete_input_scope",
        "canonical_camera": calibration["canonical_right_padded_engine_pass"] and calibration["causal_prefix_max_relative_l2"] == 0.0,
        "behavior_boundary": behavior["behavior_qualified_families"] == [],
        "descriptive_replication": all(closure["canonical_results"]["family_descriptive_replication"].values()) and closure["canonical_results"]["shared_descriptive_replication"],
        "source_repair": closure["canonical_results"]["old_source_artifact_max_abs"] > 1e-2 and closure["canonical_results"]["canonical_source_artifact_max_abs"] == 0.0,
        "scope": "semantic qualification absent" in closure["puzzle_update"]["evidence"],
        "math": not closure["theory"]["new_mathematics_required"],
        "next": closure["next_campaign"]["authorization"] == "preregister_c091_behavior_grounded_natural_relation_latent_use_bridge",
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    closure["checks"] = checks
    core.save(OUT / "analysis/major_stage_closure.json", closure)
    core.save(OUT / "analysis/final.json", {"phase": 1534, "campaign": "C089-C090", "status": "major_stage_closed_with_k266_descriptive", "puzzle": "K266", "authorization": closure["next_campaign"]["authorization"]})
    print(json.dumps(closure, indent=2))


if __name__ == "__main__":
    main()
