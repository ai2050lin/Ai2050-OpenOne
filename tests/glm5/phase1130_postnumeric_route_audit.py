#!/usr/bin/env python3
"""Phase 1130: deterministic post-numeric route and identifiability audit.

This phase does not run a model. It verifies the frozen Phase 1127-1129
boundaries and turns proposed continuations into explicit authorization
decisions. The output is intentionally evidence bookkeeping, not a new
language-mechanism claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "tests" / "glm5" / "result" / "phase1130_postnumeric_route_audit"

SOURCE_RESULTS = {
    "phase1127": {
        "final": REPO_ROOT
        / "tests/glm5/result/phase1127_semeval_score_anatomy_qwen14b/analysis/final_summary.json",
        "audit": REPO_ROOT
        / "tests/glm5/result/phase1127_semeval_score_anatomy_qwen14b/audit/result_audit.json",
        "expected_final_digest": "a19ddd00efa84f20bf07ada921018e52334d173d91ce5e2bf29f8d1d7511589b",
        "expected_audit_digest": "03f02de899b6e08571a9260184320a50d8c4c4f8270715f7b0503e7ee74f4151",
    },
    "phase1128": {
        "final": REPO_ROOT
        / "tests/glm5/result/phase1128_fp16_numeric_formation/analysis/final_summary.json",
        "audit": REPO_ROOT
        / "tests/glm5/result/phase1128_fp16_numeric_formation/audit/result_audit.json",
        "expected_final_digest": "3dfbf3391f062e52ed70aef0d477de191ed9351b81a63d209d0732bb7f9d6c52",
        "expected_audit_digest": "3765156f8b200cef873f998e555e094be3d05a645b5544a448c309d138bc7364",
    },
    "phase1129": {
        "final": REPO_ROOT
        / "tests/glm5/result/phase1129_ds7b_attention_numeric_refinement/analysis/final_summary.json",
        "audit": REPO_ROOT
        / "tests/glm5/result/phase1129_ds7b_attention_numeric_refinement/audit/result_audit.json",
        "expected_final_digest": "d53c1328b61fdfeae0815dd18aa0a29b587eab5b3e445e90544e637f494fe9df",
        "expected_audit_digest": "de91b135149e797e760bf6ad1840c858f4821c0a52d666aaca720d3ca6f18a69",
    },
}

EXTERNAL_ANALYSES = [
    {
        "source_id": "analysis_1",
        "sha256": "1e3c50f9ae4bf177b9ddf0a09b78f9e0ea3f25ecb55dea1308b58c46c0fc34c3",
        "bytes": 9757,
        "judgment": "mostly_correct_with_boundary_overreach",
    },
    {
        "source_id": "analysis_2",
        "sha256": "15be36f46baa6cfeb019812aab28b78434b1e80792d43227684ad39bae57a368",
        "bytes": 11458,
        "judgment": "most_rigorous_with_local_overclaims",
    },
    {
        "source_id": "analysis_3",
        "sha256": "439fa0b27b3a0ed79fb907fc27198269fe532fda249f79a82fcc354a285ff684",
        "bytes": 11634,
        "judgment": "correct_core_but_proposed_continuations_bypass_gates",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_digest(payload: dict[str, Any], digest_key: str) -> str:
    body = dict(payload)
    body.pop(digest_key, None)
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def check_source_integrity() -> tuple[dict[str, Any], dict[str, bool]]:
    sources: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for source_id, spec in SOURCE_RESULTS.items():
        final = read_json(spec["final"])
        audit = read_json(spec["audit"])
        sources[source_id] = {
            "phase": final["phase"],
            "final_digest": final["final_digest"],
            "audit_digest": audit["audit_digest"],
            "audit_passed": audit["passed"],
        }
        checks[f"{source_id}_final_digest"] = (
            final["final_digest"] == spec["expected_final_digest"]
        )
        checks[f"{source_id}_audit_digest"] = (
            audit["audit_digest"] == spec["expected_audit_digest"]
        )
        checks[f"{source_id}_audit_passed"] = audit["passed"] is True
    return sources, checks


def build_protocol(source_manifest: dict[str, Any]) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "schema_version": "phase1130_postnumeric_route_audit_protocol.v1",
        "phase": 1130,
        "analysis_type": "deterministic_evidence_and_route_authorization_audit",
        "model_execution": False,
        "source_results": source_manifest,
        "external_analysis_manifest": EXTERNAL_ANALYSES,
        "frozen_boundaries": {
            "phase1127": "The frozen SemEval cross-model gate remained failed; same-carrier hidden, component, and causal work was not authorized.",
            "phase1128": "Numerical localization was descriptive and authorized exactly one DS7B numerical subcomponent audit.",
            "phase1129": "The one-shot numerical refinement is closed regardless of result; no automatic precision, placement, layer, or suboperator search is permitted.",
        },
        "restart_gate": {
            "formula": "M_new AND A_independent AND N_prefrozen AND Q_numeric_ge_2 AND B_discovery_ge_2 AND B_confirmation_ge_2 AND S_specificity AND U_natural",
            "terms": {
                "M_new": "independent material rather than the frozen SemEval carrier",
                "A_independent": "independent annotation and candidate-uniqueness audit",
                "N_prefrozen": "strong matched null frozen before model scoring",
                "Q_numeric_ge_2": "at least two models pass carrier-specific FP16 finite and replay qualification",
                "B_discovery_ge_2": "at least two qualified models pass the discovery behavior gate",
                "B_confirmation_ge_2": "the same requirement passes independent confirmation",
                "S_specificity": "active effect exceeds the matched null, not merely zero",
                "U_natural": "the candidate object predicts or controls natural use before internal localization",
            },
        },
        "prohibitions": [
            "same_carrier_hidden_or_dynamic_scan",
            "same_carrier_free_generation_patching",
            "bf16_or_fp32_semantic_backfill",
            "reopening_ds7b_l27_numeric_suboperator_search",
            "treating_other_carrier_numeric_health_as_current_carrier_qualification",
            "adaptive_llm_negative_generation_against_test_model_outputs",
        ],
        "evidence_policy": {
            "new_puzzle_allowed": False,
            "theory_update_allowed": False,
            "reason": "This phase audits existing evidence and proposed routes; it observes no new model or language object.",
        },
    }
    protocol["protocol_digest"] = canonical_digest(protocol, "protocol_digest")
    return protocol


def build_final(protocol: dict[str, Any], source_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        {
            "route": "same_carrier_qwen3_dynamic_hidden_map",
            "status": "rejected_by_frozen_gate",
            "automatic_execution": False,
            "reason": "Phase1127 P8 is false; changing the statistic from endpoint geometry to a trajectory does not create behavior or cross-model authorization.",
        },
        {
            "route": "same_carrier_free_generation_and_causal_patching",
            "status": "rejected_by_evidence_order",
            "automatic_execution": False,
            "reason": "Natural-use, hidden-repeat, matched-control, and cross-model gates are absent; patching would reopen the stopped carrier search.",
        },
        {
            "route": "bf16_or_fp32_rescue_as_semantic_backfill",
            "status": "rejected_as_backfill",
            "automatic_execution": False,
            "reason": "Phase1129 closed the one-shot numerical axis, the project requires FP16 for the current panel, and a different precision cannot retroactively qualify Phase1126 semantics.",
        },
        {
            "route": "pythia_as_second_semantic_panel_model",
            "status": "not_currently_authorized",
            "automatic_execution": False,
            "reason": "Pythia is outside the frozen Qwen3/GLM4/DS7B panel and any health result on another carrier is not carrier-specific numerical or behavioral qualification.",
        },
        {
            "route": "llm_generated_adversarial_null",
            "status": "conditionally_legal_but_material_not_identified",
            "automatic_execution": False,
            "reason": "A generator can add artifacts and adaptive leakage. The material must be independently frozen, human-audited, uniqueness-checked, and generated without test-model feedback.",
        },
        {
            "route": "new_independent_natural_semantic_material",
            "status": "legal_restart_target_but_missing_inputs",
            "automatic_execution": False,
            "reason": "This is the only route consistent with the frozen boundaries, but no new independently audited material or two-model carrier-specific FP16 panel is currently supplied.",
        },
    ]

    corrections = [
        "The observed DS7B row had no finite pre-softmax score; the audit did not isolate unmasked positions, raw RoPE output, raw dot product, scale, and mask as separate causes.",
        "The qk_score result is specific to this model, carrier, scored positions, FP16, CPU/GPU placement, and Transformers eager implementation; it is not an architecture-intrinsic defect.",
        "Numerical failure blocks semantic evaluation; it does not prove semantic equality, semantic absence, or that all behavior failures in other protocols are numerical.",
        "Qwen3 numerical health is carrier-specific measurement qualification, not universal camera calibration or evidence of a language mechanism.",
        "The 4B/14B difference does not identify parameter scale because training data, checkpoint details, placement, and interface effects remain entangled.",
        "BF16 is not established as a preferred precision, and free generation does not by itself isolate semantic use.",
    ]

    final: dict[str, Any] = {
        "schema_version": "phase1130_postnumeric_route_audit_final.v1",
        "phase": 1130,
        "protocol_digest": protocol["protocol_digest"],
        "source_state": {
            "phase1127_hidden_authorized": source_payloads["phase1127"]["predictions"]["P8_hidden_authorized"],
            "phase1128_refinement_authorized": source_payloads["phase1128"]["automatic_refinement"]["value"],
            "phase1129_auto_continue": source_payloads["phase1129"]["auto_continue"]["value"],
            "phase1129_qk_score_count": source_payloads["phase1129"]["root_counts_source_nonfinite"]["qk_score"],
        },
        "cross_analysis_judgment": {
            "shared_correct_core": [
                "Phase1128 reproduced the frozen finite/nonfinite outcomes and localized first recorded component failures.",
                "Phase1129 localized 314 DS7B failures to the recorded pre-softmax QK-score path before softmax propagation.",
                "GLM4 has only five localized failures and therefore remains a narrow E2 numerical boundary.",
                "Numerical qualification is logically prior to semantic, hidden-state, component, and causal interpretation.",
                "No new mathematics or intelligence-theory operator is needed for the numerical result.",
            ],
            "required_corrections": corrections,
            "source_ranking": [
                "analysis_2 is the most rigorous overall, subject to the score-row and camera-scope corrections.",
                "analysis_1 is broadly correct but its proposed same-carrier dynamic continuation violates Phase1127.",
                "analysis_3 has a correct numerical core but its immediate patching, BF16 standardization, and scale narrative exceed the evidence.",
            ],
        },
        "route_decisions": candidates,
        "restart_decision": {
            "auto_continue": False,
            "reason": "No candidate currently satisfies the frozen independent-material, matched-null, two-model FP16 numerical, two-split behavior, specificity, and natural-use prerequisites.",
            "next_legal_trigger": "A new independently audited natural-semantic material package plus at least two carrier-specific FP16-qualified panel models.",
        },
        "evidence_update": {
            "new_k_item": None,
            "theory_update_number": None,
            "measurement_gate_unchanged": True,
            "intelligence_theory_unchanged": True,
        },
        "interpretation_boundary": "Phase1130 is a route authorization audit. It adds no empirical language, hidden-state, component, scale, precision-preference, or causal evidence.",
    }
    final["final_digest"] = canonical_digest(final, "final_digest")
    return final


def main() -> None:
    source_manifest, checks = check_source_integrity()
    source_payloads = {
        source_id: read_json(spec["final"])
        for source_id, spec in SOURCE_RESULTS.items()
    }

    protocol = build_protocol(source_manifest)
    final = build_final(protocol, source_payloads)

    checks.update(
        {
            "phase1127_hidden_not_authorized": source_payloads["phase1127"]["predictions"]["P8_hidden_authorized"] is False,
            "phase1127_auto_continue_false": source_payloads["phase1127"]["auto_continue"]["value"] is False,
            "phase1128_refinement_was_one_shot": source_payloads["phase1128"]["automatic_refinement"]["value"] is True,
            "phase1129_qk_score_314": source_payloads["phase1129"]["root_counts_source_nonfinite"] == {"qk_score": 314},
            "phase1129_auto_continue_false": source_payloads["phase1129"]["auto_continue"]["value"] is False,
            "protocol_no_model_execution": protocol["model_execution"] is False,
            "all_routes_not_auto_executed": all(
                route["automatic_execution"] is False for route in final["route_decisions"]
            ),
            "same_carrier_hidden_rejected": final["route_decisions"][0]["status"] == "rejected_by_frozen_gate",
            "causal_patching_rejected": final["route_decisions"][1]["status"] == "rejected_by_evidence_order",
            "precision_backfill_rejected": final["route_decisions"][2]["status"] == "rejected_as_backfill",
            "new_material_is_only_restart_target": final["route_decisions"][5]["status"] == "legal_restart_target_but_missing_inputs",
            "auto_continue_false": final["restart_decision"]["auto_continue"] is False,
            "no_new_k_item": final["evidence_update"]["new_k_item"] is None,
            "no_theory_update": final["evidence_update"]["theory_update_number"] is None,
            "protocol_digest_valid": canonical_digest(protocol, "protocol_digest") == protocol["protocol_digest"],
            "final_digest_valid": canonical_digest(final, "final_digest") == final["final_digest"],
        }
    )

    audit: dict[str, Any] = {
        "schema_version": "phase1130_postnumeric_route_audit_result_audit.v1",
        "phase": 1130,
        "checks": checks,
        "passed_count": sum(bool(value) for value in checks.values()),
        "total_count": len(checks),
        "passed": all(checks.values()),
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
    }
    audit["audit_digest"] = canonical_digest(audit, "audit_digest")

    write_json(RESULT_ROOT / "protocol" / "preregistration.json", protocol)
    write_json(RESULT_ROOT / "analysis" / "final_summary.json", final)
    write_json(RESULT_ROOT / "audit" / "result_audit.json", audit)

    print(
        json.dumps(
            {
                "phase": 1130,
                "passed": audit["passed"],
                "passed_count": audit["passed_count"],
                "total_count": audit["total_count"],
                "protocol_digest": protocol["protocol_digest"],
                "final_digest": final["final_digest"],
                "audit_digest": audit["audit_digest"],
                "auto_continue": final["restart_decision"]["auto_continue"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
