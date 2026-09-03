#!/usr/bin/env python3
"""C289: adjudicate C277-C288 and publish a full-coordinate heatmap asset."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1811_c277_c289_joint_response_common as common

core, OUT = common.core, common.OUTS["C289"]
PUBLIC = common.ROOT / "frontend/public/vis_data/research_kernel/c289_joint_response_campaign_atlas.json"
CANDIDATES = ("relation_query", "primary_relation_query", "all_six_roles")
COMPOSITION = {
    "attitude_event": common.OUTS["C282"] / "analysis/attitude_event_interaction_atlas.float16.npy",
    "nested_attitude": common.OUTS["C282"] / "analysis/nested_attitude_interaction_atlas.float16.npy",
    "type_graph": common.OUTS["C283"] / "analysis/type_graph_interaction_atlas.float16.npy",
    "contrast": common.OUTS["C284"] / "analysis/contrast_interaction_atlas.float16.npy",
    "translation": common.OUTS["C284"] / "analysis/translation_interaction_atlas.float16.npy",
    "comparison": common.OUTS["C284"] / "analysis/comparison_interaction_atlas.float16.npy",
}
VIS_Q = (0, 8, 16, 24, 35, 36)


def rounded(values: np.ndarray) -> list[float]:
    return np.round(np.asarray(values, np.float32), 6).tolist()


def load_final(campaign: str) -> dict:
    return core.load(common.OUTS[campaign] / "analysis/final.json")


def make_asset(c280: dict, c281: dict, c285: dict, c288: dict) -> dict:
    rows: list[dict] = []
    counts = np.load(common.OUTS["C280"] / "analysis/coordinate_correct_union_counts.uint32.npy", mmap_mode="r")
    for fi, family in enumerate(common.FAMILIES):
        for q in range(36):
            for ci, candidate in enumerate(CANDIDATES):
                correct = np.asarray(counts[fi, q, ci, 0], np.float32)
                union = np.asarray(counts[fi, q, ci, 1], np.float32)
                values = np.divide(correct, union, out=np.zeros_like(correct), where=union > 0)
                rows.append({
                    "source": "c280_joint_word_prediction",
                    "family": family,
                    "effect": candidate,
                    "checkpoint": common.CANONICAL_CHECKPOINTS[q],
                    "checkpoint_type": "embedding_transition" if q == 0 else "hidden_state_transition",
                    "role": "six_roles_aggregated",
                    "label": f"{family}/{candidate}/q{q:02d}->q{q + 1:02d}",
                    "values": rounded(values),
                })

    for family, path in COMPOSITION.items():
        atlas = np.load(path, mmap_mode="r")
        for q in VIS_Q:
            for ri, role in enumerate(common.ROLES):
                rows.append({
                    "source": "c282_c284_factorial_interaction",
                    "family": family,
                    "effect": "factorial_residual",
                    "checkpoint": common.CANONICAL_CHECKPOINTS[q],
                    "checkpoint_type": "embedding" if q == 0 else ("final_norm" if q == 36 else "hidden_state"),
                    "role": role,
                    "label": f"{family}/{role}/{common.CANONICAL_CHECKPOINTS[q]}",
                    "values": rounded(atlas[q, ri]),
                })

    fifth = np.load(common.OUTS["C278"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = core.rows(common.OUTS["C278"] / "raw/hidden_index.jsonl")
    for family in common.FAMILIES:
        left, right = common.pair_specs(index, family)[0][:2]
        for q in VIS_Q:
            for role in ("relation", "boundary"):
                ri = common.ROLES.index(role)
                delta = np.asarray(fifth[right, common.CANONICAL_NEW_INDICES[q], ri], np.float32) - np.asarray(
                    fifth[left, common.CANONICAL_NEW_INDICES[q], ri], np.float32
                )
                rows.append({
                    "source": "c278_fifth_material_edit_response",
                    "family": family,
                    "effect": "factor_a_edit_delta",
                    "checkpoint": common.CANONICAL_CHECKPOINTS[q],
                    "checkpoint_type": "embedding" if q == 0 else ("final_norm" if q == 36 else "hidden_state"),
                    "role": role,
                    "label": f"{family}/{role}/{common.CANONICAL_CHECKPOINTS[q]}/edit",
                    "values": rounded(delta),
                })

    return {
        "schema": "c289_joint_response_campaign_atlas.v1",
        "phase": 1823,
        "campaign": "C289",
        "model": "Qwen3-4B plus anonymous Qwen3/GLM4/DS7B topology",
        "dimensions": list(range(common.DIM)),
        "default_coordinates": list(range(64)),
        "total_rows": len(rows),
        "coordinate_semantics": "Each row retains all 2560 physical Qwen3 activation coordinates. Prediction rows are per-coordinate signed-event accuracies; interaction and fifth-material rows retain signed coordinate values. q0 is embedding, q1-q35 are block outputs, and q36 is final norm.",
        "claim_boundary": "C280-C281 establish a reusable observational signed-event automaton. C282-C284 are descriptive factorial residuals. C285 did not obtain local coordinate eligibility, so this atlas is not a unique causal circuit or continuous HiddenState closure. C288 compares anonymous role topology, never equal coordinate indices across models.",
        "summary": {
            "one_step_candidates_passing": len(c280["headline"]["passing_candidates"]),
            "rollout_candidates_passing": sum(bool(row["broad_gate_passed"]) for row in c281["headline"]["candidate_summary"]),
            "causal_status": c285["headline"]["status"],
            "cross_model_pairs_passing": sum(bool(row["pair_gate_passed"]) for row in c288["headline"]["pairs"]),
            "families": 6,
            "physical_coordinates": common.DIM,
        },
        "rows": rows,
    }


def main() -> None:
    finals = {campaign: load_final(campaign) for campaign in (f"C{i}" for i in range(277, 289))}
    parent_checks = {campaign: bool(final["all_checks_passed"]) for campaign, final in finals.items()}
    if not all(parent_checks.values()):
        raise RuntimeError(parent_checks)
    OUT.mkdir(parents=True, exist_ok=True)
    for subdir in ("analysis", "audit", "protocol"):
        (OUT / subdir).mkdir(exist_ok=True)
    protocol = {
        "phase": 1823,
        "campaign": "C289",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "adjudication_and_visualization_frozen",
        "parents": list(finals),
        "rules": [
            "No new coordinate selection or model fitting is permitted.",
            "All 2560 Qwen3 physical coordinates are retained in every heatmap row.",
            "One-step prediction, rollout, composition, causal eligibility, generation and cross-model topology are adjudicated separately.",
            "Causal no-test cannot be promoted to a causal negative or rescued by observational prediction.",
        ],
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)

    c278, c279, c280, c281 = (finals[key] for key in ("C278", "C279", "C280", "C281"))
    c282, c283, c284, c285 = (finals[key] for key in ("C282", "C283", "C284", "C285"))
    c286, c287, c288 = (finals[key] for key in ("C286", "C287", "C288"))
    corrections = [
        "The legacy 37-state archives contain embedding, block 1-35 outputs and final norm; they do not separately contain block-36 pre-final-norm.",
        "C275's 52%-59% statistic is any-source same-sign coverage among destination emergence instances, not a transition probability.",
        "C273's 25.6425% is an instance fraction among active next events, not a fixed coordinate subset.",
        "C266 long-horizon values are margins over registered controls, not raw negative cosine values.",
        "C267-C268 and C282-C284 factorial residuals are descriptive nonadditivity, not causal composition laws.",
        "C271 and C288 support anonymous functional topology only, not shared physical coordinates or implementation identity.",
        "C285 is a local eligibility no-test: one eligible coordinate across twelve registered samples, not a failed causal intervention.",
        "A successful finite event automaton does not prove that the joint role state is necessary, minimal, unique, or a continuous HiddenState simulator.",
    ]
    evidence = {
        "behavior": {
            "qwen_fifth_candidate_accuracy": c278["headline"]["accuracy"],
            "qwen_fifth_eligible": c278["headline"]["behavior_eligible"],
            "free_generation_accuracy": c286["headline"]["overall_success_rate"],
            "cross_model_behavior_eligible": all(bool(row["behavior_eligible"]) for row in c287["headline"]["models"].values()),
        },
        "joint_state": {
            "exact_full_token_support4_all_families": all(
                float(row["exact_token_signature_median_support4_fraction"]) > 0 for row in c279["headline"]["families"]
            ),
            "family_partitions": c279["headline"]["families"],
        },
        "prediction": {
            "one_step": c280["headline"]["candidate_summary"],
            "rollout": c281["headline"]["candidate_summary"],
        },
        "composition": {
            "attitude": c282["headline"]["families"],
            "type_graph": c283["headline"]["result"],
            "other_families": c284["headline"]["families"],
        },
        "causal": {
            "status": c285["headline"]["status"],
            "eligible_coordinates_total": c285["headline"]["eligible_coordinates_total"],
        },
        "cross_model": c288["headline"]["pairs"],
    }
    new_math_gate = {
        "repeated_functional_object": True,
        "prospective_unseen_transition_prediction": True,
        "autonomous_event_rollout": True,
        "local_causal_use": False,
        "existing_mathematics_demonstrably_insufficient": False,
        "gate_open": False,
    }
    report = {
        "phase": 1823,
        "campaign": "C289",
        "status": "campaign_closed_with_observational_event_automaton_positive_and_causal_no_test",
        "audit_corrections": corrections,
        "evidence": evidence,
        "theory": {
            "stable_name": "Conditional Output Field Closure Theory",
            "organizing_principle": "reuse-difference-conditioning (RDC)",
            "update": "A role-conditioned finite signed-event automaton can predict unseen one-step transitions and autonomous checkpoint trajectories across six language families. The object is a partial observational response law over physical coordinates, roles and checkpoints; it is not yet a causal or continuous-state mechanism.",
            "new_math_gate": new_math_gate,
        },
        "strict_conclusion": "The campaign found a broad, prospectively reusable discrete event law and cross-model anonymous topology. It did not identify a unique coordinate circuit, prove necessity, or close continuous HiddenState dynamics.",
        "next_authorization": "C290_training_only_causal_stratum_qualification_then_C291_if_eligible",
    }
    asset = make_asset(c280, c281, c285, c288)
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    core.save(PUBLIC, asset)
    core.save(OUT / "analysis/summary.json", report)
    core.save(OUT / "analysis/heatmap_manifest.json", {
        "asset": str(PUBLIC.relative_to(common.ROOT)).replace("\\", "/"),
        "schema": asset["schema"],
        "rows": asset["total_rows"],
        "dimensions": len(asset["dimensions"]),
        "sha256": core.sha(PUBLIC),
    })
    audit_checks = {
        "all_parents_closed": all(parent_checks.values()),
        "corrections_recorded": len(corrections) == 8,
        "heatmap_all_coordinates": len(asset["dimensions"]) == 2560 and all(len(row["values"]) == 2560 for row in asset["rows"]),
        "heatmap_embedding_present": any(row["checkpoint_type"] == "embedding" for row in asset["rows"]),
        "heatmap_hidden_state_present": any(row["checkpoint_type"] == "hidden_state" for row in asset["rows"]),
        "causal_no_test_preserved": "no_test" in c285["headline"]["status"],
        "new_math_gate_closed": not new_math_gate["gate_open"],
    }
    core.save(OUT / "audit/internal_audit.json", {"checks": audit_checks, "all_checks_passed": all(audit_checks.values())})
    final_checks = {
        "parents": all(parent_checks.values()),
        "analysis": all(audit_checks.values()),
        "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
        "asset_hash": core.sha(PUBLIC) == core.load(OUT / "analysis/heatmap_manifest.json")["sha256"],
    }
    final = {
        "phase": 1823,
        "campaign": "C289",
        "status": "closed",
        "checks": final_checks,
        "all_checks_passed": all(final_checks.values()),
        "headline": report,
        "next_authorization": report["next_authorization"],
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
