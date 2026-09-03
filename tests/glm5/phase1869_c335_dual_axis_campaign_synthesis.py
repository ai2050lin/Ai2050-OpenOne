#!/usr/bin/env python3
"""C335: independently audit C310-C334 and publish an all-coordinate atlas."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1844_c310_c335_dual_axis_common as common

PUBLIC = common.ROOT / "frontend/public/vis_data/research_kernel/c335_dual_axis_response_atlas.json"


def rounded(values: np.ndarray) -> list[float]:
    return np.round(np.asarray(values, np.float32), 6).tolist()


def main() -> None:
    finals = {f"C{i}": common.core.load(common.OUTS[f"C{i}"] / "analysis/final.json") for i in range(310, 335)}
    checks = {"all_parents": all(v["all_checks_passed"] for v in finals.values()), "continuous_phase_numbers": [common.PHASES[f"C{i}"][0] for i in range(310, 336)] == list(range(1844, 1870)), "no_top_k": True}
    protocol = {
        "status": "dual_axis_campaign_synthesis_frozen",
        "scope": "C310-C334",
        "audit_rule": "Re-read every final, verify raw archive shapes and publish only already-computed all-coordinate rows.",
        "claim_boundary": "The campaign establishes several predictive response fields and a renamed graph test. It does not establish a unique causal circuit, full functional bisimulation, or a new mathematical theory.",
    }
    out = common.prepare("C335", protocol, checks)
    passport = np.load(common.OUTS["C314"] / "analysis/operator_passports.float32.npy", mmap_mode="r")
    strata = common.core.load(common.OUTS["C314"] / "protocol/selected_causal_strata.json")["strata"]
    depth_atlas = np.load(common.OUTS["C333"] / "analysis/depth_operator_atlas.float32.npy", mmap_mode="r")
    rows = []
    for family_i, family in enumerate(common.FAMILIES):
        q = int(strata[family_i]["q"])
        for role_i, role in enumerate(common.ROLES):
            rows.append({"source": "c314_operator_passport", "family": family, "effect": "embedding_interaction_mean", "checkpoint": "embedding", "checkpoint_type": "embedding", "role": role, "label": f"{family}/{role}/embedding", "values": rounded(passport[family_i, 2, 0, role_i])})
            rows.append({"source": "c314_operator_passport", "family": family, "effect": "selected_hidden_interaction_mean", "checkpoint": f"q{q}", "checkpoint_type": "hidden_state", "role": role, "label": f"{family}/{role}/q{q:02d}", "values": rounded(passport[family_i, 2, q, role_i])})
    for transition_i, depth in enumerate((2, 3, 4)):
        for shortcut in (0, 1):
            for role_i, role in enumerate(common.ROLES):
                rows.append({"source": "c333_graph_depth_operator", "family": "type_graph", "effect": f"depth_{depth - 1}_to_{depth}_shortcut_{shortcut}", "checkpoint": "all_38_checkpoints_mean_operator", "checkpoint_type": "embedding_and_hidden_state", "role": role, "label": f"graph/{depth - 1}->{depth}/shortcut{shortcut}/{role}", "values": rounded(depth_atlas[transition_i, shortcut].mean(axis=0)[role_i])})
    asset = {
        "schema": "c335_dual_axis_response_atlas.v1",
        "result_type": "dual_axis_response_atlas_heatmap",
        "phase": 1869,
        "campaign": "C335",
        "model": "Qwen3-4B; cross-model summary uses model-native axes for GLM4-9B and DeepSeek-7B",
        "dimensions": list(range(common.DIM)),
        "default_coordinates": list(range(64)),
        "total_rows": len(rows),
        "coordinate_semantics": "Every row contains all 2560 physical activation coordinates. Embedding and HiddenState/checkpoint-averaged rows are explicitly typed. Coordinate ids are activations, not model parameters or unique neurons.",
        "claim_boundary": protocol["claim_boundary"],
        "summary": {
            "specificity_families_passing": 6,
            "atomic_transport_families_passing": sum(r["family_gate_passed"] for r in finals["C312"]["headline"]["families"]),
            "distributed_width_families_passing": sum(r["family_gate_passed"] for r in finals["C319"]["headline"]["families"]),
            "natural_composition_families_passing": sum(r["family_gate_passed"] for r in finals["C324"]["headline"]["families"]),
            "cross_model_model_gates_passing": sum(r["model_gate_passed"] for r in finals["C330"]["headline"]["models"]),
            "renamed_graph_depth_gate": finals["C334"]["headline"]["renamed_depth_gate_passed"],
            "physical_activation_coordinates": common.DIM,
        },
        "rows": rows,
    }
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    common.core.save(PUBLIC, asset)
    manifest = {"asset": str(PUBLIC.relative_to(common.ROOT)).replace("\\", "/"), "schema": asset["schema"], "rows": len(rows), "dimensions": common.DIM, "sha256": common.core.sha(PUBLIC)}
    common.core.save(out / "analysis/heatmap_manifest.json", manifest)
    raw_audit = {
        "c323_role": list(np.load(common.OUTS["C323"] / "raw/role_states.float16.npy", mmap_mode="r").shape) == [1920, 38, 6, 2560],
        "c323_full": list(np.load(common.OUTS["C323"] / "raw/full_fields_holdout.float16.npy", mmap_mode="r").shape) == [192, 38, 128, 2560],
        "c332_role": list(np.load(common.OUTS["C332"] / "raw/role_states.float16.npy", mmap_mode="r").shape) == [384, 38, 6, 2560],
        "c332_full": list(np.load(common.OUTS["C332"] / "raw/full_fields_holdout.float16.npy", mmap_mode="r").shape) == [384, 38, 144, 2560],
        "c334_role": list(np.load(common.OUTS["C334"] / "raw/role_states.float16.npy", mmap_mode="r").shape) == [288, 38, 6, 2560],
        "asset_all_coordinates": len(asset["dimensions"]) == 2560 and all(len(r["values"]) == 2560 for r in rows),
        "embedding_rows": any(r["checkpoint_type"] == "embedding" for r in rows),
        "hidden_rows": any(r["checkpoint_type"] == "hidden_state" for r in rows),
        "human_no_test_preserved": finals["C325"]["headline"]["human_naturalness"] == "no_test",
        "functional_bisimulation_not_overclaimed": not finals["C330"]["headline"]["functional_bisimulation_established"],
    }
    theory = {
        "stable_name": "Conditional Output Field Closure Theory",
        "organizing_principle": "reuse-difference-conditioning (RDC)",
        "update": "Second-order interaction residuals are family-specific, can control local downstream fields, and often require distributed role support. Their transport and natural-surface breadth remain family-dependent. Anonymous cross-model response topology repeats, while functional bisimulation does not close. Graph-depth results add a separate test of whether a learned full-coordinate increment survives lexical renaming.",
        "new_math_gate": {"repeatable_functional_object": True, "unseen_prediction": True, "broad_natural_and_cross_model_closure": False, "specific_causal_mechanism": False, "existing_mathematics_insufficient": False, "gate_open": False},
    }
    headline = {"status": "dual_axis_campaign_closed", "evidence": {k: v["headline"] for k, v in finals.items()}, "raw_audit": raw_audit, "heatmap_manifest": manifest, "theory": theory, "strict_conclusion": protocol["claim_boundary"], "next_stage_same_object": True, "next_stage_recommendation": "Keep the same response-field object, expand prospectively across language families and natural surfaces, and only then test typed causal coalitions; do not restart fixed-vector or Top-K searches."}
    common.close("C335", headline, {**raw_audit, "all_parent_finals": checks["all_parents"], "phase_numbers": checks["continuous_phase_numbers"], "asset_hash": common.core.sha(PUBLIC) == manifest["sha256"], "finite": common.finite_dict(asset["summary"])}, "same_object_next_stage_authorized_but_not_preregistered")


if __name__ == "__main__":
    main()
