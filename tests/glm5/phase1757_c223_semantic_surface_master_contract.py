#!/usr/bin/env python3
"""C223: freeze semantic graphs, surfaces, partitions, gates, and stop rules."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C223"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.RESULT / "phase1756_c222_surface_conditioned_response_decomposition/audit/independent_final_audit.json")
    rows = common.material()
    compiled = common.compile_rows(common.graph_base.tokenizer(), rows)
    prompts = [row["prompt"] for row in rows]
    family_surface = {(family, surface): sum(row["family"] == family and row["surface"] == surface for row in rows) for family in common.FAMILIES for surface in common.SURFACES}
    partitions = {part: sum(row["partition"] == part for row in rows) for part in ("discovery", "confirmation", "lockbox")}
    checks = {
        "authorization": parent["all_checks_passed"],
        "rows": len(rows) == 2304,
        "family_surface_balance": set(family_surface.values()) == {72},
        "partition_balance": partitions == {"discovery": 768, "confirmation": 768, "lockbox": 768},
        "candidate_balance": sum(row["gold_position"] == 0 for row in rows) == 1152,
        "prompt_unique": len(set(prompts)) == len(prompts),
        "all_roles": all(set(row["role_positions"]) == set(common.ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) <= common.WIDTH,
        "three_calibration_families": len(common.CALIBRATION_FAMILIES) == 3,
        "five_target_families": len(common.TARGET_FAMILIES) == 5,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled), "family_surface": family_surface, "partitions": partitions})
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": 1757,
        "campaign": "C223",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "semantic_graph_surface_rendering_contract_frozen",
        "research_object": "full signed embedding/HiddenState response field indexed by semantic family, surface, unit, checkpoint, role, and physical coordinate",
        "models": ["qwen3", "glm4", "deepseek7b"],
        "sequential_model_loading": True,
        "families": {"target": list(common.TARGET_FAMILIES), "surface_calibration": list(common.CALIBRATION_FAMILIES)},
        "surfaces": list(common.SURFACES),
        "partitions": {"discovery": [0, 1, 2], "confirmation": [3, 4, 5], "lockbox": [6, 7, 8]},
        "rows": 2304,
        "hidden_rows": 1152,
        "checkpoints": list(common.CHECKPOINTS),
        "roles": list(common.ROLES),
        "physical_coordinates": common.DIM,
        "behavior_floor": 0.65,
        "transport_confirmation_gate": {"median_nrmse_max": 0.85, "median_weighted_sign_min": 0.70, "identity_nrmse_improvement_min": 0.05},
        "transport_lockbox_gate": {"median_nrmse_max": 0.85, "median_weighted_sign_min": 0.70, "identity_nrmse_improvement_min": 0.05, "all_null_nrmse_margin_min": 0.02},
        "composition_lockbox_gate": {"family_median_nrmse_max": 0.80, "family_median_weighted_sign_min": 0.70, "families_min": 3},
        "causal_eligibility": "all transport gates and at least three composition families must pass; otherwise causal intervention is typed_not_tested while observational and cross-model routes continue",
        "model_tournament": ["identity", "common_offset", "typed_scalar_affine", "typed_coordinate_gain", "typed_coordinate_affine"],
        "nulls": ["wrong_surface", "wrong_family", "factor_swap", "coordinate_permutation", "same_norm_random", "energy_only"],
        "surface_transport_fit": "calibration families and discovery units only; target confirmation selects; lockbox remains excluded until C227",
        "semantic_uniqueness_audit": "deterministic role-span, candidate-balance, prompt-uniqueness, and factorial-accounting audit",
        "naturalness_audit": "controlled-English machine audit only; no independent human naturalness panel",
        "route_policy": "a failed route is downgraded or closed locally; it does not stop the remaining preregistered observation routes",
        "claim_boundary": "Passing identifies a predictive response-field regularity in this panel, not a fixed semantic coordinate, unique circuit, complete language mechanism, fiber bundle, topology hole, or new mathematics.",
        "forbidden": ["attention", "MLP", "weights", "PCA", "post-reveal threshold edits", "lockbox-selected model", "project-level stop after one route failure"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "C224_Qwen3_full_field_observation_then_C225_C233_in_frozen_order",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "max_width": max(len(row["prompt_ids"]) for row in compiled)})
    print(json.dumps({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled)}, indent=2))


if __name__ == "__main__":
    main()

