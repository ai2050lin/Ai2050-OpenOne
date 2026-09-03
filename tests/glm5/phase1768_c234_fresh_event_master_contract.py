#!/usr/bin/env python3
"""C234: freeze fresh materials, partitions, event rules, gates, and missingness."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C234"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.RESULT / "phase1767_c233_campaign_synthesis_heatmap/audit/independent_final_audit.json")
    rows = common.material()
    compiled = common.compile_rows(common.graph_base.tokenizer(), rows)
    expected = {"discovery": 320, "confirmation": 120, "lockbox": 80, "fresh": 120}
    partition_counts = {p: sum(row["partition"] == p for row in rows) for p in common.PARTITIONS}
    family_counts = {f: sum(row["family"] == f for row in rows) for f in common.FAMILIES}
    checks = {
        "authorization": parent["all_checks_passed"] and "fresh" in parent["authorization"],
        "rows": len(rows) == 640,
        "partition_counts": partition_counts == expected,
        "family_balance": set(family_counts.values()) == {128},
        "candidate_balance": sum(row["gold_position"] == 0 for row in rows) == 320,
        "prompt_unique": len({row["prompt"] for row in rows}) == 640,
        "surface_partition_disjoint": all(row["partition"] == common.SURFACE_PARTITION[row["surface"]] for row in rows),
        "roles": all(set(row["role_positions"]) == set(common.ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) <= common.WIDTH,
        "two_orders": all(sum(row["family"] == f and row["surface"] == s and row["unit"] == u and row["factor_a"] == a and row["factor_b"] == b for row in rows) == 2 for f in common.FAMILIES for s in common.SURFACES for u in common.PARTITION_UNITS[common.SURFACE_PARTITION[s]] for a in (0, 1) for b in (0, 1)),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled), "partition_counts": partition_counts})
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": 1768,
        "campaign": "C234",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fresh_full_coordinate_event_campaign_frozen",
        "research_object": "signed coordinate events and their order over every real token, every Qwen3 HiddenState checkpoint, and every physical activation coordinate",
        "models": ["qwen3", "glm4", "deepseek7b"],
        "sequential_model_loading": True,
        "families": list(common.FAMILIES),
        "surfaces": dict(common.SURFACE_PARTITION),
        "partitions": {key: list(value) for key, value in common.PARTITION_UNITS.items()},
        "rows": 640,
        "qwen_hidden_rows": 640,
        "candidate_orders_saved": [1, -1],
        "width": common.WIDTH,
        "physical_coordinates": common.DIM,
        "checkpoints": "embedding plus every transformer block output; exact count audited at runtime",
        "standard_factorial_scale": {"factor_a": "one_half_marginal_difference", "factor_b": "one_half_marginal_difference", "interaction": "difference_of_differences"},
        "event_threshold_formula": "per checkpoint max(4*duplicate_run_max_error, 0.25*discovery_positive_effect_q75, 1e-6); numeric values frozen by C235 before event reveal",
        "event_alphabet": ["down", "zero", "up"],
        "readable_rule_floor": {"event_prevalence_min": 0.75, "dominant_sign_min": 0.80, "role_event_density_min": 0.02, "precedence_rate_min": 0.75},
        "behavior_policy": {"stratum_accuracy_min": 0.65, "route_failure_is_local": True, "incorrect_rows_are_missing_for_mechanism_claims": True},
        "unseen_event_gate": {"correct_signed_jaccard_min": 0.15, "all_control_margin_min": 0.02, "families_min": 3},
        "composition_gate": {"family_signed_jaccard_min": 0.15, "families_min": 3, "must_beat_atomic_controls_by": 0.02},
        "causal_eligibility": "C238 unseen-event gate and C240 composition gate must both pass; failure produces typed_not_tested but does not stop C242 or C243",
        "cross_model_gate": {"models_min": 2, "all_participant_pairs_cosine_min": 0.30, "all_participant_pairs_permutation_p_max": 0.05},
        "semantic_uniqueness_audit": "deterministic truth table, exact candidate balance, unique prompts, role-span compilation, disjoint surface partitions",
        "naturalness_audit": "controlled-English internal audit only; an independent human blind panel is unavailable and is preregistered missingness, so natural-language generality cannot be claimed",
        "free_generation_panel": "one base factorial cell per family-surface-unit group, evaluated externally to the A/B candidate score",
        "forbidden": ["attention", "MLP", "weights", "PCA", "Top-K discovery", "post-reveal threshold edits", "using C223-C233 as current evidence", "project-level stop after a route failure"],
        "claim_boundary": "The campaign may identify repeatable HiddenState event rules. It cannot by itself identify a unique weight circuit, a semantic neuron, a topological invariant, or new mathematics.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "C235_Qwen3_all_layer_full_token_capture_then_C236_C243_in_frozen_order",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "max_width": max(len(row["prompt_ids"]) for row in compiled), "partition_counts": partition_counts, "human_blind_audit_missing": True})
    print(json.dumps({"checks": checks, "max_width": max(len(row["prompt_ids"]) for row in compiled), "partition_counts": partition_counts}, indent=2))


if __name__ == "__main__":
    main()
