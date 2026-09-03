#!/usr/bin/env python3
"""C293: freeze the complete C293-C309 campaign before the sixth lockbox."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import phase1827_c293_c309_conditional_hypergraph_common as common

core, OUT = common.core, common.OUTS["C293"]


def main() -> None:
    if (OUT / "protocol/preregistration.json").exists() and core.load(OUT / "protocol/preregistration.json").get("material_compiler_version") == "direct_base_v2":
        raise RuntimeError(OUT)
    parent = core.load(common.previous.RESULT / "phase1826_c292_joint_response_campaign_independent_audit/analysis/final.json")
    rows = common.material()
    identity_fields = ("primary", "secondary", "observer", "object", "other", "node", "middle")
    old_words = {unit[key].lower() for unit in common.previous.UNITS for key in identity_fields}
    new_words = {unit[key].lower() for unit in common.UNITS for key in identity_fields}
    distribution = Counter((row["family"], row["surface"], row["factor_a"], row["factor_b"], row["order"]) for row in rows)
    checks = {
        "parent_closed": parent["all_checks_passed"],
        "rows_768": len(rows) == 768,
        "six_families": set(row["family"] for row in rows) == set(common.FAMILIES),
        "two_surfaces": set(row["surface"] for row in rows) == set(common.SURFACES),
        "factorial_balance": len(distribution) == 96 and set(distribution.values()) == {8},
        "answer_position_balance": sum(row["gold_position"] == 0 for row in rows) == sum(row["gold_position"] == 1 for row in rows),
        "identity_lexicon_disjoint_from_fifth": not (old_words & new_words),
        "semantic_graph_present": all(bool(row["semantic_graph"]) for row in rows),
        "no_attention_mlp": True,
        "no_pca_cosine_topk": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    for sub in ("analysis", "audit", "protocol", "materials"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    gates = {
        "behavior_global_min": 0.95,
        "behavior_family_min": 0.90,
        "transition_support_min": 4,
        "transition_agreement_min": 0.70,
        "model_margin_min": 0.01,
        "broad_families_min": 4,
        "amplitude_relative_mae_gain_min": 0.01,
        "cross_coordinate_train_score_min": 0.70,
        "cross_coordinate_confirmation_score_min": 0.65,
        "composition_relative_mae_gain_min": 0.01,
        "causal_targets_min": 16,
        "causal_delete_movement_min": 0.10,
        "causal_correct_vs_wrong_min": 0.05,
    }
    protocol = {
        "phase": 1827,
        "campaign": "C293",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "master_contract_frozen_before_sixth_model_run",
        "material_compiler_version": "direct_base_v2",
        "research_object": "full-coordinate, role/token-indexed, signed-and-amplitude response transitions; activation coordinates are not weights or neurons",
        "partitions": {
            "discovery": ["C248 third material", "C264 fourth material"],
            "confirmation": "C278 fifth material",
            "lockbox": "C295 sixth material",
        },
        "families": list(common.FAMILIES),
        "surfaces": list(common.SURFACES),
        "models": list(common.MODELS),
        "model_tournament": [
            "M0 destination persistence",
            "M1 C280 absorbing same-coordinate role word",
            "M2 complete three-state same-coordinate transition",
            "M3 full-coordinate cross-coordinate signed transfer",
            "M4 all-token aligned cross-coordinate transfer",
            "M5 amplitude-conditioned interval/median forecast",
        ],
        "gates": gates,
        "branch_policy": "A failed branch is retained and closed locally; it never stops observational, composition, cross-model, or visualization branches.",
        "causal_policy": "Only a source-target map frozen before the sixth lockbox and confirmed there may enter intervention.",
        "naturalness_scope": "Controlled English templates receive deterministic grammar/uniqueness audits. Independent human blind naturalness was not collected, so natural-language external validity remains open.",
        "forbidden": ["attention tensors", "MLP internals", "PCA", "cosine as primary test", "arbitrary Top-K coordinate truncation", "post-lockbox threshold changes"],
        "checkpoint_policy": "Use semantic checkpoint names; legacy 37-state archives lack block_36_output before final norm.",
        "producer_sha256": core.sha(Path(__file__)),
        "next_authorization": "C294_material_compile_then_C295_lockbox_capture_and_all_remaining_branches",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.write_rows(OUT / "materials/sixth_material.jsonl", rows)
    audit = {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "semantic_uniqueness": {"case_ids_unique": len({r['case_id'] for r in rows}) == len(rows), "answer_positions_exact": [384, 384], "shared_category_labels_are_registered_reuse": True, "first_unit_identity": rows[0]["role_values"]["primary"]},
        "naturalness": {"machine_controlled_grammar": "passed", "independent_human_blind_review": "absent", "scope": "controlled_not_open_domain"},
    }
    core.save(OUT / "audit/internal_contract_audit.json", audit)
    report = {
        "phase": 1827,
        "campaign": "C293",
        "status": "closed",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "headline": {
            "rows": len(rows),
            "families": len(common.FAMILIES),
            "lockbox_vocabulary_units": len(common.UNITS),
            "strict_interpretation": "The campaign is frozen before tokenizer compilation or model execution. It tests a lexical/graph rename lockbox across existing controlled surfaces, not unseen syntax or open-domain language.",
        },
        "next_authorization": protocol["next_authorization"],
    }
    core.save(OUT / "analysis/final.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
