#!/usr/bin/env python3
"""C277: freeze the fifth-material joint-response campaign before model load."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import phase1811_c277_c289_joint_response_common as common

core, OUT = common.core, common.OUTS["C277"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.previous.RESULT / "phase1810_c276_prospective_cross_role_reuse_prediction/audit/independent_audit.json")
    rows = common.material()
    compiled = common.compile_qwen(common.graph_base.tokenizer(), rows)
    identity_keys = ("primary", "secondary", "observer", "object", "other", "node", "middle")
    old_words = {unit[key] for unit in common.previous.UNITS for key in identity_keys}
    new_words = {unit[key] for unit in common.UNITS for key in identity_keys}
    per_family_position = {
        family: [sum(r["gold_position"] == position for r in rows if r["family"] == family) for position in (0, 1)]
        for family in common.FAMILIES
    }
    checks = {
        "parent": parent["all_checks_passed"],
        "rows": len(rows) == 768,
        "families": {row["family"] for row in rows} == set(common.FAMILIES),
        "surfaces": {row["surface"] for row in rows} == set(common.SURFACES),
        "disjoint_identity_lexicon": not bool(old_words & new_words),
        "global_position_balance": sum(r["gold_position"] == 0 for r in rows) == 384,
        "family_position_balance": all(left == right for left, right in per_family_position.values()),
        "unique_prompts": len({r["prompt"] for r in rows}) == 768,
        "semantic_graph": all(r["semantic_graph"].get("nodes") and r["semantic_graph"].get("edges") for r in rows),
        "roles": all(set(r["role_positions"]) == set(common.ROLES) for r in compiled),
        "width": max(len(r["prompt_ids"]) for r in compiled) <= common.WIDTH,
        "machine_naturalness": all("Question:" in r["prompt_core"] or "Decide:" in r["prompt_core"] for r in rows),
        "human_blind_review_missing_registered": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": 1811,
        "campaign": "C277",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_load",
        "research_object": "joint signed response word and multi-source response hyperedge",
        "rows": len(rows),
        "families": list(common.FAMILIES),
        "surfaces": list(common.SURFACES),
        "raw_checkpoints": list(common.RAW_CHECKPOINTS),
        "canonical_cross_material_checkpoints": list(common.CANONICAL_CHECKPOINTS),
        "checkpoint_erratum": "Old 37-state archives contain embedding, block 1..35 outputs, and final norm. They do not separately contain block-36 pre-final-norm. C278 records all 38 states; primary cross-material tests use the 37-state canonical intersection.",
        "routes": [
            "joint exact role-state words",
            "multi-source one-step prediction with persistence and coordinate-permutation controls",
            "eligible long-horizon rollout with legal abstention",
            "factorial composition panels for attitude, type graph, contrast, translation, and comparison",
            "prospectively eligible local causal test",
            "free generation and side effects",
            "sequential Qwen3/GLM4/DeepSeek-7B anonymous automaton comparison",
        ],
        "forbidden": ["PCA", "cosine-nearest-neighbor discovery", "top-k coordinate selection", "attention states", "MLP states", "post-reveal gate changes"],
        "gates": {
            "behavior_global_min": 0.85,
            "family_min": 0.70,
            "word_support_min": 4,
            "word_agreement_min": 0.70,
            "one_step_margin_min": 0.01,
            "broad_families_min": 4,
            "causal_flip_min": 0.20,
            "causal_control_margin_min": 0.10,
        },
        "route_policy": "A failed route is closed without stopping the other preregistered routes. Causal work is no-test unless its prospective prediction route qualifies.",
        "claim_boundary": "Cross-role precedence is an event-ecology statistic, not a transport probability. Joint predictors are observational until deletion and selective rescue pass.",
        "naturalness": "Controlled grammatical English with two surfaces and disjoint natural/pseudoword lexica. Independent human blind review is absent.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C278_through_C289_with_route_level_not_global_stops",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {"checks": checks, "per_family_position": per_family_position, "max_width": max(len(r["prompt_ids"]) for r in compiled), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    final = {"phase": 1811, "campaign": "C277", "status": "closed", "all_checks_passed": True, "headline": protocol, "next_authorization": "C278_qwen_fifth_material_full_field"}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
