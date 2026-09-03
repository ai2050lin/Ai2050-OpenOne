#!/usr/bin/env python3
"""C263: freeze the semantic-graph and state-conditioned operator contract."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import phase1797_c263_c272_state_operator_common as common

core, OUT = common.core, common.OUTS["C263"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.prior.RESULT / "phase1796_c262_full_word_generation_correction/audit/independent_final_audit.json")
    rows = common.material()
    compiled = common.compile_qwen(common.graph_base.tokenizer(), rows)
    core_rows = [row for row in rows if row["panel"] == "core"]
    nested = [row for row in rows if row["panel"] == "nested_composition"]
    prior_words = {u["primary"] for u in common.prior.UNITS}
    checks = {
        "parent": parent["all_checks_passed"], "rows": len(rows) == 768, "core_rows": len(core_rows) == 640,
        "nested_rows": len(nested) == 128, "five_families": {r["family"] for r in core_rows} == set(common.FAMILIES),
        "candidate_balance": sum(r["gold_position"] == 0 for r in rows) == 384,
        "new_primary_lexicon": not ({u["primary"] for u in common.UNITS} & prior_words),
        "unique_prompts": len({r["prompt"] for r in rows}) == len(rows),
        "semantic_graphs": all(r["semantic_graph"].get("nodes") and r["semantic_graph"].get("edges") for r in rows),
        "roles": all(set(r["role_positions"]) == set(common.ROLES) for r in compiled),
        "width": max(len(r["prompt_ids"]) for r in compiled) <= common.WIDTH,
        "human_blind_missing_registered": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", rows)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": 1797, "campaign": "C263", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_load", "research_object": "state-conditioned full-coordinate event operator",
        "rows": 768, "families": list(common.FAMILIES) + ["nested_attitude"], "surfaces": list(common.SURFACES),
        "operator": "guard(current baseline state, current edit response, semantic graph, role) -> signed next-checkpoint event and magnitude interval",
        "baselines": ["same-coordinate persistence", "fixed tri-material event core", "coordinate roll", "wrong family", "role-depth scaffold"],
        "forbidden_discovery_shortcuts": ["PCA", "top-k magnitude selection", "attention state", "MLP state", "post-reveal threshold changes"],
        "behavior_policy": "capture every row; mechanism claims use behavior-correct complete factorial groups, while failed strata remain visible",
        "gates": {"behavior_global_min": 0.85, "family_min": 0.70, "passport_support_min": 4, "passport_agreement_min": 0.70, "prediction_margin_min": 0.01, "causal_flip_min": 0.20, "causal_control_margin_min": 0.10},
        "stopping": "a failed route is retained as a negative result and does not stop other preregistered routes",
        "claim_boundary": "A predictive state-conditioned edge is not a unique natural circuit. Causal language is reserved for C269-C270 interventions.",
        "naturalness": "controlled grammatical English; independent human blind review is absent and explicitly limits external validity",
        "field_shape": [768, 37, 128, 2560], "producer_sha256": core.sha(Path(__file__)), "authorization": "run_C264_through_C272_without_route_level_global_stop",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {"checks": checks, "max_width": max(len(r["prompt_ids"]) for r in compiled), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    core.save(OUT / "analysis/final.json", {"phase": 1797, "campaign": "C263", "status": "closed", "all_checks_passed": True, "headline": protocol, "next_authorization": "C264_full_field"})
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

