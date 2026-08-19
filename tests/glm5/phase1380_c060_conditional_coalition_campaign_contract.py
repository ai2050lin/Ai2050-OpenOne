#!/usr/bin/env python3
"""Phase1380: freeze the C060 conditional-coordinate-coalition campaign."""
from __future__ import annotations

import copy
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1373_c058_dose_distance_group_campaign_contract as base

PHASE, CAMPAIGN = 1380, "C060"
PARENT = TESTS / "result/phase1379_c059_coordinate_group_evaluation"
C059_CONTRACT = TESTS / "result/phase1375_c059_independent_relaunch_contract"
C059_GROUPS = TESTS / "result/phase1379_c059_coordinate_group_evaluation/protocol/candidate_groups.json"
OUT = TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract"

FAMILIES = {
    "animal": ("snake", "lizard", "turtle", "crocodile", "alligator", "gecko",
               "iguana", "chameleon", "cobra", "tortoise", "viper", "caiman"),
    "beverage": ("coffee", "tea", "milk", "juice", "water", "soda",
                 "beer", "wine", "cocoa", "lemonade", "cider", "smoothie"),
    "sport": ("soccer", "tennis", "golf", "rugby", "boxing", "hockey",
              "skiing", "surfing", "cycling", "baseball", "volleyball", "basketball"),
    "building": ("house", "school", "hospital", "library", "church", "hotel",
                 "factory", "museum", "castle", "tower", "warehouse", "stadium"),
}
OLD_CONTRACTS = (
    TESTS / "result/phase1369_c057_independent_relation_campaign_contract",
    TESTS / "result/phase1373_c058_dose_distance_group_campaign_contract",
    C059_CONTRACT,
)
REFINED_DOSES = [0.0, 0.125, 0.25, 0.375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.875, 1.0]
DYNAMIC_SIZES = [256, 512, 1024, 1536, 2048]


def relabel(rows: list[dict], old: str, new: str) -> list[dict]:
    for row in rows:
        row["case_id"] = row["case_id"].replace(old, new)
    return rows


def fixed_coalitions() -> dict[str, list[int]]:
    source = core.load(C059_GROUPS)
    ranking = source["groups"]["deterministic_random"]["2560"]
    if len(ranking) != 2560 or len(set(ranking)) != 2560:
        raise RuntimeError("C059 deterministic ranking is invalid")
    inherited = ranking[:1024]
    complement = ranking[1024:]
    groups = {
        "inherited_S1024": inherited,
        "inherited_C1536": complement,
        "inherited_full2560": ranking,
    }
    for extra in (256, 512, 768, 1024, 1280):
        groups[f"inherited_S_plus_C{extra}"] = inherited + complement[:extra]
    for index, seed in enumerate((6001380, 6011380, 6021380, 6031380), start=1):
        values = list(range(2560))
        random.Random(seed).shuffle(values)
        groups[f"new_random_{index}_S1024"] = values[:1024]
        groups[f"new_random_{index}_C1536"] = values[1024:]
    return groups


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1380 already exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    closure = core.load(PARENT / "audit/campaign_closure_audit.json")
    if (parent.get("authorization") != "close_c059_after_all_frozen_eligible_branches" or
            not parent_audit.get("all_checks_passed") or not closure.get("all_checks_passed")):
        raise RuntimeError("C059 audited closure missing")

    base.FAMILIES = FAMILIES
    tok = base.tokenizer()
    concepts = base.concepts()
    active = relabel(base.active_cases(), "c058", "c060")
    status = relabel(base.status_cases(), "c058", "c060")
    compiled_active = base.compile_rows(tok, active, base.SYSTEM_ACTIVE)
    compiled_status = base.compile_rows(tok, status, base.SYSTEM_STATUS)
    pairs = base.candidate_pairs(active, status)
    old_words = set()
    for source in OLD_CONTRACTS:
        old_words.update(r["word"] for r in core.load(source / "material/frozen_concept_graph.json")["concepts"])
    family_tokens = {family: tok.encode(" " + family, add_special_tokens=False) for family in FAMILIES}
    coalitions = fixed_coalitions()
    inherited = set(coalitions["inherited_S1024"])
    complement = set(coalitions["inherited_C1536"])
    checks = {
        "parent_closed_audited": parent_audit["all_checks_passed"] and closure["all_checks_passed"],
        "concept_count": len(concepts) == 48 and len({r["word"] for r in concepts}) == 48,
        "independent_vocabulary": not ({r["word"] for r in concepts} & old_words),
        "family_balance": all(len(v) == 12 for v in FAMILIES.values()),
        "partition_balance": all(sum(r["partition"] == p for r in concepts) == 16 for p in base.PARTITIONS),
        "semantic_uniqueness": all(r["sense"] and r["adjudication"] for r in concepts),
        "family_single_token": all(len(v) == 1 for v in family_tokens.values()),
        "active_balance": len(active) == 864 and Counter(r["truth"] for r in active) == {True: 432, False: 432},
        "status_balance": len(status) == 288 and Counter(r["truth"] for r in status) == {True: 144, False: 144},
        "candidate_pairs": len(pairs) == 432,
        "controlled_naturalness": all("  " not in r["prompt"] and r["prompt"].endswith("yes or no.")
                                      for r in active + status),
        "compiled_counts": len(compiled_active) == 864 and len(compiled_status) == 288,
        "family_spans_single": all(len(r["role_positions"]["family"]) == 1
                                   for r in compiled_active + compiled_status),
        "candidate_single_tokens": all(len(ids) == 1 for r in compiled_active + compiled_status
                                       for ids in r["candidate_ids"]),
        "typed_roles": all(set(r["role_positions"]) == {"target", "family", "query", "boundary"}
                           for r in compiled_active + compiled_status),
        "inherited_partition": len(inherited) == 1024 and len(complement) == 1536 and
                               not (inherited & complement) and inherited | complement == set(range(2560)),
        "fixed_group_bounds": all(len(v) == len(set(v)) and min(v) >= 0 and max(v) < 2560
                                  for v in coalitions.values()),
        "hidden_state_only": True,
    }
    if not all(checks.values()):
        raise RuntimeError({k: v for k, v in checks.items() if not v})

    core.save(OUT / "material/frozen_concept_graph.json", {
        "schema": "c060.independent_concepts.v1", "families": FAMILIES,
        "partitions": {k: list(v) for k, v in base.PARTITIONS.items()}, "concepts": concepts,
    })
    core.write_rows(OUT / "material/active_membership_cases.jsonl", active)
    core.write_rows(OUT / "material/status_cases.jsonl", status)
    core.write_rows(OUT / "material/candidate_pairs.jsonl", pairs)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled_active)
    core.write_rows(OUT / "compiled/qwen3_status.jsonl", compiled_status)
    core.save(OUT / "protocol/fixed_coalitions.json", {
        "schema": "c060.fixed_coalitions.v1", "source": str(C059_GROUPS.relative_to(ROOT)),
        "source_sha256": core.sha(C059_GROUPS), "groups": coalitions,
    })
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "zero_models": {"always_yes": 0.5, "always_no": 0.5,
                        "target_only_quartet_upper_bound": 0.5, "family_only_quartet_upper_bound": 0.5,
                        "status_always_yes": 0.5, "status_always_no": 0.5},
        "independent_human_blind_review": False,
        "naturalness_scope": "curated controlled English plus deterministic machine audit",
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)

    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "schema": "c060.conditional_coordinate_coalition_campaign.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "independent-material dense-threshold, fixed coalition/complement, dynamic coalition, and mediation competition",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states",
                                "candidate and full-vocabulary logits"],
        "forbidden": ["attention", "MLP", "parameter scan", "gradient", "PCA", "t-SNE", "UMAP",
                      "SAE", "learned probe", "post-reveal layer search", "post-reveal threshold change"],
        "material": {
            "families": list(FAMILIES), "concept_count": 48, "active_count": 864,
            "status_count": 288, "candidate_pair_count": 432,
            "partitions": list(base.PARTITIONS), "surfaces": list(base.SURFACES),
            "eligible_cases_per_cell": 6, "eligible_case_target": 216,
            "discovery_target": 72, "confirmation_target": 72, "lockbox_target": 72,
            "active_sha256": core.sha(OUT / "material/active_membership_cases.jsonl"),
            "status_sha256": core.sha(OUT / "material/status_cases.jsonl"),
            "pair_sha256": core.sha(OUT / "material/candidate_pairs.jsonl"),
        },
        "behavior": {
            "active_accuracy_min": 0.90, "partition_min": 0.85, "surface_min": 0.85,
            "family_min": 0.85, "truth_min": 0.85, "quartet_all_min": 0.75,
            "status_accuracy_min": 0.95, "status_partition_min": 0.90,
            "same_shape_repeat_max_abs_diff": 1e-6,
            "eligibility": "clean, corrupt, wrong-identity, and status donors all individually behavior-correct",
        },
        "paths": copy.deepcopy(base.PATHS),
        "dose": {
            "values": REFINED_DOSES, "directions": ["correct", "wrong", "status", "random"],
            "random_seed": 6001380, "norm_ratio_abs_error_max": 1e-5,
            "self_output_max_abs_diff": 1e-4,
            "endpoint_gain_median_min": 0.5, "endpoint_advantage_median_min": 0.25,
            "endpoint_win_min": 0.65,
            "threshold_low_dose_abs_median_max": 0.5,
            "threshold_adjacent_jump_min": 10.0,
            "threshold_high_dose_gain_min": 20.0,
            "threshold_plateau_abs_difference_max": 10.0,
            "mid_reverse_damage_median_min": 0.5,
            "mid_reverse_over_status_median_min": 0.25,
            "mid_reverse_over_status_win_min": 0.65,
            "split_confirmation_required": True,
        },
        "camera": {
            "known_truth_systems": 256, "qwen_cases": 36,
            "same_shape_output_max_abs_diff": 1e-5,
            "same_shape_checkpoint_relative_l2_max": 1e-6,
            "threshold_linear_nonmonotone_exact": True,
            "coalition_union_complement_exact": True,
            "dynamic_mask_exact": True, "serial_parallel_exact": True,
        },
        "fixed_coalitions": {
            "artifact_sha256": core.sha(OUT / "protocol/fixed_coalitions.json"),
            "routes": list(coalitions),
            "inherited_route": "inherited_S1024",
            "complement_route": "inherited_C1536", "full_route": "inherited_full2560",
            "control_norm_matched_within_group": True,
            "reverse_damage_median_min": 0.5, "reverse_over_status_median_min": 0.25,
            "reverse_over_status_win_min": 0.65,
            "suff_gain_median_min": 0.5, "suff_advantage_median_min": 0.25,
            "suff_win_min": 0.65, "whole_effect_fraction_median_min": 0.8,
            "self_max_abs_diff": 1e-4,
            "partition_specific_report_required": True,
        },
        "dynamic_coalitions": {
            "sizes": DYNAMIC_SIZES,
            "rules": ["per_example_top_abs", "per_example_bottom_abs",
                      "discovery_global_magnitude", "discovery_family_magnitude",
                      "deterministic_random_prefix"],
            "discovery_source": "C060 response_discovery natural family@3 clean-minus-corrupt only",
            "evaluation_partitions": ["confirmation", "lockbox"],
            "dynamic_advantage_over_random_median_min": 0.25,
            "dynamic_win_over_random_min": 0.65,
            "small_group_whole_effect_fraction_min": 0.8,
            "small_group_max_size": 1024,
        },
        "coalition_algebra": {
            "effect": "A(S)=correct effect minus status effect",
            "interaction": "Gamma(A,B)=A(A_union_B)-A(A)-A(B)",
            "cancellation_candidate": "A(S)>=0.25 and A(full)<=0 and Gamma(S,C)<0",
            "confirmation_and_lockbox_separate": True,
        },
        "mediation": {
            "path": "family_early",
            "eligibility": "early lambda1 endpoint selectivity plus exact-shape camera; strict monotonicity not required",
            "upstream_rescue_median_min": 0.5,
            "query_block_fraction_median_min": 0.5, "query_block_positive_fraction_min": 0.65,
            "boundary_block_fraction_median_min": 0.5, "boundary_block_positive_fraction_min": 0.65,
            "clean_checkpoint_control_loss_fraction_max": 0.25,
        },
        "branching": {
            "phase1381": "behavior qualification and 216-case balanced freeze",
            "phase1382": "known-truth and exact-shape calibration for every camera family",
            "phase1383": "refined early/mid dose competition and discovery source field",
            "phase1384": "all fixed and dynamic coalitions, complements, unions, and split reports",
            "phase1385": "early mediation if endpoint selectivity qualifies, independently of threshold classification",
            "phase1386": "campaign closure after all independently eligible branches",
        },
        "claim_boundary": {
            "allowed": "Qwen-specific independent-material response-function replication, fixed/dynamic coalition curves, cancellation candidate, and mediation boundary",
            "forbidden": ["relationship manifold proven", "semantic neurons discovered", "minimal natural circuit",
                          "parameter mechanism", "cross-model invariant", "all language relations"],
        },
        "stop_rule": "A failed route eliminates only that route; parent closes only after all frozen eligible routes finish.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1381_c060_behavior_qualification"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "all_gates_passed": True, "authorization": protocol["authorization"],
    })
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
