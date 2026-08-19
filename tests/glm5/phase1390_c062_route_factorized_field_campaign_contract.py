#!/usr/bin/env python3
"""Phase1390: freeze C062 with family-factorized behavior authorization."""
from __future__ import annotations

import json, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1373_c058_dose_distance_group_campaign_contract as base
import phase1387_c061_full_field_transfer_campaign_contract as c061

PHASE, CAMPAIGN = 1390, "C062"
PARENT = TESTS / "result/phase1389_c061_behavior_gate_closure"
C060_RANKINGS = TESTS / "result/phase1384_c060_fixed_dynamic_coalitions/protocol/discovery_rankings.json"
C060_GRAPH = TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract/material/frozen_concept_graph.json"
C061_GRAPH = TESTS / "result/phase1387_c061_full_field_transfer_campaign_contract/material/frozen_concept_graph.json"
OUT = TESTS / "result/phase1390_c062_route_factorized_field_campaign_contract"

TRANSFER_FAMILIES = {
    "animal": ("lion", "tiger", "wolf", "fox", "cow", "pig", "goat", "sheep", "mouse", "rat", "camel", "panda"),
    "building": ("chapel", "barn", "cabin", "hut", "mosque", "cathedral", "fortress", "shrine",
                 "monastery", "convent", "barracks", "greenhouse"),
}
NOVEL_FAMILIES = {
    "appliance": ("stove", "fan", "iron", "mixer", "grill", "radio", "television", "lamp",
                  "furnace", "boiler", "cooler", "processor"),
    "vehicle": ("ship", "boat", "ferry", "airplane", "helicopter", "wagon", "jeep", "tram",
                "rocket", "canoe", "kayak", "yacht"),
    "profession": ("surgeon", "scientist", "writer", "artist", "musician", "banker", "clerk", "manager",
                   "mechanic", "librarian", "journalist", "professor"),
    "country": ("France", "Germany", "Italy", "Spain", "Canada", "Mexico", "Brazil", "India",
                "China", "Japan", "Russia", "Egypt"),
}
FAMILIES = {**TRANSFER_FAMILIES, **NOVEL_FAMILIES}
FAMILY_EDGES = (("animal", "building"), ("building", "appliance"), ("appliance", "vehicle"),
                ("vehicle", "profession"), ("profession", "country"), ("country", "animal"),
                ("animal", "vehicle"), ("building", "profession"), ("appliance", "country"))
SURFACES = {
    "ordinary": "In ordinary English classification, does {word} belong to the noun category {family}?",
    "lexicon": "Using standard dictionary meanings, is {word} a member of the category {family}?",
    "statement": "Evaluate this category claim: {word} belongs to {family}. Is the claim true?",
}
SYSTEM_ACTIVE = "Use standard ordinary meanings. Decide only noun-category membership. Output only yes or no."
SYSTEM_STATUS = base.SYSTEM_STATUS
COORDINATE_SIZES = c061.COORDINATE_SIZES
STAGE_WINDOWS = c061.STAGE_WINDOWS


def relabel(rows: list[dict]) -> list[dict]:
    for row in rows: row["case_id"] = row["case_id"].replace("c061", "c062")
    return rows


def main() -> None:
    if (OUT / "analysis/final.json").exists(): raise RuntimeError("Phase1390 already exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c062_route_factorized_behavior_and_hidden_campaign" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("C061 closure missing")
    c061.FAMILIES, c061.TRANSFER_FAMILIES, c061.NOVEL_FAMILIES = FAMILIES, TRANSFER_FAMILIES, NOVEL_FAMILIES
    c061.FAMILY_EDGES, c061.SURFACES = FAMILY_EDGES, SURFACES
    base.FAMILIES, base.SURFACES = FAMILIES, SURFACES
    tok = base.tokenizer()
    concepts = base.concepts()
    active = relabel(c061.active_cases())
    status = relabel(c061.status_cases())
    compiled_active = base.compile_rows(tok, active, SYSTEM_ACTIVE)
    compiled_status = base.compile_rows(tok, status, SYSTEM_STATUS)
    pairs = c061.candidate_pairs(active, status)
    old_words = {r["word"] for p in (C060_GRAPH, C061_GRAPH) for r in core.load(p)["concepts"]}
    family_tokens = {f: tok.encode(" " + f, add_special_tokens=False) for f in FAMILIES}
    target_tokens = {w: tok.encode(" " + w, add_special_tokens=False) for words in FAMILIES.values() for w in words}
    source = {r["case_id"]: r for r in active}
    by_surface: dict[str, list[dict]] = defaultdict(list)
    for row in compiled_active: by_surface[source[row["case_id"]]["surface"]].append(row)
    layouts = {s: {"length": len(rows[0]["prompt_ids"]), "role_positions": rows[0]["role_positions"]}
               for s, rows in by_surface.items()}
    exact_layout = all(len(r["prompt_ids"]) == layouts[s]["length"] and r["role_positions"] == layouts[s]["role_positions"]
                       for s, rows in by_surface.items() for r in rows)
    degree = Counter(f for e in FAMILY_EDGES for f in e)
    checks = {
        "parent_closed_audited": parent_audit["all_checks_passed"],
        "six_balanced_families": len(FAMILIES) == 6 and all(len(v) == 12 for v in FAMILIES.values()),
        "panel_design": len(TRANSFER_FAMILIES) == 2 and len(NOVEL_FAMILIES) == 4,
        "regular_graph": len(FAMILY_EDGES) == 9 and set(degree.values()) == {3},
        "vocabulary_independent_c060_c061": not ({w for v in FAMILIES.values() for w in v} & old_words),
        "family_single_token": all(len(v) == 1 for v in family_tokens.values()),
        "target_single_token": all(len(v) == 1 for v in target_tokens.values()),
        "active_balance": len(active) == 1296 and Counter(r["truth"] for r in active) == {True: 648, False: 648},
        "status_balance": len(status) == 432 and Counter(r["truth"] for r in status) == {True: 216, False: 216},
        "pair_count": len(pairs) == 648,
        "compiled": len(compiled_active) == 1296 and len(compiled_status) == 432,
        "candidate_single_token": all(len(ids) == 1 for r in compiled_active + compiled_status for ids in r["candidate_ids"]),
        "typed_roles": all(set(r["role_positions"]) == {"target", "family", "query", "boundary"}
                           for r in compiled_active + compiled_status),
        "surface_layout_exact": exact_layout,
        "controlled_naturalness": all("  " not in r["prompt"] and r["prompt"].endswith("yes or no.")
                                      for r in active + status),
        "hidden_state_only": True,
    }
    if not all(checks.values()): raise RuntimeError({k: v for k, v in checks.items() if not v})
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c062.route_factorized.v1",
              "families": FAMILIES, "transfer_families": list(TRANSFER_FAMILIES),
              "novel_families": list(NOVEL_FAMILIES), "family_edges": [list(v) for v in FAMILY_EDGES],
              "partitions": {k: list(v) for k, v in base.PARTITIONS.items()}, "concepts": concepts})
    core.write_rows(OUT / "material/active_membership_cases.jsonl", active)
    core.write_rows(OUT / "material/status_cases.jsonl", status)
    core.write_rows(OUT / "material/candidate_pairs.jsonl", pairs)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled_active)
    core.write_rows(OUT / "compiled/qwen3_status.jsonl", compiled_status)
    core.save(OUT / "protocol/surface_layouts.json", layouts)
    pre = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()),
           "total": len(checks), "all_checks_passed": all(checks.values()),
           "zero_models": {"always_yes": 0.5, "always_no": 0.5, "target_only_quartet_upper_bound": 0.5,
                           "family_only_quartet_upper_bound": 0.5, "status_always_yes": 0.5},
           "semantic_scope": "frozen standard senses; route-level behavior additionally tests construct viability",
           "naturalness_scope": "curated controlled English plus deterministic machine audit",
           "known_ambiguity_risk": ["appliance has polysemous members fan/iron/radio/processor"],
           "independent_human_blind_review": False}
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", pre)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c062.route_factorized_field.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "family-route-factorized full hidden field, coordinate transfer/rediscovery, and event mediation",
        "allowed_observables": ["input token embeddings", "all layers/all positions full-dimensional hidden states", "logits"],
        "forbidden": ["attention", "MLP", "parameter scan", "gradient", "PCA", "t-SNE", "UMAP", "SAE", "learned probe",
                      "post-reveal word/family/surface/layer/threshold/route replacement"],
        "material": {"families": list(FAMILIES), "transfer_families": list(TRANSFER_FAMILIES),
                     "novel_families": list(NOVEL_FAMILIES), "active_count": 1296, "status_count": 432,
                     "pair_count": 648, "partitions": list(base.PARTITIONS), "surfaces": list(SURFACES),
                     "eligible_cases_per_family_cell": 6, "selected_per_family": 54,
                     "selected_per_family_partition": 18, "minimum_qualified_families": 4,
                     "minimum_qualified_transfer_families": 1, "minimum_qualified_novel_families": 2,
                     "active_sha256": core.sha(OUT / "material/active_membership_cases.jsonl"),
                     "status_sha256": core.sha(OUT / "material/status_cases.jsonl"),
                     "pair_sha256": core.sha(OUT / "material/candidate_pairs.jsonl"), "human_naturalness_lock": False},
        "behavior": {"family_active_accuracy_min": 0.85, "family_partition_min": 0.80,
                     "family_surface_min": 0.80, "family_true_min": 0.80, "family_false_min": 0.90,
                     "family_quartet_all_min": 0.70, "status_accuracy_min": 0.95,
                     "same_shape_repeat_max_abs_diff": 1e-6,
                     "route_rule": "qualify or eliminate each family independently before applying frozen breadth gate"},
        "source": {"layer": 3, "role": "family"},
        "camera": {"known_truth_systems": 256, "qwen_cases": 24, "zero_write_output_max_abs_diff": 1e-5,
                   "zero_write_all_state_relative_l2_max": 1e-6, "coordinate_mask_exact": True,
                   "full_position_field_exact": True, "multi_checkpoint_reset_exact": True},
        "observation": {"partition": "response_discovery", "all_hidden_state_indices": list(range(37)),
                        "all_physical_positions": True, "interventions": ["self", "correct", "wrong", "status"],
                        "event_strength_ratio_min": 0.01, "event_selectivity_median_min": 0.05,
                        "event_selectivity_win_min": 0.65, "stage_windows": [list(v) for v in STAGE_WINDOWS],
                        "top_events_per_stage_surface": 2, "candidate_source_is_discovery_only": True},
        "coordinates": {"sizes": list(COORDINATE_SIZES),
                        "routes": ["c060_family_fixed", "c060_global_fixed", "c062_family_discovery",
                                   "c062_global_discovery", "per_example_top_abs", "per_example_bottom_abs", "deterministic_random"],
                        "primary_size": 512, "evaluation_partitions": ["confirmation", "lockbox"],
                        "suff_gain_median_min": 0.5, "suff_advantage_median_min": 0.25, "suff_win_min": 0.65,
                        "whole_effect_fraction_median_min": 0.80, "reverse_damage_median_min": 0.5,
                        "reverse_over_status_median_min": 0.25, "reverse_over_status_win_min": 0.65,
                        "self_max_abs_diff": 1e-4, "same_family_transfer_primary": "c060_family_fixed@512",
                        "rediscovery_primary": "c062_family_discovery@512",
                        "c060_rankings_sha256": core.sha(C060_RANKINGS)},
        "mediation": {"bundles": ["top1", "stage_top1", "stage_top2", "query_reference", "boundary_reference"],
                      "evaluation_partitions": ["confirmation", "lockbox"], "upstream_rescue_median_min": 0.5,
                      "block_fraction_median_min": 0.5, "block_positive_fraction_min": 0.65,
                      "clean_checkpoint_control_loss_fraction_max": 0.25,
                      "reference_query": {"state_index": 15, "role": "query"},
                      "reference_boundary": {"state_index": 27, "role": "boundary"}},
        "branching": {"phase1391": "family-factorized behavior and breadth gate", "phase1392": "camera calibration",
                      "phase1393": "discovery full field and candidate freeze", "phase1394": "coordinate curves",
                      "phase1395": "event bundle mediation", "phase1396": "campaign closure"},
        "claim_boundary": {"allowed": "Qwen-specific qualified-family hidden response and operational intervention results",
                           "forbidden": ["semantic neurons", "fixed relation vector", "minimal/unique natural circuit",
                                         "attention/MLP/parameter mechanism", "cross-model or open-language law"]},
        "stop_rule": "A family or downstream route failure eliminates only that route; breadth/camera failure blocks dependent hidden access.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1391_c062_family_factorized_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN,
              "contract_sha256": protocol["contract_sha256"], "all_gates_passed": True,
              "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": pre, "protocol": protocol}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
