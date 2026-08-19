#!/usr/bin/env python3
"""Phase1387: freeze the C061 full-field and coordinate-transfer campaign."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1373_c058_dose_distance_group_campaign_contract as base

PHASE, CAMPAIGN = 1387, "C061"
PARENT = TESTS / "result/phase1386_c060_campaign_closure"
C060_CONTRACT = TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract"
C060_RANKINGS = TESTS / "result/phase1384_c060_fixed_dynamic_coalitions/protocol/discovery_rankings.json"
OUT = TESTS / "result/phase1387_c061_full_field_transfer_campaign_contract"

TRANSFER_FAMILIES = {
    "animal": ("rabbit", "horse", "dolphin", "eagle", "frog", "whale",
               "shark", "deer", "elephant", "monkey", "owl", "bear"),
    "beverage": ("punch", "vodka", "whiskey", "rum", "gin", "champagne",
                 "cocktail", "tonic", "ale", "liquor", "brew", "sake"),
    "sport": ("cricket", "swimming", "running", "fencing", "wrestling", "skating",
              "diving", "sailing", "racing", "bowling", "softball", "squash"),
    "building": ("apartment", "office", "clinic", "courthouse", "theater", "temple",
                 "prison", "mansion", "cottage", "garage", "airport", "palace"),
}
NOVEL_FAMILIES = {
    "appliance": ("refrigerator", "oven", "toaster", "blender", "freezer", "dishwasher",
                  "microwave", "kettle", "vacuum", "dryer", "washer", "heater"),
    "vehicle": ("car", "truck", "bus", "bicycle", "scooter", "tractor",
                "van", "taxi", "motorcycle", "ambulance", "subway", "train"),
    "profession": ("doctor", "lawyer", "teacher", "nurse", "plumber", "dentist",
                   "pilot", "farmer", "chef", "architect", "engineer", "accountant"),
    "furniture": ("chair", "table", "sofa", "desk", "cabinet", "dresser",
                  "stool", "couch", "bench", "wardrobe", "bed", "shelf"),
}
FAMILIES = {**TRANSFER_FAMILIES, **NOVEL_FAMILIES}
FAMILY_EDGES = (
    ("animal", "beverage"), ("beverage", "sport"),
    ("sport", "building"), ("building", "animal"),
    ("appliance", "vehicle"), ("vehicle", "profession"),
    ("profession", "furniture"), ("furniture", "appliance"),
    ("animal", "appliance"), ("beverage", "vehicle"),
    ("sport", "profession"), ("building", "furniture"),
)
SURFACES = {
    "ordinary": "In ordinary English, is the noun {word} a {family}?",
    "lexicon": "In a general dictionary, should the noun {word} be listed as a {family}?",
    "statement": "Is the claim that the noun {word} is a member of the category {family} correct?",
}
COORDINATE_SIZES = (64, 128, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048, 2560)
STAGE_WINDOWS = ((4, 11), (12, 19), (20, 27), (28, 35))


def active_cases() -> list[dict]:
    rows = []
    for partition, indexes in base.PARTITIONS.items():
        for family_a, family_b in FAMILY_EDGES:
            for index in indexes:
                word_a, word_b = FAMILIES[family_a][index], FAMILIES[family_b][index]
                for surface, template in SURFACES.items():
                    quartet = f"{partition}:{family_a}__{family_b}:{index}:{surface}"
                    cells = (("aa", word_a, family_a, family_a, True),
                             ("ab", word_a, family_a, family_b, False),
                             ("ba", word_b, family_b, family_a, False),
                             ("bb", word_b, family_b, family_b, True))
                    for cell, target, target_family, tested_family, truth in cells:
                        prompt = template.format(word=target, family=tested_family) + " Output only yes or no."
                        rows.append({
                            "case_id": f"c061-a-{len(rows):05d}", "partition": partition,
                            "family_pair": f"{family_a}__{family_b}", "surface": surface,
                            "quartet_key": quartet, "cell": cell, "target": target,
                            "target_family": target_family, "tested_family": tested_family,
                            "panel": "transfer" if target_family in TRANSFER_FAMILIES else "novel",
                            "truth": truth, "prompt": prompt, "candidates": ["yes", "no"],
                            "gold_position": 0 if truth else 1,
                        })
    return rows


def status_cases() -> list[dict]:
    rows = []
    for partition, indexes in base.PARTITIONS.items():
        for family_a, family_b in FAMILY_EDGES:
            for index in indexes:
                word_a, word_b = FAMILIES[family_a][index], FAMILIES[family_b][index]
                quartet = f"{partition}:{family_a}__{family_b}:{index}:status"
                cells = (("aa", word_a, family_a, "approved", True),
                         ("ab", word_a, family_b, "rejected", False),
                         ("ba", word_b, family_a, "rejected", False),
                         ("bb", word_b, family_b, "approved", True))
                for cell, target, tested_family, status, truth in cells:
                    prompt = (f'Review record: "{target}"; proposed category: {tested_family}; '
                              f"status: {status}. Is the status approved? Output only yes or no.")
                    target_family = family_a if target == word_a else family_b
                    rows.append({
                        "case_id": f"c061-s-{len(rows):05d}", "partition": partition,
                        "family_pair": f"{family_a}__{family_b}", "surface": "status",
                        "quartet_key": quartet, "cell": cell, "target": target,
                        "target_family": target_family, "tested_family": tested_family,
                        "panel": "transfer" if target_family in TRANSFER_FAMILIES else "novel",
                        "status": status, "truth": truth, "prompt": prompt,
                        "candidates": ["yes", "no"], "gold_position": 0 if truth else 1,
                    })
    return rows


def candidate_pairs(active: list[dict], status: list[dict]) -> list[dict]:
    quartets: dict[str, list[dict]] = defaultdict(list)
    for row in active:
        quartets[row["quartet_key"]].append(row)
    status_by = {(row["quartet_key"], row["cell"]): row for row in status}
    pairs = []
    for key, values in sorted(quartets.items()):
        cells = {row["cell"]: row for row in values}
        status_key = key.rsplit(":", 1)[0] + ":status"
        for direction, clean_cell, corrupt_cell, wrong_cell in (
                ("a_to_b", "aa", "ab", "bb"), ("b_to_a", "bb", "ba", "aa")):
            clean, corrupt, wrong = cells[clean_cell], cells[corrupt_cell], cells[wrong_cell]
            status_true = status_by[(status_key, wrong_cell)]
            pairs.append({
                "pair_id": f"{key}:{direction}", "partition": clean["partition"],
                "surface": clean["surface"], "family_pair": clean["family_pair"],
                "direction": direction, "target": clean["target"],
                "target_family": clean["target_family"], "tested_family": clean["tested_family"],
                "wrong_family": corrupt["tested_family"], "panel": clean["panel"],
                "clean_true": clean["case_id"], "corrupt_false": corrupt["case_id"],
                "wrong_identity_true": wrong["case_id"], "status_true": status_true["case_id"],
            })
    return pairs


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1387 already exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent.get("status") != "closed_after_all_frozen_eligible_routes" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("C060 audited closure missing")
    if not C060_RANKINGS.exists():
        raise RuntimeError("C060 frozen rankings missing")

    base.FAMILIES = FAMILIES
    base.SURFACES = SURFACES
    tok = base.tokenizer()
    concepts = base.concepts()
    active, status = active_cases(), status_cases()
    compiled_active = base.compile_rows(tok, active, base.SYSTEM_ACTIVE)
    compiled_status = base.compile_rows(tok, status, base.SYSTEM_STATUS)
    pairs = candidate_pairs(active, status)
    c060_graph = core.load(C060_CONTRACT / "material/frozen_concept_graph.json")
    c060_words = {r["word"] for r in c060_graph["concepts"]}
    c060_rankings = core.load(C060_RANKINGS)
    family_tokens = {f: tok.encode(" " + f, add_special_tokens=False) for f in FAMILIES}
    target_tokens = {w: tok.encode(" " + w, add_special_tokens=False)
                     for words in FAMILIES.values() for w in words}
    by_surface: dict[str, list[dict]] = defaultdict(list)
    source_by_id = {r["case_id"]: r for r in active}
    for row in compiled_active:
        by_surface[source_by_id[row["case_id"]]["surface"]].append(row)
    layouts = {
        surface: {
            "length": len(rows[0]["prompt_ids"]),
            "role_positions": rows[0]["role_positions"],
        }
        for surface, rows in by_surface.items()
    }
    layout_exact = all(
        len(row["prompt_ids"]) == layouts[surface]["length"]
        and row["role_positions"] == layouts[surface]["role_positions"]
        for surface, rows in by_surface.items() for row in rows
    )
    family_degree = Counter(f for edge in FAMILY_EDGES for f in edge)
    checks = {
        "parent_closed_audited": bool(parent_audit.get("all_checks_passed")),
        "family_count_balance": len(FAMILIES) == 8 and all(len(v) == 12 for v in FAMILIES.values()),
        "panel_balance": len(TRANSFER_FAMILIES) == len(NOVEL_FAMILIES) == 4,
        "family_graph_regular": len(FAMILY_EDGES) == 12 and set(family_degree.values()) == {3},
        "c060_vocabulary_independent": not ({w for values in TRANSFER_FAMILIES.values() for w in values} & c060_words),
        "all_vocabulary_independent_of_c060": not ({w for values in FAMILIES.values() for w in values} & c060_words),
        "semantic_uniqueness": all(r["sense"] and r["adjudication"] for r in concepts),
        "family_single_token": all(len(v) == 1 for v in family_tokens.values()),
        "target_single_token": all(len(v) == 1 for v in target_tokens.values()),
        "active_balance": len(active) == 1728 and Counter(r["truth"] for r in active) == {True: 864, False: 864},
        "status_balance": len(status) == 576 and Counter(r["truth"] for r in status) == {True: 288, False: 288},
        "pair_count": len(pairs) == 864,
        "panel_pair_balance": Counter(r["panel"] for r in pairs) == {"transfer": 432, "novel": 432},
        "controlled_naturalness": all("  " not in r["prompt"] and r["prompt"].endswith("yes or no.")
                                      for r in active + status),
        "compiled_counts": len(compiled_active) == 1728 and len(compiled_status) == 576,
        "candidate_single_tokens": all(len(ids) == 1 for r in compiled_active + compiled_status for ids in r["candidate_ids"]),
        "typed_roles": all(set(r["role_positions"]) == {"target", "family", "query", "boundary"}
                           for r in compiled_active + compiled_status),
        "surface_physical_layout_exact": layout_exact,
        "c060_rankings_complete": set(c060_rankings["families"]) == set(TRANSFER_FAMILIES)
                                  and len(c060_rankings["global"]) == 2560,
        "hidden_state_only": True,
    }
    if not all(checks.values()):
        raise RuntimeError({k: v for k, v in checks.items() if not v})

    core.save(OUT / "material/frozen_concept_graph.json", {
        "schema": "c061.full_field_transfer.v1", "families": FAMILIES,
        "transfer_families": list(TRANSFER_FAMILIES), "novel_families": list(NOVEL_FAMILIES),
        "family_edges": [list(v) for v in FAMILY_EDGES],
        "partitions": {k: list(v) for k, v in base.PARTITIONS.items()}, "concepts": concepts,
    })
    core.write_rows(OUT / "material/active_membership_cases.jsonl", active)
    core.write_rows(OUT / "material/status_cases.jsonl", status)
    core.write_rows(OUT / "material/candidate_pairs.jsonl", pairs)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled_active)
    core.write_rows(OUT / "compiled/qwen3_status.jsonl", compiled_status)
    core.save(OUT / "protocol/surface_layouts.json", layouts)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "zero_models": {"always_yes": 0.5, "always_no": 0.5,
                        "target_only_quartet_upper_bound": 0.5, "family_only_quartet_upper_bound": 0.5,
                        "status_always_yes": 0.5, "status_always_no": 0.5},
        "semantic_scope": "ordinary mutually exclusive noun-family senses in the frozen inventory",
        "naturalness_scope": "curated controlled English plus deterministic machine audit",
        "independent_human_blind_review": False,
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)

    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c061.full_field_transfer_campaign.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "full hidden-state response field, C060 coordinate transfer, C061 rediscovery, and downstream event mediation",
        "allowed_observables": ["input token embeddings", "all layers and all token positions of full-dimensional hidden states",
                                "candidate and full-vocabulary logits"],
        "forbidden": ["attention", "MLP", "parameter scan", "gradient", "PCA", "t-SNE", "UMAP",
                      "SAE", "learned probe", "post-reveal material/layer/threshold/route replacement"],
        "material": {
            "families": list(FAMILIES), "transfer_families": list(TRANSFER_FAMILIES),
            "novel_families": list(NOVEL_FAMILIES), "concept_count": 96,
            "active_count": 1728, "status_count": 576, "candidate_pair_count": 864,
            "partitions": list(base.PARTITIONS), "surfaces": list(SURFACES),
            "eligible_cases_per_cell": 4, "eligible_case_target": 288,
            "discovery_target": 96, "confirmation_target": 96, "lockbox_target": 96,
            "active_sha256": core.sha(OUT / "material/active_membership_cases.jsonl"),
            "status_sha256": core.sha(OUT / "material/status_cases.jsonl"),
            "pair_sha256": core.sha(OUT / "material/candidate_pairs.jsonl"),
            "human_naturalness_lock": False,
        },
        "behavior": {
            "active_accuracy_min": 0.90, "partition_min": 0.85, "surface_min": 0.85,
            "family_min": 0.85, "panel_min": 0.85, "truth_min": 0.85, "quartet_all_min": 0.75,
            "status_accuracy_min": 0.95, "status_partition_min": 0.90,
            "same_shape_repeat_max_abs_diff": 1e-6,
            "eligibility": "clean, corrupt, wrong-identity, and status donors individually behavior-correct",
        },
        "source": {"layer": 3, "role": "family"},
        "camera": {
            "known_truth_systems": 256, "qwen_cases": 24,
            "zero_write_output_max_abs_diff": 1e-5,
            "zero_write_all_state_relative_l2_max": 1e-6,
            "coordinate_mask_exact": True, "full_position_field_exact": True,
            "multi_checkpoint_reset_exact": True,
        },
        "observation": {
            "partition": "response_discovery", "case_count": 96,
            "all_hidden_state_indices": list(range(37)), "all_physical_positions": True,
            "interventions": ["self", "correct", "wrong", "status"],
            "event_strength_ratio_min": 0.01, "event_selectivity_median_min": 0.05,
            "event_selectivity_win_min": 0.65,
            "stage_windows": [list(v) for v in STAGE_WINDOWS], "top_events_per_stage_surface": 2,
            "candidate_source_is_discovery_only": True,
        },
        "coordinates": {
            "sizes": list(COORDINATE_SIZES),
            "routes": ["c060_family_fixed", "c060_global_fixed", "c061_family_discovery",
                       "c061_global_discovery", "per_example_top_abs", "per_example_bottom_abs",
                       "deterministic_random"],
            "primary_size": 512, "evaluation_partitions": ["confirmation", "lockbox"],
            "suff_gain_median_min": 0.5, "suff_advantage_median_min": 0.25,
            "suff_win_min": 0.65, "whole_effect_fraction_median_min": 0.80,
            "reverse_damage_median_min": 0.5, "reverse_over_status_median_min": 0.25,
            "reverse_over_status_win_min": 0.65, "self_max_abs_diff": 1e-4,
            "same_family_transfer_primary": "c060_family_fixed@512",
            "novel_family_rediscovery_primary": "c061_family_discovery@512",
            "overlap_intersection_fraction_enrichment_min": 0.10,
            "c060_rankings_sha256": core.sha(C060_RANKINGS),
        },
        "mediation": {
            "candidate_source": "discovery-only full-position response field",
            "bundles": ["top1", "stage_top1", "stage_top2", "query_reference", "boundary_reference"],
            "evaluation_partitions": ["confirmation", "lockbox"],
            "upstream_rescue_median_min": 0.5,
            "block_fraction_median_min": 0.5, "block_positive_fraction_min": 0.65,
            "clean_checkpoint_control_loss_fraction_max": 0.25,
            "reference_query": {"state_index": 15, "role": "query"},
            "reference_boundary": {"state_index": 27, "role": "boundary"},
        },
        "branching": {
            "phase1388": "behavior qualification and balanced 288-case freeze",
            "phase1389": "known-truth and Qwen full-field/mask/reset camera calibration",
            "phase1390": "discovery-only all-layer all-position field and frozen candidates",
            "phase1391": "all coordinate routes and sizes on confirmation plus lockbox",
            "phase1392": "all discovery event bundles and fixed reference checkpoints on confirmation plus lockbox",
            "phase1393": "independent campaign closure after every eligible route",
        },
        "claim_boundary": {
            "allowed": "Qwen-specific cross-material coordinate transfer/rediscovery curves and operational event-reset mediation",
            "forbidden": ["fixed relation vector", "semantic neurons", "minimal or unique natural circuit",
                          "attention/MLP/parameter mechanism", "cross-model invariant", "open-language mechanism"],
        },
        "stop_rule": "Failure eliminates only that frozen route; camera failure blocks dependent hidden-state routes; campaign closes after all independently eligible routes.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1388_c061_behavior_qualification"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "all_gates_passed": True, "authorization": protocol["authorization"],
    })
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
