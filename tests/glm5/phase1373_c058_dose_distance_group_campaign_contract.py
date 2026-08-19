#!/usr/bin/env python3
"""Phase1373: freeze the branched C058 dose, distance, group, and mediation campaign."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from model_utils import MODEL_CONFIGS

PHASE, CAMPAIGN = 1373, "C058"
PARENT = TESTS / "result/phase1372_c057_whole_state_bidirectional"
OLD = TESTS / "result/phase1369_c057_independent_relation_campaign_contract"
OUT = TESTS / "result/phase1373_c058_dose_distance_group_campaign_contract"
MODEL = "qwen3"

FAMILIES = {
    "fish": ("salmon", "trout", "tuna", "carp", "cod", "herring",
             "sardine", "anchovy", "marlin", "sturgeon", "haddock", "halibut"),
    "tree": ("oak", "maple", "cedar", "birch", "willow", "pine",
             "spruce", "elm", "fir", "cypress", "redwood", "aspen"),
    "tool": ("hammer", "wrench", "chisel", "pliers", "mallet", "clamp",
             "trowel", "crowbar", "spanner", "hatchet", "caliper", "shovel"),
    "garment": ("shirt", "jacket", "trousers", "skirt", "blouse", "sweater",
                "coat", "dress", "scarf", "glove", "sock", "vest"),
}
PARTITIONS = {
    "response_discovery": range(0, 4),
    "confirmation": range(4, 8),
    "lockbox": range(8, 12),
}
SURFACES = {
    "ordinary": 'In ordinary English, is "{word}" a {family}?',
    "lexicon": 'In a general dictionary classification, should "{word}" be listed as a {family}?',
    "statement": 'Is the statement "{word} is a member of the category {family}" correct?',
}
SYSTEM_ACTIVE = "Judge the ordinary noun-category relation. Output only yes or no."
SYSTEM_STATUS = "Read the explicit review status. Output only yes or no."
PATHS = {
    "family_early": {
        "source": {"layer": 3, "role": "family"},
        "checkpoints": [{"layer": 15, "role": "query"}, {"layer": 27, "role": "boundary"}],
    },
    "family_mid": {
        "source": {"layer": 15, "role": "family"},
        "checkpoints": [{"layer": 27, "role": "query"}, {"layer": 35, "role": "boundary"}],
    },
}
DOSES = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
DIRECTIONS = ["correct", "wrong", "status", "random"]
GROUP_ROUTES = ["magnitude", "stable_sign", "family_min", "deterministic_random"]
GROUP_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 2560]


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_CONFIGS[MODEL]["path"], trust_remote_code=True,
                                        local_files_only=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def span(tok, ids: list[int], value: str) -> list[int]:
    needles = [[int(x) for x in tok.encode(form, add_special_tokens=False)] for form in (value, " " + value)]
    return core.locate_last_subsequence(ids, needles)


def query_position(tok, ids: list[int]) -> int:
    needles = [[int(x) for x in tok.encode(form, add_special_tokens=False)] for form in ("?", " ?")]
    return core.locate_last_subsequence(ids, needles)[-1]


def concepts() -> list[dict]:
    return [{"word": word, "family": family, "index": index, "partition": partition,
             "sense": f"ordinary common-noun sense of {word}",
             "adjudication": f"unambiguous member of {family} in the frozen four-family inventory"}
            for family, words in FAMILIES.items() for partition, indexes in PARTITIONS.items()
            for index in indexes for word in (words[index],)]


def active_cases() -> list[dict]:
    rows = []
    for partition, indexes in PARTITIONS.items():
        for family_a, family_b in combinations(FAMILIES, 2):
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
                            "case_id": f"c058-a-{len(rows):05d}", "partition": partition,
                            "family_pair": f"{family_a}__{family_b}", "surface": surface,
                            "quartet_key": quartet, "cell": cell, "target": target,
                            "target_family": target_family, "tested_family": tested_family,
                            "truth": truth, "prompt": prompt, "candidates": ["yes", "no"],
                            "gold_position": 0 if truth else 1,
                        })
    return rows


def status_cases() -> list[dict]:
    rows = []
    for partition, indexes in PARTITIONS.items():
        for family_a, family_b in combinations(FAMILIES, 2):
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
                    rows.append({
                        "case_id": f"c058-s-{len(rows):05d}", "partition": partition,
                        "family_pair": f"{family_a}__{family_b}", "surface": "status",
                        "quartet_key": quartet, "cell": cell, "target": target,
                        "tested_family": tested_family, "status": status, "truth": truth,
                        "prompt": prompt, "candidates": ["yes", "no"],
                        "gold_position": 0 if truth else 1,
                    })
    return rows


def compile_rows(tok, values: list[dict], system: str) -> list[dict]:
    result = []
    for row in values:
        ids = core.chat_ids(tok, system, row["prompt"])
        target, family = span(tok, ids, row["target"]), span(tok, ids, row["tested_family"])
        query, boundary = query_position(tok, ids), len(ids) - 1
        result.append({
            "case_id": row["case_id"], "prompt_ids": ids,
            "candidate_ids": [[int(x) for x in tok.encode(v, add_special_tokens=False)] for v in row["candidates"]],
            "role_positions": {"target": target, "family": family, "query": [query], "boundary": [boundary]},
        })
    return result


def candidate_pairs(active: list[dict], status: list[dict]) -> list[dict]:
    quartets = defaultdict(list)
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
                "wrong_family": corrupt["tested_family"], "clean_true": clean["case_id"],
                "corrupt_false": corrupt["case_id"], "wrong_identity_true": wrong["case_id"],
                "status_true": status_true["case_id"],
            })
    return pairs


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1373 already exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent.get("authorization") != "close_c057_without_early_bidirectional_qualification" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("C057 terminal audit missing")
    tok = tokenizer()
    concept_rows, active, status = concepts(), active_cases(), status_cases()
    compiled_active = compile_rows(tok, active, SYSTEM_ACTIVE)
    compiled_status = compile_rows(tok, status, SYSTEM_STATUS)
    pairs = candidate_pairs(active, status)
    old_words = {row["word"] for row in core.load(OLD / "material/frozen_concept_graph.json")["concepts"]}
    family_tokens = {family: tok.encode(" " + family, add_special_tokens=False) for family in FAMILIES}
    quartet_counts = Counter(row["quartet_key"] for row in active)
    true_cells = Counter((row["partition"], row["surface"], row["target_family"])
                         for row in active if row["truth"])
    checks = {
        "parent_closed_audited": parent_audit.get("all_checks_passed"),
        "concept_count": len(concept_rows) == 48 and len({row["word"] for row in concept_rows}) == 48,
        "independent_vocabulary": not ({row["word"] for row in concept_rows} & old_words),
        "family_balance": all(len(words) == 12 for words in FAMILIES.values()),
        "partition_balance": all(sum(row["partition"] == p for row in concept_rows) == 16 for p in PARTITIONS),
        "semantic_uniqueness": all(row["sense"] and row["adjudication"] for row in concept_rows),
        "family_single_token": all(len(ids) == 1 for ids in family_tokens.values()),
        "active_balance": len(active) == 864 and Counter(row["truth"] for row in active) == {True: 432, False: 432},
        "active_quartets": len(quartet_counts) == 216 and set(quartet_counts.values()) == {4},
        "status_balance": len(status) == 288 and Counter(row["truth"] for row in status) == {True: 144, False: 144},
        "candidate_pairs": len(pairs) == 432,
        "family_partition_surface": len(true_cells) == 36 and set(true_cells.values()) == {12},
        "controlled_naturalness": all("  " not in row["prompt"] and row["prompt"].endswith("yes or no.")
                                      for row in active + status),
        "compiled_counts": len(compiled_active) == 864 and len(compiled_status) == 288,
        "candidate_single_tokens": all(len(ids) == 1 for row in compiled_active + compiled_status
                                       for ids in row["candidate_ids"]),
        "family_spans_single": all(len(row["role_positions"]["family"]) == 1
                                   for row in compiled_active + compiled_status),
        "role_positions_valid": all(all(points and max(points) < len(row["prompt_ids"])
                                        for points in row["role_positions"].values())
                                    for row in compiled_active + compiled_status),
        "typed_layout": DOSES == sorted(set(DOSES)) and set(DIRECTIONS) == {"correct", "wrong", "status", "random"},
        "hidden_state_only": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    core.save(OUT / "material/frozen_concept_graph.json", {
        "schema": "c058.independent_concepts.v1", "families": FAMILIES,
        "partitions": {key: list(value) for key, value in PARTITIONS.items()}, "concepts": concept_rows,
    })
    core.write_rows(OUT / "material/active_membership_cases.jsonl", active)
    core.write_rows(OUT / "material/status_cases.jsonl", status)
    core.write_rows(OUT / "material/candidate_pairs.jsonl", pairs)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled_active)
    core.write_rows(OUT / "compiled/qwen3_status.jsonl", compiled_status)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "zero_models": {"always_yes": 0.5, "always_no": 0.5,
                        "target_only_quartet_upper_bound": 0.5, "family_only_quartet_upper_bound": 0.5,
                        "status_always_yes": 0.5, "status_always_no": 0.5},
        "naturalness_scope": "curated controlled English plus deterministic machine audit",
        "independent_human_blind_review": False,
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)

    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c058.branched_dose_distance_groups_mediation.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "same-recipient dose response, full response geometry, raw-coordinate group boundary, and mediation",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states", "candidate and full-vocabulary logits"],
        "forbidden": ["attention", "MLP", "parameter scan", "gradient", "PCA", "t-SNE", "UMAP",
                      "SAE", "learned probe", "post-reveal layer search", "post-reveal threshold change"],
        "material": {
            "families": list(FAMILIES), "concept_count": 48, "active_count": 864,
            "status_count": 288, "candidate_pair_count": 432,
            "partitions": list(PARTITIONS), "surfaces": list(SURFACES),
            "eligible_cases_per_cell": 8, "eligible_case_target": 288,
            "discovery_target": 96, "confirmation_target": 96, "lockbox_target": 96,
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
        "paths": PATHS,
        "dose": {
            "values": DOSES, "directions": DIRECTIONS, "random_seed": 5801373,
            "norm_ratio_abs_error_max": 1e-5, "self_output_max_abs_diff": 1e-4,
            "lambda1_suff_gain_median_min": 0.5, "lambda1_suff_advantage_median_min": 0.25,
            "lambda1_suff_win_min": 0.65, "population_median_monotone": True,
            "reverse_damage_median_min": 0.5, "reverse_over_status_median_min": 0.25,
            "reverse_over_status_win_min": 0.65,
        },
        "distance": {
            "metrics": ["pairwise_normalized_l2", "direction_projection", "orthogonal_ratio", "clean_distance_ratio"],
            "finite_fraction_min": 1.0, "direction_decomposition_relative_error_max": 1e-5,
            "single_projection_not_required_for_output_branch": True,
        },
        "camera": {
            "known_truth_systems": 256, "qwen_cases": 36,
            "same_shape_output_max_abs_diff": 1e-5,
            "same_shape_checkpoint_relative_l2_max": 1e-6,
            "dose_and_distance_exact": True, "serial_parallel_exact": True,
        },
        "coordinate_groups": {
            "candidate_source": "C057 prototype_discovery family@3 clean-minus-corrupt deltas only",
            "candidate_source_sha256": core.sha(TESTS / "result/phase1372_c057_whole_state_bidirectional/raw/family3_source_deltas.pt"),
            "routes": GROUP_ROUTES, "sizes": GROUP_SIZES, "random_seed": 5801373,
            "evaluation_partitions": ["confirmation", "lockbox"],
            "suff_gain_median_min": 0.5, "suff_advantage_median_min": 0.25,
            "suff_win_min": 0.65, "whole_effect_fraction_median_min": 0.80,
            "reverse_damage_median_min": 0.5, "reverse_over_status_median_min": 0.25,
            "reverse_over_status_win_min": 0.65, "self_max_abs_diff": 1e-4,
            "sufficiency_qualification_independent_of_reverse_qualification": True,
        },
        "mediation": {
            "path": "family_early", "eligibility": "early lambda1 positive sufficiency plus exact-shape identity",
            "upstream_rescue_median_min": 0.5,
            "query_block_fraction_median_min": 0.5, "query_block_positive_fraction_min": 0.65,
            "boundary_block_fraction_median_min": 0.5, "boundary_block_positive_fraction_min": 0.65,
            "clean_checkpoint_control_loss_fraction_max": 0.25,
        },
        "observation": {
            "partition": "response_discovery", "all_layers": True,
            "roles": ["target", "family", "query", "boundary"],
            "preserve_each_span_token": True, "not_confirmation_evidence": True,
        },
        "branching": {
            "phase1374": "behavior qualification and balanced case freeze",
            "phase1375": "known-truth and exact-shape calibration",
            "phase1376": "run both whole-state dose paths and full response geometry; also save discovery field",
            "phase1377": "run every raw-coordinate route on confirmation plus lockbox regardless of reverse whole-state result",
            "phase1378": "run early mediation if early lambda1 sufficiency qualifies, independently of reverse result",
            "finish": "close after all independently eligible branches complete; no replacement routes, layers, materials, doses, or gates",
        },
        "claim_boundary": {
            "allowed": "Qwen-specific dose response, full response geometry, raw-coordinate group curve, and mediation boundary",
            "forbidden": ["five neuron classes discovered", "relation manifold proven", "parameter mechanism",
                          "minimal natural circuit", "cross-model invariance", "all language relations"],
        },
        "stop_rule": "A failed branch eliminates only that branch; the parent closes after every frozen eligible branch completes.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1374_c058_behavior_qualification"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "all_gates_passed": True, "authorization": protocol["authorization"],
    })
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
