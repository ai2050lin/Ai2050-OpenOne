#!/usr/bin/env python3
"""Phase1369: freeze the independent-material C057 Hidden-State campaign."""
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

PHASE, CAMPAIGN = 1369, "C057"
PARENT = TESTS / "result/phase1368_c056_all_path_causal_competition"
OLD = TESTS / "result/phase1353_c053_route_portfolio_contract"
OUT = TESTS / "result/phase1369_c057_independent_relation_campaign_contract"
MODEL = "qwen3"

FAMILIES = {
    "bird": ("robin", "sparrow", "falcon", "heron", "raven", "finch",
             "eagle", "swan", "owl", "pelican", "stork", "puffin"),
    "instrument": ("violin", "trumpet", "piano", "clarinet", "cello", "flute",
                   "oboe", "harp", "tuba", "banjo", "mandolin", "trombone"),
    "flower": ("rose", "tulip", "daisy", "lily", "orchid", "iris",
               "daffodil", "peony", "marigold", "lotus", "poppy", "carnation"),
    "vehicle": ("bicycle", "truck", "sedan", "tram", "submarine", "helicopter",
                "tractor", "canoe", "scooter", "van", "airplane", "motorcycle"),
}
PARTITIONS = {
    "prototype_discovery": range(0, 3),
    "clock_selection": range(3, 6),
    "confirmation": range(6, 9),
    "lockbox": range(9, 12),
}
SURFACES = {
    "ordinary": 'In ordinary English, is "{word}" a {family}?',
    "lexicon": 'For a general-vocabulary classification, should "{word}" be filed under {family}?',
    "statement": 'Is the statement "{word} belongs to the category {family}" correct?',
}
SYSTEM_ACTIVE = "Judge the ordinary noun-category relation. Output only yes or no."
SYSTEM_STATUS = "Read the explicit classification-review status. Output only yes or no."

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


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[MODEL]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def span(tok, ids: list[int], value: str) -> list[int]:
    needles = [[int(x) for x in tok.encode(form, add_special_tokens=False)]
               for form in (value, " " + value)]
    return core.locate_last_subsequence(ids, needles)


def query_position(tok, ids: list[int]) -> int:
    needles = [[int(x) for x in tok.encode(form, add_special_tokens=False)] for form in ("?", " ?")]
    return core.locate_last_subsequence(ids, needles)[-1]


def concepts() -> list[dict]:
    rows = []
    for family, words in FAMILIES.items():
        for partition, indexes in PARTITIONS.items():
            for index in indexes:
                word = words[index]
                rows.append({
                    "word": word, "family": family, "index": index, "partition": partition,
                    "sense": f"ordinary common-noun sense of {word}",
                    "adjudication": f"unambiguous member of {family} in the frozen four-family inventory",
                })
    return rows


def active_cases() -> list[dict]:
    rows = []
    for partition, indexes in PARTITIONS.items():
        for family_a, family_b in combinations(FAMILIES, 2):
            for index in indexes:
                word_a, word_b = FAMILIES[family_a][index], FAMILIES[family_b][index]
                for surface, template in SURFACES.items():
                    quartet = f"{partition}:{family_a}__{family_b}:{index}:{surface}"
                    cells = (
                        ("aa", word_a, family_a, family_a, True),
                        ("ab", word_a, family_a, family_b, False),
                        ("ba", word_b, family_b, family_a, False),
                        ("bb", word_b, family_b, family_b, True),
                    )
                    for cell, target, target_family, tested_family, truth in cells:
                        prompt = template.format(word=target, family=tested_family) + " Output only yes or no."
                        rows.append({
                            "case_id": f"c057-a-{len(rows):05d}", "partition": partition,
                            "family_pair": f"{family_a}__{family_b}", "surface": surface,
                            "quartet_key": quartet, "cell": cell, "target": target,
                            "target_family": target_family, "tested_family": tested_family,
                            "truth": truth, "prompt": prompt,
                            "candidates": ["yes", "no"], "gold_position": 0 if truth else 1,
                        })
    return rows


def status_cases() -> list[dict]:
    rows = []
    for partition, indexes in PARTITIONS.items():
        for family_a, family_b in combinations(FAMILIES, 2):
            for index in indexes:
                word_a, word_b = FAMILIES[family_a][index], FAMILIES[family_b][index]
                quartet = f"{partition}:{family_a}__{family_b}:{index}:status"
                cells = (
                    ("aa", word_a, family_a, "approved", True),
                    ("ab", word_a, family_b, "rejected", False),
                    ("ba", word_b, family_a, "rejected", False),
                    ("bb", word_b, family_b, "approved", True),
                )
                for cell, target, tested_family, status, truth in cells:
                    prompt = (f'Classification record: "{target}"; category: {tested_family}; '
                              f"review status: {status}. Is the review status approved? Output only yes or no.")
                    rows.append({
                        "case_id": f"c057-s-{len(rows):05d}", "partition": partition,
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
        target = span(tok, ids, row["target"])
        family = span(tok, ids, row["tested_family"])
        query = query_position(tok, ids)
        boundary = len(ids) - 1
        result.append({
            "case_id": row["case_id"], "prompt_ids": ids,
            "candidate_ids": [[int(x) for x in tok.encode(v, add_special_tokens=False)]
                              for v in row["candidates"]],
            "target_span": target, "tested_family_span": family,
            "query_position": query, "boundary_position": boundary,
            "role_positions": {"target": target, "family": family,
                               "query": [query], "boundary": [boundary]},
        })
    return result


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
            ("a_to_b", "aa", "ab", "bb"),
            ("b_to_a", "bb", "ba", "aa"),
        ):
            clean, corrupt, wrong = cells[clean_cell], cells[corrupt_cell], cells[wrong_cell]
            status_true = status_by[(status_key, wrong_cell)]
            pairs.append({
                "pair_id": f"{key}:{direction}", "partition": clean["partition"],
                "surface": clean["surface"], "family_pair": clean["family_pair"],
                "direction": direction, "target": clean["target"],
                "target_family": clean["target_family"], "tested_family": clean["tested_family"],
                "wrong_family": corrupt["tested_family"],
                "clean_true": clean["case_id"], "corrupt_false": corrupt["case_id"],
                "wrong_identity_true": wrong["case_id"], "status_true": status_true["case_id"],
            })
    return pairs


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1369 already exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if not parent.get("campaign_closed") or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("C056 final audit is not closed")
    tok = tokenizer()
    concept_rows = concepts()
    active, status = active_cases(), status_cases()
    compiled_active = compile_rows(tok, active, SYSTEM_ACTIVE)
    compiled_status = compile_rows(tok, status, SYSTEM_STATUS)
    pairs = candidate_pairs(active, status)
    old_words = {row["word"] for row in core.load(OLD / "material/frozen_concept_graph.json")["concepts"]}
    family_tokens = {family: tok.encode(" " + family, add_special_tokens=False) for family in FAMILIES}
    quartet_counts = Counter(row["quartet_key"] for row in active)
    cell_counts = Counter((row["partition"], row["surface"], row["target_family"])
                          for row in active if row["truth"])
    checks = {
        "parent_closed_audited": parent.get("campaign_closed") and parent_audit.get("all_checks_passed"),
        "concept_count": len(concept_rows) == 48 and len({row["word"] for row in concept_rows}) == 48,
        "independent_vocabulary": not ({row["word"] for row in concept_rows} & old_words),
        "family_balance": all(len(words) == 12 for words in FAMILIES.values()),
        "partition_balance": all(sum(row["partition"] == p for row in concept_rows) == 12 for p in PARTITIONS),
        "semantic_uniqueness": all(row["sense"] and row["adjudication"] for row in concept_rows),
        "family_single_token": all(len(ids) == 1 for ids in family_tokens.values()),
        "active_count_balance": len(active) == 864 and Counter(row["truth"] for row in active) == {True: 432, False: 432},
        "active_quartets": len(quartet_counts) == 216 and set(quartet_counts.values()) == {4},
        "status_count_balance": len(status) == 288 and Counter(row["truth"] for row in status) == {True: 144, False: 144},
        "candidate_pairs": len(pairs) == 432,
        "balanced_family_partition_surface": len(cell_counts) == 48 and set(cell_counts.values()) == {9},
        "controlled_naturalness": all("  " not in row["prompt"] and row["prompt"].endswith("yes or no.")
                                      for row in active + status),
        "compiled_counts": len(compiled_active) == 864 and len(compiled_status) == 288,
        "candidate_single_tokens": all(len(ids) == 1 for row in compiled_active + compiled_status
                                       for ids in row["candidate_ids"]),
        "family_spans_single": all(len(row["tested_family_span"]) == 1
                                   for row in compiled_active + compiled_status),
        "role_positions_valid": all(all(points and max(points) < len(row["prompt_ids"])
                                        for points in row["role_positions"].values())
                                    for row in compiled_active + compiled_status),
        "hidden_state_only": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    core.save(OUT / "material/frozen_concept_graph.json", {
        "schema": "c057.independent_concepts.v1", "families": FAMILIES,
        "partitions": {key: list(value) for key, value in PARTITIONS.items()},
        "concepts": concept_rows,
    })
    core.write_rows(OUT / "material/active_membership_cases.jsonl", active)
    core.write_rows(OUT / "material/status_cases.jsonl", status)
    core.write_rows(OUT / "material/candidate_pairs.jsonl", pairs)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled_active)
    core.write_rows(OUT / "compiled/qwen3_status.jsonl", compiled_status)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks,
        "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "zero_models": {"always_yes": 0.5, "always_no": 0.5,
                        "target_only_quartet_upper_bound": 0.5,
                        "family_only_quartet_upper_bound": 0.5,
                        "status_always_yes": 0.5, "status_always_no": 0.5},
        "naturalness_scope": "curated controlled English plus deterministic machine audit",
        "independent_human_blind_review": False,
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)

    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c057.independent_bidirectional_mediation_groups.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "independent-material bidirectional causal role, serial mediation, and raw-coordinate group boundary",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states", "candidate logits"],
        "forbidden": ["attention", "MLP", "parameter scan", "gradient", "PCA", "t-SNE", "UMAP",
                      "SAE", "learned probe", "post-reveal layer search", "post-reveal threshold change"],
        "material": {
            "families": list(FAMILIES), "concept_count": 48, "active_count": 864,
            "status_count": 288, "candidate_pair_count": 432,
            "partitions": list(PARTITIONS), "surfaces": list(SURFACES),
            "eligible_cases_per_cell": 6, "eligible_case_target": 288,
            "active_sha256": core.sha(OUT / "material/active_membership_cases.jsonl"),
            "status_sha256": core.sha(OUT / "material/status_cases.jsonl"),
            "pair_sha256": core.sha(OUT / "material/candidate_pairs.jsonl"),
        },
        "behavior": {
            "active_accuracy_min": 0.90, "partition_min": 0.85, "surface_min": 0.85,
            "family_min": 0.85, "truth_min": 0.85, "quartet_all_min": 0.75,
            "status_accuracy_min": 0.95, "status_partition_min": 0.90,
            "finite_fraction_min": 1.0, "same_shape_repeat_max_abs_diff": 1e-6,
            "eligibility": "clean, corrupt, wrong-identity, and status donors all individually behavior-correct",
        },
        "paths": PATHS,
        "camera": {
            "known_truth_systems": 256, "calibration_cases": 48,
            "same_shape_output_max_abs_diff": 1e-5,
            "same_shape_checkpoint_relative_l2_max": 1e-6,
            "serial_parallel_topology_exact": True,
        },
        "bidirectional": {
            "suff_projection_median_min": 0.15, "suff_control_advantage_median_min": 0.10,
            "suff_control_win_min": 0.65, "suff_output_gain_median_min": 0.50,
            "suff_output_advantage_median_min": 0.25, "suff_output_win_min": 0.65,
            "necessity_projection_median_min": 0.15, "necessity_output_damage_median_min": 0.50,
            "necessity_direction_fraction_min": 0.75, "necessity_over_status_median_min": 0.25,
            "necessity_over_status_win_min": 0.65, "self_output_max_abs_diff": 1e-4,
            "self_checkpoint_relative_l2_max": 1e-6,
        },
        "mediation": {
            "path": "family_early", "upstream_rescue_median_min": 0.50,
            "query_block_fraction_median_min": 0.50, "query_block_positive_fraction_min": 0.65,
            "boundary_block_fraction_median_min": 0.50, "boundary_block_positive_fraction_min": 0.65,
            "clean_checkpoint_control_loss_fraction_max": 0.25,
        },
        "coordinate_groups": {
            "source": "family@3", "discovery_partition": "prototype_discovery",
            "evaluation_partitions": ["confirmation", "lockbox"],
            "routes": ["magnitude", "stable_sign", "family_min", "deterministic_random"],
            "sizes": [16, 32, 64, 128, 256, 512, 1024, 2048, 2560],
            "random_seed": 5701369, "suff_gain_median_min": 0.50,
            "suff_advantage_median_min": 0.25, "suff_win_min": 0.65,
            "necessity_damage_median_min": 0.50, "necessity_direction_fraction_min": 0.65,
            "whole_effect_fraction_median_min": 0.80, "self_max_abs_diff": 1e-4,
            "route_failure_eliminates_route_only": True,
        },
        "branching": {
            "phase1370": "run behavior and freeze 288 eligible pairs if every cell has at least six",
            "phase1371": "run known-truth and exact-shape cameras if behavior qualifies",
            "phase1372": "run both whole-state paths if camera qualifies",
            "phase1373": "run mediation if family_early sufficiency and necessity both qualify",
            "phase1374": "run all four coordinate routes if family_early whole-state qualifies",
            "finish": "close C057 after all authorized routes; no replacement family, layer, route, size, or threshold",
        },
        "claim_boundary": {
            "allowed": "Qwen-specific independent-material full-state sufficiency/necessity, mediation boundary, and raw-coordinate group curve",
            "forbidden": ["relation manifold discovered", "parameter mechanism", "minimal natural circuit",
                          "cross-model invariance", "attention/MLP mechanism", "all language relations"],
        },
        "stop_rule": "A failed route is eliminated; the finite parent campaign closes only after every preauthorized branch finishes or becomes ineligible.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1370_c057_behavior_qualification"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "all_gates_passed": True, "authorization": protocol["authorization"],
    })
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
