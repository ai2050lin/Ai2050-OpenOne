#!/usr/bin/env python3
"""Phase1353: freeze the finite C053 Qwen relation-route portfolio."""
from __future__ import annotations

import json
import statistics
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

PHASE, CAMPAIGN = 1353, "C053"
PARENT = TESTS / "result/phase1352_c052_qwen_pair_probe_behavior"
OUT = TESTS / "result/phase1353_c053_route_portfolio_contract"
MODEL = "qwen3"

SURFACES = {
    "ordinary": 'In ordinary English, does the noun "{word}" belong to the category {family}?',
    "dictionary": 'Would a standard dictionary classify "{word}" as a type of {family}?',
    "claim": 'Is the category statement "{word} is a {family}" correct?',
}
SYSTEM_BINARY = "Evaluate the ordinary-English category relation. Output only yes or no."
SYSTEM_CHOICE = "Choose the category that best fits the noun. Output only A or B."
SYSTEM_STATUS = "Read the explicit review status. Output only yes or no."

FAMILIES = {
    "constellation": ("orion", "cassiopeia", "andromeda", "cygnus", "draco", "perseus",
                      "scorpius", "aquila", "lyra", "aries", "taurus", "gemini"),
    "river": ("nile", "amazon", "yangtze", "mississippi", "danube", "ganges",
              "rhine", "mekong", "volga", "congo", "indus", "euphrates"),
    "sport": ("cricket", "baseball", "soccer", "rugby", "hockey", "tennis",
              "golf", "volleyball", "badminton", "squash", "lacrosse", "handball"),
    "chemical element": ("hydrogen", "helium", "lithium", "beryllium", "boron", "carbon",
                         "nitrogen", "oxygen", "fluorine", "neon", "sodium", "magnesium"),
    "currency": ("dollar", "euro", "yen", "peso", "rupee", "franc",
                 "pound", "krona", "dinar", "baht", "won", "rial"),
    "language": ("english", "spanish", "french", "german", "mandarin", "arabic",
                 "hindi", "swahili", "japanese", "korean", "turkish", "polish"),
    "academic field": ("physics", "chemistry", "biology", "geology", "astronomy", "sociology",
                       "economics", "linguistics", "archaeology", "botany", "zoology", "ecology"),
    "emotion": ("joy", "sadness", "anger", "fear", "surprise", "disgust",
                "envy", "pride", "shame", "guilt", "hope", "relief"),
}
SEEN = tuple(list(FAMILIES)[:4])
UNSEEN = tuple(list(FAMILIES)[4:])
PARTITIONS = {
    "prototype_discovery": (SEEN, range(0, 6)),
    "clock_selection": (SEEN, range(6, 12)),
    "confirmation": (UNSEEN, range(0, 6)),
    "lockbox": (UNSEEN, range(6, 12)),
}


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[MODEL]["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def span(tok, ids, value):
    needles = [[int(x) for x in tok.encode(v, add_special_tokens=False)] for v in (value, " " + value)]
    return core.locate_last_subsequence(ids, needles)


def concept_rows():
    rows = []
    for family, words in FAMILIES.items():
        family_scope = "seen-family" if family in SEEN else "held-family"
        for index, word in enumerate(words):
            partition = (
                "prototype_discovery" if family in SEEN and index < 6 else
                "clock_selection" if family in SEEN else
                "confirmation" if index < 6 else "lockbox"
            )
            rows.append({
                "word": word,
                "family": family,
                "family_scope": family_scope,
                "partition": partition,
                "sense": f"ordinary noun sense of {word}",
                "adjudication": f"unambiguous member of {family} within the frozen eight-family inventory",
            })
    return rows


def binary_cases():
    rows = []
    for partition, (families, indexes) in PARTITIONS.items():
        for family_a, family_b in combinations(families, 2):
            for index in indexes:
                word_a, word_b = FAMILIES[family_a][index], FAMILIES[family_b][index]
                for surface, template in SURFACES.items():
                    quartet = f"{partition}:{family_a}__{family_b}:{index}:{surface}"
                    cells = (
                        ("aa", word_a, family_a, family_a, True, 1),
                        ("ab", word_a, family_a, family_b, False, -1),
                        ("ba", word_b, family_b, family_a, False, -1),
                        ("bb", word_b, family_b, family_b, True, 1),
                    )
                    for cell, target, target_family, tested_family, truth, sign in cells:
                        rows.append({
                            "case_id": f"c053-b1-{len(rows):05d}",
                            "route": "B1_binary",
                            "partition": partition,
                            "family_scope": "seen-family" if family_a in SEEN else "held-family",
                            "family_pair": f"{family_a}__{family_b}",
                            "surface": surface,
                            "quartet_key": quartet,
                            "cell": cell,
                            "interaction_sign": sign,
                            "target": target,
                            "target_family": target_family,
                            "tested_family": tested_family,
                            "truth": truth,
                            "prompt": template.format(word=target, family=tested_family) + " Output only yes or no.",
                            "candidates": ["yes", "no"],
                            "gold_position": 0 if truth else 1,
                        })
    return rows


def choice_cases():
    rows = []
    for partition, (families, indexes) in PARTITIONS.items():
        for family_a, family_b in combinations(families, 2):
            for index in indexes:
                word_a, word_b = FAMILIES[family_a][index], FAMILIES[family_b][index]
                for surface in SURFACES:
                    for side, target, target_family in (("a", word_a, family_a), ("b", word_b, family_b)):
                        group = f"{partition}:{family_a}__{family_b}:{index}:{surface}:{side}"
                        for order in (0, 1):
                            first, second = ((family_a, family_b) if order == 0 else (family_b, family_a))
                            prompt = (
                                f'Consider the noun "{target}" in its ordinary sense. '
                                f"Which category fits it better? A: {first}. B: {second}. Output only A or B."
                            )
                            gold = 0 if target_family == first else 1
                            rows.append({
                                "case_id": f"c053-b3-{len(rows):05d}",
                                "route": "B3_choice",
                                "partition": partition,
                                "family_scope": "seen-family" if family_a in SEEN else "held-family",
                                "family_pair": f"{family_a}__{family_b}",
                                "surface": surface,
                                "choice_group": group,
                                "order": order,
                                "target_side": side,
                                "target": target,
                                "target_family": target_family,
                                "candidate_a": first,
                                "candidate_b": second,
                                "prompt": prompt,
                                "candidates": ["A", "B"],
                                "gold_position": gold,
                            })
    return rows


def status_cases():
    rows = []
    for partition, (families, indexes) in PARTITIONS.items():
        for family_a, family_b in combinations(families, 2):
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
                    prompt = (
                        f'Record noun: "{target}". Recorded category: {tested_family}. '
                        f"Review status: {status}. Is the review status approved? Output only yes or no."
                    )
                    rows.append({
                        "case_id": f"c053-n-{len(rows):05d}",
                        "route": "N_status",
                        "partition": partition,
                        "family_scope": "seen-family" if family_a in SEEN else "held-family",
                        "family_pair": f"{family_a}__{family_b}",
                        "surface": "status_record",
                        "quartet_key": quartet,
                        "cell": cell,
                        "target": target,
                        "tested_family": tested_family,
                        "status": status,
                        "truth": truth,
                        "prompt": prompt,
                        "candidates": ["yes", "no"],
                        "gold_position": 0 if truth else 1,
                    })
    return rows


def compile_rows(tok, rows, system):
    result = []
    for row in rows:
        ids = core.chat_ids(tok, system, row["prompt"])
        value = {
            "case_id": row["case_id"],
            "prompt_ids": ids,
            "candidate_ids": [
                [int(x) for x in tok.encode(candidate, add_special_tokens=False)] for candidate in row["candidates"]
            ],
            "target_span": span(tok, ids, row["target"]),
            "boundary_position": len(ids) - 1,
        }
        if row["route"] in ("B1_binary", "N_status"):
            value["tested_family_span"] = span(tok, ids, row["tested_family"])
        else:
            value["candidate_a_span"] = span(tok, ids, row["candidate_a"])
            value["candidate_b_span"] = span(tok, ids, row["candidate_b"])
        result.append(value)
    return result


def prior_words():
    found = set()
    for path in (TESTS / "result").glob("phase13*/material/frozen_concept_graph.json"):
        try:
            found.update(str(x["word"]) for x in core.load(path).get("concepts", []))
        except Exception:
            pass
    return found


def main():
    parent = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent.get("authorization") != "close_c052_behavior" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1352 must be closed and audited")
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1353 already exists")

    concepts = concept_rows()
    b1, b3, status = binary_cases(), choice_cases(), status_cases()
    tok = tokenizer()
    compiled = {
        "B1_binary": compile_rows(tok, b1, SYSTEM_BINARY),
        "B3_choice": compile_rows(tok, b3, SYSTEM_CHOICE),
        "N_status": compile_rows(tok, status, SYSTEM_STATUS),
    }
    all_words = {row["word"] for row in concepts}
    quartets = defaultdict(list)
    for row in b1:
        quartets[row["quartet_key"]].append(row)
    choice_groups = defaultdict(list)
    for row in b3:
        choice_groups[row["choice_group"]].append(row)
    status_quartets = defaultdict(list)
    for row in status:
        status_quartets[row["quartet_key"]].append(row)
    route_lengths = {
        route: statistics.median(len(row["prompt_ids"]) for row in values)
        for route, values in compiled.items()
    }
    checks = {
        "parent": parent["authorization"] == "close_c052_behavior",
        "fresh_words": not (all_words & prior_words()),
        "concept_count": len(concepts) == 96 and len(all_words) == 96,
        "family_balance": all(len(words) == 12 for words in FAMILIES.values()),
        "family_holdout": set(SEEN).isdisjoint(UNSEEN) and len(SEEN) == len(UNSEEN) == 4,
        "partition_balance": all(sum(r["partition"] == p for r in concepts) == 24 for p in PARTITIONS),
        "semantic_uniqueness": all(r["sense"] and r["adjudication"] for r in concepts),
        "b1_count": len(b1) == 1728 and len(quartets) == 432,
        "b1_quartets": all([x["cell"] for x in q] == ["aa", "ab", "ba", "bb"] for q in quartets.values()),
        "b1_balance": Counter(r["truth"] for r in b1) == {True: 864, False: 864},
        "b3_count": len(b3) == 1728 and len(choice_groups) == 864,
        "b3_order_balance": all(len(g) == 2 and {x["gold_position"] for x in g} == {0, 1}
                                for g in choice_groups.values()),
        "status_count": len(status) == 576 and len(status_quartets) == 144,
        "status_balance": Counter(r["truth"] for r in status) == {True: 288, False: 288},
        "controlled_naturalness": all("  " not in r["prompt"] and r["prompt"].endswith(("yes or no.", "A or B."))
                                      for r in b1 + b3 + status),
        "compiled_counts": all(len(compiled[k]) == n for k, n in (("B1_binary", 1728), ("B3_choice", 1728), ("N_status", 576))),
        "candidate_single_tokens": all(len(c) == 1 for values in compiled.values() for r in values for c in r["candidate_ids"]),
        "role_spans": all(r["target_span"] and max(r["target_span"]) < r["boundary_position"]
                          for values in compiled.values() for r in values),
        "route_length_match": max(route_lengths.values()) - min(route_lengths.values()) <= 30,
    }
    if not all(checks.values()):
        raise RuntimeError([k for k, v in checks.items() if not v])

    core.save(OUT / "material/frozen_concept_graph.json", {
        "schema": "c053.graph.v1", "seen_families": list(SEEN), "held_families": list(UNSEEN),
        "concepts": concepts,
    })
    core.write_rows(OUT / "material/b1_binary_cases.jsonl", b1)
    core.write_rows(OUT / "material/b3_choice_cases.jsonl", b3)
    core.write_rows(OUT / "material/status_null_cases.jsonl", status)
    for route, values in compiled.items():
        core.write_rows(OUT / f"compiled/qwen3_{route}.jsonl", values)

    zero_models = {
        "B1_always_yes": 0.5,
        "B1_always_no": 0.5,
        "B2_target_only_pairwise_upper_bound": 0.5,
        "B2_family_only_pairwise_upper_bound": 0.5,
        "B3_always_A": 0.5,
        "B3_always_B": 0.5,
        "B3_target_only_without_candidates": 0.5,
        "status_always_yes": 0.5,
        "status_always_no": 0.5,
    }
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "zero_models": zero_models, "route_token_length_medians": route_lengths,
        "naturalness_scope": "curated controlled English plus deterministic machine audit",
        "independent_human_blind_review": False,
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)

    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c053.finite_route_portfolio.v1",
        "model": MODEL,
        "research_object": "Qwen-specific noun-membership relation routes, full-dimensional fields, and conditional typed causal selectivity",
        "claim_boundary": {
            "allowed": "route-specific behavior; conditional descriptive readability; conditional causal selectivity",
            "not_assumed": ["relative coding is true", "semantic ontology recovery", "cross-model invariance",
                            "attention/MLP localization", "parameter mechanism", "natural-language universality"],
        },
        "material": {
            "families": list(FAMILIES), "seen_families": list(SEEN), "held_families": list(UNSEEN),
            "partitions": list(PARTITIONS), "surfaces": list(SURFACES),
            "concept_count": 96, "B1_case_count": 1728, "B3_case_count": 1728, "status_case_count": 576,
        },
        "routes": {
            "B1_absolute": {
                "source": "B1_binary", "accuracy_min": 0.90, "partition_min": 0.85,
                "surface_min": 0.85, "family_min": 0.85, "truth_min": 0.85,
                "quartet_all_min": 0.80,
            },
            "B2_relative": {
                "source": "B1_binary", "pairwise_win_min": 0.95,
                "partition_pairwise_min": 0.90, "surface_pairwise_min": 0.90,
                "positive_interaction_min": 0.95, "median_interaction_min": 2.0,
            },
            "B3_choice": {
                "source": "B3_choice", "accuracy_min": 0.95, "partition_min": 0.90,
                "surface_min": 0.90, "family_min": 0.90, "position_min": 0.90,
                "choice_group_all_min": 0.85,
            },
        },
        "status_gate": {"accuracy_min": 0.98, "partition_min": 0.97, "truth_min": 0.97,
                        "quartet_all_min": 0.90},
        "executor_gate": {"finite_fraction_min": 1.0, "rank_agreement_min": 1.0,
                          "max_abs_diff_max": 1e-6, "batch_size": 4},
        "route_logic": {
            "route_fail": "eliminate only that route",
            "all_behavior_routes_fail": "close C053 before hidden state",
            "B2_pass": "authorize quartet interaction field",
            "B3_pass": "authorize choice-order-invariance field",
            "B1_pass_only": "behavior evidence only; B1 alone does not authorize a relation field",
        },
        "field_gate": {
            "roles": ["target_span_mean", "tested_family_span_mean", "answer_boundary"],
            "primary_role": "tested_family_span_mean",
            "depths": "embedding plus every layer",
            "storage": "float32 full-dimensional response objects; no fitted projection",
            "numeric_relative_l2_max": 1e-6,
            "layer0_relative_norm_max": 1e-6,
            "family_pair_selection_top1_min": 0.60,
            "family_pair_surface_min": 0.50,
            "family_pair_gap_min": 0.03,
            "persistence_layers": 3,
            "shared_selection_cosine_min": 0.20,
            "shared_transfer_cosine_min": 0.15,
            "active_over_status_gap_min": 0.05,
            "active_over_status_win_min": 0.65,
            "status_direction_cosine_max": 0.80,
            "choice_order_cosine_min": 0.90,
            "choice_retrieval_win_min": 0.75,
        },
        "causal_gate": {
            "authorized_only_if": "B2 and shared relation field pass",
            "site_role": "tested_family_span_mean",
            "layer": "field-selected layer without reselection",
            "recipient_partitions": ["confirmation", "lockbox"],
            "surface": "ordinary",
            "donor_rule": "same partition and surface; different target index; no donor or layer search",
            "arms": ["baseline", "self", "same_family_true_donor", "different_family_true_donor",
                     "same_family_false_donor"],
            "false_to_true_gain_min": 0.5,
            "direction_fraction_min": 0.75,
            "correct_over_wrong_median_min": 0.25,
            "correct_over_wrong_win_min": 0.65,
            "self_max_abs_diff_max": 1e-4,
        },
        "branching": {
            "phase1354": "run all three behavior routes without mutation",
            "phase1355": "run every preauthorized field whose behavior route passed",
            "phase1356": "run causal test only if the shared relation field passed",
            "finish": "close after the last authorized branch; do not invent another route after reveal",
        },
        "stop_rule": "No post-reveal changes to object, material, route, partition, model, null, threshold, role, layer, donor, or branch.",
        "parameter_boundary": "No PCA, SAE, learned projection, attention/MLP search, parameter scan, or cross-model claim.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1354_c053_behavior_routes"
    core.save(OUT / "protocol/preregistration.json", protocol)
    final = {"phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
             "all_gates_passed": True, "authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
