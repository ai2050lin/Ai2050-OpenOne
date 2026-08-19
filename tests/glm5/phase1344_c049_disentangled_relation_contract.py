#!/usr/bin/env python3
"""Phase1344: freeze C049 disentangled relation-field campaign."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from model_utils import MODEL_CONFIGS

PHASE = 1344
CAMPAIGN = "C049"
OUT = TESTS / "result/phase1344_c049_disentangled_relation_contract"
PARENT = TESTS / "result/phase1343_c048_factorial_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "confirmation", "holdout")
FAMILIES = ("cheese", "pasta", "disease", "profession")
SURFACES = {
    "ordinary": 'In ordinary English, does the noun "{word}" belong to the category {family}?',
    "dictionary": 'Would a standard dictionary classify "{word}" as a type of {family}?',
    "claim": 'Is the category statement "{word} is a {family}" correct?',
}
SYSTEM = "Evaluate the ordinary-English category relation. Output only yes or no."

WORDS = {
    "discovery": {
        "cheese": ("cheddar", "mozzarella", "parmesan", "gouda"),
        "pasta": ("spaghetti", "linguine", "penne", "fusilli"),
        "disease": ("malaria", "influenza", "measles", "mumps"),
        "profession": ("plumber", "dentist", "surgeon", "carpenter"),
    },
    "confirmation": {
        "cheese": ("brie", "camembert", "ricotta", "feta"),
        "pasta": ("ravioli", "tortellini", "farfalle", "rigatoni"),
        "disease": ("cholera", "tetanus", "rabies", "rubella"),
        "profession": ("electrician", "architect", "pharmacist", "mechanic"),
    },
    "holdout": {
        "cheese": ("gruyere", "provolone", "mascarpone", "roquefort"),
        "pasta": ("tagliatelle", "vermicelli", "lasagne", "macaroni"),
        "disease": ("diphtheria", "scurvy", "rickets", "hepatitis"),
        "profession": ("accountant", "librarian", "baker", "pilot"),
    },
}


def concepts():
    rows = []
    for partition in PARTITIONS:
        for family in FAMILIES:
            for word in WORDS[partition][family]:
                rows.append(
                    {
                        "word": word,
                        "family": family,
                        "partition": partition,
                        "sense": f"ordinary noun sense of {word}",
                        "adjudication": f"unambiguous member of {family} among the four frozen families",
                    }
                )
    return rows


def cases():
    rows = []
    for partition in PARTITIONS:
        for family_a, family_b in combinations(FAMILIES, 2):
            words_a, words_b = WORDS[partition][family_a], WORDS[partition][family_b]
            for index, word_a in enumerate(words_a):
                for offset in (0, 1):
                    word_b = words_b[(index + offset) % 4]
                    for surface, template in SURFACES.items():
                        quartet = f"{partition}:{family_a}__{family_b}:{index}:o{offset}:{surface}"
                        cells = (
                            ("aa", word_a, family_a, family_a, True, 1),
                            ("ab", word_a, family_a, family_b, False, -1),
                            ("ba", word_b, family_b, family_a, False, -1),
                            ("bb", word_b, family_b, family_b, True, 1),
                        )
                        for cell, target, target_family, tested_family, truth, sign in cells:
                            rows.append(
                                {
                                    "case_id": f"c049-b-{len(rows):05d}",
                                    "partition": partition,
                                    "family_pair": f"{family_a}__{family_b}",
                                    "pair_index": index,
                                    "pair_offset": offset,
                                    "surface": surface,
                                    "quartet_key": quartet,
                                    "cell": cell,
                                    "interaction_sign": sign,
                                    "target": target,
                                    "target_family": target_family,
                                    "tested_family": tested_family,
                                    "truth": truth,
                                    "prompt": template.format(word=target, family=tested_family)
                                    + " Output only yes or no.",
                                    "candidates": ["yes", "no"],
                                    "gold_position": 0 if truth else 1,
                                }
                            )
    return rows


def prior_words():
    found = set()
    for path in (TESTS / "result").glob("phase13*/material/frozen_concept_graph.json"):
        try:
            found.update(str(x["word"]) for x in core.load(path).get("concepts", []))
        except Exception:
            pass
    return found


def tokenizer(model_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[model_name]["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def span(tokenizer, token_ids, value):
    variants = [value, " " + value]
    needles = [[int(x) for x in tokenizer.encode(v, add_special_tokens=False)] for v in variants]
    return core.locate_last_subsequence(token_ids, needles)


def compile_for(model_name, rows):
    tokenizer_ = tokenizer(model_name)
    out = []
    for row in rows:
        prompt_ids = core.chat_ids(tokenizer_, SYSTEM, row["prompt"])
        out.append(
            {
                "case_id": row["case_id"],
                "prompt_ids": prompt_ids,
                "candidate_ids": [
                    [int(x) for x in tokenizer_.encode(candidate, add_special_tokens=False)]
                    for candidate in ("yes", "no")
                ],
                "target_span": span(tokenizer_, prompt_ids, row["target"]),
                "tested_family_span": span(tokenizer_, prompt_ids, row["tested_family"]),
                "boundary_position": len(prompt_ids) - 1,
            }
        )
    return out


def main():
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent.get("authorization") != "close_c048_behavior" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("C048 must be formally closed before C049")
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1344 result already exists")

    concept_rows, case_rows = concepts(), cases()
    compiled = {model: compile_for(model, case_rows) for model in MODELS}
    quartets = {}
    for row in case_rows:
        quartets.setdefault(row["quartet_key"], []).append(row)

    selected_words = {row["word"] for row in concept_rows}
    checks = {
        "fresh_words": not (selected_words & prior_words()),
        "concept_count": len(concept_rows) == 48 and len(selected_words) == 48,
        "concept_balance": all(
            sum(row["partition"] == p and row["family"] == f for row in concept_rows) == 4
            for p in PARTITIONS
            for f in FAMILIES
        ),
        "semantic_uniqueness": all(row["adjudication"] for row in concept_rows)
        and all(sum(row["word"] == w for row in concept_rows) == 1 for w in selected_words),
        "case_count": len(case_rows) == 1728 and len({row["case_id"] for row in case_rows}) == 1728,
        "quartet_count": len(quartets) == 432
        and all(
            [x["cell"] for x in q] == ["aa", "ab", "ba", "bb"]
            and [x["interaction_sign"] for x in q] == [1, -1, -1, 1]
            for q in quartets.values()
        ),
        "truth_balance": Counter(row["truth"] for row in case_rows) == {True: 864, False: 864},
        "surface_balance": all(sum(row["surface"] == s for row in case_rows) == 576 for s in SURFACES),
        "partition_balance": all(sum(row["partition"] == p for row in case_rows) == 576 for p in PARTITIONS),
        "pairing_coverage": all(
            sum(
                row["partition"] == p
                and row["family_pair"] == f"{a}__{b}"
                and row["pair_offset"] == o
                for row in case_rows
            )
            == 48
            for p in PARTITIONS
            for a, b in combinations(FAMILIES, 2)
            for o in (0, 1)
        ),
        "machine_naturalness": all(
            "  " not in row["prompt"]
            and row["prompt"].endswith("yes or no.")
            and row["target"] in row["prompt"]
            and row["tested_family"] in row["prompt"]
            for row in case_rows
        ),
    }
    for model, rows in compiled.items():
        checks[f"{model}_compiled"] = len(rows) == 1728 and all(
            a["case_id"] == b["case_id"] for a, b in zip(case_rows, rows)
        )
        checks[f"{model}_candidate_tokens"] = all(
            all(len(candidate) == 1 for candidate in row["candidate_ids"]) for row in rows
        )
        checks[f"{model}_role_spans"] = all(
            row["target_span"]
            and row["tested_family_span"]
            and max(row["target_span"] + row["tested_family_span"]) < row["boundary_position"]
            for row in rows
        )
    if not all(checks.values()):
        raise RuntimeError([key for key, passed in checks.items() if not passed])

    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c049.graph.v1", "concepts": concept_rows})
    core.write_rows(OUT / "material/frozen_factorial_cases.jsonl", case_rows)
    for model in MODELS:
        core.write_rows(OUT / f"compiled/{model}_factorial.jsonl", compiled[model])

    zero_models = {
        "always_yes_accuracy": 0.5,
        "always_no_accuracy": 0.5,
        "surface_only_accuracy": 0.5,
        "target_only_without_tested_family_upper_bound": 0.5,
        "tested_family_only_without_target_upper_bound": 0.5,
        "additive_target_plus_family_second_order_interaction": 0.0,
        "generic_truth_response": "may pass behavior but must fail six-way family-pair identity",
    }
    core.save(
        OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json",
        {
            "checks": checks,
            "passed": sum(checks.values()),
            "total": len(checks),
            "all_checks_passed": all(checks.values()),
            "zero_models": zero_models,
            "naturalness_scope": "curated controlled English plus deterministic machine audit",
            "independent_human_blind_review": "not available; no natural-language universality claim is authorized",
        },
    )

    sentinel = [q[0]["case_id"] for q in list(quartets.values())[:24]]
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c049.disentangled_relation.v1",
        "parent_c048_status": "formally closed without reinterpretation or threshold change",
        "research_object": "full-dimensional role-aligned target-by-family second-order interaction field with separately typed behavior, joint-reliability, single-model, and cross-model ledgers",
        "claim_boundary": {
            "allowed": [
                "marginal relation behavior",
                "second-order output interaction",
                "quartet joint reliability",
                "model-specific descriptive interaction field",
                "conditionally model-specific same-label causal swaps",
                "cross-model repetition only if at least two models independently pass",
            ],
            "not_assumed": [
                "semantic ontology recovery",
                "relative coding correctness",
                "attention or MLP localization",
                "parameter-level mechanism",
                "cross-model physical-coordinate invariance",
                "human-natural-language generality",
            ],
        },
        "material": {
            "case_count": 1728,
            "quartet_count": 432,
            "partitions": list(PARTITIONS),
            "families": list(FAMILIES),
            "family_pairs": 6,
            "pair_offsets": [0, 1],
            "surfaces": list(SURFACES),
            "graph_sha256": core.sha(OUT / "material/frozen_concept_graph.json"),
            "cases_sha256": core.sha(OUT / "material/frozen_factorial_cases.jsonl"),
        },
        "models": list(MODELS),
        "model_order": list(MODELS),
        "precision": "bfloat16-no-quantization",
        "batch_size": 4,
        "zero_models": zero_models,
        "executor_gate": {
            "sentinel_case_ids": sentinel,
            "finite_fraction_min": 1.0,
            "rank_agreement_min": 1.0,
            "max_abs_diff_max": 1e-6,
        },
        "behavior_ledgers": {
            "relation_interaction_authorization": {
                "accuracy_min": 0.90,
                "partition_min": 0.85,
                "surface_min": 0.85,
                "family_min": 0.85,
                "truth_min": 0.85,
                "pairwise_true_over_false_min": 0.95,
                "positive_interaction_fraction_min": 0.95,
                "median_interaction_min": 2.0,
            },
            "quartet_joint_reliability_report": {
                "point_rate_target": 0.85,
                "wilson_95_lower_bound_target": 0.80,
                "authorization_effect": "reported independently and does not gate the interaction-field branch",
            },
        },
        "field_gate": {
            "depths": "embedding plus every model-layer output",
            "roles": ["target_span_mean", "tested_family_span_mean", "answer_boundary"],
            "primary_role": "tested_family_span_mean",
            "object": "signed quartet interaction H_aa-H_ab-H_ba+H_bb",
            "storage": "float32 full-dimensional signed interactions; no PCA or learned projection",
            "numeric_relative_l2_p95_max": 1e-5,
            "numeric_relative_l2_max": 1e-4,
            "layer0_relative_norm_max": 1e-5,
            "discovery_family_pair_top1_min": 0.70,
            "discovery_median_gap_min": 0.05,
            "discovery_relative_norm_min": 0.001,
            "confirmation_family_pair_top1_min": 0.60,
            "holdout_family_pair_top1_min": 0.60,
            "transfer_median_gap_min": 0.03,
            "permuted_label_top1_max": 0.30,
            "selection": "earliest non-embedding layer passing discovery at tested-family role; no reselection",
            "single_model_authorization": "each behavior-qualified model is evaluated independently",
            "cross_model_minimum": 2,
        },
        "causal_gate": {
            "site_role": "tested_family_span_mean",
            "layer": "per-model field-selected layer",
            "target_partitions": ["confirmation", "holdout"],
            "donor_partition": "discovery",
            "arms": [
                "baseline",
                "self_patch",
                "same_label_true_donor_to_false_recipient",
                "same_label_false_donor_to_true_recipient",
                "same_label_false_wrong_donor",
                "same_label_true_wrong_donor",
            ],
            "false_to_true_median_gain_min": 0.5,
            "true_to_false_median_damage_min": 0.5,
            "direction_fraction_min": 0.75,
            "flip_fraction_min": 0.30,
            "correct_over_wrong_median_min": 0.25,
            "correct_over_wrong_win_min": 0.65,
            "self_max_abs_margin_diff_max": 1e-4,
            "partition_direction_fraction_min": 0.65,
            "single_model_authorization": "each field-qualified model is evaluated independently",
            "cross_model_minimum": 2,
        },
        "branching": {
            "behavior_model_fail": "close that model before hidden-state capture",
            "behavior_model_pass": "run that model's full-dimensional interaction field",
            "field_model_fail": "close that model without a causal claim",
            "field_model_pass": "run frozen same-label bidirectional swaps for that model",
            "causal_model_fail": "close at model-specific causal-selectivity boundary",
            "causal_model_pass": "register model-specific typed causal evidence",
            "all_models_fail": "close C049",
            "cross_model_claim": "requires at least two independent model-specific passes",
        },
        "stop_rule": "After reveal, do not change object, material, partition, model, null, threshold, role, layer rule, donor rule, or branch.",
        "parameter_boundary": "No attention, MLP, sparse dictionary, or parameter scan is authorized.",
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1345_c049_disentangled_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(
        OUT / "analysis/final.json",
        {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "all_gates_passed": True,
            "authorization": protocol["authorization"],
            "contract_sha256": protocol["contract_sha256"],
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        json.dumps(
            {
                "checks": checks,
                "contract_sha256": protocol["contract_sha256"],
                "authorization": protocol["authorization"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
