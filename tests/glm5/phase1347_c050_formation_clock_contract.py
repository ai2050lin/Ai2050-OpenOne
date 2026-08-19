#!/usr/bin/env python3
"""Phase1347: freeze C050 relation-identity formation-clock campaign."""
from __future__ import annotations

import json
import math
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

PHASE, CAMPAIGN = 1347, "C050"
OUT = TESTS / "result/phase1347_c050_formation_clock_contract"
PARENT = TESTS / "result/phase1346_c049_full_interaction_field"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("prototype_discovery", "clock_selection", "confirmation", "holdout")
FAMILIES = ("bird", "fish", "insect", "reptile")
CORE_SURFACES = {
    "ordinary": 'In ordinary English, does the noun "{word}" belong to the category {family}?',
    "dictionary": 'Would a standard dictionary classify "{word}" as a type of {family}?',
}
SYSTEM = "Answer the stated yes-or-no question. Output only yes or no."
WORDS = {
    "prototype_discovery": {
        "bird": ("robin", "heron", "magpie", "canary"),
        "fish": ("haddock", "halibut", "sturgeon", "tilapia"),
        "insect": ("moth", "wasp", "flea", "cockroach"),
        "reptile": ("crocodile", "alligator", "gecko", "iguana"),
    },
    "clock_selection": {
        "bird": ("albatross", "finch", "woodpecker", "thrush"),
        "fish": ("marlin", "barracuda", "grouper", "bluegill"),
        "insect": ("locust", "aphid", "weevil", "gnat"),
        "reptile": ("chameleon", "cobra", "viper", "tortoise"),
    },
    "confirmation": {
        "bird": ("warbler", "partridge", "pheasant", "kingfisher"),
        "fish": ("coelacanth", "sunfish", "clownfish", "lionfish"),
        "insect": ("hornet", "silverfish", "earwig", "bedbug"),
        "reptile": ("skink", "anole", "caiman", "rattlesnake"),
    },
    "holdout": {
        "bird": ("cuckoo", "ptarmigan", "toucan", "starling"),
        "fish": ("monkfish", "tarpon", "swordtail", "guppy"),
        "insect": ("mayfly", "lacewing", "damselfly", "stonefly"),
        "reptile": ("copperhead", "mamba", "terrapin", "gharial"),
    },
}


def concepts():
    return [
        {
            "word": word,
            "family": family,
            "partition": partition,
            "sense": f"ordinary zoological noun sense of {word}",
            "adjudication": f"a taxonomic everyday-English member of {family}, and not a member of the other frozen families",
        }
        for partition in PARTITIONS
        for family in FAMILIES
        for word in WORDS[partition][family]
    ]


def add_quartet(rows, panel, partition, family_a, family_b, index, offset, surface, template):
    word_a = WORDS[partition][family_a][index]
    word_b = WORDS[partition][family_b][(index + offset) % 4]
    quartet_key = f"{panel}:{partition}:{family_a}__{family_b}:{index}:o{offset}:{surface}"
    if panel == "core_membership":
        cells = (
            ("aa", word_a, family_a, family_a, word_a, True, 1),
            ("ab", word_a, family_a, family_b, word_b, False, -1),
            ("ba", word_b, family_b, family_a, word_a, False, -1),
            ("bb", word_b, family_b, family_b, word_b, True, 1),
        )
    elif panel == "label_only":
        cells = (
            ("aa", word_a, family_a, family_a, word_a, True, 1),
            ("ab", word_a, family_a, family_b, word_b, True, -1),
            ("ba", word_b, family_b, family_a, word_a, True, -1),
            ("bb", word_b, family_b, family_b, word_b, True, 1),
        )
    elif panel == "generic_equality":
        cells = (
            ("aa", word_a, family_a, family_a, word_a, True, 1),
            ("ab", word_a, family_a, family_b, word_b, False, -1),
            ("ba", word_b, family_b, family_a, word_a, False, -1),
            ("bb", word_b, family_b, family_b, word_b, True, 1),
        )
    else:
        raise ValueError(panel)
    for cell, target, target_family, tested_family, reference, truth, sign in cells:
        if panel == "core_membership":
            question = template.format(word=target, family=tested_family)
        elif panel == "label_only":
            question = template.format(word=target, family=tested_family)
        else:
            question = template.format(word=target, family=tested_family, reference=reference)
        rows.append(
            {
                "case_id": f"c050-b-{len(rows):05d}",
                "panel": panel,
                "partition": partition,
                "family_pair": f"{family_a}__{family_b}",
                "pair_index": index,
                "pair_offset": offset,
                "surface": surface,
                "quartet_key": quartet_key,
                "cell": cell,
                "interaction_sign": sign,
                "target": target,
                "target_family": target_family,
                "tested_family": tested_family,
                "reference": reference,
                "truth": truth,
                "prompt": question + " Output only yes or no.",
                "candidates": ["yes", "no"],
                "gold_position": 0 if truth else 1,
            }
        )


def cases():
    rows = []
    label_template = (
        'A vocabulary card prints the noun "{word}" beside the heading "{family}". '
        "Does the card contain both quoted expressions?"
    )
    equality_template = (
        'A note headed "{family}" asks whether the words "{word}" and "{reference}" are exactly identical. '
        "Are they identical?"
    )
    for partition in PARTITIONS:
        for family_a, family_b in combinations(FAMILIES, 2):
            for index in range(4):
                for offset in (0, 1):
                    for surface, template in CORE_SURFACES.items():
                        add_quartet(
                            rows,
                            "core_membership",
                            partition,
                            family_a,
                            family_b,
                            index,
                            offset,
                            surface,
                            template,
                        )
                    add_quartet(
                        rows,
                        "label_only",
                        partition,
                        family_a,
                        family_b,
                        index,
                        offset,
                        "card",
                        label_template,
                    )
                    add_quartet(
                        rows,
                        "generic_equality",
                        partition,
                        family_a,
                        family_b,
                        index,
                        offset,
                        "equality",
                        equality_template,
                    )
    return rows


def prior_words():
    found = set()
    for path in (TESTS / "result").glob("phase13*/material/frozen_concept_graph.json"):
        try:
            found.update(str(row["word"]) for row in core.load(path).get("concepts", []))
        except Exception:
            pass
    return found


def tokenizer(model_name):
    from transformers import AutoTokenizer

    value = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[model_name]["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if value.pad_token is None:
        value.pad_token = value.eos_token
    value.padding_side = "right"
    return value


def locate(tokenizer_, token_ids, value, first=False):
    needles = [
        [int(x) for x in tokenizer_.encode(variant, add_special_tokens=False)]
        for variant in (value, " " + value)
    ]
    matches = []
    for needle in needles:
        for start in range(len(token_ids) - len(needle) + 1):
            if token_ids[start : start + len(needle)] == needle:
                matches.append(list(range(start, start + len(needle))))
    if not matches:
        return None
    return min(matches, key=lambda span: span[0]) if first else max(matches, key=lambda span: span[0])


def compile_for(model_name, rows):
    tokenizer_ = tokenizer(model_name)
    compiled = []
    for row in rows:
        prompt_ids = core.chat_ids(tokenizer_, SYSTEM, row["prompt"])
        compiled.append(
            {
                "case_id": row["case_id"],
                "prompt_ids": prompt_ids,
                "candidate_ids": [
                    [int(x) for x in tokenizer_.encode(candidate, add_special_tokens=False)]
                    for candidate in ("yes", "no")
                ],
                "target_span": locate(tokenizer_, prompt_ids, row["target"], first=True),
                "tested_family_span": locate(tokenizer_, prompt_ids, row["tested_family"], first=False),
                "boundary_position": len(prompt_ids) - 1,
            }
        )
    return compiled


def binomial_upper_tail(n, p, threshold):
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(threshold, n + 1))


def main():
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent.get("authorization") != "close_c049_descriptive_field" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("C049 must be formally closed")
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1347 already exists")

    concept_rows, case_rows = concepts(), cases()
    compiled = {model: compile_for(model, case_rows) for model in MODELS}
    groups = defaultdict(list)
    for row in case_rows:
        groups[row["quartet_key"]].append(row)
    panel_counts = Counter(row["panel"] for row in case_rows)
    selected_words = {row["word"] for row in concept_rows}
    checks = {
        "fresh_words": not (selected_words & prior_words()),
        "concept_count": len(concept_rows) == 64 and len(selected_words) == 64,
        "concept_balance": all(
            sum(row["partition"] == p and row["family"] == f for row in concept_rows) == 4
            for p in PARTITIONS
            for f in FAMILIES
        ),
        "semantic_uniqueness": all(row["adjudication"] for row in concept_rows),
        "case_count": len(case_rows) == 3072 and len({row["case_id"] for row in case_rows}) == 3072,
        "panel_counts": panel_counts
        == {"core_membership": 1536, "label_only": 768, "generic_equality": 768},
        "quartet_count": len(groups) == 768
        and all([row["cell"] for row in quartet] == ["aa", "ab", "ba", "bb"] for quartet in groups.values()),
        "core_truth_balance": Counter(
            row["truth"] for row in case_rows if row["panel"] == "core_membership"
        )
        == {True: 768, False: 768},
        "equality_truth_balance": Counter(
            row["truth"] for row in case_rows if row["panel"] == "generic_equality"
        )
        == {True: 384, False: 384},
        "label_only_truth": all(row["truth"] for row in case_rows if row["panel"] == "label_only"),
        "partition_balance": all(sum(row["partition"] == p for row in case_rows) == 768 for p in PARTITIONS),
        "pairing_coverage": all(
            sum(
                row["panel"] == panel
                and row["partition"] == partition
                and row["family_pair"] == f"{family_a}__{family_b}"
                and row["pair_offset"] == offset
                for row in case_rows
            )
            == (32 if panel == "core_membership" else 16)
            for panel in ("core_membership", "label_only", "generic_equality")
            for partition in PARTITIONS
            for family_a, family_b in combinations(FAMILIES, 2)
            for offset in (0, 1)
        ),
        "machine_naturalness": all(
            "  " not in row["prompt"]
            and row["prompt"].endswith("yes or no.")
            and row["target"] in row["prompt"]
            and row["tested_family"] in row["prompt"]
            for row in case_rows
        ),
    }
    for model_name, rows in compiled.items():
        checks[f"{model_name}_compiled"] = len(rows) == len(case_rows) and all(
            left["case_id"] == right["case_id"] for left, right in zip(case_rows, rows)
        )
        checks[f"{model_name}_candidate_tokens"] = all(
            all(len(candidate) == 1 for candidate in row["candidate_ids"]) for row in rows
        )
        checks[f"{model_name}_spans"] = all(
            row["target_span"]
            and row["tested_family_span"]
            and max(row["target_span"] + row["tested_family_span"]) < row["boundary_position"]
            for row in rows
        )
    if not all(checks.values()):
        raise RuntimeError([key for key, value in checks.items() if not value])

    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c050.graph.v1", "concepts": concept_rows})
    core.write_rows(OUT / "material/frozen_cases.jsonl", case_rows)
    for model_name in MODELS:
        core.write_rows(OUT / f"compiled/{model_name}_cases.jsonl", compiled[model_name])

    power = {
        "six_class_chance": 1 / 6,
        "clock_query_count": 96,
        "top1_min": 0.70,
        "minimum_successes": 68,
        "chance_upper_tail": binomial_upper_tail(96, 1 / 6, 68),
        "null_top1_max": 0.30,
        "null_maximum_successes": 28,
        "persistence_layers": 3,
        "note": "Basic exact binomial calibration only; no learned statistic or C049-derived layer is used.",
    }
    zero_models = {
        "core_always_yes_accuracy": 0.5,
        "core_always_no_accuracy": 0.5,
        "equality_always_yes_accuracy": 0.5,
        "label_only_always_yes_accuracy": 1.0,
        "additive_second_order_interaction": 0.0,
        "pair_independent_truth_six_class_accuracy": 1 / 6,
        "label_pair_detector_warning": "can pass core pair identity but should also align label-only null and therefore fail null exclusion",
    }
    core.save(
        OUT / "audit/pre_model_material_zero_power_audit.json",
        {
            "checks": checks,
            "passed": sum(checks.values()),
            "total": len(checks),
            "all_checks_passed": all(checks.values()),
            "power": power,
            "zero_models": zero_models,
            "naturalness_scope": "researcher-curated controlled English plus deterministic syntax and truth audit",
            "independent_human_blind_review": "not available; claims restricted to the frozen controlled-language contract",
        },
    )

    sentinels = [quartet[0]["case_id"] for quartet in list(groups.values())[:24]]
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c050.formation_clock.v1",
        "research_object": "earliest persistent all-layer event at which full-dimensional target-by-family interaction transfers across unseen words and surfaces while rejecting label-only, generic-equality, and permuted-identity nulls",
        "claim_boundary": {
            "allowed": [
                "controlled membership behavior",
                "model-specific descriptive formation clock",
                "conditionally model-specific causal use around a confirmed clock",
                "cross-model repetition only after independent passes",
            ],
            "not_assumed": [
                "relative coding theory is correct",
                "late readability is upstream computation",
                "family-pair identity is a shared is-a operator",
                "fixed physical coordinates across models",
                "attention or MLP localization",
                "natural-language universality",
            ],
        },
        "material": {
            "case_count": 3072,
            "quartet_count": 768,
            "panels": dict(panel_counts),
            "partitions": list(PARTITIONS),
            "families": list(FAMILIES),
            "surfaces": list(CORE_SURFACES),
            "graph_sha256": core.sha(OUT / "material/frozen_concept_graph.json"),
            "cases_sha256": core.sha(OUT / "material/frozen_cases.jsonl"),
        },
        "models": list(MODELS),
        "model_order": list(MODELS),
        "precision": "bfloat16-no-quantization",
        "batch_size": 4,
        "zero_models": zero_models,
        "power": power,
        "executor_gate": {
            "sentinel_case_ids": sentinels,
            "finite_fraction_min": 1.0,
            "rank_agreement_min": 1.0,
            "max_abs_diff_max": 1e-6,
        },
        "behavior_gate": {
            "core_accuracy_min": 0.90,
            "core_partition_min": 0.85,
            "core_surface_min": 0.85,
            "core_family_min": 0.85,
            "core_truth_min": 0.85,
            "core_pairwise_min": 0.95,
            "core_positive_interaction_min": 0.95,
            "core_median_interaction_min": 2.0,
            "label_only_accuracy_min": 0.95,
            "generic_equality_accuracy_min": 0.95,
            "generic_equality_truth_min": 0.90,
        },
        "formation_gate": {
            "depths": "embedding plus every model layer",
            "core_roles": ["target_span_mean", "tested_family_span_mean", "answer_boundary"],
            "null_roles": ["tested_family_span_mean", "answer_boundary"],
            "primary_role": "tested_family_span_mean",
            "object": "full-dimensional signed quartet interaction without PCA or learned projection",
            "prototype_partition": "prototype_discovery",
            "clock_partition": "clock_selection",
            "clock_top1_min": 0.70,
            "clock_surface_top1_min": 0.60,
            "clock_median_gap_min": 0.05,
            "clock_relative_norm_min": 0.001,
            "clock_label_only_top1_max": 0.30,
            "clock_generic_equality_top1_max": 0.30,
            "clock_permuted_top1_max": 0.30,
            "persistence_layers": 3,
            "selection": "earliest start of a three-consecutive-layer run passing every clock criterion; confirmation and holdout are never used for selection",
            "confirmation_top1_min": 0.70,
            "holdout_top1_min": 0.70,
            "transfer_surface_top1_min": 0.60,
            "transfer_median_gap_min": 0.05,
            "transfer_label_only_top1_max": 0.30,
            "transfer_generic_equality_top1_max": 0.30,
            "transfer_permuted_top1_max": 0.30,
            "numeric_relative_l2_p95_max": 1e-5,
            "numeric_relative_l2_max": 1e-4,
            "layer0_relative_norm_max": 1e-5,
            "single_model_authorization": True,
            "cross_model_minimum": 2,
        },
        "causal_gate": {
            "layers": ["tau_minus_1", "tau", "tau_plus_2", "last"],
            "roles": ["target", "tested_family", "answer_boundary"],
            "primary_site": "tested_family",
            "target_partitions": ["confirmation", "holdout"],
            "donor_partition": "prototype_discovery",
            "arms": [
                "baseline",
                "self",
                "same_label_true_to_false",
                "same_label_false_to_true",
                "wrong_pair_same_label",
                "generic_truth",
                "label_only",
                "norm_matched_random",
            ],
            "damage_or_gain_median_min": 0.5,
            "direction_fraction_min": 0.75,
            "flip_fraction_min": 0.30,
            "correct_over_each_null_median_min": 0.25,
            "correct_over_each_null_win_min": 0.65,
            "self_max_abs_margin_diff_max": 1e-4,
            "tau_must_outperform_last_or_last_only_interpretation": "If only last passes, register output-preparation sufficiency, not formation-event causality.",
            "single_model_authorization": True,
            "cross_model_minimum": 2,
        },
        "branching": {
            "behavior_model_fail": "close that model before hidden states",
            "behavior_model_pass": "run that model's all-layer core and null field",
            "formation_model_fail": "close that model without causal work",
            "formation_model_pass": "run frozen layer-comparison causal panel for that model",
            "causal_model_fail": "close at typed causal boundary",
            "causal_model_pass": "register model-specific typed relation-use evidence",
            "all_models_fail": "close C050",
            "cross_model_claim": "requires two independent model-specific passes",
        },
        "stop_rule": "After reveal, do not change material, partition, model, panel, null, threshold, persistence, layer rule, role, donor, or branch.",
        "parameter_boundary": "No parameter, attention-head, MLP, dictionary-learning, or sparse-feature search is authorized.",
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1348_c050_behavior"
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
                "power": power,
                "contract_sha256": protocol["contract_sha256"],
                "authorization": protocol["authorization"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
