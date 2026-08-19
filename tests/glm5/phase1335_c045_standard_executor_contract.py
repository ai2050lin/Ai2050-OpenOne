#!/usr/bin/env python3
"""Phase1335: freeze C045 standard-executor language relation campaign."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1331_relational_measurement_core as core  # noqa: E402
import phase1333_c044_relational_measurement_contract as compiler  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402

PHASE, CAMPAIGN = 1335, "C045"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1335_c045_standard_executor_contract_audit.py"
COMPILER = T / "phase1333_c044_relational_measurement_contract.py"
PARENT = T / "result/phase1334_c044_numeric_factorial"
OUT = T / "result/phase1335_c045_standard_executor_contract"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "confirmation", "holdout")
FAMILIES = ("fish", "flower", "footwear", "kitchen_utensil")
LABELS = {"fish": "fish", "flower": "flower", "footwear": "footwear", "kitchen_utensil": "kitchen utensil"}
WORDS = {
    "discovery": {
        "fish": ("salmon", "tuna", "trout", "cod"),
        "flower": ("rose", "tulip", "daisy", "lily"),
        "footwear": ("shoe", "boot", "sandal", "slipper"),
        "kitchen_utensil": ("spoon", "fork", "spatula", "whisk"),
    },
    "confirmation": {
        "fish": ("herring", "sardine", "mackerel", "catfish"),
        "flower": ("orchid", "sunflower", "daffodil", "carnation"),
        "footwear": ("sneaker", "loafer", "moccasin", "clog"),
        "kitchen_utensil": ("ladle", "peeler", "grater", "colander"),
    },
    "holdout": {
        "fish": ("swordfish", "angelfish", "seahorse", "anchovy"),
        "flower": ("marigold", "hibiscus", "camellia", "zinnia"),
        "footwear": ("flip-flop", "galosh", "espadrille", "ballet shoe"),
        "kitchen_utensil": ("corkscrew", "strainer", "measuring cup", "rolling pin"),
    },
}
BINARY_SURFACES = {
    "ordinary_kind": "In ordinary English, is a {word} a kind of {label}? Answer yes or no.",
    "statement_truth": "Consider the statement: A {word} is a {label}. Is that statement true? Answer yes or no.",
    "lexical_check": "Would a standard dictionary classify a {word} as a {label}? Answer yes or no.",
}
CHOICE_SURFACES = {
    "direct_choice": "Which category best fits a {word}: {choices}? Output only the category.",
    "dictionary_choice": "A dictionary must file the term {word} under one heading. Choose from {choices}. Output only the heading.",
}
GENERATION_SURFACES = {
    "category_generation": "Name the ordinary category of a {word}. Output only the category name.",
    "heading_generation": "What dictionary heading best fits a {word}? Output only the heading.",
}
CONTEXT_SURFACES = {
    "passage": "I read a short passage about the {word}.",
    "guide": "The guide briefly mentioned the {word}.",
    "photograph": "A clear photograph showed the {word}.",
}


def graph() -> dict[str, Any]:
    values = []
    for partition in PARTITIONS:
        for family in FAMILIES:
            for word in WORDS[partition][family]:
                values.append({"word": word, "family": family, "label": LABELS[family], "partition": partition,
                               "part_of_speech": "common_noun_or_noun_phrase",
                               "ordinary_sense_relation": f"{word} is ordinarily classifiable as {LABELS[family]}"})
    return {"schema": "c045.fresh_common_noun_graph.v1", "concepts": values}


def wrong_family(family: str, local_index: int, surface_index: int) -> str:
    current = FAMILIES.index(family)
    return FAMILIES[(current + 1 + (local_index + surface_index) % 3) % 4]


def candidate_order(gold: str, concept_index: int, surface_index: int):
    position = (concept_index * 2 + surface_index) % 4
    result: list[str | None] = [None] * 4
    result[position] = gold
    others = [family for family in FAMILIES if family != gold]
    cursor = 0
    for index in range(4):
        if result[index] is None:
            result[index] = others[cursor]; cursor += 1
    return [LABELS[str(value)] for value in result], position


def material():
    behavior, contexts, binary = [], [], []
    bi = ci = gi = hi = 0
    concepts = graph()["concepts"]
    choices = ", ".join(LABELS[family] for family in FAMILIES)
    for concept_index, concept in enumerate(concepts):
        word, family, partition = concept["word"], concept["family"], concept["partition"]
        local = WORDS[partition][family].index(word)
        for surface_index, (surface, template) in enumerate(BINARY_SURFACES.items()):
            for truth in (True, False):
                tested = family if truth else wrong_family(family, local, surface_index)
                row = {"case_id": f"c045-b-{bi:04d}", "interface": "binary", "partition": partition,
                       "surface": surface, "target": word, "target_family": family, "tested_family": tested,
                       "truth": truth, "pair_key": f"{partition}:{word}:{surface}",
                       "prompt": template.format(word=word, label=LABELS[tested]), "candidates": ["yes", "no"],
                       "gold_position": 0 if truth else 1, "gold_value": "yes" if truth else "no"}
                behavior.append(row); binary.append(row); bi += 1
        for surface_index, (surface, template) in enumerate(CHOICE_SURFACES.items()):
            candidates, position = candidate_order(family, concept_index, surface_index)
            behavior.append({"case_id": f"c045-c-{ci:04d}", "interface": "choice", "partition": partition,
                             "surface": surface, "target": word, "target_family": family,
                             "prompt": template.format(word=word, choices=choices), "candidates": candidates,
                             "gold_position": position, "gold_value": LABELS[family]}); ci += 1
        for surface, template in GENERATION_SURFACES.items():
            behavior.append({"case_id": f"c045-g-{gi:04d}", "interface": "generation", "partition": partition,
                             "surface": surface, "target": word, "target_family": family,
                             "prompt": template.format(word=word), "gold_value": LABELS[family],
                             "accepted_normalized_outputs": [LABELS[family]]}); gi += 1
        for surface, template in CONTEXT_SURFACES.items():
            contexts.append({"case_id": f"c045-h-{hi:04d}", "partition": partition, "surface": surface,
                             "target": word, "target_family": family, "text": template.format(word=word)}); hi += 1
    return behavior, contexts, binary


def prior_words() -> set[str]:
    result = set()
    for path in (
        T / "result/phase1329_c042_relational_ecology_contract/material/frozen_concept_graph.json",
        T / "result/phase1331_c043_native_relational_contract/material/frozen_concept_graph.json",
        T / "result/phase1333_c044_relational_measurement_contract/material/frozen_concept_graph.json",
    ):
        result.update(row["word"] for row in core.load(path)["concepts"])
    return result


def build(force: bool) -> None:
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "close_c044_numeric_factorial" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1334 is not the audited terminal parent")
    final_path = OUT / "analysis/final.json"
    if final_path.exists() and not force: raise RuntimeError(f"{final_path} exists")
    frozen_graph = graph(); behavior, contexts, binary = material()
    compiled = {}
    for model in MODELS:
        tokenizer, b, h = compiler.compile_model(model, behavior, contexts)
        compiled[model] = {"behavior": b, "contexts": h, "tokenizer_class": type(tokenizer).__name__}
    concepts = frozen_graph["concepts"]
    choice = [row for row in behavior if row["interface"] == "choice"]
    generation = [row for row in behavior if row["interface"] == "generation"]
    checks = {
        "fresh": not ({row["word"] for row in concepts} & prior_words()),
        "graph": len(concepts) == 48 and len({row["word"] for row in concepts}) == 48,
        "graph_balance": all(sum(row["partition"] == p and row["family"] == f for row in concepts) == 4
                             for p in PARTITIONS for f in FAMILIES),
        "counts": len(binary) == 288 and len(choice) == 96 and len(generation) == 96 and len(contexts) == 144,
        "binary_balance": Counter(row["gold_value"] for row in binary) == {"yes": 144, "no": 144},
        "choice_balance": Counter(row["gold_position"] for row in choice) == {0: 24, 1: 24, 2: 24, 3: 24},
        "generation_balance": Counter(row["gold_value"] for row in generation) == {
            "fish": 24, "flower": 24, "footwear": 24, "kitchen utensil": 24},
        "semantic_uniqueness": all(row["ordinary_sense_relation"] and row["word"].strip() == row["word"] for row in concepts),
        "naturalness": all("  " not in row["prompt"] and row["prompt"].endswith(("no.", "category.", "heading.", "name."))
                           for row in behavior) and all(row["text"].endswith(".") for row in contexts),
    }
    for model in MODELS:
        checks[f"{model}_compiled"] = len(compiled[model]["behavior"]) == 480 and len(compiled[model]["contexts"]) == 144
        checks[f"{model}_binary_tokens"] = all(all(len(c) == 1 for c in row["candidate_ids"])
                                                for row in compiled[model]["behavior"] if row["interface"] == "binary")
        checks[f"{model}_spans"] = all(row["target_span"] and max(row["target_span"]) < row["boundary_position"]
                                       for row in compiled[model]["contexts"])
    if not all(checks.values()): raise RuntimeError([key for key, value in checks.items() if not value])
    core.save(OUT / "material/frozen_concept_graph.json", frozen_graph)
    core.write_rows(OUT / "material/frozen_behavior_cases.jsonl", behavior)
    core.write_rows(OUT / "material/frozen_context_cases.jsonl", contexts)
    for model in MODELS:
        core.write_rows(OUT / f"compiled/{model}_behavior.jsonl", compiled[model]["behavior"])
        core.write_rows(OUT / f"compiled/{model}_context.jsonl", compiled[model]["contexts"])
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json",
              {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()),
               "total": len(checks), "all_checks_passed": all(checks.values()),
               "zero_models": {"binary": .5, "choice": .25, "generation": .25},
               "human_blind_naturalness": "not_available; scope limited to curated controlled English"})
    sentinel_ids = [row["case_id"] for row in binary if row["surface"] == "ordinary_kind"
                    and WORDS[row["partition"]][row["target_family"]].index(row["target"]) < 2]
    hidden_ids = [row["case_id"] for row in contexts
                  if WORDS[row["partition"]][row["target_family"]].index(row["target"]) == 0]
    protocol: dict[str, Any] = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "phase1335.c045.standard_executor_contract.v1",
        "research_object": "multi-interface noun-family behavior and full-dimensional relation fields under one standard physical executor",
        "claim_boundary": {"allowed": "mechanism claims are indexed by model and frozen executor",
                           "not_assumed": ["cross-shape invariance", "relational coding", "causal use of geometry", "single-parameter semantics"]},
        "parent_terminal_sha256": core.sha(PARENT / "analysis/final.json"),
        "material": {"graph_sha256": core.sha(OUT / "material/frozen_concept_graph.json"),
                     "behavior_sha256": core.sha(OUT / "material/frozen_behavior_cases.jsonl"),
                     "context_sha256": core.sha(OUT / "material/frozen_context_cases.jsonl"),
                     "partitions": list(PARTITIONS), "families": list(FAMILIES),
                     "fresh_from": ["C042", "C043", "C044"]},
        "models": [{"name": model, "path": MODEL_CONFIGS[model]["path"], "dtype": "bfloat16",
                    "quantization": "none", "execution": "sequential_cuda_then_release"} for model in MODELS],
        "standard_executor": {"batch_size": 8, "fixed_width_per_model": True, "padding_side": "right",
                              "explicit_position_ids": True, "device_map_frozen_per_model": True,
                              "cohort_rule": "source order chunks of eight", "cross_shape_status": "engineering_diagnostic_only"},
        "executor_gate": {"case_ids": sentinel_ids, "conditions": ["cohort_a", "cohort_permuted", "cohort_a_repeat"],
                          "finite_fraction_min": 1.0, "permuted_rank_agreement_min": 1.0,
                          "permuted_max_abs_score_diff_max": 1e-6, "repeat_max_abs_score_diff_max": 1e-6,
                          "minimum_authorized_models": 2,
                          "failure": "fewer than two models closes C045 before behavior and hidden states"},
        "behavior": {
            "binary_gate": {"accuracy_min": .85, "partition_min": .80, "surface_min": .80,
                            "polarity_min": .80, "paired_success_min": .75, "median_margin_min": .50},
            "choice_gate": {"accuracy_min": .80, "partition_min": .70, "surface_min": .70, "median_margin_min": .25},
            "generation_gate": {"exact_normalized_accuracy_min": .70, "partition_min": .60, "surface_min": .60},
            "minimum_authorized_models": 2,
            "failure": "fewer than two multi-interface models closes C045 before hidden states"},
        "hidden_numeric": {"case_ids": hidden_ids, "normalized_depths": [0, .25, .5, .75, 1],
                           "positions": ["target_span_mean", "boundary"],
                           "conditions": ["cohort_a", "cohort_permuted", "cohort_a_repeat"],
                           "gate": {"finite_fraction_min": 1.0, "relative_l2_median_max": .01,
                                    "relative_l2_p95_max": .03, "relation_distance_cosine_min": .98,
                                    "passing_depths_min": 4}},
        "relation": {"storage": "lossless float32 representation of complete BF16 vectors; no primary compression",
                     "gate": {"family_knn_k": 3, "embedding_family_purity_min": .45,
                              "hidden_family_purity_min": .55, "cross_surface_distance_cosine_min": .70,
                              "semantic_over_char_advantage_min": .10, "cross_model_distance_cosine_min": .35,
                              "passing_depths_min": 3, "minimum_authorized_models": 2},
                     "pass": "authorize known-truth reparameterizable parameter-camera calibration only",
                     "failure": "close C045 without post-hoc layer/head/parameter rescue"},
        "parameter_boundary": "No natural-model parameter, attention-hotspot, sparse-dictionary, or rescue search is authorized.",
        "stop_rule": "After unblinding, do not change material, executor, threshold, model vote, interface, or branch.",
        "script_sha256": core.sha(SCRIPT), "auditor_sha256": core.sha(AUDITOR),
        "compiler_sha256": core.sha(COMPILER), "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol); protocol["authorization"] = "run_phase1336_c045_standard_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    final = {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True,
             "authorization": protocol["authorization"], "contract_sha256": protocol["contract_sha256"],
             "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(final_path, final); print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--force", action="store_true"); build(parser.parse_args().force)
