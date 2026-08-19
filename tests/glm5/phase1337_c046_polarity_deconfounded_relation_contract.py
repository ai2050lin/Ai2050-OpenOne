#!/usr/bin/env python3
"""Phase1337: freeze C046 polarity-deconfounded noun-relation campaign."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1331_relational_measurement_core as core  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402

PHASE, CAMPAIGN = 1337, "C046"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1337_c046_polarity_deconfounded_relation_contract_audit.py"
PARENT = T / "result/phase1336_c045_standard_behavior"
OUT = T / "result/phase1337_c046_polarity_deconfounded_relation_contract"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "confirmation", "holdout")
FAMILIES = ("mammal", "gemstone", "vehicle", "vegetable")
LABELS = {family: family for family in FAMILIES}
WORDS = {
    "discovery": {
        "mammal": ("porcupine", "elephant", "weasel", "armadillo"),
        "gemstone": ("diamond", "ruby", "sapphire", "emerald"),
        "vehicle": ("roadster", "lorry", "bus", "monorail"),
        "vegetable": ("carrot", "broccoli", "spinach", "cabbage"),
    },
    "confirmation": {
        "mammal": ("hyena", "gazelle", "kangaroo", "bison"),
        "gemstone": ("opal", "topaz", "amethyst", "garnet"),
        "vehicle": ("trolleybus", "van", "tram", "snowmobile"),
        "vegetable": ("radish", "celery", "asparagus", "lettuce"),
    },
    "holdout": {
        "mammal": ("rhinoceros", "manatee", "wombat", "meerkat"),
        "gemstone": ("aquamarine", "peridot", "tourmaline", "moonstone"),
        "vehicle": ("hovercraft", "forklift", "minibus", "rickshaw"),
        "vegetable": ("turnip", "beetroot", "cauliflower", "artichoke"),
    },
}
SURFACES = {
    "noun_class": "Does the noun \"{word}\" ordinarily name a kind of {label}?",
    "dictionary_relation": "In its common everyday sense, is \"{word}\" classified as a {label}?",
    "category_claim": "Consider the category claim: \"{word}\" belongs to the category {label}. Is the claim correct?",
}
CODEBOOKS = {
    "standard": "Output code: use yes when the claim is correct and no when it is incorrect.",
    "reversed": "Output code: use no when the claim is correct and yes when it is incorrect.",
}
SYSTEM = "Follow the stated output code exactly. Evaluate the ordinary-English category claim. Output only yes or no."


def graph() -> dict[str, Any]:
    concepts = []
    for partition in PARTITIONS:
        for family in FAMILIES:
            for word in WORDS[partition][family]:
                concepts.append({
                    "word": word,
                    "family": family,
                    "label": LABELS[family],
                    "partition": partition,
                    "part_of_speech": "common_noun",
                    "ordinary_sense_relation": f"the common noun {word} ordinarily names a kind of {family}",
                })
    return {"schema": "c046.polarity_deconfounded_noun_graph.v1", "concepts": concepts}


def cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for concept in graph()["concepts"]:
        for surface, template in SURFACES.items():
            for tested_family in FAMILIES:
                truth = tested_family == concept["family"]
                claim = template.format(word=concept["word"], label=LABELS[tested_family])
                semantic_key = f'{concept["partition"]}:{concept["word"]}:{surface}:{tested_family}'
                for codebook, instruction in CODEBOOKS.items():
                    gold_yes = truth if codebook == "standard" else not truth
                    rows.append({
                        "case_id": f"c046-b-{index:04d}",
                        "interface": "binary_coded_relation",
                        "partition": concept["partition"],
                        "surface": surface,
                        "target": concept["word"],
                        "target_family": concept["family"],
                        "tested_family": tested_family,
                        "truth": truth,
                        "codebook": codebook,
                        "semantic_key": semantic_key,
                        "quartet_key": f'{concept["partition"]}:{concept["word"]}:{surface}',
                        "prompt": f"{instruction} {claim} Output only yes or no.",
                        "candidates": ["yes", "no"],
                        "gold_position": 0 if gold_yes else 1,
                        "gold_value": "yes" if gold_yes else "no",
                    })
                    index += 1
    return rows


def prior_words() -> set[str]:
    values: set[str] = set()
    paths = list((T / "result").glob("phase13*_c04*_*/material/frozen_concept_graph.json"))
    paths += list((T / "result").glob("phase13*_c04*_*/material/frozen_*graph.json"))
    for path in set(paths):
        try:
            value = core.load(path)
            values.update(str(item["word"]) for item in value.get("concepts", []) if "word" in item)
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    return values


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[model_name]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def span(tokenizer: Any, prompt_ids: list[int], value: str) -> list[int]:
    needles = [[int(token) for token in tokenizer.encode(form, add_special_tokens=False)]
               for form in (value, " " + value)]
    return core.locate_last_subsequence(prompt_ids, needles)


def compile_model(model_name: str, material: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer = load_tokenizer(model_name)
    compiled = []
    for row in material:
        prompt_ids = core.chat_ids(tokenizer, SYSTEM, row["prompt"])
        compiled.append({
            "case_id": row["case_id"],
            "prompt_ids": prompt_ids,
            "candidate_ids": [[int(token) for token in tokenizer.encode(candidate, add_special_tokens=False)]
                              for candidate in row["candidates"]],
            "target_span": span(tokenizer, prompt_ids, row["target"]),
            "tested_family_span": span(tokenizer, prompt_ids, LABELS[row["tested_family"]]),
            "boundary_position": len(prompt_ids) - 1,
        })
    return compiled


def zero_model_scores(material: list[dict[str, Any]]) -> dict[str, float]:
    def accuracy(rule):
        return sum(rule(row) == row["gold_value"] for row in material) / len(material)
    return {
        "always_yes": accuracy(lambda row: "yes"),
        "always_no": accuracy(lambda row: "no"),
        "semantic_truth_ignore_codebook": accuracy(lambda row: "yes" if row["truth"] else "no"),
        "codebook_assume_true": accuracy(lambda row: "yes" if row["codebook"] == "standard" else "no"),
        "codebook_assume_false": accuracy(lambda row: "no" if row["codebook"] == "standard" else "yes"),
    }


def build(force: bool) -> None:
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent.get("authorization") != "close_c045_standard_behavior" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("Phase1336 is not the audited terminal parent")
    final_path = OUT / "analysis/final.json"
    if final_path.exists() and not force:
        raise RuntimeError(f"{final_path} exists")
    material = cases()
    frozen_graph = graph()
    compiled = {model: compile_model(model, material) for model in MODELS}
    zeros = zero_model_scores(material)
    semantic_groups: dict[str, list[dict[str, Any]]] = {}
    for row in material:
        semantic_groups.setdefault(row["semantic_key"], []).append(row)
    checks = {
        "fresh": not ({row["word"] for row in frozen_graph["concepts"]} & prior_words()),
        "graph": len(frozen_graph["concepts"]) == 48 and len({row["word"] for row in frozen_graph["concepts"]}) == 48,
        "graph_balance": all(sum(row["partition"] == p and row["family"] == f for row in frozen_graph["concepts"]) == 4
                             for p in PARTITIONS for f in FAMILIES),
        "count": len(material) == 1152 and len({row["case_id"] for row in material}) == 1152,
        "factorial": all(sum(row[key] == value for row in material) == expected for key, values, expected in (
            ("partition", PARTITIONS, 384), ("surface", tuple(SURFACES), 384),
            ("codebook", tuple(CODEBOOKS), 576)) for value in values),
        "family_test_balance": all(sum(row["tested_family"] == f for row in material) == 288 for f in FAMILIES),
        "truth_counts": Counter(row["truth"] for row in material) == {True: 288, False: 864},
        "gold_balance": Counter(row["gold_value"] for row in material) == {"yes": 576, "no": 576},
        "semantic_pairs": len(semantic_groups) == 576 and all(
            len(group) == 2 and {row["codebook"] for row in group} == set(CODEBOOKS)
            and {row["gold_value"] for row in group} == {"yes", "no"} for group in semantic_groups.values()),
        "truth_definition": all(row["truth"] == (row["tested_family"] == row["target_family"]) for row in material),
        "semantic_uniqueness": all(row["word"].strip() == row["word"] and row["ordinary_sense_relation"]
                                   for row in frozen_graph["concepts"]),
        "naturalness": all("  " not in row["prompt"] and row["prompt"].endswith("yes or no.") for row in material),
        "zero_models": zeros == {"always_yes": .5, "always_no": .5,
                                  "semantic_truth_ignore_codebook": .5,
                                  "codebook_assume_true": .25, "codebook_assume_false": .75},
    }
    for model in MODELS:
        rows = compiled[model]
        checks[f"{model}_compiled"] = len(rows) == len(material) and all(
            a["case_id"] == b["case_id"] for a, b in zip(material, rows))
        checks[f"{model}_candidate_tokens"] = all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in rows)
        checks[f"{model}_spans"] = all(row["target_span"] and row["tested_family_span"]
                                       and max(row["target_span"] + row["tested_family_span"]) < row["boundary_position"]
                                       for row in rows)
    if not all(checks.values()):
        raise RuntimeError([key for key, value in checks.items() if not value])
    core.save(OUT / "material/frozen_concept_graph.json", frozen_graph)
    core.write_rows(OUT / "material/frozen_behavior_cases.jsonl", material)
    for model in MODELS:
        core.write_rows(OUT / f"compiled/{model}_behavior.jsonl", compiled[model])
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "zero_models": zeros,
        "human_blind_naturalness": "not_available; claims limited to curated controlled English",
    })
    sentinel_ids = [row["case_id"] for row in material
                    if row["surface"] == "noun_class" and row["truth"] and row["codebook"] == "standard"]
    protocol: dict[str, Any] = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "phase1337.c046.polarity_deconfounded_relation.v1",
        "research_object": "noun-family truth relation invariant to an explicit normal or reversed yes-no output code",
        "claim_boundary": {
            "allowed": "behavioral and, conditionally, descriptive full-dimensional relation-field claims indexed by model and executor",
            "not_assumed": ["shared ontology", "cross-interface unification", "causal use", "attention or MLP localization", "parameter semantics"],
        },
        "parent_terminal_sha256": core.sha(PARENT / "analysis/final.json"),
        "material": {
            "graph_sha256": core.sha(OUT / "material/frozen_concept_graph.json"),
            "behavior_sha256": core.sha(OUT / "material/frozen_behavior_cases.jsonl"),
            "partitions": list(PARTITIONS), "families": list(FAMILIES), "surfaces": list(SURFACES),
            "codebooks": list(CODEBOOKS), "case_count": len(material), "fresh_from": ["C042", "C043", "C044", "C045"],
        },
        "models": [{"name": model, "path": MODEL_CONFIGS[model]["path"], "dtype": "bfloat16",
                    "quantization": "none", "execution": "sequential_then_release"} for model in MODELS],
        "standard_executor": {"batch_size": 8, "fixed_width_per_model": True, "padding_side": "right",
                              "explicit_position_ids": True, "device_map_frozen_per_model": True,
                              "cohort_rule": "source order chunks of eight"},
        "executor_gate": {"case_ids": sentinel_ids, "finite_fraction_min": 1.0,
                          "permuted_rank_agreement_min": 1.0, "permuted_max_abs_score_diff_max": 1e-6,
                          "repeat_max_abs_score_diff_max": 1e-6, "minimum_authorized_models": 2},
        "behavior_gate": {
            "accuracy_min": .90, "partition_min": .85, "surface_min": .85, "family_min": .85,
            "codebook_min": .85, "truth_min": .85, "truth_codebook_cell_min": .80,
            "semantic_pair_success_min": .80, "median_margin_min": .50,
            "minimum_authorized_models": 2,
            "failure": "fewer than two models closes C046 before hidden-state capture",
        },
        "hidden_numeric_gate": {
            "case_ids": sentinel_ids, "normalized_depths": [0, .25, .5, .75, 1],
            "positions": ["target_span_mean", "tested_family_span_mean", "boundary"],
            "conditions": ["cohort_a", "cohort_permuted", "cohort_a_repeat"],
            "relative_l2_p95_max": 1e-5, "relative_l2_max": 1e-4,
            "minimum_authorized_models": 2,
        },
        "relation_gate": {
            "storage": "float32 lossless representation of selected complete BF16 vectors; no fitted or lossy primary projection",
            "normalized_depths": [0, .25, .5, .75, 1],
            "positions": ["target_span_mean", "tested_family_span_mean", "boundary"],
            "evaluated_depths": [.25, .5, .75, 1],
            "cross_codebook_identity_win_min": .60,
            "truth_contrast_cosine_min": .15,
            "cross_surface_identity_win_min": .50,
            "passing_depths_per_model_min": 2,
            "cross_model_centered_signature_cosine_min": .35,
            "cross_model_over_permuted_gap_min": .10,
            "cross_model_passing_depths_min": 2,
            "minimum_authorized_models": 2,
            "success": "authorize a separate preregistered causal relation-field campaign only",
            "failure": "close C046 without layer, head, MLP, probe, parameter, or rescue search",
        },
        "zero_models": zeros,
        "stop_rule": "After unblinding, do not change material, codebook, model vote, metric, threshold, role, depth, or branch.",
        "parameter_boundary": "No natural-model component, parameter, sparse dictionary, or causal rescue search is authorized in C046.",
        "script_sha256": core.sha(SCRIPT), "auditor_sha256": core.sha(AUDITOR),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1338_c046_deconfounded_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    final = {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True,
             "authorization": protocol["authorization"], "contract_sha256": protocol["contract_sha256"],
             "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(final_path, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args().force)
