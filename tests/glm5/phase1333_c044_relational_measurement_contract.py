#!/usr/bin/env python3
"""Phase1333: freeze C044 execution-conditioned relational measurement campaign."""
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
from model_utils import MODEL_CONFIGS  # noqa: E402

PHASE, CAMPAIGN = 1333, "C044"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1333_c044_relational_measurement_contract_audit.py"
PARENT = T / "result/phase1332_c043_bf16_numeric_qualification"
OUT = T / "result/phase1333_c044_relational_measurement_contract"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "confirmation", "holdout")
FAMILIES = ("bird", "garment", "musical_instrument", "furniture")
LABELS = {
    "bird": "bird",
    "garment": "garment",
    "musical_instrument": "musical instrument",
    "furniture": "furniture",
}
WORDS = {
    "discovery": {
        "bird": ("sparrow", "falcon", "parrot", "ostrich"),
        "garment": ("shirt", "jacket", "sweater", "trousers"),
        "musical_instrument": ("violin", "trumpet", "piano", "flute"),
        "furniture": ("chair", "sofa", "bookcase", "wardrobe"),
    },
    "confirmation": {
        "bird": ("owl", "raven", "pelican", "stork"),
        "garment": ("blouse", "cardigan", "overcoat", "pajamas"),
        "musical_instrument": ("clarinet", "trombone", "banjo", "ukulele"),
        "furniture": ("armchair", "nightstand", "sideboard", "recliner"),
    },
    "holdout": {
        "bird": ("swan", "duck", "goose", "flamingo"),
        "garment": ("raincoat", "waistcoat", "bathrobe", "sweatshirt"),
        "musical_instrument": ("accordion", "xylophone", "saxophone", "harmonica"),
        "furniture": ("ottoman", "loveseat", "headboard", "footstool"),
    },
}
BINARY_SURFACES = {
    "ordinary_kind": "In ordinary English, is a {word} a kind of {label}? Answer yes or no.",
    "statement_truth": "Consider the statement: A {word} is a {label}. Is that statement true? Answer yes or no.",
    "lexical_check": "Would a standard dictionary classify a {word} as a {label}? Answer yes or no.",
}
CHOICE_SURFACES = {
    "direct_choice": "Which category best fits a {word}: {choices}? Output only the category.",
    "dictionary_choice": "A dictionary must file the word {word} under one heading. Choose from {choices}. Output only the heading.",
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
SYSTEM_BINARY = "Answer the ordinary-English category question. Output only yes or no."
SYSTEM_CATEGORY = "Use ordinary word meanings. Output only one requested category label and no explanation."
SYSTEM_CONTEXT = "Read the sentence. Output only OK."


def concept_graph() -> dict[str, Any]:
    values = []
    for partition in PARTITIONS:
        for family in FAMILIES:
            for word in WORDS[partition][family]:
                values.append({
                    "word": word,
                    "family": family,
                    "label": LABELS[family],
                    "partition": partition,
                    "part_of_speech": "common_noun",
                    "ordinary_sense_relation": f"{word} is ordinarily classifiable as {LABELS[family]}",
                })
    return {"schema": "c044.fresh_common_noun_graph.v1", "concepts": values}


def wrong_family(family: str, local_index: int, surface_index: int) -> str:
    current = FAMILIES.index(family)
    return FAMILIES[(current + 1 + (local_index + surface_index) % 3) % 4]


def rotated_candidates(gold_family: str, concept_index: int, surface_index: int) -> tuple[list[str], int]:
    gold_position = (concept_index * len(CHOICE_SURFACES) + surface_index) % 4
    others = [family for family in FAMILIES if family != gold_family]
    values: list[str | None] = [None] * 4
    values[gold_position] = gold_family
    cursor = 0
    for position in range(4):
        if values[position] is None:
            values[position] = others[cursor]
            cursor += 1
    families = [str(value) for value in values]
    return [LABELS[family] for family in families], gold_position


def build_material() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    binary: list[dict[str, Any]] = []
    choice: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    case_index = 0
    choice_index = 0
    generation_index = 0
    context_index = 0
    concepts = concept_graph()["concepts"]
    choices_text = ", ".join(LABELS[family] for family in FAMILIES)
    for concept_index, concept in enumerate(concepts):
        word, family, partition = concept["word"], concept["family"], concept["partition"]
        local_index = WORDS[partition][family].index(word)
        for surface_index, (surface, template) in enumerate(BINARY_SURFACES.items()):
            for truth in (True, False):
                tested = family if truth else wrong_family(family, local_index, surface_index)
                binary.append({
                    "case_id": f"c044-b-{case_index:04d}",
                    "interface": "binary",
                    "partition": partition,
                    "surface": surface,
                    "target": word,
                    "target_family": family,
                    "tested_family": tested,
                    "truth": truth,
                    "pair_key": f"{partition}:{word}:{surface}",
                    "prompt": template.format(word=word, label=LABELS[tested]),
                    "candidates": ["yes", "no"],
                    "gold_position": 0 if truth else 1,
                    "gold_value": "yes" if truth else "no",
                })
                case_index += 1
        for surface_index, (surface, template) in enumerate(CHOICE_SURFACES.items()):
            candidates, gold_position = rotated_candidates(family, concept_index, surface_index)
            choice.append({
                "case_id": f"c044-c-{choice_index:04d}",
                "interface": "choice",
                "partition": partition,
                "surface": surface,
                "target": word,
                "target_family": family,
                "prompt": template.format(word=word, choices=choices_text),
                "candidates": candidates,
                "gold_position": gold_position,
                "gold_value": LABELS[family],
            })
            choice_index += 1
        for surface, template in GENERATION_SURFACES.items():
            generation.append({
                "case_id": f"c044-g-{generation_index:04d}",
                "interface": "generation",
                "partition": partition,
                "surface": surface,
                "target": word,
                "target_family": family,
                "prompt": template.format(word=word),
                "gold_value": LABELS[family],
                "accepted_normalized_outputs": [LABELS[family]],
            })
            generation_index += 1
        for surface, template in CONTEXT_SURFACES.items():
            contexts.append({
                "case_id": f"c044-h-{context_index:04d}",
                "partition": partition,
                "surface": surface,
                "target": word,
                "target_family": family,
                "text": template.format(word=word),
            })
            context_index += 1
    behavior = binary + choice + generation
    return behavior, contexts, binary


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


def compile_model(model_name: str, behavior: list[dict[str, Any]], contexts: list[dict[str, Any]]):
    tokenizer = load_tokenizer(model_name)
    compiled_behavior = []
    for row in behavior:
        system = SYSTEM_BINARY if row["interface"] == "binary" else SYSTEM_CATEGORY
        prompt_ids = core.chat_ids(tokenizer, system, row["prompt"])
        item = {"case_id": row["case_id"], "interface": row["interface"], "prompt_ids": prompt_ids}
        if "candidates" in row:
            item["candidate_ids"] = [
                [int(value) for value in tokenizer.encode(candidate, add_special_tokens=False)]
                for candidate in row["candidates"]
            ]
        compiled_behavior.append(item)
    compiled_contexts = []
    for row in contexts:
        prompt_ids = core.chat_ids(tokenizer, SYSTEM_CONTEXT, row["text"])
        needles = [
            [int(value) for value in tokenizer.encode(form, add_special_tokens=False)]
            for form in (row["target"], " " + row["target"])
        ]
        span = core.locate_last_subsequence(prompt_ids, needles)
        compiled_contexts.append({
            "case_id": row["case_id"],
            "prompt_ids": prompt_ids,
            "target_span": span,
            "boundary_position": len(prompt_ids) - 1,
        })
    return tokenizer, compiled_behavior, compiled_contexts


def old_targets() -> set[str]:
    values: set[str] = set()
    for path in (
        T / "result/phase1329_c042_relational_ecology_contract/material/frozen_concept_graph.json",
        T / "result/phase1331_c043_native_relational_contract/material/frozen_concept_graph.json",
    ):
        values.update(item["word"] for item in core.load(path)["concepts"])
    return values


def audit_material(graph: dict[str, Any], behavior: list[dict[str, Any]], contexts: list[dict[str, Any]], compiled: dict[str, Any]):
    concepts = graph["concepts"]
    binary = [row for row in behavior if row["interface"] == "binary"]
    choice = [row for row in behavior if row["interface"] == "choice"]
    generation = [row for row in behavior if row["interface"] == "generation"]
    checks: dict[str, bool] = {}
    checks["concept_count"] = len(concepts) == 48 and len({row["word"] for row in concepts}) == 48
    checks["fresh_from_c042_c043"] = not ({row["word"] for row in concepts} & old_targets())
    checks["balanced_graph"] = all(
        sum(row["partition"] == partition and row["family"] == family for row in concepts) == 4
        for partition in PARTITIONS for family in FAMILIES
    )
    checks["counts"] = len(binary) == 288 and len(choice) == 96 and len(generation) == 96 and len(contexts) == 144
    checks["binary_balance"] = (
        Counter(row["gold_value"] for row in binary) == {"yes": 144, "no": 144}
        and all(Counter(row["gold_value"] for row in binary if row["surface"] == surface) == {"yes": 48, "no": 48}
                for surface in BINARY_SURFACES)
    )
    checks["binary_pairs"] = all(
        len(values) == 2 and {row["truth"] for row in values} == {True, False}
        for values in _groups(binary, "pair_key").values()
    )
    checks["choice_positions"] = Counter(row["gold_position"] for row in choice) == {0: 24, 1: 24, 2: 24, 3: 24}
    checks["interface_family_balance"] = all(
        sum(row["target_family"] == family for row in choice) == 24
        and sum(row["target_family"] == family for row in generation) == 24
        for family in FAMILIES
    )
    checks["semantic_uniqueness"] = all(
        row["word"].islower() and row["label"] == LABELS[row["family"]]
        and row["word"] not in {value.replace(" ", "") for value in LABELS.values()}
        for row in concepts
    )
    checks["naturalness"] = all(
        row["prompt"].endswith(("no.", "category.", "heading.", "name."))
        and "  " not in row["prompt"] for row in behavior
    ) and all(row["text"].endswith(".") and "  " not in row["text"] for row in contexts)
    checks["zero_models"] = (
        max(Counter(row["gold_value"] for row in binary).values()) / len(binary) == 0.5
        and max(Counter(row["gold_position"] for row in choice).values()) / len(choice) == 0.25
        and max(Counter(row["gold_value"] for row in generation).values()) / len(generation) == 0.25
    )
    for model_name in MODELS:
        rows_b = compiled[model_name]["behavior"]
        rows_h = compiled[model_name]["contexts"]
        binary_compiled = [row for row in rows_b if row["interface"] == "binary"]
        checks[f"{model_name}_compiled"] = len(rows_b) == 480 and len(rows_h) == 144
        checks[f"{model_name}_binary_tokens"] = all(
            len(row["candidate_ids"]) == 2 and len(row["candidate_ids"][0]) == len(row["candidate_ids"][1]) == 1
            for row in binary_compiled
        )
        checks[f"{model_name}_spans"] = all(
            row["target_span"] and max(row["target_span"]) < row["boundary_position"] < len(row["prompt_ids"])
            for row in rows_h
        )
    return checks


def _groups(values: list[dict[str, Any]], key: str):
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in values:
        output[value[key]].append(value)
    return output


def build(force: bool) -> None:
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "close_c043_numeric_ineligible" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1332 is not the audited terminal parent")
    final_path = OUT / "analysis/final.json"
    if final_path.exists() and not force:
        raise RuntimeError(f"{final_path} exists")

    graph = concept_graph()
    behavior, contexts, binary = build_material()
    compiled: dict[str, Any] = {}
    token_audit: dict[str, Any] = {}
    for model_name in MODELS:
        tokenizer, compiled_behavior, compiled_contexts = compile_model(model_name, behavior, contexts)
        compiled[model_name] = {"behavior": compiled_behavior, "contexts": compiled_contexts}
        token_audit[model_name] = {
            "tokenizer_class": type(tokenizer).__name__,
            "binary_candidate_token_lengths": sorted({
                len(candidate) for row in compiled_behavior if row["interface"] == "binary"
                for candidate in row["candidate_ids"]
            }),
            "choice_candidate_token_lengths": sorted({
                len(candidate) for row in compiled_behavior if row["interface"] == "choice"
                for candidate in row["candidate_ids"]
            }),
        }

    checks = audit_material(graph, behavior, contexts, compiled)
    if not all(checks.values()):
        raise RuntimeError(f"pre-model material audit failed: {[key for key, value in checks.items() if not value]}")
    sentinel_ids = [
        row["case_id"] for row in binary
        if row["surface"] == "ordinary_kind"
        and WORDS[row["partition"]][row["target_family"]].index(row["target"]) < 2
    ]
    hidden_sentinel_ids = [
        row["case_id"] for row in contexts
        if WORDS[row["partition"]][row["target_family"]].index(row["target"]) == 0
    ]
    if len(sentinel_ids) != 48 or len(hidden_sentinel_ids) != 36:
        raise RuntimeError("frozen sentinel construction mismatch")

    core.save(OUT / "material/frozen_concept_graph.json", graph)
    core.write_rows(OUT / "material/frozen_behavior_cases.jsonl", behavior)
    core.write_rows(OUT / "material/frozen_context_cases.jsonl", contexts)
    for model_name in MODELS:
        core.write_rows(OUT / f"compiled/{model_name}_behavior.jsonl", compiled[model_name]["behavior"])
        core.write_rows(OUT / f"compiled/{model_name}_context.jsonl", compiled[model_name]["contexts"])
    audit = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "zero_models": {"binary_constant": 0.5, "choice_position": 0.25, "generation_family": 0.25},
        "token_audit": token_audit,
        "human_blind_naturalness": "not_available; claims restricted to curated controlled English",
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", audit)

    protocol: dict[str, Any] = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "phase1333.c044.relational_measurement_contract.v1",
        "research_object": "execution-conditioned multi-interface noun-family behavior and full-dimensional relation fields",
        "claim_boundary": {
            "allowed": "full-resolution relational coding is a falsifiable descriptive candidate",
            "not_assumed": [
                "relational coding exists", "PCA caused historical failures", "single parameters are semantic atoms",
                "descriptive relation geometry is causally used",
            ],
        },
        "parent_terminal_sha256": core.sha(PARENT / "analysis/final.json"),
        "material": {
            "graph_sha256": core.sha(OUT / "material/frozen_concept_graph.json"),
            "behavior_sha256": core.sha(OUT / "material/frozen_behavior_cases.jsonl"),
            "context_sha256": core.sha(OUT / "material/frozen_context_cases.jsonl"),
            "partitions": list(PARTITIONS), "families": list(FAMILIES),
            "binary_surfaces": list(BINARY_SURFACES), "choice_surfaces": list(CHOICE_SURFACES),
            "generation_surfaces": list(GENERATION_SURFACES), "context_surfaces": list(CONTEXT_SURFACES),
            "fresh_from_campaigns": ["C042", "C043"],
        },
        "models": [{
            "name": model, "path": MODEL_CONFIGS[model]["path"], "dtype": "bfloat16",
            "quantization": "none", "execution": "sequential_cuda_then_release",
        } for model in MODELS],
        "numeric": {
            "case_ids": sentinel_ids,
            "standard_execution": "right-padded fixed-width cohort batch8 with explicit position_ids",
            "conditions": ["solo_fixed_width", "replicated_batch8", "cohort_batch8", "cohort_batch8_repeat"],
            "diagnostics": ["absolute_score_drift", "candidate_common_drift", "absolute_margin_drift"],
            "gate": {
                "finite_fraction_min": 1.0,
                "shape_rank_agreement_min": 0.98,
                "composition_rank_agreement_min": 0.98,
                "shape_normalized_margin_drift_p95_max": 0.10,
                "composition_normalized_margin_drift_p95_max": 0.10,
                "shape_normalized_margin_drift_max": 0.50,
                "composition_normalized_margin_drift_max": 0.50,
                "repeat_max_abs_score_diff_max": 1e-6,
                "minimum_authorized_models": 2,
            },
            "failure": "fewer than two models closes C044 before behavior and hidden states",
        },
        "behavior": {
            "binary_gate": {"accuracy_min": 0.85, "partition_min": 0.80, "surface_min": 0.80,
                            "polarity_min": 0.80, "paired_success_min": 0.75, "median_margin_min": 0.50},
            "choice_gate": {"accuracy_min": 0.80, "partition_min": 0.70, "surface_min": 0.70,
                            "median_margin_min": 0.25},
            "generation_gate": {"exact_normalized_accuracy_min": 0.70, "partition_min": 0.60,
                                "surface_min": 0.60},
            "minimum_authorized_models": 2,
            "failure": "fewer than two multi-interface-qualified models closes C044 before hidden states",
        },
        "hidden_numeric": {
            "case_ids": hidden_sentinel_ids,
            "normalized_depths": [0.0, 0.25, 0.5, 0.75, 1.0],
            "positions": ["target_span_mean", "boundary"],
            "gate": {"finite_fraction_min": 1.0, "relative_l2_median_max": 0.01,
                     "relative_l2_p95_max": 0.03, "relation_distance_cosine_min": 0.98,
                     "passing_depths_min": 4},
            "failure": "model excluded from descriptive relation field",
        },
        "relation": {
            "storage": "lossless float32 representations of full BF16 vectors; no PCA/sketch/SAE/hidden-dimension averaging",
            "gate": {"family_knn_k": 3, "embedding_family_purity_min": 0.45,
                     "hidden_family_purity_min": 0.55, "cross_surface_distance_cosine_min": 0.70,
                     "semantic_over_char_advantage_min": 0.10, "cross_model_distance_cosine_min": 0.35,
                     "passing_depths_min": 3, "minimum_authorized_models": 2},
            "pass": "authorize known-truth reparameterizable parameter-camera calibration only",
            "failure": "close C044 without natural-model layer/head/parameter rescue search",
        },
        "parameter_boundary": "No natural-model single-parameter, sparse-dictionary, attention-hotspot, or post-hoc rescue search is authorized.",
        "stop_rule": "No threshold, object, material, model vote, interface, or branch may change after unblinding.",
        "script_sha256": core.sha(SCRIPT),
        "auditor_sha256": core.sha(AUDITOR),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1334_c044_numeric_factorial"
    core.save(OUT / "protocol/preregistration.json", protocol)
    final = {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": True,
             "authorization": protocol["authorization"], "contract_sha256": protocol["contract_sha256"],
             "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(final_path, final)
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args().force)
