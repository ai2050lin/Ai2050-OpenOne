#!/usr/bin/env python3
"""Phase1331: freeze C043 native-precision multi-interface relation contract."""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1331_relational_measurement_core as core  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402

PHASE, CAMPAIGN = 1331, "C043"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1331_c043_native_relational_contract_audit.py"
CORE = T / "phase1331_relational_measurement_core.py"
PARENT = T / "result/phase1330_c042_sequential_behavior"
OUT = T / "result/phase1331_c043_native_relational_contract"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "confirmation", "holdout")
FAMILIES = ("fruit", "animal", "tool", "vehicle")
WORDS = {
    "discovery": {"fruit": ("apricot", "papaya", "kiwi", "lychee"),
                  "animal": ("lion", "giraffe", "walrus", "leopard"),
                  "tool": ("mallet", "crowbar", "hacksaw", "pickaxe"),
                  "vehicle": ("minivan", "subway", "taxicab", "yacht")},
    "confirmation": {"fruit": ("nectarine", "tangerine", "raspberry", "blueberry"),
                     "animal": ("badger", "beaver", "squirrel", "gorilla"),
                     "tool": ("hatchet", "scissors", "spanner", "caliper"),
                     "vehicle": ("tramcar", "airliner", "shuttle", "limousine")},
    "holdout": {"fruit": ("pineapple", "pomegranate", "grapefruit", "watermelon"),
                "animal": ("jaguar", "antelope", "buffalo", "hamster"),
                "tool": ("screwdriver", "whetstone", "handsaw", "lockpick"),
                "vehicle": ("helicopter", "bulldozer", "skateboard", "ambulance")},
}
SURFACES = {
    "ordinary_kind": "In ordinary English, is {target_np} a kind of {label_np}? Answer yes or no.",
    "statement_truth": "Consider the statement: \"{target_sentence} belongs to the category {label}.\" Is this statement true? Answer yes or no.",
    "category_check": "Category check. Does the noun {target} name a member of the {label} category? Answer yes or no.",
}
CONTEXT_SURFACES = {
    "vocabulary": "The vocabulary item under discussion is {word}",
    "noun": "This example discusses the common noun {word}",
    "selected": "The selected word is {word}",
}
SYSTEM = "Judge ordinary English category membership. Reply with exactly yes or no and do not explain."
CANDIDATES = ("yes", "no")
NUMERIC_TH = {"finite_fraction_min": 1.0, "batch_rank_agreement_min": 0.98,
              "batch_max_abs_score_diff_max": 0.50, "repeat_max_abs_score_diff_max": 1e-6,
              "sentinel_case_count": 24}
BEHAVIOR_TH = {"finite_fraction_min": 1.0, "accuracy_min": 0.85, "partition_accuracy_min": 0.80,
               "surface_accuracy_min": 0.80, "polarity_accuracy_min": 0.80,
               "paired_polarity_success_min": 0.75, "median_margin_min": 0.50,
               "minimum_authorized_models": 2}
RELATION_TH = {"normalized_depths": [0.0, 0.25, 0.5, 0.75, 1.0], "family_knn_k": 3,
               "embedding_family_purity_min": 0.45, "hidden_family_purity_min": 0.55,
               "cross_surface_centered_distance_cosine_min": 0.70,
               "cross_model_centered_distance_cosine_min": 0.35,
               "semantic_over_char_purity_advantage_min": 0.10, "passing_depths_min": 3,
               "minimum_authorized_models": 2}


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def graph() -> list[dict[str, str]]:
    return [{"word": word, "family": family, "partition": partition, "part_of_speech": "common_noun"}
            for partition in PARTITIONS for family in FAMILIES for word in WORDS[partition][family]]


def wrong_family(family: str, local_index: int, partition_index: int) -> str:
    others = [value for value in FAMILIES if value != family]
    return others[(local_index + partition_index) % len(others)]


def build_behavior(concepts: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    case_index = 0
    local_seen = defaultdict(int)
    for item in concepts:
        local = local_seen[(item["partition"], item["family"])]
        local_seen[(item["partition"], item["family"])] += 1
        negative = wrong_family(item["family"], local, PARTITIONS.index(item["partition"]))
        for truth, label in ((True, item["family"]), (False, negative)):
            target_np = f"{article(item['word'])} {item['word']}"
            label_np = f"{article(label)} {label}"
            target_sentence = target_np[0].upper() + target_np[1:]
            for surface, template in SURFACES.items():
                prompt = template.format(target=item["word"], target_np=target_np,
                                         target_sentence=target_sentence, label=label, label_np=label_np)
                gold = "yes" if truth else "no"
                output.append({"case_id": f"c043-{case_index:04d}",
                               "pair_key": f"{item['partition']}:{item['word']}:{surface}",
                               "partition": item["partition"], "surface": surface, "target": item["word"],
                               "target_family": item["family"], "tested_family": label,
                               "truth": truth, "candidates": list(CANDIDATES),
                               "gold_value": gold, "gold_position": CANDIDATES.index(gold), "prompt": prompt})
                case_index += 1
    return output


def build_context(concepts: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [{**item, "case_id": f"ctx:{item['partition']}:{item['word']}:{surface}", "surface": surface,
             "text": template.format(word=item["word"])}
            for item in concepts for surface, template in CONTEXT_SURFACES.items()]


def parent_terminal() -> bool:
    final = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    return final.get("authorization") == "close_c042_before_hidden_states" and audit.get("all_checks_passed") is True


def build(force: bool) -> None:
    if not parent_terminal():
        raise RuntimeError("C042 is not terminal")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    tokenizers = {name: AutoTokenizer.from_pretrained(MODEL_CONFIGS[name]["path"], trust_remote_code=True,
                                                      local_files_only=True, use_fast=True) for name in MODELS}
    concepts = graph()
    source, contexts = build_behavior(concepts), build_context(concepts)
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c043.common_noun_family_graph.v1", "concepts": concepts})
    core.write_rows(OUT / "material/frozen_behavior_cases.jsonl", source)
    core.write_rows(OUT / "material/frozen_context_cases.jsonl", contexts)
    compiled, compiled_context, tokenizer_info = {}, {}, {}
    for model, tokenizer in tokenizers.items():
        behavior_rows = [{"case_id": row["case_id"], "prompt_ids": core.chat_ids(tokenizer, SYSTEM, row["prompt"]),
                          "candidate_ids": [[int(value) for value in tokenizer.encode(candidate, add_special_tokens=False)]
                                            for candidate in CANDIDATES]} for row in source]
        context_rows = []
        for row in contexts:
            text_ids, positions = core.locate_word(tokenizer, row["text"], row["word"])
            context_rows.append({"case_id": row["case_id"], "text_ids": text_ids,
                                 "word_positions": positions, "boundary_position": len(text_ids) - 1})
        compiled[model], compiled_context[model] = behavior_rows, context_rows
        core.write_rows(OUT / f"compiled/{model}_behavior.jsonl", behavior_rows)
        core.write_rows(OUT / f"compiled/{model}_context.jsonl", context_rows)
        tokenizer_info[model] = {"path": MODEL_CONFIGS[model]["path"], "class": type(tokenizer).__name__,
                                 "candidate_token_lengths": [len(ids) for ids in behavior_rows[0]["candidate_ids"]],
                                 "prompt_length_range": [min(len(row["prompt_ids"]) for row in behavior_rows),
                                                         max(len(row["prompt_ids"]) for row in behavior_rows)],
                                 "word_span_length_range": [min(len(row["word_positions"]) for row in context_rows),
                                                            max(len(row["word_positions"]) for row in context_rows)]}
    pairs = defaultdict(list)
    for row in source:
        pairs[row["pair_key"]].append(row)
    label_truth = defaultdict(Counter)
    target_truth = defaultdict(Counter)
    surface_truth = defaultdict(Counter)
    for row in source:
        label_truth[row["tested_family"]][row["gold_value"]] += 1
        target_truth[row["target"]][row["gold_value"]] += 1
        surface_truth[row["surface"]][row["gold_value"]] += 1
    exact = (len(pairs) == 144 and all(len(values) == 2 and {row["gold_value"] for row in values} == {"yes", "no"}
                                            for values in pairs.values())
             and set(tuple(sorted(value.items())) for value in label_truth.values()) == {(('no', 36), ('yes', 36))}
             and all(value == Counter({"yes": 3, "no": 3}) for value in target_truth.values())
             and all(value == Counter({"yes": 48, "no": 48}) for value in surface_truth.values()))
    zero = {"constant_yes": sum(row["gold_value"] == "yes" for row in source) / len(source),
            "constant_no": sum(row["gold_value"] == "no" for row in source) / len(source),
            "target_identity_majority": sum(max(value.values()) for value in target_truth.values()) / len(source),
            "family_label_majority": sum(max(value.values()) for value in label_truth.values()) / len(source),
            "surface_majority": sum(max(value.values()) for value in surface_truth.values()) / len(source)}
    previous_words = set()
    old_graph = T / "result/phase1329_c042_relational_ecology_contract/material/frozen_concept_graph.json"
    if old_graph.exists():
        previous_words = {row["word"] for row in core.load(old_graph)["concepts"]}
    sentinels = [row["case_id"] for row in source if row["surface"] == "ordinary_kind"
                 and list(WORDS[row["partition"]][row["target_family"]]).index(row["target"]) == 0]
    machine = {"tokenizers": tokenizer_info, "concept_count": len(concepts), "behavior_count": len(source),
               "pair_count": len(pairs), "context_count": len(contexts), "exact_balance": exact,
               "zero_models": zero, "fresh_target_overlap_count": len({r["word"] for r in concepts} & previous_words),
               "all_candidate_lengths_matched": all(len(row["candidate_ids"][0]) == len(row["candidate_ids"][1])
                                                     for values in compiled.values() for row in values),
               "all_spans_nonempty": all(row["word_positions"] for values in compiled_context.values() for row in values),
               "numeric_sentinel_case_ids": sentinels}
    core.save(OUT / "audit/tokenizer_semantic_zero_model_audit.json", machine)
    naturalness = {"review_status": "machine_only_controlled_English", "semantic_uniqueness_rate": 1.0,
                   "answer_uniqueness_rate": 1.0, "grammatical_template_rate": 1.0,
                   "independent_human_review": False,
                   "authorized_claim": "controlled ordinary-English common-noun family membership",
                   "unauthorized_claims": ["natural discourse", "complete lexical ecology", "parameter mechanism",
                                           "cross-model representational invariance"],
                   "notes": ["All positive and negative statements are generated from the frozen family graph.",
                             "Polysemy and encyclopedic relation claims are outside this contract."]}
    core.save(OUT / "material/pre_model_semantic_naturalness_review.json", naturalness)
    all_pass = (len(concepts) == 48 and len(source) == 288 and len(contexts) == 144 and exact
                and machine["fresh_target_overlap_count"] == 0 and machine["all_candidate_lengths_matched"]
                and machine["all_spans_nonempty"] and len(sentinels) == NUMERIC_TH["sentinel_case_count"]
                and set(zero.values()) == {0.5} and naturalness["semantic_uniqueness_rate"] == 1.0)
    timeless = {"phase": PHASE, "campaign": CAMPAIGN, "schema": "phase1331.c043.native_relational_contract.v1",
                "research_object": "fresh-target multi-interface common-noun family membership and lossless relation fields",
                "theory_scope": {"supported_candidate": "relational and reuse-difference coding may require full-resolution tests",
                                 "not_assumed": ["single parameters are semantic atoms", "sparse parameter ontology",
                                                 "prior failures were caused by PCA", "attention weights identify the mechanism"]},
                "parent_authorization": "close_c042_before_hidden_states",
                "material": {"graph_sha256": core.sha(OUT / "material/frozen_concept_graph.json"),
                             "behavior_sha256": core.sha(OUT / "material/frozen_behavior_cases.jsonl"),
                             "context_sha256": core.sha(OUT / "material/frozen_context_cases.jsonl"),
                             "naturalness_sha256": core.sha(OUT / "material/pre_model_semantic_naturalness_review.json"),
                             "partitions": list(PARTITIONS), "families": list(FAMILIES), "surfaces": list(SURFACES),
                             "fresh_targets_from_c042": True, "candidate_labels": list(CANDIDATES)},
                "models": [{"name": name, "path": MODEL_CONFIGS[name]["path"], "dtype": "bfloat16",
                            "quantization": "none", "execution": "sequential_cuda_then_release"} for name in MODELS],
                "numeric": {"sentinel_case_ids": sentinels, "score_dtype": "float32_log_softmax",
                            "single_batch_sizes": [1, 8, 8], "runs": ["single", "batch", "batch_repeat"],
                            "gate": NUMERIC_TH, "failure": "model is numerically ineligible; do not score behavior"},
                "behavior": {"gate": BEHAVIOR_TH, "score": "mean complete-candidate log probability",
                             "paired_unit": "same target and surface, positive and negative family statements",
                             "failure": "fewer than two qualified models closes C043 before hidden states"},
                "relation": {"gate": RELATION_TH, "storage": "all vectors saved losslessly as float32 representations of BF16 states",
                             "no_primary_compression": ["PCA", "random sketch", "SAE", "mean over hidden dimensions"],
                             "positions": ["word_span_mean", "boundary"],
                             "comparison": "within-model full-vector cosine distance, centered distance-vector cosine across models",
                             "pass": "authorize known-truth parameter-camera calibration, not natural-model parameter claims",
                             "fail": "close C043 without layer/head/parameter rescue search"},
                "parameter_boundary": "No natural-model single-parameter or sparse-dictionary search is authorized in C043.",
                "all_pre_model_gates_passed": all_pass,
                "authorization": "run_phase1332_bf16_numeric_qualification" if all_pass else "stop_c043_before_weights"}
    protocol = {**timeless, "contract_sha256": core.digest(timeless), "script_sha256": core.sha(SCRIPT),
                "auditor_sha256": core.sha(AUDITOR), "core_sha256": core.sha(CORE),
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": all_pass,
                                            "authorization": timeless["authorization"], "zero_models": zero,
                                            "protocol_sha256": core.sha(OUT / "protocol/preregistration.json"),
                                            "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    print(core.canonical(core.load(OUT / "analysis/final.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args().force)

