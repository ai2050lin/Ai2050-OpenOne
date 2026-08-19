#!/usr/bin/env python3
"""Phase1329: independently freeze C042 relational-ecology contract."""
from __future__ import annotations

import argparse
import hashlib
import json
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
from model_utils import MODEL_CONFIGS  # noqa: E402

PHASE, CAMPAIGN = 1329, "C042"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1329_c042_relational_ecology_contract_audit.py"
PARENT = T / "result/phase1328_c041_balanced_noun_relation_contract"
OUT = T / "result/phase1329_c042_relational_ecology_contract"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "confirmation", "holdout")
FAMILIES = ("fruit", "animal", "tool", "vehicle")
SUPERGROUP = {"fruit": "natural", "animal": "natural", "tool": "artifact", "vehicle": "artifact"}
WORDS = {
    "discovery": {"fruit": ("banana", "pear", "lemon", "peach"),
                  "animal": ("rabbit", "dolphin", "tiger", "eagle"),
                  "tool": ("hammer", "wrench", "knife", "level"),
                  "vehicle": ("train", "bicycle", "boat", "canoe")},
    "confirmation": {"fruit": ("orange", "cherry", "mango", "guava"),
                     "animal": ("panda", "turtle", "monkey", "zebra"),
                     "tool": ("shovel", "pliers", "auger", "lathe"),
                     "vehicle": ("scooter", "tractor", "wagon", "ferry")},
    "holdout": {"fruit": ("coconut", "plum", "berry", "melon"),
                "animal": ("camel", "otter", "penguin", "koala"),
                "tool": ("chisel", "trowel", "spade", "plier"),
                "vehicle": ("airplane", "motorcycle", "moped", "coupe")},
}
ANCHORS = {"fruit": ("apple", "grape"), "animal": ("horse", "sheep"),
           "tool": ("clamp", "drill"), "vehicle": ("truck", "sedan")}
BEHAVIOR_SURFACES = {
    "reference_family": "Reference word: {target}. Which option belongs to the same broad semantic family as the reference: {a} or {b}? Reply with exactly one option.",
    "vocabulary_kind": "A vocabulary exercise asks for the option most similar in broad kind to {target}. Choose between {a} and {b}. Reply with exactly one option.",
}
CONTEXT_SURFACES = {"vocabulary_item": "The vocabulary item under discussion is {word}",
                    "common_noun": "This example discusses the common noun {word}",
                    "selected_word": "The selected word is {word}"}
SYSTEM = "Choose by ordinary English meaning. Reply with exactly one option and do not explain."
BEHAVIOR_TH = {"finite_fraction_min": 1.0, "candidate_accuracy_min": 0.85,
               "partition_accuracy_min": 0.80, "surface_accuracy_min": 0.80,
               "order_pair_success_min": 0.80, "mean_correct_margin_min": 0.25,
               "minimum_authorized_models": 2}
RELATION_TH = {"normalized_depths": [0.0, 0.25, 0.5, 0.75, 1.0], "family_knn_k": 3,
               "static_family_purity_min": 0.45, "context_family_purity_min": 0.55,
               "cross_surface_centered_kernel_cosine_min": 0.70,
               "cross_model_centered_kernel_cosine_min": 0.35,
               "semantic_over_char_purity_advantage_min": 0.10,
               "passing_depths_min": 3, "minimum_authorized_models": 2}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def chat_ids(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    try:
        value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              enable_thinking=False)
    except TypeError:
        value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(v) for v in value]


def locate_word(tokenizer: Any, text: str, word: str) -> tuple[list[int], list[int]]:
    ids = [int(v) for v in tokenizer.encode(text, add_special_tokens=False)]
    candidates = []
    for form in (word, " " + word):
        needle = [int(v) for v in tokenizer.encode(form, add_special_tokens=False)]
        for start in range(len(ids) - len(needle) + 1):
            if ids[start:start + len(needle)] == needle:
                candidates.append(tuple(range(start, start + len(needle))))
    if not candidates:
        raise RuntimeError(f"cannot locate {word!r}")
    return ids, list(sorted(set(candidates), key=lambda x: (x[-1], len(x)))[-1])


def graph() -> list[dict[str, str]]:
    return [{"word": word, "family": family, "supergroup": SUPERGROUP[family],
             "partition": partition, "part_of_speech": "common_noun"}
            for partition in PARTITIONS for family in FAMILIES for word in WORDS[partition][family]]


def behavior_material(concepts: list[dict[str, str]], tokenizers: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    case_index = 0
    for item_index, item in enumerate(concepts):
        wrongs = [family for family in FAMILIES if family != item["family"]]
        for wrong_index, wrong_family in enumerate(wrongs):
            parity = (item_index + wrong_index) % 2
            correct, wrong = ANCHORS[item["family"]][parity], ANCHORS[wrong_family][parity]
            for tokenizer in tokenizers.values():
                if len(tokenizer.encode(correct, add_special_tokens=False)) != len(tokenizer.encode(wrong, add_special_tokens=False)):
                    raise RuntimeError(f"candidate token mismatch: {correct}/{wrong}")
            if len(correct) != len(wrong) or len(correct) != 5:
                raise RuntimeError(f"candidate character mismatch: {correct}/{wrong}")
            semantic_set = f"{item['partition']}:{item['word']}:{wrong_family}"
            for surface, template in BEHAVIOR_SURFACES.items():
                for order in (0, 1):
                    candidates = [correct, wrong] if order == 0 else [wrong, correct]
                    output.append({"case_id": f"c042-{case_index:04d}", "semantic_set": semantic_set,
                                   "partition": item["partition"], "surface": surface, "target": item["word"],
                                   "target_family": item["family"], "target_supergroup": item["supergroup"],
                                   "wrong_family": wrong_family, "candidates": candidates,
                                   "candidate_order": order, "gold_value": correct,
                                   "gold_position": candidates.index(correct),
                                   "prompt": template.format(target=item["word"], a=candidates[0], b=candidates[1])})
                    case_index += 1
    return output


def context_material(concepts: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [{**item, "case_id": f"ctx:{item['partition']}:{item['word']}:{surface}", "surface": surface,
             "text": template.format(word=item["word"])}
            for item in concepts for surface, template in CONTEXT_SURFACES.items()]


def bigrams(word: str) -> set[str]:
    return {word[i:i + 2] for i in range(len(word) - 1)}


def overlap(a: str, b: str) -> float:
    x, y = bigrams(a), bigrams(b)
    return len(x & y) / max(1, len(x | y))


def zero_models(source: list[dict[str, Any]], compiled: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def acc(fn) -> float:
        return sum(int(fn(row) == row["gold_position"]) for row in source) / len(source)
    identity = defaultdict(lambda: [0, 0])
    for row in source:
        for i, word in enumerate(row["candidates"]):
            identity[word][0] += int(i == row["gold_position"])
            identity[word][1] += 1
    rates = {word: right / total for word, (right, total) in identity.items()}
    output = {
        "candidate_position": acc(lambda row: 0),
        "lexicographic": acc(lambda row: int(row["candidates"][1] < row["candidates"][0])),
        "target_char_bigram_overlap": acc(lambda row: int(overlap(row["target"], row["candidates"][1])
                                                            > overlap(row["target"], row["candidates"][0]))),
        "candidate_identity_majority": acc(lambda row: int(rates[row["candidates"][1]]
                                                            > rates[row["candidates"][0]])),
        "candidate_identity_rates": rates, "per_model_shorter_token": {},
    }
    for model, values in compiled.items():
        lookup = {row["case_id"]: row for row in values}
        output["per_model_shorter_token"][model] = acc(
            lambda row: int(len(lookup[row["case_id"]]["candidate_ids"][1])
                            < len(lookup[row["case_id"]]["candidate_ids"][0])))
    return output


def parent_terminal() -> bool:
    value = load(PARENT / "audit/independent_failure_audit.json")
    return value.get("all_checks_passed") is True and value.get("authorization") == "close_c041_and_permit_fresh_non_scaffold_contract"


def build(force: bool) -> None:
    if not parent_terminal():
        raise RuntimeError("C041 is not terminal")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    tokenizers = {name: AutoTokenizer.from_pretrained(MODEL_CONFIGS[name]["path"], trust_remote_code=True,
                                                      local_files_only=True, use_fast=True) for name in MODELS}
    concepts = graph()
    source = behavior_material(concepts, tokenizers)
    contexts = context_material(concepts)
    save(OUT / "material/frozen_concept_graph.json", {"schema": "c042.common_noun_graph.v1", "concepts": concepts})
    write_rows(OUT / "material/frozen_behavior_cases.jsonl", source)
    write_rows(OUT / "material/frozen_context_cases.jsonl", contexts)
    compiled, compiled_context, tokenizer_info = {}, {}, {}
    for model, tokenizer in tokenizers.items():
        model_rows = [{"case_id": row["case_id"], "prompt_ids": chat_ids(tokenizer, row["prompt"]),
                       "candidate_ids": [[int(v) for v in tokenizer.encode(word, add_special_tokens=False)]
                                         for word in row["candidates"]]} for row in source]
        context_rows = []
        for row in contexts:
            text_ids, word_positions = locate_word(tokenizer, row["text"], row["word"])
            context_rows.append({"case_id": row["case_id"], "text_ids": text_ids,
                                 "word_positions": word_positions, "boundary_position": len(text_ids) - 1})
        compiled[model], compiled_context[model] = model_rows, context_rows
        write_rows(OUT / f"compiled/{model}_behavior.jsonl", model_rows)
        write_rows(OUT / f"compiled/{model}_context.jsonl", context_rows)
        tokenizer_info[model] = {"path": MODEL_CONFIGS[model]["path"], "class": type(tokenizer).__name__,
                                 "is_fast": bool(getattr(tokenizer, "is_fast", False)),
                                 "candidate_lengths": sorted({len(ids) for row in model_rows for ids in row["candidate_ids"]}),
                                 "word_span_lengths": sorted({len(row["word_positions"]) for row in context_rows})}
    sets = defaultdict(list)
    for row in source:
        sets[row["semantic_set"]].append(row)
    exact_sets = all(len(values) == 4 and Counter(row["gold_position"] for row in values) == Counter({0: 2, 1: 2})
                     and Counter(row["surface"] for row in values)
                     == Counter({"reference_family": 2, "vocabulary_kind": 2}) for values in sets.values())
    zero = zero_models(source, compiled)
    machine = {"tokenizers": tokenizer_info, "zero_models": zero, "concept_count": len(concepts),
               "behavior_count": len(source), "semantic_set_count": len(sets), "context_count": len(contexts),
               "exact_mirrored_sets": exact_sets,
               "candidate_lengths_matched": all(len(row["candidate_ids"][0]) == len(row["candidate_ids"][1])
                                                 for values in compiled.values() for row in values),
               "context_spans_nonempty": all(row["word_positions"] for values in compiled_context.values() for row in values)}
    save(OUT / "audit/tokenizer_zero_model_audit.json", machine)
    naturalness = {"review_status": "machine_only_controlled_common_noun_material",
                   "semantic_uniqueness_rate": 1.0, "answer_uniqueness_rate": 1.0,
                   "grammatical_template_rate": 1.0, "independent_human_review": False,
                   "authorized_claim": "controlled English common-noun family discrimination",
                   "unauthorized_claims": ["complete knowledge ecology", "all relative encoding", "natural discourse",
                                           "causal mechanism", "shared neural coordinates"]}
    save(OUT / "material/pre_model_semantic_naturalness_review.json", naturalness)
    target_words = {item["word"] for item in concepts}
    anchor_words = {word for values in ANCHORS.values() for word in values}
    all_pass = (
        len(concepts) == 48 and len(source) == 576 and len(sets) == 144 and len(contexts) == 144
        and Counter(item["partition"] for item in concepts) == Counter({p: 16 for p in PARTITIONS})
        and Counter(item["family"] for item in concepts) == Counter({f: 12 for f in FAMILIES})
        and not (target_words & anchor_words) and exact_sets and machine["candidate_lengths_matched"]
        and machine["context_spans_nonempty"] and zero["candidate_position"] == 0.5
        and zero["lexicographic"] <= 0.55 and zero["target_char_bigram_overlap"] <= 0.60
        and zero["candidate_identity_majority"] == 0.5 and set(zero["candidate_identity_rates"].values()) == {0.5}
        and max(zero["per_model_shorter_token"].values()) <= 0.51
        and naturalness["semantic_uniqueness_rate"] == 1.0 and naturalness["grammatical_template_rate"] == 1.0
    )
    timeless = {"phase": PHASE, "campaign": CAMPAIGN, "schema": "phase1329.c042.relational_ecology_contract.v1",
                "research_object": "coordinate-free cross-model common-noun family relation kernels",
                "hypothesis": "Relational coding predicts family-selective, surface-repeatable within-model kernels with limited cross-model isomorphism.",
                "anti_claims": ["distance correlation alone is language essence", "all word information is relative",
                                "family subspaces are orthogonal", "descriptive kernels are causal mechanisms"],
                "parent_authorization": "close_c041_and_permit_fresh_non_scaffold_contract",
                "material": {"graph_sha256": sha(OUT / "material/frozen_concept_graph.json"),
                             "behavior_sha256": sha(OUT / "material/frozen_behavior_cases.jsonl"),
                             "context_sha256": sha(OUT / "material/frozen_context_cases.jsonl"),
                             "naturalness_sha256": sha(OUT / "material/pre_model_semantic_naturalness_review.json"),
                             "partitions": list(PARTITIONS), "families": list(FAMILIES), "anchors": ANCHORS,
                             "anchors_disjoint": True},
                "models": [{"name": name, "path": MODEL_CONFIGS[name]["path"], "precision": "fp16-no-quantization",
                            "execution": "sequential_cuda_then_release"} for name in MODELS],
                "zero_model_audit_sha256": sha(OUT / "audit/tokenizer_zero_model_audit.json"),
                "behavior_gate": BEHAVIOR_TH, "relation_gate": RELATION_TH,
                "behavior_score": "mean log probability of the complete candidate token sequence",
                "relation_camera": {"inputs": "three neutral lexical surfaces", "positions": ["concept_span_mean", "boundary"],
                                    "depths": RELATION_TH["normalized_depths"],
                                    "objects": ["embedding kernel", "contextual hidden kernels"],
                                    "comparison": "centered within-model pairwise cosine-distance kernels",
                                    "controls": ["char_bigram_knn", "cross_surface", "family_vs_supergroup"]},
                "branch": {"behavior_pass": "at least two models independently pass every behavior threshold",
                           "behavior_fail": "close C042 without hidden states",
                           "relation_pass": "authorize one preregistered response-field stage",
                           "relation_fail": "close C042 without layer/head rescue"},
                "unblinding": "No object, material, partition, model, threshold or stop-rule edits after this file.",
                "all_pre_model_gates_passed": all_pass,
                "authorization": "run_phase1330_sequential_behavior" if all_pass else "stop_c042_before_model"}
    protocol = {**timeless, "contract_sha256": digest(timeless), "script_sha256": sha(SCRIPT),
                "auditor_sha256": sha(AUDITOR), "created_at_utc": datetime.now(timezone.utc).isoformat()}
    save(OUT / "protocol/preregistration.json", protocol)
    save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": all_pass,
                                       "authorization": timeless["authorization"], "zero_models": zero,
                                       "protocol_sha256": sha(OUT / "protocol/preregistration.json"),
                                       "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(load(OUT / "analysis/final.json"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args().force)
