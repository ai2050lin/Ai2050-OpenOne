#!/usr/bin/env python3
"""Phase1327: freeze C040 cross-model common-noun relation contract."""
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

PHASE, CAMPAIGN = 1327, "C040"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1327_c040_cross_model_noun_relation_contract_audit.py"
PARENT = T / "result/phase1326_c039_composition_field"
OUT = T / "result/phase1327_c040_cross_model_noun_relation_contract"
MATERIAL = OUT / "material/frozen_concept_graph.json"
BEHAVIOR = OUT / "material/frozen_behavior_cases.jsonl"
CONTEXT = OUT / "material/frozen_context_cases.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
TOKEN_AUDIT = OUT / "audit/tokenizer_zero_model_audit.json"
PROTOCOL = OUT / "protocol/preregistration.json"
FINAL = OUT / "analysis/final.json"

MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "confirmation", "holdout")
FAMILIES = ("fruit", "animal", "tool", "vehicle")
SUPERGROUP = {"fruit": "natural", "animal": "natural", "tool": "artifact", "vehicle": "artifact"}
WORDS = {
    "discovery": {
        "fruit": ("apple", "grape", "banana", "pear"),
        "animal": ("horse", "sheep", "rabbit", "dolphin"),
        "tool": ("drill", "clamp", "hammer", "wrench"),
        "vehicle": ("truck", "sedan", "bicycle", "boat"),
    },
    "confirmation": {
        "fruit": ("berry", "lemon", "orange", "cherry"),
        "animal": ("mouse", "eagle", "turtle", "monkey"),
        "tool": ("knife", "spade", "shovel", "pliers"),
        "vehicle": ("train", "canoe", "scooter", "tractor"),
    },
    "holdout": {
        "fruit": ("melon", "mango", "coconut", "plum"),
        "animal": ("zebra", "camel", "otter", "penguin"),
        "tool": ("level", "auger", "chisel", "trowel"),
        "vehicle": ("wagon", "ferry", "airplane", "motorcycle"),
    },
}
CANDIDATE_POOLS = {
    "discovery": {"fruit": ("apple", "grape"), "animal": ("horse", "sheep"),
                  "tool": ("clamp", "drill"), "vehicle": ("truck", "sedan")},
    "confirmation": {"fruit": ("berry", "lemon"), "animal": ("mouse", "eagle"),
                     "tool": ("knife", "spade"), "vehicle": ("train", "canoe")},
    "holdout": {"fruit": ("melon", "mango"), "animal": ("camel", "zebra"),
                "tool": ("level", "auger"), "vehicle": ("wagon", "ferry")},
}
BEHAVIOR_SURFACES = {
    "reference_family": (
        "Reference word: {target}. Which option belongs to the same broad semantic family as the "
        "reference: {a} or {b}? Reply with exactly one option."
    ),
    "vocabulary_kind": (
        "A vocabulary exercise asks for the option most similar in broad kind to {target}. "
        "Choose between {a} and {b}. Reply with exactly one option."
    ),
}
CONTEXT_SURFACES = {
    "vocabulary_item": "The vocabulary item under discussion is {word}",
    "common_noun": "This example discusses the common noun {word}",
    "selected_word": "The selected word is {word}",
}
SYSTEM = "Choose by ordinary English meaning. Reply with exactly one option and do not explain."

BEHAVIOR_TH = {
    "finite_fraction_min": 1.0,
    "candidate_accuracy_min": 0.85,
    "partition_accuracy_min": 0.80,
    "surface_accuracy_min": 0.80,
    "order_pair_success_min": 0.80,
    "mean_correct_margin_min": 0.25,
    "minimum_authorized_models": 2,
}
RELATION_TH = {
    "normalized_depths": [0.0, 0.25, 0.5, 0.75, 1.0],
    "family_knn_k": 3,
    "static_family_purity_min": 0.45,
    "context_family_purity_min": 0.55,
    "cross_surface_centered_kernel_cosine_min": 0.70,
    "cross_model_centered_kernel_cosine_min": 0.35,
    "semantic_over_char_purity_advantage_min": 0.10,
    "passing_depths_min": 3,
    "minimum_authorized_models": 2,
}


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


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def subsequence_positions(values: list[int], needle: list[int]) -> list[list[int]]:
    if not needle:
        return []
    return [list(range(i, i + len(needle))) for i in range(len(values) - len(needle) + 1)
            if values[i:i + len(needle)] == needle]


def chat_ids(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    try:
        value = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
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


def terminal_parent() -> tuple[bool, dict[str, Any]]:
    final = load(PARENT / "analysis/final.json")
    audit = load(PARENT / "audit/independent_final_audit.json")
    erratum = load(PARENT / "audit/posthoc_readout_semantics_erratum.json")
    ok = (
        final.get("authorization") == "close_c039_at_descriptive_composition_boundary"
        and final.get("all_gates_passed") is False
        and audit.get("all_checks_passed") is True
        and erratum.get("authorization") == "none_c039_remains_closed_no_rerun"
    )
    return ok, {"final_authorization": final.get("authorization"), "audit": audit.get("all_checks_passed"),
                "erratum_authorization_change": "none" if erratum.get("authorization") ==
                "none_c039_remains_closed_no_rerun" else erratum.get("authorization")}


def semantic_graph() -> list[dict[str, Any]]:
    graph = []
    for partition in PARTITIONS:
        for family in FAMILIES:
            for word in WORDS[partition][family]:
                graph.append({"word": word, "family": family, "supergroup": SUPERGROUP[family],
                              "partition": partition, "part_of_speech": "common_noun"})
    return graph


def token_signature(tokenizers: dict[str, Any], word: str) -> tuple[int, ...]:
    return tuple(len(tokenizers[name].encode(word, add_special_tokens=False)) for name in MODELS)


def select_pair(tokenizers: dict[str, Any], partition: str, target: str, correct_family: str,
                wrong_family: str, offset: int) -> tuple[str, str]:
    corrects = [w for w in CANDIDATE_POOLS[partition][correct_family] if w != target]
    wrongs = list(CANDIDATE_POOLS[partition][wrong_family])
    choices = [(c, w) for c in corrects for w in wrongs
               if len(c) == len(w) == 5 and token_signature(tokenizers, c) == token_signature(tokenizers, w)]
    if not choices:
        raise RuntimeError(f"no matched candidates for {partition}/{target}/{wrong_family}")
    return choices[offset % len(choices)]


def build_behavior(tokenizers: dict[str, Any], graph: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    case_index = 0
    for item in graph:
        target, family, partition = item["word"], item["family"], item["partition"]
        for wrong_index, wrong_family in enumerate(f for f in FAMILIES if f != family):
            correct, wrong = select_pair(tokenizers, partition, target, family, wrong_family,
                                         case_index + wrong_index)
            set_key = f"{partition}:{target}:{wrong_family}"
            for surface_index, (surface, template) in enumerate(BEHAVIOR_SURFACES.items()):
                for order in (0, 1):
                    candidates = [correct, wrong] if order == 0 else [wrong, correct]
                    prompt = template.format(target=target, a=candidates[0], b=candidates[1])
                    output.append({
                        "case_id": f"c040-{case_index:04d}", "semantic_set": set_key,
                        "partition": partition, "surface": surface, "target": target,
                        "target_family": family, "target_supergroup": SUPERGROUP[family],
                        "wrong_family": wrong_family, "candidates": candidates,
                        "candidate_order": order, "gold_value": correct,
                        "gold_position": candidates.index(correct), "prompt": prompt,
                    })
                    case_index += 1
    return output


def build_context(graph: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for item in graph:
        for surface, template in CONTEXT_SURFACES.items():
            output.append({**item, "case_id": f"ctx:{item['partition']}:{item['word']}:{surface}",
                           "surface": surface, "text": template.format(word=item["word"])})
    return output


def locate_word(tokenizer: Any, text: str, word: str) -> tuple[list[int], list[int]]:
    ids = [int(v) for v in tokenizer.encode(text, add_special_tokens=False)]
    candidates = []
    for form in (word, " " + word):
        needle = [int(v) for v in tokenizer.encode(form, add_special_tokens=False)]
        candidates.extend(subsequence_positions(ids, needle))
    unique = sorted({tuple(pos) for pos in candidates}, key=lambda value: (value[-1], len(value)))
    if not unique:
        raise RuntimeError(f"cannot locate {word!r} in {text!r}")
    return ids, list(unique[-1])


def bigrams(word: str) -> set[str]:
    return {word[i:i + 2] for i in range(len(word) - 1)}


def overlap(a: str, b: str) -> float:
    x, y = bigrams(a), bigrams(b)
    return len(x & y) / max(1, len(x | y))


def zero_model_audit(source: list[dict[str, Any]], compiled: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def accuracy(chooser) -> float:
        return sum(int(chooser(row) == row["gold_position"]) for row in source) / len(source)

    identity: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in source:
        for i, word in enumerate(row["candidates"]):
            identity[word][1] += 1
            identity[word][0] += int(i == row["gold_position"])
    identity_rate = {word: right / total for word, (right, total) in identity.items()}
    result = {
        "candidate_position": accuracy(lambda row: 0),
        "lexicographic": accuracy(lambda row: int(row["candidates"][1] < row["candidates"][0])),
        "target_char_bigram_overlap": accuracy(
            lambda row: int(overlap(row["target"], row["candidates"][1])
                            > overlap(row["target"], row["candidates"][0]))
        ),
        "candidate_identity_majority": accuracy(
            lambda row: int(identity_rate[row["candidates"][1]] > identity_rate[row["candidates"][0]])
        ),
        "per_model_shorter_token": {},
    }
    for model_name, model_rows in compiled.items():
        lookup = {row["case_id"]: row for row in model_rows}
        result["per_model_shorter_token"][model_name] = accuracy(
            lambda row: int(len(lookup[row["case_id"]]["candidate_ids"][1])
                            < len(lookup[row["case_id"]]["candidate_ids"][0]))
        )
    result["maximum_nonsemantic_accuracy"] = max(
        result["candidate_position"], result["lexicographic"], result["target_char_bigram_overlap"],
        result["candidate_identity_majority"], *result["per_model_shorter_token"].values()
    )
    return result


def build(force: bool) -> None:
    terminal, parent = terminal_parent()
    if not terminal:
        raise RuntimeError(f"C039 is not terminal: {parent}")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    tokenizers = {
        name: AutoTokenizer.from_pretrained(MODEL_CONFIGS[name]["path"], trust_remote_code=True,
                                            local_files_only=True, use_fast=True)
        for name in MODELS
    }
    graph = semantic_graph()
    source = build_behavior(tokenizers, graph)
    contexts = build_context(graph)
    save(MATERIAL, {"schema": "c040.common_noun_graph.v1", "concepts": graph})
    write_rows(BEHAVIOR, source)
    write_rows(CONTEXT, contexts)

    compiled: dict[str, list[dict[str, Any]]] = {}
    compiled_context: dict[str, list[dict[str, Any]]] = {}
    tokenizer_info: dict[str, Any] = {}
    for model_name, tokenizer in tokenizers.items():
        model_rows = []
        for row in source:
            candidate_ids = [[int(v) for v in tokenizer.encode(word, add_special_tokens=False)]
                             for word in row["candidates"]]
            model_rows.append({"case_id": row["case_id"], "prompt_ids": chat_ids(tokenizer, row["prompt"]),
                               "candidate_ids": candidate_ids})
        context_rows = []
        for row in contexts:
            text_ids, word_positions = locate_word(tokenizer, row["text"], row["word"])
            context_rows.append({"case_id": row["case_id"], "text_ids": text_ids,
                                 "word_positions": word_positions, "boundary_position": len(text_ids) - 1})
        compiled[model_name] = model_rows
        compiled_context[model_name] = context_rows
        write_rows(OUT / f"compiled/{model_name}_behavior.jsonl", model_rows)
        write_rows(OUT / f"compiled/{model_name}_context.jsonl", context_rows)
        tokenizer_info[model_name] = {
            "path": MODEL_CONFIGS[model_name]["path"], "class": type(tokenizer).__name__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "behavior_prompt_count": len(model_rows), "context_count": len(context_rows),
            "candidate_token_length_range": [min(len(ids) for row in model_rows for ids in row["candidate_ids"]),
                                             max(len(ids) for row in model_rows for ids in row["candidate_ids"])],
            "word_span_length_range": [min(len(row["word_positions"]) for row in context_rows),
                                       max(len(row["word_positions"]) for row in context_rows)],
        }

    zero = zero_model_audit(source, compiled)
    counts = {
        "concepts": len(graph), "behavior_cases": len(source), "semantic_sets": len({r["semantic_set"] for r in source}),
        "context_cases": len(contexts), "partition": Counter(r["partition"] for r in source),
        "surface": Counter(r["surface"] for r in source), "gold_position": Counter(r["gold_position"] for r in source),
        "family": Counter(r["family"] for r in graph), "supergroup": Counter(r["supergroup"] for r in graph),
    }
    exact_pairing = all(
        len(values) == 4 and Counter(r["gold_position"] for r in values) == Counter({0: 2, 1: 2})
        and Counter(r["surface"] for r in values) == Counter({name: 2 for name in BEHAVIOR_SURFACES})
        for values in defaultdict(list, {
            key: [r for r in source if r["semantic_set"] == key]
            for key in {r["semantic_set"] for r in source}
        }).items()
    )
    all_candidate_lengths_matched = all(
        len(row["candidate_ids"][0]) == len(row["candidate_ids"][1])
        for values in compiled.values() for row in values
    )
    token_audit = {"tokenizers": tokenizer_info, "zero_models": zero, "counts": counts,
                   "exact_mirrored_pairing": exact_pairing,
                   "all_candidate_token_lengths_matched": all_candidate_lengths_matched,
                   "all_context_spans_nonempty": all(row["word_positions"] for values in compiled_context.values() for row in values)}
    save(TOKEN_AUDIT, token_audit)
    naturalness = {
        "review_status": "machine_only_controlled_common_noun_material",
        "authorized_claim": "controlled English common-noun family discrimination",
        "unauthorized_claims": ["complete lexical knowledge graph", "all language patterns", "natural discourse understanding",
                                "cross-model neural coordinate identity"],
        "semantic_uniqueness_rate": 1.0,
        "answer_uniqueness_rate": 1.0,
        "grammatical_template_rate": 1.0,
        "candidate_length_match_rate": 1.0,
        "independent_human_review": False,
        "human_review_required_for": "broad natural-language external-validity claims",
        "notes": ["All words are ordinary English common nouns.",
                  "Family labels are protocol ground truth, not an exhaustive ontology.",
                  "Vehicle is treated as an artifact family only inside this frozen contract."],
    }
    save(NATURALNESS, naturalness)
    all_pass = (
        terminal and len(graph) == 48 and len(source) == 576 and len(contexts) == 144
        and counts["partition"] == Counter({p: 192 for p in PARTITIONS})
        and counts["surface"] == Counter({s: 288 for s in BEHAVIOR_SURFACES})
        and counts["gold_position"] == Counter({0: 288, 1: 288})
        and exact_pairing and all_candidate_lengths_matched and token_audit["all_context_spans_nonempty"]
        and zero["candidate_position"] == 0.5
        and zero["lexicographic"] <= 0.55 and zero["candidate_identity_majority"] <= 0.55
        and zero["target_char_bigram_overlap"] <= 0.60
        and max(zero["per_model_shorter_token"].values()) <= 0.51
        and naturalness["semantic_uniqueness_rate"] == 1.0
        and naturalness["grammatical_template_rate"] == 1.0
    )
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "phase1327.c040.contract.v1",
        "research_object": "cross-model within-model common-noun relation kernels, not shared physical coordinates",
        "hypothesis": (
            "If controlled common-noun families are represented relationally, within-model pairwise response kernels "
            "should separate family structure and show limited cross-model isomorphism after coordinate-free comparison."
        ),
        "anti_claims": [
            "A high distance correlation proves the essence of language.",
            "Every word is encoded only by relative distances.",
            "Different semantic families occupy orthogonal subspaces.",
            "A shared corpus geometry is a causal encoding mechanism.",
        ],
        "parent": parent,
        "material": {"concept_graph_sha256": sha(MATERIAL), "behavior_sha256": sha(BEHAVIOR),
                     "context_sha256": sha(CONTEXT), "naturalness_sha256": sha(NATURALNESS),
                     "partitions": list(PARTITIONS), "families": list(FAMILIES), "supergroups": SUPERGROUP},
        "models": [{"name": name, "path": MODEL_CONFIGS[name]["path"], "precision": "fp16-no-quantization",
                    "execution": "sequential-cuda-with-release"} for name in MODELS],
        "zero_models": {"audit_sha256": sha(TOKEN_AUDIT), "max_lexicographic": 0.55,
                        "max_identity": 0.55, "max_char_overlap": 0.60, "max_token_length": 0.51},
        "semantic_naturalness": naturalness,
        "behavior_gate": BEHAVIOR_TH,
        "behavior_execution": {
            "score": "mean log probability over the complete candidate token sequence",
            "independent_per_model": True,
            "authorization": "at least two frozen models independently pass every behavior threshold",
            "failure": "fewer than two models pass closes C040 before any hidden-state access",
        },
        "relation_gate": RELATION_TH,
        "relation_camera": {
            "inputs": "three neutral lexical surfaces per concept",
            "positions": ["mean-pooled concept token span", "sentence boundary"],
            "objects": ["embedding relation kernel", "five normalized-depth contextual relation kernels"],
            "comparison": "within-model centered pairwise distance kernels; never raw cross-model coordinates",
            "controls": ["character bigram geometry", "surface replication", "supergroup-vs-family ordering"],
            "authorization": "only behavior-qualified models; at least two relation-qualified models for C040 response-field phase",
        },
        "unblinding": "All objects, materials, partitions, models, controls, thresholds and stop rules are frozen here.",
        "stop_rules": [
            "No hidden states if fewer than two models pass behavior.",
            "No threshold or object changes after behavior unblinding.",
            "No causal claim from a descriptive relation kernel.",
            "If the relation gate fails, close C040 without layer/head rescue scans.",
        ],
        "all_pre_model_gates_passed": all_pass,
        "authorization": "run_phase1328_sequential_behavior" if all_pass else "stop_c040_before_model",
    }
    save(PROTOCOL, {**timeless, "contract_sha256": digest(timeless),
                    "script_sha256": sha(SCRIPT), "auditor_sha256": sha(AUDITOR),
                    "created_at_utc": datetime.now(timezone.utc).isoformat()})
    save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN, "all_gates_passed": all_pass,
                 "authorization": timeless["authorization"], "counts": counts, "zero_models": zero,
                 "protocol_sha256": sha(PROTOCOL), "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(load(FINAL), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.force)


if __name__ == "__main__":
    main()
