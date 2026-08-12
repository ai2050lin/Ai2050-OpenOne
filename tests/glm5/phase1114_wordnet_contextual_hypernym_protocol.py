#!/usr/bin/env python3
"""Freeze Phase1114 natural-example contextual noun-sense behavior protocol."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1018_language_pattern_protocol import tokenizer_for
import phase1098_relative_relation_geometry_protocol as relation_tools
import phase1101_relation_identity_routing_protocol as relation_source
import phase1113_wordnet_semantic_quadrant_protocol as wordnet_source


PHASE = 1114
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
SPLITS = ("discovery", "independent_confirmation", "heldout")
ITEMS_PER_SPLIT = 12
SELECTED_ITEM_COUNT = ITEMS_PER_SPLIT * len(SPLITS)
SENSES = (0, 1)
ASSISTANT_PREFILL = "Category:"
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1114_wordnet_contextual_hypernym"
)
WORDNET_ARCHIVE = wordnet_source.WORDNET_ARCHIVE
WORDNET_SHA256 = wordnet_source.WORDNET_SHA256
WORDNET_MEMBERS = wordnet_source.WORDNET_MEMBERS


TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1, 2, 3, 4, 5),
    "independent_confirmation": (6, 7, 8, 9, 10, 11),
    "heldout": (12, 13, 14, 15, 16, 17),
}


TEMPLATES = (
    (
        "Read this ordinary English sentence:\n{sentence}\n"
        "In this use, the noun \"{term}\" is most nearly a kind of what? "
        "Reply with one lowercase noun only."
    ),
    (
        "Sentence in context: {sentence}\n"
        "Complete the semantic category for the noun \"{term}\" as used here. "
        "Give one lowercase noun."
    ),
    (
        "Consider the meaning selected by this sentence:\n{sentence}\n"
        "The intended sense of \"{term}\" belongs most directly under which noun? "
        "Answer with one lowercase word."
    ),
    (
        "Interpret the highlighted noun from its sentence, not from spelling alone.\n"
        "Sentence: {sentence}\nNoun: {term}\n"
        "State its nearest semantic kind using one lowercase noun."
    ),
    (
        "Use the sentence to resolve the noun's intended meaning.\n{sentence}\n"
        "Here, \"{term}\" is a type of what? Return one lowercase noun only."
    ),
    (
        "Context decides which noun sense is active.\nContext: {sentence}\n"
        "For this occurrence of \"{term}\", provide its closest broader noun category "
        "as one lowercase word."
    ),
    (
        "Infer the local sense from the quoted usage.\n\"{sentence}\"\n"
        "Which immediate noun category best fits \"{term}\" in that usage? "
        "Write one lowercase noun."
    ),
    (
        "Resolve this contextual noun meaning:\n{sentence}\n"
        "Name the closest kind of thing denoted by \"{term}\" here. "
        "Use exactly one lowercase noun."
    ),
    (
        "The same spelling can express different noun senses. Read: {sentence}\n"
        "For the sense expressed by \"{term}\", give its nearest broader noun type "
        "in one lowercase word."
    ),
    (
        "Identify the meaning carried by the noun in this natural example.\n"
        "Example: {sentence}\nTarget noun: {term}\n"
        "Respond with its closest semantic category, one lowercase noun only."
    ),
    (
        "Use only the local context to choose the intended noun sense.\n{sentence}\n"
        "What noun kind most directly contains the meaning of \"{term}\" here? "
        "Answer in one lowercase word."
    ),
    (
        "Read the usage before categorizing the word.\nUsage: {sentence}\n"
        "The occurrence of \"{term}\" is most specifically a kind of what? "
        "Return a single lowercase noun."
    ),
    (
        "Interpret this noun occurrence in its sentence:\n{sentence}\n"
        "Give the nearest broader noun meaning for \"{term}\". "
        "Your response must be one lowercase noun."
    ),
    (
        "A sentence selects one meaning of an ambiguous noun.\nSentence: {sentence}\n"
        "Which noun category is closest to the selected meaning of \"{term}\"? "
        "Use one lowercase word."
    ),
    (
        "Determine the contextual sense in the following example.\n{sentence}\n"
        "Classify \"{term}\" under its closest broader noun, replying with one "
        "lowercase noun only."
    ),
    (
        "Read this usage as a normal English sentence: {sentence}\n"
        "For \"{term}\" in this sentence, supply the nearest semantic kind. "
        "Give one lowercase noun."
    ),
    (
        "Disambiguate the noun from context.\nNatural example: {sentence}\n"
        "The intended \"{term}\" sense is a kind of what noun? "
        "Answer with one lowercase word only."
    ),
    (
        "Use the full sentence to infer what the noun denotes.\n{sentence}\n"
        "Name the immediate broader noun category of \"{term}\" in this usage. "
        "Return exactly one lowercase noun."
    ),
)


THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_overall_candidate_accuracy": 0.75,
    "minimum_split_candidate_accuracy": 0.70,
    "minimum_sense_candidate_accuracy": 0.70,
    "minimum_split_sense_candidate_accuracy": 0.65,
    "minimum_template_candidate_accuracy": 0.60,
    "minimum_context_direction_accuracy": 0.80,
    "minimum_split_context_direction_accuracy": 0.75,
    "minimum_template_context_direction_accuracy": 0.65,
    "minimum_bidirectional_pair_accuracy": 0.60,
    "minimum_split_bidirectional_pair_accuracy": 0.50,
    "maximum_sense_accuracy_gap": 0.20,
    "minimum_behavior_qualified_models": 2,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "The verified WordNet source, native-example parser, source-only selection, "
        "split lexical isolation, candidate nonleakage, tokenization, and digests pass."
    ),
    "P2": (
        "At least two FP16/no-quantization models satisfy every case-level and paired "
        "context behavior gate on discovery, independent confirmation, and heldout."
    ),
    "P3": (
        "The paired context effect z0-z1 has the correct sign at or above the frozen "
        "overall, split, and template gates."
    ),
    "P4": (
        "Both members of a sense pair cross their candidate decision boundary at or "
        "above the frozen bidirectional gates, not merely shift in the right direction."
    ),
    "P5": (
        "Static candidate-token preference is canceled only in the paired context "
        "effect. Direct top-token generation remains diagnostic and is not a gate."
    ),
    "P6": (
        "A pass qualifies this contextual hypernym behavior object only. It does not "
        "by itself establish a hidden semantic direction or authorize a confounded scan."
    ),
}


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def exact_count(text: str, term: str) -> int:
    pattern = rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+", text))


def parse_native_examples() -> tuple[dict[str, list[str]], dict[str, Any]]:
    if file_sha256(WORDNET_ARCHIVE) != WORDNET_SHA256:
        raise RuntimeError("WordNet archive SHA256 mismatch")
    with tarfile.open(WORDNET_ARCHIVE, "r:gz") as archive:
        raw = archive.extractfile(WORDNET_MEMBERS["data_noun"]).read()
    examples: dict[str, list[str]] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line or line[0].isspace():
            continue
        main, gloss = line.split("|", 1)
        offset = main.split()[0]
        examples[offset] = [
            value.strip() for value in re.findall(r'"([^"\r\n]+)"', gloss)
        ]
    manifest = {
        "parser": "quoted_examples_from_WordNet-3.0/dict/data.noun",
        "synsets_with_examples": sum(bool(values) for values in examples.values()),
        "example_count": sum(len(values) for values in examples.values()),
        "data_noun_sha256": hashlib.sha256(raw).hexdigest(),
    }
    manifest["example_manifest_digest"] = digest(manifest)
    return examples, manifest


def one_token_all(tokenizers: dict[str, Any], term: str) -> bool:
    return all(
        len(tokenizer.encode(" " + term, add_special_tokens=False)) == 1
        for tokenizer in tokenizers.values()
    )


def eligible_inventory(
    synsets: dict[str, dict[str, Any]],
    lemma_synsets: dict[str, list[str]],
    metadata: dict[str, Any],
    examples: dict[str, list[str]],
    tokenizers: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sense_number = metadata["sense_number"]
    tag_count = metadata["tag_count"]
    rejection: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for base, offsets in sorted(lemma_synsets.items()):
        if not re.fullmatch(r"[a-z]+", base) or not 4 <= len(base) <= 12:
            rejection["base_form"] += 1
            continue
        ordered = sorted(
            set(offsets), key=lambda value: sense_number.get((base, value), 10**9)
        )
        if len(ordered) < 2:
            rejection["fewer_than_two_senses"] += 1
            continue
        sense_offsets = ordered[:2]
        senses = [synsets[offset] for offset in sense_offsets]
        if (
            tag_count.get((base, sense_offsets[0]), 0) < 5
            or tag_count.get((base, sense_offsets[1]), 0) < 1
        ):
            rejection["base_tag_count"] += 1
            continue
        if senses[0]["lexfile_index"] == senses[1]["lexfile_index"]:
            rejection["same_lexicographer_file"] += 1
            continue
        if any(len(sense["hypernyms"]) != 1 for sense in senses):
            rejection["nonunique_direct_hypernym"] += 1
            continue
        hypernym_offsets = [sense["hypernyms"][0] for sense in senses]
        hypernyms = [synsets[offset]["lemmas"][0] for offset in hypernym_offsets]
        if (
            hypernym_offsets[0] == hypernym_offsets[1]
            or hypernyms[0] == hypernyms[1]
            or any(not re.fullmatch(r"[a-z]+", term) for term in hypernyms)
            or len({
                wordnet_source.simple_morphology_root(base),
                *(wordnet_source.simple_morphology_root(term) for term in hypernyms),
            }) != 3
        ):
            rejection["hypernym_identity_or_form"] += 1
            continue
        if not all(one_token_all(tokenizers, term) for term in hypernyms):
            rejection["hypernym_not_one_token_all_models"] += 1
            continue
        if any(
            metadata["tag_count"].get((term, offset), 0) < 1
            for term, offset in zip(hypernyms, hypernym_offsets)
        ):
            rejection["hypernym_tag_count"] += 1
            continue
        if any(
            exact_count(template.format(sentence="", term=base), term) > 0
            for template in TEMPLATES
            for term in hypernyms
        ):
            rejection["hypernym_leaks_from_template"] += 1
            continue
        selected_examples: list[str] = []
        valid_examples: list[list[str]] = []
        for offset in sense_offsets:
            values = [
                example
                for example in examples.get(offset, [])
                if exact_count(example, base) == 1
                and 4 <= word_count(example) <= 22
                and all(exact_count(example, term) == 0 for term in hypernyms)
            ]
            valid_examples.append(values)
            if values:
                selected_examples.append(values[0])
        if any(not values for values in valid_examples):
            rejection["missing_native_example"] += 1
            continue
        if selected_examples[0].casefold() == selected_examples[1].casefold():
            rejection["identical_native_examples"] += 1
            continue
        eligible.append({
            "base": base,
            "sense_offsets": sense_offsets,
            "sense_numbers": [
                sense_number.get((base, offset)) for offset in sense_offsets
            ],
            "lexnames": [sense["lexname"] for sense in senses],
            "hypernym_offsets": hypernym_offsets,
            "hypernyms": hypernyms,
            "examples": selected_examples,
            "available_example_counts": [len(values) for values in valid_examples],
            "base_tag_counts": [
                tag_count.get((base, offset), 0) for offset in sense_offsets
            ],
            "hypernym_tag_counts": [
                tag_count.get((term, offset), 0)
                for term, offset in zip(hypernyms, hypernym_offsets)
            ],
        })
    return eligible, dict(sorted(rejection.items()))


def select_inventory(
    eligible: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    isolated: list[dict[str, Any]] = []
    used_terms: set[str] = set()
    for row in sorted(
        eligible,
        key=lambda value: stable_hash(
            "phase1114-global-lexical-isolation",
            value["base"],
            *value["sense_offsets"],
        ),
    ):
        terms = {row["base"], *row["hypernyms"]}
        if terms & used_terms:
            continue
        isolated.append(dict(row))
        used_terms.update(terms)
    if len(isolated) < SELECTED_ITEM_COUNT:
        raise RuntimeError(
            f"Only {len(isolated)} globally disjoint candidates; need {SELECTED_ITEM_COUNT}"
        )
    selected = sorted(
        isolated[:SELECTED_ITEM_COUNT],
        key=lambda value: stable_hash(
            "phase1114-split-assignment", value["base"], *value["sense_offsets"]
        ),
    )
    for split_index, split in enumerate(SPLITS):
        start = split_index * ITEMS_PER_SPLIT
        for item_index, row in enumerate(selected[start:start + ITEMS_PER_SPLIT]):
            row["split"] = split
            row["item_index"] = item_index
            row["concept_id"] = (
                f"wn-{row['sense_offsets'][0]}-{row['sense_offsets'][1]}"
            )
    return selected, len(isolated)


def render_prompt(concept: dict[str, Any], template: int, sense: int) -> str:
    return TEMPLATES[template].format(
        sentence=concept["examples"][sense], term=concept["base"]
    )


def build_case(
    tokenizer: Any,
    model_name: str,
    concept: dict[str, Any],
    template: int,
    sense: int,
    case_index: int,
) -> dict[str, Any]:
    raw_prompt = render_prompt(concept, template, sense)
    rendered = (
        relation_source.base.behavior_tools.render_native(
            tokenizer, model_name, raw_prompt, with_system=False
        )
        + ASSISTANT_PREFILL
    )
    input_ids = [
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    candidate_labels = {
        "sense0": concept["hypernyms"][0],
        "sense1": concept["hypernyms"][1],
    }
    candidate_token_ids = {
        key: relation_tools.continuation_ids(tokenizer, rendered, label)
        for key, label in candidate_labels.items()
    }
    expected_class = f"sense{sense}"
    pair_id = (
        f"phase1114.{model_name}.{concept['split']}.{concept['concept_id']}.t{template}"
    )
    record_id = f"{pair_id}.s{sense}"
    return {
        "schema_version": "phase1114_contextual_hypernym_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": record_id,
        "pair_id": pair_id,
        "concept_id": concept["concept_id"],
        "item_index": concept["item_index"],
        "split": concept["split"],
        "template": template,
        "sense": sense,
        "base": concept["base"],
        "sense_offset": concept["sense_offsets"][sense],
        "lexname": concept["lexnames"][sense],
        "native_example": concept["examples"][sense],
        "candidate_labels": candidate_labels,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {
            key: [int(values[0])] for key, values in candidate_token_ids.items()
        },
        "expected_class": expected_class,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "query_position": len(input_ids) - 1,
        "prompt_digest": hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
    }


def build_model_cases(
    tokenizer: Any, model_name: str, selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in selected:
        for template in TEMPLATES_BY_SPLIT[concept["split"]]:
            for sense in SENSES:
                rows.append(
                    build_case(
                        tokenizer,
                        model_name,
                        concept,
                        template,
                        sense,
                        len(rows),
                    )
                )
    return rows


def audit_model(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    expected_count = SELECTED_ITEM_COUNT * 6 * len(SENSES)
    split_counts = Counter(row["split"] for row in rows)
    split_template_sense = Counter(
        (row["split"], row["template"], row["sense"]) for row in rows
    )
    pair_counts = Counter(row["pair_id"] for row in rows)
    checks = {
        "case_count": len(rows) == expected_count,
        "record_ids_unique": len({row["record_id"] for row in rows}) == len(rows),
        "case_indices_contiguous": [row["case_index"] for row in rows]
        == list(range(len(rows))),
        "split_counts_balanced": set(split_counts.values())
        == {expected_count // len(SPLITS)},
        "split_template_sense_balanced": set(split_template_sense.values())
        == {ITEMS_PER_SPLIT},
        "pairs_complete": set(pair_counts.values()) == {2},
        "candidate_continuations_one_token": all(
            len(values) == 1
            for row in rows
            for values in row["candidate_token_ids"].values()
        ),
        "candidate_tokens_distinct": all(
            row["candidate_first_token_ids"]["sense0"]
            != row["candidate_first_token_ids"]["sense1"]
            for row in rows
        ),
        "candidate_labels_not_shown": all(
            all(exact_count(row["raw_prompt"], label) == 0 for label in row["candidate_labels"].values())
            for row in rows
        ),
        "native_example_contains_base_once": all(
            exact_count(row["native_example"], row["base"]) == 1 for row in rows
        ),
        "expected_class_matches_sense": all(
            row["expected_class"] == f"sense{row['sense']}" for row in rows
        ),
        "prompts_unique": len({row["prompt_digest"] for row in rows}) == len(rows),
        "input_nonempty": all(row["input_ids"] for row in rows),
        "query_at_last_input_token": all(
            row["query_position"] == len(row["input_ids"]) - 1 for row in rows
        ),
    }
    return {
        "model": model_name,
        "case_count": len(rows),
        "pair_count": len(pair_counts),
        "split_counts": dict(sorted(split_counts.items())),
        "minimum_token_length": min(len(row["input_ids"]) for row in rows),
        "maximum_token_length": max(len(row["input_ids"]) for row in rows),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(rows),
    }


def main() -> None:
    synsets, lemma_synsets, metadata = wordnet_source.parse_wordnet()
    examples, example_manifest = parse_native_examples()
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    eligible, rejection_counts = eligible_inventory(
        synsets, lemma_synsets, metadata, examples, tokenizers
    )
    selected, globally_disjoint_count = select_inventory(eligible)
    split_terms = {
        split: {
            term
            for row in selected
            if row["split"] == split
            for term in (row["base"], *row["hypernyms"])
        }
        for split in SPLITS
    }
    lexical_isolation = all(
        not (split_terms[left] & split_terms[right])
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1 :]
    )
    global_checks = {
        "wordnet_archive_sha256_verified": file_sha256(WORDNET_ARCHIVE)
        == WORDNET_SHA256,
        "eligible_inventory_sufficient": len(eligible) >= SELECTED_ITEM_COUNT,
        "globally_disjoint_inventory_sufficient": globally_disjoint_count
        >= SELECTED_ITEM_COUNT,
        "selected_item_count": len(selected) == SELECTED_ITEM_COUNT,
        "split_item_counts": all(
            sum(row["split"] == split for row in selected) == ITEMS_PER_SPLIT
            for split in SPLITS
        ),
        "split_base_and_candidate_terms_disjoint": lexical_isolation,
        "template_families_disjoint": len(
            set().union(*(set(values) for values in TEMPLATES_BY_SPLIT.values()))
        )
        == len(TEMPLATES),
        "source_examples_native_not_generated": True,
        "no_model_outputs_used_for_selection": True,
    }
    if not all(global_checks.values()):
        raise RuntimeError(f"global protocol checks failed: {global_checks}")

    protocol_root = OUT_ROOT / "protocol"
    source_manifest = {
        **metadata["source"],
        **example_manifest,
    }
    source_manifest["phase1114_source_digest"] = digest(source_manifest)
    write_json(protocol_root / "source_manifest.json", source_manifest)
    write_json(
        protocol_root / "eligible_inventory.json",
        {
            "eligible_count": len(eligible),
            "globally_disjoint_count": globally_disjoint_count,
            "rejection_counts": rejection_counts,
            "eligible": eligible,
            "inventory_digest": digest(eligible),
        },
    )
    write_json(
        protocol_root / "selected_concepts.json",
        {
            "selected_count": len(selected),
            "selected": selected,
            "selected_digest": digest(selected),
        },
    )

    model_audits: dict[str, Any] = {}
    case_digests: dict[str, str] = {}
    for model_name in MODELS:
        rows = build_model_cases(tokenizers[model_name], model_name, selected)
        audit = audit_model(rows, model_name)
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"{model_name} protocol audit failed: {audit['checks']}")
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", rows)
        model_audits[model_name] = audit
        case_digests[model_name] = audit["case_digest"]

    preregistration = {
        "schema_version": "phase1114_contextual_hypernym_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "splits": list(SPLITS),
        "items_per_split": ITEMS_PER_SPLIT,
        "senses": list(SENSES),
        "templates_by_split": {
            key: list(values) for key, values in TEMPLATES_BY_SPLIT.items()
        },
        "thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_manifest_digest": source_manifest["phase1114_source_digest"],
        "selected_concepts_digest": digest(selected),
        "case_digests": case_digests,
        "interpretive_limits": [
            "This is source-native contextual word-sense classification, not unrestricted free continuation.",
            "The two scored hypernym candidates are hidden from the prompt. Candidate-pair logit comparison remains a forced candidate audit.",
            "Only the paired context effect cancels a context-independent candidate-token prior; individual candidate accuracy does not.",
            "Repeated templates improve interface coverage but do not increase the number of independent semantic concepts beyond 36.",
            "The two native example sentences differ in many words, so a hidden-state difference would conflate sense with context surface without an additional matched-control protocol.",
            "No hidden-state or causal scan is automatic in Phase1114 even if behavior passes.",
        ],
        "hard_stop": (
            "If fewer than two models qualify, deny hidden-state access and do not revise "
            "this protocol after seeing outputs. If behavior qualifies, require a separate "
            "matched-control preregistration before any hidden-state claim."
        ),
    }
    preregistration["protocol_digest"] = digest(preregistration)
    write_json(protocol_root / "preregistration.json", preregistration)
    audit = {
        "schema_version": "phase1114_contextual_hypernym_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "global_checks": global_checks,
        "model_audits": model_audits,
        "all_checks_passed": all(global_checks.values())
        and all(row["all_checks_passed"] for row in model_audits.values()),
    }
    audit["audit_digest"] = digest(audit)
    write_json(protocol_root / "audit.json", audit)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "eligible_count": len(eligible),
                "globally_disjoint_count": globally_disjoint_count,
                "selected_count": len(selected),
                "case_count_per_model": model_audits["qwen3"]["case_count"],
                "pair_count_per_model": model_audits["qwen3"]["pair_count"],
                "all_checks_passed": audit["all_checks_passed"],
                "protocol_digest": preregistration["protocol_digest"],
                "audit_digest": audit["audit_digest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
