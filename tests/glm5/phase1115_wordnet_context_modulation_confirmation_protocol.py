#!/usr/bin/env python3
"""Freeze independent Phase1115 confirmation of contextual margin modulation."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1018_language_pattern_protocol import tokenizer_for
import phase1098_relative_relation_geometry_protocol as relation_tools
import phase1101_relation_identity_routing_protocol as relation_source
import phase1114_wordnet_contextual_hypernym_protocol as base


PHASE = 1115
PROTOCOL_REVISION = 1
MODELS = base.MODELS
PRECISION = base.PRECISION
QUANTIZATION = base.QUANTIZATION
SPLITS = base.SPLITS
ITEMS_PER_SPLIT = 7
SELECTED_ITEM_COUNT = ITEMS_PER_SPLIT * len(SPLITS)
SENSES = base.SENSES
ASSISTANT_PREFILL = base.ASSISTANT_PREFILL
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1115_wordnet_context_modulation_confirmation"
)
WORDNET_ARCHIVE = base.WORDNET_ARCHIVE
WORDNET_SHA256 = base.WORDNET_SHA256


TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1, 2, 3, 4, 5),
    "independent_confirmation": (6, 7, 8, 9, 10, 11),
    "heldout": (12, 13, 14, 15, 16, 17),
}


TEMPLATES = (
    (
        "Example sentence:\n{sentence}\n"
        "What broader noun best captures \"{term}\" in this particular sentence? "
        "Give a single lowercase noun."
    ),
    (
        "Interpret the target from the surrounding words.\nTarget: {term}\n"
        "Sentence: {sentence}\nReply with the nearest containing noun category, "
        "using one lowercase word."
    ),
    (
        "Which reading of \"{term}\" is active below?\n{sentence}\n"
        "Name its closest parent noun in one lowercase word."
    ),
    (
        "Treat the following as an attested dictionary example, then categorize the "
        "target occurrence.\n{sentence}\nTarget occurrence: {term}\n"
        "Respond with one lowercase noun."
    ),
    (
        "The surrounding sentence fixes the intended reading.\n{sentence}\n"
        "Supply a one-word lowercase noun that most closely subsumes \"{term}\" here."
    ),
    (
        "Read the full example before answering.\n{sentence}\n"
        "For the local reading of \"{term}\", write its nearest parent noun and "
        "nothing else."
    ),
    (
        "Contextual noun interpretation:\n{sentence}\n"
        "Under what broader noun should this occurrence of \"{term}\" be placed? "
        "Answer with one lowercase word."
    ),
    (
        "Use the words around the target to determine what it denotes.\n"
        "Sentence: {sentence}\nTarget: {term}\n"
        "Return the closest superordinate noun as one lowercase word."
    ),
    (
        "Read: {sentence}\n"
        "In that sentence, what is the nearest broader noun for the intended "
        "\"{term}\" reading? Use one lowercase noun only."
    ),
    (
        "An ambiguous spelling is resolved by the sentence below.\n{sentence}\n"
        "Categorize the resolved occurrence of \"{term}\" with one lowercase noun."
    ),
    (
        "Infer the denotation of the target in its attested usage.\n"
        "Usage: {sentence}\nTarget noun: {term}\n"
        "Give its closest parent category in one lowercase word."
    ),
    (
        "Do not classify the spelling in isolation; classify its use here.\n"
        "{sentence}\nThe noun \"{term}\" is most nearly an instance of what broader "
        "noun? Reply with one lowercase word."
    ),
    (
        "Determine the intended category from this occurrence:\n{sentence}\n"
        "Write the nearest noun above \"{term}\" for this reading, as one lowercase word."
    ),
    (
        "The sentence provides the disambiguating evidence.\nSentence: {sentence}\n"
        "Place \"{term}\" beneath its closest broader noun. Return only that lowercase noun."
    ),
    (
        "Interpret, then categorize.\n{sentence}\n"
        "For the meaning expressed by \"{term}\" here, supply one nearest parent noun "
        "in lowercase."
    ),
    (
        "This is a local-meaning question about the sentence below.\n{sentence}\n"
        "What broader noun most directly includes the intended \"{term}\" meaning? "
        "Answer with one lowercase word."
    ),
    (
        "Resolve the target using only its natural sentence context.\n"
        "Target: {term}\nContext: {sentence}\n"
        "Respond with the closest superordinate noun, one lowercase word only."
    ),
    (
        "Examine the attested use: {sentence}\n"
        "Which broader noun is nearest to \"{term}\" in this use? "
        "Write exactly one lowercase noun."
    ),
)


THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_context_direction_accuracy": 0.80,
    "minimum_split_context_direction_accuracy": 0.75,
    "minimum_template_context_direction_accuracy": 0.65,
    "minimum_concept_direction_accuracy": 0.80,
    "minimum_split_concept_direction_accuracy": 5 / 7,
    "minimum_shared_two_model_concept_fraction": 0.70,
    "minimum_modulation_qualified_models": 2,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "All 21 concepts, both candidate terms, and all prompt templates are disjoint "
        "from Phase1114; source, tokenization, pairing, and digest audits pass."
    ),
    "P2": (
        "At least two FP16/no-quantization models pass the overall, split, template, "
        "and concept-level context-direction gates."
    ),
    "P3": (
        "At least 70 percent of concepts have a positive median context effect in at "
        "least two qualified models."
    ),
    "P4": (
        "Candidate accuracy, direct generation, and bidirectional boundary crossing "
        "remain diagnostics; they cannot be upgraded by this modulation-only confirmation."
    ),
    "P5": (
        "Regardless of result, no hidden-state scan is authorized because the two native "
        "sentences differ in uncontrolled surface content."
    ),
}


def isolated_inventory(eligible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    isolated: list[dict[str, Any]] = []
    used_terms: set[str] = set()
    for row in sorted(
        eligible,
        key=lambda value: base.stable_hash(
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
    return isolated


def select_inventory(
    isolated: list[dict[str, Any]], phase1114_selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    used_concepts = {row["base"] for row in phase1114_selected}
    used_terms = {
        term
        for row in phase1114_selected
        for term in (row["base"], *row["hypernyms"])
    }
    tail = []
    for rank, row in enumerate(isolated):
        terms = {row["base"], *row["hypernyms"]}
        if row["base"] in used_concepts or terms & used_terms:
            continue
        if any(
            base.exact_count(template.format(sentence="", term=row["base"]), candidate)
            > 0
            for template in TEMPLATES
            for candidate in row["hypernyms"]
        ):
            continue
        value = dict(row)
        value["source_inventory_rank"] = rank
        tail.append(value)
    if len(tail) < SELECTED_ITEM_COUNT:
        raise RuntimeError(
            f"Only {len(tail)} Phase1114-disjoint candidates; need {SELECTED_ITEM_COUNT}"
        )
    selected = sorted(
        tail[:SELECTED_ITEM_COUNT],
        key=lambda value: base.stable_hash(
            "phase1115-split-assignment", value["base"], *value["sense_offsets"]
        ),
    )
    for split_index, split in enumerate(SPLITS):
        start = split_index * ITEMS_PER_SPLIT
        for item_index, row in enumerate(selected[start : start + ITEMS_PER_SPLIT]):
            row["split"] = split
            row["item_index"] = item_index
            row["concept_id"] = (
                f"wn-{row['sense_offsets'][0]}-{row['sense_offsets'][1]}"
            )
    return selected


def build_case(
    tokenizer: Any,
    model_name: str,
    concept: dict[str, Any],
    template: int,
    sense: int,
    case_index: int,
) -> dict[str, Any]:
    raw_prompt = TEMPLATES[template].format(
        sentence=concept["examples"][sense], term=concept["base"]
    )
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
    pair_id = (
        f"phase1115.{model_name}.{concept['split']}.{concept['concept_id']}.t{template}"
    )
    return {
        "schema_version": "phase1115_context_modulation_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{pair_id}.s{sense}",
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
        "expected_class": f"sense{sense}",
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "query_position": len(input_ids) - 1,
        "prompt_digest": hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
    }


def build_cases(
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
    pair_counts = Counter(row["pair_id"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    cells = Counter(
        (row["split"], row["template"], row["sense"]) for row in rows
    )
    checks = {
        "case_count": len(rows) == expected_count,
        "record_ids_unique": len({row["record_id"] for row in rows}) == len(rows),
        "case_indices_contiguous": [row["case_index"] for row in rows]
        == list(range(len(rows))),
        "pairs_complete": set(pair_counts.values()) == {2},
        "split_counts_balanced": set(split_counts.values())
        == {expected_count // len(SPLITS)},
        "cells_balanced": set(cells.values()) == {ITEMS_PER_SPLIT},
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
            all(
                base.exact_count(row["raw_prompt"], label) == 0
                for label in row["candidate_labels"].values()
            )
            for row in rows
        ),
        "native_example_contains_base_once": all(
            base.exact_count(row["native_example"], row["base"]) == 1
            for row in rows
        ),
        "prompts_unique": len({row["prompt_digest"] for row in rows}) == len(rows),
        "query_at_last_input_token": all(
            row["query_position"] == len(row["input_ids"]) - 1 for row in rows
        ),
    }
    return {
        "model": model_name,
        "case_count": len(rows),
        "pair_count": len(pair_counts),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": base.digest(rows),
    }


def main() -> None:
    synsets, lemma_synsets, metadata = base.wordnet_source.parse_wordnet()
    examples, example_manifest = base.parse_native_examples()
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    eligible, rejection_counts = base.eligible_inventory(
        synsets, lemma_synsets, metadata, examples, tokenizers
    )
    isolated = isolated_inventory(eligible)
    phase1114_payload = base.read_json(
        base.OUT_ROOT / "protocol" / "selected_concepts.json"
    )
    phase1114_selected = phase1114_payload["selected"]
    selected = select_inventory(isolated, phase1114_selected)

    old_terms = {
        term
        for row in phase1114_selected
        for term in (row["base"], *row["hypernyms"])
    }
    new_terms = {
        term for row in selected for term in (row["base"], *row["hypernyms"])
    }
    split_terms = {
        split: {
            term
            for row in selected
            if row["split"] == split
            for term in (row["base"], *row["hypernyms"])
        }
        for split in SPLITS
    }
    global_checks = {
        "wordnet_archive_sha256_verified": base.file_sha256(WORDNET_ARCHIVE)
        == WORDNET_SHA256,
        "selected_item_count": len(selected) == SELECTED_ITEM_COUNT,
        "phase1114_concepts_and_terms_disjoint": not (old_terms & new_terms),
        "phase1114_templates_disjoint": not (set(TEMPLATES) & set(base.TEMPLATES)),
        "split_item_counts": all(
            sum(row["split"] == split for row in selected) == ITEMS_PER_SPLIT
            for split in SPLITS
        ),
        "split_terms_disjoint": all(
            not (split_terms[left] & split_terms[right])
            for left_index, left in enumerate(SPLITS)
            for right in SPLITS[left_index + 1 :]
        ),
        "source_only_tail_selection": True,
        "no_phase1114_or_phase1115_model_outputs_used_for_selection": True,
    }
    if not all(global_checks.values()):
        raise RuntimeError(f"Phase1115 global audit failed: {global_checks}")

    protocol_root = OUT_ROOT / "protocol"
    source_manifest = {
        **metadata["source"],
        **example_manifest,
        "phase1114_selected_digest": phase1114_payload["selected_digest"],
        "phase1114_protocol_digest": base.read_json(
            base.OUT_ROOT / "protocol" / "preregistration.json"
        )["protocol_digest"],
    }
    source_manifest["phase1115_source_digest"] = base.digest(source_manifest)
    base.write_json(protocol_root / "source_manifest.json", source_manifest)
    base.write_json(
        protocol_root / "eligible_tail_inventory.json",
        {
            "eligible_count": len(eligible),
            "globally_isolated_count": len(isolated),
            "rejection_counts": rejection_counts,
            "tail_count_after_phase1114_and_template_filters": sum(
                1
                for row in isolated
                if row["base"] not in {value["base"] for value in phase1114_selected}
                and not ({row["base"], *row["hypernyms"]} & old_terms)
                and not any(
                    base.exact_count(
                        template.format(sentence="", term=row["base"]), candidate
                    )
                    > 0
                    for template in TEMPLATES
                    for candidate in row["hypernyms"]
                )
            ),
            "isolated_inventory_digest": base.digest(isolated),
        },
    )
    base.write_json(
        protocol_root / "selected_concepts.json",
        {
            "selected_count": len(selected),
            "selected": selected,
            "selected_digest": base.digest(selected),
        },
    )

    model_audits: dict[str, Any] = {}
    case_digests: dict[str, str] = {}
    for model_name in MODELS:
        rows = build_cases(tokenizers[model_name], model_name, selected)
        audit = audit_model(rows, model_name)
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"{model_name} audit failed: {audit['checks']}")
        base.write_jsonl(protocol_root / f"cases.{model_name}.jsonl", rows)
        model_audits[model_name] = audit
        case_digests[model_name] = audit["case_digest"]

    preregistration = {
        "schema_version": "phase1115_context_modulation_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "splits": list(SPLITS),
        "items_per_split": ITEMS_PER_SPLIT,
        "templates_by_split": {
            key: list(values) for key, values in TEMPLATES_BY_SPLIT.items()
        },
        "thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_manifest_digest": source_manifest["phase1115_source_digest"],
        "selected_concepts_digest": base.digest(selected),
        "case_digests": case_digests,
        "primary_object": (
            "The sign of the paired context effect z0-z1, aggregated both by case pair "
            "and by independent concept."
        ),
        "interpretive_limits": [
            "This phase independently confirms only contextual candidate-margin modulation; candidate use is not a primary gate.",
            "Six templates per concept are repeated interfaces, not six independent semantic facts; concept-level gates are mandatory.",
            "A positive context effect does not prove an abstract semantic coordinate, because the two source sentences differ in many surface words.",
            "No hidden-state or causal scan is authorized regardless of the result.",
        ],
    }
    preregistration["protocol_digest"] = base.digest(preregistration)
    base.write_json(protocol_root / "preregistration.json", preregistration)
    audit = {
        "schema_version": "phase1115_context_modulation_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "global_checks": global_checks,
        "model_audits": model_audits,
        "all_checks_passed": all(global_checks.values())
        and all(row["all_checks_passed"] for row in model_audits.values()),
    }
    audit["audit_digest"] = base.digest(audit)
    base.write_json(protocol_root / "audit.json", audit)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "eligible_count": len(eligible),
                "isolated_count": len(isolated),
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
