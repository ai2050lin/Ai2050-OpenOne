#!/usr/bin/env python3
"""Audit WordNet-native examples for a prospective Phase1114 protocol."""

from __future__ import annotations

import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1018_language_pattern_protocol import tokenizer_for
import phase1113_wordnet_semantic_quadrant_protocol as phase1113


WORD_RE = re.compile(r"(?<![A-Za-z]){}(?![A-Za-z])", re.IGNORECASE)


def contains_once(text: str, term: str) -> bool:
    return len(re.findall(WORD_RE.pattern.format(re.escape(term)), text)) == 1


def simple_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text.casefold())


def parse_examples() -> dict[str, list[str]]:
    with tarfile.open(phase1113.WORDNET_ARCHIVE, "r:gz") as archive:
        raw = archive.extractfile(phase1113.WORDNET_MEMBERS["data_noun"]).read()
    examples: dict[str, list[str]] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line or line[0].isspace():
            continue
        main, gloss = line.split("|", 1)
        offset = main.split()[0]
        examples[offset] = [
            value.strip() for value in re.findall(r'"([^"\r\n]+)"', gloss)
        ]
    return examples


def single_token_all(tokenizers: dict[str, object], term: str) -> bool:
    return all(
        len(tokenizer.encode(" " + term, add_special_tokens=False)) == 1
        for tokenizer in tokenizers.values()
    )


def main() -> None:
    synsets, lemma_synsets, metadata = phase1113.parse_wordnet()
    examples = parse_examples()
    sense_number = metadata["sense_number"]
    tag_count = metadata["tag_count"]
    tokenizers = {model: tokenizer_for(model) for model in phase1113.MODELS}
    rejection: Counter[str] = Counter()
    base_native: list[dict[str, object]] = []
    cross_surface_native: list[dict[str, object]] = []
    synonym_target_native: list[dict[str, object]] = []

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
        offsets2 = ordered[:2]
        senses = [synsets[offset] for offset in offsets2]
        if (
            tag_count.get((base, offsets2[0]), 0) < 5
            or tag_count.get((base, offsets2[1]), 0) < 1
        ):
            rejection["base_tag_count"] += 1
            continue
        if senses[0]["lexfile_index"] == senses[1]["lexfile_index"]:
            rejection["same_lexfile"] += 1
            continue
        if any(len(sense["hypernyms"]) != 1 for sense in senses):
            rejection["nonunique_hypernym"] += 1
            continue
        hypernyms = [synsets[sense["hypernyms"][0]]["lemmas"][0] for sense in senses]
        if (
            hypernyms[0] == hypernyms[1]
            or any(not re.fullmatch(r"[a-z]+", value) for value in hypernyms)
            or any(not single_token_all(tokenizers, value) for value in hypernyms)
        ):
            rejection["hypernym_token_or_identity"] += 1
            continue
        if any(
            tag_count.get((hypernym, sense["hypernyms"][0]), 0) < 1
            for hypernym, sense in zip(hypernyms, senses)
        ):
            rejection["hypernym_tag_count"] += 1
            continue
        base_examples = []
        for offset in offsets2:
            valid = [
                example
                for example in examples.get(offset, [])
                if contains_once(example, base)
                and 4 <= len(simple_words(example)) <= 22
            ]
            base_examples.append(valid)
        if any(not values for values in base_examples):
            rejection["missing_base_native_example"] += 1
            continue
        if any(
            any(
                contains_once(example, hypernym)
                for hypernym in hypernyms
                for example in side
            )
            for side in base_examples
        ):
            rejection["hypernym_leak_in_base_example"] += 1
            continue

        alternatives: list[list[tuple[str, str]]] = []
        synonym_targets: list[list[str]] = []
        for sense_index, (offset, sense) in enumerate(zip(offsets2, senses)):
            other_lemmas = set(senses[1 - sense_index]["lemmas"])
            candidates: list[tuple[str, str]] = []
            target_candidates: list[str] = []
            for lemma in sense["lemmas"]:
                if (
                    lemma == base
                    or lemma in other_lemmas
                    or not re.fullmatch(r"[a-z]+", lemma)
                    or tag_count.get((lemma, offset), 0) < 1
                    or phase1113.simple_morphology_root(lemma)
                    == phase1113.simple_morphology_root(base)
                ):
                    continue
                if (
                    single_token_all(tokenizers, lemma)
                    and not any(
                        contains_once(example, lemma)
                        for side in base_examples
                        for example in side
                    )
                ):
                    target_candidates.append(lemma)
                native = [
                    example
                    for example in examples.get(offset, [])
                    if contains_once(example, lemma)
                    and 4 <= len(simple_words(example)) <= 22
                    and not any(contains_once(example, value) for value in hypernyms)
                ]
                if native:
                    candidates.append((lemma, native[0]))
            candidates.sort(key=lambda pair: (-tag_count.get((pair[0], offset), 0), pair[0]))
            target_candidates.sort(
                key=lambda lemma: (-tag_count.get((lemma, offset), 0), lemma)
            )
            alternatives.append(candidates)
            synonym_targets.append(target_candidates)

        row = {
            "base": base,
            "offsets": offsets2,
            "lexnames": [sense["lexname"] for sense in senses],
            "hypernyms": hypernyms,
            "base_examples": [values[0] for values in base_examples],
            "alternatives": [values[0] if values else None for values in alternatives],
            "synonym_targets": [values[0] if values else None for values in synonym_targets],
        }
        base_native.append(row)
        if all(synonym_targets):
            synonym_target_native.append(row)
        else:
            rejection["missing_single_token_synonym_target"] += 1
        if all(alternatives):
            cross_surface_native.append(row)
        else:
            rejection["missing_alternate_native_example"] += 1

    def greedy(rows: list[dict[str, object]], term_field: str | None) -> list[dict[str, object]]:
        selected: list[dict[str, object]] = []
        used: set[str] = set()
        for row in sorted(
            rows,
            key=lambda value: phase1113.stable_hash(
                "phase1114-native-feasibility", value["base"], *value["offsets"]
            ),
        ):
            terms = {str(row["base"]), *map(str, row["hypernyms"])}
            if term_field == "alternatives":
                terms.update(str(pair[0]) for pair in row[term_field] if pair)
            elif term_field == "synonym_targets":
                terms.update(str(value) for value in row[term_field] if value)
            roots = {phase1113.simple_morphology_root(term) for term in terms}
            if len(roots) != len(terms) or terms & used:
                continue
            selected.append(row)
            used.update(terms)
        return selected

    report = {
        "base_native_candidate_count": len(base_native),
        "base_native_globally_disjoint_count": len(greedy(base_native, None)),
        "cross_surface_native_candidate_count": len(cross_surface_native),
        "cross_surface_native_globally_disjoint_count": len(
            greedy(cross_surface_native, "alternatives")
        ),
        "synonym_target_native_candidate_count": len(synonym_target_native),
        "synonym_target_native_globally_disjoint_count": len(
            greedy(synonym_target_native, "synonym_targets")
        ),
        "base_native_globally_disjoint": greedy(base_native, None),
        "cross_surface_native_globally_disjoint": greedy(
            cross_surface_native, "alternatives"
        ),
        "synonym_target_native_globally_disjoint": greedy(
            synonym_target_native, "synonym_targets"
        ),
        "rejection_counts": dict(sorted(rejection.items())),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
