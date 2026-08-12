#!/usr/bin/env python3
"""Freeze the Phase1113 WordNet surface-by-sense behavior protocol."""

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
import phase1098_relative_relation_geometry_protocol as tools
import phase1101_relation_identity_routing_protocol as relation_source


PHASE = 1113
PROTOCOL_REVISION = 3
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
SPLITS = ("discovery", "independent_confirmation", "heldout")
ITEMS_PER_SPLIT = 6
SELECTED_ITEM_COUNT = ITEMS_PER_SPLIT * len(SPLITS)
QUADRANTS = (
    "same_surface_same_sense",
    "same_surface_different_sense",
    "different_surface_same_sense",
    "different_surface_different_sense",
)
ANSWER_ORDERS = (0, 1)
ASSISTANT_PREFILL = "Answer:"
CONTINUATION_PREFIX = " "
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1113_wordnet_semantic_quadrant"
WORDNET_ARCHIVE = (
    ROOT
    / "tests"
    / "gpt5"
    / "result"
    / "phase602_three_track_semantics"
    / "source"
    / "WordNet-3.0.tar.gz"
)
WORDNET_SHA256 = "640db279c949a88f61f851dd54ebbb22d003f8b90b85267042ef85a3781d3a52"
WORDNET_MEMBERS = {
    "data_noun": "WordNet-3.0/dict/data.noun",
    "index_sense": "WordNet-3.0/dict/index.sense",
    "lexnames": "WordNet-3.0/dict/lexnames",
}


TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1, 2, 3, 4, 5),
    "independent_confirmation": (6, 7, 8, 9, 10, 11),
    "heldout": (12, 13, 14, 15, 16, 17),
}


TEMPLATES = (
    (
        "Use only the following WordNet 3.0 evidence.\n"
        "Reference definition: {definition}\n"
        "Candidate entry: the noun '{term}' is in lexical file '{lexname}' and has "
        "the immediate hypernym '{hypernym}'.\n"
        "Do the reference and candidate denote the same WordNet noun synset?"
    ),
    (
        "This is a WordNet 3.0 noun-sense comparison.\n"
        "A reference sense is defined as: {definition}\n"
        "For the candidate, '{term}' is classified under '{lexname}', directly below "
        "'{hypernym}'.\n"
        "Are these two entries the same noun sense?"
    ),
    (
        "Judge the entries from the supplied WordNet 3.0 facts, not from spelling alone.\n"
        "Definition for the reference entry: {definition}\n"
        "The candidate noun is '{term}'; its lexical file is '{lexname}' and its direct "
        "hypernym is '{hypernym}'.\n"
        "Do both entries identify one noun synset?"
    ),
    (
        "Compare two noun entries using only this WordNet 3.0 record.\n"
        "Reference gloss: {definition}\n"
        "Candidate record: '{term}', lexical file '{lexname}', immediate parent "
        "'{hypernym}'.\n"
        "Is the candidate record the same noun sense as the reference gloss?"
    ),
    (
        "The task is to match WordNet 3.0 noun senses.\n"
        "One entry has this definition: {definition}\n"
        "Another entry uses '{term}' in lexical file '{lexname}' with direct hypernym "
        "'{hypernym}'.\n"
        "Do the entries point to the identical synset?"
    ),
    (
        "Read the registered dictionary evidence below.\n"
        "WordNet reference meaning: {definition}\n"
        "WordNet candidate: noun '{term}'; file '{lexname}'; nearest hypernym "
        "'{hypernym}'.\n"
        "Are the reference meaning and candidate the same noun sense?"
    ),
    (
        "Resolve a noun-sense identity question from WordNet 3.0.\n"
        "The reference is defined by: {definition}\n"
        "The candidate expression is '{term}', recorded in '{lexname}' beneath the "
        "hypernym '{hypernym}'.\n"
        "Does the candidate express exactly the reference synset?"
    ),
    (
        "Treat each line as evidence from WordNet 3.0.\n"
        "Reference entry definition: {definition}\n"
        "Candidate entry evidence: '{term}' / lexical file '{lexname}' / direct parent "
        "'{hypernym}'.\n"
        "Do these entries have one shared noun-sense identity?"
    ),
    (
        "Use the frozen WordNet 3.0 evidence to compare noun senses.\n"
        "Reference evidence says: {definition}\n"
        "Candidate evidence says that '{term}' belongs to '{lexname}' and immediately "
        "inherits from '{hypernym}'.\n"
        "Are both descriptions for the same noun synset?"
    ),
    (
        "Decide whether two WordNet 3.0 noun records are sense-identical.\n"
        "Record one is defined as: {definition}\n"
        "Record two contains '{term}', uses lexical file '{lexname}', and lists "
        "'{hypernym}' as its immediate hypernym.\n"
        "Do record one and record two denote the same sense?"
    ),
    (
        "Compare the semantic identities specified by this WordNet 3.0 evidence.\n"
        "Reference definition: {definition}\n"
        "The candidate noun '{term}' occurs in '{lexname}' directly under "
        "'{hypernym}'.\n"
        "Is its noun sense identical to the reference?"
    ),
    (
        "Answer a WordNet noun-synset identity check.\n"
        "Reference gloss: {definition}\n"
        "Candidate facts: term '{term}', lexical file '{lexname}', closest hypernym "
        "'{hypernym}'.\n"
        "Are the two sense records identical?"
    ),
    (
        "Determine noun-sense identity from the listed WordNet 3.0 fields.\n"
        "The reference field defines: {definition}\n"
        "The comparison field gives noun '{term}', lexical file '{lexname}', and direct "
        "hypernym '{hypernym}'.\n"
        "Do both fields specify one synset?"
    ),
    (
        "Use this WordNet 3.0 extract for a dictionary decision.\n"
        "Reference meaning: {definition}\n"
        "Comparison meaning: '{term}' in '{lexname}', immediately subordinate to "
        "'{hypernym}'.\n"
        "Is the comparison meaning the same noun sense as the reference?"
    ),
    (
        "Check whether the supplied WordNet noun evidence is co-referential.\n"
        "Reference definition: {definition}\n"
        "Candidate noun evidence: '{term}' has lexical file '{lexname}' and immediate "
        "hypernym '{hypernym}'.\n"
        "Do the records refer to exactly one noun synset?"
    ),
    (
        "Make a sense-level, not merely word-level, WordNet 3.0 comparison.\n"
        "The reference sense means: {definition}\n"
        "The candidate uses '{term}' under '{lexname}', with '{hypernym}' as direct parent.\n"
        "Are candidate and reference the identical noun sense?"
    ),
    (
        "Consult only these WordNet 3.0 sense clues.\n"
        "Reference clue: {definition}\n"
        "Candidate clues: noun '{term}'; lexical file '{lexname}'; immediate hypernym "
        "'{hypernym}'.\n"
        "Do the clues resolve to the same noun synset?"
    ),
    (
        "Perform a WordNet noun-entry identity test.\n"
        "Entry A is defined as: {definition}\n"
        "Entry B uses '{term}', belongs to '{lexname}', and sits directly below "
        "'{hypernym}'.\n"
        "Do Entry A and Entry B have the same sense identity?"
    ),
)


STOPWORDS = frozenset(
    "a an the and or of to in on at by for from with as is are was were be being been "
    "that which who whom whose this these those it its their his her into within through "
    "over under about used often usually especially any each other than something someone "
    "some one two more most having has have".split()
)


THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_overall_candidate_accuracy": 0.85,
    "minimum_split_candidate_accuracy": 0.80,
    "minimum_quadrant_candidate_accuracy": 0.80,
    "minimum_split_quadrant_candidate_accuracy": 0.75,
    "minimum_template_quadrant_candidate_accuracy": 0.75,
    "minimum_anti_shortcut_quadrant_accuracy": 0.80,
    "maximum_answer_order_accuracy_gap": 0.10,
    "minimum_behavior_qualified_models": 2,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "The verified WordNet archive, noun-sense parser, source-only candidate filters, "
        "split lexical isolation, four-quadrant balance, tokenization, and case digests pass."
    ),
    "P2": (
        "At least two FP16/no-quantization models satisfy all behavior gates on all three "
        "splits, including the independent heldout split."
    ),
    "P3": (
        "The two anti-shortcut quadrants -- same surface/different sense and different "
        "surface/same sense -- each reach the preregistered accuracy gate."
    ),
    "P4": (
        "Answer-order counterbalancing does not produce an accuracy gap above the frozen gate."
    ),
    "P5": (
        "Passing Phase1113 qualifies only this public-source metalinguistic behavior object. "
        "It does not authorize an automatic hidden-state scan or establish natural semantic routing."
    ),
}


def stable_hash(*parts: object) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(payload: Any) -> str:
    frozen = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(frozen.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def content_words(text: str) -> list[str]:
    return [
        value
        for value in re.findall(r"[a-z]+", text.casefold())
        if value not in STOPWORDS
    ]


def simple_morphology_root(term: str) -> str:
    """Reject obvious inflection-only 'different surface' controls."""
    value = term.casefold()
    for suffix in ("ing", "ies", "es", "s", "ed"):
        if value.endswith(suffix) and len(value) > len(suffix) + 3:
            return value[:-len(suffix)]
    return value


def parse_wordnet() -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    if file_sha256(WORDNET_ARCHIVE) != WORDNET_SHA256:
        raise RuntimeError("WordNet archive SHA256 mismatch")
    with tarfile.open(WORDNET_ARCHIVE, "r:gz") as archive:
        member_bytes = {
            key: archive.extractfile(member).read()
            for key, member in WORDNET_MEMBERS.items()
        }
    data_text = member_bytes["data_noun"].decode("utf-8")
    sense_text = member_bytes["index_sense"].decode("utf-8")
    lexname_text = member_bytes["lexnames"].decode("utf-8")
    lexnames = {
        int(line.split()[0]): line.split()[1]
        for line in lexname_text.splitlines()
        if line.strip()
    }
    synsets: dict[str, dict[str, Any]] = {}
    lemma_synsets: dict[str, list[str]] = defaultdict(list)
    for line in data_text.splitlines():
        if not line or line[0].isspace():
            continue
        main, gloss = line.split("|", 1)
        fields = main.split()
        offset = fields[0]
        lexfile = int(fields[1])
        word_count = int(fields[3], 16)
        cursor = 4
        lemmas = [
            fields[cursor + 2 * index].replace("_", " ").casefold()
            for index in range(word_count)
        ]
        cursor += 2 * word_count
        pointer_count = int(fields[cursor])
        cursor += 1
        hypernyms: list[str] = []
        for _index in range(pointer_count):
            symbol, target, pos, _source_target = fields[cursor:cursor + 4]
            cursor += 4
            if symbol == "@" and pos == "n":
                hypernyms.append(target)
        definition = gloss.strip().split(";", 1)[0].strip()
        synsets[offset] = {
            "offset": offset,
            "lexfile_index": lexfile,
            "lexname": lexnames[lexfile],
            "lemmas": lemmas,
            "hypernyms": hypernyms,
            "definition": definition,
        }
        for lemma in lemmas:
            lemma_synsets[lemma].append(offset)
    sense_number: dict[tuple[str, str], int] = {}
    tag_count: dict[tuple[str, str], int] = {}
    for line in sense_text.splitlines():
        if not line.strip():
            continue
        sense_key, offset, number, count = line.split()[:4]
        if "%1:" not in sense_key:
            continue
        lemma = sense_key.split("%", 1)[0].replace("_", " ").casefold()
        sense_number[(lemma, offset)] = int(number)
        tag_count[(lemma, offset)] = int(count)
    source = {
        "archive_path": str(WORDNET_ARCHIVE.relative_to(ROOT)),
        "archive_sha256": WORDNET_SHA256,
        "member_sha256": {
            key: hashlib.sha256(value).hexdigest() for key, value in member_bytes.items()
        },
        "noun_synset_count": len(synsets),
        "noun_lemma_count": len(lemma_synsets),
        "lexname_count": len(lexnames),
    }
    source["source_digest"] = digest(source)
    return synsets, lemma_synsets, {
        "sense_number": sense_number,
        "tag_count": tag_count,
        "source": source,
    }


def eligible_inventory(
    synsets: dict[str, dict[str, Any]],
    lemma_synsets: dict[str, list[str]],
    metadata: dict[str, Any],
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
            rejection["fewer_than_two_noun_senses"] += 1
            continue
        first_offset, second_offset = ordered[:2]
        first = synsets[first_offset]
        second = synsets[second_offset]
        if tag_count.get((base, first_offset), 0) < 5:
            rejection["first_sense_tag_count"] += 1
            continue
        if tag_count.get((base, second_offset), 0) < 1:
            rejection["second_sense_tag_count"] += 1
            continue
        if first["lexfile_index"] == second["lexfile_index"]:
            rejection["same_lexicographer_file"] += 1
            continue
        if len(first["hypernyms"]) != 1 or len(second["hypernyms"]) != 1:
            rejection["nonunique_direct_hypernym"] += 1
            continue
        if first["hypernyms"][0] == second["hypernyms"][0]:
            rejection["shared_direct_hypernym"] += 1
            continue
        first_words = content_words(first["definition"])
        second_words = content_words(second["definition"])
        if not (3 <= len(first_words) <= 20 and 3 <= len(second_words) <= 20):
            rejection["definition_length"] += 1
            continue

        def alternatives(
            synset: dict[str, Any], offset: str, forbidden_lemmas: set[str],
        ) -> list[str]:
            return sorted(
                (
                    lemma
                    for lemma in synset["lemmas"]
                    if lemma != base
                    and lemma not in forbidden_lemmas
                    and re.fullmatch(r"[a-z]+", lemma)
                    and 4 <= len(lemma) <= 15
                    and tag_count.get((lemma, offset), 0) >= 1
                ),
                key=lambda lemma: (-tag_count.get((lemma, offset), 0), lemma),
            )

        first_alternatives = alternatives(
            first, first_offset, set(second["lemmas"])
        )
        second_alternatives = alternatives(
            second, second_offset, set(first["lemmas"])
        )
        if not first_alternatives or not second_alternatives:
            rejection["missing_tagged_alternate"] += 1
            continue
        alternate_first = first_alternatives[0]
        alternate_second = second_alternatives[0]
        if alternate_first == alternate_second:
            rejection["alternate_collision"] += 1
            continue
        if len({
            simple_morphology_root(base),
            simple_morphology_root(alternate_first),
            simple_morphology_root(alternate_second),
        }) != 3:
            rejection["inflection_only_surface_difference"] += 1
            continue
        first_hypernym = synsets[first["hypernyms"][0]]
        second_hypernym = synsets[second["hypernyms"][0]]
        target_words = set(content_words(f"{base} {alternate_first} {alternate_second}"))
        reference_words = set(first_words)
        if target_words & reference_words:
            rejection["reference_definition_target_leakage"] += 1
            continue
        if reference_words & set(content_words(" ".join(first_hypernym["lemmas"]))):
            rejection["reference_same_hypernym_leakage"] += 1
            continue
        if reference_words & set(content_words(" ".join(second_hypernym["lemmas"]))):
            rejection["reference_different_hypernym_leakage"] += 1
            continue
        eligible.append({
            "base": base,
            "alternate_first": alternate_first,
            "alternate_second": alternate_second,
            "first_synset_offset": first_offset,
            "second_synset_offset": second_offset,
            "first_definition": first["definition"],
            "second_definition": second["definition"],
            "first_lexname": first["lexname"],
            "second_lexname": second["lexname"],
            "first_hypernym_offset": first_hypernym["offset"],
            "second_hypernym_offset": second_hypernym["offset"],
            "first_hypernym": first_hypernym["lemmas"][0],
            "second_hypernym": second_hypernym["lemmas"][0],
            "base_first_tag_count": tag_count.get((base, first_offset), 0),
            "base_second_tag_count": tag_count.get((base, second_offset), 0),
            "alternate_first_tag_count": tag_count.get((alternate_first, first_offset), 0),
            "alternate_second_tag_count": tag_count.get((alternate_second, second_offset), 0),
        })
    return eligible, dict(sorted(rejection.items()))


def select_inventory(eligible: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    used_terms: set[str] = set()
    for row in sorted(
        eligible,
        key=lambda value: stable_hash(
            "phase1113-lexical-isolation",
            value["base"],
            value["first_synset_offset"],
            value["second_synset_offset"],
        ),
    ):
        terms = {
            row["base"], row["alternate_first"], row["alternate_second"],
        }
        if terms & used_terms:
            continue
        unique.append(dict(row))
        used_terms.update(terms)
    if len(unique) < SELECTED_ITEM_COUNT:
        raise RuntimeError(
            f"Only {len(unique)} globally term-disjoint candidates; need {SELECTED_ITEM_COUNT}"
        )
    selected = sorted(
        unique[:SELECTED_ITEM_COUNT],
        key=lambda value: stable_hash(
            "phase1113-split", value["base"], value["first_synset_offset"]
        ),
    )
    for split_index, split in enumerate(SPLITS):
        start = split_index * ITEMS_PER_SPLIT
        for item_index, row in enumerate(selected[start:start + ITEMS_PER_SPLIT]):
            row["split"] = split
            row["item_index"] = item_index
            row["concept_id"] = f"wn-{row['first_synset_offset']}-{row['second_synset_offset']}"
    return selected, len(unique)


def quadrant_state(concept: dict[str, Any], quadrant: str) -> dict[str, Any]:
    same_surface = quadrant.startswith("same_surface")
    same_sense = quadrant.endswith("same_sense")
    if same_sense:
        offset = concept["first_synset_offset"]
        term = concept["base"] if same_surface else concept["alternate_first"]
        lexname = concept["first_lexname"]
        hypernym = concept["first_hypernym"]
    else:
        offset = concept["second_synset_offset"]
        term = concept["base"] if same_surface else concept["alternate_second"]
        lexname = concept["second_lexname"]
        hypernym = concept["second_hypernym"]
    return {
        "surface_same": same_surface,
        "semantic_same": same_sense,
        "right_term": term,
        "right_synset_offset": offset,
        "right_lexname": lexname,
        "right_hypernym": hypernym,
    }


def option_text(answer_order: int) -> tuple[str, str]:
    return (
        ("same noun sense", "different noun senses")
        if answer_order == 0
        else ("different noun senses", "same noun sense")
    )


def render_prompt(
    concept: dict[str, Any], template: int, quadrant: str, answer_order: int,
) -> tuple[str, dict[str, Any]]:
    state = quadrant_state(concept, quadrant)
    body = TEMPLATES[template].format(
        definition=concept["first_definition"],
        term=state["right_term"],
        lexname=state["right_lexname"].replace("noun.", ""),
        hypernym=state["right_hypernym"],
    )
    a_text, b_text = option_text(answer_order)
    raw_prompt = f"{body}\nA. {a_text}\nB. {b_text}\nAnswer with A or B only."
    expected_class = (
        "a" if state["semantic_same"] == (answer_order == 0) else "b"
    )
    state["expected_class"] = expected_class
    return raw_prompt, state


def build_case(
    tokenizer,
    model_name: str,
    concept: dict[str, Any],
    template: int,
    quadrant: str,
    answer_order: int,
    case_index: int,
) -> dict[str, Any]:
    raw_prompt, state = render_prompt(concept, template, quadrant, answer_order)
    rendered = (
        relation_source.base.behavior_tools.render_native(
            tokenizer, model_name, raw_prompt, with_system=False
        )
        + ASSISTANT_PREFILL
    )
    input_ids = [
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    candidate_labels = {"a": "A", "b": "B"}
    candidate_token_ids = {
        key: tools.continuation_ids(tokenizer, rendered, label)
        for key, label in candidate_labels.items()
    }
    record_id = (
        f"phase1113.{model_name}.{concept['split']}.{concept['concept_id']}."
        f"t{template}.q{QUADRANTS.index(quadrant)}.o{answer_order}"
    )
    return {
        "schema_version": "phase1113_wordnet_semantic_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": record_id,
        "concept_id": concept["concept_id"],
        "item_index": concept["item_index"],
        "split": concept["split"],
        "template": template,
        "quadrant": quadrant,
        "surface_same": state["surface_same"],
        "semantic_same": state["semantic_same"],
        "answer_order": answer_order,
        "base": concept["base"],
        "right_term": state["right_term"],
        "reference_synset_offset": concept["first_synset_offset"],
        "right_synset_offset": state["right_synset_offset"],
        "reference_definition": concept["first_definition"],
        "right_lexname": state["right_lexname"],
        "right_hypernym": state["right_hypernym"],
        "expected_class": state["expected_class"],
        "candidate_labels": candidate_labels,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {
            key: [int(values[0])] for key, values in candidate_token_ids.items()
        },
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "query_position": len(input_ids) - 1,
        "prompt_digest": hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
    }


def build_model_cases(
    tokenizer, model_name: str, selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in selected:
        for template in TEMPLATES_BY_SPLIT[concept["split"]]:
            for quadrant in QUADRANTS:
                for answer_order in ANSWER_ORDERS:
                    rows.append(build_case(
                        tokenizer,
                        model_name,
                        concept,
                        template,
                        quadrant,
                        answer_order,
                        len(rows),
                    ))
    return rows


def audit_model(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    templates_per_split = len(next(iter(TEMPLATES_BY_SPLIT.values())))
    expected_count = (
        SELECTED_ITEM_COUNT
        * templates_per_split
        * len(QUADRANTS)
        * len(ANSWER_ORDERS)
    )
    split_counts = Counter(row["split"] for row in rows)
    cell_counts = Counter(
        (row["split"], row["template"], row["quadrant"], row["answer_order"])
        for row in rows
    )
    class_counts = Counter(row["expected_class"] for row in rows)
    checks = {
        "case_count": len(rows) == expected_count,
        "record_ids_unique": len({row["record_id"] for row in rows}) == len(rows),
        "case_indices_contiguous": [row["case_index"] for row in rows] == list(range(len(rows))),
        "split_counts_balanced": set(split_counts.values()) == {expected_count // len(SPLITS)},
        "factor_cells_balanced": set(cell_counts.values()) == {ITEMS_PER_SPLIT},
        "expected_classes_balanced": class_counts["a"] == class_counts["b"],
        "candidate_continuations_one_token": all(
            len(values) == 1
            for row in rows
            for values in row["candidate_token_ids"].values()
        ),
        "candidate_tokens_distinct": all(
            row["candidate_first_token_ids"]["a"]
            != row["candidate_first_token_ids"]["b"]
            for row in rows
        ),
        "semantic_factor_matches_synset": all(
            row["semantic_same"]
            == (row["reference_synset_offset"] == row["right_synset_offset"])
            for row in rows
        ),
        "surface_factor_matches_term": all(
            row["surface_same"] == (row["base"] == row["right_term"])
            for row in rows
        ),
        "reference_definition_has_no_target_term": all(
            not (
                set(content_words(row["reference_definition"]))
                & set(content_words(f"{row['base']} {row['right_term']}"))
            )
            for row in rows
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
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "minimum_token_length": min(len(row["input_ids"]) for row in rows),
        "maximum_token_length": max(len(row["input_ids"]) for row in rows),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(rows),
    }


def main() -> None:
    synsets, lemma_synsets, metadata = parse_wordnet()
    eligible, rejection_counts = eligible_inventory(synsets, lemma_synsets, metadata)
    selected, globally_disjoint_count = select_inventory(eligible)
    split_terms = {
        split: {
            term
            for row in selected
            if row["split"] == split
            for term in (row["base"], row["alternate_first"], row["alternate_second"])
        }
        for split in SPLITS
    }
    lexical_isolation = all(
        not (split_terms[left] & split_terms[right])
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1:]
    )
    global_checks = {
        "wordnet_archive_sha256_verified": file_sha256(WORDNET_ARCHIVE) == WORDNET_SHA256,
        "eligible_inventory_nonempty": len(eligible) >= SELECTED_ITEM_COUNT,
        "globally_term_disjoint_inventory_sufficient": globally_disjoint_count >= SELECTED_ITEM_COUNT,
        "selected_item_count": len(selected) == SELECTED_ITEM_COUNT,
        "split_item_counts": all(
            sum(row["split"] == split for row in selected) == ITEMS_PER_SPLIT
            for split in SPLITS
        ),
        "split_target_terms_disjoint": lexical_isolation,
        "template_families_disjoint": len(set().union(
            *(set(values) for values in TEMPLATES_BY_SPLIT.values())
        )) == len(TEMPLATES),
        "no_model_outputs_used_for_selection": True,
    }
    if not all(global_checks.values()):
        raise RuntimeError(f"global protocol checks failed: {global_checks}")

    protocol_root = OUT_ROOT / "protocol"
    write_json(protocol_root / "source_manifest.json", metadata["source"])
    write_json(protocol_root / "eligible_inventory.json", {
        "eligible_count": len(eligible),
        "globally_term_disjoint_count": globally_disjoint_count,
        "rejection_counts": rejection_counts,
        "eligible": eligible,
        "inventory_digest": digest(eligible),
    })
    write_json(protocol_root / "selected_concepts.json", {
        "selected_count": len(selected),
        "selected": selected,
        "selected_digest": digest(selected),
    })

    model_audits: dict[str, Any] = {}
    case_digests: dict[str, str] = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        rows = build_model_cases(tokenizer, model_name, selected)
        audit = audit_model(rows, model_name)
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"{model_name} protocol audit failed: {audit['checks']}")
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", rows)
        model_audits[model_name] = audit
        case_digests[model_name] = audit["case_digest"]

    preregistration = {
        "schema_version": "phase1113_wordnet_semantic_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "splits": list(SPLITS),
        "items_per_split": ITEMS_PER_SPLIT,
        "quadrants": list(QUADRANTS),
        "answer_orders": list(ANSWER_ORDERS),
        "templates_by_split": {
            key: list(values) for key, values in TEMPLATES_BY_SPLIT.items()
        },
        "thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_manifest_digest": metadata["source"]["source_digest"],
        "selected_concepts_digest": digest(selected),
        "case_digests": case_digests,
        "interpretive_limits": [
            "The factors are highlighted-noun surface identity and WordNet noun-synset identity, not total prompt-token overlap and unrestricted natural meaning.",
            "WordNet definitions, lexicographer files, and immediate hypernyms make this a source-grounded metalinguistic task, not a natural next-token semantic-routing task.",
            "A behavior pass qualifies the material and interface only; it is not evidence for a hidden semantic coordinate or payload reader.",
            "No old model output participates in item selection. Stable hashes and source-only filters freeze all concepts before Phase1113 model execution.",
            "No hidden-state or causal experiment is automatic in Phase1113, even if two models pass behavior gates.",
        ],
    }
    preregistration["protocol_digest"] = digest(preregistration)
    write_json(protocol_root / "preregistration.json", preregistration)
    audit = {
        "schema_version": "phase1113_wordnet_semantic_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "global_checks": global_checks,
        "model_audits": model_audits,
        "all_checks_passed": all(global_checks.values()) and all(
            row["all_checks_passed"] for row in model_audits.values()
        ),
    }
    audit["audit_digest"] = digest(audit)
    write_json(protocol_root / "audit.json", audit)
    print(json.dumps({
        "phase": PHASE,
        "eligible_count": len(eligible),
        "globally_term_disjoint_count": globally_disjoint_count,
        "selected_count": len(selected),
        "case_count_per_model": model_audits["qwen3"]["case_count"],
        "all_checks_passed": audit["all_checks_passed"],
        "protocol_digest": preregistration["protocol_digest"],
        "audit_digest": audit["audit_digest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
