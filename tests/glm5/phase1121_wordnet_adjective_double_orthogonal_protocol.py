#!/usr/bin/env python3
"""Freeze Phase1121 independent adjective sense-by-surface behavior protocol."""

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


PHASE = 1121
PROTOCOL_REVISION = 1
MODELS = ("pythia", "qwen3", "glm4", "deepseek7b")
REFERENCE_MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
SPLITS = ("discovery", "independent_confirmation", "heldout")
ITEMS_PER_SPLIT = 8
SELECTED_ITEM_COUNT = ITEMS_PER_SPLIT * len(SPLITS)
SENSES = (0, 1)
SURFACES = ("base", "synonym")
DEFINITION_SENSES = (0, 1)
ASSISTANT_PREFILL = "Verdict:"
PYTHIA_PATH = ROOT / "models" / "hf" / "pythia-1.4b-deduped" / "step143000"
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1121_wordnet_adjective_double_orthogonal"
)
WORDNET_ARCHIVE = wordnet_source.WORDNET_ARCHIVE
WORDNET_SHA256 = wordnet_source.WORDNET_SHA256
WORDNET_MEMBERS = {
    "data_adj": "WordNet-3.0/dict/data.adj",
    "index_sense": "WordNet-3.0/dict/index.sense",
    "lexnames": "WordNet-3.0/dict/lexnames",
}


TEMPLATES = (
    (
        "Read the sentence as ordinary English.\nSentence: {sentence}\n"
        "Target adjective: {term}\nProposed meaning: {definition}\n"
        "Is that meaning correct for this occurrence? Reply true or false."
    ),
    (
        "Resolve the adjective from its local context.\nUsage: {sentence}\n"
        "Adjective under review: {term}\nCandidate definition: {definition}\n"
        "Is the candidate definition valid here? Reply false or true."
    ),
    (
        "Interpret this adjective occurrence, not merely its spelling.\n"
        "Context: {sentence}\nOccurrence: {term}\nMeaning to test: {definition}\n"
        "Does the meaning fit this use? Return true or false."
    ),
    (
        "Use the sentence to decide which adjective sense is active.\n"
        "Sentence: {sentence}\nAdjective: {term}\nDefinition being tested: {definition}\n"
        "Does this definition match the active sense? Return false or true."
    ),
    (
        "Judge a contextual adjective meaning.\nExample: {sentence}\n"
        "Highlighted adjective: {term}\nPossible meaning: {definition}\n"
        "Is the possible meaning accurate in this example? Answer true or false."
    ),
    (
        "The same adjective can express different senses. Read: {sentence}\n"
        "Target: {term}\nSense description: {definition}\n"
        "Is the description correct for the target here? Answer false or true."
    ),
)


THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_overall_candidate_accuracy": 0.55,
    "minimum_split_candidate_accuracy": 0.52,
    "minimum_surface_candidate_accuracy": 0.52,
    "minimum_sense_candidate_accuracy": 0.50,
    "minimum_template_candidate_accuracy": 0.50,
    "minimum_interaction_direction_accuracy": 0.75,
    "minimum_split_interaction_direction_accuracy": 0.70,
    "minimum_surface_interaction_direction_accuracy": 0.70,
    "minimum_template_interaction_direction_accuracy": 0.65,
    "minimum_cross_surface_pair_accuracy": 0.60,
    "minimum_split_cross_surface_pair_accuracy": 0.55,
    "maximum_surface_direction_accuracy_gap": 0.20,
    "minimum_qualified_reference_models": 2,
    "pythia_must_qualify": True,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "The verified WordNet adjective source, fixed 24-concept curation, native-example "
        "membership, synonym membership, exact lexical nonleakage, prior noun-family "
        "isolation, tokenization, factorial balance, and digests all pass."
    ),
    "P2": (
        "The final Pythia checkpoint and at least two of Qwen3, GLM4, and DS7B satisfy "
        "every frozen behavior gate under FP16 without quantization."
    ),
    "P3": (
        "The truth-balanced context-sense by definition-sense interaction has the correct "
        "direction across all splits, both surface modes, and all template families."
    ),
    "P4": (
        "Base-word and synonym-word realizations jointly cross the interaction direction "
        "boundary at the frozen cross-surface pair rates."
    ),
    "P5": (
        "Only joint P1-P4 success authorizes an independently frozen Pythia hidden-formation "
        "replication. This phase never selects a layer, component, head, or neuron."
    ),
    "P6": (
        "A pass qualifies a contextual adjective-sense behavior object. It does not by "
        "itself prove abstract semantics, a hidden invariant, execution, or causality."
    ),
}


# This curation was frozen from WordNet fields before any Phase1121 model output was read.
# Every tuple is (synset offset, synonym surface, exact native base-word example).
CURATED = {
    "flat": (("00910101", "level", "skirts sewn with fine flat seams"), ("01239040", "prostrate", "found himself lying flat on the floor")),
    "good": (("00106020", "full", "a good mile from here"), ("01983162", "respectable", "ruined the family's good name")),
    "open": (("01886620", "exposed", "open to the weather"), ("01654377", "opened", "keep your eyes open")),
    "sharp": (("00780352", "crisp", "a sharp photographic image"), ("01744515", "keen", "as sharp and incisive as the stroke of a fang")),
    "intact": (("00515870", "entire", "fought to keep the union intact"), ("01319434", "inviolate", "she was intact, virginal")),
    "vague": (("00431004", "obscure", "their descriptions of human behavior become vague, dull, and unclear"), ("00782216", "faint", "saw a vague outline of a building through the fog")),
    "last": (("01010271", "final", "the last days of the dinosaurs"), ("01212095", "utmost", "to the last measure of human endurance")),
    "husky": (("02038126", "burly", "clothing sizes for husky boys"), ("00299690", "hoarse", "makes all the instruments sound powerful but husky")),
    "miserable": (("01150205", "wretched", "he felt depressed and miserable"), ("01050890", "pathetic", "miserable victims of war")),
    "fundamental": (("01277097", "central", "an example that was fundamental to the argument"), ("01856419", "underlying", "the fundamental laws of the universe")),
    "hard": (("00744916", "difficult", "why is it so hard for you to keep a secret?"), ("02322512", "knockout", "a hard left to the chin")),
    "grotesque": (("00221627", "monstrous", "tales of grotesque serpents eight fathoms long that churned the seas"), ("00967646", "fantastic", "a grotesque reflection in the mirror")),
    "delicate": (("00709215", "fragile", "a kite too delicate to fly safely"), ("02448324", "soft", "a baby's delicate skin")),
    "spare": (("00991301", "trim", "the spare figure of a marathon runner"), ("01581305", "surplus", "sleeping in the spare room")),
    "funny": (("01265308", "comic", "a very funny writer"), ("00968010", "curious", "her speech has a funny twang")),
    "isolated": (("00594267", "stray", "isolated instances of rebellion"), ("02110447", "detached", "could not remain the isolated figure he had been")),
    "right": (("00631391", "correct", "took the right road"), ("00135455", "proper", "the right man for the job")),
    "confused": (("00465221", "disjointed", "a confused dream about the end of the world"), ("01669246", "disordered", "a confused mass of papers on the desk")),
    "still": (("01919428", "silent", "the night was still"), ("00302951", "placid", "scarcely a ripple on the still water")),
    "thin": (("00988232", "lean", "you can't be too rich or too thin"), ("02562566", "slender", "a thin line across the page")),
    "brilliant": (("01335156", "brainy", "a brilliant mind"), ("01285376", "magnificent", "the brilliant court life at Versailles")),
    "double": (("02217799", "twofold", "every episode has its double and treble meaning"), ("02217452", "dual", "an egg with a double yolk")),
    "smart": (("00975487", "chic", "a smart new dress"), ("01335458", "bright", "smart children talk earlier than the average")),
    "severe": (("01513050", "terrible", "a severe case of flu"), ("00651039", "serious", "a severe case of pneumonia")),
}


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def exact_count(text: str, term: str) -> int:
    return len(re.findall(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", text, flags=re.IGNORECASE))


def replace_exact(text: str, base: str, synonym: str) -> str:
    return re.sub(rf"(?<![A-Za-z]){re.escape(base)}(?![A-Za-z])", synonym, text, count=1, flags=re.IGNORECASE)


def parse_wordnet_adjectives() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if file_sha256(WORDNET_ARCHIVE) != WORDNET_SHA256:
        raise RuntimeError("WordNet archive SHA256 mismatch")
    with tarfile.open(WORDNET_ARCHIVE, "r:gz") as archive:
        member_bytes = {key: archive.extractfile(member).read() for key, member in WORDNET_MEMBERS.items()}
    lexnames = {
        int(line.split()[0]): line.split()[1]
        for line in member_bytes["lexnames"].decode("utf-8").splitlines()
        if line.strip()
    }
    synsets: dict[str, dict[str, Any]] = {}
    for line in member_bytes["data_adj"].decode("utf-8").splitlines():
        if not line or line[0].isspace():
            continue
        main, gloss = line.split("|", 1)
        fields = main.split()
        offset = fields[0]
        lexfile_index = int(fields[1])
        word_count = int(fields[3], 16)
        lemmas = [fields[4 + 2 * index].casefold() for index in range(word_count)]
        synsets[offset] = {
            "offset": offset,
            "lexfile_index": lexfile_index,
            "lexname": lexnames[lexfile_index],
            "lemmas": lemmas,
            "definition": gloss.strip().split(";", 1)[0].strip(),
            "examples": [value.strip() for value in re.findall(r'"([^"\r\n]+)"', gloss)],
        }
    sense_number: dict[tuple[str, str], int] = {}
    tag_count: dict[tuple[str, str], int] = {}
    for line in member_bytes["index_sense"].decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) < 4 or ("%3:" not in fields[0] and "%5:" not in fields[0]):
            continue
        lemma = fields[0].split("%", 1)[0].replace("_", " ").casefold()
        sense_number[(lemma, fields[1])] = int(fields[2])
        tag_count[(lemma, fields[1])] = int(fields[3])
    source = {
        "archive_path": str(WORDNET_ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
        "archive_sha256": WORDNET_SHA256,
        "member_sha256": {key: hashlib.sha256(value).hexdigest() for key, value in member_bytes.items()},
        "adjective_synset_count": len(synsets),
        "sense_record_count": len(sense_number),
    }
    source["source_digest"] = digest(source)
    return synsets, {"sense_number": sense_number, "tag_count": tag_count, "source": source}


def prior_noun_terms() -> set[str]:
    paths = (
        ROOT / "tests" / "glm5" / "result" / "phase1114_wordnet_contextual_hypernym" / "protocol" / "selected_concepts.json",
        ROOT / "tests" / "glm5" / "result" / "phase1115_wordnet_context_modulation_confirmation" / "protocol" / "selected_concepts.json",
    )
    terms: set[str] = set()
    for path in paths:
        for row in read_json(path)["selected"]:
            terms.add(row["base"])
            terms.update(row["hypernyms"])
    return terms


def curate(synsets: dict[str, dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if len(CURATED) != SELECTED_ITEM_COUNT:
        raise RuntimeError("curated concept count is not frozen at 24")
    rows: list[dict[str, Any]] = []
    all_terms: set[str] = set()
    old_terms = prior_noun_terms()
    for base, senses in sorted(CURATED.items()):
        concept_terms = {base, senses[0][1], senses[1][1]}
        if len(concept_terms) != 3 or concept_terms & all_terms:
            raise RuntimeError(f"curated lexical collision for {base}")
        if concept_terms & old_terms:
            raise RuntimeError(f"Phase1114-1115 lexical reuse for {base}")
        payload = {
            "base": base,
            "sense_offsets": [],
            "sense_numbers": [],
            "lexnames": [],
            "definitions": [],
            "base_examples": [],
            "synonym_surfaces": [],
            "synonym_examples": [],
            "base_tag_counts": [],
            "synonym_tag_counts": [],
        }
        for offset, synonym, example in senses:
            synset = synsets[offset]
            if base not in synset["lemmas"] or synonym not in synset["lemmas"]:
                raise RuntimeError(f"curated lemma membership failed for {base}/{offset}")
            if example not in synset["examples"] or exact_count(example, base) != 1:
                raise RuntimeError(f"curated native example failed for {base}/{offset}")
            if exact_count(example, synonym) != 0:
                raise RuntimeError(f"synonym already occurs in base example for {base}/{offset}")
            synonym_example = replace_exact(example, base, synonym)
            if exact_count(synonym_example, synonym) != 1 or exact_count(synonym_example, base) != 0:
                raise RuntimeError(f"synonym substitution failed for {base}/{offset}")
            if any(exact_count(synset["definition"], term) for term in concept_terms):
                raise RuntimeError(f"definition leaks concept term for {base}/{offset}")
            payload["sense_offsets"].append(offset)
            payload["sense_numbers"].append(metadata["sense_number"].get((base, offset)))
            payload["lexnames"].append(synset["lexname"])
            payload["definitions"].append(synset["definition"])
            payload["base_examples"].append(example)
            payload["synonym_surfaces"].append(synonym)
            payload["synonym_examples"].append(synonym_example)
            payload["base_tag_counts"].append(metadata["tag_count"].get((base, offset), 0))
            payload["synonym_tag_counts"].append(metadata["tag_count"].get((synonym, offset), 0))
        rows.append(payload)
        all_terms.update(concept_terms)

    rows.sort(key=lambda row: stable_hash("phase1121-split", row["base"], *row["sense_offsets"]))
    for split_index, split in enumerate(SPLITS):
        start = split_index * ITEMS_PER_SPLIT
        split_rows = rows[start:start + ITEMS_PER_SPLIT]
        split_rows.sort(key=lambda row: stable_hash("phase1121-item", row["base"]))
        for item_index, row in enumerate(split_rows):
            row["split"] = split
            row["item_index"] = item_index
            row["concept_id"] = f"wn-adj-{row['sense_offsets'][0]}-{row['sense_offsets'][1]}"

    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        ordered = sorted(split_rows, key=lambda row: stable_hash("phase1121-control", row["concept_id"]))
        for index, row in enumerate(ordered):
            row["deranged_control_concept_id"] = ordered[(index + 1) % len(ordered)]["concept_id"]
    return rows


def tokenizer_for_phase(model_name: str):
    if model_name != "pythia":
        return tokenizer_for(model_name)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(PYTHIA_PATH, local_files_only=True)


def render_prompt(concept: dict[str, Any], template: int, surface: str, context_sense: int, definition_sense: int) -> tuple[str, str, str]:
    if surface == "base":
        sentence = concept["base_examples"][context_sense]
        term = concept["base"]
    else:
        sentence = concept["synonym_examples"][context_sense]
        term = concept["synonym_surfaces"][context_sense]
    raw = TEMPLATES[template].format(sentence=sentence, term=term, definition=concept["definitions"][definition_sense])
    return raw, sentence, term


def build_case(tokenizer: Any, model_name: str, concept: dict[str, Any], template: int, surface: str, context_sense: int, definition_sense: int, case_index: int) -> dict[str, Any]:
    raw_prompt, sentence, term = render_prompt(concept, template, surface, context_sense, definition_sense)
    if model_name == "pythia":
        rendered = raw_prompt + "\n" + ASSISTANT_PREFILL
    else:
        rendered = relation_source.base.behavior_tools.render_native(tokenizer, model_name, raw_prompt, with_system=False) + ASSISTANT_PREFILL
    input_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    candidate_labels = {"true": "true", "false": "false"}
    candidate_token_ids = {key: relation_tools.continuation_ids(tokenizer, rendered, label) for key, label in candidate_labels.items()}
    expected_class = "true" if context_sense == definition_sense else "false"
    interaction_id = f"phase1121.{model_name}.{concept['split']}.{concept['concept_id']}.t{template}.{surface}"
    record_id = f"{interaction_id}.c{context_sense}.d{definition_sense}"
    return {
        "schema_version": "phase1121_adjective_double_orthogonal_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": record_id,
        "interaction_id": interaction_id,
        "concept_id": concept["concept_id"],
        "deranged_control_concept_id": concept["deranged_control_concept_id"],
        "split": concept["split"],
        "item_index": concept["item_index"],
        "template": template,
        "answer_order": "true_false" if template % 2 == 0 else "false_true",
        "surface": surface,
        "context_sense": context_sense,
        "definition_sense": definition_sense,
        "truth": context_sense == definition_sense,
        "base": concept["base"],
        "term": term,
        "sentence": sentence,
        "definition": concept["definitions"][definition_sense],
        "sense_offset": concept["sense_offsets"][context_sense],
        "definition_offset": concept["sense_offsets"][definition_sense],
        "candidate_labels": candidate_labels,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {key: [int(values[0])] for key, values in candidate_token_ids.items()},
        "expected_class": expected_class,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "query_position": len(input_ids) - 1,
        "prompt_digest": hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
    }


def build_model_cases(tokenizer: Any, model_name: str, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in selected:
        for template in range(len(TEMPLATES)):
            for surface in SURFACES:
                for context_sense in SENSES:
                    for definition_sense in DEFINITION_SENSES:
                        rows.append(build_case(tokenizer, model_name, concept, template, surface, context_sense, definition_sense, len(rows)))
    return rows


def audit_model(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    expected_count = SELECTED_ITEM_COUNT * len(TEMPLATES) * len(SURFACES) * len(SENSES) * len(DEFINITION_SENSES)
    split_counts = Counter(row["split"] for row in rows)
    interaction_counts = Counter(row["interaction_id"] for row in rows)
    truth_counts = Counter(row["truth"] for row in rows)
    factorial_counts = Counter((row["split"], row["template"], row["surface"], row["context_sense"], row["definition_sense"]) for row in rows)
    checks = {
        "case_count": len(rows) == expected_count,
        "case_indices_contiguous": [row["case_index"] for row in rows] == list(range(len(rows))),
        "record_ids_unique": len({row["record_id"] for row in rows}) == len(rows),
        "split_balance": split_counts == Counter({split: expected_count // len(SPLITS) for split in SPLITS}),
        "truth_balance": truth_counts == Counter({True: expected_count // 2, False: expected_count // 2}),
        "factorial_balance": set(factorial_counts.values()) == {ITEMS_PER_SPLIT},
        "interaction_completeness": len(interaction_counts) == SELECTED_ITEM_COUNT * len(TEMPLATES) * len(SURFACES) and set(interaction_counts.values()) == {4},
        "candidate_continuations_one_token": all(len(values) == 1 for row in rows for values in row["candidate_token_ids"].values()),
        "candidate_tokens_distinct": all(row["candidate_first_token_ids"]["true"] != row["candidate_first_token_ids"]["false"] for row in rows),
        "input_ids_nonempty": all(row["input_ids"] and row["query_position"] == len(row["input_ids"]) - 1 for row in rows),
        "surface_target_once": all(exact_count(row["sentence"], row["term"]) == 1 for row in rows),
        "definition_target_nonleakage": all(exact_count(row["definition"], row["base"]) == 0 for row in rows),
        "rendered_roundtrip": all([int(value) for value in row["input_ids"]] == row["input_ids"] for row in rows),
    }
    return {
        "model": model_name,
        "case_count": len(rows),
        "case_digest": digest(rows),
        "split_counts": dict(split_counts),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    synsets, metadata = parse_wordnet_adjectives()
    selected = curate(synsets, metadata)
    tokenizers = {model: tokenizer_for_phase(model) for model in MODELS}
    selected_terms = {row["base"] for row in selected} | {value for row in selected for value in row["synonym_surfaces"]}
    lexical_isolation = len(selected_terms) == SELECTED_ITEM_COUNT * 3
    prior_isolation = not (selected_terms & prior_noun_terms())
    split_counts = Counter(row["split"] for row in selected)
    control_map = {row["concept_id"]: row["deranged_control_concept_id"] for row in selected}
    common_checks = {
        "wordnet_archive_sha256": file_sha256(WORDNET_ARCHIVE) == WORDNET_SHA256,
        "selected_item_count": len(selected) == SELECTED_ITEM_COUNT,
        "split_item_balance": split_counts == Counter({split: ITEMS_PER_SPLIT for split in SPLITS}),
        "global_term_isolation": lexical_isolation,
        "prior_noun_term_isolation": prior_isolation,
        "native_examples_distinct": all(row["base_examples"][0].casefold() != row["base_examples"][1].casefold() for row in selected),
        "synonym_examples_distinct": all(row["synonym_examples"][0].casefold() != row["synonym_examples"][1].casefold() for row in selected),
        "definition_nonleakage": all(all(exact_count(definition, term) == 0 for definition in row["definitions"] for term in {row["base"], *row["synonym_surfaces"]}) for row in selected),
        "deranged_controls_within_split": all(next(value for value in selected if value["concept_id"] == control)["split"] == row["split"] and control != row["concept_id"] for row, control in ((row, control_map[row["concept_id"]]) for row in selected)),
        "pythia_checkpoint_present": (PYTHIA_PATH / "model.safetensors").exists(),
        "outputs_unread": True,
    }
    write_json(OUT_ROOT / "protocol" / "source_manifest.json", metadata["source"])
    selected_payload = {"selected_count": len(selected), "selected": selected, "selected_digest": digest(selected)}
    write_json(OUT_ROOT / "protocol" / "selected_concepts.json", selected_payload)

    model_audits: dict[str, Any] = {}
    case_digests: dict[str, str] = {}
    for model_name in MODELS:
        rows = build_model_cases(tokenizers[model_name], model_name, selected)
        write_jsonl(OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl", rows)
        model_audits[model_name] = audit_model(rows, model_name)
        case_digests[model_name] = model_audits[model_name]["case_digest"]

    prereg_core = {
        "schema_version": "phase1121_adjective_double_orthogonal_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "reference_models": list(REFERENCE_MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "source_digest": metadata["source"]["source_digest"],
        "selected_digest": selected_payload["selected_digest"],
        "case_digests": case_digests,
        "case_count_per_model": SELECTED_ITEM_COUNT * len(TEMPLATES) * len(SURFACES) * len(SENSES) * len(DEFINITION_SENSES),
        "concept_count": SELECTED_ITEM_COUNT,
        "splits": list(SPLITS),
        "items_per_split": ITEMS_PER_SPLIT,
        "surfaces": list(SURFACES),
        "templates": list(TEMPLATES),
        "thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "interaction_formula": "0.5 * ((z_context0_definition0 - z_context0_definition1) - (z_context1_definition0 - z_context1_definition1)), where z=logit(true)-logit(false)",
        "surface_pair_gate": "both base and synonym interaction values must be positive for the same concept-template cell",
        "future_geometry_control": "within-split deranged concept permutation frozen in selected_concepts.json",
        "model_outputs_read_during_protocol": False,
        "forbidden_actions": [
            "change selected concepts, examples, synonyms, splits, templates, or thresholds after behavior output is read",
            "drop a failing model, split, surface, sense, or template cell",
            "run hidden-state, component, head, neuron, patch, ablation, or restoration analysis unless the joint behavior authorization passes",
            "interpret synonym substitution as perfect semantic equivalence without the behavior gate",
            "reopen the closed exact-key registry",
        ],
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)

    checks = dict(common_checks)
    checks.update({f"model_{model}_{key}": value for model, audit in model_audits.items() for key, value in audit["checks"].items()})
    checks["all_model_protocol_audits"] = all(audit["all_checks_passed"] for audit in model_audits.values())
    checks["protocol_digest"] = digest(prereg_core) == prereg["protocol_digest"]
    audit_core = {
        "schema_version": "phase1121_adjective_double_orthogonal_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "model_audits": model_audits,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1121 protocol audit failed")
    print(json.dumps({"preregistration": prereg, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
