#!/usr/bin/env python3
"""Freeze Phase1081 cue-family orthogonal natural routing protocol.

Every family uses the same eight record labels in a balanced Latin schedule.
Two records contain different natural cloze problems.  Content order, queried
record, label assignment, shell wording, and an identical-content negative
control are fully crossed.  No internal result is used to construct the
protocol or its held-out taxonomy prediction.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior
import phase1079_output_orthogonal_pattern_protocol as source


PHASE = 1081
PROTOCOL_REVISION = 3
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
BASE_FAMILIES = (
    "height_relation",
    "taxonomy_fruit",
    "taxonomy_animal",
    "color_property",
    "rare_semantics",
    "translation",
    "sentiment_polarity",
    "tense_marker",
)
EXPLORATORY_FAMILIES = (
    "contrast_conjunction",
    "punctuation_rule",
)
HELDOUT_FAMILY = "taxonomy_plant"
FAMILIES = BASE_FAMILIES + EXPLORATORY_FAMILIES + (HELDOUT_FAMILY,)
SPLITS = ("discovery", "confirmation")
PANELS = ("active", "duplicate")
LABELS = tuple(f"K{index}" for index in range(1, 9))
LABEL_PAIRS = tuple(
    (LABELS[index], LABELS[index + 1]) for index in range(0, 8, 2)
)
STATES = tuple(
    f"t{template}_c{panel}_m{mapping}_q{query}_w{label_swap}"
    for template in (0, 1)
    for panel in PANELS
    for mapping in (0, 1)
    for query in (0, 1)
    for label_swap in (0, 1)
)
CAPTURE_ROLES = (
    "record0_end",
    "record1_end",
    "selected_label",
    "request_end",
    "answer_boundary",
)
PRE_QUERY_ROLES = ("record0_end", "record1_end")
INTERMEDIATE_ROLES = ("request_end",)
CONDITIONINGS = ("all_finite", "behavior_supported")
ITEMS_PER_FAMILY_SPLIT = 12
GENERATION_UNITS_PER_FAMILY_SPLIT = 6
GENERATION_STEPS = 16
ASSISTANT_PREFILL = "Completion:"
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result" / "phase1081_latin_route_atlas"
)
SOURCE_PHASE1080 = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1080_natural_relevance_atlas"
    / "analysis"
    / "final_summary.json"
)


EVIDENCE_THRESHOLDS = {
    "candidate_accuracy_for_family_behavior": 0.80,
    "generation_first_accuracy_diagnostic": 0.50,
    "generation_target_before_distractor_accuracy": 0.75,
    "unit_behavior_support_fraction": 0.75,
    "permutation_p_max": 0.01,
    "minimum_base_family_top1": 5,
    "minimum_repeated_models_or_pairs": 2,
    "minimum_behavior_families": 5,
    "minimum_cross_model_content_advantage": 0.05,
    "minimum_content_advantage_pairs": 4,
    "maximum_control_to_content_ratio": 1.0,
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_hidden_finite_fraction": 0.97,
    "pre_query_tolerance": 1e-8,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "The family-label Latin audit, output matching, candidate-token, "
        "and pre-query prefix audits pass for every model."
    ),
    "P2": (
        "At least five of eight base families pass candidate and natural "
        "target-before-distractor generation gates in at least two models."
    ),
    "P3": (
        "Family-centered content-dependent route topology retrieves at "
        "least five of eight families across independent splits in at "
        "least two models under the exact label null."
    ),
    "P4": (
        "Confirmation content-dependent route topology retrieves at least "
        "five of eight families in at least two directed cross-model pairs."
    ),
    "P5": (
        "Content-dependent route identity score exceeds the matched "
        "duplicate-route identity score by at least 0.05 in at least four "
        "of six directed model pairs."
    ),
    "P6": (
        "Content-route topology computed separately under the two label "
        "assignments retrieves at least five of eight families in at least "
        "two models."
    ),
    "P7": (
        "The pooled median maximum label/shell control-to-content-route "
        "ratio is at most one in at least two models."
    ),
    "P8": (
        "Held-out taxonomy_plant is nearest taxonomy_fruit or "
        "taxonomy_animal in at least two models, and its strongest "
        "request-end response is middle-depth Attention or MLP."
    ),
    "P9": (
        "Every model retains at least 95% finite candidate observations, "
        "97% finite hidden-role observations, zero identity-repeat error, "
        "and zero query-only difference before the query divergence."
    ),
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


SHELLS = {
    0: (
        "Notebook:\n"
        "Record {label0}: {record0}\n"
        "Record {label1}: {record1}\n"
        "Request: Fill the blank in record {selected}.\n"
        "Return only the missing text."
    ),
    1: (
        "Archive:\n"
        "Entry {label0}: {record0}\n"
        "Entry {label1}: {record1}\n"
        "Request: Supply the blank for entry {selected}.\n"
        "Write only the missing text."
    ),
}

BANNED_MODE_WORDS = (
    "semantic", "index", "reason", "copy", "mode", "option",
    "choices", "candidate", "instruction",
)


FRUITS = (
    "apple", "orange", "banana", "grape", "peach", "pear",
    "plum", "mango", "lemon", "cherry", "pineapple", "strawberry",
    "blueberry", "raspberry", "watermelon", "papaya", "kiwi",
    "apricot", "coconut", "fig", "guava", "pomegranate",
    "tangerine", "grapefruit",
)
VEGETABLES = (
    "carrot", "celery", "potato", "onion", "lettuce", "radish",
    "broccoli", "spinach", "cabbage", "beet", "turnip", "asparagus",
    "cauliflower", "leek", "pea", "kale", "artichoke", "parsnip",
    "yam", "okra", "Brussels sprout", "bok choy", "fennel", "chard",
)
ANIMALS = (
    "dog", "cat", "tiger", "whale", "eagle", "dolphin", "horse",
    "rabbit", "bear", "shark", "owl", "fox", "lion", "giraffe",
    "zebra", "penguin", "turtle", "frog", "camel", "goat", "sheep",
    "monkey", "elephant", "kangaroo",
)
PLANTS = (
    "oak", "rose", "fern", "pine", "daisy", "moss", "tulip",
    "bamboo", "cactus", "maple", "lily", "grass", "orchid", "birch",
    "ivy", "sunflower", "cedar", "wheat", "clover", "palm", "spruce",
    "lavender", "basil", "willow",
)
OBJECTS = (
    "marker", "folder", "badge", "ribbon", "button", "card", "box",
    "cup", "tile", "flag", "sign", "lamp", "book", "bottle",
    "notebook", "pencil", "envelope", "plate", "ball", "key",
    "ticket", "stamp", "label", "token",
)
NAME_PAIRS = (
    ("Ava", "Ben"), ("Mia", "Noah"), ("Liam", "Emma"),
    ("Owen", "Lily"), ("Ethan", "Grace"), ("Lucas", "Chloe"),
    ("Henry", "Ella"), ("Jack", "Sofia"), ("Leo", "Ruby"),
    ("Adam", "Nora"), ("Ryan", "Iris"), ("Caleb", "Maya"),
    ("Dylan", "Zoe"), ("Evan", "Lucy"), ("Aaron", "Clara"),
    ("Isaac", "Alice"), ("Mason", "Eva"), ("Logan", "Sarah"),
    ("Oscar", "Julia"), ("Felix", "Anna"), ("Simon", "Laura"),
    ("Peter", "Diana"), ("David", "Elena"), ("James", "Maria"),
)
CONTRAST_ACTIONS = (
    ("tired", "kept working", "went to bed"),
    ("hungry", "finished the task", "ate dinner"),
    ("cold", "stayed outside", "put on a coat"),
    ("busy", "answered the call", "postponed the meeting"),
    ("ill", "attended class", "rested at home"),
    ("late", "walked slowly", "called a taxi"),
    ("afraid", "entered the room", "locked the door"),
    ("wet", "continued hiking", "changed clothes"),
    ("sleepy", "read another chapter", "went to sleep"),
    ("injured", "completed the race", "visited a doctor"),
    ("nervous", "gave the speech", "practiced again"),
    ("poor", "donated money", "saved every coin"),
    ("angry", "spoke calmly", "left the room"),
    ("exhausted", "washed the dishes", "took a nap"),
    ("confused", "solved the puzzle", "asked for help"),
    ("sick", "went to work", "took medicine"),
    ("worried", "smiled", "checked the report"),
    ("weak", "lifted the box", "sat down"),
    ("sad", "cheered the team", "called a friend"),
    ("thirsty", "kept running", "drank water"),
    ("uncertain", "made a decision", "reviewed the facts"),
    ("sunburned", "stayed on the beach", "used sunscreen"),
    ("overloaded", "accepted more work", "reduced the schedule"),
    ("dizzy", "continued dancing", "sat on a chair"),
)
PUNCTUATION_PAIRS = (
    ("Is the gate open", "The gate is open"),
    ("Did the train arrive", "The train arrived"),
    ("Can the bird fly", "The bird can fly"),
    ("Are the lights on", "The lights are on"),
    ("Was the package delivered", "The package was delivered"),
    ("Will the store close", "The store will close"),
    ("Does the key fit", "The key fits"),
    ("Is the water warm", "The water is warm"),
    ("Did Maya call", "Maya called"),
    ("Can Noah swim", "Noah can swim"),
    ("Are the windows clean", "The windows are clean"),
    ("Was the answer correct", "The answer was correct"),
    ("Will the bell ring", "The bell will ring"),
    ("Does the clock work", "The clock works"),
    ("Is the road clear", "The road is clear"),
    ("Did the child laugh", "The child laughed"),
    ("Can the door lock", "The door can lock"),
    ("Are the papers ready", "The papers are ready"),
    ("Was the room quiet", "The room was quiet"),
    ("Will the sun rise", "The sun will rise"),
    ("Does the phone charge", "The phone charges"),
    ("Is the soup hot", "The soup is hot"),
    ("Did the dog bark", "The dog barked"),
    ("Can the camera focus", "The camera can focus"),
)
TRANSLATIONS = (
    ("gato", "cat", "perro", "dog"),
    ("casa", "house", "libro", "book"),
    ("agua", "water", "fuego", "fire"),
    ("sol", "sun", "luna", "moon"),
    ("pan", "bread", "leche", "milk"),
    ("mesa", "table", "silla", "chair"),
    ("puerta", "door", "ventana", "window"),
    ("árbol", "tree", "flor", "flower"),
    ("cielo", "sky", "mar", "sea"),
    ("rojo", "red", "azul", "blue"),
    ("uno", "one", "dos", "two"),
    ("madre", "mother", "padre", "father"),
    ("hermano", "brother", "hermana", "sister"),
    ("día", "day", "noche", "night"),
    ("grande", "big", "pequeño", "small"),
    ("rápido", "fast", "lento", "slow"),
    ("feliz", "happy", "triste", "sad"),
    ("frío", "cold", "caliente", "hot"),
    ("blanco", "white", "negro", "black"),
    ("camino", "road", "ciudad", "city"),
    ("pájaro", "bird", "pez", "fish"),
    ("manzana", "apple", "naranja", "orange"),
    ("queso", "cheese", "huevo", "egg"),
    ("mano", "hand", "pie", "foot"),
)
RARE_WORDS = (
    ("饕餮", "gluttony", "petrichor", "aroma"),
    ("sesquipedalian", "verbosity", "defenestration", "ejection"),
    ("susurrus", "whisper", "crepuscular", "twilight"),
    ("ultracrepidarian", "presumption", "apricity", "warmth"),
    ("psithurism", "rustling", "zugzwang", "disadvantage"),
    ("logorrhea", "wordiness", "liminal", "boundary"),
    ("cynosure", "focus", "ataraxia", "tranquility"),
    ("palimpsest", "manuscript", "syzygy", "alignment"),
    ("lacuna", "gap", "anomie", "normlessness"),
    ("eldritch", "eeriness", "ineffable", "mystery"),
    ("mellifluous", "melody", "noctambulist", "sleepwalker"),
    ("oneiric", "dream", "pareidolia", "illusion"),
    ("quiddity", "essence", "raconteur", "storyteller"),
    ("sibilant", "hissing", "tintinnabulation", "ringing"),
    ("uxorious", "devotion", "velleity", "wish"),
    ("weltschmerz", "sadness", "xenial", "hospitality"),
    ("yūgen", "depth", "zenith", "peak"),
    ("abecedarian", "beginner", "bruxism", "grinding"),
    ("chiaroscuro", "contrast", "deliquescent", "dissolution"),
    ("epistolary", "letters", "floccinaucinihilipilification", "worthlessness"),
    ("garrulous", "talkative", "hapax", "uniqueness"),
    ("irenic", "peacefulness", "jejune", "dullness"),
    ("kakistocracy", "misrule", "limerence", "infatuation"),
    ("mondegreen", "mishearing", "numinous", "awe"),
)
MINERALS = (
    "quartz", "granite", "mica", "feldspar", "calcite", "gypsum",
    "halite", "magnetite", "hematite", "graphite", "fluorite",
    "dolomite", "topaz", "corundum", "pyrite", "malachite",
    "azurite", "kaolinite", "talc", "olivine", "zircon", "beryl",
    "apatite", "tourmaline",
)
SENTIMENT_PAIRS = (
    ("excellent", "terrible"), ("wonderful", "awful"),
    ("delightful", "horrible"), ("impressive", "disappointing"),
    ("pleasant", "unpleasant"), ("brilliant", "dreadful"),
    ("enjoyable", "miserable"), ("superb", "inferior"),
    ("admirable", "deplorable"), ("satisfying", "frustrating"),
    ("uplifting", "depressing"), ("charming", "repulsive"),
    ("favorable", "unfavorable"), ("successful", "unsuccessful"),
    ("rewarding", "regrettable"), ("encouraging", "discouraging"),
    ("appealing", "distasteful"), ("commendable", "shameful"),
    ("reassuring", "alarming"), ("valuable", "worthless"),
    ("joyful", "sorrowful"), ("promising", "hopeless"),
    ("lovely", "ugly"), ("generous", "cruel"),
)
TENSE_ACTIONS = (
    "walk home", "open the gate", "read the letter", "paint the wall",
    "call the office", "visit the museum", "clean the room",
    "finish the report", "repair the chair", "cook the meal",
    "carry the box", "water the garden", "close the window",
    "check the engine", "deliver the parcel", "wash the cup",
    "cross the bridge", "write the note", "lock the door",
    "move the table", "answer the question", "start the machine",
    "meet the teacher", "return the book",
)


def make_item(
    family: str,
    index: int,
    record0: str,
    answer0: str,
    record1: str,
    answer1: str,
) -> dict[str, Any]:
    return {
        "item_id": f"{family}.{index:02d}",
        "record0": record0,
        "record1": record1,
        "answer0": answer0,
        "answer1": answer1,
    }


def build_family_items() -> dict[str, tuple[dict[str, Any], ...]]:
    result: dict[str, list[dict[str, Any]]] = {
        family: [] for family in FAMILIES
    }
    for index in range(24):
        a, b = NAME_PAIRS[index]
        result["height_relation"].append(make_item(
            "height_relation", index,
            f"{a} is taller than {b}; therefore {a} is ___ than {b}.",
            "taller",
            f"{a} is shorter than {b}; therefore {a} is ___ than {b}.",
            "shorter",
        ))
        condition, contrary, expected = CONTRAST_ACTIONS[index]
        subject = a
        result["contrast_conjunction"].append(make_item(
            "contrast_conjunction", index,
            f"{subject} was {condition}, ___ {subject} {contrary}.",
            "but",
            f"{subject} was {condition}, ___ {subject} {expected}.",
            "so",
        ))
        question, statement = PUNCTUATION_PAIRS[index]
        result["punctuation_rule"].append(make_item(
            "punctuation_rule", index,
            f"The final punctuation in '{question}___' is missing.", "?",
            f"The final punctuation in '{statement}___' is missing.", ".",
        ))
        result["taxonomy_fruit"].append(make_item(
            "taxonomy_fruit", index,
            f"In everyday food grouping, a {FRUITS[index]} is a ___.", "fruit",
            f"In everyday food grouping, a {VEGETABLES[index]} is a ___.",
            "vegetable",
        ))
        result["taxonomy_animal"].append(make_item(
            "taxonomy_animal", index,
            f"In broad living categories, a {ANIMALS[index]} is an ___.", "animal",
            f"In broad living categories, an {PLANTS[index]} is a ___.", "plant",
        ))
        result["color_property"].append(make_item(
            "color_property", index,
            f"The {OBJECTS[index]} is painted red; its basic color is ___.", "red",
            f"The {OBJECTS[(index + 7) % 24]} is painted blue; its basic color is ___.",
            "blue",
        ))
        rare0, meaning0, rare1, meaning1 = RARE_WORDS[index]
        result["rare_semantics"].append(make_item(
            "rare_semantics", index,
            f"The rare term '{rare0}' is defined here by the word {meaning0}; "
            "its recorded meaning is ___.",
            meaning0,
            f"The rare term '{rare1}' is defined here by the word {meaning1}; "
            "its recorded meaning is ___.",
            meaning1,
        ))
        src0, dst0, src1, dst1 = TRANSLATIONS[index]
        result["translation"].append(make_item(
            "translation", index,
            f"The English translation of Spanish '{src0}' is ___.", dst0,
            f"The English translation of Spanish '{src1}' is ___.", dst1,
        ))
        positive, negative = SENTIMENT_PAIRS[index]
        result["sentiment_polarity"].append(make_item(
            "sentiment_polarity", index,
            f"The reviewer called the result {positive}; the evaluation was ___.",
            "positive",
            f"The reviewer called the result {negative}; the evaluation was ___.",
            "negative",
        ))
        action = TENSE_ACTIONS[index]
        result["tense_marker"].append(make_item(
            "tense_marker", index,
            f"Yesterday, {a} had to {action}; the event belongs to the ___.",
            "past",
            f"Tomorrow, {b} will {action}; the event belongs to the ___.",
            "future",
        ))
        result["taxonomy_plant"].append(make_item(
            "taxonomy_plant", index,
            f"In a plant-or-mineral classification, {PLANTS[index]} is a ___.",
            "plant",
            f"In a plant-or-mineral classification, {MINERALS[index]} is a ___.",
            "mineral",
        ))
    return {family: tuple(rows) for family, rows in result.items()}


ITEMS_BY_FAMILY = build_family_items()


def split_items(family: str, split: str) -> tuple[dict[str, Any], ...]:
    start = 0 if split == "discovery" else 12
    return ITEMS_BY_FAMILY[family][start:start + ITEMS_PER_FAMILY_SPLIT]


def state_factors(state: str) -> tuple[int, str, int, int, int]:
    match = re.fullmatch(
        r"t([01])_c(active|duplicate)_m([01])_q([01])_w([01])",
        state,
    )
    if not match:
        raise ValueError(f"invalid state: {state}")
    return (
        int(match.group(1)),
        str(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
    )


def label_pair(family: str, item_local_index: int) -> tuple[str, str]:
    family_index = FAMILIES.index(family)
    return LABEL_PAIRS[(family_index + item_local_index) % len(LABEL_PAIRS)]


def render_records(
    item: dict[str, Any], panel: str, mapping: int
) -> tuple[str, str, int, int]:
    if panel == "active":
        content = (
            (item["record0"], item["record1"])
            if mapping == 0
            else (item["record1"], item["record0"])
        )
        answers = (mapping, 1 - mapping)
        return str(content[0]), str(content[1]), answers[0], answers[1]
    record = str(item[f"record{mapping}"])
    return record, record, mapping, mapping


def encoded_width(tokenizer, value: str) -> int:
    return len(tokenizer.encode(" " + value, add_special_tokens=False))


def build_case(
    tokenizer,
    model_name: str,
    family: str,
    split: str,
    item: dict[str, Any],
    item_local_index: int,
    state: str,
    case_index: int,
) -> dict[str, Any]:
    template, panel, mapping, query, label_swap = state_factors(state)
    pair = label_pair(family, item_local_index)
    labels = pair if label_swap == 0 else (pair[1], pair[0])
    record0, record1, answer_index0, answer_index1 = render_records(
        item, panel, mapping
    )
    answer_index = answer_index0 if query == 0 else answer_index1
    selected = labels[query]
    raw_prompt = SHELLS[template].format(
        label0=labels[0],
        label1=labels[1],
        record0=record0,
        record1=record1,
        selected=selected,
    )
    line0 = f"{'Record' if template == 0 else 'Entry'} {labels[0]}: {record0}"
    line1 = f"{'Record' if template == 0 else 'Entry'} {labels[1]}: {record1}"
    request = (
        f"Request: Fill the blank in record {selected}."
        if template == 0
        else f"Request: Supply the blank for entry {selected}."
    )
    raw_spans = {
        "record0_end": source.mark(raw_prompt, line0, occurrence="first"),
        "record1_end": source.mark(raw_prompt, line1, occurrence="first"),
        "selected_label": source.mark(
            raw_prompt, selected, occurrence="last"
        ),
        "request_end": source.mark(raw_prompt, request, occurrence="last"),
    }
    rendered = behavior.render_native(
        tokenizer, model_name, raw_prompt, with_system=False
    )
    rendered += ASSISTANT_PREFILL
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_spans = offset_token_spans(
        tokenizer, rendered, raw_prompt, raw_spans
    )
    role_spans["answer_boundary"] = (
        len(input_ids) - 1,
        len(input_ids) - 1,
    )
    answers = (str(item["answer0"]), str(item["answer1"]))
    prefix = " "
    candidate_token_ids = {
        f"a{index}": behavior.continuation_ids(
            tokenizer, rendered, prefix, answer
        )
        for index, answer in enumerate(answers)
    }
    return {
        "schema_version": "phase1081_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{model_name}.{family}.{split}.{item['item_id']}.{state}",
        "unit_id": f"{family}.{split}.{item['item_id']}",
        "family": family,
        "split": split,
        "item_id": item["item_id"],
        "item_local_index": item_local_index,
        "state": state,
        "template": template,
        "panel": panel,
        "mapping": mapping,
        "query": query,
        "label_swap": label_swap,
        "label_pair": list(pair),
        "position_labels": list(labels),
        "selected_label_text": selected,
        "selected_position": query,
        "record0": record0,
        "record1": record1,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_spans": {
            role: [int(span[0]), int(span[1])]
            for role, span in role_spans.items()
        },
        "role_positions": {
            role: int(span[1]) for role, span in role_spans.items()
        },
        "answer_labels": list(answers),
        "answer_index": answer_index,
        "target_answer": answers[answer_index],
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {
            key: [int(values[0])]
            for key, values in candidate_token_ids.items()
        },
        "expected_class": f"a{answer_index}",
        "continuation_prefix": prefix,
        "label_token_widths": [
            encoded_width(tokenizer, value) for value in labels
        ],
    }


def common_prefix_divergence(token_rows: list[list[int]]) -> int:
    common = min(len(value) for value in token_rows)
    for index in range(common):
        if len({value[index] for value in token_rows}) > 1:
            return index
    return common


def audit_model(
    model_name: str,
    tokenizer,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roles_valid = True
    role_order = True
    candidates_disjoint = True
    expected_correct = True
    no_banned_words = True
    label_width_equal = True
    active_answer_depends_on_query = True
    duplicate_answer_independent_of_query = True
    label_swap_output_matched = True
    query_prefix_equal = True
    pre_query_roles_before_divergence = True
    active_records_distinct = True
    duplicate_records_equal = True
    for row in cases:
        by_unit[row["unit_id"]].append(row)
        width = len(row["input_ids"])
        roles_valid &= all(
            0 <= row["role_spans"][role][0]
            <= row["role_spans"][role][1] < width
            for role in CAPTURE_ROLES
        )
        positions = row["role_positions"]
        role_order &= (
            positions["record0_end"]
            < positions["record1_end"]
            < positions["selected_label"]
            <= positions["request_end"]
            < positions["answer_boundary"]
        )
        left = set(row["candidate_first_token_ids"]["a0"])
        right = set(row["candidate_first_token_ids"]["a1"])
        candidates_disjoint &= bool(left) and bool(right) and left.isdisjoint(right)
        expected_correct &= row["expected_class"] == f"a{row['answer_index']}"
        prompt_folded = row["raw_prompt"].casefold()
        no_banned_words &= not any(
            re.search(rf"\b{re.escape(word)}\b", prompt_folded)
            for word in BANNED_MODE_WORDS
        )
        label_width_equal &= len(set(row["label_token_widths"])) == 1
        if row["panel"] == "active":
            active_records_distinct &= row["record0"] != row["record1"]
        else:
            duplicate_records_equal &= row["record0"] == row["record1"]

    for rows in by_unit.values():
        lookup = {row["state"]: row for row in rows}
        for template in (0, 1):
            for panel in PANELS:
                for mapping in (0, 1):
                    for label_swap in (0, 1):
                        left = lookup[
                            f"t{template}_c{panel}_m{mapping}_q0_w{label_swap}"
                        ]
                        right = lookup[
                            f"t{template}_c{panel}_m{mapping}_q1_w{label_swap}"
                        ]
                        divergence = common_prefix_divergence([
                            left["input_ids"], right["input_ids"]
                        ])
                        query_prefix_equal &= divergence < min(
                            len(left["input_ids"]), len(right["input_ids"])
                        )
                        pre_query_roles_before_divergence &= all(
                            row["role_positions"][role] < divergence
                            for row in (left, right)
                            for role in PRE_QUERY_ROLES
                        )
                        if panel == "active":
                            active_answer_depends_on_query &= (
                                left["answer_index"] != right["answer_index"]
                            )
                        else:
                            duplicate_answer_independent_of_query &= (
                                left["answer_index"] == right["answer_index"]
                            )
                    for query in (0, 1):
                        first = lookup[
                            f"t{template}_c{panel}_m{mapping}_q{query}_w0"
                        ]
                        second = lookup[
                            f"t{template}_c{panel}_m{mapping}_q{query}_w1"
                        ]
                        label_swap_output_matched &= (
                            first["target_answer"] == second["target_answer"]
                        )

    label_counts = Counter(
        (row["family"], row["split"], label)
        for row in cases
        if row["state"] == "t0_cactive_m0_q0_w0"
        for label in row["label_pair"]
    )
    expected_label_units = ITEMS_PER_FAMILY_SPLIT // len(LABEL_PAIRS)
    labels_balanced = all(
        label_counts[(family, split, label)] == expected_label_units
        for family in FAMILIES
        for split in SPLITS
        for label in LABELS
    )
    item_ids = {
        (family, split): {
            row["item_id"]
            for row in cases
            if row["family"] == family and row["split"] == split
        }
        for family in FAMILIES
        for split in SPLITS
    }
    counts = Counter((row["family"], row["split"]) for row in cases)
    checks = {
        "case_count": len(cases)
        == len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT * len(STATES),
        "unit_count": len(by_unit)
        == len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT,
        "complete_units": all(len(rows) == len(STATES) for rows in by_unit.values()),
        "family_split_counts": all(
            count == ITEMS_PER_FAMILY_SPLIT * len(STATES)
            for count in counts.values()
        ) and len(counts) == len(FAMILIES) * len(SPLITS),
        "role_spans_valid": roles_valid,
        "role_order_valid": role_order,
        "candidate_first_tokens_disjoint": candidates_disjoint,
        "expected_class_matches_answer": expected_correct,
        "explicit_mode_words_absent": no_banned_words,
        "within_pair_label_width_equal": label_width_equal,
        "labels_balanced_within_every_family_split": labels_balanced,
        "active_answer_depends_on_query": active_answer_depends_on_query,
        "duplicate_answer_independent_of_query": duplicate_answer_independent_of_query,
        "label_swap_output_matched": label_swap_output_matched,
        "query_prefix_equal_before_divergence": query_prefix_equal,
        "pre_query_roles_before_divergence": pre_query_roles_before_divergence,
        "active_records_distinct": active_records_distinct,
        "duplicate_records_equal": duplicate_records_equal,
        "independent_item_splits": all(
            item_ids[(family, "discovery")].isdisjoint(
                item_ids[(family, "confirmation")]
            )
            for family in FAMILIES
        ),
    }
    return {
        "schema_version": "phase1081_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "label_counts": {
            f"{family}.{split}.{label}": label_counts[(family, split, label)]
            for family in FAMILIES
            for split in SPLITS
            for label in LABELS
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(cases),
    }


def build_model_cases(model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = tokenizer_for(model_name)
    cases: list[dict[str, Any]] = []
    case_index = 0
    for family in FAMILIES:
        for split in SPLITS:
            for item_local_index, item in enumerate(split_items(family, split)):
                for state in STATES:
                    cases.append(build_case(
                        tokenizer,
                        model_name,
                        family,
                        split,
                        item,
                        item_local_index,
                        state,
                        case_index,
                    ))
                    case_index += 1
    audit = audit_model(model_name, tokenizer, cases)
    return cases, audit


def main() -> None:
    protocol_root = OUT_ROOT / "protocol"
    model_case_digests = {}
    model_audits = {}
    for model_name in MODELS:
        cases, audit = build_model_cases(model_name)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"protocol audit failed for {model_name}: {audit}"
            )
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", cases)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_case_digests[model_name] = audit["case_digest"]
        model_audits[model_name] = audit

    source_summary = read_json(SOURCE_PHASE1080)
    prereg = {
        "schema_version": "phase1081_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "families": list(FAMILIES),
        "base_families": list(BASE_FAMILIES),
        "exploratory_families": list(EXPLORATORY_FAMILIES),
        "heldout_family": HELDOUT_FAMILY,
        "heldout_prediction": {
            "nearest_family_set": ["taxonomy_fruit", "taxonomy_animal"],
            "request_end_component_set": ["attention_output", "mlp_output"],
            "request_end_relative_depth_range": [1 / 3, 2 / 3],
        },
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "labels": list(LABELS),
        "label_pairs": [list(value) for value in LABEL_PAIRS],
        "states": list(STATES),
        "factor_definition": {
            "T": "two shared notebook/archive shell wordings",
            "C": "active distinct records or duplicate-content negative control",
            "M": "active content order or duplicated answer identity",
            "Q": "queried physical record position",
            "W": "record-label assignment swap",
        },
        "capture_roles": list(CAPTURE_ROLES),
        "pre_query_roles": list(PRE_QUERY_ROLES),
        "intermediate_roles": list(INTERMEDIATE_ROLES),
        "conditionings": list(CONDITIONINGS),
        "assistant_prefill": ASSISTANT_PREFILL,
        "items_per_family_split": ITEMS_PER_FAMILY_SPLIT,
        "case_count_per_model": (
            len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT * len(STATES)
        ),
        "unit_count_per_model": (
            len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT
        ),
        "model_case_digests": model_case_digests,
        "generation_units_per_family_split": GENERATION_UNITS_PER_FAMILY_SPLIT,
        "generation_steps": GENERATION_STEPS,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_phase1080_protocol_digest": source_summary["protocol_digest"],
        "source_phase1080_summary_digest": source_summary["summary_digest"],
        "primary_population": (
            "All finite preregistered observations. Behavior annotations "
            "never delete a unit from the descriptive map."
        ),
        "secondary_population": (
            "Units with at least 75% correct candidate comparisons across "
            "all 32 factorial states."
        ),
        "behavior_gate_definition": (
            "Candidate accuracy must reach 0.80 and, in 16-token natural "
            "generation, the target answer must occur before the paired "
            "distractor in at least 0.75 of audited cases. Strict first-token "
            "accuracy remains diagnostic and is not used to excuse an "
            "incorrect first candidate."
        ),
        "measurement_order": [
            "tokenizer and Latin-balance protocol audits",
            "behavior-only calibration without hidden-state inspection",
            "FP16 finite coverage and identity repeat",
            "query-only prefix causality audit",
            "active output-matched route",
            "duplicate route negative control",
            "content-dependent route subtraction",
            "label assignment and shell controls",
            "independent split retrieval",
            "cross-model retrieval and content advantage",
            "held-out taxonomy prediction",
            "automatic gate",
        ],
        "evidence_levels": {
            "L0": "finite active, duplicate, content, answer, label, and shell fields mapped",
            "L1": "content route retrieves across independent splits in at least two models",
            "L2": "L1 plus cross-label-assignment retrieval in at least two models",
            "L3": "L2 plus cross-model content retrieval and duplicate-route advantage",
            "L4": "L3 plus behavior support in at least two models",
            "L5": "L4 plus held-out family prediction",
            "L6": "causal support; forbidden in Phase1081",
        },
        "interpretation_limits": [
            "The shared record labels are routing handles, not claimed latent symbols.",
            "Content-route subtraction assumes approximately additive matched controls.",
            "A successful family profile may still reflect task syntax or cloze structure.",
            "Output-matched diagonal differences compare distinct prefixes and are descriptive.",
            "Natural generation success is necessary but not sufficient for mechanism claims.",
            "Target-before-distractor scoring tolerates explanatory prefixes "
            "but cannot by itself prove exact-format instruction following.",
            "Normalized-depth similarity is not physical-coordinate homology.",
            "No observation establishes minimality, optimality, brain homology, or a language law.",
        ],
        "automatic_next": {
            "continue_only_if": (
                "P1-P9 all pass, at least five base families reach L3, "
                "integrity passes, and no protocol leak is found."
            ),
            "next_task_if_passed": (
                "Pre-register component-level causal tests for a predicted "
                "family route without selecting raw activation peaks."
            ),
            "stop_if_failed": (
                "Retain the descriptive atlas and diagnose the failed "
                "identification or behavior gate; do not select neurons."
            ),
        },
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    audit = {
        "schema_version": "phase1081_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_audits": model_audits,
        "checks": {
            "all_model_audits_passed": all(
                value["all_checks_passed"] for value in model_audits.values()
            ),
            "model_order_frozen": tuple(prereg["models"]) == MODELS,
            "precision_fp16": prereg["precision"] == "fp16",
            "quantization_none": prereg["quantization"] == "none",
            "heldout_family_frozen": prereg["heldout_family"] == HELDOUT_FAMILY,
            "predictions_frozen": set(prereg["prospective_predictions"])
            == set(PROSPECTIVE_PREDICTIONS),
        },
    }
    audit["all_checks_passed"] = all(audit["checks"].values())
    audit["audit_digest"] = digest(audit)
    write_json(protocol_root / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"global protocol audit failed: {audit}")
    print({
        "phase": PHASE,
        "status": "protocol_frozen",
        "case_count_per_model": prereg["case_count_per_model"],
        "unit_count_per_model": prereg["unit_count_per_model"],
        "protocol_digest": prereg["protocol_digest"],
    })


if __name__ == "__main__":
    main()
