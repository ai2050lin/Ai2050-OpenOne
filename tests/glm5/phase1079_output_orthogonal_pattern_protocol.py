#!/usr/bin/env python3
"""Freeze the Phase1079 output-orthogonal predictive pattern atlas.

The primary controlled panel places a semantic request and an index-copy
request in the same prompt.  Both requests have the same answer.  Only the
late active-mode word changes, so the operation differential does not change
the expected output token.  A second panel removes the explicit candidate
list and tests whether the observed family topology transfers to completion.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior
import phase1077_nonblocking_pattern_atlas_protocol as source
import phase1078_shared_shell_pattern_atlas_protocol as source1078


PHASE = 1079
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
FAMILIES = (
    "height_relation",
    "contrast_conjunction",
    "punctuation_rule",
    "taxonomy_fruit",
    "taxonomy_animal",
    "color_property",
    "rare_semantics",
    "translation",
    "causal_connector",
)
BASE_FAMILIES = FAMILIES[:-1]
HELDOUT_FAMILY = "causal_connector"
SPLITS = ("discovery", "confirmation")
PANELS = ("controlled", "natural")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
CONTROLLED_STATES = tuple(
    f"t{template}_o{operation}_a{answer}_l{surface}"
    for template in (0, 1)
    for operation in (0, 1)
    for answer in (0, 1)
    for surface in (0, 1)
)
NATURAL_STATES = tuple(
    f"t{template}_a{answer}_l{surface}"
    for template in (0, 1)
    for answer in (0, 1)
    for surface in (0, 1)
)
STATES_BY_PANEL = {
    "controlled": CONTROLLED_STATES,
    "natural": NATURAL_STATES,
}
CAPTURE_ROLES = (
    "content_anchor",
    "context_or_options",
    "semantic_request",
    "active_mode",
    "answer_boundary",
)
PRE_MODE_ROLES = (
    "content_anchor",
    "context_or_options",
    "semantic_request",
)
CONDITIONINGS = ("all_finite", "behavior_supported")
ITEMS_PER_FAMILY_SPLIT = 12
NATURAL_GENERATION_CASES_PER_FAMILY_SPLIT = 8
NATURAL_GENERATION_STEPS = 5
ASSISTANT_PREFILL = "Completion:"
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1079_output_orthogonal_pattern_atlas"
)
SOURCE_PHASE1078 = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1078_shared_shell_pattern_atlas"
    / "analysis"
    / "final_summary.json"
)


EVIDENCE_THRESHOLDS = {
    "candidate_accuracy_for_behavior_annotation": 0.70,
    "natural_generation_first_accuracy": 0.50,
    "unit_behavior_support_fraction": 0.75,
    "permutation_p_max": 0.01,
    "minimum_repeated_models_or_pairs": 2,
    "minimum_base_family_top1": 5,
    "natural_controlled_transfer_top1": 5,
    "phase1078_alignment_drop_min": 0.15,
    "pre_mode_operation_tolerance": 1e-8,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "The output-matched semantic-versus-index operation topology "
        "retrieves base-family identity across independent splits in at "
        "least two models under an exact family-label permutation null."
    ),
    "P2": (
        "A family-centered output-matched operation topology retrieves at "
        "least five of eight base families in at least two directed "
        "cross-model comparisons."
    ),
    "P3": (
        "Removing the shared yes/no answer pair reduces late cross-family "
        "answer-direction alignment by at least 0.15 relative to Phase1078 "
        "in at least two models."
    ),
    "P4": (
        "Candidate-absent natural answer topology matches the controlled "
        "semantic-answer topology for at least five of eight base families "
        "in at least two models under an exact permutation null."
    ),
    "P5": (
        "The preregistered held-out causal_connector family is closest to "
        "contrast_conjunction in answer-boundary operation topology in at "
        "least two models, and its peak is a middle-to-late Attention or "
        "MLP event."
    ),
    "P6": (
        "Operation differences before the active-mode token are numerically "
        "zero within 1e-8, providing a causal-order instrumentation audit."
    ),
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


CONTROLLED_SHELLS = {
    0: (
        "Information: {evidence}\n"
        "Options: {options}\n"
        "Semantic request: {request}\n"
        "Index request: Copy the {ordinal} option.\n"
        "Active request: {mode}\n"
        "Output rule: return only the requested completion."
    ),
    1: (
        "Reference: {evidence}\n"
        "Choices: {options}\n"
        "Meaning request: {request}\n"
        "Position request: Return choice {ordinal_number}.\n"
        "Selected request: {mode}\n"
        "Completion rule: provide only the requested text."
    ),
    2: (
        "Evidence: {evidence}\n"
        "Candidate completions: {options}\n"
        "Language request: {request}\n"
        "Copy request: Use the {ordinal} candidate.\n"
        "Active instruction: {mode}\n"
        "Response rule: emit only the completion."
    ),
    3: (
        "Context: {evidence}\n"
        "Available completions: {options}\n"
        "Content request: {request}\n"
        "Slot request: Select candidate {ordinal_number}.\n"
        "Chosen instruction: {mode}\n"
        "Answer rule: write only the completion."
    ),
}
NATURAL_SHELLS = {
    0: (
        "Information: {evidence}\n"
        "Semantic request: {request}\n"
        "Generation mode: natural completion.\n"
        "Output rule: return only the completion."
    ),
    1: (
        "Reference: {evidence}\n"
        "Meaning request: {request}\n"
        "Generation mode: natural completion.\n"
        "Completion rule: provide only the requested text."
    ),
    2: (
        "Evidence: {evidence}\n"
        "Language request: {request}\n"
        "Generation mode: natural completion.\n"
        "Response rule: emit only the completion."
    ),
    3: (
        "Context: {evidence}\n"
        "Content request: {request}\n"
        "Generation mode: natural completion.\n"
        "Answer rule: write only the completion."
    ),
}


VEGETABLE_DISCOVERY = (
    "carrot", "celery", "cabbage", "spinach", "broccoli",
    "onion", "turnip", "radish", "lettuce", "asparagus",
    "cauliflower", "beet", "leek", "kale", "artichoke",
)
VEGETABLE_CONFIRMATION = (
    "parsnip", "fennel", "okra", "zucchini", "eggplant",
    "pumpkin", "pepper", "pea", "bean", "chard",
    "endive", "yam", "cassava", "rutabaga", "watercress",
)
MAMMAL_DISCOVERY = (
    "dog", "horse", "cow", "sheep", "goat", "lion",
    "tiger", "bear", "wolf", "fox", "rabbit", "elephant",
    "giraffe", "zebra", "otter",
)
BIRD_DISCOVERY = (
    "eagle", "sparrow", "robin", "owl", "penguin", "falcon",
    "swan", "duck", "goose", "parrot", "raven", "pigeon",
    "heron", "peacock", "flamingo",
)
MAMMAL_CONFIRMATION = (
    "deer", "monkey", "panda", "camel", "dolphin",
    "kangaroo", "leopard", "rhinoceros", "buffalo", "seal",
    "whale", "bat", "moose", "badger", "weasel",
)
BIRD_CONFIRMATION = (
    "albatross", "canary", "crane", "finch", "hawk",
    "ibis", "jay", "kingfisher", "lark", "magpie",
    "oriole", "pelican", "quail", "tern", "vulture",
)

RARE_DISCOVERY_KEYS = (
    "gluttony", "wealth", "steles", "fire", "ambition",
    "aspiration", "sunrise", "justice", "auspiciousness",
    "knowledge", "persistence", "pursuit", "moon", "repair", "rain",
)
RARE_CONFIRMATION_KEYS = (
    "chaos", "wrongdoing", "stubbornness", "tails", "east",
    "south", "north", "west", "fire", "renewal", "sun",
    "moon", "cosmology", "square", "floods",
)

CAUSAL_DISCOVERY = (
    ("ice", "The road froze overnight", "the bus moved slowly"),
    ("rain", "Heavy rain flooded the field", "the match was canceled"),
    ("power", "The power failed", "the room became dark"),
    ("heat", "The metal was heated", "it expanded"),
    ("alarm", "The alarm rang", "everyone left the building"),
    ("traffic", "Traffic was blocked", "the delivery arrived late"),
    ("study", "Mira studied carefully", "she passed the exam"),
    ("wind", "Strong wind damaged the line", "the signal stopped"),
    ("drought", "The drought continued", "the reservoir shrank"),
    ("cold", "The temperature fell below zero", "the lake froze"),
    ("fuel", "The car ran out of fuel", "the engine stopped"),
    ("practice", "Noah practiced daily", "his timing improved"),
    ("cloud", "Dark clouds gathered", "the street became dim"),
    ("rust", "The tool stayed wet", "rust formed"),
    ("delay", "The train was delayed", "the meeting started late"),
)
CAUSAL_CONFIRMATION = (
    ("snow", "Snow covered the runway", "the flight was postponed"),
    ("virus", "A virus infected the server", "the service crashed"),
    ("light", "Sunlight reached the panel", "the battery charged"),
    ("leak", "A pipe began leaking", "the floor became wet"),
    ("effort", "Lena revised every draft", "the report improved"),
    ("storm", "A storm damaged the bridge", "the road was closed"),
    ("noise", "The noise grew louder", "the speaker paused"),
    ("medicine", "The patient took the medicine", "the fever declined"),
    ("erosion", "Waves struck the cliff", "the rock eroded"),
    ("signal", "The signal turned red", "the driver stopped"),
    ("shortage", "Supplies became scarce", "prices increased"),
    ("exercise", "Iris exercised regularly", "her endurance increased"),
    ("fog", "Dense fog covered the harbor", "ships moved slowly"),
    ("battery", "The battery was empty", "the device shut down"),
    ("inspection", "The inspector found a crack", "the machine was halted"),
)


def item_number(item_id: str) -> int:
    match = re.search(r"(\d+)$", item_id)
    if match:
        return int(match.group(1))
    return sum(
        (index + 1) * ord(char)
        for index, char in enumerate(item_id)
    )


def mark(
    text: str,
    value: str,
    *,
    occurrence: str = "first",
) -> tuple[int, int, str]:
    start = text.find(value) if occurrence == "first" else text.rfind(value)
    if start < 0:
        raise RuntimeError(f"missing marked value {value!r}")
    return start, start + len(value), value


def controlled_factors(state: str) -> tuple[int, int, int, int]:
    match = re.fullmatch(r"t([01])_o([01])_a([01])_l([01])", state)
    if not match:
        raise ValueError(f"invalid controlled state: {state}")
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def natural_factors(state: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"t([01])_a([01])_l([01])", state)
    if not match:
        raise ValueError(f"invalid natural state: {state}")
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def cyclic_pair(values: tuple[Any, ...], index: int) -> tuple[Any, Any]:
    return values[index % len(values)], values[(index + 1) % len(values)]


def height_worlds(item: tuple[str, ...], surface: int) -> list[dict[str, str]]:
    _, high, middle, low = item
    if surface == 0:
        evidence0 = (
            f"{high} is taller than {middle}; "
            f"{middle} is taller than {low}"
        )
        evidence1 = (
            f"{low} is taller than {middle}; "
            f"{middle} is taller than {high}"
        )
        request = "The tallest person is"
    else:
        evidence0 = (
            f"{low} is shorter than {middle}; "
            f"{middle} is shorter than {high}"
        )
        evidence1 = (
            f"{high} is shorter than {middle}; "
            f"{middle} is shorter than {low}"
        )
        request = "The person with the greatest height is"
    return [
        {
            "evidence": evidence0,
            "anchor": high,
            "tail": low,
            "request": request,
            "answer": high,
        },
        {
            "evidence": evidence1,
            "anchor": low,
            "tail": high,
            "request": request,
            "answer": low,
        },
    ]


def contrast_worlds(item: tuple[str, ...], surface: int) -> list[dict[str, str]]:
    _, premise, aligned, opposed = item
    if surface == 0:
        evidence0 = f"First clause: {premise}. Second clause: {opposed}"
        evidence1 = f"First clause: {premise}. Second clause: {aligned}"
        request = "The natural connector between the clauses is"
    else:
        evidence0 = f"{premise}. The continuation is: {opposed}"
        evidence1 = f"{premise}. The continuation is: {aligned}"
        request = "The conjunction that best links the two parts is"
    return [
        {
            "evidence": evidence0,
            "anchor": premise,
            "tail": opposed,
            "request": request,
            "answer": "but",
        },
        {
            "evidence": evidence1,
            "anchor": premise,
            "tail": aligned,
            "request": request,
            "answer": "and",
        },
    ]


def punctuation_worlds(
    item: tuple[str, ...],
    surface: int,
) -> list[dict[str, str]]:
    _, statement, question = item
    request = (
        "The final punctuation mark should be"
        if surface == 0
        else "The mark that naturally closes this text is"
    )
    worlds = []
    for carrier, answer in ((question, "?"), (statement, ".")):
        words = carrier.split()
        evidence = (
            f"Unpunctuated text: {carrier}"
            if surface == 0
            else f"Text awaiting its final mark: {carrier}"
        )
        worlds.append({
            "evidence": evidence,
            "anchor": words[0],
            "tail": words[-1],
            "request": request,
            "answer": answer,
        })
    return worlds


def taxonomy_fruit_worlds(
    item: tuple[str, ...],
    surface: int,
    *,
    split: str,
) -> list[dict[str, str]]:
    item_id, fruit, _ = item
    vegetables = (
        VEGETABLE_DISCOVERY
        if split == "discovery"
        else VEGETABLE_CONFIRMATION
    )
    vegetable = vegetables[item_number(item_id) % len(vegetables)]
    request = (
        "The biological category is"
        if surface == 0
        else "The correct food category is"
    )
    worlds = []
    for entity, answer in ((fruit, "fruit"), (vegetable, "vegetable")):
        evidence = (
            f"Biological item {entity} is presented for classification"
            if surface == 0
            else f"Classify the food item named {entity} by type"
        )
        worlds.append({
            "evidence": evidence,
            "anchor": entity,
            "tail": "classification" if surface == 0 else "type",
            "request": request,
            "answer": answer,
        })
    return worlds


def taxonomy_animal_worlds(
    item: tuple[str, ...],
    surface: int,
    *,
    split: str,
) -> list[dict[str, str]]:
    index = item_number(item[0]) % 15
    mammals = (
        MAMMAL_DISCOVERY
        if split == "discovery"
        else MAMMAL_CONFIRMATION
    )
    birds = (
        BIRD_DISCOVERY
        if split == "discovery"
        else BIRD_CONFIRMATION
    )
    request = (
        "The zoological class is"
        if surface == 0
        else "The animal group is"
    )
    worlds = []
    for entity, answer in (
        (mammals[index], "mammal"),
        (birds[index], "bird"),
    ):
        evidence = (
            f"Animal {entity} is presented for zoological classification"
            if surface == 0
            else f"Identify the broad animal group of {entity}"
        )
        worlds.append({
            "evidence": evidence,
            "anchor": entity,
            "tail": "classification" if surface == 0 else "group",
            "request": request,
            "answer": answer,
        })
    return worlds


def color_worlds(
    item: tuple[str, ...],
    surface: int,
    *,
    split: str,
) -> list[dict[str, str]]:
    values = (
        source1078.COLOR_DISCOVERY
        if split == "discovery"
        else source1078.COLOR_CONFIRMATION
    )
    index = item_number(item[0]) % len(values)
    left, right = cyclic_pair(values, index)
    request = (
        "The usual color is"
        if surface == 0
        else "The color normally associated with the object is"
    )
    worlds = []
    for _, entity, color, _ in (left, right):
        evidence = (
            f"Object {entity} is considered under ordinary lighting"
            if surface == 0
            else f"Ordinary color knowledge is requested for {entity}"
        )
        worlds.append({
            "evidence": evidence,
            "anchor": entity,
            "tail": "lighting" if surface == 0 else "knowledge",
            "request": request,
            "answer": color,
        })
    return worlds


def rare_worlds(
    item: tuple[str, ...],
    surface: int,
    *,
    split: str,
) -> list[dict[str, str]]:
    values = (
        source.RARE_DISCOVERY
        if split == "discovery"
        else source.RARE_CONFIRMATION
    )
    keys = (
        RARE_DISCOVERY_KEYS
        if split == "discovery"
        else RARE_CONFIRMATION_KEYS
    )
    index = item_number(item[0]) % len(values)
    pair = ((values[index], keys[index]), (
        values[(index + 1) % len(values)],
        keys[(index + 1) % len(values)],
    ))
    request = (
        "The best-known one-word association is"
        if surface == 0
        else "The traditional one-word association is"
    )
    worlds = []
    for rare_item, answer in pair:
        _, term, _, _, _ = rare_item
        evidence = (
            f"Rare name {term} appears in Chinese mythology"
            if surface == 0
            else f"Chinese tradition contains the uncommon name {term}"
        )
        worlds.append({
            "evidence": evidence,
            "anchor": term,
            "tail": "mythology" if surface == 0 else "tradition",
            "request": request,
            "answer": answer,
        })
    return worlds


def translation_worlds(
    item: tuple[str, ...],
    surface: int,
) -> list[dict[str, str]]:
    _, english0, french0, english1, french1 = item
    request = (
        "The English translation is"
        if surface == 0
        else "Rendered naturally in English, the word is"
    )
    worlds = []
    for english, french in ((english0, french0), (english1, french1)):
        evidence = (
            f"French source word: {french}"
            if surface == 0
            else f"The source-language entry {french} is French"
        )
        worlds.append({
            "evidence": evidence,
            "anchor": french,
            "tail": "French",
            "request": request,
            "answer": english,
        })
    return worlds


def causal_worlds(
    item: tuple[str, ...],
    surface: int,
) -> list[dict[str, str]]:
    _, cause, effect = item
    if surface == 0:
        evidence0 = f"Cause: {cause}. Result: {effect}"
        evidence1 = f"Result: {effect}. Cause: {cause}"
        request0 = "The connector introducing the result is"
        request1 = "The connector introducing the cause is"
    else:
        evidence0 = f"{cause}; as a result, {effect}"
        evidence1 = f"{effect}; this happened due to {cause}"
        request0 = "A natural result connector is"
        request1 = "A natural cause connector is"
    return [
        {
            "evidence": evidence0,
            "anchor": cause,
            "tail": effect,
            "request": request0,
            "answer": "therefore",
        },
        {
            "evidence": evidence1,
            "anchor": effect,
            "tail": cause,
            "request": request1,
            "answer": "because",
        },
    ]


def worlds_for(
    family: str,
    item: tuple[str, ...],
    surface: int,
    split: str,
) -> list[dict[str, str]]:
    if family == "height_relation":
        return height_worlds(item, surface)
    if family == "contrast_conjunction":
        return contrast_worlds(item, surface)
    if family == "punctuation_rule":
        return punctuation_worlds(item, surface)
    if family == "taxonomy_fruit":
        return taxonomy_fruit_worlds(
            item, surface, split=split
        )
    if family == "taxonomy_animal":
        return taxonomy_animal_worlds(
            item, surface, split=split
        )
    if family == "color_property":
        return color_worlds(item, surface, split=split)
    if family == "rare_semantics":
        return rare_worlds(item, surface, split=split)
    if family == "translation":
        return translation_worlds(item, surface)
    if family == "causal_connector":
        return causal_worlds(item, surface)
    raise KeyError(family)


ITEMS_BY_FAMILY_SPLIT = {
    "height_relation": {
        "discovery": source.HEIGHT_DISCOVERY[:ITEMS_PER_FAMILY_SPLIT],
        "confirmation": source.HEIGHT_CONFIRMATION[:ITEMS_PER_FAMILY_SPLIT],
    },
    "contrast_conjunction": {
        "discovery": source.CONTRAST_DISCOVERY[:ITEMS_PER_FAMILY_SPLIT],
        "confirmation": source.CONTRAST_CONFIRMATION[:ITEMS_PER_FAMILY_SPLIT],
    },
    "punctuation_rule": {
        "discovery": source.PUNCTUATION_DISCOVERY[:ITEMS_PER_FAMILY_SPLIT],
        "confirmation": source.PUNCTUATION_CONFIRMATION[:ITEMS_PER_FAMILY_SPLIT],
    },
    "taxonomy_fruit": {
        "discovery": source.TAXONOMY_DISCOVERY[:ITEMS_PER_FAMILY_SPLIT],
        "confirmation": source.TAXONOMY_CONFIRMATION[:ITEMS_PER_FAMILY_SPLIT],
    },
    "taxonomy_animal": {
        "discovery": source.TAXONOMY_DISCOVERY[:ITEMS_PER_FAMILY_SPLIT],
        "confirmation": source.TAXONOMY_CONFIRMATION[:ITEMS_PER_FAMILY_SPLIT],
    },
    "color_property": {
        "discovery": source1078.COLOR_DISCOVERY[:ITEMS_PER_FAMILY_SPLIT],
        "confirmation": source1078.COLOR_CONFIRMATION[:ITEMS_PER_FAMILY_SPLIT],
    },
    "rare_semantics": {
        "discovery": source.RARE_DISCOVERY[:ITEMS_PER_FAMILY_SPLIT],
        "confirmation": source.RARE_CONFIRMATION[:ITEMS_PER_FAMILY_SPLIT],
    },
    "translation": {
        "discovery": source.TRANSLATION_DISCOVERY[:ITEMS_PER_FAMILY_SPLIT],
        "confirmation": source.TRANSLATION_CONFIRMATION[:ITEMS_PER_FAMILY_SPLIT],
    },
    "causal_connector": {
        "discovery": CAUSAL_DISCOVERY[:ITEMS_PER_FAMILY_SPLIT],
        "confirmation": CAUSAL_CONFIRMATION[:ITEMS_PER_FAMILY_SPLIT],
    },
}


def build_case(
    tokenizer,
    model_name: str,
    family: str,
    split: str,
    item: tuple[str, ...],
    panel: str,
    state: str,
    case_index: int,
) -> dict[str, Any]:
    if panel == "controlled":
        template_local, operation, answer, surface = controlled_factors(
            state
        )
    else:
        template_local, answer, surface = natural_factors(state)
        operation = 1
    template_index = TEMPLATES_BY_SPLIT[split][template_local]
    worlds = worlds_for(family, item, surface, split)
    world = worlds[answer]
    answers = [str(value["answer"]) for value in worlds]
    if answers[0].casefold() == answers[1].casefold():
        raise RuntimeError(f"answer collision {family}/{item[0]}")

    reverse_options = item_number(str(item[0])) % 2 == 0
    option_answers = list(reversed(answers)) if reverse_options else answers
    option_index = option_answers.index(answers[answer])
    options = f"{option_answers[0]} | {option_answers[1]}"
    ordinal = "first" if option_index == 0 else "second"
    ordinal_number = "one" if option_index == 0 else "two"

    if panel == "controlled":
        mode = "semantic" if operation == 1 else "index"
        raw_prompt = CONTROLLED_SHELLS[template_index].format(
            evidence=world["evidence"],
            options=options,
            request=world["request"],
            ordinal=ordinal,
            ordinal_number=ordinal_number,
            mode=mode,
        )
        raw_spans = {
            "content_anchor": mark(
                raw_prompt, world["anchor"], occurrence="first"
            ),
            "context_or_options": mark(raw_prompt, options),
            "semantic_request": mark(raw_prompt, world["request"]),
            "active_mode": mark(
                raw_prompt, mode, occurrence="last"
            ),
        }
    else:
        mode = "natural"
        raw_prompt = NATURAL_SHELLS[template_index].format(
            evidence=world["evidence"],
            request=world["request"],
        )
        raw_spans = {
            "content_anchor": mark(
                raw_prompt, world["anchor"], occurrence="first"
            ),
            "context_or_options": mark(
                raw_prompt, world["tail"], occurrence="last"
            ),
            "semantic_request": mark(raw_prompt, world["request"]),
            "active_mode": mark(
                raw_prompt,
                "natural completion",
                occurrence="last",
            ),
        }

    rendered = behavior.render_native(
        tokenizer,
        model_name,
        raw_prompt,
        with_system=False,
    )
    rendered += ASSISTANT_PREFILL
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_spans = offset_token_spans(
        tokenizer,
        rendered,
        raw_prompt,
        raw_spans,
    )
    role_spans["answer_boundary"] = (
        len(input_ids) - 1,
        len(input_ids) - 1,
    )
    prefix = " "
    candidate_token_ids = {
        f"a{branch}": behavior.continuation_ids(
            tokenizer,
            rendered,
            prefix,
            label,
        )
        for branch, label in enumerate(answers)
    }
    candidate_first_token_ids = {
        key: [int(values[0])]
        for key, values in candidate_token_ids.items()
    }
    return {
        "schema_version": "phase1079_pattern_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "semantic_case_index": case_index,
        "record_id": (
            f"{model_name}.{family}.{split}.{item[0]}."
            f"{panel}.{state}"
        ),
        "unit_id": f"{family}.{split}.{item[0]}",
        "family": family,
        "split": split,
        "item_id": str(item[0]),
        "panel": panel,
        "state": state,
        "template_local_branch": template_local,
        "template_index": template_index,
        "operation_branch": operation,
        "answer_branch": answer,
        "surface_branch": surface,
        "mode": mode,
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
        "answer_labels": answers,
        "option_answers": option_answers if panel == "controlled" else None,
        "target_option_index": option_index if panel == "controlled" else None,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": candidate_first_token_ids,
        "expected_class": f"a{answer}",
        "acceptable_labels": [answers[answer]],
        "continuation_prefix": prefix,
        "target_answer": answers[answer],
        "distractor_answer": answers[1 - answer],
        "shared_output_matched_across_operation": panel == "controlled",
        "candidate_absent": panel == "natural",
    }


def audit_model(
    model_name: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_unit_panel: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    counts = Counter(
        (row["family"], row["split"], row["panel"])
        for row in cases
    )
    roles_valid = True
    roles_distinct = True
    candidates_disjoint = True
    expected_correct = True
    operation_pair_prefix_equal = True
    operation_pair_answer_equal = True
    natural_candidate_list_absent = True
    for row in cases:
        by_unit_panel[(row["unit_id"], row["panel"])].append(row)
        width = len(row["input_ids"])
        positions = []
        for role in CAPTURE_ROLES:
            start, end = row["role_spans"][role]
            roles_valid &= 0 <= start <= end < width
            positions.append(int(row["role_positions"][role]))
        roles_distinct &= len(set(positions)) == len(positions)
        left = set(row["candidate_first_token_ids"]["a0"])
        right = set(row["candidate_first_token_ids"]["a1"])
        candidates_disjoint &= bool(left) and bool(right) and left.isdisjoint(
            right
        )
        expected_correct &= row["expected_class"] == (
            f"a{row['answer_branch']}"
        )
        if row["panel"] == "natural":
            prompt_casefold = row["raw_prompt"].casefold()
            natural_candidate_list_absent &= (
                "options:" not in prompt_casefold
                and "choices:" not in prompt_casefold
                and "candidate completions:" not in prompt_casefold
                and "available completions:" not in prompt_casefold
                and "index request:" not in prompt_casefold
                and "position request:" not in prompt_casefold
                and "copy request:" not in prompt_casefold
                and "slot request:" not in prompt_casefold
            )

    for (unit_id, panel), rows in by_unit_panel.items():
        if panel != "controlled":
            continue
        lookup = {row["state"]: row for row in rows}
        for template in (0, 1):
            for answer in (0, 1):
                for surface in (0, 1):
                    left = lookup[
                        f"t{template}_o0_a{answer}_l{surface}"
                    ]
                    right = lookup[
                        f"t{template}_o1_a{answer}_l{surface}"
                    ]
                    operation_pair_answer_equal &= (
                        left["target_answer"] == right["target_answer"]
                    )
                    left_mode_start = left["role_spans"]["active_mode"][0]
                    right_mode_start = right["role_spans"]["active_mode"][0]
                    operation_pair_prefix_equal &= (
                        left["input_ids"][:left_mode_start]
                        == right["input_ids"][:right_mode_start]
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
    checks = {
        "case_count": len(cases)
        == len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT
        * (len(CONTROLLED_STATES) + len(NATURAL_STATES)),
        "unit_count": len({
            row["unit_id"] for row in cases
        }) == len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT,
        "panel_case_counts": all(
            counts[(family, split, panel)]
            == ITEMS_PER_FAMILY_SPLIT * len(STATES_BY_PANEL[panel])
            for family in FAMILIES
            for split in SPLITS
            for panel in PANELS
        ),
        "complete_panel_units": all(
            {row["state"] for row in rows}
            == set(STATES_BY_PANEL[panel])
            for (_, panel), rows in by_unit_panel.items()
        ),
        "role_spans_valid": roles_valid,
        "role_end_positions_distinct": roles_distinct,
        "candidate_first_tokens_disjoint": candidates_disjoint,
        "expected_class_matches_answer": expected_correct,
        "operation_pair_prefix_equal_before_mode": (
            operation_pair_prefix_equal
        ),
        "operation_pair_target_answer_equal": operation_pair_answer_equal,
        "natural_candidate_list_absent": natural_candidate_list_absent,
        "independent_item_splits": all(
            item_ids[(family, "discovery")].isdisjoint(
                item_ids[(family, "confirmation")]
            )
            for family in FAMILIES
        ),
    }
    return {
        "schema_version": "phase1079_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len({row["unit_id"] for row in cases}),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def build_protocol() -> dict[str, Any]:
    if not SOURCE_PHASE1078.exists():
        raise RuntimeError("missing formal Phase1078 final summary")
    source_summary = read_json(SOURCE_PHASE1078)
    model_audits = {}
    model_case_digests = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        cases = []
        case_index = 0
        for family in FAMILIES:
            for split in SPLITS:
                items = ITEMS_BY_FAMILY_SPLIT[family][split]
                if len(items) != ITEMS_PER_FAMILY_SPLIT:
                    raise RuntimeError(
                        f"{family}/{split} has {len(items)} items"
                    )
                for item in items:
                    for panel in PANELS:
                        for state in STATES_BY_PANEL[panel]:
                            cases.append(build_case(
                                tokenizer,
                                model_name,
                                family,
                                split,
                                item,
                                panel,
                                state,
                                case_index,
                            ))
                            case_index += 1
        audit = audit_model(model_name, cases)
        audit["case_digest"] = digest(cases)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"protocol audit failed for {model_name}: {audit}"
            )
        write_jsonl(
            OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl",
            cases,
        )
        write_json(
            OUT_ROOT / "protocol" / f"audit.{model_name}.json",
            audit,
        )
        model_audits[model_name] = audit
        model_case_digests[model_name] = audit["case_digest"]

    unit_count = (
        len(FAMILIES) * len(SPLITS) * ITEMS_PER_FAMILY_SPLIT
    )
    case_count = unit_count * (
        len(CONTROLLED_STATES) + len(NATURAL_STATES)
    )
    payload = {
        "schema_version": "phase1079_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "families": list(FAMILIES),
        "base_families": list(BASE_FAMILIES),
        "heldout_family": HELDOUT_FAMILY,
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "controlled_states": list(CONTROLLED_STATES),
        "natural_states": list(NATURAL_STATES),
        "factor_definition": {
            "T": "shared shell wording with disjoint split templates",
            "O": (
                "active semantic versus index request; both requests have "
                "the same expected answer"
            ),
            "A": "same operation with two different answer identities",
            "L": "meaning-preserving content surface realization",
            "P": (
                "candidate-present controlled panel versus explicit-"
                "candidate-list-free natural panel"
            ),
        },
        "capture_roles": list(CAPTURE_ROLES),
        "pre_mode_roles": list(PRE_MODE_ROLES),
        "conditionings": list(CONDITIONINGS),
        "assistant_prefill": ASSISTANT_PREFILL,
        "case_count_per_model": case_count,
        "unit_count_per_model": unit_count,
        "model_case_digests": model_case_digests,
        "natural_generation_cases_per_family_split": (
            NATURAL_GENERATION_CASES_PER_FAMILY_SPLIT
        ),
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "evidence_thresholds": dict(EVIDENCE_THRESHOLDS),
        "prospective_predictions": dict(PROSPECTIVE_PREDICTIONS),
        "source_phase1078_protocol_digest": source_summary[
            "protocol_digest"
        ],
        "source_phase1078_summary_digest": source_summary[
            "summary_digest"
        ],
        "primary_population": (
            "All preregistered finite states. Behavior never deletes a "
            "sample from the descriptive physical map."
        ),
        "secondary_population": (
            "Units with at least 75% correct candidate comparisons across "
            "controlled semantic and candidate-list-free natural states."
        ),
        "evidence_levels": {
            "L0": "finite output-orthogonal field mapped",
            "L1": (
                "controlled operation topology retrieves across independent "
                "splits in at least two models"
            ),
            "L2": (
                "family-centered operation topology repeats across at least "
                "two directed model comparisons"
            ),
            "L3": (
                "candidate-list-free natural topology matches controlled "
                "semantic topology in at least two models"
            ),
            "L4": "L3 plus behavior annotation in at least two models",
            "L5": "causal mechanism support; forbidden in Phase1079",
        },
        "measurement_order": [
            "freeze families, independent items, factor states, roles, predictions, and automatic gate",
            "audit exact answer matching across operation and explicit candidate-list removal in the natural panel",
            "capture all finite residual, Attention-output, and MLP-output states",
            "measure operation, answer, surface, shell, and interaction fields",
            "verify causal-order zero response before the active-mode token",
            "compare independent split and cross-model normalized-depth topologies",
            "test all assignments against all 9! family-label permutations",
            "test candidate-list-free natural versus controlled semantic topology transfer",
            "evaluate the frozen causal_connector prediction",
            "stop before component, neuron, transport, or closure claims",
        ],
        "interpretation_limits": [
            "The controlled panel is an artificial routing protocol with visible candidates.",
            "An operation differential measures semantic-versus-index request selection, not a complete natural language operation.",
            "A natural answer differential still changes source content and expected output identity.",
            "Repeated normalized-depth topology is functional similarity, not coordinate homology.",
            "A family signature can still contain domain, syntax, length, and tokenizer statistics.",
            "Rare-word natural behavior tests model knowledge; controlled behavior can use visible options.",
            "No result establishes minimal coding, optimality, brain homology, or a complete pattern ontology.",
            "No Phase1079 result can establish causal necessity or sufficiency.",
        ],
        "automatic_next": {
            "continue_only_if": (
                "P1, P2, P4, P5, and P6 pass; at least five base families "
                "reach L3; integrity audit passes; and no protocol leak is "
                "found."
            ),
            "next_task_if_passed": (
                "Pre-register a candidate-list-free component-level predictive "
                "map for the strongest output-independent family without "
                "using response peaks as causal proof."
            ),
            "stop_if_failed": (
                "Do not select heads or neurons. Redesign the operation "
                "control or natural behavior protocol first."
            ),
        },
        "model_audits": model_audits,
    }
    payload["protocol_digest"] = digest(payload)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", payload)
    global_audit = {
        "schema_version": "phase1079_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": payload["protocol_digest"],
        "model_audits": model_audits,
        "checks": {
            "all_model_audits_passed": all(
                row["all_checks_passed"]
                for row in model_audits.values()
            ),
            "model_order_frozen": tuple(payload["models"]) == MODELS,
            "precision_fp16": payload["precision"] == "fp16",
            "quantization_none": payload["quantization"] == "none",
            "heldout_family_frozen": (
                payload["heldout_family"] == HELDOUT_FAMILY
            ),
            "predictions_frozen": set(
                payload["prospective_predictions"]
            ) == set(PROSPECTIVE_PREDICTIONS),
            "automatic_gate_frozen": bool(payload["automatic_next"]),
        },
    }
    global_audit["all_checks_passed"] = all(
        global_audit["checks"].values()
    )
    global_audit["audit_digest"] = digest(global_audit)
    write_json(OUT_ROOT / "protocol" / "audit.json", global_audit)
    return payload


def main() -> None:
    payload = build_protocol()
    print({
        "phase": PHASE,
        "status": "protocol_frozen",
        "case_count_per_model": payload["case_count_per_model"],
        "unit_count_per_model": payload["unit_count_per_model"],
        "protocol_digest": payload["protocol_digest"],
    })


if __name__ == "__main__":
    main()
