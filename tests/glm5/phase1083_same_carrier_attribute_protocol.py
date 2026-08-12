#!/usr/bin/env python3
"""Freeze Phase1083 same-carrier multi-attribute selection protocol.

Every attribute query reuses the exact same dossier prefix.  The requested
attribute appears only after that prefix, and a matched duplicate-candidate
panel preserves the selector, candidates, and output shell while removing the
semantic consequence of choosing A versus B.  This phase maps one focused
pattern family: conditional attribute-slot selection.  It does not claim to
represent translation, contrast, punctuation, or language as a whole.
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
import phase1079_output_orthogonal_pattern_protocol as source


PHASE = 1083
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
OPERATIONS = (
    "category",
    "color",
    "size",
    "material",
    "location",
    "alias",
    "schedule",
    "condition",
)
WORLDS = ("produce", "living", "artifacts", "rare_terms")
HELDOUT_WORLD = "rare_terms"
CELLS = tuple(
    f"{operation}__{world}"
    for operation in OPERATIONS
    for world in WORLDS
)
# Compatibility names consumed by the shared atlas engine.
FAMILIES = CELLS
BASE_FAMILIES = CELLS
EXPLORATORY_FAMILIES: tuple[str, ...] = ()
SPLITS = ("discovery", "confirmation")
PANELS = ("active", "duplicate")
OUTPUT_PAIRS = (("Yes", "No"), ("True", "False"))
CODE_WORDS = tuple(value for pair in OUTPUT_PAIRS for value in pair)
CODE_PAIRS = OUTPUT_PAIRS
STATES = tuple(
    f"t{template}_c{panel}_m{mapping}_q{query}_w{output_set}"
    for template in (0, 1)
    for panel in PANELS
    for mapping in (0, 1)
    for query in (0, 1)
    for output_set in (0, 1)
)
CAPTURE_ROLES = (
    "rule_end",
    "dossier_end",
    "field_selector",
    "candidate0_end",
    "candidate1_end",
    "selected_candidate",
    "answer_boundary",
)
PRIMARY_PROFILE_ROLES = ("selected_candidate", "answer_boundary")
PRE_QUERY_ROLES = (
    "rule_end", "dossier_end", "field_selector",
    "candidate0_end", "candidate1_end",
)
INTERMEDIATE_ROLES = ("selected_candidate",)
CONDITIONINGS = ("all_finite", "behavior_supported")
ITEMS_PER_CELL_SPLIT = 6
GENERATION_UNITS_PER_FAMILY_SPLIT = 4
GENERATION_STEPS = 12
ASSISTANT_PREFILL = "Answer:"
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1083_same_carrier_attribute_atlas"
)
SOURCE_PHASE1082 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1082_semantic_output_operation_world_atlas"
    / "analysis" / "final_summary.json"
)


EVIDENCE_THRESHOLDS = {
    "candidate_accuracy_for_operation_behavior": 0.80,
    "generation_target_before_distractor_accuracy": 0.75,
    "minimum_behavior_worlds_per_operation": 3,
    "minimum_behavior_operations": 6,
    "unit_behavior_support_fraction": 0.75,
    "permutation_p_max": 0.01,
    "minimum_operation_top1": 6,
    "minimum_repeated_models_or_pairs": 2,
    "minimum_heldout_world_folds": 3,
    "minimum_cross_world_content_advantage": 0.05,
    "minimum_cross_world_advantage_pairs": 8,
    "minimum_cross_model_content_advantage": 0.05,
    "minimum_cross_model_advantage_pairs": 4,
    "maximum_control_to_content_ratio": 1.0,
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_hidden_finite_fraction": 0.97,
    "pre_query_tolerance": 1e-8,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "All attribute-world cells, exact common-dossier prefixes, fixed "
        "output semantics, token, role, and query-order audits pass."
    ),
    "P2": (
        "At least six of eight attributes pass candidate and natural "
        "generation behavior in at least three worlds for at least two models."
    ),
    "P3": (
        "World-averaged downstream content profiles retrieve at least six of "
        "eight attributes across independent item splits in at least two models."
    ),
    "P4": (
        "For at least two models, at least three of four held-out lexical-world "
        "folds retrieve six of eight operations from the other three worlds."
    ),
    "P5": (
        "Content profiles under the two semantic output vocabularies retrieve at "
        "least six of eight operations in at least two models."
    ),
    "P6": (
        "Content identity exceeds carrier-only duplicate-route identity by 0.05 "
        "in at least eight of twelve directed cross-world pairs."
    ),
    "P7": (
        "The pooled median maximum output/shell control-to-content ratio is at "
        "most one in at least two models."
    ),
    "P8": (
        "Cross-model downstream content profiles retrieve six of eight "
        "attributes in at least two directed pairs, and content beats duplicate "
        "by 0.05 in at least four of six directed model pairs."
    ),
    "P9": (
        "Every model retains at least 95% finite candidate observations, 97% "
        "finite hidden-role observations, exact identity repeat, and zero "
        "query-only difference before the request selector."
    ),
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


SHELLS = {
    0: (
        "Decision rule: answer {support_code} if the selected candidate exactly "
        "matches the named field in the dossier; answer {contradict_code} "
        "otherwise.\n"
        "Dossier: {dossier}\n"
        "Field to check: {field}.\n"
        "Candidate A: {candidate0}.\n"
        "Candidate B: {candidate1}.\n"
        "Selected candidate: {selected}.\n"
        "Write only the answer word."
    ),
    1: (
        "Answer key: use {support_code} when the chosen value equals the dossier "
        "value for the requested field; use {contradict_code} when it does not.\n"
        "Profile: {dossier}\n"
        "Requested field: {field}.\n"
        "Option A: {candidate0}.\n"
        "Option B: {candidate1}.\n"
        "Chosen option: {selected}.\n"
        "Return only the answer word."
    ),
}

BANNED_FACTOR_WORDS = (
    "operation", "world", "family", "semantic", "latent", "mode",
    "pattern", "neuron", "mechanism",
)


WORLD_DATA = {
    "produce": {
        "class0": "fruit",
        "class1": "vegetable",
        "entities0": (
            "apple", "orange", "banana", "grape", "peach", "pear",
            "plum", "mango", "lemon", "cherry", "pineapple", "kiwi",
        ),
        "entities1": (
            "carrot", "celery", "potato", "onion", "lettuce", "radish",
            "broccoli", "spinach", "cabbage", "beet", "turnip", "okra",
        ),
    },
    "living": {
        "class0": "animal",
        "class1": "plant",
        "entities0": (
            "dog", "cat", "tiger", "whale", "eagle", "dolphin",
            "horse", "rabbit", "bear", "shark", "owl", "fox",
        ),
        "entities1": (
            "oak", "rose", "fern", "pine", "daisy", "moss",
            "tulip", "bamboo", "cactus", "maple", "lily", "grass",
        ),
    },
    "artifacts": {
        "class0": "tool",
        "class1": "container",
        "entities0": (
            "hammer", "wrench", "saw", "drill", "chisel", "pliers",
            "mallet", "screwdriver", "shovel", "rake", "ladder", "compass",
        ),
        "entities1": (
            "cup", "bottle", "box", "bucket", "basket", "jar",
            "canister", "flask", "barrel", "pouch", "crate", "vial",
        ),
    },
    "rare_terms": {
        "class0": "rare term",
        "class1": "common word",
        "entities0": (
            "饕餮", "petrichor", "susurrus", "syzygy", "ataraxia", "liminal",
            "oneiric", "quiddity", "apricity", "eldritch", "palimpsest", "numinous",
        ),
        "entities1": (
            "chair", "window", "garden", "bridge", "candle", "paper",
            "clock", "mirror", "cloud", "stone", "road", "table",
        ),
    },
}

ATTRIBUTE_PAIRS = {
    "produce": {
        "color": (("red", "green"), ("yellow", "purple"), ("orange", "pink")),
        "size": (("small", "large"), ("short", "tall"), ("narrow", "wide")),
        "material": (("wood", "metal"), ("glass", "plastic"), ("paper", "stone")),
        "location": (("orchard", "harbor"), ("market", "garden"), ("pantry", "station")),
        "alias": (("mepo", "zori"), ("lanu", "fena"), ("savi", "galo")),
        "schedule": (("dawn", "dusk"), ("noon", "midnight"), ("morning", "evening")),
        "condition": (("fresh", "stale"), ("ripe", "unripe"), ("whole", "damaged")),
    },
    "living": {
        "color": (("brown", "white"), ("black", "golden"), ("gray", "beige")),
        "size": (("tiny", "huge"), ("light", "heavy"), ("thin", "thick")),
        "material": (("wool", "leather"), ("bark", "silk"), ("feather", "clay")),
        "location": (("forest", "desert"), ("meadow", "cavern"), ("river", "mountain")),
        "alias": (("humi", "jaro"), ("wesi", "xanu"), ("yelo", "qira")),
        "schedule": (("spring", "autumn"), ("summer", "winter"), ("weekday", "weekend")),
        "condition": (("awake", "asleep"), ("healthy", "injured"), ("calm", "agitated")),
    },
    "artifacts": {
        "color": (("blue", "cyan"), ("magenta", "teal"), ("navy", "maroon")),
        "size": (("compact", "oversized"), ("low", "high"), ("shallow", "deep")),
        "material": (("steel", "rubber"), ("ceramic", "fabric"), ("copper", "concrete")),
        "location": (("office", "kitchen"), ("museum", "factory"), ("school", "warehouse")),
        "alias": (("lomi", "senu"), ("davi", "banu"), ("celo", "neli")),
        "schedule": (("early", "late"), ("today", "tomorrow"), ("Monday", "Friday")),
        "condition": (("open", "closed"), ("clean", "dirty"), ("active", "inactive")),
    },
    "rare_terms": {
        "color": (("crimson", "amber"), ("violet", "indigo"), ("scarlet", "turquoise")),
        "size": (("minute", "gigantic"), ("slender", "broad"), ("diminutive", "massive")),
        "material": (("ivory", "obsidian"), ("linen", "granite"), ("bronze", "velvet")),
        "location": (("archive", "theater"), ("temple", "laboratory"), ("library", "observatory")),
        "alias": (("riso", "veka"), ("tuma", "pavo"), ("kima", "doren")),
        "schedule": (("sunrise", "sunset"), ("beforehand", "afterward"), ("yesterday", "tonight")),
        "condition": (("stable", "unstable"), ("intact", "broken"), ("full", "empty")),
    },
}


def cell_id(operation: str, world: str) -> str:
    return f"{operation}__{world}"


def split_cell(cell: str) -> tuple[str, str]:
    operation, world = cell.split("__", 1)
    return operation, world


def dossier_item(world: str, index: int) -> dict[str, Any]:
    data = WORLD_DATA[world]
    even = index % 2 == 0
    entity = str(data["entities0"][index]) if even else str(data["entities1"][index])
    category = str(data["class0"] if even else data["class1"])
    category_other = str(data["class1"] if even else data["class0"])
    values: dict[str, str] = {"category": category}
    distractors: dict[str, str] = {"category": category_other}
    for attribute_index, attribute in enumerate(OPERATIONS[1:], start=1):
        pair = ATTRIBUTE_PAIRS[world][attribute][index % 3]
        side = (index + attribute_index) % 2
        values[attribute] = str(pair[side])
        distractors[attribute] = str(pair[1 - side])
    dossier = (
        f"entity '{entity}'; category {values['category']}; color {values['color']}; "
        f"size {values['size']}; material {values['material']}; "
        f"location {values['location']}; alias {values['alias']}; "
        f"schedule {values['schedule']}; condition {values['condition']}."
    )
    return {
        "base_item_id": f"{world}.{index:02d}",
        "entity": entity,
        "dossier": dossier,
        "values": values,
        "distractors": distractors,
    }


def build_items() -> dict[str, tuple[dict[str, Any], ...]]:
    output: dict[str, tuple[dict[str, Any], ...]] = {}
    for world in WORLDS:
        base_items = tuple(
            dossier_item(world, index)
            for index in range(ITEMS_PER_CELL_SPLIT * len(SPLITS))
        )
        for operation in OPERATIONS:
            cell = cell_id(operation, world)
            rows = []
            for index, base in enumerate(base_items):
                rows.append({
                    "item_id": f"{cell}.{index:02d}",
                    "base_item_id": base["base_item_id"],
                    "operation": operation,
                    "world": world,
                    "entity": base["entity"],
                    "dossier": base["dossier"],
                    "target_value": base["values"][operation],
                    "distractor_value": base["distractors"][operation],
                })
            output[cell] = tuple(rows)
    return output


ITEMS_BY_CELL = build_items()


def split_items(cell: str, split: str) -> tuple[dict[str, Any], ...]:
    start = 0 if split == "discovery" else ITEMS_PER_CELL_SPLIT
    return ITEMS_BY_CELL[cell][start:start + ITEMS_PER_CELL_SPLIT]


def state_factors(state: str) -> tuple[int, str, int, int, int]:
    match = re.fullmatch(
        r"t([01])_c(active|duplicate)_m([01])_q([01])_w([01])",
        state,
    )
    if not match:
        raise ValueError(f"invalid state: {state}")
    return (
        int(match.group(1)), str(match.group(2)),
        int(match.group(3)), int(match.group(4)), int(match.group(5)),
    )


def output_code_pair(output_set: int) -> tuple[str, str]:
    return OUTPUT_PAIRS[output_set]


def render_candidates(
    item: dict[str, Any], panel: str, mapping: int
) -> tuple[str, str, int, int]:
    target = str(item["target_value"])
    distractor = str(item["distractor_value"])
    if panel == "active":
        candidates = (
            (target, distractor)
            if mapping == 0 else (distractor, target)
        )
        semantic_answers = (mapping, 1 - mapping)
        return (
            candidates[0], candidates[1],
            semantic_answers[0], semantic_answers[1],
        )
    candidate = target if mapping == 0 else distractor
    return candidate, candidate, mapping, mapping


def encoded_width(tokenizer, text: str) -> int:
    return len(tokenizer.encode(" " + text, add_special_tokens=False))


def build_case(
    tokenizer,
    model_name: str,
    cell: str,
    split: str,
    item: dict[str, Any],
    item_local_index: int,
    state: str,
    case_index: int,
) -> dict[str, Any]:
    operation, world = split_cell(cell)
    template, panel, mapping, query, output_set = state_factors(state)
    semantic_codes = output_code_pair(output_set)
    candidate0, candidate1, semantic0, semantic1 = render_candidates(
        item, panel, mapping
    )
    semantic_answer = semantic0 if query == 0 else semantic1
    target_answer = semantic_codes[semantic_answer]
    selected = "A" if query == 0 else "B"
    raw_prompt = SHELLS[template].format(
        support_code=semantic_codes[0],
        contradict_code=semantic_codes[1],
        dossier=item["dossier"],
        field=operation,
        candidate0=candidate0,
        candidate1=candidate1,
        selected=selected,
    )
    rule = (
        f"Decision rule: answer {semantic_codes[0]} if the selected candidate exactly "
        f"matches the named field in the dossier; answer {semantic_codes[1]} otherwise."
        if template == 0 else
        f"Answer key: use {semantic_codes[0]} when the chosen value equals the dossier "
        f"value for the requested field; use {semantic_codes[1]} when it does not."
    )
    dossier_line = (
        f"Dossier: {item['dossier']}" if template == 0
        else f"Profile: {item['dossier']}"
    )
    field_line = (
        f"Field to check: {operation}." if template == 0
        else f"Requested field: {operation}."
    )
    candidate0_line = (
        f"Candidate A: {candidate0}." if template == 0
        else f"Option A: {candidate0}."
    )
    candidate1_line = (
        f"Candidate B: {candidate1}." if template == 0
        else f"Option B: {candidate1}."
    )
    raw_spans = {
        "rule_end": source.mark(raw_prompt, rule, occurrence="first"),
        "dossier_end": source.mark(raw_prompt, dossier_line, occurrence="first"),
        "field_selector": source.mark(raw_prompt, field_line, occurrence="first"),
        "candidate0_end": source.mark(
            raw_prompt, candidate0_line, occurrence="first"
        ),
        "candidate1_end": source.mark(
            raw_prompt, candidate1_line, occurrence="first"
        ),
        "selected_candidate": source.mark(
            raw_prompt, selected, occurrence="last"
        ),
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
    role_spans["answer_boundary"] = (len(input_ids) - 1, len(input_ids) - 1)
    prefix = " "
    candidate_token_ids = {
        f"a{index}": behavior.continuation_ids(
            tokenizer, rendered, prefix, answer
        )
        for index, answer in enumerate(semantic_codes)
    }
    return {
        "schema_version": "phase1083_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{model_name}.{cell}.{split}.{item['item_id']}.{state}",
        "unit_id": f"{cell}.{split}.{item['item_id']}",
        "family": cell,
        "cell": cell,
        "operation": operation,
        "world": world,
        "split": split,
        "item_id": item["item_id"],
        "base_item_id": item["base_item_id"],
        "item_local_index": item_local_index,
        "state": state,
        "template": template,
        "panel": panel,
        "mapping": mapping,
        "query": query,
        "label_swap": output_set,
        "code_swap": output_set,
        "output_set": output_set,
        "label_pair": list(semantic_codes),
        "semantic_codes": list(semantic_codes),
        "selected_label_text": selected,
        "selected_position": query,
        "entity": item["entity"],
        "dossier": item["dossier"],
        "field": operation,
        "target_value": item["target_value"],
        "distractor_value": item["distractor_value"],
        "candidate0": candidate0,
        "candidate1": candidate1,
        "record0": candidate0,
        "record1": candidate1,
        "control_type": "carrier_only_duplicate_candidate" if panel == "duplicate" else None,
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
        "answer_labels": list(semantic_codes),
        "semantic_answer_index": semantic_answer,
        "answer_index": semantic_answer,
        "target_answer": target_answer,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {
            key: [int(values[0])]
            for key, values in candidate_token_ids.items()
        },
        "expected_class": f"a{semantic_answer}",
        "continuation_prefix": prefix,
        "label_token_widths": [
            encoded_width(tokenizer, value) for value in semantic_codes
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
    candidate_first_tokens_disjoint = True
    candidate_width_one = True
    expected_correct = True
    no_banned_words = True
    active_candidates_distinct = True
    duplicate_candidates_equal = True
    target_distractor_distinct = True
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
            positions["rule_end"] < positions["dossier_end"]
            < positions["field_selector"] < positions["candidate0_end"]
            < positions["candidate1_end"] < positions["selected_candidate"]
            < positions["answer_boundary"]
        )
        first0 = set(row["candidate_first_token_ids"]["a0"])
        first1 = set(row["candidate_first_token_ids"]["a1"])
        candidate_first_tokens_disjoint &= (
            bool(first0) and bool(first1) and first0.isdisjoint(first1)
        )
        candidate_width_one &= row["label_token_widths"] == [1, 1]
        expected_correct &= row["expected_class"] == (
            f"a{row['semantic_answer_index']}"
        )
        prompt_folded = row["raw_prompt"].casefold()
        no_banned_words &= not any(
            re.search(rf"\b{re.escape(word)}\b", prompt_folded)
            for word in BANNED_FACTOR_WORDS
        )
        if row["panel"] == "active":
            active_candidates_distinct &= row["candidate0"] != row["candidate1"]
        else:
            duplicate_candidates_equal &= row["candidate0"] == row["candidate1"]
        target_distractor_distinct &= (
            row["target_value"] != row["distractor_value"]
        )

    active_answer_depends_on_query = True
    duplicate_answer_independent_of_query = True
    output_vocabulary_changes_words_not_semantics = True
    query_prefix_equal = True
    pre_query_roles_before_divergence = True
    for rows in by_unit.values():
        lookup = {row["state"]: row for row in rows}
        for template in (0, 1):
            for panel in PANELS:
                for mapping in (0, 1):
                    for output_set in (0, 1):
                        left = lookup[
                            f"t{template}_c{panel}_m{mapping}_q0_w{output_set}"
                        ]
                        right = lookup[
                            f"t{template}_c{panel}_m{mapping}_q1_w{output_set}"
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
                                left["semantic_answer_index"]
                                != right["semantic_answer_index"]
                            )
                        else:
                            duplicate_answer_independent_of_query &= (
                                left["semantic_answer_index"]
                                == right["semantic_answer_index"]
                            )
                    for query in (0, 1):
                        first = lookup[
                            f"t{template}_c{panel}_m{mapping}_q{query}_w0"
                        ]
                        second = lookup[
                            f"t{template}_c{panel}_m{mapping}_q{query}_w1"
                        ]
                        output_vocabulary_changes_words_not_semantics &= (
                            first["semantic_answer_index"]
                            == second["semantic_answer_index"]
                            and first["target_answer"] != second["target_answer"]
                        )

    code_counts = Counter(
        (row["operation"], row["split"], row["output_set"], code)
        for row in cases
        if row["template"] == 0 and row["panel"] == "active"
        and row["mapping"] == 0 and row["query"] == 0
        for code in row["label_pair"]
    )
    expected_code_units = len(WORLDS) * ITEMS_PER_CELL_SPLIT
    codes_balanced = all(
        code_counts[(operation, split, output_set, code)] == expected_code_units
        for operation in OPERATIONS
        for split in SPLITS
        for output_set, pair in enumerate(OUTPUT_PAIRS)
        for code in pair
    )
    counts = Counter(
        (row["operation"], row["world"], row["split"])
        for row in cases
    )
    item_ids = {
        (cell, split): {
            row["item_id"] for row in cases
            if row["cell"] == cell and row["split"] == split
        }
        for cell in CELLS for split in SPLITS
    }
    lexical_sets = [
        set(data["entities0"]) | set(data["entities1"])
        for data in WORLD_DATA.values()
    ]
    lexical_worlds_disjoint = all(
        lexical_sets[left].isdisjoint(lexical_sets[right])
        for left in range(len(lexical_sets))
        for right in range(left + 1, len(lexical_sets))
    )
    same_dossier_text = True
    same_dossier_token_prefix = True
    operation_selector_after_dossier = True
    by_base_state: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        key = (
            row["world"], row["split"], row["base_item_id"],
            row["template"], row["panel"], row["mapping"],
            row["query"], row["output_set"],
        )
        by_base_state[key].append(row)
    for rows in by_base_state.values():
        same_dossier_text &= (
            len(rows) == len(OPERATIONS)
            and len({row["dossier"] for row in rows}) == 1
        )
        prefix_rows = [
            row["input_ids"][:row["role_positions"]["dossier_end"] + 1]
            for row in rows
        ]
        same_dossier_token_prefix &= (
            len(prefix_rows) == len(OPERATIONS)
            and all(value == prefix_rows[0] for value in prefix_rows[1:])
        )
        operation_selector_after_dossier &= all(
            row["role_positions"]["dossier_end"]
            < row["role_positions"]["field_selector"]
            for row in rows
        )

    checks = {
        "case_count": len(cases)
        == len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT * len(STATES),
        "unit_count": len(by_unit)
        == len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT,
        "complete_units": all(len(rows) == len(STATES) for rows in by_unit.values()),
        "full_operation_world_cross": (
            len(counts) == len(OPERATIONS) * len(WORLDS) * len(SPLITS)
            and all(
                count == ITEMS_PER_CELL_SPLIT * len(STATES)
                for count in counts.values()
            )
        ),
        "entity_worlds_disjoint": lexical_worlds_disjoint,
        "same_dossier_text_across_attributes": same_dossier_text,
        "same_dossier_token_prefix_across_attributes": same_dossier_token_prefix,
        "attribute_selector_after_common_dossier": operation_selector_after_dossier,
        "role_spans_valid": roles_valid,
        "role_order_valid": role_order,
        "candidate_first_tokens_disjoint": candidate_first_tokens_disjoint,
        "candidate_codes_single_token": candidate_width_one,
        "expected_class_matches_semantic_answer": expected_correct,
        "factor_names_absent_from_prompts": no_banned_words,
        "output_words_balanced_per_operation_split": codes_balanced,
        "active_answer_depends_on_query": active_answer_depends_on_query,
        "duplicate_answer_independent_of_query": duplicate_answer_independent_of_query,
        "output_vocabulary_changes_words_not_semantics": (
            output_vocabulary_changes_words_not_semantics
        ),
        "query_prefix_equal_before_divergence": query_prefix_equal,
        "pre_query_roles_before_divergence": pre_query_roles_before_divergence,
        "active_candidates_distinct": active_candidates_distinct,
        "duplicate_candidates_equal": duplicate_candidates_equal,
        "target_distractor_distinct": target_distractor_distinct,
        "independent_item_splits": all(
            item_ids[(cell, "discovery")].isdisjoint(
                item_ids[(cell, "confirmation")]
            )
            for cell in CELLS
        ),
    }
    return {
        "schema_version": "phase1083_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "code_counts": {
            f"{operation}.{split}.w{output_set}.{code}": (
                code_counts[(operation, split, output_set, code)]
            )
            for operation in OPERATIONS for split in SPLITS
            for output_set, pair in enumerate(OUTPUT_PAIRS) for code in pair
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(cases),
    }


def build_model_cases(
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    case_index = 0
    for cell in CELLS:
        for split in SPLITS:
            for item_local_index, item in enumerate(split_items(cell, split)):
                for state in STATES:
                    cases.append(build_case(
                        tokenizer, model_name, cell, split, item,
                        item_local_index, state, case_index,
                    ))
                    case_index += 1
    return cases, audit_model(model_name, tokenizer, cases)


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

    source_summary = read_json(SOURCE_PHASE1082)
    prereg = {
        "schema_version": "phase1083_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "operations": list(OPERATIONS),
        "worlds": list(WORLDS),
        "heldout_world": HELDOUT_WORLD,
        "cells": list(CELLS),
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "output_words": list(CODE_WORDS),
        "output_pairs": [list(value) for value in OUTPUT_PAIRS],
        "states": list(STATES),
        "factor_definition": {
            "O": "eight requested attributes over one shared dossier",
            "X": "four disjoint entity worlds fully crossed with O",
            "T": "two shared surface shells",
            "C": "active target/distractor candidates or carrier-only duplicate control",
            "M": "target order or duplicated candidate truth identity",
            "Q": "selected physical candidate position",
            "W": "shared semantic output vocabulary: Yes/No or True/False",
        },
        "capture_roles": list(CAPTURE_ROLES),
        "primary_profile_roles": list(PRIMARY_PROFILE_ROLES),
        "pre_query_roles": list(PRE_QUERY_ROLES),
        "conditionings": list(CONDITIONINGS),
        "assistant_prefill": ASSISTANT_PREFILL,
        "items_per_cell_split": ITEMS_PER_CELL_SPLIT,
        "case_count_per_model": (
            len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT * len(STATES)
        ),
        "unit_count_per_model": (
            len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT
        ),
        "generation_units_per_family_split": GENERATION_UNITS_PER_FAMILY_SPLIT,
        "generation_steps": GENERATION_STEPS,
        "model_case_digests": model_case_digests,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_phase1082_protocol_digest": source_summary["protocol_digest"],
        "source_phase1082_summary_digest": source_summary["summary_digest"],
        "primary_population": (
            "All finite preregistered observations; behavior never deletes a "
            "cell from the descriptive map."
        ),
        "secondary_population": (
            "Units with at least 75% correct active-panel candidate comparisons."
        ),
        "behavior_gate_definition": (
            "For an attribute-world cell, active candidate accuracy must reach "
            "0.80 and the target answer must occur before its paired answer in at "
            "least 0.75 of 12-token generations. An operation passes a model "
            "only when at least three worlds pass both gates."
        ),
        "measurement_order": [
            "complete same-dossier attribute-world and output-token audits",
            "behavior-only pilot without hidden-state access",
            "freeze protocol digest and all predictions",
            "sequential Qwen3, GLM4, DS7B FP16 scans",
            "active, duplicate, and content difference-in-differences fields",
            "downstream-only attribute profiles",
            "independent item-split retrieval",
            "held-out lexical-world transfer",
            "semantic output-vocabulary transfer",
            "cross-world content-versus-duplicate advantage",
            "cross-model transfer and numerical integrity",
            "automatic non-causal gate",
        ],
        "evidence_levels": {
            "L0": "finite attribute-world cell mapped",
            "L1": "attribute retrieves across independent item splits",
            "L2": "L1 plus held-out lexical-world and output-vocabulary transfer",
            "L3": "L2 plus cross-world duplicate-route advantage",
            "L4": "L3 plus cross-model attribute retrieval and advantage",
            "L5": "L4 plus behavior support in at least two models",
            "L6": "causal support; forbidden in Phase1083",
        },
        "interpretation_limits": [
            "Shared output words are experimental handles, not latent symbols.",
            "The failed revision-1 arbitrary-code pilot is diagnostic only and is not pooled.",
            "All tasks are binary candidate matching tasks and share one generic shell.",
            "Attribute names and attribute-specific candidate words remain part of the route.",
            "Difference-in-differences assumes approximate additivity.",
            "Cross-world transfer is not cross-model conservation.",
            "Alias lookup is not natural-language translation.",
            "This phase does not test contrast, punctuation, syntax, or stored knowledge mechanisms.",
            "No result establishes a causal path, minimal code, optimality, or brain homology.",
        ],
        "automatic_next": {
            "continue_to_local_causal_only_if": (
                "P1-P9 all pass and at least six operations reach L4."
            ),
            "continue_global_atlas_if": (
                "P4 passes but P8 fails; preserve cross-world structure and "
                "design a cross-model functional-coordinate alignment phase."
            ),
            "stop_and_diagnose_if": (
                "P4 fails or controls dominate; do not select components or neurons."
            ),
        },
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    global_audit = {
        "schema_version": "phase1083_protocol_audit.v1",
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
            "full_cross_frozen": len(CELLS) == len(OPERATIONS) * len(WORLDS),
            "heldout_world_frozen": prereg["heldout_world"] == HELDOUT_WORLD,
            "predictions_frozen": set(prereg["prospective_predictions"])
            == set(PROSPECTIVE_PREDICTIONS),
        },
    }
    global_audit["all_checks_passed"] = all(
        global_audit["checks"].values()
    )
    global_audit["audit_digest"] = digest(global_audit)
    write_json(protocol_root / "audit.json", global_audit)
    if not global_audit["all_checks_passed"]:
        raise RuntimeError(f"global protocol audit failed: {global_audit}")
    print({
        "phase": PHASE,
        "status": "protocol_frozen",
        "case_count_per_model": prereg["case_count_per_model"],
        "unit_count_per_model": prereg["unit_count_per_model"],
        "protocol_digest": prereg["protocol_digest"],
    })


if __name__ == "__main__":
    main()
