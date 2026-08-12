#!/usr/bin/env python3
"""Freeze Phase1082 revision-2 operation-by-lexical-world protocol.

The protocol crosses eight language operations with four disjoint lexical
worlds. Every task uses one of two shared, semantically fixed output pairs so
that arbitrary symbol remapping cannot dominate the language operation.
Internal measurements are forbidden until this file has generated and audited
every model-specific case plus the preregistration.
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


PHASE = 1082
PROTOCOL_REVISION = 2
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
OPERATIONS = (
    "class_membership",
    "color_property",
    "size_relation",
    "glossary_translation",
    "sentiment_polarity",
    "tense_status",
    "contrast_coherence",
    "punctuation_status",
)
WORLDS = ("produce", "living", "artifacts", "rare_terms")
HELDOUT_WORLD = "rare_terms"
CELLS = tuple(
    f"{operation}__{world}"
    for operation in OPERATIONS
    for world in WORLDS
)
# Compatibility names consumed by the shared Phase1081 scan engine.
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
    "codebook_end",
    "record0_end",
    "record1_end",
    "selected_case",
    "request_end",
    "answer_boundary",
)
PRIMARY_PROFILE_ROLES = ("request_end", "answer_boundary")
PRE_QUERY_ROLES = ("codebook_end", "record0_end", "record1_end")
INTERMEDIATE_ROLES = ("request_end",)
CONDITIONINGS = ("all_finite", "behavior_supported")
ITEMS_PER_CELL_SPLIT = 6
GENERATION_UNITS_PER_FAMILY_SPLIT = 4
GENERATION_STEPS = 12
ASSISTANT_PREFILL = "Answer:"
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1082_semantic_output_operation_world_atlas"
)
SOURCE_PHASE1081 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1081_latin_route_atlas" / "analysis" / "final_summary.json"
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
        "All operation-world cells, fixed output semantics, output token, "
        "token, role, and query-causal-order audits pass for every model."
    ),
    "P2": (
        "At least six of eight operations pass candidate and natural "
        "generation behavior in at least three worlds for at least two models."
    ),
    "P3": (
        "World-averaged downstream content profiles retrieve at least six of "
        "eight operations across independent item splits in at least two models."
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
        "Content identity exceeds duplicate-route identity by at least 0.05 "
        "in at least eight of twelve directed cross-world pairs."
    ),
    "P7": (
        "The pooled median maximum output/shell control-to-content ratio is at "
        "most one in at least two models."
    ),
    "P8": (
        "Cross-model downstream content profiles retrieve six of eight "
        "operations in at least two directed pairs, and content beats duplicate "
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
        "Response key: write {support_code} when the selected claim is supported "
        "by its evidence; write {contradict_code} when it is contradicted.\n"
        "Case A: {record0}\n"
        "Case B: {record1}\n"
        "Request: Judge case {selected}.\n"
        "Write only the response word."
    ),
    1: (
        "Answer rule: use {support_code} if the chosen statement agrees with "
        "its evidence; use {contradict_code} if it disagrees.\n"
        "Item A: {record0}\n"
        "Item B: {record1}\n"
        "Request: Evaluate item {selected}.\n"
        "Output only the answer word."
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

ALIASES0 = (
    "mepo", "lanu", "savi", "doren", "kima", "pavo",
    "neli", "tuma", "riso", "veka", "banu", "celo",
)
ALIASES1 = (
    "zori", "fena", "galo", "humi", "jaro", "wesi",
    "xanu", "yelo", "qira", "lomi", "senu", "davi",
)


def cell_id(operation: str, world: str) -> str:
    return f"{operation}__{world}"


def split_cell(cell: str) -> tuple[str, str]:
    operation, world = cell.split("__", 1)
    return operation, world


def operation_records(
    operation: str,
    world: str,
    index: int,
) -> tuple[str, str]:
    data = WORLD_DATA[world]
    entity0 = str(data["entities0"][index])
    entity1 = str(data["entities1"][index])
    class0 = str(data["class0"])
    class1 = str(data["class1"])
    even = index % 2 == 0
    support_entity = entity0 if even else entity1
    other_entity = entity1 if even else entity0
    support_class = class0 if even else class1
    other_class = class1 if even else class0
    target_color = "red" if even else "blue"
    other_color = "blue" if even else "red"
    if operation == "class_membership":
        supported = (
            f"Evidence: '{support_entity}' is registered as a {support_class}. "
            f"Claim: '{support_entity}' is a {support_class}."
        )
        contradicted = (
            f"Evidence: '{other_entity}' is registered as a {other_class}. "
            f"Claim: '{other_entity}' is a {support_class}."
        )
    elif operation == "color_property":
        supported = (
            f"Evidence: the card for '{support_entity}' is {target_color}. "
            f"Claim: that card is {target_color}."
        )
        contradicted = (
            f"Evidence: the card for '{other_entity}' is {other_color}. "
            f"Claim: that card is {target_color}."
        )
    elif operation == "size_relation":
        supported = (
            f"Evidence: the display for '{support_entity}' is larger than the "
            f"display for '{other_entity}'. Claim: '{support_entity}' has the "
            "larger display."
        )
        contradicted = (
            f"Evidence: the display for '{other_entity}' is smaller than the "
            f"display for '{support_entity}'. Claim: '{other_entity}' has the "
            "larger display."
        )
    elif operation == "glossary_translation":
        alias0 = ALIASES0[index]
        alias1 = ALIASES1[index]
        support_alias = alias0 if even else alias1
        other_alias = alias1 if even else alias0
        supported = (
            f"Evidence: in a local glossary, '{support_alias}' translates to "
            f"'{support_entity}'. Claim: '{support_alias}' translates to "
            f"'{support_entity}'."
        )
        contradicted = (
            f"Evidence: in a local glossary, '{other_alias}' translates to "
            f"'{other_entity}'. Claim: '{other_alias}' translates to "
            f"'{support_entity}'."
        )
    elif operation == "sentiment_polarity":
        positive = "excellent" if even else "wonderful"
        negative = "awful" if even else "terrible"
        supported = (
            f"Evidence: a reviewer called the entry for '{support_entity}' "
            f"{positive}. Claim: the evaluation is positive."
        )
        contradicted = (
            f"Evidence: a reviewer called the entry for '{other_entity}' "
            f"{negative}. Claim: the evaluation is positive."
        )
    elif operation == "tense_status":
        supported = (
            f"Evidence: yesterday the catalog added '{support_entity}'. "
            "Claim: the event is in the past."
        )
        contradicted = (
            f"Evidence: tomorrow the catalog will add '{other_entity}'. "
            "Claim: the event is in the past."
        )
    elif operation == "contrast_coherence":
        supported = (
            f"Evidence: although the entry for '{support_entity}' was disputed, "
            "it remained in the catalog. Claim: the sentence expresses a contrast."
        )
        contradicted = (
            f"Evidence: because the entry for '{other_entity}' was confirmed, "
            "it remained in the catalog. Claim: the sentence expresses a contrast."
        )
    elif operation == "punctuation_status":
        supported = (
            f"Evidence: the text is 'Is {support_entity} listed'. "
            "Claim: the text requires a question mark."
        )
        contradicted = (
            f"Evidence: the text is '{other_entity} is listed'. "
            "Claim: the text requires a question mark."
        )
    else:
        raise KeyError(operation)
    return supported, contradicted


def build_items() -> dict[str, tuple[dict[str, Any], ...]]:
    output: dict[str, tuple[dict[str, Any], ...]] = {}
    for operation in OPERATIONS:
        for world in WORLDS:
            cell = cell_id(operation, world)
            rows = []
            for index in range(ITEMS_PER_CELL_SPLIT * len(SPLITS)):
                supported, contradicted = operation_records(
                    operation, world, index
                )
                rows.append({
                    "item_id": f"{cell}.{index:02d}",
                    "operation": operation,
                    "world": world,
                    "record_supported": supported,
                    "record_contradicted": contradicted,
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


def render_records(
    item: dict[str, Any], panel: str, mapping: int
) -> tuple[str, str, int, int]:
    supported = str(item["record_supported"])
    contradicted = str(item["record_contradicted"])
    if panel == "active":
        records = (
            (supported, contradicted)
            if mapping == 0 else (contradicted, supported)
        )
        semantic_answers = (mapping, 1 - mapping)
        return records[0], records[1], semantic_answers[0], semantic_answers[1]
    record = supported if mapping == 0 else contradicted
    return record, record, mapping, mapping


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
    record0, record1, semantic0, semantic1 = render_records(
        item, panel, mapping
    )
    semantic_answer = semantic0 if query == 0 else semantic1
    target_answer = semantic_codes[semantic_answer]
    selected = "A" if query == 0 else "B"
    raw_prompt = SHELLS[template].format(
        support_code=semantic_codes[0],
        contradict_code=semantic_codes[1],
        record0=record0,
        record1=record1,
        selected=selected,
    )
    codebook = (
        f"Response key: write {semantic_codes[0]} when the selected claim is "
        f"supported by its evidence; write {semantic_codes[1]} when it is contradicted."
        if template == 0 else
        f"Answer rule: use {semantic_codes[0]} if the chosen statement agrees with "
        f"its evidence; use {semantic_codes[1]} if it disagrees."
    )
    line0 = f"{'Case' if template == 0 else 'Item'} A: {record0}"
    line1 = f"{'Case' if template == 0 else 'Item'} B: {record1}"
    request = (
        f"Request: Judge case {selected}."
        if template == 0 else
        f"Request: Evaluate item {selected}."
    )
    raw_spans = {
        "codebook_end": source.mark(raw_prompt, codebook, occurrence="first"),
        "record0_end": source.mark(raw_prompt, line0, occurrence="first"),
        "record1_end": source.mark(raw_prompt, line1, occurrence="first"),
        "selected_case": source.mark(
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
    role_spans["answer_boundary"] = (len(input_ids) - 1, len(input_ids) - 1)
    prefix = " "
    candidate_token_ids = {
        f"a{index}": behavior.continuation_ids(
            tokenizer, rendered, prefix, answer
        )
        for index, answer in enumerate(semantic_codes)
    }
    return {
        "schema_version": "phase1082_case.v2",
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
            positions["codebook_end"] < positions["record0_end"]
            < positions["record1_end"] < positions["selected_case"]
            <= positions["request_end"] < positions["answer_boundary"]
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
            active_records_distinct &= row["record0"] != row["record1"]
        else:
            duplicate_records_equal &= row["record0"] == row["record1"]

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
        "lexical_worlds_disjoint": lexical_worlds_disjoint,
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
        "active_records_distinct": active_records_distinct,
        "duplicate_records_equal": duplicate_records_equal,
        "independent_item_splits": all(
            item_ids[(cell, "discovery")].isdisjoint(
                item_ids[(cell, "confirmation")]
            )
            for cell in CELLS
        ),
    }
    return {
        "schema_version": "phase1082_protocol_model_audit.v2",
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

    source_summary = read_json(SOURCE_PHASE1081)
    prereg = {
        "schema_version": "phase1082_preregistration.v2",
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
            "O": "eight natural verification operations",
            "X": "four disjoint lexical worlds fully crossed with O",
            "T": "two shared surface shells",
            "C": "active distinct truth records or duplicate-truth control",
            "M": "truth-record order or duplicated truth identity",
            "Q": "queried physical case position",
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
        "source_phase1081_protocol_digest": source_summary["protocol_digest"],
        "source_phase1081_summary_digest": source_summary["summary_digest"],
        "primary_population": (
            "All finite preregistered observations; behavior never deletes a "
            "cell from the descriptive map."
        ),
        "secondary_population": (
            "Units with at least 75% correct active-panel candidate comparisons."
        ),
        "behavior_gate_definition": (
            "For an operation-world cell, active candidate accuracy must reach "
            "0.80 and the target answer must occur before its paired answer in at "
            "least 0.75 of 12-token generations. An operation passes a model "
            "only when at least three worlds pass both gates."
        ),
        "measurement_order": [
            "complete operation-world and output-token audits",
            "behavior-only pilot without hidden-state access",
            "freeze protocol digest and all predictions",
            "sequential Qwen3, GLM4, DS7B FP16 scans",
            "active, duplicate, and content difference-in-differences fields",
            "downstream-only operation profiles",
            "independent item-split retrieval",
            "held-out lexical-world transfer",
            "semantic output-vocabulary transfer",
            "cross-world content-versus-duplicate advantage",
            "cross-model transfer and numerical integrity",
            "automatic non-causal gate",
        ],
        "evidence_levels": {
            "L0": "finite operation-world cell mapped",
            "L1": "operation retrieves across independent item splits",
            "L2": "L1 plus held-out lexical-world and output-vocabulary transfer",
            "L3": "L2 plus cross-world duplicate-route advantage",
            "L4": "L3 plus cross-model operation retrieval and advantage",
            "L5": "L4 plus behavior support in at least two models",
            "L6": "causal support; forbidden in Phase1082",
        },
        "interpretation_limits": [
            "Shared output words are experimental handles, not latent symbols.",
            "The failed revision-1 arbitrary-code pilot is diagnostic only and is not pooled.",
            "All tasks are binary verification tasks and may share a verification shell.",
            "Operation wording remains part of the operation and is not separated here.",
            "Difference-in-differences assumes approximate additivity.",
            "Cross-world transfer is not cross-model conservation.",
            "Local glossary translation tests in-context mapping, not stored multilingual translation.",
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
        "schema_version": "phase1082_protocol_audit.v2",
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
