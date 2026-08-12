#!/usr/bin/env python3
"""Freeze Phase1084 two-entity, shared-candidate attribute protocol.

All eight attribute queries share the same two entity candidates and the same
two-record dossier.  The requested field and target value appear only after
the common dossier.  In the active panel the entities have opposite values for
every field.  In the matched duplicate panel the two distinct entities receive
the same complete profile, so selecting A versus B has no semantic consequence
while the field selector, target value, entity vocabulary, shell, and output
protocol remain present.

This phase is a preregistered, descriptive test of one local pattern family.
It does not presume a latent formula or claim a causal language mechanism.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1083_same_carrier_attribute_protocol as source


PHASE = 1084
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
OPERATIONS = source.OPERATIONS
WORLDS = source.WORLDS
HELDOUT_WORLD = source.HELDOUT_WORLD
CELLS = tuple(
    f"{operation}__{world}"
    for operation in OPERATIONS
    for world in WORLDS
)
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

# Phase1083's post-hoc candidate is converted into a narrow preregistered
# measurement.  The scanner captures all three component types only in the
# middle third, at one pre-selection role, the selected-entity role, and the
# answer boundary.  A full-role/full-depth atlas is forbidden unless the
# frozen carrier-purity gates pass.
TARGET_RELATIVE_DEPTH_MIN = 1.0 / 3.0
TARGET_RELATIVE_DEPTH_MAX = 2.0 / 3.0
CAPTURE_ROLES = (
    "candidate1_end",
    "selected_candidate",
    "answer_boundary",
)
PRIMARY_PROFILE_ROLES = ("answer_boundary",)
PRE_QUERY_ROLES = ("candidate1_end",)
INTERMEDIATE_ROLES = ("selected_candidate",)
CONDITIONINGS = ("all_finite", "behavior_supported")
ITEMS_PER_CELL_SPLIT = 6
GENERATION_UNITS_PER_FAMILY_SPLIT = 4
GENERATION_STEPS = 12
ASSISTANT_PREFILL = "Answer:"
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1084_two_entity_attribute_route"
)
SOURCE_PHASE1083 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1083_same_carrier_attribute_atlas"
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
        "All two-entity, shared-candidate, exact-prefix, role, token, truth, "
        "split, and factor audits pass before hidden-state collection."
    ),
    "P2": (
        "At least six of eight attributes pass candidate and natural-generation "
        "behavior in at least three worlds for at least two models."
    ),
    "P3": (
        "Preregistered middle-answer-boundary content profiles retrieve at "
        "least six of eight attributes across independent item splits in at "
        "least two models."
    ),
    "P4": (
        "For at least two models, at least three of four held-out entity-world "
        "folds retrieve six of eight attributes."
    ),
    "P5": (
        "Yes/No to True/False content-profile transfer retrieves at least six "
        "of eight attributes in at least two models."
    ),
    "P6": (
        "In at least two models, middle-answer-boundary content identity exceeds "
        "the matched duplicate-profile route by 0.05 in at least eight of twelve "
        "directed cross-world pairs."
    ),
    "P7": (
        "The median maximum output/shell control-to-content ratio is at most one "
        "in at least two models."
    ),
    "P8": (
        "At least two directed model pairs retrieve six of eight attributes and "
        "at least four pairs show a 0.05 content-over-duplicate advantage."
    ),
    "P9": (
        "Every model passes FP16/no-quantization, finite-value, identity-repeat, "
        "and exact pre-selection causal-order audits."
    ),
}


write_json = source.write_json
write_jsonl = source.write_jsonl
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest
tokenizer_for = source.tokenizer_for
offset_token_spans = source.offset_token_spans
behavior = source.behavior
mark_source = source.source
WORLD_DATA = source.WORLD_DATA
ATTRIBUTE_PAIRS = source.ATTRIBUTE_PAIRS


SHELLS = {
    0: (
        "Decision rule: answer {support_code} if the selected entity has the "
        "target value for the named field in the dossier; answer "
        "{contradict_code} otherwise.\n"
        "Dossier:\n{record0}\n{record1}\n"
        "Field to check: {field}.\n"
        "Target value: {target_value}.\n"
        "Candidate A: {candidate0}.\n"
        "Candidate B: {candidate1}.\n"
        "Selected candidate: {selected}.\n"
        "Write only the answer word."
    ),
    1: (
        "Answer key: use {support_code} when the chosen entity carries the "
        "requested target value in its profile; use {contradict_code} when it "
        "does not.\n"
        "Profile records:\n{record0}\n{record1}\n"
        "Requested field: {field}.\n"
        "Requested value: {target_value}.\n"
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


def cell_id(operation: str, world: str) -> str:
    return f"{operation}__{world}"


def split_cell(cell: str) -> tuple[str, str]:
    operation, world = cell.split("__", 1)
    return operation, world


def profile_text(entity: str, values: dict[str, str], label: str) -> str:
    return (
        f"Record {label}: entity '{entity}'; category {values['category']}; "
        f"color {values['color']}; size {values['size']}; "
        f"material {values['material']}; location {values['location']}; "
        f"alias {values['alias']}; schedule {values['schedule']}; "
        f"condition {values['condition']}."
    )


def paired_item(world: str, index: int) -> dict[str, Any]:
    data = WORLD_DATA[world]
    entity0 = str(data["entities0"][index])
    entity1 = str(data["entities1"][index])
    values0: dict[str, str] = {"category": str(data["class0"])}
    values1: dict[str, str] = {"category": str(data["class1"])}
    for attribute_index, attribute in enumerate(OPERATIONS[1:], start=1):
        pair = ATTRIBUTE_PAIRS[world][attribute][index % 3]
        side = (index + attribute_index) % 2
        values0[attribute] = str(pair[side])
        values1[attribute] = str(pair[1 - side])
    return {
        "base_item_id": f"{world}.{index:02d}",
        "entity0": entity0,
        "entity1": entity1,
        "values0": values0,
        "values1": values1,
    }


def build_items() -> dict[str, tuple[dict[str, Any], ...]]:
    output: dict[str, tuple[dict[str, Any], ...]] = {}
    for world in WORLDS:
        bases = tuple(
            paired_item(world, index)
            for index in range(ITEMS_PER_CELL_SPLIT * len(SPLITS))
        )
        for operation in OPERATIONS:
            cell = cell_id(operation, world)
            output[cell] = tuple({
                **base,
                "item_id": f"{cell}.{index:02d}",
                "operation": operation,
                "world": world,
                "target_value": base["values0"][operation],
                "alternate_value": base["values1"][operation],
            } for index, base in enumerate(bases))
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


def render_panel(
    item: dict[str, Any], panel: str, mapping: int
) -> dict[str, Any]:
    entities = (str(item["entity0"]), str(item["entity1"]))
    values0 = dict(item["values0"])
    values1 = dict(item["values1"])
    if panel == "active":
        semantic_profiles = (values0, values1)
        semantic_truth = (0, 1)
    else:
        repeated = values0 if mapping == 0 else values1
        semantic_profiles = (repeated, repeated)
        duplicate_truth = 0 if mapping == 0 else 1
        semantic_truth = (duplicate_truth, duplicate_truth)
    order = (0, 1) if mapping == 0 else (1, 0)
    candidates = (entities[order[0]], entities[order[1]])
    profiles = (
        semantic_profiles[order[0]], semantic_profiles[order[1]]
    )
    truths = (semantic_truth[order[0]], semantic_truth[order[1]])
    return {
        "candidate0": candidates[0],
        "candidate1": candidates[1],
        "record0": profile_text(candidates[0], profiles[0], "A"),
        "record1": profile_text(candidates[1], profiles[1], "B"),
        "profile0": profiles[0],
        "profile1": profiles[1],
        "semantic0": truths[0],
        "semantic1": truths[1],
    }


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
    panel_data = render_panel(item, panel, mapping)
    semantic_answer = int(
        panel_data["semantic0"] if query == 0 else panel_data["semantic1"]
    )
    target_answer = semantic_codes[semantic_answer]
    selected = "A" if query == 0 else "B"
    raw_prompt = SHELLS[template].format(
        support_code=semantic_codes[0],
        contradict_code=semantic_codes[1],
        record0=panel_data["record0"],
        record1=panel_data["record1"],
        field=operation,
        target_value=item["target_value"],
        candidate0=panel_data["candidate0"],
        candidate1=panel_data["candidate1"],
        selected=selected,
    )
    rule = (
        f"Decision rule: answer {semantic_codes[0]} if the selected entity has "
        f"the target value for the named field in the dossier; answer "
        f"{semantic_codes[1]} otherwise."
        if template == 0 else
        f"Answer key: use {semantic_codes[0]} when the chosen entity carries "
        f"the requested target value in its profile; use "
        f"{semantic_codes[1]} when it does not."
    )
    field_line = (
        f"Field to check: {operation}." if template == 0
        else f"Requested field: {operation}."
    )
    target_line = (
        f"Target value: {item['target_value']}." if template == 0
        else f"Requested value: {item['target_value']}."
    )
    candidate0_line = (
        f"Candidate A: {panel_data['candidate0']}." if template == 0
        else f"Option A: {panel_data['candidate0']}."
    )
    candidate1_line = (
        f"Candidate B: {panel_data['candidate1']}." if template == 0
        else f"Option B: {panel_data['candidate1']}."
    )
    raw_spans = {
        "rule_end": mark_source.mark(raw_prompt, rule, occurrence="first"),
        "dossier_end": mark_source.mark(
            raw_prompt, panel_data["record1"], occurrence="first"
        ),
        "field_selector": mark_source.mark(
            raw_prompt, field_line, occurrence="first"
        ),
        "target_value": mark_source.mark(
            raw_prompt, target_line, occurrence="first"
        ),
        "candidate0_end": mark_source.mark(
            raw_prompt, candidate0_line, occurrence="first"
        ),
        "candidate1_end": mark_source.mark(
            raw_prompt, candidate1_line, occurrence="first"
        ),
        "selected_candidate": mark_source.mark(
            raw_prompt, selected, occurrence="last"
        ),
    }
    rendered = behavior.render_native(
        tokenizer, model_name, raw_prompt, with_system=False
    )
    rendered += ASSISTANT_PREFILL
    input_ids = [
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_spans = offset_token_spans(tokenizer, rendered, raw_prompt, raw_spans)
    role_spans["answer_boundary"] = (len(input_ids) - 1, len(input_ids) - 1)
    prefix = " "
    candidate_token_ids = {
        f"a{index}": behavior.continuation_ids(
            tokenizer, rendered, prefix, answer
        )
        for index, answer in enumerate(semantic_codes)
    }
    return {
        "schema_version": "phase1084_case.v1",
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
        "entity0": item["entity0"],
        "entity1": item["entity1"],
        "field": operation,
        "target_value": item["target_value"],
        "distractor_value": item["alternate_value"],
        "candidate0": panel_data["candidate0"],
        "candidate1": panel_data["candidate1"],
        "record0": panel_data["record0"],
        "record1": panel_data["record1"],
        "profile0": panel_data["profile0"],
        "profile1": panel_data["profile1"],
        "control_type": (
            "matched_duplicate_entity_profiles" if panel == "duplicate" else None
        ),
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
            key: [int(values[0])] for key, values in candidate_token_ids.items()
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
    role_names = (
        "rule_end", "dossier_end", "field_selector", "target_value",
        "candidate0_end", "candidate1_end", "selected_candidate",
        "answer_boundary",
    )
    checks_acc = Counter()
    checks_total = Counter()

    def observe(name: str, condition: bool) -> None:
        checks_total[name] += 1
        checks_acc[name] += int(bool(condition))

    for row in cases:
        by_unit[row["unit_id"]].append(row)
        width = len(row["input_ids"])
        observe("role_spans_valid", all(
            0 <= row["role_spans"][role][0]
            <= row["role_spans"][role][1] < width
            for role in role_names
        ))
        positions = row["role_positions"]
        observe("role_order_valid", (
            positions["rule_end"] < positions["dossier_end"]
            < positions["field_selector"] < positions["target_value"]
            < positions["candidate0_end"] < positions["candidate1_end"]
            < positions["selected_candidate"] < positions["answer_boundary"]
        ))
        first0 = set(row["candidate_first_token_ids"]["a0"])
        first1 = set(row["candidate_first_token_ids"]["a1"])
        observe(
            "candidate_first_tokens_disjoint",
            bool(first0) and bool(first1) and first0.isdisjoint(first1),
        )
        observe("candidate_codes_single_token", row["label_token_widths"] == [1, 1])
        observe(
            "expected_class_matches_semantic_answer",
            row["expected_class"] == f"a{row['semantic_answer_index']}",
        )
        observe("entity_candidates_distinct", row["candidate0"] != row["candidate1"])
        observe(
            "target_alternate_distinct",
            row["target_value"] != row["distractor_value"],
        )
        prompt_folded = row["raw_prompt"].casefold()
        observe("factor_names_absent_from_prompts", not any(
            re.search(rf"\b{re.escape(word)}\b", prompt_folded)
            for word in BANNED_FACTOR_WORDS
        ))
        if row["panel"] == "active":
            observe("active_profiles_differ_all_fields", all(
                row["profile0"][field] != row["profile1"][field]
                for field in OPERATIONS
            ))
        else:
            observe("duplicate_profiles_equal_all_fields", all(
                row["profile0"][field] == row["profile1"][field]
                for field in OPERATIONS
            ))

    active_answer_depends_on_query = True
    duplicate_answer_independent_of_query = True
    output_vocabulary_changes_words_not_semantics = True
    pre_query_roles_before_divergence = True
    for rows in by_unit.values():
        lookup = {row["state"]: row for row in rows}
        for template in (0, 1):
            for panel in PANELS:
                for mapping in (0, 1):
                    for output_set in (0, 1):
                        left = lookup[f"t{template}_c{panel}_m{mapping}_q0_w{output_set}"]
                        right = lookup[f"t{template}_c{panel}_m{mapping}_q1_w{output_set}"]
                        divergence = common_prefix_divergence([
                            left["input_ids"], right["input_ids"]
                        ])
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
                        first = lookup[f"t{template}_c{panel}_m{mapping}_q{query}_w0"]
                        second = lookup[f"t{template}_c{panel}_m{mapping}_q{query}_w1"]
                        output_vocabulary_changes_words_not_semantics &= (
                            first["semantic_answer_index"]
                            == second["semantic_answer_index"]
                            and first["target_answer"] != second["target_answer"]
                        )

    by_base_state: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        key = (
            row["world"], row["split"], row["base_item_id"],
            row["template"], row["panel"], row["mapping"],
            row["query"], row["output_set"],
        )
        by_base_state[key].append(row)
    same_dossier_prefix = True
    same_entity_candidates = True
    selector_after_dossier = True
    for rows in by_base_state.values():
        same_dossier_prefix &= len(rows) == len(OPERATIONS)
        prefixes = [
            row["input_ids"][:row["role_positions"]["dossier_end"] + 1]
            for row in rows
        ]
        same_dossier_prefix &= all(value == prefixes[0] for value in prefixes[1:])
        same_entity_candidates &= len({
            (row["candidate0"], row["candidate1"]) for row in rows
        }) == 1
        selector_after_dossier &= all(
            row["role_positions"]["dossier_end"]
            < row["role_positions"]["field_selector"]
            for row in rows
        )

    lexical_sets = [
        set(data["entities0"]) | set(data["entities1"])
        for data in WORLD_DATA.values()
    ]
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
    checks = {
        name: checks_acc[name] == checks_total[name]
        for name in checks_total
    }
    checks.update({
        "case_count": len(cases)
        == len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT * len(STATES),
        "unit_count": len(by_unit)
        == len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT,
        "complete_units": all(len(rows) == len(STATES) for rows in by_unit.values()),
        "full_attribute_world_cross": (
            len(counts) == len(OPERATIONS) * len(WORLDS) * len(SPLITS)
            and all(
                count == ITEMS_PER_CELL_SPLIT * len(STATES)
                for count in counts.values()
            )
        ),
        "entity_worlds_disjoint": all(
            lexical_sets[left].isdisjoint(lexical_sets[right])
            for left in range(len(lexical_sets))
            for right in range(left + 1, len(lexical_sets))
        ),
        "same_dossier_token_prefix_across_attributes": same_dossier_prefix,
        "same_entity_candidates_across_attributes": same_entity_candidates,
        "attribute_selector_after_common_dossier": selector_after_dossier,
        "active_answer_depends_on_query": active_answer_depends_on_query,
        "duplicate_answer_independent_of_query": duplicate_answer_independent_of_query,
        "output_vocabulary_changes_words_not_semantics": (
            output_vocabulary_changes_words_not_semantics
        ),
        "pre_query_roles_before_divergence": pre_query_roles_before_divergence,
        "independent_item_splits": all(
            item_ids[(cell, "discovery")].isdisjoint(
                item_ids[(cell, "confirmation")]
            )
            for cell in CELLS
        ),
    })
    return {
        "schema_version": "phase1084_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
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
            failed = [
                name for name, passed in audit["checks"].items() if not passed
            ]
            raise RuntimeError(f"protocol audit failed for {model_name}: {failed}")
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", cases)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_case_digests[model_name] = audit["case_digest"]
        model_audits[model_name] = audit

    source_summary = read_json(SOURCE_PHASE1083)
    prereg = {
        "schema_version": "phase1084_preregistration.v1",
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
            "O": "eight late-selected fields over the same two-entity dossier",
            "X": "four disjoint entity worlds fully crossed with O",
            "T": "two shared natural shells",
            "C": "opposite active profiles or matched duplicate complete profiles",
            "M": "entity order and duplicate truth identity",
            "Q": "selected physical entity candidate A or B",
            "W": "Yes/No or True/False with fixed truth semantics",
        },
        "capture_scope": {
            "relative_depth_min": TARGET_RELATIVE_DEPTH_MIN,
            "relative_depth_max": TARGET_RELATIVE_DEPTH_MAX,
            "components": ["residual", "attention_output", "mlp_output"],
            "reason": (
                "Independent preregistration of the Phase1083 post-hoc "
                "middle-answer-boundary candidate."
            ),
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
        "source_phase1083_protocol_digest": source_summary["protocol_digest"],
        "source_phase1083_summary_digest": source_summary["summary_digest"],
        "primary_population": (
            "All finite preregistered observations; behavior is reported "
            "separately and never used to erase descriptive cells."
        ),
        "behavior_gate_definition": (
            "An attribute-world cell requires active candidate accuracy >=0.80 "
            "and target-before-distractor generation >=0.75. An attribute passes "
            "a model in at least three worlds; at least six attributes must pass."
        ),
        "measurement_order": [
            "freeze exact two-entity shared-candidate protocol and digest",
            "run behavior-only Qwen3, GLM4, DS7B sequentially",
            "run preregistered middle-band three-role scans sequentially",
            "compute active, duplicate, and difference-in-differences fields",
            "test item split, held-out world, and output vocabulary transfer",
            "test content-over-duplicate advantage and control ratio",
            "audit direct cross-model alignment and FP16 numerical integrity",
            "authorize a full-role/full-depth atlas only if purity gates pass",
        ],
        "evidence_levels": {
            "L0": "finite targeted field mapped",
            "L1": "independent item-split retrieval",
            "L2": "held-out world and output vocabulary transfer",
            "L3": "cross-world content-over-duplicate advantage",
            "L4": "cross-model retrieval and advantage",
            "L5": "behavior support in at least two models",
            "L6": "causal evidence; not tested in Phase1084",
        },
        "interpretation_limits": [
            "The content route is a panel-by-selection interaction, not a pure latent variable.",
            "Difference-in-differences assumes matched scaling and approximate additivity.",
            "Selectors and target values remain natural words and may interact with the dossier.",
            "Selector-only and value-only omissions are not formal controls because they make the task underdetermined.",
            "Within-model decoding is not cross-model physical conservation.",
            "A middle-band confirmation does not identify a head, neuron, or minimal causal circuit.",
            "Attribute matching is not a theory of translation, contrast, punctuation, syntax, or stored knowledge.",
            "No result establishes optimality, brain homology, or a new mathematical theory.",
        ],
        "automatic_next": {
            "continue_full_atlas_only_if": (
                "P1-P7 and P9 pass, including content-over-duplicate advantage "
                "and control/content <= 1 in at least two models."
            ),
            "continue_cross_model_alignment_if": (
                "P1-P7 and P9 pass but P8 fails."
            ),
            "stop_hidden_escalation_if": (
                "P2, P6, P7, or P9 fails; preserve the map and diagnose the failed gate."
            ),
        },
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    global_audit = {
        "schema_version": "phase1084_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_audits": model_audits,
        "all_checks_passed": all(
            audit["all_checks_passed"] for audit in model_audits.values()
        ),
    }
    global_audit["audit_digest"] = digest(global_audit)
    write_json(protocol_root / "audit.json", global_audit)
    print({
        "phase": PHASE,
        "case_count_per_model": prereg["case_count_per_model"],
        "unit_count_per_model": prereg["unit_count_per_model"],
        "all_checks_passed": global_audit["all_checks_passed"],
        "protocol_digest": prereg["protocol_digest"],
    })


if __name__ == "__main__":
    main()
