#!/usr/bin/env python3
"""Freeze Phase1085 natural direct-entity attribute protocol.

Phase1084 established that A/B indirection made the relation task unreliable
in GLM4 and DS7B.  Phase1085 preserves the two-entity shared dossier, all eight
late-selected attributes, output vocabularies, and matched duplicate-profile
control, but names the selected entity directly at the end of the request.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1084_two_entity_attribute_protocol as base


PHASE = 1085
PROTOCOL_REVISION = 1
MODELS = base.MODELS
PRECISION = base.PRECISION
QUANTIZATION = base.QUANTIZATION
OPERATIONS = base.OPERATIONS
WORLDS = base.WORLDS
HELDOUT_WORLD = base.HELDOUT_WORLD
CELLS = base.CELLS
FAMILIES = CELLS
BASE_FAMILIES = CELLS
EXPLORATORY_FAMILIES: tuple[str, ...] = ()
SPLITS = base.SPLITS
PANELS = base.PANELS
OUTPUT_PAIRS = base.OUTPUT_PAIRS
CODE_WORDS = base.CODE_WORDS
CODE_PAIRS = OUTPUT_PAIRS
STATES = base.STATES
TARGET_RELATIVE_DEPTH_MIN = base.TARGET_RELATIVE_DEPTH_MIN
TARGET_RELATIVE_DEPTH_MAX = base.TARGET_RELATIVE_DEPTH_MAX
CAPTURE_ROLES = ("dossier_end", "query_entity", "answer_boundary")
PRIMARY_PROFILE_ROLES = ("answer_boundary",)
PRE_QUERY_ROLES = ("dossier_end",)
INTERMEDIATE_ROLES = ("query_entity",)
CONDITIONINGS = base.CONDITIONINGS
ITEMS_PER_CELL_SPLIT = base.ITEMS_PER_CELL_SPLIT
GENERATION_UNITS_PER_FAMILY_SPLIT = base.GENERATION_UNITS_PER_FAMILY_SPLIT
GENERATION_STEPS = base.GENERATION_STEPS
ASSISTANT_PREFILL = base.ASSISTANT_PREFILL
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1085_direct_entity_attribute_route"
)
SOURCE_PHASE1084 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1084_two_entity_attribute_route"
    / "analysis" / "behavior_stop_summary.json"
)
EVIDENCE_THRESHOLDS = dict(base.EVIDENCE_THRESHOLDS)
PROSPECTIVE_PREDICTIONS = {
    key: value.replace("two-entity, shared-candidate", "two-entity, direct-name")
    for key, value in base.PROSPECTIVE_PREDICTIONS.items()
}

write_json = base.write_json
write_jsonl = base.write_jsonl
read_json = base.read_json
read_jsonl = base.read_jsonl
digest = base.digest
tokenizer_for = base.tokenizer_for
offset_token_spans = base.offset_token_spans
behavior = base.behavior
mark_source = base.mark_source
WORLD_DATA = base.WORLD_DATA
ITEMS_BY_CELL = base.ITEMS_BY_CELL
split_items = base.split_items
split_cell = base.split_cell
state_factors = base.state_factors
output_code_pair = base.output_code_pair
render_panel = base.render_panel
encoded_width = base.encoded_width
common_prefix_divergence = base.common_prefix_divergence
BANNED_FACTOR_WORDS = base.BANNED_FACTOR_WORDS


SHELLS = {
    0: (
        "Decision rule: answer {support_code} if the named entity has the "
        "target value for the requested field in the dossier; answer "
        "{contradict_code} otherwise.\n"
        "Dossier:\n{record0}\n{record1}\n"
        "Field to check: {field}.\n"
        "Target value: {target_value}.\n"
        "Entity to check: {selected_entity}.\n"
        "Write only the answer word."
    ),
    1: (
        "Answer key: use {support_code} when the requested field of the named "
        "entity equals the target value in its profile; use {contradict_code} "
        "when it does not.\n"
        "Profile records:\n{record0}\n{record1}\n"
        "Requested field: {field}.\n"
        "Requested value: {target_value}.\n"
        "Entity under review: {selected_entity}.\n"
        "Return only the answer word."
    ),
}


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
    selected_entity = str(
        panel_data["candidate0"] if query == 0 else panel_data["candidate1"]
    )
    target_answer = semantic_codes[semantic_answer]
    raw_prompt = SHELLS[template].format(
        support_code=semantic_codes[0],
        contradict_code=semantic_codes[1],
        record0=panel_data["record0"],
        record1=panel_data["record1"],
        field=operation,
        target_value=item["target_value"],
        selected_entity=selected_entity,
    )
    rule = (
        f"Decision rule: answer {semantic_codes[0]} if the named entity has the "
        f"target value for the requested field in the dossier; answer "
        f"{semantic_codes[1]} otherwise."
        if template == 0 else
        f"Answer key: use {semantic_codes[0]} when the requested field of the "
        f"named entity equals the target value in its profile; use "
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
    query_line = (
        f"Entity to check: {selected_entity}." if template == 0
        else f"Entity under review: {selected_entity}."
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
        "query_entity": mark_source.mark(
            raw_prompt, query_line, occurrence="first"
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
        "schema_version": "phase1085_case.v1",
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
        "selected_label_text": selected_entity,
        "selected_position": query,
        "selected_entity": selected_entity,
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


def audit_model(
    model_name: str,
    tokenizer,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checks_acc = Counter()
    checks_total = Counter()

    def observe(name: str, condition: bool) -> None:
        checks_total[name] += 1
        checks_acc[name] += int(bool(condition))

    roles = (
        "rule_end", "dossier_end", "field_selector", "target_value",
        "query_entity", "answer_boundary",
    )
    for row in cases:
        by_unit[row["unit_id"]].append(row)
        width = len(row["input_ids"])
        observe("role_spans_valid", all(
            0 <= row["role_spans"][role][0]
            <= row["role_spans"][role][1] < width
            for role in roles
        ))
        positions = row["role_positions"]
        observe("role_order_valid", (
            positions["rule_end"] < positions["dossier_end"]
            < positions["field_selector"] < positions["target_value"]
            < positions["query_entity"] < positions["answer_boundary"]
        ))
        first0 = set(row["candidate_first_token_ids"]["a0"])
        first1 = set(row["candidate_first_token_ids"]["a1"])
        observe(
            "candidate_first_tokens_disjoint",
            bool(first0) and bool(first1) and first0.isdisjoint(first1),
        )
        observe("candidate_codes_single_token", row["label_token_widths"] == [1, 1])
        observe("entity_candidates_distinct", row["candidate0"] != row["candidate1"])
        observe("selected_entity_is_candidate", row["selected_entity"] in (
            row["candidate0"], row["candidate1"]
        ))
        observe("target_alternate_distinct", row["target_value"] != row["distractor_value"])
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
    selected_entity_changes_with_query = True
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
                        selected_entity_changes_with_query &= (
                            left["selected_entity"] != right["selected_entity"]
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
    common_dossier_prefix = True
    shared_entity_pair = True
    for rows in by_base_state.values():
        prefixes = [
            row["input_ids"][:row["role_positions"]["dossier_end"] + 1]
            for row in rows
        ]
        common_dossier_prefix &= (
            len(rows) == len(OPERATIONS)
            and all(value == prefixes[0] for value in prefixes[1:])
        )
        shared_entity_pair &= len({
            (row["candidate0"], row["candidate1"], row["selected_entity"])
            for row in rows
        }) == 1

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
        "same_dossier_token_prefix_across_attributes": common_dossier_prefix,
        "same_entity_pair_and_query_across_attributes": shared_entity_pair,
        "selected_entity_changes_with_query": selected_entity_changes_with_query,
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
        "schema_version": "phase1085_protocol_model_audit.v1",
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
            failed = [name for name, passed in audit["checks"].items() if not passed]
            raise RuntimeError(f"protocol audit failed for {model_name}: {failed}")
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", cases)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_case_digests[model_name] = audit["case_digest"]
        model_audits[model_name] = audit

    prior = read_json(SOURCE_PHASE1084)
    prereg = {
        "schema_version": "phase1085_preregistration.v1",
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
            "O": "eight requested fields over one common two-entity dossier",
            "X": "four disjoint entity worlds fully crossed with O",
            "T": "two shared natural shells",
            "C": "opposite active profiles or matched duplicate complete profiles",
            "M": "record order and duplicate truth identity",
            "Q": "late natural selected-entity name",
            "W": "Yes/No or True/False with fixed truth semantics",
        },
        "capture_scope": {
            "relative_depth_min": TARGET_RELATIVE_DEPTH_MIN,
            "relative_depth_max": TARGET_RELATIVE_DEPTH_MAX,
            "components": ["residual", "attention_output", "mlp_output"],
            "reason": "Independent middle-answer-boundary confirmation after a fresh behavior gate.",
        },
        "capture_roles": list(CAPTURE_ROLES),
        "primary_profile_roles": list(PRIMARY_PROFILE_ROLES),
        "pre_query_roles": list(PRE_QUERY_ROLES),
        "conditionings": list(CONDITIONINGS),
        "assistant_prefill": ASSISTANT_PREFILL,
        "items_per_cell_split": ITEMS_PER_CELL_SPLIT,
        "case_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT * len(STATES),
        "unit_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT,
        "generation_units_per_family_split": GENERATION_UNITS_PER_FAMILY_SPLIT,
        "generation_steps": GENERATION_STEPS,
        "model_case_digests": model_case_digests,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_phase1084_protocol_digest": prior["protocol_digest"],
        "source_phase1084_summary_digest": prior["summary_digest"],
        "primary_population": "All finite preregistered observations.",
        "behavior_gate_definition": (
            "An attribute-world cell requires active candidate accuracy >=0.80 "
            "and target-before-distractor generation >=0.75. An attribute passes "
            "in at least three worlds; at least six attributes and two models pass."
        ),
        "measurement_order": [
            "freeze direct-entity protocol and all thresholds",
            "run Qwen3, GLM4, DS7B FP16 behavior sequentially",
            "stop if fewer than two models pass six attributes",
            "otherwise run the middle-band targeted scans sequentially",
            "test independent split, world, output, carrier, control, and model transfer",
            "authorize full atlas only after every purity gate",
        ],
        "evidence_levels": base.read_json(
            base.OUT_ROOT / "protocol" / "preregistration.json"
        )["evidence_levels"],
        "interpretation_limits": [
            "Direct entity names remove A/B indirection but introduce entity-token differences captured by the duplicate route.",
            "The content route remains a difference-in-differences interaction, not a pure latent variable.",
            "Approximate additivity, matched scale, and behavior support remain necessary assumptions.",
            "Within-model retrieval is not cross-model physical conservation or causality.",
            "This focused attribute task does not explain translation, punctuation, contrast, syntax, or language globally.",
            "No result establishes neural optimality, brain homology, or a new mathematics.",
        ],
        "automatic_next": {
            "continue_targeted_scan_only_if": "P1 and P2 behavior gates pass.",
            "continue_full_atlas_only_if": "P1-P7 and P9 pass.",
            "continue_alignment_if": "P1-P7 and P9 pass but P8 fails.",
            "stop_if": "P2, P6, P7, or P9 fails.",
        },
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    global_audit = {
        "schema_version": "phase1085_protocol_audit.v1",
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
