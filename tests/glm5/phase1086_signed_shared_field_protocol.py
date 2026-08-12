#!/usr/bin/env python3
"""Freeze Phase1086 signed shared-field and attribute-residual protocol.

The same natural two-entity dossier supports eight late questions.  The
registered analysis first asks whether a truth-aligned signed response field
is shared across the eight questions, then asks whether attribute-specific
residuals transfer across lexical worlds.  These are separate gates.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1085_direct_entity_attribute_protocol as base


PHASE = 1086
PROTOCOL_REVISION = 1
MODELS = base.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
OPERATIONS = base.OPERATIONS
WORLDS = base.WORLDS
CELLS = tuple(
    f"{operation}__{world}"
    for operation in OPERATIONS
    for world in WORLDS
)
FAMILIES = CELLS
SPLITS = base.SPLITS
PANELS = ("active", "field_null")
OUTPUT_PAIRS = base.OUTPUT_PAIRS
CODE_WORDS = base.CODE_WORDS
STATES = tuple(
    f"t{template}_c{panel}_m{mapping}_q{query}_w{output_set}"
    for template in (0, 1)
    for panel in PANELS
    for mapping in (0, 1)
    for query in (0, 1)
    for output_set in (0, 1)
)

TARGET_RELATIVE_DEPTH_MIN = 1.0 / 3.0
TARGET_RELATIVE_DEPTH_MAX = 2.0 / 3.0
CAPTURE_ROLES = ("dossier_end", "query_end", "answer_boundary")
PRIMARY_PROFILE_ROLES = ("answer_boundary",)
PRE_QUERY_ROLES = ("dossier_end",)
ITEMS_PER_CELL_SPLIT = 6
GENERATION_UNITS_PER_FAMILY_SPLIT = 4
GENERATION_STEPS = 12
ASSISTANT_PREFILL = "Answer:"
SIGNED_PROJECTION_DIM = 96
SIGNED_PROJECTION_REPLICATES = 2
SIGNED_PROJECTION_SEED = 1086001
SIGNED_FIELDS = ("active_truth", "field_null", "content")
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1086_signed_shared_field"
)
SOURCE_PHASE1085 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1085_direct_entity_attribute_route"
    / "analysis" / "final_summary.json"
)


EVIDENCE_THRESHOLDS = {
    "candidate_accuracy_for_operation_behavior": 0.80,
    "generation_target_before_distractor_accuracy": 0.75,
    "minimum_behavior_worlds_per_operation": 3,
    "minimum_behavior_operations": 6,
    "minimum_behavior_models": 2,
    "maximum_projection_median_abs_norm_error": 0.08,
    "maximum_projection_p95_abs_norm_error": 0.20,
    "minimum_shared_split_cosine": 0.50,
    "minimum_shared_content_over_null_advantage": 0.10,
    "minimum_shared_worlds": 3,
    "minimum_cross_world_pairs": 8,
    "minimum_surface_transfer_cosine": 0.40,
    "minimum_output_transfer_cosine": 0.40,
    "minimum_attribute_top1": 6,
    "permutation_p_max": 0.01,
    "minimum_attribute_heldout_worlds": 3,
    "maximum_surface_to_content_ratio": 1.0,
    "minimum_cross_model_geometry_cosine": 0.50,
    "minimum_cross_model_geometry_pairs": 2,
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_hidden_finite_fraction": 0.97,
    "pre_query_tolerance": 1e-8,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": "All protocol, prefix, truth, role, split, surface, and token audits pass.",
    "P2": "At least two models pass at least six operations in three lexical worlds.",
    "P3": "Both signed sketches pass their preregistered norm-distortion audits.",
    "P4": (
        "In at least two models and both sketches, the truth-aligned content "
        "centroid repeats across item splits in three worlds and exceeds the "
        "field-null centroid by at least 0.10."
    ),
    "P5": (
        "In at least two models and both sketches, at least eight of twelve "
        "directed world pairs repeat the shared content field with cosine at "
        "least 0.50 and content-over-null advantage at least 0.10."
    ),
    "P6": (
        "The shared signed content field transfers across natural surfaces and "
        "Yes/No versus True/False in at least two models and both sketches."
    ),
    "P7": (
        "Centered attribute residuals retrieve at least six of eight operations "
        "across independent items in at least two models and both sketches."
    ),
    "P8": (
        "Centered attribute residuals retrieve at least six of eight operations "
        "in at least three held-out worlds for two models and beat field-null."
    ),
    "P9": (
        "At least two directed model pairs repeat the within-model operation "
        "relation geometry; projected coordinates are never compared directly."
    ),
    "P10": "All models pass FP16, finite-value, identity, and pre-query audits.",
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
ATTRIBUTE_PAIRS = base.base.ATTRIBUTE_PAIRS


FACT_PATTERNS = {
    0: {
        "category": "{entity} belongs to the {value} class.",
        "color": "{entity} appears {value}.",
        "size": "{entity} has a {value} scale.",
        "material": "{entity} is made of {value}.",
        "location": "{entity} is kept at {value}.",
        "alias": "{entity} is also called {value}.",
        "schedule": "{entity} is scheduled for {value}.",
        "condition": "{entity} is currently {value}.",
    },
    1: {
        "category": "The kind assigned to {entity} is {value}.",
        "color": "The recorded hue of {entity} is {value}.",
        "size": "The recorded scale of {entity} is {value}.",
        "material": "The substance recorded for {entity} is {value}.",
        "location": "The storage place for {entity} is {value}.",
        "alias": "The alternate name for {entity} is {value}.",
        "schedule": "The recorded time for {entity} is {value}.",
        "condition": "The present state of {entity} is {value}.",
    },
}

QUERY_PATTERNS = {
    0: {
        "category": "Does {entity} belong to the {value} class?",
        "color": "Does {entity} appear {value}?",
        "size": "Does {entity} have a {value} scale?",
        "material": "Is {entity} made of {value}?",
        "location": "Is {entity} kept at {value}?",
        "alias": "Is {entity} also called {value}?",
        "schedule": "Is {entity} due at {value}?",
        "condition": "Is {entity} currently {value}?",
    },
    1: {
        "category": "Is {value} the kind assigned to {entity}?",
        "color": "Is the recorded hue of {entity} {value}?",
        "size": "Is the recorded scale of {entity} {value}?",
        "material": "Is the recorded substance of {entity} {value}?",
        "location": "Is the recorded storage place of {entity} {value}?",
        "alias": "Is the alternate name of {entity} {value}?",
        "schedule": "Is {value} the appointed time for {entity}?",
        "condition": "Is the present state of {entity} {value}?",
    },
}

SHELLS = {
    0: (
        "Use only the dossier. Reply {true_word} when the question agrees "
        "with it, and {false_word} when it does not.\n"
        "Dossier:\n{record0}\n{record1}\n"
        "Question: {question}\n"
        "Write only the reply word."
    ),
    1: (
        "Judge the final question from the descriptions below. Return "
        "{true_word} for agreement and {false_word} for disagreement.\n"
        "Descriptions:\n{record0}\n{record1}\n"
        "Check: {question}\n"
        "Return just the reply word."
    ),
}

# Twelve deterministic, distinct orders.  Each item uses one base order and
# the second surface uses its reverse, preventing a fixed field position.
FIELD_ORDERS = tuple(
    tuple(OPERATIONS[(index + shift) % len(OPERATIONS)] for index in range(len(OPERATIONS)))
    for shift in range(8)
) + tuple(
    tuple(reversed(tuple(
        OPERATIONS[(index + shift) % len(OPERATIONS)]
        for index in range(len(OPERATIONS))
    )))
    for shift in range(4)
)

BANNED_CANONICAL_QUERY_LABELS = tuple(OPERATIONS)


def cell_id(operation: str, world: str) -> str:
    return f"{operation}__{world}"


def split_cell(cell: str) -> tuple[str, str]:
    return tuple(cell.split("__", 1))  # type: ignore[return-value]


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
        "field_order_index": index,
    }


BASE_ITEMS = {
    world: tuple(paired_item(world, index) for index in range(12))
    for world in WORLDS
}


def split_items(cell: str, split: str) -> tuple[dict[str, Any], ...]:
    operation, world = split_cell(cell)
    start = 0 if split == "discovery" else ITEMS_PER_CELL_SPLIT
    return tuple({
        **item,
        "item_id": f"{cell}.{index:02d}",
        "operation": operation,
        "world": world,
    } for index, item in enumerate(
        BASE_ITEMS[world][start:start + ITEMS_PER_CELL_SPLIT], start=start
    ))


def state_factors(state: str) -> tuple[int, str, int, int, int]:
    match = re.fullmatch(
        r"t([01])_c(active|field_null)_m([01])_q([01])_w([01])",
        state,
    )
    if not match:
        raise ValueError(f"invalid state: {state}")
    return (
        int(match.group(1)), str(match.group(2)), int(match.group(3)),
        int(match.group(4)), int(match.group(5)),
    )


def output_code_pair(output_set: int) -> tuple[str, str]:
    return OUTPUT_PAIRS[output_set]


def field_order(item: dict[str, Any], template: int) -> tuple[str, ...]:
    base_order = FIELD_ORDERS[int(item["field_order_index"])]
    return base_order if template == 0 else tuple(reversed(base_order))


def render_record(
    entity: str,
    values: dict[str, str],
    template: int,
    order: tuple[str, ...],
) -> tuple[str, str]:
    sentences = [
        FACT_PATTERNS[template][field].format(
            entity=entity, value=values[field]
        )
        for field in order
    ]
    return " ".join(sentences), sentences[-1]


def render_panel(
    item: dict[str, Any], operation: str, panel: str, mapping: int,
) -> dict[str, Any]:
    values0 = dict(item["values0"])
    values1 = dict(item["values1"])
    target_value = values0[operation] if mapping == 0 else values1[operation]
    if panel == "active":
        profile0, profile1 = values0, values1
        semantic_answers = (
            0 if mapping == 0 else 1,
            0 if mapping == 1 else 1,
        )
    else:
        profile0, profile1 = dict(values0), dict(values1)
        repeated_value = values0[operation]
        profile0[operation] = repeated_value
        profile1[operation] = repeated_value
        null_answer = 0 if mapping == 0 else 1
        semantic_answers = (null_answer, null_answer)
    return {
        "profile0": profile0,
        "profile1": profile1,
        "target_value": target_value,
        "semantic_answers": semantic_answers,
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
    true_word, false_word = output_code_pair(output_set)
    panel_data = render_panel(item, operation, panel, mapping)
    order = field_order(item, template)
    record0, _ = render_record(
        str(item["entity0"]), panel_data["profile0"], template, order
    )
    record1, record1_last = render_record(
        str(item["entity1"]), panel_data["profile1"], template, order
    )
    selected_entity = str(item[f"entity{query}"])
    question = QUERY_PATTERNS[template][operation].format(
        entity=selected_entity,
        value=panel_data["target_value"],
    )
    semantic_answer = int(panel_data["semantic_answers"][query])
    semantic_codes = (true_word, false_word)
    target_answer = semantic_codes[semantic_answer]
    raw_prompt = SHELLS[template].format(
        true_word=true_word,
        false_word=false_word,
        record0=record0,
        record1=record1,
        question=question,
    )
    instruction = raw_prompt.splitlines()[0]
    raw_spans = {
        "instruction_end": mark_source.mark(
            raw_prompt, instruction, occurrence="first"
        ),
        "dossier_end": mark_source.mark(
            raw_prompt, record1_last, occurrence="last"
        ),
        "query_end": mark_source.mark(
            raw_prompt, question, occurrence="last"
        ),
    }
    rendered = behavior.render_native(
        tokenizer, model_name, raw_prompt, with_system=False
    ) + ASSISTANT_PREFILL
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
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
        "schema_version": "phase1086_case.v1",
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
        "field_order_index": int(item["field_order_index"]),
        "field_order": list(order),
        "state": state,
        "template": template,
        "panel": panel,
        "mapping": mapping,
        "query": query,
        "label_swap": output_set,
        "output_set": output_set,
        "label_pair": list(semantic_codes),
        "answer_labels": list(semantic_codes),
        "selected_entity": selected_entity,
        "entity0": item["entity0"],
        "entity1": item["entity1"],
        "field": operation,
        "target_value": panel_data["target_value"],
        "profile0": panel_data["profile0"],
        "profile1": panel_data["profile1"],
        "record0": record0,
        "record1": record1,
        "question": question,
        "control_type": (
            "queried_field_matched_other_fields_retained"
            if panel == "field_null" else None
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


def audit_model(
    model_name: str,
    tokenizer,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)

    checks: dict[str, bool] = {}
    checks["complete_factorial_units"] = all(
        {row["state"] for row in rows} == set(STATES)
        for rows in by_unit.values()
    )
    checks["role_positions_valid"] = all(
        all(0 <= int(position) < len(row["input_ids"])
            for position in row["role_positions"].values())
        for row in cases
    )
    checks["query_after_dossier"] = all(
        row["role_positions"]["dossier_end"]
        < row["role_positions"]["query_end"]
        <= row["role_positions"]["answer_boundary"]
        for row in cases
    )
    checks["canonical_field_labels_absent_from_questions"] = all(
        re.search(
            rf"(?<![\w]){re.escape(row['field'].casefold())}(?![\w])",
            row["question"].casefold(),
        ) is None
        for row in cases
    )
    checks["active_answer_flips_with_entity"] = all(
        len({
            row["semantic_answer_index"] for row in rows
            if row["panel"] == "active"
            and row["template"] == 0
            and row["mapping"] == 0
            and row["output_set"] == 0
        }) == 2
        for rows in by_unit.values()
    )
    checks["field_null_answer_independent_of_entity"] = all(
        all(len({
            row["semantic_answer_index"] for row in rows
            if row["panel"] == "field_null"
            and row["template"] == template
            and row["mapping"] == mapping
            and row["output_set"] == output_set
        }) == 1
        for template in (0, 1)
        for mapping in (0, 1)
        for output_set in (0, 1))
        for rows in by_unit.values()
    )
    checks["field_null_retains_nonqueried_differences"] = all(
        all(
            row["profile0"][field] != row["profile1"][field]
            for field in OPERATIONS if field != row["operation"]
        )
        for row in cases if row["panel"] == "field_null"
    )
    checks["field_null_matches_queried_field"] = all(
        row["profile0"][row["operation"]]
        == row["profile1"][row["operation"]]
        for row in cases if row["panel"] == "field_null"
    )
    checks["output_words_single_token_in_context"] = all(
        all(len(values) >= 1 for values in row["candidate_token_ids"].values())
        for row in cases
    )
    checks["independent_entity_splits"] = all(
        {
            row["entity0"] for row in cases
            if row["world"] == world and row["split"] == "discovery"
        }.isdisjoint({
            row["entity0"] for row in cases
            if row["world"] == world and row["split"] == "confirmation"
        })
        and {
            row["entity1"] for row in cases
            if row["world"] == world and row["split"] == "discovery"
        }.isdisjoint({
            row["entity1"] for row in cases
            if row["world"] == world and row["split"] == "confirmation"
        })
        for world in WORLDS
    )

    active_dossiers: dict[tuple[str, str, str, int, int, int], set[str]] = defaultdict(set)
    for row in cases:
        if row["panel"] != "active":
            continue
        key = (
            row["world"], row["split"], row["base_item_id"],
            int(row["template"]), int(row["mapping"]), int(row["output_set"]),
        )
        active_dossiers[key].add(row["record0"] + "\n" + row["record1"])
    checks["same_active_dossier_across_attributes"] = all(
        len(values) == 1 for values in active_dossiers.values()
    )
    position_coverage: dict[str, set[int]] = defaultdict(set)
    for row in cases:
        for position, field in enumerate(row["field_order"]):
            position_coverage[field].add(position)
    checks["every_field_uses_every_dossier_position"] = all(
        len(position_coverage[field]) == len(OPERATIONS)
        for field in OPERATIONS
    )
    checks["all_checks_boolean"] = all(isinstance(value, bool) for value in checks.values())
    return {
        "schema_version": "phase1086_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "position_coverage": {
            field: sorted(values) for field, values in position_coverage.items()
        },
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
            failed = [name for name, value in audit["checks"].items() if not value]
            raise RuntimeError(f"protocol audit failed for {model_name}: {failed}")
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", cases)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_case_digests[model_name] = audit["case_digest"]
        model_audits[model_name] = audit

    source = read_json(SOURCE_PHASE1085)
    prereg = {
        "schema_version": "phase1086_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "operations": list(OPERATIONS),
        "worlds": list(WORLDS),
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "states": list(STATES),
        "capture_roles": list(CAPTURE_ROLES),
        "primary_profile_roles": list(PRIMARY_PROFILE_ROLES),
        "relative_depth_range": [
            TARGET_RELATIVE_DEPTH_MIN, TARGET_RELATIVE_DEPTH_MAX
        ],
        "signed_fields": list(SIGNED_FIELDS),
        "projection": {
            "type": "deterministic_rademacher",
            "dimension_per_replicate": SIGNED_PROJECTION_DIM,
            "replicates": SIGNED_PROJECTION_REPLICATES,
            "seed": SIGNED_PROJECTION_SEED,
            "cross_model_rule": (
                "Never compare projected coordinates across models; compare "
                "within-model operation Gram geometry only."
            ),
        },
        "items_per_cell_split": ITEMS_PER_CELL_SPLIT,
        "case_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT * len(STATES),
        "unit_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT,
        "generation_units_per_family_split": GENERATION_UNITS_PER_FAMILY_SPLIT,
        "generation_steps": GENERATION_STEPS,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "model_case_digests": model_case_digests,
        "source_phase1085_protocol_digest": source["protocol_digest"],
        "source_phase1085_summary_digest": source["summary_digest"],
        "interpretation_limits": [
            "Natural predicates remain operation carriers even though canonical field labels are absent.",
            "The field-null control removes the queried relation consequence, not all reading or judgement work.",
            "Signed random projections preserve within-model geometry approximately; they are not shared model coordinates.",
            "A shared truth-aligned field may still be a generic verification or output protocol rather than semantic storage.",
            "Attribute retrieval is evidence for a conditional residual, not a fixed neuron or context-free vector.",
            "No result establishes brain homology, optimality, or a new mathematical theory.",
        ],
        "automatic_next": {
            "stop_before_hidden_scan_if": "P1 or P2 fails.",
            "stop_full_escalation_if": "P3, P4, P5, P6, P7, P8, or P10 fails.",
            "shared_only_if": (
                "P3-P6 pass but P7-P8 fail: retain a shared operation map, "
                "without claiming attribute semantic coding."
            ),
            "causal_authorization": (
                "Only if P1-P8 and P10 pass prospectively in at least two models."
            ),
        },
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    global_audit = {
        "schema_version": "phase1086_protocol_audit.v1",
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
