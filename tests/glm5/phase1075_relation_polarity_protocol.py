#!/usr/bin/env python3
"""Freeze the Phase1075 held-out relation-polarity protocol."""

from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1051_natural_behavior_protocol as behavior
import phase1068_reasoning_generalization_protocol as name_source
import phase1074_polarity_dynamics_protocol as source


PHASE = 1075
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
RELATIONS = ("height", "age", "weight", "arrival", "score")
RELATION_PRIORITY = ("weight", "height", "age", "score", "arrival")
TASKS = ("max", "min")
PATHS = ("direct", "transitive")
LAYOUTS = ("forward", "reverse", "interleaved")
TEMPLATES = (0, 1, 2, 3)
REPLICATES = (0, 1, 2)
INTERNAL_REPLICATES = (0,)
SPLITS = ("discovery", "confirmation")
ORIENTATIONS = (0, 1)
LEXICAL_BRANCHES = (0, 1)
ASSISTANT_PREFILL = "Answer:"
NATURAL_GENERATION_STEPS = 10
MAX_INTERNAL_RELATIONS = 2

CAPTURE_ROLES = (
    "high_edge",
    "low_edge",
    "anchor_edge",
    "fact_high",
    "fact_low",
    "null_a",
    "null_b",
    "candidate_a",
    "candidate_b",
    "branch_probe",
    "task_cue",
    "operator",
    "query",
    "answer_boundary",
)
PRE_BRANCH_ROLES = (
    "high_edge",
    "low_edge",
    "anchor_edge",
    "fact_high",
    "fact_low",
    "null_a",
    "null_b",
    "candidate_a",
    "candidate_b",
    "branch_probe",
)
PRIMARY_INTERNAL_ROLES = (
    "task_cue",
    "query",
    "answer_boundary",
)
SEMANTIC_SOURCES = (
    "fact_high",
    "fact_low",
    "candidate_high",
    "candidate_low",
    "null_a",
    "null_b",
)
SOURCE_PAIRS = (
    ("fact", "fact_high", "fact_low"),
    ("candidate", "candidate_high", "candidate_low"),
    ("null_control", "null_a", "null_b"),
)
ATTENTION_DESTINATIONS = ("query", "answer_boundary")

OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1075_heldout_relation_polarity"
)
SOURCE_ROOT = source.OUT_ROOT

# Frozen before any Phase1075 model forward pass. These are independent
# confirmation gates, not post-hoc changes to the Phase1074 gates.
GATES = {
    "relation_finite_rate_min": 0.995,
    "relation_candidate_accuracy_min": 0.82,
    "relation_task_accuracy_min": 0.80,
    "relation_path_accuracy_min": 0.78,
    "relation_layout_accuracy_min": 0.78,
    "relation_lexical_accuracy_min": 0.78,
    "relation_orientation_accuracy_min": 0.78,
    "relation_confirmation_accuracy_min": 0.80,
    "relation_natural_semantic_first_min": 0.70,
    "relation_confirmation_natural_min": 0.65,
    "minimum_models_per_relation": 2,
    "internal_finite_rate_min": 0.995,
    "prebranch_interaction_relative_max": 1e-6,
    "local_selection_confirmation_positive_min": 0.65,
    "local_selection_split_profile_cosine_min": 0.50,
    "local_selection_path_profile_cosine_min": 0.30,
    "raw_interaction_split_profile_cosine_min": 0.40,
    "attention_confirmation_positive_min": 0.60,
    "attention_fact_to_null_ratio_min": 1.20,
    "minimum_internal_models_per_relation": 2,
}

RELATION_SPECS = {
    "height": {
        "positive": ("stands taller than", "has greater height than"),
        "inverse": ("stands shorter than", "has less height than"),
        "max": (
            "the endpoint that stands taller",
            "the endpoint with greater height",
        ),
        "min": (
            "the endpoint that stands shorter",
            "the endpoint with less height",
        ),
    },
    "age": {
        "positive": ("has lived longer than", "has greater age than"),
        "inverse": ("has lived fewer years than", "has less age than"),
        "max": (
            "the endpoint that has lived longer",
            "the endpoint with greater age",
        ),
        "min": (
            "the endpoint that has lived fewer years",
            "the endpoint with less age",
        ),
    },
    "weight": {
        "positive": ("outweighs", "has greater mass than"),
        "inverse": ("weighs less than", "has less mass than"),
        "max": (
            "the endpoint that outweighs the other",
            "the endpoint with greater mass",
        ),
        "min": (
            "the endpoint that weighs less",
            "the endpoint with less mass",
        ),
    },
    "arrival": {
        "positive": ("got there ahead of", "completed arrival before"),
        "inverse": ("got there behind", "completed arrival after"),
        "max": (
            "the endpoint that got there first",
            "the endpoint that completed arrival earlier",
        ),
        "min": (
            "the endpoint that got there last",
            "the endpoint that completed arrival later",
        ),
    },
    "score": {
        "positive": ("outscored", "finished with more points than"),
        "inverse": ("was outscored by", "finished with fewer points than"),
        "max": (
            "the endpoint that outscored the other",
            "the endpoint with more points",
        ),
        "min": (
            "the endpoint that was outscored",
            "the endpoint with fewer points",
        ),
    },
}

# This pool is explicit and frozen. The protocol keeps only names that are
# one continuation token in all three tokenizers and excludes every Phase1074
# name. More candidates are listed than needed so tokenizer drift fails
# transparently rather than silently reusing the old pool.
HELDOUT_NAME_CANDIDATES = (
    "Adam", "Alan", "Albert", "Alex", "Amanda", "Amy", "Andrea", "Andrew",
    "Angela", "Anna", "Anne", "Annie", "Anthony", "Antonio", "Arthur",
    "Ashley", "Audrey", "Austin", "Barbara", "Benjamin", "Betty",
    "Beverly", "Brandon", "Brenda", "Brian", "Brittany", "Bruce", "Bryan",
    "Carl", "Carolyn", "Catherine", "Charles", "Charlotte", "Cheryl",
    "Chris", "Christian", "Christina", "Christine", "Christopher", "Cindy",
    "Claire", "Clarence", "Claudia", "Craig", "Crystal", "Curtis", "Daniel",
    "Danielle", "Deborah", "Denise", "Dennis", "Derek", "Donald", "Donna",
    "Dorothy", "Douglas", "Dylan", "Edward", "Elizabeth", "Emily", "Eric",
    "Eugene", "Evelyn", "Gary", "Gloria", "Gregory", "Heather", "Irene",
    "Janet", "Jason", "Jennifer", "Jessica", "Joan", "John", "Jonathan",
    "Jordan", "Joseph", "Joshua", "Joyce", "Justin", "Keith", "Kelly",
    "Kenneth", "Kimberly", "Larry", "Linda", "Lisa", "Louis", "Louise",
    "Margaret", "Maria", "Marie", "Mark", "Martha", "Martin", "Mary",
    "Matthew", "Melissa", "Michael", "Michelle", "Nancy", "Nicholas",
    "Nicole", "Pamela", "Patricia", "Patrick", "Paul", "Philip", "Rebecca",
    "Richard", "Robert", "Ronald", "Rose", "Roy", "Russell", "Ryan",
    "Sarah", "Scott", "Sharon", "Shirley", "Sophia", "Stephanie", "Stephen",
    "Steven", "Susan", "Thomas", "Timothy", "Todd", "Walter", "Wayne",
    "William",
)

write_json = source.write_json
write_jsonl = source.write_jsonl
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest
tokenizer_for = source.tokenizer_for
offset_token_spans = source.offset_token_spans


def split_for_template(template_index: int) -> str:
    return "discovery" if template_index < 2 else "confirmation"


def phrase_set_for_template(template_index: int) -> int:
    return template_index % 2


def mark(
    text: str,
    value: str,
    start: int = 0,
) -> tuple[int, int, str]:
    position = text.find(value, start)
    if position < 0:
        raise RuntimeError(f"missing marked text: {value!r}")
    return position, position + len(value), value


def heldout_names(old_names: set[str]) -> tuple[str, ...]:
    tokenizers = {
        model: tokenizer_for(model) for model in MODELS
    }
    selected = []
    used_ids = {model: set() for model in MODELS}
    for name in HELDOUT_NAME_CANDIDATES:
        if name in old_names:
            continue
        ids = {}
        valid = True
        for model, tokenizer in tokenizers.items():
            values = tokenizer.encode(
                " " + name, add_special_tokens=False
            )
            if len(values) != 1 or int(values[0]) in used_ids[model]:
                valid = False
                break
            ids[model] = int(values[0])
        if not valid:
            continue
        selected.append(name)
        for model, token_id in ids.items():
            used_ids[model].add(token_id)
    required = len(TEMPLATES) * len(REPLICATES) * 5
    if len(selected) < required:
        raise RuntimeError(
            f"Phase1075 needs {required} held-out names, found "
            f"{len(selected)}"
        )
    return tuple(selected)


def deterministic_name_sets(
    names: tuple[str, ...],
    cell_id: str,
) -> dict[tuple[int, int], list[str]]:
    required = len(TEMPLATES) * len(REPLICATES) * 5
    ranked = sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"phase1075-heldout|{cell_id}|{name}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ranked) < required:
        raise RuntimeError("held-out name pool is too small")
    result = {}
    cursor = 0
    for template_index in TEMPLATES:
        for replicate in REPLICATES:
            result[(template_index, replicate)] = ranked[cursor:cursor + 5]
            cursor += 5
    return result


def relation_clause(
    relation: str,
    high: str,
    low: str,
    lexical_branch: int,
    phrase_set: int,
) -> str:
    spec = RELATION_SPECS[relation]
    if lexical_branch == 0:
        return f"{high} {spec['positive'][phrase_set]} {low}"
    return f"{low} {spec['inverse'][phrase_set]} {high}"


def arrange_facts(
    facts: list[tuple[str, str]],
    layout: str,
) -> tuple[str, dict[str, tuple[int, int, str]]]:
    if layout == "forward":
        order = (0, 1, 2)
    elif layout == "reverse":
        order = (2, 1, 0)
    elif layout == "interleaved":
        order = (0, 2, 1)
    else:
        raise ValueError(f"unknown layout: {layout}")
    pieces = []
    spans = {}
    cursor = 0
    for displayed_index, source_index in enumerate(order):
        if displayed_index:
            pieces.append(". ")
            cursor += 2
        role, value = facts[source_index]
        start = cursor
        pieces.append(value)
        cursor += len(value)
        spans[role] = (start, cursor, value)
    pieces.append(".")
    return "".join(pieces), spans


def task_tail(
    relation: str,
    task: str,
    template_index: int,
) -> tuple[str, str, str]:
    cue = RELATION_SPECS[relation][task][
        phrase_set_for_template(template_index)
    ]
    if template_index == 0:
        operator = "Select"
        tail = f"{operator} {cue}. Output the selected name only."
    elif template_index == 1:
        operator = "Apply"
        tail = (
            f"{operator} this criterion: {cue}. Reply with one endpoint "
            "name."
        )
    elif template_index == 2:
        operator = "Write"
        tail = (
            f"Your criterion is {cue}. {operator} exactly the matching "
            "endpoint."
        )
    else:
        operator = "Choose"
        tail = (
            f"{operator} according to this request: {cue}. State one name "
            "and nothing else."
        )
    return tail, cue, operator


def render_prompt(
    relation: str,
    task: str,
    path: str,
    layout: str,
    names: list[str],
    orientation: int,
    lexical_branch: int,
    template_index: int,
) -> tuple[
    str,
    dict[str, tuple[int, int, str]],
    dict[str, tuple[int, int, str]],
    dict[str, list[str]],
    dict[str, Any],
]:
    endpoint_a, middle, endpoint_b, null_a, null_b = names
    high, low = (
        (endpoint_a, endpoint_b)
        if orientation == 0
        else (endpoint_b, endpoint_a)
    )
    phrase_set = phrase_set_for_template(template_index)
    if path == "direct":
        main = relation_clause(
            relation, high, low, lexical_branch, phrase_set
        )
        context = relation_clause(
            relation, middle, null_a, lexical_branch, phrase_set
        )
        high_role = low_role = "main_edge"
        facts = [
            ("main_edge", main),
            ("context_edge", context),
            (
                "anchor_edge",
                relation_clause(
                    relation, null_a, null_b, lexical_branch, phrase_set
                ),
            ),
        ]
    elif path == "transitive":
        upper = relation_clause(
            relation, high, middle, lexical_branch, phrase_set
        )
        lower = relation_clause(
            relation, middle, low, lexical_branch, phrase_set
        )
        high_role, low_role = "upper_edge", "lower_edge"
        facts = [
            ("upper_edge", upper),
            ("lower_edge", lower),
            (
                "anchor_edge",
                relation_clause(
                    relation, null_a, null_b, lexical_branch, phrase_set
                ),
            ),
        ]
    else:
        raise ValueError(f"unknown path: {path}")

    facts_text, fact_spans = arrange_facts(facts, layout)
    candidate_text = f"Endpoints: {endpoint_a} and {endpoint_b}"
    branch_probe = "Decision begins now"
    tail, cue, operator = task_tail(relation, task, template_index)
    raw_prompt = (
        f"Evidence: {facts_text} {candidate_text}. {branch_probe}. {tail}"
    )
    facts_start = raw_prompt.find(facts_text)
    raw_spans = {
        role: (
            facts_start + span[0],
            facts_start + span[1],
            span[2],
        )
        for role, span in fact_spans.items()
    }
    raw_spans["high_edge"] = raw_spans[high_role]
    raw_spans["low_edge"] = raw_spans[low_role]
    high_start = raw_spans["high_edge"][0]
    low_start = raw_spans["low_edge"][0]
    anchor_start = raw_spans["anchor_edge"][0]
    raw_spans["fact_high"] = mark(raw_prompt, high, high_start)
    raw_spans["fact_low"] = mark(raw_prompt, low, low_start)
    raw_spans["null_a"] = mark(raw_prompt, null_a, anchor_start)
    raw_spans["null_b"] = mark(raw_prompt, null_b, anchor_start)

    candidate_span = mark(raw_prompt, candidate_text)
    a_offset = candidate_text.find(endpoint_a)
    b_offset = candidate_text.find(
        endpoint_b, a_offset + len(endpoint_a)
    )
    raw_spans["candidate_a"] = (
        candidate_span[0] + a_offset,
        candidate_span[0] + a_offset + len(endpoint_a),
        endpoint_a,
    )
    raw_spans["candidate_b"] = (
        candidate_span[0] + b_offset,
        candidate_span[0] + b_offset + len(endpoint_b),
        endpoint_b,
    )
    raw_spans["branch_probe"] = mark(raw_prompt, branch_probe)
    raw_spans["task_cue"] = mark(
        raw_prompt, cue, raw_spans["branch_probe"][1]
    )
    raw_spans["operator"] = mark(
        raw_prompt, operator, raw_spans["branch_probe"][1]
    )
    query_start = min(
        raw_spans["operator"][0], raw_spans["task_cue"][0]
    )
    query_text = raw_prompt[query_start:].rstrip()
    raw_spans["query"] = (
        query_start,
        query_start + len(query_text),
        query_text,
    )

    semantic_spans = {
        "fact_high": raw_spans["fact_high"],
        "fact_low": raw_spans["fact_low"],
        "candidate_high": (
            raw_spans["candidate_a"]
            if high == endpoint_a
            else raw_spans["candidate_b"]
        ),
        "candidate_low": (
            raw_spans["candidate_b"]
            if low == endpoint_b
            else raw_spans["candidate_a"]
        ),
        "null_a": raw_spans["null_a"],
        "null_b": raw_spans["null_b"],
    }
    expected_answer = high if task == "max" else low
    expected_class = "b0" if expected_answer == endpoint_a else "b1"
    classes = {"b0": [endpoint_a], "b1": [endpoint_b]}
    return (
        raw_prompt,
        raw_spans,
        semantic_spans,
        classes,
        {
            "endpoint_a": endpoint_a,
            "endpoint_b": endpoint_b,
            "middle": middle,
            "null_a": null_a,
            "null_b": null_b,
            "high": high,
            "low": low,
            "expected_answer": expected_answer,
            "expected_class": expected_class,
            "task_cue": cue,
            "query_text": query_text,
            "facts_text": facts_text,
        },
    )


def skeleton_hash(raw_prompt: str, names: list[str]) -> str:
    normalized = raw_prompt
    for index, name in sorted(
        enumerate(names), key=lambda item: len(item[1]), reverse=True
    ):
        normalized = normalized.replace(name, f"<N{index}>")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def encode_case(
    tokenizer,
    model_name: str,
    names: tuple[str, ...],
    relation: str,
    task: str,
    path: str,
    layout: str,
    template_index: int,
    replicate: int,
    orientation: int,
    lexical_branch: int,
    semantic_case_index: int,
) -> dict[str, Any]:
    cell_id = f"{relation}.{path}.{layout}"
    cell_names = deterministic_name_sets(names, cell_id)[
        (template_index, replicate)
    ]
    (
        raw_prompt,
        raw_spans,
        semantic_raw_spans,
        classes,
        metadata,
    ) = render_prompt(
        relation,
        task,
        path,
        layout,
        cell_names,
        orientation,
        lexical_branch,
        template_index,
    )
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
    semantic_source_spans = offset_token_spans(
        tokenizer, rendered, raw_prompt, semantic_raw_spans
    )
    role_spans["answer_boundary"] = (
        len(input_ids) - 1,
        len(input_ids) - 1,
    )
    candidate_token_ids = {
        class_name: [
            behavior.continuation_ids(tokenizer, rendered, " ", label)
            for label in labels
        ]
        for class_name, labels in classes.items()
    }
    candidate_first_token_ids = {
        class_name: sorted({
            int(values[0]) for values in tokenizations
        })
        for class_name, tokenizations in candidate_token_ids.items()
    }
    pair_id = (
        f"{relation}.{path}.{layout}.t{template_index}.r{replicate}."
        f"o{orientation}.l{lexical_branch}"
    )
    unit_id = (
        f"{relation}.{path}.{layout}.t{template_index}.r{replicate}"
    )
    return {
        "schema_version": "phase1075_heldout_polarity_case.v1",
        "phase": PHASE,
        "model": model_name,
        "semantic_case_index": semantic_case_index,
        "record_id": f"phase1075.{model_name}.{pair_id}.{task}",
        "pair_id": pair_id,
        "unit_id": unit_id,
        "relation": relation,
        "task": task,
        "path": path,
        "layout": layout,
        "template_index": template_index,
        "replicate": replicate,
        "split": split_for_template(template_index),
        "phrase_set": phrase_set_for_template(template_index),
        "orientation": orientation,
        "lexical_branch": lexical_branch,
        "cell_names": list(cell_names),
        "semantic_names": {
            key: metadata[key]
            for key in (
                "endpoint_a",
                "endpoint_b",
                "middle",
                "null_a",
                "null_b",
                "high",
                "low",
            )
        },
        "expected_answer": metadata["expected_answer"],
        "expected_class": metadata["expected_class"],
        "acceptable_labels": classes[metadata["expected_class"]],
        "candidate_labels": classes,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": candidate_first_token_ids,
        "facts_text": metadata["facts_text"],
        "task_cue_text": metadata["task_cue"],
        "query_text": metadata["query_text"],
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "prompt_skeleton_sha256": skeleton_hash(
            raw_prompt, cell_names
        ),
        "input_ids": input_ids,
        "role_spans": {
            role: [int(span[0]), int(span[1])]
            for role, span in role_spans.items()
        },
        "role_positions": {
            role: int(span[1]) for role, span in role_spans.items()
        },
        "semantic_source_spans": {
            role: [int(span[0]), int(span[1])]
            for role, span in semantic_source_spans.items()
        },
        "continuation_prefix": " ",
    }


def audit_model(
    model_name: str,
    cases: list[dict[str, Any]],
    old_names: set[str],
    old_skeletons: set[str],
) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_pair[str(row["pair_id"])].append(row)
        by_unit[str(row["unit_id"])].append(row)

    exact_prefix = True
    pair_equal = True
    answer_conflict = True
    for rows in by_pair.values():
        if len(rows) != len(TASKS):
            exact_prefix = pair_equal = answer_conflict = False
            continue
        prefixes = []
        for row in rows:
            end = int(row["role_positions"]["branch_probe"]) + 1
            prefixes.append(tuple(row["input_ids"][:end]))
        exact_prefix &= len(set(prefixes)) == 1
        pair_equal &= (
            len({row["facts_text"] for row in rows}) == 1
            and len({tuple(row["cell_names"]) for row in rows}) == 1
            and len({row["orientation"] for row in rows}) == 1
            and len({row["lexical_branch"] for row in rows}) == 1
        )
        answer_conflict &= len({
            row["expected_answer"] for row in rows
        }) == 2

    expected_cases = (
        len(RELATIONS)
        * len(TASKS)
        * len(PATHS)
        * len(LAYOUTS)
        * len(TEMPLATES)
        * len(REPLICATES)
        * len(ORIENTATIONS)
        * len(LEXICAL_BRANCHES)
    )
    expected_pairs = expected_cases // len(TASKS)
    expected_units = (
        len(RELATIONS)
        * len(PATHS)
        * len(LAYOUTS)
        * len(TEMPLATES)
        * len(REPLICATES)
    )
    balanced = Counter(
        (
            row["relation"],
            row["task"],
            row["path"],
            row["layout"],
            row["split"],
        )
        for row in cases
    )
    checks = {
        "case_count": len(cases) == expected_cases,
        "pair_count": len(by_pair) == expected_pairs,
        "unit_count": len(by_unit) == expected_units,
        "complete_units": all(len(rows) == 8 for rows in by_unit.values()),
        "balanced_cells": all(
            balanced[(relation, task, path, layout, split)] == 24
            for relation in RELATIONS
            for task in TASKS
            for path in PATHS
            for layout in LAYOUTS
            for split in SPLITS
        ),
        "exact_identical_prefix_through_branch_probe": exact_prefix,
        "pair_fields_equal_before_task": pair_equal,
        "task_answers_conflict_in_every_pair": answer_conflict,
        "role_spans_valid": all(
            all(
                0 <= int(row["role_spans"][role][0])
                <= int(row["role_spans"][role][1])
                < len(row["input_ids"])
                for role in CAPTURE_ROLES
            )
            for row in cases
        ),
        "semantic_source_spans_valid": all(
            all(
                0 <= int(row["semantic_source_spans"][role][0])
                <= int(row["semantic_source_spans"][role][1])
                < len(row["input_ids"])
                for role in SEMANTIC_SOURCES
            )
            for row in cases
        ),
        "candidate_continuations_single_token": all(
            len(tokenization) == 1
            for row in cases
            for tokenizations in row["candidate_token_ids"].values()
            for tokenization in tokenizations
        ),
        "candidate_first_tokens_disjoint": all(
            set(row["candidate_first_token_ids"]["b0"]).isdisjoint(
                row["candidate_first_token_ids"]["b1"]
            )
            for row in cases
        ),
        "heldout_names_disjoint_from_phase1074": not any(
            name in old_names
            for row in cases
            for name in row["cell_names"]
        ),
        "prompt_skeletons_disjoint_from_phase1074": not any(
            row["prompt_skeleton_sha256"] in old_skeletons
            for row in cases
        ),
        "task_answer_valid": all(
            row["expected_answer"] == (
                row["semantic_names"]["high"]
                if row["task"] == "max"
                else row["semantic_names"]["low"]
            )
            for row in cases
        ),
        "orientation_truth_table_valid": all(
            (
                row["semantic_names"]["high"]
                == row["semantic_names"]["endpoint_a"]
            )
            == (int(row["orientation"]) == 0)
            for row in cases
        ),
        "task_cue_strictly_after_common_prefix": all(
            int(row["role_positions"]["task_cue"])
            > int(row["role_positions"]["branch_probe"])
            for row in cases
        ),
    }
    return {
        "schema_version": "phase1075_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "pair_count": len(by_pair),
        "unit_count": len(by_unit),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def build_protocol() -> dict[str, Any]:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_decision = read_json(
        SOURCE_ROOT / "analysis" / "automatic_next.json"
    )
    source_audit = read_json(
        SOURCE_ROOT / "analysis" / "integrity_audit.json"
    )
    if (
        source_decision["route"] != "stop_at_behavior_foundation"
        or bool(source_decision["should_continue_automatically"])
        or not source_audit["all_integrity_checks_passed"]
    ):
        raise RuntimeError("Phase1074 source state drift")

    old_names = set(source_prereg["cross_model_single_token_names"])
    old_skeletons = {
        str(row["prompt_skeleton_sha256"])
        for model in MODELS
        for row in read_jsonl(
            SOURCE_ROOT / "protocol" / f"cases.{model}.jsonl"
        )
    }
    names = heldout_names(old_names)
    model_audits = {}
    expected_cases = 2880
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        cases = []
        semantic_case_index = 0
        for relation in RELATIONS:
            for path in PATHS:
                for layout in LAYOUTS:
                    for template_index in TEMPLATES:
                        for replicate in REPLICATES:
                            for orientation in ORIENTATIONS:
                                for lexical_branch in LEXICAL_BRANCHES:
                                    for task in TASKS:
                                        cases.append(encode_case(
                                            tokenizer,
                                            model_name,
                                            names,
                                            relation,
                                            task,
                                            path,
                                            layout,
                                            template_index,
                                            replicate,
                                            orientation,
                                            lexical_branch,
                                            semantic_case_index,
                                        ))
                                        semantic_case_index += 1
        if len(cases) != expected_cases:
            raise RuntimeError("Phase1075 case count drift")
        audit = audit_model(
            model_name, cases, old_names, old_skeletons
        )
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1075 audit failed for {model_name}: {audit}"
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

    payload = {
        "schema_version": "phase1075_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "relations": list(RELATIONS),
        "relation_priority": list(RELATION_PRIORITY),
        "tasks": list(TASKS),
        "paths": list(PATHS),
        "layouts": list(LAYOUTS),
        "templates": list(TEMPLATES),
        "replicates": list(REPLICATES),
        "internal_replicates": list(INTERNAL_REPLICATES),
        "splits": list(SPLITS),
        "orientations": list(ORIENTATIONS),
        "lexical_branches": list(LEXICAL_BRANCHES),
        "capture_roles": list(CAPTURE_ROLES),
        "pre_branch_roles": list(PRE_BRANCH_ROLES),
        "primary_internal_roles": list(PRIMARY_INTERNAL_ROLES),
        "semantic_sources": list(SEMANTIC_SOURCES),
        "source_pairs": [list(value) for value in SOURCE_PAIRS],
        "attention_destinations": list(ATTENTION_DESTINATIONS),
        "case_count_per_model": expected_cases,
        "pair_count_per_model": expected_cases // 2,
        "unit_count_per_model": 360,
        "internal_cases_per_relation_model": 192,
        "internal_units_per_relation_model": 24,
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "assistant_prefill": ASSISTANT_PREFILL,
        "heldout_name_count": len(names),
        "heldout_single_token_names": list(names),
        "max_internal_relations": MAX_INTERNAL_RELATIONS,
        "gates": dict(GATES),
        "source_phase1074_digest": source_prereg["protocol_digest"],
        "source_phase1074_decision": source_decision,
        "measurement_definitions": {
            "raw_task_contrast": "Q_o = h[max,o] - h[min,o]",
            "raw_selection_interaction": (
                "I_h = (h[max,o0]-h[min,o0]) - "
                "(h[max,o1]-h[min,o1])"
            ),
            "layer_interaction_write": (
                "I_u[d] = I_h[d+1] - I_h[d]"
            ),
            "local_high_low_margin": (
                "m[tau,o,d,r] = logit(high|h) - logit(low|h)"
            ),
            "local_selection_separation": (
                "S[d,r] = mean_o(m[max,o,d,r]-m[min,o,d,r])/2"
            ),
            "attention_route_selectivity": (
                "R_A = mean_o[(A[max,high]-A[max,low]) - "
                "(A[min,high]-A[min,low])]"
            ),
        },
        "authorization_order": [
            "freeze held-out names, templates, phrases, factors, and gates",
            "verify complete separation from Phase1074 names and skeletons",
            "run all three models sequentially in FP16 without quantization",
            "authorize each model-relation cell independently",
            "promote only relations independently confirmed in at least two models",
            "select at most two relations by the frozen priority order",
            "scan only authorized model-relation cells",
            "treat raw interaction, local readout, and Attention routing as separate observations",
            "select internal candidates on discovery templates",
            "evaluate all internal claims on confirmation templates",
        ],
        "interpretation_limits": [
            "Behavioral max/min competence does not prove an internal operation module.",
            "I_h is a factorial interaction, not a recovered mechanism equation.",
            "A local logit-lens margin is an observer coordinate, not the native code.",
            "Attention routing is not causal necessity or sufficiency.",
            "A repeated relation-level profile is not neuron or head homology.",
            "No result can establish brain homology, evolutionary optimality, or a complete language theory.",
        ],
        "hypotheses_under_test": [
            "At least one ordered relation repeats on fully held-out names and prompt skeletons in two or more models.",
            "A valid relation shows a post-cue high-versus-low selection separation that is absent before the late cue.",
            "Some physical writes or source-routing responses repeat across templates and models without requiring a single fixed neuron.",
            "Shared relation evidence and opposite endpoint selection can reuse components while differing in conditional readout.",
        ],
        "automatic_next": {
            "internal_stage": (
                "Run internal mapping only for relations that pass fresh "
                "behavior gates in at least two models."
            ),
            "causal_stage": (
                "Recommend a new frozen causal phase only if one relation "
                "passes held-out local-selection and Attention-routing "
                "gates in at least two models."
            ),
        },
        "model_audits": model_audits,
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        OUT_ROOT / "protocol" / "preregistration.json", payload
    )
    write_json(
        OUT_ROOT / "protocol" / "audit.json",
        {
            "schema_version": "phase1075_protocol_audit.v1",
            "phase": PHASE,
            "protocol_digest": payload["protocol_digest"],
            "model_audits": model_audits,
            "all_checks_passed": all(
                audit["all_checks_passed"]
                for audit in model_audits.values()
            ),
        },
    )
    return payload


def main() -> None:
    payload = build_protocol()
    print(
        f"Phase{PHASE} protocol {payload['protocol_digest']} "
        f"cases={payload['case_count_per_model']}/model "
        f"heldout_names={payload['heldout_name_count']}"
    )


if __name__ == "__main__":
    main()
