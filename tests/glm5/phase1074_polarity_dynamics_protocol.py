#!/usr/bin/env python3
"""Freeze the Phase1074 late-polarity system-identification protocol."""

from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1051_natural_behavior_protocol as behavior
import phase1070_process_answer_protocol as factorial
import phase1073_late_query_protocol as source


PHASE = 1074
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
RELATIONS = ("height", "age", "weight", "arrival", "score")
TASKS = ("max", "min")
PATHS = ("direct", "transitive")
LAYOUTS = ("forward", "reverse")
TEMPLATES = (0, 1, 2, 3)
REPLICATES = (0, 1)
INTERNAL_REPLICATES = (0,)
SPLITS = ("discovery", "confirmation")
ORIENTATIONS = (0, 1)
LEXICAL_BRANCHES = (0, 1)
ASSISTANT_PREFILL = "Answer:"
NATURAL_GENERATION_STEPS = 10

CAPTURE_ROLES = (
    "upper_edge",
    "switch_edge",
    "lower_edge",
    "anchor_edge",
    "fact_high",
    "fact_low",
    "query_candidate_a",
    "query_candidate_b",
    "branch_probe",
    "task_cue",
    "operator",
    "query",
    "answer_boundary",
)
PRE_BRANCH_ROLES = (
    "upper_edge",
    "switch_edge",
    "lower_edge",
    "anchor_edge",
    "fact_high",
    "fact_low",
    "query_candidate_a",
    "query_candidate_b",
    "branch_probe",
)
PRIMARY_DYNAMIC_ROLES = (
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
ATTENTION_DESTINATIONS = ("task_cue", "answer_boundary")

OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1074_late_polarity_dynamics"
)
SOURCE_ROOT = source.OUT_ROOT

# Frozen before any Phase1074 model forward pass.
GATES = {
    "candidate_finite_rate_min": 0.995,
    "overall_candidate_accuracy_min": 0.86,
    "per_task_candidate_accuracy_min": 0.82,
    "confirmation_candidate_accuracy_min": 0.82,
    "per_path_candidate_accuracy_min": 0.80,
    "relation_task_candidate_accuracy_min": 0.76,
    "semantic_first_natural_rate_min": 0.75,
    "confirmation_semantic_first_rate_min": 0.70,
    "minimum_strong_relations_per_model": 3,
    "minimum_behavior_models": 2,
    "internal_finite_rate_min": 0.995,
    "prebranch_selection_interaction_max": 1e-6,
    "embedding_selection_interaction_max": 1e-5,
    "selection_interaction_relative_magnitude_min": 0.002,
    "transition_interaction_relative_magnitude_min": 0.001,
    "selection_lexical_reuse_cosine_min": 0.20,
    "selection_split_profile_cosine_min": 0.55,
    "selection_path_profile_cosine_min": 0.30,
    "attention_confirmation_positive_fraction_min": 0.65,
    "attention_fact_to_null_effect_ratio_min": 1.25,
    "attention_selected_head_count": 8,
    "minimum_dynamic_relations_per_model": 2,
    "minimum_dynamic_models": 2,
}

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
    return 0 if template_index < 2 else 1


def mark(
    text: str,
    value: str,
    start: int = 0,
) -> tuple[int, int, str]:
    position = text.find(value, start)
    if position < 0:
        raise RuntimeError(f"missing marked text: {value!r}")
    return position, position + len(value), value


def deterministic_name_sets(
    names: tuple[str, ...],
    cell_id: str,
) -> dict[tuple[int, int], list[str]]:
    required = len(TEMPLATES) * len(REPLICATES) * 6
    if len(names) < required:
        raise RuntimeError(
            f"{cell_id} needs {required} names, found {len(names)}"
        )
    ranked = sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"phase1074|{cell_id}|{name}".encode("utf-8")
        ).hexdigest(),
    )
    result: dict[tuple[int, int], list[str]] = {}
    cursor = 0
    for template_index in TEMPLATES:
        for replicate in REPLICATES:
            result[(template_index, replicate)] = ranked[
                cursor:cursor + 6
            ]
            cursor += 6
    return result


def compose_facts(
    tagged: list[tuple[str, str]],
    layout: str,
) -> tuple[str, dict[str, tuple[int, int, str]]]:
    displayed = tagged if layout == "forward" else list(reversed(tagged))
    pieces: list[str] = []
    spans: dict[str, tuple[int, int, str]] = {}
    cursor = 0
    for index, (role, text) in enumerate(displayed):
        if index:
            pieces.append(". ")
            cursor += 2
        start = cursor
        pieces.append(text)
        cursor += len(text)
        spans[role] = (start, cursor, text)
    pieces.append(".")
    return "".join(pieces), spans


def task_tail(
    relation: str,
    task: str,
    template_index: int,
) -> tuple[str, str, str]:
    phrase_set = phrase_set_for_template(template_index)
    cue = factorial.RELATIONS[relation][
        f"{task}_query"
    ][phrase_set]
    if template_index == 0:
        query = f"Between these candidates, {cue}?"
        operator = "Return"
        tail = (
            f"Task: {query} {operator} exactly one candidate name "
            "and nothing else."
        )
    elif template_index == 1:
        query = f"Of these two candidates, {cue}?"
        operator = "Give"
        tail = (
            f"Question: {query} {operator} one name only, with no "
            "explanation."
        )
    elif template_index == 2:
        query = f"From the evidence, decide {cue}."
        operator = "Write"
        tail = f"Instruction: {query} {operator} just the candidate name."
    else:
        query = f"For the two candidates, {cue}?"
        operator = "Respond"
        tail = (
            f"Decision: {query} {operator} with exactly one candidate "
            "name."
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
    (
        endpoint_a,
        middle_a,
        middle_b,
        endpoint_b,
        null_a,
        null_b,
    ) = names
    if orientation == 0:
        high, low = endpoint_a, endpoint_b
    else:
        high, low = endpoint_b, endpoint_a
    phrase_set = phrase_set_for_template(template_index)
    upper = factorial.relation_clause(
        relation, high, middle_a, lexical_branch, phrase_set
    )
    lower = factorial.relation_clause(
        relation, middle_b, low, lexical_branch, phrase_set
    )
    if path == "direct":
        switch = factorial.relation_clause(
            relation, high, low, lexical_branch, phrase_set
        )
    elif path == "transitive":
        switch = factorial.relation_clause(
            relation, middle_a, middle_b, lexical_branch, phrase_set
        )
    else:
        raise ValueError(f"unknown path: {path}")
    anchor = factorial.relation_clause(
        relation, null_a, null_b, lexical_branch, phrase_set
    )
    tagged = [
        ("upper_edge", upper),
        ("switch_edge", switch),
        ("lower_edge", lower),
        ("anchor_edge", anchor),
    ]
    facts, fact_spans = compose_facts(tagged, layout)
    candidate_text = f"Candidates: {endpoint_a} and {endpoint_b}"
    branch_probe = "Operation marker"
    tail, cue, operator = task_tail(relation, task, template_index)
    raw_prompt = (
        f"Facts: {facts} {candidate_text}. {branch_probe}. {tail}"
    )
    facts_start = raw_prompt.find(facts)
    if facts_start < 0:
        raise RuntimeError("facts missing from raw prompt")
    raw_spans = {
        role: (
            facts_start + span[0],
            facts_start + span[1],
            span[2],
        )
        for role, span in fact_spans.items()
    }

    upper_start = raw_spans["upper_edge"][0]
    lower_start = raw_spans["lower_edge"][0]
    anchor_start = raw_spans["anchor_edge"][0]
    raw_spans["fact_high"] = mark(raw_prompt, high, upper_start)
    raw_spans["fact_low"] = mark(raw_prompt, low, lower_start)
    raw_spans["null_a"] = mark(raw_prompt, null_a, anchor_start)
    raw_spans["null_b"] = mark(raw_prompt, null_b, anchor_start)

    candidate_span = mark(raw_prompt, candidate_text)
    endpoint_a_offset = candidate_text.find(endpoint_a)
    endpoint_b_offset = candidate_text.find(
        endpoint_b, endpoint_a_offset + len(endpoint_a)
    )
    raw_spans["query_candidate_a"] = (
        candidate_span[0] + endpoint_a_offset,
        candidate_span[0] + endpoint_a_offset + len(endpoint_a),
        endpoint_a,
    )
    raw_spans["query_candidate_b"] = (
        candidate_span[0] + endpoint_b_offset,
        candidate_span[0] + endpoint_b_offset + len(endpoint_b),
        endpoint_b,
    )
    raw_spans["branch_probe"] = mark(raw_prompt, branch_probe)
    raw_spans["task_cue"] = mark(
        raw_prompt, cue, raw_spans["branch_probe"][1]
    )
    raw_spans["operator"] = mark(
        raw_prompt, operator, raw_spans["task_cue"][1]
    )
    query_start = raw_prompt.rfind(
        tail.split(": ", 1)[-1].split(f" {operator}", 1)[0]
    )
    if query_start < 0:
        raise RuntimeError("query span drift")
    query_text = raw_prompt[
        query_start:raw_spans["operator"][0]
    ].rstrip()
    raw_spans["query"] = (
        query_start,
        query_start + len(query_text),
        query_text,
    )

    semantic_spans = {
        "fact_high": raw_spans["fact_high"],
        "fact_low": raw_spans["fact_low"],
        "candidate_high": (
            raw_spans["query_candidate_a"]
            if high == endpoint_a
            else raw_spans["query_candidate_b"]
        ),
        "candidate_low": (
            raw_spans["query_candidate_b"]
            if low == endpoint_b
            else raw_spans["query_candidate_a"]
        ),
        "null_a": raw_spans["null_a"],
        "null_b": raw_spans["null_b"],
    }
    expected_answer = high if task == "max" else low
    expected_class = (
        "b0" if expected_answer == endpoint_a else "b1"
    )
    classes = {"b0": [endpoint_a], "b1": [endpoint_b]}
    metadata = {
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "middle_a": middle_a,
        "middle_b": middle_b,
        "null_a": null_a,
        "null_b": null_b,
        "high": high,
        "low": low,
        "expected_answer": expected_answer,
        "expected_class": expected_class,
        "task_cue": cue,
        "query_text": query_text,
        "facts_text": facts,
    }
    return raw_prompt, raw_spans, semantic_spans, classes, metadata


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
        for value in tokenizer.encode(
            rendered, add_special_tokens=False
        )
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
            behavior.continuation_ids(
                tokenizer, rendered, " ", label
            )
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
        "schema_version": "phase1074_late_polarity_case.v1",
        "phase": PHASE,
        "model": model_name,
        "semantic_case_index": semantic_case_index,
        "record_id": f"phase1074.{model_name}.{pair_id}.{task}",
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
                "middle_a",
                "middle_b",
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
) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_pair[str(row["pair_id"])].append(row)
        by_unit[str(row["unit_id"])].append(row)

    exact_prefix = True
    pair_fields_equal = True
    conflicting_answers = True
    shared_width_possible = True
    for rows in by_pair.values():
        if len(rows) != len(TASKS):
            exact_prefix = False
            pair_fields_equal = False
            conflicting_answers = False
            continue
        rows = sorted(rows, key=lambda row: TASKS.index(row["task"]))
        prefixes = []
        for row in rows:
            end = int(row["role_positions"]["branch_probe"]) + 1
            prefixes.append(tuple(row["input_ids"][:end]))
        exact_prefix = exact_prefix and len(set(prefixes)) == 1
        pair_fields_equal = (
            pair_fields_equal
            and len({row["facts_text"] for row in rows}) == 1
            and len({tuple(row["cell_names"]) for row in rows}) == 1
            and len({row["orientation"] for row in rows}) == 1
            and len({row["lexical_branch"] for row in rows}) == 1
        )
        conflicting_answers = (
            conflicting_answers
            and len({row["expected_answer"] for row in rows}) == 2
        )
        shared_width_possible = (
            shared_width_possible
            and all(len(row["input_ids"]) > 0 for row in rows)
        )

    role_spans_valid = all(
        all(
            0 <= int(row["role_spans"][role][0])
            <= int(row["role_spans"][role][1])
            < len(row["input_ids"])
            for role in CAPTURE_ROLES
        )
        for row in cases
    )
    semantic_spans_valid = all(
        all(
            0 <= int(row["semantic_source_spans"][role][0])
            <= int(row["semantic_source_spans"][role][1])
            < len(row["input_ids"])
            for role in SEMANTIC_SOURCES
        )
        for row in cases
    )
    candidate_single_token = all(
        len(tokenization) == 1
        for row in cases
        for tokenizations in row["candidate_token_ids"].values()
        for tokenization in tokenizations
    )
    candidate_disjoint = all(
        set(row["candidate_first_token_ids"]["b0"]).isdisjoint(
            row["candidate_first_token_ids"]["b1"]
        )
        for row in cases
    )
    task_answer_valid = all(
        row["expected_answer"] == (
            row["semantic_names"]["high"]
            if row["task"] == "max"
            else row["semantic_names"]["low"]
        )
        for row in cases
    )
    orientation_valid = all(
        (
            row["semantic_names"]["high"]
            == row["semantic_names"]["endpoint_a"]
        )
        == (int(row["orientation"]) == 0)
        for row in cases
    )
    branch_after_prefix = all(
        int(row["role_positions"]["task_cue"])
        > int(row["role_positions"]["branch_probe"])
        for row in cases
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
        "case_count": len(cases) == 1280,
        "pair_count": len(by_pair) == 640,
        "unit_count": len(by_unit) == 160,
        "complete_units": all(len(rows) == 8 for rows in by_unit.values()),
        "balanced_cells": all(
            balanced[(relation, task, path, layout, split)] == 16
            for relation in RELATIONS
            for task in TASKS
            for path in PATHS
            for layout in LAYOUTS
            for split in SPLITS
        ),
        "exact_identical_prefix_through_branch_probe": exact_prefix,
        "pair_fields_equal_before_task": pair_fields_equal,
        "task_answers_conflict_in_every_pair": conflicting_answers,
        "shared_width_execution_possible": shared_width_possible,
        "role_spans_valid": role_spans_valid,
        "semantic_source_spans_valid": semantic_spans_valid,
        "candidate_continuations_single_token": candidate_single_token,
        "candidate_first_tokens_disjoint": candidate_disjoint,
        "task_answer_valid": task_answer_valid,
        "orientation_truth_table_valid": orientation_valid,
        "task_cue_strictly_after_common_prefix": branch_after_prefix,
    }
    return {
        "schema_version": "phase1074_protocol_model_audit.v1",
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
    source_next = read_json(
        SOURCE_ROOT / "analysis" / "automatic_next.json"
    )
    source_audit = read_json(
        SOURCE_ROOT / "analysis" / "integrity_audit.json"
    )
    if (
        bool(source_next["should_continue_automatically"])
        or source_next["route"]
        != "stop_at_late_query_operation_selection"
        or not source_audit["all_integrity_checks_passed"]
    ):
        raise RuntimeError("Phase1073 source decision or audit drift")
    phase1070_prereg = read_json(
        factorial.OUT_ROOT / "protocol" / "preregistration.json"
    )
    names = tuple(
        str(value)
        for value in phase1070_prereg[
            "cross_model_single_token_names"
        ]
    )
    if len(names) != 48:
        raise RuntimeError("Phase1074 requires the frozen 48-name pool")

    model_audits = {}
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
        audit = audit_model(model_name, cases)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1074 audit failed for {model_name}: {audit}"
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
        "schema_version": "phase1074_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "relations": list(RELATIONS),
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
        "primary_dynamic_roles": list(PRIMARY_DYNAMIC_ROLES),
        "semantic_sources": list(SEMANTIC_SOURCES),
        "source_pairs": [list(value) for value in SOURCE_PAIRS],
        "attention_destinations": list(ATTENTION_DESTINATIONS),
        "case_count_per_model": 1280,
        "pair_count_per_model": 640,
        "unit_count_per_model": 160,
        "internal_case_count_per_model": 640,
        "internal_unit_count_per_model": 80,
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "assistant_prefill": ASSISTANT_PREFILL,
        "cross_model_single_token_names": list(names),
        "gates": dict(GATES),
        "source_phase1073_digest": source_prereg[
            "protocol_digest"
        ],
        "source_phase1073_decision": source_next,
        "measurement_definitions": {
            "task_contrast": "Q_o = h[max,o] - h[min,o]",
            "selection_interaction": (
                "I_h = (h[max,o0]-h[min,o0]) - "
                "(h[max,o1]-h[min,o1])"
            ),
            "layer_update": "u[tau,o,d] = h[tau,o,d+1] - h[tau,o,d]",
            "transition_interaction": (
                "I_u[d] = I_h[d+1] - I_h[d]"
            ),
            "attention_route_selectivity": (
                "R_A = (A[max,high]-A[max,low]) - "
                "(A[min,high]-A[min,low])"
            ),
        },
        "measurement_order": [
            "freeze all prompts, names, factors, gates, and splits",
            "verify every max/min pair has an exact token-identical prefix",
            "run candidate and natural-generation behavior on all three models",
            "authorize internal mapping only after the cross-model behavior gate",
            "force paired tasks to the same padded tensor width",
            "measure all-depth residual selection interactions",
            "measure per-layer transition increments rather than treating state differences as recovered F",
            "measure real-head Attention and A-times-V routing from semantic sources",
            "select candidate heads on discovery templates only",
            "evaluate frozen heads on confirmation templates and held-out names",
            "authorize causal component validation only after frozen dynamic gates",
        ],
        "interpretation_limits": [
            "The state increment is an observed layer write, not recovery of the full nonlinear transition function.",
            "Max/min polarity selection is one ordered-relation operation family, not a universal reasoning module.",
            "Attention mass and A-times-V norm are routing observations, not causal proof.",
            "A low-rank or repeated profile can reflect shared architecture and must not be named a language law.",
            "Cross-model repetition is functional and depth-normalized, not neuron or head homology.",
            "No result can establish brain homology, evolutionary optimality, or a complete theory of language.",
        ],
        "hypotheses_under_test": [
            "The same relation evidence can support two behaviorally valid late operations without changing the pre-branch state.",
            "Operation-specific answer selection appears as an operation-by-world interaction rather than a raw max-minus-min prompt difference.",
            "A repeated interaction should be carried by layer writes and semantic-source routing after the late cue.",
            "Shared relation processing and polarity-specific selection can reuse physical components while differing in their conditional routing.",
        ],
        "automatic_next": {
            "behavior_stage": (
                "Run internal dynamics only if at least two models pass "
                "the frozen behavior gate."
            ),
            "causal_stage": (
                "Run a frozen component intervention only if at least "
                "two behavior-valid models pass the residual-transition "
                "and held-out Attention-routing gates."
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
            "schema_version": "phase1074_protocol_audit.v1",
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
        f"pairs={payload['pair_count_per_model']}/model"
    )


if __name__ == "__main__":
    main()
