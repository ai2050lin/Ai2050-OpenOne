#!/usr/bin/env python3
"""Freeze the Phase1070 process/answer/surface orthogonal protocol."""

from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior
import phase1068_reasoning_generalization_protocol as source
import phase1069_local_coordinate_protocol as previous


PHASE = 1070
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
RELATION_NAMES = ("height", "age", "weight", "arrival", "score")
QUERY_TYPES = ("max", "min")
LAYOUTS = ("forward", "reverse", "distractor")
TEMPLATES = (0, 1, 2, 3)
REPLICATES = (0, 1)
SPLITS = ("discovery", "confirmation")
ANCHOR_BRANCHES = (0, 1)
SWITCH_BRANCHES = (0, 1)
ANSWER_BRANCHES = (0, 1)
LEXICAL_BRANCHES = (0, 1)
STATES = tuple(
    f"a{anchor}_b{switch}_y{answer}_l{lexical}"
    for anchor in ANCHOR_BRANCHES
    for switch in SWITCH_BRANCHES
    for answer in ANSWER_BRANCHES
    for lexical in LEXICAL_BRANCHES
)
PATH_NAMES = {
    (0, 0): "shortcut_only",
    (0, 1): "transitive_only",
    (1, 0): "duplicated_direct",
    (1, 1): "direct_plus_bridge",
}
CAPTURE_ROLES = (
    "upper_edge",
    "switch_edge",
    "lower_edge",
    "anchor_edge",
    "query_candidate_a",
    "query_candidate_b",
    "operator",
    "query",
    "answer_boundary",
)
PROCESS_ROLES = (
    "query_candidate_a",
    "query_candidate_b",
    "operator",
    "query",
    "answer_boundary",
)
ASSISTANT_PREFILL = "Answer:"
NATURAL_AUDIT_PER_PATH = 24
NATURAL_GENERATION_STEPS = 12
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1070_process_answer_orthogonal"
)
SOURCE_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1069_local_coordinate_reasoning"
)

# Frozen before any Phase1070 model forward pass. These gates authorize an
# automatic component-level follow-up; failure does not delete atlas evidence.
GATES = {
    "candidate_first_token_accuracy_min": 0.90,
    "semantic_first_natural_rate_min": 0.85,
    "per_query_candidate_accuracy_min": 0.85,
    "per_path_candidate_accuracy_min": 0.82,
    "valid_process_quad_per_relation_min": 100,
    "valid_answer_pair_per_relation_min": 160,
    "complete_factorial_unit_per_relation_min": 12,
    "candidate_finite_rate_min": 0.995,
    "internal_finite_rate_min": 0.995,
    "minimum_strong_relations_per_model": 2,
    "minimum_repeated_models": 2,
    "process_window_start": 0.35,
    "process_did_relative_magnitude_min": 0.004,
    "process_lexical_reuse_cosine_min": 0.20,
    "process_answer_invariance_cosine_min": 0.15,
    "process_discovery_confirmation_profile_cosine_min": 0.65,
    "process_depth_reversal_gap_min": 0.05,
    "embedding_process_did_relative_magnitude_max": 1e-5,
    "late_depth_start": 0.70,
    "process_to_answer_readout_ratio_max": 0.60,
}

RELATIONS = {
    "height": {
        "positive": ("is taller than", "stands higher than"),
        "inverse": ("is shorter than", "stands lower than"),
        "max_query": ("who is taller", "which person has greater height"),
        "min_query": ("who is shorter", "which person has less height"),
    },
    "age": {
        "positive": ("is older than", "has lived longer than"),
        "inverse": ("is younger than", "has lived for less time than"),
        "max_query": ("who is older", "which person has greater age"),
        "min_query": ("who is younger", "which person has less age"),
    },
    "weight": {
        "positive": ("is heavier than", "weighs more than"),
        "inverse": ("is lighter than", "weighs less than"),
        "max_query": ("who is heavier", "which person weighs more"),
        "min_query": ("who is lighter", "which person weighs less"),
    },
    "arrival": {
        "positive": ("arrived before", "reached the place earlier than"),
        "inverse": ("arrived after", "reached the place later than"),
        "max_query": (
            "who arrived earlier",
            "which person reached the place first",
        ),
        "min_query": (
            "who arrived later",
            "which person reached the place last",
        ),
    },
    "score": {
        "positive": ("has a higher score than", "earned more points than"),
        "inverse": ("has a lower score than", "earned fewer points than"),
        "max_query": (
            "who has the higher score",
            "which person earned more points",
        ),
        "min_query": (
            "who has the lower score",
            "which person earned fewer points",
        ),
    },
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest
tokenizer_for = source.tokenizer_for
offset_token_spans = source.offset_token_spans


def state_factors(state: str) -> tuple[int, int, int, int]:
    fields = state.split("_")
    if len(fields) != 4:
        raise ValueError(f"invalid Phase1070 state: {state}")
    return tuple(int(field[1:]) for field in fields)  # type: ignore[return-value]


def split_for_template(template_index: int) -> str:
    return "discovery" if template_index < 2 else "confirmation"


def phrase_set_for_template(template_index: int) -> int:
    return 0 if template_index < 2 else 1


def mark(text: str, value: str, start: int = 0) -> tuple[int, int, str]:
    position = text.find(value, start)
    if position < 0:
        raise RuntimeError(f"missing marked text: {value!r}")
    return position, position + len(value), value


def single_token_names(
    source_names: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    tokenizers = {
        model: tokenizer_for(model) for model in MODELS
    }
    selected = []
    for name in source_names:
        if all(
            len(tokenizer.encode(
                " " + str(name), add_special_tokens=False
            )) == 1
            for tokenizer in tokenizers.values()
        ):
            selected.append(str(name))
    if len(selected) < 48:
        raise RuntimeError(
            f"Phase1070 needs 48 cross-model single-token names, "
            f"found {len(selected)}"
        )
    return tuple(selected)


def cell_name_sets(
    names: tuple[str, ...],
    cell_id: str,
) -> dict[tuple[int, int], list[str]]:
    required = len(TEMPLATES) * len(REPLICATES) * 6
    if required > len(names):
        raise RuntimeError(
            f"{cell_id} needs {required} names but only {len(names)} exist"
        )
    ranked = sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"phase1070|{cell_id}|{name}".encode("utf-8")
        ).hexdigest(),
    )
    result = {}
    cursor = 0
    for template_index in TEMPLATES:
        for replicate in REPLICATES:
            result[(template_index, replicate)] = ranked[cursor:cursor + 6]
            cursor += 6
    return result


def relation_clause(
    relation: str,
    higher: str,
    lower: str,
    lexical_branch: int,
    phrase_set: int,
) -> str:
    spec = RELATIONS[relation]
    if lexical_branch == 0:
        return f"{higher} {spec['positive'][phrase_set]} {lower}"
    return f"{lower} {spec['inverse'][phrase_set]} {higher}"


def tagged_fact_items(
    relation: str,
    names: list[str],
    query_type: str,
    anchor_branch: int,
    switch_branch: int,
    answer_branch: int,
    lexical_branch: int,
    phrase_set: int,
) -> tuple[
    list[tuple[str | None, str]],
    dict[str, str],
    str,
]:
    endpoint_a, middle_a, middle_b, endpoint_b, null_a, null_b = names
    endpoints = (endpoint_a, endpoint_b)
    answer = endpoints[answer_branch]
    other = endpoints[1 - answer_branch]
    if query_type == "max":
        higher, lower = answer, other
    else:
        higher, lower = other, answer

    upper = relation_clause(
        relation, higher, middle_a, lexical_branch, phrase_set
    )
    lower_edge = relation_clause(
        relation, middle_b, lower, lexical_branch, phrase_set
    )
    switch = (
        relation_clause(
            relation, higher, lower, lexical_branch, phrase_set
        )
        if switch_branch == 0
        else relation_clause(
            relation, middle_a, middle_b, lexical_branch, phrase_set
        )
    )
    anchor = (
        relation_clause(
            relation, null_a, null_b, lexical_branch, phrase_set
        )
        if anchor_branch == 0
        else relation_clause(
            relation, higher, lower, lexical_branch, phrase_set
        )
    )
    tagged = [
        ("upper_edge", upper),
        ("switch_edge", switch),
        ("lower_edge", lower_edge),
        ("anchor_edge", anchor),
    ]
    metadata = {
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "middle_a": middle_a,
        "middle_b": middle_b,
        "null_a": null_a,
        "null_b": null_b,
        "higher": higher,
        "lower": lower,
        "answer": answer,
    }
    return tagged, metadata, answer


def compose_facts(
    tagged: list[tuple[str | None, str]],
    layout: str,
) -> tuple[str, dict[str, tuple[int, int, str]]]:
    distractor = (
        None,
        "Unrelated note: the room had blue curtains",
    )
    if layout == "forward":
        displayed = list(tagged)
    elif layout == "reverse":
        displayed = list(reversed(tagged))
    elif layout == "distractor":
        displayed = [tagged[0], tagged[1], distractor, tagged[2], tagged[3]]
    else:
        raise ValueError(f"unknown layout: {layout}")

    pieces = []
    spans: dict[str, tuple[int, int, str]] = {}
    cursor = 0
    for index, (role, text) in enumerate(displayed):
        if index:
            pieces.append(". ")
            cursor += 2
        start = cursor
        pieces.append(text)
        cursor += len(text)
        if role is not None:
            spans[role] = (start, cursor, text)
    pieces.append(".")
    return "".join(pieces), spans


def render_prompt(
    relation: str,
    query_type: str,
    layout: str,
    names: list[str],
    anchor_branch: int,
    switch_branch: int,
    answer_branch: int,
    lexical_branch: int,
    template_index: int,
) -> tuple[
    str,
    dict[str, tuple[int, int, str]],
    dict[str, list[str]],
    dict[str, Any],
]:
    phrase_set = phrase_set_for_template(template_index)
    tagged, semantic_names, answer = tagged_fact_items(
        relation,
        names,
        query_type,
        anchor_branch,
        switch_branch,
        answer_branch,
        lexical_branch,
        phrase_set,
    )
    facts, fact_spans = compose_facts(tagged, layout)
    endpoint_a = semantic_names["endpoint_a"]
    endpoint_b = semantic_names["endpoint_b"]
    query_tail = RELATIONS[relation][
        f"{query_type}_query"
    ][phrase_set]
    query = f"Between {endpoint_a} and {endpoint_b}, {query_tail}?"

    if template_index == 0:
        operator = "Find"
        raw_prompt = (
            f"Ordering facts: {facts} {operator} the answer to this "
            f"question: {query} Write exactly one person's name and stop."
        )
    elif template_index == 1:
        operator = "Select"
        raw_prompt = (
            f"Question: {query} Use these ordering facts: {facts} "
            f"{operator} one name only, with no explanation."
        )
    elif template_index == 2:
        operator = "Infer"
        raw_prompt = (
            f"Evidence: {facts} {operator} the answer: {query} "
            "Return exactly the person's name, then end."
        )
    else:
        operator = "Report"
        raw_prompt = (
            f"{operator} one name for this question: {query} "
            f"Use only this evidence: {facts} Your entire answer must be "
            "one person's name."
        )

    facts_start = raw_prompt.find(facts)
    if facts_start < 0:
        raise RuntimeError("facts missing from rendered raw prompt")
    raw_spans = {
        role: (
            facts_start + span[0],
            facts_start + span[1],
            span[2],
        )
        for role, span in fact_spans.items()
    }
    query_span = mark(raw_prompt, query)
    endpoint_a_start = query.find(endpoint_a)
    endpoint_b_start = query.find(endpoint_b, endpoint_a_start + len(endpoint_a))
    if endpoint_a_start < 0 or endpoint_b_start < 0:
        raise RuntimeError("query candidate span drift")
    raw_spans.update({
        "query_candidate_a": (
            query_span[0] + endpoint_a_start,
            query_span[0] + endpoint_a_start + len(endpoint_a),
            endpoint_a,
        ),
        "query_candidate_b": (
            query_span[0] + endpoint_b_start,
            query_span[0] + endpoint_b_start + len(endpoint_b),
            endpoint_b,
        ),
        "operator": mark(raw_prompt, operator),
        "query": query_span,
    })
    classes = {
        "b0": [endpoint_a],
        "b1": [endpoint_b],
    }
    metadata = {
        **semantic_names,
        "query_text": query,
        "query_tail": query_tail,
        "answer": answer,
        "path_name": PATH_NAMES[(anchor_branch, switch_branch)],
        "direct_edge_present": bool(
            switch_branch == 0 or anchor_branch == 1
        ),
        "bridge_edge_present": bool(switch_branch == 1),
        "requires_transitive_chain": bool(
            anchor_branch == 0 and switch_branch == 1
        ),
        "fact_count": 4,
    }
    return raw_prompt, raw_spans, classes, metadata


def response_buckets(
    relation: str,
    query_type: str,
    layout: str,
    path_name: str,
) -> list[str]:
    return [
        "global:all",
        f"relation:{relation}",
        f"relation_query:{relation}:{query_type}",
        f"relation_path:{relation}:{path_name}",
        f"layout:{layout}",
        f"path:{path_name}",
    ]


def build_model_case(
    tokenizer,
    model_name: str,
    names: tuple[str, ...],
    relation: str,
    query_type: str,
    layout: str,
    template_index: int,
    replicate: int,
    state: str,
    semantic_case_index: int,
) -> dict[str, Any]:
    anchor, switch, answer, lexical = state_factors(state)
    cell_id = f"{relation}.{query_type}.{layout}"
    name_sets = cell_name_sets(names, cell_id)
    cell_names = name_sets[(template_index, replicate)]
    raw_prompt, raw_spans, classes, metadata = render_prompt(
        relation,
        query_type,
        layout,
        cell_names,
        anchor,
        switch,
        answer,
        lexical,
        template_index,
    )
    rendered = behavior.render_native(
        tokenizer,
        model_name,
        raw_prompt,
        with_system=False,
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
    unit_id = f"{cell_id}.t{template_index}.r{replicate}"
    path_name = PATH_NAMES[(anchor, switch)]
    return {
        "schema_version": "phase1070_process_answer_case.v1",
        "phase": PHASE,
        "model": model_name,
        "semantic_case_index": semantic_case_index,
        "record_id": f"{model_name}.{unit_id}.{state}",
        "unit_id": unit_id,
        "cell_id": cell_id,
        "relation": relation,
        "query_type": query_type,
        "layout": layout,
        "split": split_for_template(template_index),
        "template_index": template_index,
        "replicate": replicate,
        "phrase_set": phrase_set_for_template(template_index),
        "state": state,
        "anchor_branch": anchor,
        "switch_branch": switch,
        "answer_branch": answer,
        "lexical_branch": lexical,
        "path_name": path_name,
        "direct_edge_present": metadata["direct_edge_present"],
        "bridge_edge_present": metadata["bridge_edge_present"],
        "requires_transitive_chain": metadata[
            "requires_transitive_chain"
        ],
        "cell_names": cell_names,
        "semantic_names": {
            key: metadata[key]
            for key in (
                "endpoint_a",
                "endpoint_b",
                "middle_a",
                "middle_b",
                "null_a",
                "null_b",
                "higher",
                "lower",
                "answer",
            )
        },
        "query_text": metadata["query_text"],
        "query_tail": metadata["query_tail"],
        "fact_count": metadata["fact_count"],
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_spans": {
            role: [int(span[0]), int(span[1])]
            for role, span in role_spans.items()
        },
        "role_positions": {
            role: int(span[1])
            for role, span in role_spans.items()
        },
        "candidate_labels": classes,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": candidate_first_token_ids,
        "expected_class": f"b{answer}",
        "acceptable_labels": classes[f"b{answer}"],
        "continuation_prefix": " ",
        "response_buckets": response_buckets(
            relation, query_type, layout, path_name
        ),
        "mismatch_unit_id": None,
    }


def assign_mismatch_units(cases: list[dict[str, Any]]) -> None:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)
    unit_keys = {}
    for unit_id, rows in by_unit.items():
        row = rows[0]
        unit_keys[(
            row["relation"],
            row["query_type"],
            row["layout"],
            int(row["replicate"]),
            int(row["template_index"]),
        )] = unit_id
    for unit_id, rows in by_unit.items():
        row = rows[0]
        template = int(row["template_index"])
        paired_template = template + 1 if template % 2 == 0 else template - 1
        mismatch_id = unit_keys[(
            row["relation"],
            row["query_type"],
            row["layout"],
            int(row["replicate"]),
            paired_template,
        )]
        for value in rows:
            value["mismatch_unit_id"] = mismatch_id


def audit_model(
    model_name: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)
    complete_units = all(
        {row["state"] for row in rows} == set(STATES)
        for rows in by_unit.values()
    )
    role_spans_valid = True
    candidates_disjoint = True
    candidate_single_token = True
    mismatch_disjoint = True
    mismatch_symmetric = True
    query_fixed = True
    path_answers_preserved = True
    answer_changes = True
    lexical_answers_preserved = True
    fixed_lexical_lengths_equal = True
    truth_table_valid = True
    for row in cases:
        width = len(row["input_ids"])
        role_spans_valid = role_spans_valid and all(
            0 <= row["role_spans"][role][0]
            <= row["role_spans"][role][1] < width
            for role in CAPTURE_ROLES
        )
        left = set(row["candidate_first_token_ids"]["b0"])
        right = set(row["candidate_first_token_ids"]["b1"])
        candidates_disjoint = (
            candidates_disjoint
            and bool(left)
            and bool(right)
            and left.isdisjoint(right)
        )
        candidate_single_token = candidate_single_token and all(
            len(values) == 1
            for class_values in row["candidate_token_ids"].values()
            for values in class_values
        )
        mismatch = by_unit[str(row["mismatch_unit_id"])][0]
        mismatch_ids = (
            set(mismatch["candidate_first_token_ids"]["b0"])
            | set(mismatch["candidate_first_token_ids"]["b1"])
        )
        mismatch_disjoint = (
            mismatch_disjoint
            and (left | right).isdisjoint(mismatch_ids)
        )
        mismatch_symmetric = (
            mismatch_symmetric
            and str(mismatch["mismatch_unit_id"])
            == str(row["unit_id"])
        )
        anchor = int(row["anchor_branch"])
        switch = int(row["switch_branch"])
        truth_table_valid = truth_table_valid and (
            row["path_name"] == PATH_NAMES[(anchor, switch)]
            and bool(row["requires_transitive_chain"])
            == bool(anchor == 0 and switch == 1)
            and bool(row["direct_edge_present"])
            == bool(switch == 0 or anchor == 1)
            and bool(row["bridge_edge_present"])
            == bool(switch == 1)
        )

    for rows in by_unit.values():
        states = {str(row["state"]): row for row in rows}
        query_fixed = query_fixed and len({
            row["query_text"] for row in rows
        }) == 1
        for answer in ANSWER_BRANCHES:
            for lexical in LEXICAL_BRANCHES:
                labels = {
                    tuple(states[
                        f"a{anchor}_b{switch}_y{answer}_l{lexical}"
                    ]["acceptable_labels"])
                    for anchor in ANCHOR_BRANCHES
                    for switch in SWITCH_BRANCHES
                }
                path_answers_preserved = (
                    path_answers_preserved and len(labels) == 1
                )
        for anchor in ANCHOR_BRANCHES:
            for switch in SWITCH_BRANCHES:
                for lexical in LEXICAL_BRANCHES:
                    left = states[
                        f"a{anchor}_b{switch}_y0_l{lexical}"
                    ]["acceptable_labels"]
                    right = states[
                        f"a{anchor}_b{switch}_y1_l{lexical}"
                    ]["acceptable_labels"]
                    answer_changes = answer_changes and left != right
                for answer in ANSWER_BRANCHES:
                    left = states[
                        f"a{anchor}_b{switch}_y{answer}_l0"
                    ]["acceptable_labels"]
                    right = states[
                        f"a{anchor}_b{switch}_y{answer}_l1"
                    ]["acceptable_labels"]
                    lexical_answers_preserved = (
                        lexical_answers_preserved and left == right
                    )
        for lexical in LEXICAL_BRANCHES:
            lengths = {
                len(row["input_ids"])
                for row in rows
                if int(row["lexical_branch"]) == lexical
            }
            fixed_lexical_lengths_equal = (
                fixed_lexical_lengths_equal and len(lengths) == 1
            )

    counts = Counter(
        (
            row["relation"],
            row["query_type"],
            row["layout"],
            row["split"],
        )
        for row in cases
    )
    checks = {
        "case_count": len(cases) == 3840,
        "unit_count": len(by_unit) == 240,
        "complete_factorial_units": complete_units,
        "balanced_cells": all(
            counts[(relation, query, layout, split)] == 64
            for relation in RELATION_NAMES
            for query in QUERY_TYPES
            for layout in LAYOUTS
            for split in SPLITS
        ),
        "role_spans_valid": role_spans_valid,
        "candidate_first_tokens_disjoint": candidates_disjoint,
        "candidate_continuations_single_token": candidate_single_token,
        "mismatch_candidate_tokens_disjoint": mismatch_disjoint,
        "mismatch_pairing_symmetric": mismatch_symmetric,
        "query_fixed_within_unit": query_fixed,
        "path_answers_preserved": path_answers_preserved,
        "answer_branch_changes_answer": answer_changes,
        "lexical_branch_preserves_answer": lexical_answers_preserved,
        "fixed_lexical_token_lengths_equal": fixed_lexical_lengths_equal,
        "path_truth_table_valid": truth_table_valid,
    }
    return {
        "schema_version": "phase1070_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
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
        != "stop_at_local_coordinate_atlas"
        or not source_audit["all_integrity_checks_passed"]
    ):
        raise RuntimeError("Phase1069 source decision or audit drift")
    names = single_token_names(
        source_prereg["cross_tokenizer_names"]
    )
    names = names[:48]
    model_audits = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        cases = []
        semantic_case_index = 0
        for relation in RELATION_NAMES:
            for query_type in QUERY_TYPES:
                for layout in LAYOUTS:
                    for template_index in TEMPLATES:
                        for replicate in REPLICATES:
                            for state in STATES:
                                cases.append(build_model_case(
                                    tokenizer,
                                    model_name,
                                    names,
                                    relation,
                                    query_type,
                                    layout,
                                    template_index,
                                    replicate,
                                    state,
                                    semantic_case_index,
                                ))
                                semantic_case_index += 1
        assign_mismatch_units(cases)
        audit = audit_model(model_name, cases)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1070 audit failed for {model_name}: {audit}"
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
        "schema_version": "phase1070_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "relations": list(RELATION_NAMES),
        "query_types": list(QUERY_TYPES),
        "layouts": list(LAYOUTS),
        "templates": list(TEMPLATES),
        "replicates": list(REPLICATES),
        "splits": list(SPLITS),
        "states": list(STATES),
        "path_names": {
            f"a{anchor}_b{switch}": name
            for (anchor, switch), name in PATH_NAMES.items()
        },
        "capture_components": ["residual"],
        "capture_roles": list(CAPTURE_ROLES),
        "process_roles": list(PROCESS_ROLES),
        "assistant_prefill": ASSISTANT_PREFILL,
        "case_count_per_model": 3840,
        "unit_count_per_model": 240,
        "natural_audit_per_path": NATURAL_AUDIT_PER_PATH,
        "natural_audit_per_model": (
            len(RELATION_NAMES)
            * len(PATH_NAMES)
            * NATURAL_AUDIT_PER_PATH
        ),
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "cross_model_single_token_name_count": len(names),
        "cross_model_single_token_names": list(names),
        "gates": dict(GATES),
        "source_phase1069_digest": source_prereg[
            "protocol_digest"
        ],
        "source_phase1069_decision": source_next,
        "core_contrast": {
            "switch_without_direct_anchor": (
                "h[a0,b1,y,l] - h[a0,b0,y,l]"
            ),
            "matched_switch_with_direct_anchor": (
                "h[a1,b1,y,l] - h[a1,b0,y,l]"
            ),
            "process_difference_in_differences": (
                "(h[a0,b1,y,l] - h[a0,b0,y,l]) - "
                "(h[a1,b1,y,l] - h[a1,b0,y,l])"
            ),
            "answer_contrast": (
                "h[a,b,y1,l] - h[a,b,y0,l]"
            ),
            "surface_contrast": (
                "h[a,b,y,l1] - h[a,b,y,l0]"
            ),
        },
        "measurement_order": [
            "freeze all path, answer, lexical, template, name, and numeric gates",
            "audit identical query text and fixed-token-length path cells",
            "measure candidate behavior without candidate enumeration in prompts",
            "separate semantic-first, strict-format, and termination behavior",
            "capture all-depth residual states at four evidence and five query/readout roles",
            "measure raw switch response with no direct anchor",
            "subtract the matched switch response while a direct answer anchor remains",
            "measure answer identity while computation path is fixed",
            "measure lexical realization while process and answer are fixed",
            "report all-pair atlas before behavior-conditioned strata",
            "test discovery/confirmation and cross-model repetition",
            "authorize component localization only if every frozen gate passes",
        ],
        "interpretation_limits": [
            "The difference-in-differences is a matched measurement contrast, not a pre-assumed law of reasoning.",
            "A nonzero process contrast can reflect confidence, redundancy, or conflict handling in addition to transitive computation.",
            "The direct-anchor control reduces but cannot remove every prompt-distribution difference.",
            "Local W_U readout is an output-coordinate diagnostic, not proof that intermediate layers literally decode with W_U.",
            "All-pair metrics are primary; behavior-conditioned metrics expose, but do not correct, selection bias.",
            "Residual-state evidence does not identify attention heads, MLP neurons, or a necessary causal circuit.",
            "Cross-model agreement is functional and depth-normalized, not neuron-to-neuron homology.",
            "No result establishes brain homology, evolutionary optimality, an ecological niche law, or a complete language theory.",
            "Small-model failure can reflect capability limits, but it cannot be dismissed without a larger-model replication.",
        ],
        "hypotheses_under_test": [
            "Language-relevant computation may be encoded by relative, context-conditioned differences rather than fixed global vectors.",
            "A computation-path response should survive answer and lexical changes after a matched clause-substitution control is subtracted.",
            "Answer identity should remain readable across fixed computation paths and should be stronger than process leakage on the final answer axis.",
            "Repeated structure across names, templates, relations, and models is stronger evidence than top-activation or single-neuron rank.",
        ],
        "automatic_next": {
            "continue_only_if": (
                "At least two models each have at least two relations that "
                "pass behavior, numerical, split-replication, process "
                "difference-in-differences, answer/lexical reuse, and "
                "readout-separation gates."
            ),
            "next_phase": (
                "frozen component-level localization of the repeated "
                "process difference-in-differences, with the same direct-"
                "anchor and local-answer controls"
            ),
        },
        "model_audits": model_audits,
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        OUT_ROOT / "protocol" / "preregistration.json",
        payload,
    )
    write_json(
        OUT_ROOT / "protocol" / "audit.json",
        {
            "schema_version": "phase1070_protocol_audit.v1",
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
        f"units={payload['unit_count_per_model']}/model "
        f"names={payload['cross_model_single_token_name_count']}"
    )


if __name__ == "__main__":
    main()
