#!/usr/bin/env python3
"""Freeze the Phase1071 causal-exposure-aware pattern-family protocol."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1051_natural_behavior_protocol as behavior
import phase1070_process_answer_protocol as previous
import phase1071_behavior_calibration_protocol as calibration


PHASE = 1071
PROTOCOL_REVISION = 1
MODELS = previous.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
RELATION_NAMES = previous.RELATION_NAMES
QUERY_TYPES = previous.QUERY_TYPES
LAYOUTS = ("clean", "early_distractor", "mid_distractor")
TEMPLATES = previous.TEMPLATES
REPLICATES = previous.REPLICATES
SPLITS = previous.SPLITS
ANCHOR_BRANCHES = previous.ANCHOR_BRANCHES
SWITCH_BRANCHES = previous.SWITCH_BRANCHES
ANSWER_BRANCHES = previous.ANSWER_BRANCHES
LEXICAL_BRANCHES = previous.LEXICAL_BRANCHES
STATES = previous.STATES
PATH_NAMES = previous.PATH_NAMES
ASSISTANT_PREFILL = previous.ASSISTANT_PREFILL
NATURAL_AUDIT_PER_PATH = previous.NATURAL_AUDIT_PER_PATH
NATURAL_GENERATION_STEPS = previous.NATURAL_GENERATION_STEPS
OUT_ROOT = calibration.OUT_ROOT
SOURCE_ROOT = previous.OUT_ROOT
CALIBRATION_ROOT = calibration.CALIBRATION_ROOT

CAPTURE_ROLES = (
    "pre_probe",
    "upper_edge",
    "switch_edge",
    "switch_probe",
    "lower_edge",
    "anchor_edge",
    "evidence_probe",
    "query_candidate_a",
    "query_candidate_b",
    "operator",
    "query",
    "answer_boundary",
)
PROCESS_ROLES = (
    "evidence_probe",
    "operator",
    "query",
    "answer_boundary",
)
PRIMARY_PROCESS_ROLES = (
    "evidence_probe",
    "answer_boundary",
)
HARD_NEGATIVE_ROLES = (
    "pre_probe",
    "switch_probe",
)
ROLE_EXPOSURE = {
    "pre_probe": "none",
    "upper_edge": "none",
    "switch_edge": "switch_only",
    "switch_probe": "switch_only",
    "lower_edge": "switch_only",
    "anchor_edge": "both",
    "evidence_probe": "both",
    "query_candidate_a": "both",
    "query_candidate_b": "both",
    "operator": "both",
    "query": "both",
    "answer_boundary": "both",
}

# Frozen before any Phase1071 hidden-state forward pass. These are
# authorization gates, not definitions of the phenomenon.
GATES = {
    "candidate_first_token_accuracy_min": 0.88,
    "semantic_first_natural_rate_min": 0.82,
    "per_query_candidate_accuracy_min": 0.80,
    "per_path_candidate_accuracy_min": 0.78,
    "valid_process_quad_per_relation_min": 100,
    "valid_answer_pair_per_relation_min": 160,
    "complete_factorial_unit_per_relation_min": 12,
    "candidate_finite_rate_min": 0.995,
    "internal_finite_rate_min": 0.995,
    "minimum_strong_relations_per_model": 2,
    "minimum_repeated_models": 2,
    "process_window_start": 0.35,
    "process_did_relative_magnitude_min": 0.004,
    "evidence_probe_did_relative_magnitude_min": 0.003,
    "answer_boundary_did_relative_magnitude_min": 0.004,
    "process_lexical_reuse_cosine_min": 0.20,
    "process_answer_invariance_cosine_min": 0.20,
    "process_discovery_confirmation_profile_cosine_min": 0.70,
    "process_depth_reversal_gap_min": 0.05,
    "hard_negative_process_did_max": 1e-6,
    "embedding_process_did_relative_magnitude_max": 1e-5,
    "late_depth_start": 0.70,
    "process_to_answer_readout_ratio_max": 0.60,
}

write_json = previous.write_json
write_jsonl = previous.write_jsonl
read_json = previous.read_json
read_jsonl = previous.read_jsonl
digest = previous.digest
tokenizer_for = previous.tokenizer_for
offset_token_spans = previous.offset_token_spans
state_factors = previous.state_factors
split_for_template = previous.split_for_template
phrase_set_for_template = previous.phrase_set_for_template
single_token_names = previous.single_token_names
cell_name_sets = previous.cell_name_sets
relation_clause = previous.relation_clause
tagged_fact_items = previous.tagged_fact_items
mark = previous.mark


def compose_exposure_block(
    tagged: list[tuple[str | None, str]],
    layout: str,
) -> tuple[str, dict[str, tuple[int, int, str]]]:
    by_role = {
        str(role): text
        for role, text in tagged
        if role is not None
    }
    distractor = "Unrelated note: the room had blue curtains"
    sequence: list[tuple[str | None, str]] = [
        ("pre_probe", "Preparation marker"),
    ]
    if layout == "early_distractor":
        sequence.append((None, distractor))
    sequence.extend([
        ("upper_edge", by_role["upper_edge"]),
        ("switch_edge", by_role["switch_edge"]),
        ("switch_probe", "Switch evidence recorded"),
    ])
    if layout == "mid_distractor":
        sequence.append((None, distractor))
    elif layout not in ("clean", "early_distractor"):
        raise ValueError(f"unknown layout: {layout}")
    sequence.extend([
        ("lower_edge", by_role["lower_edge"]),
        ("anchor_edge", by_role["anchor_edge"]),
        ("evidence_probe", "All evidence is now complete"),
    ])

    pieces = []
    spans: dict[str, tuple[int, int, str]] = {}
    cursor = 0
    for index, (role, text) in enumerate(sequence):
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


def style_prompt(
    selected_style: int,
    task_prefix: str,
    evidence: str,
    query: str,
    operator: str,
) -> str:
    if selected_style == 0:
        return (
            f"{task_prefix} Facts: {evidence} Question: {query} "
            f"{operator} exactly one person's name."
        )
    if selected_style == 1:
        return (
            f"{task_prefix} Use the ordering facts transitively when "
            f"necessary. Facts: {evidence} Question: {query} "
            f"{operator} one name only."
        )
    if selected_style == 2:
        return (
            f"{task_prefix} Ordering evidence: {evidence} Using only "
            f"this evidence, answer: {query} {operator} exactly one name."
        )
    if selected_style == 3:
        return (
            f"{task_prefix} Read every fact before answering. {evidence} "
            f"{query} {operator} with one person's name and nothing else."
        )
    raise ValueError(f"unknown selected style: {selected_style}")


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
    selected_style: int,
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
    evidence, evidence_spans = compose_exposure_block(
        tagged, layout
    )
    endpoint_a = semantic_names["endpoint_a"]
    endpoint_b = semantic_names["endpoint_b"]
    query_tail = previous.RELATIONS[relation][
        f"{query_type}_query"
    ][phrase_set]
    query = f"Between {endpoint_a} and {endpoint_b}, {query_tail}?"
    task_prefixes = (
        "Ordering task.",
        "Comparison exercise.",
        "Relation evidence.",
        "Reasoning item.",
    )
    operators = ("Give", "Select", "Write", "Respond")
    task_prefix = task_prefixes[template_index]
    operator = operators[template_index]
    raw_prompt = style_prompt(
        selected_style,
        task_prefix,
        evidence,
        query,
        operator,
    )
    evidence_start = raw_prompt.find(evidence)
    if evidence_start < 0:
        raise RuntimeError("evidence block missing from prompt")
    raw_spans = {
        role: (
            evidence_start + span[0],
            evidence_start + span[1],
            span[2],
        )
        for role, span in evidence_spans.items()
    }
    query_span = mark(raw_prompt, query)
    endpoint_a_start = query.find(endpoint_a)
    endpoint_b_start = query.find(
        endpoint_b, endpoint_a_start + len(endpoint_a)
    )
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
    classes = {"b0": [endpoint_a], "b1": [endpoint_b]}
    metadata = {
        **semantic_names,
        "answer": answer,
        "query_text": query,
        "query_tail": query_tail,
        "path_name": PATH_NAMES[
            (anchor_branch, switch_branch)
        ],
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
    selected_style: int,
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
        selected_style,
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
        "schema_version": "phase1071_exposure_pattern_case.v1",
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
        "selected_prompt_style": selected_style,
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
        "role_exposure": dict(ROLE_EXPOSURE),
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
    previous.assign_mismatch_units(cases)


def audit_model(
    model_name: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)
    role_spans_valid = True
    candidate_single_token = True
    candidate_disjoint = True
    mismatch_disjoint = True
    mismatch_symmetric = True
    truth_table_valid = True
    role_exposure_valid = True
    downstream_after_evidence = True
    pre_probe_prefix_invariant = True
    switch_probe_anchor_invariant = True
    fixed_lexical_lengths_equal = True
    query_fixed = True
    answer_and_lexical_truth = True
    for row in cases:
        width = len(row["input_ids"])
        role_spans_valid = role_spans_valid and all(
            0 <= row["role_spans"][role][0]
            <= row["role_spans"][role][1] < width
            for role in CAPTURE_ROLES
        )
        candidate_single_token = candidate_single_token and all(
            len(values) == 1
            for class_values in row["candidate_token_ids"].values()
            for values in class_values
        )
        left = set(row["candidate_first_token_ids"]["b0"])
        right = set(row["candidate_first_token_ids"]["b1"])
        candidate_disjoint = (
            candidate_disjoint
            and bool(left)
            and bool(right)
            and left.isdisjoint(right)
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
        )
        role_exposure_valid = (
            role_exposure_valid
            and row["role_exposure"] == ROLE_EXPOSURE
        )
        evidence_position = row["role_positions"]["evidence_probe"]
        downstream_after_evidence = (
            downstream_after_evidence
            and all(
                row["role_positions"][role] >= evidence_position
                for role in (
                    "query_candidate_a",
                    "query_candidate_b",
                    "operator",
                    "query",
                    "answer_boundary",
                )
            )
        )

    for rows in by_unit.values():
        states = {str(row["state"]): row for row in rows}
        if set(states) != set(STATES):
            continue
        query_fixed = query_fixed and len({
            row["query_text"] for row in rows
        }) == 1
        pre_prefixes = {
            tuple(
                row["input_ids"][
                    :row["role_positions"]["pre_probe"] + 1
                ]
            )
            for row in rows
        }
        pre_probe_prefix_invariant = (
            pre_probe_prefix_invariant and len(pre_prefixes) == 1
        )
        for switch in SWITCH_BRANCHES:
            for answer in ANSWER_BRANCHES:
                for lexical in LEXICAL_BRANCHES:
                    left = states[
                        f"a0_b{switch}_y{answer}_l{lexical}"
                    ]
                    right = states[
                        f"a1_b{switch}_y{answer}_l{lexical}"
                    ]
                    left_prefix = left["input_ids"][
                        :left["role_positions"]["switch_probe"] + 1
                    ]
                    right_prefix = right["input_ids"][
                        :right["role_positions"]["switch_probe"] + 1
                    ]
                    switch_probe_anchor_invariant = (
                        switch_probe_anchor_invariant
                        and left_prefix == right_prefix
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
        for answer in ANSWER_BRANCHES:
            for lexical in LEXICAL_BRANCHES:
                labels = {
                    tuple(states[
                        f"a{anchor}_b{switch}_y{answer}_l{lexical}"
                    ]["acceptable_labels"])
                    for anchor in ANCHOR_BRANCHES
                    for switch in SWITCH_BRANCHES
                }
                answer_and_lexical_truth = (
                    answer_and_lexical_truth and len(labels) == 1
                )
        for anchor in ANCHOR_BRANCHES:
            for switch in SWITCH_BRANCHES:
                for lexical in LEXICAL_BRANCHES:
                    answer_and_lexical_truth = (
                        answer_and_lexical_truth
                        and states[
                            f"a{anchor}_b{switch}_y0_l{lexical}"
                        ]["acceptable_labels"]
                        != states[
                            f"a{anchor}_b{switch}_y1_l{lexical}"
                        ]["acceptable_labels"]
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
    expected_case_count = (
        len(RELATION_NAMES)
        * len(QUERY_TYPES)
        * len(LAYOUTS)
        * len(TEMPLATES)
        * len(REPLICATES)
        * len(STATES)
    )
    expected_units = expected_case_count // len(STATES)
    checks = {
        "case_count": len(cases) == expected_case_count,
        "unit_count": len(by_unit) == expected_units,
        "complete_factorial_units": all(
            {row["state"] for row in rows} == set(STATES)
            for rows in by_unit.values()
        ),
        "balanced_cells": all(
            counts[(relation, query, layout, split)] == 64
            for relation in RELATION_NAMES
            for query in QUERY_TYPES
            for layout in LAYOUTS
            for split in SPLITS
        ),
        "role_spans_valid": role_spans_valid,
        "candidate_continuations_single_token": candidate_single_token,
        "candidate_first_tokens_disjoint": candidate_disjoint,
        "mismatch_candidate_tokens_disjoint": mismatch_disjoint,
        "mismatch_pairing_symmetric": mismatch_symmetric,
        "path_truth_table_valid": truth_table_valid,
        "role_exposure_labels_valid": role_exposure_valid,
        "all_downstream_roles_after_evidence_probe": (
            downstream_after_evidence
        ),
        "pre_probe_prefix_invariant": pre_probe_prefix_invariant,
        "switch_probe_prefix_invariant_to_future_anchor": (
            switch_probe_anchor_invariant
        ),
        "fixed_lexical_token_lengths_equal": (
            fixed_lexical_lengths_equal
        ),
        "query_fixed_within_unit": query_fixed,
        "answer_and_lexical_truth_preserved": (
            answer_and_lexical_truth
        ),
    }
    return {
        "schema_version": "phase1071_protocol_model_audit.v1",
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
    calibration_prereg = read_json(
        CALIBRATION_ROOT / "protocol" / "preregistration.json"
    )
    calibration_selection = read_json(
        CALIBRATION_ROOT / "analysis" / "prompt_selection.json"
    )
    if (
        bool(source_next["should_continue_automatically"])
        or source_next["route"] != "stop_at_process_answer_atlas"
        or not source_audit["all_integrity_checks_passed"]
    ):
        raise RuntimeError("Phase1070 source decision or audit drift")
    if (
        calibration_selection["protocol_digest"]
        != calibration_prereg["protocol_digest"]
    ):
        raise RuntimeError("Phase1071 calibration digest drift")
    selected_style = int(
        calibration_selection["selected_prompt_style"]
    )
    names = tuple(calibration_prereg["reserved_mechanism_names"])
    if set(names) & set(calibration_prereg["calibration_names"]):
        raise RuntimeError("calibration/mechanism name leakage")

    model_audits = {}
    case_count = unit_count = None
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
                                    selected_style,
                                ))
                                semantic_case_index += 1
        assign_mismatch_units(cases)
        audit = audit_model(model_name, cases)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1071 audit failed for {model_name}: {audit}"
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
        case_count = len(cases)
        unit_count = len(cases) // len(STATES)

    payload = {
        "schema_version": "phase1071_preregistration.v1",
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
        "primary_process_roles": list(PRIMARY_PROCESS_ROLES),
        "hard_negative_roles": list(HARD_NEGATIVE_ROLES),
        "role_exposure": ROLE_EXPOSURE,
        "assistant_prefill": ASSISTANT_PREFILL,
        "case_count_per_model": case_count,
        "unit_count_per_model": unit_count,
        "natural_audit_per_path": NATURAL_AUDIT_PER_PATH,
        "natural_audit_per_model": (
            len(RELATION_NAMES)
            * len(PATH_NAMES)
            * NATURAL_AUDIT_PER_PATH
        ),
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "selected_prompt_style": selected_style,
        "selected_prompt_style_label": calibration.STYLE_LABELS[
            selected_style
        ],
        "calibration_gate_passed": calibration_selection[
            "calibration_gate_passed"
        ],
        "calibration_protocol_digest": calibration_prereg[
            "protocol_digest"
        ],
        "mechanism_names": list(names),
        "gates": dict(GATES),
        "source_phase1070_digest": source_prereg["protocol_digest"],
        "source_phase1070_decision": source_next,
        "user_override_reason": (
            "The user explicitly requested continuation with a repaired "
            "protocol. Phase1070 remains a frozen failed automatic gate."
        ),
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
            "shared_process_across_lexical_realizations": (
                "cos(D[y,l0], D[y,l1])"
            ),
            "shared_process_across_answer_identities": (
                "cos(D[y0,l], D[y1,l])"
            ),
            "lexical_specific_difference": (
                "h[a,b,y,l1] - h[a,b,y,l0]"
            ),
        },
        "measurement_order": [
            "select one common prompt using behavior-only held-out names",
            "freeze mechanism names, templates, roles, states, and gates",
            "verify exact causal-prefix invariance at hard-negative probes",
            "measure all-pair candidate and natural behavior",
            "capture all-depth residual states at fixed exposure roles",
            "require process DiD to remain zero before both factors are visible",
            "measure process DiD only at fixed post-evidence probes",
            "separate shared process reuse from lexical-specific differences",
            "test discovery/confirmation and cross-model repetition",
            "authorize component localization only through frozen gates",
        ],
        "interpretation_limits": [
            "The process DiD is a contrast, not a theory of transitive reasoning.",
            "Nonzero post-evidence DiD can still include confidence, redundancy, and conflict handling.",
            "Cosine reuse measures repeated direction, not a stored symbolic pattern.",
            "Surface difference and shared process reuse may coexist; neither proves optimal compression.",
            "Residual evidence does not identify attention, K/V, MLP, or neuron implementation.",
            "Cross-model profile agreement is not neuron homology.",
            "No result establishes brain homology, evolutionary optimality, or a complete language theory.",
            "Small-model failures remain evidence about these models and cannot be dismissed as an arbitrary percentage.",
        ],
        "hypotheses_under_test": [
            "A repeated process-sensitive structure should appear only after causal exposure to both matched factors.",
            "Within one ordered-relation family, some process direction may be reused across answer identities and lexical realizations.",
            "Lexical realization may retain a distinct differential component while sharing a downstream process skeleton.",
            "Repeated structure across names, templates, relations, and models is stronger evidence than top-neuron rank.",
        ],
        "automatic_next": {
            "continue_only_if": (
                "At least two models each have at least two relations "
                "passing behavior, numerical, hard-negative, split, "
                "post-evidence process reuse, and readout-separation gates."
            ),
            "next_phase": (
                "frozen component localization of the repeated post-"
                "evidence process field, preserving all hard negatives"
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
            "schema_version": "phase1071_protocol_audit.v1",
            "phase": PHASE,
            "protocol_digest": payload["protocol_digest"],
            "model_audits": model_audits,
            "all_checks_passed": all(
                row["all_checks_passed"]
                for row in model_audits.values()
            ),
        },
    )
    return payload


def main() -> None:
    payload = build_protocol()
    print(
        f"Phase{PHASE} mechanism protocol "
        f"{payload['protocol_digest']} "
        f"style={payload['selected_prompt_style']} "
        f"cases={payload['case_count_per_model']}/model "
        f"units={payload['unit_count_per_model']}/model"
    )


if __name__ == "__main__":
    main()
