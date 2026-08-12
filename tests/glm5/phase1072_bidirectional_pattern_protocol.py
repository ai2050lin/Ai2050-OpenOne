#!/usr/bin/env python3
"""Freeze the Phase1072 bidirectional exposure and task-control protocol."""

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
import phase1071_exposure_pattern_protocol as source


PHASE = 1072
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
BASE_RELATIONS = tuple(factorial.RELATION_NAMES)
TASK_FAMILIES = ("transitive", "direct_key_control")
PROMPT_BRANCHES = ("natural", "explicit")
EVIDENCE_ORDERS = ("switch_first", "anchor_first")
QUERY_TYPES = factorial.QUERY_TYPES
LAYOUTS = ("clean",)
TEMPLATES = factorial.TEMPLATES
REPLICATES = factorial.REPLICATES
SPLITS = factorial.SPLITS
ANCHOR_BRANCHES = factorial.ANCHOR_BRANCHES
SWITCH_BRANCHES = factorial.SWITCH_BRANCHES
ANSWER_BRANCHES = factorial.ANSWER_BRANCHES
LEXICAL_BRANCHES = factorial.LEXICAL_BRANCHES
STATES = factorial.STATES
PATH_NAMES = factorial.PATH_NAMES
ASSISTANT_PREFILL = factorial.ASSISTANT_PREFILL
NATURAL_AUDIT_PER_PATH = 8
NATURAL_GENERATION_STEPS = 10
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1072_bidirectional_pattern_specificity"
)
CALIBRATION_ROOT = OUT_ROOT / "calibration"
SOURCE_ROOT = source.OUT_ROOT
SOURCE_CALIBRATION_ROOT = source.CALIBRATION_ROOT


def condition_key(
    relation: str,
    task_family: str,
    prompt_branch: str,
    evidence_order: str,
) -> str:
    return "::".join(
        (relation, task_family, prompt_branch, evidence_order)
    )


def parse_condition(value: str) -> dict[str, str]:
    fields = value.split("::")
    if len(fields) != 4:
        raise ValueError(f"invalid Phase1072 condition: {value}")
    return {
        "base_relation": fields[0],
        "task_family": fields[1],
        "prompt_branch": fields[2],
        "evidence_order": fields[3],
    }


RELATION_NAMES = tuple(
    condition_key(relation, task_family, prompt_branch, evidence_order)
    for relation in BASE_RELATIONS
    for task_family in TASK_FAMILIES
    for prompt_branch in PROMPT_BRANCHES
    for evidence_order in EVIDENCE_ORDERS
)

CAPTURE_ROLES = (
    "pre_probe",
    "upper_edge",
    "switch_edge",
    "anchor_edge",
    "first_factor_probe",
    "lower_edge",
    "evidence_probe",
    "decision_cue",
    "query_candidate_a",
    "query_candidate_b",
    "operator",
    "query",
    "answer_boundary",
)
PROCESS_ROLES = (
    "evidence_probe",
    "decision_cue",
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
    "first_factor_probe",
)

# Frozen before any Phase1072 model result exists.
GATES = {
    "calibration_candidate_accuracy_min": 0.84,
    "calibration_semantic_first_rate_min": 0.76,
    "formal_candidate_accuracy_min": 0.86,
    "formal_semantic_first_rate_min": 0.76,
    "candidate_first_token_accuracy_min": 0.86,
    "semantic_first_natural_rate_min": 0.76,
    "calibration_formal_candidate_gap_max": 0.10,
    "calibration_formal_semantic_gap_max": 0.12,
    "per_query_candidate_accuracy_min": 0.78,
    "per_path_candidate_accuracy_min": 0.74,
    "valid_process_quad_per_relation_min": 24,
    "valid_answer_pair_per_relation_min": 48,
    "complete_factorial_unit_per_relation_min": 4,
    "candidate_finite_rate_min": 0.995,
    "internal_finite_rate_min": 0.995,
    "process_window_start": 0.35,
    "late_depth_start": 0.70,
    "target_process_did_relative_magnitude_min": 0.004,
    "process_lexical_reuse_cosine_min": 0.20,
    "process_answer_invariance_cosine_min": 0.20,
    "process_discovery_confirmation_profile_cosine_min": 0.65,
    "bidirectional_order_profile_cosine_min": 0.40,
    "natural_explicit_profile_cosine_min": 0.35,
    "target_control_process_ratio_min": 1.20,
    "target_control_process_gap_min": 0.002,
    "hard_negative_process_did_max": 1e-6,
    "embedding_process_did_relative_magnitude_max": 1e-5,
    "process_to_answer_readout_ratio_max": 0.60,
    "minimum_strong_relations_per_model": 2,
    "minimum_repeated_models": 2,
}

write_json = factorial.write_json
write_jsonl = factorial.write_jsonl
read_json = factorial.read_json
read_jsonl = factorial.read_jsonl
digest = factorial.digest
tokenizer_for = factorial.tokenizer_for
offset_token_spans = factorial.offset_token_spans
state_factors = factorial.state_factors
split_for_template = factorial.split_for_template
phrase_set_for_template = factorial.phrase_set_for_template
relation_clause = factorial.relation_clause
tagged_fact_items = factorial.tagged_fact_items
mark = factorial.mark


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
            f"phase1072|{cell_id}|{name}".encode("utf-8")
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


def compose_exposure_block(
    tagged: list[tuple[str | None, str]],
    evidence_order: str,
) -> tuple[
    str,
    dict[str, tuple[int, int, str]],
    tuple[str, ...],
]:
    by_role = {
        str(role): text
        for role, text in tagged
        if role is not None
    }
    if evidence_order == "switch_first":
        sequence: list[tuple[str | None, str]] = [
            ("pre_probe", "Preparation marker"),
            ("upper_edge", by_role["upper_edge"]),
            ("switch_edge", by_role["switch_edge"]),
            ("first_factor_probe", "First factor recorded"),
            ("lower_edge", by_role["lower_edge"]),
            ("anchor_edge", by_role["anchor_edge"]),
            ("evidence_probe", "All evidence is now complete"),
        ]
    elif evidence_order == "anchor_first":
        sequence = [
            ("pre_probe", "Preparation marker"),
            ("anchor_edge", by_role["anchor_edge"]),
            ("first_factor_probe", "First factor recorded"),
            ("upper_edge", by_role["upper_edge"]),
            ("switch_edge", by_role["switch_edge"]),
            ("lower_edge", by_role["lower_edge"]),
            ("evidence_probe", "All evidence is now complete"),
        ]
    else:
        raise ValueError(f"unknown evidence order: {evidence_order}")

    pieces: list[str] = []
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
    fact_multiset = tuple(sorted(by_role.values()))
    return "".join(pieces), spans, fact_multiset


def instruction_text(
    task_family: str,
    prompt_branch: str,
    template_index: int,
) -> str:
    natural_target = (
        "Read every fact before answering",
        "Review all comparisons before choosing",
        "Consider all relation evidence before responding",
        "Use all listed facts to decide",
    )
    explicit_target = (
        "Use the ordering facts transitively when necessary",
        "Apply transitive ordering when the answer needs it",
        "Combine the comparison facts transitively when required",
        "Infer through the ordering chain when necessary",
    )
    natural_control = (
        "Read every fact; the answer key after the facts decides",
        "Review the comparisons, then follow the answer key",
        "Consider the evidence, but use the later answer key",
        "Read all listed facts and follow the answer key",
    )
    explicit_control = (
        "Do not infer an ordering answer; copy the later answer key",
        "Ignore transitive inference and use the later answer key",
        "The explicit answer key, not the comparison chain, decides",
        "Copy the provided answer key instead of solving the order",
    )
    table = {
        ("transitive", "natural"): natural_target,
        ("transitive", "explicit"): explicit_target,
        ("direct_key_control", "natural"): natural_control,
        ("direct_key_control", "explicit"): explicit_control,
    }
    return table[(task_family, prompt_branch)][template_index]


def operator_text(template_index: int) -> str:
    return ("Respond", "Give", "Write", "Return")[template_index]


def role_exposure(evidence_order: str) -> dict[str, str]:
    common = {
        "pre_probe": "none",
        "evidence_probe": "both",
        "decision_cue": "both",
        "query_candidate_a": "both",
        "query_candidate_b": "both",
        "operator": "both",
        "query": "both",
        "answer_boundary": "both",
    }
    if evidence_order == "switch_first":
        common.update({
            "upper_edge": "none",
            "switch_edge": "switch_only",
            "first_factor_probe": "switch_only",
            "lower_edge": "switch_only",
            "anchor_edge": "both",
        })
    elif evidence_order == "anchor_first":
        common.update({
            "anchor_edge": "anchor_only",
            "first_factor_probe": "anchor_only",
            "upper_edge": "anchor_only",
            "switch_edge": "both",
            "lower_edge": "both",
        })
    else:
        raise ValueError(f"unknown evidence order: {evidence_order}")
    return common


def render_prompt(
    base_relation: str,
    task_family: str,
    prompt_branch: str,
    evidence_order: str,
    query_type: str,
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
        base_relation,
        names,
        query_type,
        anchor_branch,
        switch_branch,
        answer_branch,
        lexical_branch,
        phrase_set,
    )
    evidence, evidence_spans, fact_multiset = (
        compose_exposure_block(tagged, evidence_order)
    )
    endpoint_a = semantic_names["endpoint_a"]
    endpoint_b = semantic_names["endpoint_b"]
    query_tail = factorial.RELATIONS[base_relation][
        f"{query_type}_query"
    ][phrase_set]
    query = f"Between {endpoint_a} and {endpoint_b}, {query_tail}?"
    decision_cue = (
        "No answer key is provided"
        if task_family == "transitive"
        else f"Answer key: {answer}"
    )
    instruction = instruction_text(
        task_family, prompt_branch, template_index
    )
    operator = operator_text(template_index)
    raw_prompt = (
        f"{instruction}. Facts: {evidence} {decision_cue}. "
        f"Question: {query} {operator} with one person's name "
        f"and nothing else."
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
        "decision_cue": mark(raw_prompt, decision_cue),
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
        "decision_cue": decision_cue,
        "instruction": instruction,
        "path_name": PATH_NAMES[(anchor_branch, switch_branch)],
        "requires_transitive_chain": bool(
            anchor_branch == 0 and switch_branch == 1
        ),
        "fact_multiset": fact_multiset,
        "fact_count": 4,
    }
    return raw_prompt, raw_spans, classes, metadata


def skeleton_hash(raw_prompt: str, names: list[str]) -> str:
    normalized = raw_prompt
    for index, name in sorted(
        enumerate(names), key=lambda item: len(item[1]), reverse=True
    ):
        normalized = normalized.replace(name, f"<N{index}>")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def response_buckets(
    condition: str,
    query_type: str,
    path_name: str,
) -> list[str]:
    parsed = parse_condition(condition)
    return [
        "global:all",
        f"relation:{condition}",
        f"base_relation:{parsed['base_relation']}",
        f"task_family:{parsed['task_family']}",
        f"prompt_branch:{parsed['prompt_branch']}",
        f"evidence_order:{parsed['evidence_order']}",
        f"relation_query:{condition}:{query_type}",
        f"relation_path:{condition}:{path_name}",
        f"path:{path_name}",
    ]


def encode_case(
    tokenizer,
    model_name: str,
    cell_names: list[str],
    condition: str,
    query_type: str,
    template_index: int,
    replicate: int,
    state: str,
    semantic_case_index: int,
    record_prefix: str,
) -> dict[str, Any]:
    parsed = parse_condition(condition)
    anchor, switch, answer, lexical = state_factors(state)
    raw_prompt, raw_spans, classes, metadata = render_prompt(
        parsed["base_relation"],
        parsed["task_family"],
        parsed["prompt_branch"],
        parsed["evidence_order"],
        query_type,
        cell_names,
        anchor,
        switch,
        answer,
        lexical,
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
    unit_id = (
        f"{condition}.{query_type}.t{template_index}.r{replicate}"
    )
    path_name = PATH_NAMES[(anchor, switch)]
    return {
        "schema_version": "phase1072_pattern_case.v1",
        "phase": PHASE,
        "model": model_name,
        "semantic_case_index": semantic_case_index,
        "record_id": (
            f"{record_prefix}.{model_name}.{unit_id}.{state}"
        ),
        "unit_id": unit_id,
        "cell_id": f"{parsed['base_relation']}.{query_type}",
        "relation": condition,
        **parsed,
        "query_type": query_type,
        "layout": "clean",
        "split": split_for_template(template_index),
        "template_index": template_index,
        "replicate": replicate,
        "phrase_set": phrase_set_for_template(template_index),
        "selected_prompt_style": parsed["prompt_branch"],
        "state": state,
        "anchor_branch": anchor,
        "switch_branch": switch,
        "answer_branch": answer,
        "lexical_branch": lexical,
        "path_name": path_name,
        "requires_transitive_chain": metadata[
            "requires_transitive_chain"
        ],
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
                "higher",
                "lower",
                "answer",
            )
        },
        "query_text": metadata["query_text"],
        "query_tail": metadata["query_tail"],
        "instruction": metadata["instruction"],
        "decision_cue_text": metadata["decision_cue"],
        "facts_text": raw_prompt[
            raw_spans["pre_probe"][0]:
            raw_spans["evidence_probe"][1]
        ],
        "fact_multiset": list(metadata["fact_multiset"]),
        "fact_count": metadata["fact_count"],
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
            role: int(span[1])
            for role, span in role_spans.items()
        },
        "role_exposure": role_exposure(
            parsed["evidence_order"]
        ),
        "candidate_labels": classes,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": candidate_first_token_ids,
        "expected_class": f"b{answer}",
        "acceptable_labels": classes[f"b{answer}"],
        "continuation_prefix": " ",
        "response_buckets": response_buckets(
            condition, query_type, path_name
        ),
        "mismatch_unit_id": None,
    }


def assign_mismatch_units(cases: list[dict[str, Any]]) -> None:
    factorial.assign_mismatch_units(cases)


def audit_model(
    model_name: str,
    cases: list[dict[str, Any]],
    calibration_skeletons: set[str],
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)

    role_spans_valid = True
    candidate_single_token = True
    candidate_disjoint = True
    mismatch_disjoint = True
    mismatch_symmetric = True
    exposure_labels_valid = True
    decision_cues_valid = True
    for row in cases:
        width = len(row["input_ids"])
        role_spans_valid = role_spans_valid and all(
            0 <= row["role_spans"][role][0]
            <= row["role_spans"][role][1] < width
            for role in CAPTURE_ROLES
        )
        candidate_single_token = candidate_single_token and all(
            len(values) == 1
            for class_values in row[
                "candidate_token_ids"
            ].values()
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
        exposure_labels_valid = (
            exposure_labels_valid
            and row["role_exposure"]
            == role_exposure(row["evidence_order"])
        )
        if row["task_family"] == "transitive":
            decision_cues_valid = (
                decision_cues_valid
                and row["decision_cue_text"]
                == "No answer key is provided"
            )
        else:
            decision_cues_valid = (
                decision_cues_valid
                and row["decision_cue_text"]
                == "Answer key: "
                + row["semantic_names"]["answer"]
            )

    pre_probe_invariant = True
    first_probe_unseen_factor_invariant = True
    complete_units = True
    query_fixed = True
    answer_truth_valid = True
    fixed_lexical_lengths_equal = True
    for rows in by_unit.values():
        states = {str(row["state"]): row for row in rows}
        complete_units = (
            complete_units and set(states) == set(STATES)
        )
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
        pre_probe_invariant = (
            pre_probe_invariant and len(pre_prefixes) == 1
        )
        order = rows[0]["evidence_order"]
        if order == "switch_first":
            for switch in SWITCH_BRANCHES:
                for answer in ANSWER_BRANCHES:
                    for lexical in LEXICAL_BRANCHES:
                        left = states[
                            f"a0_b{switch}_y{answer}_l{lexical}"
                        ]
                        right = states[
                            f"a1_b{switch}_y{answer}_l{lexical}"
                        ]
                        position = left["role_positions"][
                            "first_factor_probe"
                        ]
                        first_probe_unseen_factor_invariant = (
                            first_probe_unseen_factor_invariant
                            and left["input_ids"][:position + 1]
                            == right["input_ids"][:position + 1]
                        )
        else:
            for anchor in ANCHOR_BRANCHES:
                for answer in ANSWER_BRANCHES:
                    for lexical in LEXICAL_BRANCHES:
                        left = states[
                            f"a{anchor}_b0_y{answer}_l{lexical}"
                        ]
                        right = states[
                            f"a{anchor}_b1_y{answer}_l{lexical}"
                        ]
                        position = left["role_positions"][
                            "first_factor_probe"
                        ]
                        first_probe_unseen_factor_invariant = (
                            first_probe_unseen_factor_invariant
                            and left["input_ids"][:position + 1]
                            == right["input_ids"][:position + 1]
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
                answer_truth_valid = (
                    answer_truth_valid and len(labels) == 1
                )

    evidence_grouped: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    fact_grouped: dict[tuple[Any, ...], set[tuple[str, ...]]] = (
        defaultdict(set)
    )
    for row in cases:
        evidence_grouped[(
            row["base_relation"],
            row["query_type"],
            row["template_index"],
            row["replicate"],
            row["state"],
            row["evidence_order"],
        )].add(row["facts_text"])
        fact_grouped[(
            row["base_relation"],
            row["query_type"],
            row["template_index"],
            row["replicate"],
            row["state"],
            row["task_family"],
            row["prompt_branch"],
        )].add(tuple(row["fact_multiset"]))

    expected_cases = (
        len(RELATION_NAMES)
        * len(QUERY_TYPES)
        * len(LAYOUTS)
        * len(TEMPLATES)
        * len(REPLICATES)
        * len(STATES)
    )
    checks = {
        "case_count": len(cases) == expected_cases,
        "unit_count": len(by_unit) == expected_cases // len(STATES),
        "complete_factorial_units": complete_units,
        "role_spans_valid": role_spans_valid,
        "candidate_continuations_single_token": candidate_single_token,
        "candidate_first_tokens_disjoint": candidate_disjoint,
        "mismatch_candidate_tokens_disjoint": mismatch_disjoint,
        "mismatch_pairing_symmetric": mismatch_symmetric,
        "role_exposure_labels_valid": exposure_labels_valid,
        "decision_cues_valid": decision_cues_valid,
        "pre_probe_prefix_invariant": pre_probe_invariant,
        "first_probe_invariant_to_unseen_factor": (
            first_probe_unseen_factor_invariant
        ),
        "query_fixed_within_unit": query_fixed,
        "answer_truth_preserved_across_paths": answer_truth_valid,
        "fixed_lexical_token_lengths_equal": (
            fixed_lexical_lengths_equal
        ),
        "same_evidence_across_task_and_prompt_branches": all(
            len(values) == 1 for values in evidence_grouped.values()
        ),
        "same_fact_multiset_across_orders": all(
            len(values) == 1 for values in fact_grouped.values()
        ),
        "all_calibration_skeletons_have_formal_match": (
            calibration_skeletons
            <= {
                str(row["prompt_skeleton_sha256"])
                for row in cases
            }
        ),
    }
    return {
        "schema_version": "phase1072_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def build_protocol() -> dict[str, Any]:
    source_next = read_json(
        SOURCE_ROOT / "analysis" / "automatic_next.json"
    )
    source_audit = read_json(
        SOURCE_ROOT / "analysis" / "integrity_audit.json"
    )
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    if (
        bool(source_next["should_continue_automatically"])
        or source_next["route"] != "stop_at_exposure_pattern_atlas"
        or not source_audit["all_integrity_checks_passed"]
    ):
        raise RuntimeError("Phase1071 source decision or audit drift")

    calibration_prereg = read_json(
        CALIBRATION_ROOT / "protocol" / "preregistration.json"
    )
    source_calibration = read_json(
        SOURCE_CALIBRATION_ROOT
        / "protocol"
        / "preregistration.json"
    )
    formal_names = tuple(
        source_calibration["reserved_mechanism_names"]
    )
    calibration_names = set(
        source_calibration["calibration_names"]
    )
    if calibration_names & set(formal_names):
        raise RuntimeError("calibration/formal names overlap")

    model_audits = {}
    case_counts = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        calibration_cases = read_jsonl(
            CALIBRATION_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        calibration_skeletons = {
            str(row["prompt_skeleton_sha256"])
            for row in calibration_cases
        }
        cases: list[dict[str, Any]] = []
        semantic_index = 0
        for condition in RELATION_NAMES:
            parsed = parse_condition(condition)
            for query_type in QUERY_TYPES:
                name_sets = deterministic_name_sets(
                    formal_names,
                    f"{parsed['base_relation']}.{query_type}",
                )
                for template_index in TEMPLATES:
                    for replicate in REPLICATES:
                        cell_names = name_sets[
                            (template_index, replicate)
                        ]
                        for state in STATES:
                            cases.append(encode_case(
                                tokenizer,
                                model_name,
                                cell_names,
                                condition,
                                query_type,
                                template_index,
                                replicate,
                                state,
                                semantic_index,
                                "formal",
                            ))
                            semantic_index += 1
        assign_mismatch_units(cases)
        audit = audit_model(
            model_name, cases, calibration_skeletons
        )
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1072 protocol audit failed: {model_name}: "
                f"{audit}"
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
        case_counts[model_name] = len(cases)

    payload = {
        "schema_version": "phase1072_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "base_relations": list(BASE_RELATIONS),
        "task_families": list(TASK_FAMILIES),
        "prompt_branches": list(PROMPT_BRANCHES),
        "evidence_orders": list(EVIDENCE_ORDERS),
        "condition_count": len(RELATION_NAMES),
        "conditions": list(RELATION_NAMES),
        "query_types": list(QUERY_TYPES),
        "templates": list(TEMPLATES),
        "replicates": list(REPLICATES),
        "states": list(STATES),
        "path_names": {
            f"a{a}_b{b}": value
            for (a, b), value in PATH_NAMES.items()
        },
        "capture_roles": list(CAPTURE_ROLES),
        "primary_process_roles": list(PRIMARY_PROCESS_ROLES),
        "hard_negative_roles": list(HARD_NEGATIVE_ROLES),
        "natural_audit_per_path": NATURAL_AUDIT_PER_PATH,
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "case_counts": case_counts,
        "calibration_protocol_digest": calibration_prereg[
            "protocol_digest"
        ],
        "source_phase1071_digest": source_prereg[
            "protocol_digest"
        ],
        "source_phase1071_automatic_next": source_next,
        "gates": dict(GATES),
        "measurement_definitions": {
            "process_did": (
                "(h[a0,b1]-h[a0,b0])-(h[a1,b1]-h[a1,b0])"
            ),
            "task_specificity": (
                "transitive process magnitude minus matched "
                "direct-key-control process magnitude"
            ),
            "order_reuse": (
                "cosine between relative-depth process profiles "
                "under switch-first and anchor-first evidence"
            ),
            "prompt_reuse": (
                "cosine between natural and explicit prompt "
                "process profiles"
            ),
        },
        "frozen_claim_limits": [
            "A nonzero DiD is an interaction contrast, not a pure reasoning vector.",
            "The direct-key branch is a matched task-demand control, not a complete non-language control.",
            "Profile reuse does not identify a shared physical circuit.",
            "No gate tests optimal compression, brain homology, or training plasticity.",
            "Component localization requires the frozen automatic gate.",
        ],
        "interpretation_limits": [
            "A nonzero DiD is an interaction contrast, not a pure reasoning vector.",
            "The direct-key branch is a matched task-demand control, not a complete non-language control.",
            "Profile reuse does not identify a shared physical circuit.",
            "No gate tests optimal compression, brain homology, or training plasticity.",
            "Component localization requires the frozen automatic gate.",
        ],
        "automatic_authorization": {
            "minimum_models": GATES["minimum_repeated_models"],
            "minimum_relations_per_model": GATES[
                "minimum_strong_relations_per_model"
            ],
            "requires": [
                "exact calibration-to-formal prompt skeleton transfer",
                "formal and calibration behavior gates",
                "FP16 numerical gates",
                "both causal-order hard negatives",
                "natural and explicit prompt reuse",
                "bidirectional order reuse",
                "target-over-control task specificity",
                "lexical and answer reuse",
                "process/answer readout separation",
            ],
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
            "schema_version": "phase1072_protocol_audit.v1",
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
        f"Phase1072 protocol frozen: "
        f"{payload['protocol_digest']} "
        f"conditions={payload['condition_count']} "
        f"cases={payload['case_counts']}"
    )


if __name__ == "__main__":
    main()
