#!/usr/bin/env python3
"""Freeze the Phase1073 identical-prefix late-query protocol."""

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
import phase1072_bidirectional_pattern_protocol as source


PHASE = 1073
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
BASE_RELATIONS = source.BASE_RELATIONS
TASK_FAMILIES = ("transitive", "key_copy")
PROMPT_BRANCHES = ("natural", "explicit")
KEY_ALIGNMENTS = ("congruent", "incongruent")
EVIDENCE_ORDERS = source.EVIDENCE_ORDERS
QUERY_TYPES = source.QUERY_TYPES
TEMPLATES = source.TEMPLATES
REPLICATES = (0,)
SPLITS = source.SPLITS
ANCHOR_BRANCHES = source.ANCHOR_BRANCHES
SWITCH_BRANCHES = source.SWITCH_BRANCHES
ANSWER_BRANCHES = source.ANSWER_BRANCHES
LEXICAL_BRANCHES = source.LEXICAL_BRANCHES
STATES = source.STATES
PATH_NAMES = source.PATH_NAMES
ASSISTANT_PREFILL = source.ASSISTANT_PREFILL
NATURAL_AUDIT_PER_CONDITION = 16
NATURAL_GENERATION_STEPS = 10

OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1073_late_query_operation_selection"
)
CALIBRATION_ROOT = OUT_ROOT / "calibration"
SOURCE_ROOT = source.OUT_ROOT
SOURCE_CALIBRATION_ROOT = source.CALIBRATION_ROOT


def condition_key(
    relation: str,
    task_family: str,
    prompt_branch: str,
    key_alignment: str,
    evidence_order: str,
) -> str:
    return "::".join(
        (
            relation,
            task_family,
            prompt_branch,
            key_alignment,
            evidence_order,
        )
    )


def operation_condition_key(
    relation: str,
    prompt_branch: str,
    key_alignment: str,
    evidence_order: str,
) -> str:
    return "::".join(
        (relation, prompt_branch, key_alignment, evidence_order)
    )


def parse_condition(value: str) -> dict[str, str]:
    fields = value.split("::")
    if len(fields) != 5:
        raise ValueError(f"invalid Phase1073 condition: {value}")
    return {
        "base_relation": fields[0],
        "task_family": fields[1],
        "prompt_branch": fields[2],
        "key_alignment": fields[3],
        "evidence_order": fields[4],
    }


RELATION_NAMES = tuple(
    condition_key(
        relation,
        task_family,
        prompt_branch,
        key_alignment,
        evidence_order,
    )
    for relation in BASE_RELATIONS
    for task_family in TASK_FAMILIES
    for prompt_branch in PROMPT_BRANCHES
    for key_alignment in KEY_ALIGNMENTS
    for evidence_order in EVIDENCE_ORDERS
)

OPERATION_CONDITIONS = tuple(
    operation_condition_key(
        relation, prompt_branch, key_alignment, evidence_order
    )
    for relation in BASE_RELATIONS
    for prompt_branch in PROMPT_BRANCHES
    for key_alignment in KEY_ALIGNMENTS
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
    "answer_key",
    "branch_probe",
    "task_cue",
    "query_candidate_a",
    "query_candidate_b",
    "operator",
    "query",
    "answer_boundary",
)
PRIMARY_OPERATION_ROLES = (
    "task_cue",
    "query",
    "answer_boundary",
)
PRE_BRANCH_HARD_NEGATIVE_ROLES = (
    "pre_probe",
    "first_factor_probe",
    "evidence_probe",
    "answer_key",
    "branch_probe",
)

# Frozen before any Phase1073 hidden-state result exists.
GATES = {
    "calibration_candidate_accuracy_min": 0.84,
    "calibration_semantic_first_rate_min": 0.76,
    "formal_candidate_accuracy_min": 0.86,
    "formal_semantic_first_rate_min": 0.76,
    "per_task_candidate_accuracy_min": 0.84,
    "per_alignment_candidate_accuracy_min": 0.80,
    "per_path_candidate_accuracy_min": 0.74,
    "calibration_formal_candidate_gap_max": 0.10,
    "calibration_formal_semantic_gap_max": 0.12,
    "candidate_finite_rate_min": 0.995,
    "internal_finite_rate_min": 0.995,
    "operation_window_start": 0.30,
    "late_depth_start": 0.70,
    "operation_contrast_relative_magnitude_min": 0.003,
    "congruent_operation_relative_magnitude_min": 0.002,
    "congruent_to_incongruent_magnitude_ratio_min": 0.25,
    "operation_lexical_reuse_cosine_min": 0.20,
    "operation_answer_invariance_cosine_min": 0.20,
    "operation_discovery_confirmation_cosine_min": 0.65,
    "operation_order_profile_cosine_min": 0.40,
    "operation_prompt_profile_cosine_min": 0.35,
    "operation_alignment_profile_cosine_min": 0.20,
    "pre_branch_operation_contrast_max": 1e-6,
    "embedding_operation_contrast_max": 1e-5,
    "minimum_strong_relations_per_model": 2,
    "minimum_repeated_models": 2,
}

write_json = source.write_json
write_jsonl = source.write_jsonl
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest
tokenizer_for = source.tokenizer_for
offset_token_spans = source.offset_token_spans
state_factors = source.state_factors
split_for_template = source.split_for_template
phrase_set_for_template = source.phrase_set_for_template
tagged_fact_items = source.tagged_fact_items
compose_exposure_block = source.compose_exposure_block
mark = source.mark


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
            f"phase1073|{cell_id}|{name}".encode("utf-8")
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


def task_cue_text(
    task_family: str,
    prompt_branch: str,
    template_index: int,
) -> str:
    natural_transitive = (
        "Use the comparison facts and disregard the response key",
        "Answer from the comparisons rather than the response key",
        "Rely on the comparison facts, not the response key",
        "Decide from the comparisons and disregard the response key",
    )
    explicit_transitive = (
        "Ignore the response key and infer transitively from the comparisons",
        "Do not copy the key; combine the comparison chain transitively",
        "Use transitive comparison inference and ignore the response key",
        "Infer through the comparison chain, not from the response key",
    )
    natural_key = (
        "Use the response key and disregard the comparison facts",
        "Answer from the response key rather than the comparisons",
        "Rely on the response key, not the comparison facts",
        "Copy the response key and disregard the comparisons",
    )
    explicit_key = (
        "Ignore comparison inference and copy the response key",
        "Do not solve the chain; return the response key",
        "Use direct key copying and ignore transitive comparison",
        "Return the response key without inferring through the chain",
    )
    table = {
        ("transitive", "natural"): natural_transitive,
        ("transitive", "explicit"): explicit_transitive,
        ("key_copy", "natural"): natural_key,
        ("key_copy", "explicit"): explicit_key,
    }
    return table[(task_family, prompt_branch)][template_index]


def operator_text(template_index: int) -> str:
    return ("Respond", "Give", "Write", "Return")[template_index]


def key_answer_for(
    chain_answer: str,
    endpoint_a: str,
    endpoint_b: str,
    key_alignment: str,
) -> str:
    if key_alignment == "congruent":
        return chain_answer
    if key_alignment != "incongruent":
        raise ValueError(f"unknown key alignment: {key_alignment}")
    return endpoint_b if chain_answer == endpoint_a else endpoint_a


def role_exposure(evidence_order: str) -> dict[str, str]:
    result = source.role_exposure(evidence_order)
    result.update({
        "answer_key": "both",
        "branch_probe": "both",
        "task_cue": "both_plus_task",
    })
    return result


def render_prompt(
    base_relation: str,
    task_family: str,
    prompt_branch: str,
    key_alignment: str,
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
    tagged, semantic_names, chain_answer = tagged_fact_items(
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
    key_answer = key_answer_for(
        chain_answer, endpoint_a, endpoint_b, key_alignment
    )
    expected_answer = (
        chain_answer if task_family == "transitive" else key_answer
    )
    query_tail = factorial.RELATIONS[base_relation][
        f"{query_type}_query"
    ][phrase_set]
    query = f"Between {endpoint_a} and {endpoint_b}, {query_tail}?"
    answer_key_text = f"Response key: {key_answer}"
    branch_probe_text = "Operation marker"
    task_cue = task_cue_text(
        task_family, prompt_branch, template_index
    )
    operator = operator_text(template_index)
    raw_prompt = (
        f"Facts: {evidence} {answer_key_text}. "
        f"{branch_probe_text}. Task: {task_cue}. "
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
        "answer_key": mark(raw_prompt, answer_key_text),
        "branch_probe": mark(raw_prompt, branch_probe_text),
        "task_cue": mark(raw_prompt, task_cue),
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
    expected_class = (
        "b0" if expected_answer == endpoint_a else "b1"
    )
    metadata = {
        **semantic_names,
        "chain_answer": chain_answer,
        "key_answer": key_answer,
        "expected_answer": expected_answer,
        "expected_class": expected_class,
        "query_text": query,
        "query_tail": query_tail,
        "task_cue": task_cue,
        "answer_key_text": answer_key_text,
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
        parsed["key_alignment"],
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
    operation_condition = operation_condition_key(
        parsed["base_relation"],
        parsed["prompt_branch"],
        parsed["key_alignment"],
        parsed["evidence_order"],
    )
    unit_suffix = (
        f"{query_type}.t{template_index}.r{replicate}"
    )
    unit_id = f"{condition}.{unit_suffix}"
    operation_unit_id = f"{operation_condition}.{unit_suffix}"
    prefix_unit_id = "::".join((
        parsed["base_relation"],
        parsed["key_alignment"],
        parsed["evidence_order"],
        unit_suffix,
    ))
    return {
        "schema_version": "phase1073_late_query_case.v1",
        "phase": PHASE,
        "model": model_name,
        "semantic_case_index": semantic_case_index,
        "record_id": (
            f"{record_prefix}.{model_name}.{unit_id}.{state}"
        ),
        "unit_id": unit_id,
        "operation_unit_id": operation_unit_id,
        "prefix_unit_id": prefix_unit_id,
        "operation_condition": operation_condition,
        "relation": condition,
        **parsed,
        "query_type": query_type,
        "split": split_for_template(template_index),
        "template_index": template_index,
        "replicate": replicate,
        "phrase_set": phrase_set_for_template(template_index),
        "state": state,
        "anchor_branch": anchor,
        "switch_branch": switch,
        "answer_branch": answer,
        "lexical_branch": lexical,
        "path_name": metadata["path_name"],
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
            )
        },
        "chain_answer": metadata["chain_answer"],
        "key_answer": metadata["key_answer"],
        "expected_answer": metadata["expected_answer"],
        "query_text": metadata["query_text"],
        "query_tail": metadata["query_tail"],
        "task_cue_text": metadata["task_cue"],
        "answer_key_text": metadata["answer_key_text"],
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
        "expected_class": metadata["expected_class"],
        "acceptable_labels": classes[metadata["expected_class"]],
        "continuation_prefix": " ",
    }


def audit_model(
    model_name: str,
    cases: list[dict[str, Any]],
    calibration_skeletons: set[str],
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prefix: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)
        by_operation[str(row["operation_unit_id"])].append(row)
        by_prefix[(str(row["prefix_unit_id"]), str(row["state"]))].append(
            row
        )

    complete_units = all(
        len(rows) == len(STATES)
        and {row["state"] for row in rows} == set(STATES)
        for rows in by_unit.values()
    )
    complete_operation_pairs = all(
        len(rows) == len(STATES) * len(TASK_FAMILIES)
        and {row["task_family"] for row in rows}
        == set(TASK_FAMILIES)
        for rows in by_operation.values()
    )
    role_spans_valid = all(
        all(
            0 <= row["role_spans"][role][0]
            <= row["role_spans"][role][1]
            < len(row["input_ids"])
            for role in CAPTURE_ROLES
        )
        for row in cases
    )
    candidates_single_token = all(
        len(values) == 1
        for row in cases
        for class_values in row["candidate_token_ids"].values()
        for values in class_values
    )
    candidates_disjoint = all(
        set(row["candidate_first_token_ids"]["b0"]).isdisjoint(
            set(row["candidate_first_token_ids"]["b1"])
        )
        for row in cases
    )

    exact_late_branch_prefix = True
    same_prefix_fields = True
    for rows in by_prefix.values():
        if len(rows) != (
            len(TASK_FAMILIES) * len(PROMPT_BRANCHES)
        ):
            exact_late_branch_prefix = False
            same_prefix_fields = False
            continue
        prefixes = []
        for row in rows:
            end = int(row["role_positions"]["branch_probe"]) + 1
            prefixes.append(tuple(row["input_ids"][:end]))
        exact_late_branch_prefix = (
            exact_late_branch_prefix
            and len(set(prefixes)) == 1
        )
        same_prefix_fields = (
            same_prefix_fields
            and len({row["facts_text"] for row in rows}) == 1
            and len({row["answer_key_text"] for row in rows}) == 1
            and len({row["query_text"] for row in rows}) == 1
            and len({tuple(row["cell_names"]) for row in rows}) == 1
        )

    expected_answer_valid = all(
        row["expected_answer"] == (
            row["chain_answer"]
            if row["task_family"] == "transitive"
            else row["key_answer"]
        )
        and row["expected_answer"]
        in (
            row["semantic_names"]["endpoint_a"],
            row["semantic_names"]["endpoint_b"],
        )
        for row in cases
    )
    alignment_valid = all(
        (
            row["chain_answer"] == row["key_answer"]
            if row["key_alignment"] == "congruent"
            else row["chain_answer"] != row["key_answer"]
        )
        for row in cases
    )

    first_probe_unseen_factor_invariant = True
    for rows in by_unit.values():
        states = {str(row["state"]): row for row in rows}
        if set(states) != set(STATES):
            first_probe_unseen_factor_invariant = False
            continue
        order = rows[0]["evidence_order"]
        end = int(
            rows[0]["role_positions"]["first_factor_probe"]
        ) + 1
        for answer in ANSWER_BRANCHES:
            for lexical in LEXICAL_BRANCHES:
                if order == "switch_first":
                    for switch in SWITCH_BRANCHES:
                        left = states[
                            f"a0_b{switch}_y{answer}_l{lexical}"
                        ]
                        right = states[
                            f"a1_b{switch}_y{answer}_l{lexical}"
                        ]
                        first_probe_unseen_factor_invariant = (
                            first_probe_unseen_factor_invariant
                            and left["input_ids"][:end]
                            == right["input_ids"][:end]
                        )
                else:
                    for anchor in ANCHOR_BRANCHES:
                        left = states[
                            f"a{anchor}_b0_y{answer}_l{lexical}"
                        ]
                        right = states[
                            f"a{anchor}_b1_y{answer}_l{lexical}"
                        ]
                        first_probe_unseen_factor_invariant = (
                            first_probe_unseen_factor_invariant
                            and left["input_ids"][:end]
                            == right["input_ids"][:end]
                        )

    same_fact_multiset_across_orders = True
    fact_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in cases:
        fact_groups[(
            row["base_relation"],
            row["task_family"],
            row["prompt_branch"],
            row["key_alignment"],
            row["query_type"],
            row["template_index"],
            row["replicate"],
            row["state"],
        )].append(row)
    for rows in fact_groups.values():
        same_fact_multiset_across_orders = (
            same_fact_multiset_across_orders
            and len(rows) == len(EVIDENCE_ORDERS)
            and len({
                tuple(row["fact_multiset"]) for row in rows
            }) == 1
        )

    formal_skeletons = {
        str(row["prompt_skeleton_sha256"]) for row in cases
    }
    checks = {
        "case_count": len(cases) == (
            len(RELATION_NAMES)
            * len(QUERY_TYPES)
            * len(TEMPLATES)
            * len(REPLICATES)
            * len(STATES)
        ),
        "unit_count": len(by_unit) == (
            len(RELATION_NAMES)
            * len(QUERY_TYPES)
            * len(TEMPLATES)
            * len(REPLICATES)
        ),
        "operation_unit_count": len(by_operation) == (
            len(OPERATION_CONDITIONS)
            * len(QUERY_TYPES)
            * len(TEMPLATES)
            * len(REPLICATES)
        ),
        "complete_factorial_units": complete_units,
        "complete_task_pairs": complete_operation_pairs,
        "role_spans_valid": role_spans_valid,
        "candidate_continuations_single_token": candidates_single_token,
        "candidate_first_tokens_disjoint": candidates_disjoint,
        "exact_late_branch_prefix": exact_late_branch_prefix,
        "same_prefix_semantics": same_prefix_fields,
        "expected_answer_valid": expected_answer_valid,
        "key_alignment_valid": alignment_valid,
        "first_probe_invariant_to_unseen_factor": (
            first_probe_unseen_factor_invariant
        ),
        "same_fact_multiset_across_orders": (
            same_fact_multiset_across_orders
        ),
        "calibration_skeletons_link_to_formal": (
            calibration_skeletons <= formal_skeletons
        ),
        "semantic_indices_contiguous": sorted(
            int(row["semantic_case_index"]) for row in cases
        ) == list(range(len(cases))),
    }
    return {
        "schema_version": "phase1073_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "operation_unit_count": len(by_operation),
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
        or source_next["route"]
        != "stop_at_bidirectional_pattern_specificity"
        or not source_audit["all_integrity_checks_passed"]
    ):
        raise RuntimeError("Phase1072 source decision or audit drift")

    calibration_prereg = read_json(
        CALIBRATION_ROOT / "protocol" / "preregistration.json"
    )
    source_calibration = read_json(
        SOURCE_CALIBRATION_ROOT
        / "protocol"
        / "preregistration.json"
    )
    formal_names = tuple(source_calibration["reserved_formal_names"])
    calibration_names = set(source_calibration["calibration_names"])
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
        name_cache: dict[tuple[str, str], dict[
            tuple[int, int], list[str]
        ]] = {}
        cases: list[dict[str, Any]] = []
        semantic_index = 0
        for condition in RELATION_NAMES:
            parsed = parse_condition(condition)
            for query_type in QUERY_TYPES:
                cache_key = (
                    parsed["base_relation"], query_type
                )
                if cache_key not in name_cache:
                    name_cache[cache_key] = deterministic_name_sets(
                        formal_names,
                        f"{parsed['base_relation']}.{query_type}",
                    )
                name_sets = name_cache[cache_key]
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
        audit = audit_model(
            model_name, cases, calibration_skeletons
        )
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1073 protocol audit failed: {model_name}: "
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
        "schema_version": "phase1073_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "base_relations": list(BASE_RELATIONS),
        "task_families": list(TASK_FAMILIES),
        "prompt_branches": list(PROMPT_BRANCHES),
        "key_alignments": list(KEY_ALIGNMENTS),
        "evidence_orders": list(EVIDENCE_ORDERS),
        "condition_count": len(RELATION_NAMES),
        "operation_condition_count": len(OPERATION_CONDITIONS),
        "conditions": list(RELATION_NAMES),
        "operation_conditions": list(OPERATION_CONDITIONS),
        "query_types": list(QUERY_TYPES),
        "templates": list(TEMPLATES),
        "replicates": list(REPLICATES),
        "states": list(STATES),
        "path_names": {
            f"a{a}_b{b}": value
            for (a, b), value in PATH_NAMES.items()
        },
        "capture_roles": list(CAPTURE_ROLES),
        "primary_operation_roles": list(PRIMARY_OPERATION_ROLES),
        "pre_branch_hard_negative_roles": list(
            PRE_BRANCH_HARD_NEGATIVE_ROLES
        ),
        "natural_audit_per_condition": NATURAL_AUDIT_PER_CONDITION,
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "case_counts": case_counts,
        "calibration_protocol_digest": calibration_prereg[
            "protocol_digest"
        ],
        "source_phase1072_digest": source_prereg["protocol_digest"],
        "source_phase1072_automatic_next": source_next,
        "gates": dict(GATES),
        "measurement_definitions": {
            "within_task_did": (
                "(h[a0,b1]-h[a0,b0])-(h[a1,b1]-h[a1,b0])"
            ),
            "operation_contrast": (
                "within_task_did[transitive] - "
                "within_task_did[key_copy]"
            ),
            "exact_prebranch_control": (
                "task branches have identical token prefixes through "
                "branch_probe, so operation contrast must be zero"
            ),
            "alignment_control": (
                "congruent key and chain answers preserve output "
                "identity; incongruent answers force task selection"
            ),
        },
        "frozen_claim_limits": [
            "A nonzero operation contrast is a task-conditioned path interaction, not a complete reasoning algorithm.",
            "The key-copy task controls path relevance but remains an instruction-following language task.",
            "Congruent and incongruent key conditions separate operation selection from final answer identity only descriptively until intervention.",
            "Profile reuse does not identify a shared physical circuit.",
            "No gate tests optimal compression, brain homology, or training plasticity.",
            "Component localization requires the frozen automatic gate.",
        ],
        "interpretation_limits": [
            "A nonzero operation contrast is a task-conditioned path interaction, not a complete reasoning algorithm.",
            "The key-copy task controls path relevance but remains an instruction-following language task.",
            "Congruent and incongruent key conditions separate operation selection from final answer identity only descriptively until intervention.",
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
                "exact identical prefixes through the late branch probe",
                "formal and held-out calibration behavior gates",
                "FP16 numerical gates",
                "pre-branch operation contrast exactly zero",
                "post-query operation contrast above the frozen floor",
                "congruent output-identity control",
                "lexical and answer reuse",
                "split, order, prompt, and alignment profile reuse",
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
            "schema_version": "phase1073_protocol_audit.v1",
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
        f"Phase1073 protocol frozen: "
        f"{payload['protocol_digest']} "
        f"conditions={payload['condition_count']} "
        f"operation_conditions={payload['operation_condition_count']} "
        f"cases={payload['case_counts']}"
    )


if __name__ == "__main__":
    main()
