#!/usr/bin/env python3
"""Freeze a multi-relation, multi-structure reasoning atlas protocol."""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior


PHASE = 1068
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
RELATION_NAMES = ("height", "age", "weight", "arrival", "score")
CHAIN_LENGTHS = (1, 2, 3, 4, 5)
QUERY_TYPES = ("max", "min")
LAYOUTS = ("forward", "reverse", "distractor")
TEMPLATES = (0, 1, 2, 3)
SPLITS = ("discovery", "confirmation")
STATES = ("b0_l0", "b1_l0", "b0_l1", "b1_l1")
CAPTURE_ROLES = (
    "logical_first",
    "logical_last",
    "query_near_premise",
    "query_far_premise",
    "operator",
    "query",
    "answer_boundary",
)
DIRECTION_ROLES = (
    "logical_first",
    "logical_last",
    "query_near_premise",
    "answer_boundary",
)
ASSISTANT_PREFILL = "Answer:"
NATURAL_AUDIT_PER_RELATION = 80
NATURAL_GENERATION_STEPS = 8
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1068_reasoning_generalization"
)
SOURCE_PHASE1067 = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1067_reasoning_necessity_coalition"
)
GATES = {
    "candidate_first_token_accuracy_min": 0.90,
    "valid_semantic_pair_per_relation_min": 180,
    "valid_semantic_pair_per_chain_min": 30,
    "valid_semantic_pair_per_query_min": 90,
    "natural_exact_rate_min": 0.75,
    "minimum_strong_relations": 2,
    "minimum_repeated_models": 2,
    "late_answer_discovery_confirmation_cosine_min": 0.50,
    "cross_model_depth_profile_cosine_min": 0.75,
}

RELATIONS = {
    "height": {
        "positive": "is taller than",
        "inverse": "is shorter than",
        "max_query": "who is the tallest person in the main chain",
        "min_query": "who is the shortest person in the main chain",
    },
    "age": {
        "positive": "is older than",
        "inverse": "is younger than",
        "max_query": "who is the oldest person in the main chain",
        "min_query": "who is the youngest person in the main chain",
    },
    "weight": {
        "positive": "is heavier than",
        "inverse": "is lighter than",
        "max_query": "who is the heaviest person in the main chain",
        "min_query": "who is the lightest person in the main chain",
    },
    "arrival": {
        "positive": "arrived before",
        "inverse": "arrived after",
        "max_query": "who arrived earliest in the main chain",
        "min_query": "who arrived latest in the main chain",
    },
    "score": {
        "positive": "has a higher score than",
        "inverse": "has a lower score than",
        "max_query": "who has the highest score in the main chain",
        "min_query": "who has the lowest score in the main chain",
    },
}

NAME_CANDIDATES = (
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Maya", "Nora", "Owen", "Paula",
    "Quinn", "Rita", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zane", "Amber", "Bruno", "Celia", "Damon", "Esme", "Felix",
    "Gail", "Hugo", "Ines", "Jonas", "Kara", "Leon", "Mabel", "Nina",
    "Oscar", "Petra", "Ravi", "Sonia", "Theo", "Vera", "Wade", "Xena",
    "Aiden", "Bella", "Caleb", "Diana", "Ethan", "Fiona", "Gavin", "Helen",
    "Isaac", "Julia", "Kevin", "Laura", "Miles", "Naomi", "Peter", "Rachel",
    "Simon", "Teresa", "Ursula", "Vincent", "Willow", "Yvonne", "Aaron",
    "Bianca", "Colin", "Delia", "Edwin", "Flora", "George", "Hannah",
)


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def state_factors(state: str) -> tuple[int, int]:
    return int(state[1]), int(state[-1])


def split_for_template(template_index: int) -> str:
    return "discovery" if template_index < 2 else "confirmation"


def mark(text: str, value: str) -> tuple[int, int, str]:
    start = text.find(value)
    if start < 0:
        raise RuntimeError(f"missing marked text: {value!r}")
    return start, start + len(value), value


def common_names() -> tuple[str, ...]:
    tokenizers = {
        model: tokenizer_for(model) for model in MODELS
    }
    used = {model: set() for model in MODELS}
    selected = []
    for name in NAME_CANDIDATES:
        first_ids = {}
        valid = True
        for model, tokenizer in tokenizers.items():
            ids = tokenizer.encode(
                " " + name, add_special_tokens=False
            )
            if not ids or int(ids[0]) in used[model]:
                valid = False
                break
            first_ids[model] = int(ids[0])
        if not valid:
            continue
        selected.append(name)
        for model, token_id in first_ids.items():
            used[model].add(token_id)
    if len(selected) < 48:
        raise RuntimeError(
            f"only {len(selected)} cross-tokenizer names are distinct"
        )
    return tuple(selected)


def names_for_cell(
    names: tuple[str, ...],
    cell_id: str,
    count: int,
) -> list[str]:
    seed = int(hashlib.sha256(
        cell_id.encode("utf-8")
    ).hexdigest()[:16], 16)
    offset = seed % len(names)
    step = 7
    values = [
        names[(offset + index * step) % len(names)]
        for index in range(count)
    ]
    if len(set(values)) != count:
        raise RuntimeError(f"name cycle collided for {cell_id}")
    return values


def relation_clause(
    relation: str,
    higher: str,
    lower: str,
    lexical_branch: int,
) -> str:
    spec = RELATIONS[relation]
    if lexical_branch == 0:
        return f"{higher} {spec['positive']} {lower}"
    return f"{lower} {spec['inverse']} {higher}"


def layout_parts(
    clauses: list[str],
    layout: str,
    distractor_names: tuple[str, str],
) -> list[str]:
    distractor = (
        f"Unrelated note: {distractor_names[0]} enjoys painting with "
        f"{distractor_names[1]}"
    )
    if layout == "forward":
        return list(clauses)
    if layout == "reverse":
        return list(reversed(clauses))
    ordered = list(reversed(clauses))
    insert_at = len(ordered) // 2
    return ordered[:insert_at] + [distractor] + ordered[insert_at:]


def render_reasoning_prompt(
    relation: str,
    chain_length: int,
    query_type: str,
    layout: str,
    names: list[str],
    semantic_branch: int,
    lexical_branch: int,
    template_index: int,
) -> tuple[
    str,
    dict[str, tuple[int, int, str]],
    dict[str, list[str]],
]:
    base_chain = names[:chain_length + 1]
    logical_order = (
        base_chain
        if semantic_branch == 0
        else list(reversed(base_chain))
    )
    clauses = [
        relation_clause(
            relation,
            logical_order[index],
            logical_order[index + 1],
            lexical_branch,
        )
        for index in range(chain_length)
    ]
    displayed = layout_parts(
        clauses,
        layout,
        (names[chain_length + 1], names[chain_length + 2]),
    )
    facts = ". ".join(displayed) + "."
    query = str(RELATIONS[relation][f"{query_type}_query"])
    if template_index == 0:
        operator = "Determine"
        text = (
            f"Main chain facts: {facts} {operator} {query}. "
            "Return only the person's name."
        )
    elif template_index == 1:
        operator = "Identify"
        text = (
            f"{operator} {query} from these facts: {facts} "
            "Reply with one name only."
        )
    elif template_index == 2:
        operator = "Use"
        text = (
            f"Facts for the main chain: {facts} {operator} only these facts "
            f"to answer {query}. Give just the name."
        )
    else:
        operator = "Decide"
        text = (
            f"{operator} {query}. The relevant facts are: {facts} "
            "Output the name and nothing else."
        )

    query_span = mark(text, query)
    clause_spans = [mark(text, clause) for clause in clauses]
    query_center = 0.5 * (query_span[0] + query_span[1])
    by_distance = sorted(
        clause_spans,
        key=lambda span: abs(
            0.5 * (span[0] + span[1]) - query_center
        ),
    )
    raw_spans = {
        "logical_first": clause_spans[0],
        "logical_last": clause_spans[-1],
        "query_near_premise": by_distance[0],
        "query_far_premise": by_distance[-1],
        "operator": mark(text, operator),
        "query": query_span,
    }
    answer_b0 = (
        base_chain[0] if query_type == "max" else base_chain[-1]
    )
    answer_b1 = (
        base_chain[-1] if query_type == "max" else base_chain[0]
    )
    return text, raw_spans, {
        "b0": [answer_b0],
        "b1": [answer_b1],
    }


def response_buckets(
    relation: str,
    chain_length: int,
    query_type: str,
    layout: str,
) -> list[str]:
    task_kind = "direct" if chain_length == 1 else "transitive"
    return [
        f"relation:{relation}",
        f"relation_query:{relation}:{query_type}",
        f"chain_query:{chain_length}:{query_type}",
        f"layout:{task_kind}:{layout}",
        f"task_kind:{task_kind}",
    ]


def build_model_case(
    tokenizer,
    model_name: str,
    names: tuple[str, ...],
    relation: str,
    chain_length: int,
    query_type: str,
    layout: str,
    template_index: int,
    state: str,
    semantic_case_index: int,
) -> dict[str, Any]:
    semantic_branch, lexical_branch = state_factors(state)
    cell_id = (
        f"{relation}.c{chain_length}.{query_type}.{layout}"
    )
    cell_names = names_for_cell(
        names, cell_id, chain_length + 3
    )
    raw_prompt, raw_spans, classes = render_reasoning_prompt(
        relation,
        chain_length,
        query_type,
        layout,
        cell_names,
        semantic_branch,
        lexical_branch,
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
    expected_class = f"b{semantic_branch}"
    task_kind = "direct" if chain_length == 1 else "transitive"
    return {
        "schema_version": "phase1068_reasoning_case.v1",
        "phase": PHASE,
        "model": model_name,
        "semantic_case_index": semantic_case_index,
        "record_id": (
            f"{model_name}.{cell_id}.t{template_index}.{state}"
        ),
        "unit_id": f"{cell_id}.t{template_index}",
        "cell_id": cell_id,
        "relation": relation,
        "chain_length": chain_length,
        "query_type": query_type,
        "layout": layout,
        "task_kind": task_kind,
        "split": split_for_template(template_index),
        "template_index": template_index,
        "state": state,
        "semantic_branch": semantic_branch,
        "lexical_branch": lexical_branch,
        "cell_names": cell_names,
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
        "expected_class": expected_class,
        "acceptable_labels": classes[expected_class],
        "continuation_prefix": " ",
        "response_buckets": response_buckets(
            relation, chain_length, query_type, layout
        ),
    }


def audit_model(
    model_name: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for row in cases:
        by_unit.setdefault(str(row["unit_id"]), []).append(row)
    candidate_disjoint = True
    role_spans_valid = True
    branch_answers_change = True
    lexical_answers_preserved = True
    for row in cases:
        width = len(row["input_ids"])
        for role in CAPTURE_ROLES:
            start, end = row["role_spans"][role]
            role_spans_valid = (
                role_spans_valid and 0 <= start <= end < width
            )
        left = set(row["candidate_first_token_ids"]["b0"])
        right = set(row["candidate_first_token_ids"]["b1"])
        candidate_disjoint = (
            candidate_disjoint
            and bool(left)
            and bool(right)
            and left.isdisjoint(right)
        )
        branch_answers_change = (
            branch_answers_change
            and row["candidate_labels"]["b0"]
            != row["candidate_labels"]["b1"]
        )
    for values in by_unit.values():
        by_state = {str(row["state"]): row for row in values}
        for branch in (0, 1):
            lexical_answers_preserved = (
                lexical_answers_preserved
                and by_state[f"b{branch}_l0"]["acceptable_labels"]
                == by_state[f"b{branch}_l1"]["acceptable_labels"]
            )
    counts = Counter(
        (
            row["relation"],
            int(row["chain_length"]),
            row["query_type"],
            row["layout"],
            row["split"],
        )
        for row in cases
    )
    complete_units = all(
        {row["state"] for row in values} == set(STATES)
        for values in by_unit.values()
    )
    split_prompts = {
        split: {
            row["rendered_prompt"]
            for row in cases
            if row["split"] == split
        }
        for split in SPLITS
    }
    checks = {
        "case_count": len(cases) == 2400,
        "unit_count": len(by_unit) == 600,
        "complete_factorial_units": complete_units,
        "balanced_cells": all(
            counts[(relation, chain, query, layout, split)] == 8
            for relation in RELATION_NAMES
            for chain in CHAIN_LENGTHS
            for query in QUERY_TYPES
            for layout in LAYOUTS
            for split in SPLITS
        ),
        "role_spans_valid": role_spans_valid,
        "candidate_first_tokens_disjoint": candidate_disjoint,
        "semantic_branch_answers_change": branch_answers_change,
        "lexical_branch_answers_preserved": (
            lexical_answers_preserved
        ),
        "discovery_confirmation_prompt_disjoint": split_prompts[
            "discovery"
        ].isdisjoint(split_prompts["confirmation"]),
    }
    return {
        "schema_version": "phase1068_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def build_protocol() -> dict[str, Any]:
    source_prereg = read_json(
        SOURCE_PHASE1067 / "protocol" / "preregistration.json"
    )
    source_next = read_json(
        SOURCE_PHASE1067 / "analysis" / "automatic_next.json"
    )
    if (
        source_next["should_continue_automatically"]
        or source_next["route"]
        != "stop_at_controlled_reasoning_milestone"
    ):
        raise RuntimeError("Phase1067 source decision drift")
    names = common_names()
    model_audits = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        cases = []
        semantic_case_index = 0
        for relation in RELATION_NAMES:
            for chain_length in CHAIN_LENGTHS:
                for query_type in QUERY_TYPES:
                    for layout in LAYOUTS:
                        for template_index in TEMPLATES:
                            for state in STATES:
                                cases.append(build_model_case(
                                    tokenizer,
                                    model_name,
                                    names,
                                    relation,
                                    chain_length,
                                    query_type,
                                    layout,
                                    template_index,
                                    state,
                                    semantic_case_index,
                                ))
                                semantic_case_index += 1
        audit = audit_model(model_name, cases)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1068 audit failed for {model_name}: {audit}"
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

    bucket_ids = sorted({
        bucket
        for relation in RELATION_NAMES
        for chain_length in CHAIN_LENGTHS
        for query_type in QUERY_TYPES
        for layout in LAYOUTS
        for bucket in response_buckets(
            relation, chain_length, query_type, layout
        )
    })
    payload = {
        "schema_version": "phase1068_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "relations": list(RELATION_NAMES),
        "chain_lengths": list(CHAIN_LENGTHS),
        "query_types": list(QUERY_TYPES),
        "layouts": list(LAYOUTS),
        "templates": list(TEMPLATES),
        "splits": list(SPLITS),
        "states": list(STATES),
        "capture_components": ["residual"],
        "capture_roles": list(CAPTURE_ROLES),
        "direction_roles": list(DIRECTION_ROLES),
        "response_buckets": bucket_ids,
        "assistant_prefill": ASSISTANT_PREFILL,
        "case_count_per_model": 2400,
        "unit_count_per_model": 600,
        "semantic_pair_count_per_relation_model": 240,
        "natural_audit_per_relation": NATURAL_AUDIT_PER_RELATION,
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "cross_tokenizer_name_count": len(names),
        "cross_tokenizer_names": list(names),
        "gates": dict(GATES),
        "source_phase1067_digest": source_prereg[
            "protocol_digest"
        ],
        "source_phase1067_decision": source_next,
        "measurement_order": [
            "freeze relations, chains, queries, layouts, roles, and gates",
            "measure candidate-absent first-token behavior",
            "audit stratified natural generation",
            "map residual differences at all depths and roles",
            "compare semantic direction across inverse lexical forms",
            "compare discovery and confirmation templates",
            "compare normalized response profiles across models",
            "only then authorize a generalized causal test",
        ],
        "interpretation_limits": [
            "These are synthetic ordered chains, not general reasoning.",
            "Changing semantic branch also changes entity order inside clauses.",
            "Inverse lexical forms are controlled paraphrases, not every natural paraphrase.",
            "Residual response repetition is not causal transport.",
            "Candidate names occur in the facts and can support retrieval or copying.",
            "Direct chains are controls, not a complete no-reasoning baseline.",
            "No result establishes brain homology, plasticity, or efficiency optimality.",
        ],
        "automatic_next": {
            "continue_only_if": (
                "At least two relations repeat strong behavior, late "
                "answer-direction stability, and cross-model residual "
                "depth profiles in at least two models."
            ),
            "next_phase": (
                "role-position-orthogonal K/V causal replication across "
                "relations, chain lengths, queries, and layouts"
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
            "schema_version": "phase1068_protocol_audit.v1",
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
        f"names={payload['cross_tokenizer_name_count']}"
    )


if __name__ == "__main__":
    main()
