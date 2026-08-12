#!/usr/bin/env python3
"""Freeze the Phase1069 local-coordinate reasoning protocol."""

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


PHASE = 1069
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
NATURAL_AUDIT_PER_RELATION = 100
NATURAL_GENERATION_STEPS = 12
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1069_local_coordinate_reasoning"
)
SOURCE_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1068_reasoning_generalization"
)

# These gates are frozen before any Phase1069 model forward pass.
GATES = {
    "candidate_first_token_accuracy_min": 0.90,
    "semantic_first_natural_rate_min": 0.85,
    "per_query_candidate_accuracy_min": 0.85,
    "valid_semantic_pair_per_relation_min": 180,
    "valid_semantic_pair_per_chain_min": 30,
    "minimum_strong_relations_per_model": 3,
    "minimum_repeated_models": 2,
    "late_depth_start": 0.70,
    "late_lexical_semantic_cosine_min": 0.45,
    "late_matched_readout_positive_rate_min": 0.70,
    "late_matched_vs_mismatch_positive_rate_gap_min": 0.15,
    "relation_fingerprint_retrieval_accuracy_min": 0.60,
    "relation_fingerprint_specificity_gap_min": 0.10,
}

# Discovery and confirmation use different relation wording. This prevents
# identical relation phrases from being the only source of replication.
RELATIONS = {
    "height": {
        "positive": ("is taller than", "stands higher than"),
        "inverse": ("is shorter than", "stands lower than"),
        "max_query": (
            "who is the tallest person in the main chain",
            "which person has the greatest height in the main chain",
        ),
        "min_query": (
            "who is the shortest person in the main chain",
            "which person has the least height in the main chain",
        ),
    },
    "age": {
        "positive": ("is older than", "has lived longer than"),
        "inverse": ("is younger than", "has lived for less time than"),
        "max_query": (
            "who is the oldest person in the main chain",
            "which person has the greatest age in the main chain",
        ),
        "min_query": (
            "who is the youngest person in the main chain",
            "which person has the least age in the main chain",
        ),
    },
    "weight": {
        "positive": ("is heavier than", "weighs more than"),
        "inverse": ("is lighter than", "weighs less than"),
        "max_query": (
            "who is the heaviest person in the main chain",
            "which person has the greatest weight in the main chain",
        ),
        "min_query": (
            "who is the lightest person in the main chain",
            "which person has the least weight in the main chain",
        ),
    },
    "arrival": {
        "positive": ("arrived before", "reached the place earlier than"),
        "inverse": ("arrived after", "reached the place later than"),
        "max_query": (
            "who arrived earliest in the main chain",
            "which person was first to arrive in the main chain",
        ),
        "min_query": (
            "who arrived latest in the main chain",
            "which person was last to arrive in the main chain",
        ),
    },
    "score": {
        "positive": ("has a higher score than", "earned more points than"),
        "inverse": ("has a lower score than", "earned fewer points than"),
        "max_query": (
            "who has the highest score in the main chain",
            "which person earned the most points in the main chain",
        ),
        "min_query": (
            "who has the lowest score in the main chain",
            "which person earned the fewest points in the main chain",
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


def state_factors(state: str) -> tuple[int, int]:
    return int(state[1]), int(state[-1])


def split_for_template(template_index: int) -> str:
    return "discovery" if template_index < 2 else "confirmation"


def phrase_set_for_template(template_index: int) -> int:
    return 0 if template_index < 2 else 1


def mark(text: str, value: str) -> tuple[int, int, str]:
    start = text.find(value)
    if start < 0:
        raise RuntimeError(f"missing marked text: {value!r}")
    return start, start + len(value), value


def cell_name_sets(
    names: tuple[str, ...],
    cell_id: str,
    count: int,
) -> dict[int, list[str]]:
    """Return four disjoint, deterministic name sets for one cell."""
    required = count * len(TEMPLATES)
    if required > len(names):
        raise RuntimeError(
            f"{cell_id} needs {required} names but only {len(names)} exist"
        )
    ranked = sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"phase1069|{cell_id}|{name}".encode("utf-8")
        ).hexdigest(),
    )
    return {
        template_index: ranked[
            template_index * count:
            (template_index + 1) * count
        ]
        for template_index in TEMPLATES
    }


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


def layout_parts(
    clauses: list[str],
    layout: str,
    distractor_names: tuple[str, str],
) -> list[str]:
    distractor = (
        f"Unrelated note: {distractor_names[0]} discussed music with "
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
    phrase_set = phrase_set_for_template(template_index)
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
            phrase_set,
        )
        for index in range(chain_length)
    ]
    displayed = layout_parts(
        clauses,
        layout,
        (names[chain_length + 1], names[chain_length + 2]),
    )
    facts = ". ".join(displayed) + "."
    query = str(
        RELATIONS[relation][f"{query_type}_query"][phrase_set]
    )
    if template_index == 0:
        operator = "Find"
        text = (
            f"Main ordering facts: {facts} {operator} {query}. "
            "Write exactly one person's name and then stop."
        )
    elif template_index == 1:
        operator = "Select"
        text = (
            f"From these main-chain facts, {operator} {query}: {facts} "
            "Put one person's name first and add nothing after it."
        )
    elif template_index == 2:
        operator = "Infer"
        text = (
            f"Evidence about one main chain: {facts} {operator} {query}. "
            "Return exactly the person's name, then end."
        )
    else:
        operator = "Report"
        text = (
            f"{operator} {query} using only this main-chain evidence: "
            f"{facts} Your entire answer must be one person's name."
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
        "global:all",
        f"relation:{relation}",
        f"relation_query:{relation}:{query_type}",
        f"relation_task:{relation}:{task_kind}",
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
    cell_id = f"{relation}.c{chain_length}.{query_type}.{layout}"
    name_sets = cell_name_sets(
        names, cell_id, chain_length + 3
    )
    cell_names = name_sets[template_index]
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
    unit_id = f"{cell_id}.t{template_index}"
    return {
        "schema_version": "phase1069_local_coordinate_case.v1",
        "phase": PHASE,
        "model": model_name,
        "semantic_case_index": semantic_case_index,
        "record_id": f"{model_name}.{unit_id}.{state}",
        "unit_id": unit_id,
        "cell_id": cell_id,
        "relation": relation,
        "chain_length": chain_length,
        "query_type": query_type,
        "layout": layout,
        "task_kind": task_kind,
        "split": split_for_template(template_index),
        "template_index": template_index,
        "phrase_set": phrase_set_for_template(template_index),
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
        "mismatch_unit_id": None,
    }


def assign_mismatch_units(cases: list[dict[str, Any]]) -> None:
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        units[str(row["unit_id"])].append(row)
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for unit_id, rows in units.items():
        row = rows[0]
        groups[(
            row["relation"],
            int(row["chain_length"]),
            row["query_type"],
            row["layout"],
            row["split"],
        )].append(unit_id)
    for key, unit_ids in groups.items():
        values = sorted(unit_ids)
        if len(values) != 2:
            raise RuntimeError(
                f"expected two templates per mismatch group {key}: {values}"
            )
        mapping = {values[0]: values[1], values[1]: values[0]}
        for unit_id in values:
            for row in units[unit_id]:
                row["mismatch_unit_id"] = mapping[unit_id]


def audit_model(
    model_name: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)
    complete_units = all(
        {row["state"] for row in values} == set(STATES)
        for values in by_unit.values()
    )
    candidate_disjoint = True
    mismatch_disjoint = True
    role_spans_valid = True
    lexical_answers_preserved = True
    semantic_answers_change = True
    mismatch_symmetric = True
    for row in cases:
        width = len(row["input_ids"])
        role_spans_valid = role_spans_valid and all(
            0 <= row["role_spans"][role][0]
            <= row["role_spans"][role][1] < width
            for role in CAPTURE_ROLES
        )
        left = set(row["candidate_first_token_ids"]["b0"])
        right = set(row["candidate_first_token_ids"]["b1"])
        candidate_disjoint = (
            candidate_disjoint
            and bool(left)
            and bool(right)
            and left.isdisjoint(right)
        )
        mismatch_id = str(row["mismatch_unit_id"])
        mismatch_row = by_unit[mismatch_id][0]
        mismatch_ids = set(
            mismatch_row["candidate_first_token_ids"]["b0"]
        ) | set(mismatch_row["candidate_first_token_ids"]["b1"])
        mismatch_disjoint = (
            mismatch_disjoint
            and (left | right).isdisjoint(mismatch_ids)
        )
        mismatch_symmetric = (
            mismatch_symmetric
            and str(mismatch_row["mismatch_unit_id"])
            == str(row["unit_id"])
        )
    for values in by_unit.values():
        by_state = {str(row["state"]): row for row in values}
        semantic_answers_change = (
            semantic_answers_change
            and by_state["b0_l0"]["acceptable_labels"]
            != by_state["b1_l0"]["acceptable_labels"]
        )
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
        "mismatch_candidate_tokens_disjoint": mismatch_disjoint,
        "mismatch_pairing_symmetric": mismatch_symmetric,
        "semantic_branch_answers_change": semantic_answers_change,
        "lexical_branch_answers_preserved": lexical_answers_preserved,
        "discovery_confirmation_prompt_disjoint": split_prompts[
            "discovery"
        ].isdisjoint(split_prompts["confirmation"]),
    }
    return {
        "schema_version": "phase1069_protocol_model_audit.v1",
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
    if (
        source_next["should_continue_automatically"]
        or source_next["route"]
        != "stop_and_repair_reasoning_behavior_protocol"
    ):
        raise RuntimeError("Phase1068 source decision drift")
    names = source.common_names()
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
        assign_mismatch_units(cases)
        audit = audit_model(model_name, cases)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1069 audit failed for {model_name}: {audit}"
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
        "schema_version": "phase1069_preregistration.v1",
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
        "assistant_prefill": ASSISTANT_PREFILL,
        "case_count_per_model": 2400,
        "unit_count_per_model": 600,
        "natural_audit_per_relation": NATURAL_AUDIT_PER_RELATION,
        "natural_generation_steps": NATURAL_GENERATION_STEPS,
        "cross_tokenizer_name_count": len(names),
        "cross_tokenizer_names": list(names),
        "gates": dict(GATES),
        "source_phase1068_digest": source_prereg[
            "protocol_digest"
        ],
        "source_phase1068_decision": source_next,
        "evidence_routes": {
            "shared_order_operation": (
                "Test whether different relation domains reuse a common "
                "max/min selection transition in each sample's own answer "
                "coordinate."
            ),
            "relation_domain_signature": (
                "Test whether height, age, weight, arrival, and score retain "
                "distinct centered local fingerprints after the shared "
                "depth response is removed."
            ),
        },
        "measurement_order": [
            "freeze new names, held-out relation wording, gates, and controls",
            "measure candidate behavior without candidate lists",
            "separate semantic-first correctness, strict format, and EOS",
            "capture all-depth residual states at seven semantic roles",
            "measure semantic, surface, and factorial interaction responses",
            "align each semantic delta to its own answer-token coordinate",
            "compare matched answer coordinates with paired mismatched axes",
            "compare transitive chains with direct controls",
            "center relation fingerprints before same-relation retrieval",
            "only then decide whether operation-level causal localization is authorized",
        ],
        "interpretation_limits": [
            "All five domains instantiate ordered max/min selection; shared structure may be an abstract order operation rather than relation identity.",
            "The semantic branch changes entity order and the correct answer identity; answer-coordinate alignment is a readout diagnostic, not proof of reasoning causality.",
            "Applying the final readout to intermediate residuals is a logit-lens measurement, not the model's literal intermediate output.",
            "Mismatched answer axes control answer identity but cannot remove every lexical or positional confound.",
            "Relation fingerprints can contain relation-word semantics and are not automatically reasoning mechanisms.",
            "Internal metrics are reported both unconditionally and behavior-conditioned to expose selection bias.",
            "Direct chains are controls for transitive depth, not a complete no-language baseline.",
            "No result establishes brain homology, plasticity, ecological optimality, or a scale law.",
        ],
        "automatic_next": {
            "continue_only_if": (
                "The shared-order-operation gate passes in at least two "
                "models under discovery and confirmation splits; relation "
                "fingerprint success is reported separately and is not "
                "required for shared-operation reuse."
            ),
            "next_phase": (
                "operation-invariant role/channel causal localization with "
                "matched and mismatched local-coordinate controls"
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
            "schema_version": "phase1069_protocol_audit.v1",
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
