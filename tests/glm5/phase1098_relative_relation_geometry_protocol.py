#!/usr/bin/env python3
"""Freeze Phase1098 prospective relative-relation geometry protocol."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior_tools
import phase1075_relation_polarity_protocol as name_source
import phase1096_comparison_dynamics_protocol as phase1096
import phase1097_conditional_transition_protocol as phase1097


PHASE = 1098
PROTOCOL_REVISION = 2
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
RELATIONS = ("height", "age", "weight", "arrival", "score")
SURFACES = ("en", "zh")
SPLITS = ("discovery", "confirmation")
TEMPLATES = (0, 1, 2, 3)
TEMPLATES_BY_SPLIT = {"discovery": (0, 1), "confirmation": (2, 3)}
PANELS = ("relational", "role_lookup")
TASKS = ("max", "min")
ORIENTATIONS = (0, 1)
CARRIER_ORDERS = (0, 1)
ITEMS_PER_TEMPLATE = 8
ASSISTANT_PREFILL = "Answer:"
CONTINUATION_PREFIX = " "
GENERATION_STEPS = 6
GENERATION_ITEMS_PER_CELL = 2
CAPTURE_ROLES = ("branch_probe", "query_end", "answer_boundary")
PRE_TASK_ROLES = ("branch_probe",)
FIELDS = (
    "relational_representation",
    "lookup_representation",
    "relational_execution",
    "lookup_execution",
    "relational_carrier",
    "lookup_carrier",
)
PRIMARY_FIELD = "relational_execution"
CONTROL_FIELDS = (
    "relational_representation",
    "lookup_representation",
    "lookup_execution",
    "relational_carrier",
    "lookup_carrier",
)
STATES = tuple(
    f"p{panel}_t{task}_o{orientation}_c{order}"
    for panel in PANELS
    for task in TASKS
    for orientation in ORIENTATIONS
    for order in CARRIER_ORDERS
)
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1098_relative_relation_geometry"
SOURCE_PHASE1097 = phase1097.OUT_ROOT / "analysis" / "final_summary.json"
SOURCE_PHASE1097_AUDIT = phase1097.OUT_ROOT / "audit" / "result_audit.json"
SOURCE_BLOCK_AUDIT = OUT_ROOT / "analysis" / "signature_block_audit.json"
REVISION1_ROOT = OUT_ROOT / "revision1_behavior_failure"


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


EVIDENCE_THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_candidate_accuracy": 0.85,
    "minimum_generation_accuracy": 0.70,
    "minimum_relations_per_model": 4,
    "minimum_behavior_models": 2,
    "minimum_hidden_finite_fraction": 0.97,
    "pre_task_tolerance": 1e-8,
    "minimum_geometry_cosine": 0.70,
    "minimum_permutation_margin": 0.02,
    "minimum_field_specificity_advantage": 0.05,
    "minimum_cross_model_cells": 3,
    "minimum_cross_model_pairs": 2,
    "minimum_differential_energy": 1e-5,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": "All source, token, name-holdout, factorial, natural-prefix, and matched-answer audits pass.",
    "P2": "At least two models pass four of five relations in both languages and both panels.",
    "P3": "At least two models pass hidden finiteness, duplicate identity, and exact pre-task-zero audits.",
    "P4": "The answer-boundary relational-execution relation graph repeats across independent templates in both languages, keeps the identity label permutation first by 0.02, and selects itself over every matched field control by 0.05 in two models.",
    "P5": "The same content-specific relation graph transfers English-to-Chinese and Chinese-to-English in both splits in two models.",
    "P6": "Coordinate-free relation graphs align across models in at least three of four language-split cells for at least two model pairs, while selecting relational execution over matched controls.",
    "P7": "Shared and differential energy plus eventwise graph maps are descriptive physical-atlas evidence only; they are not causal localization.",
    "P8": "No local-logit or output-margin value is included in the primary geometry signature.",
}


RELATION_SPECS = {
    "height": {
        "en": {
            "positive": (
                "{left} stands taller than {right}",
                "{left} exceeds {right} in height",
                "{left} has greater height than {right}",
                "{left} is taller than {right}",
            ),
            "max": (
                "Which person stands taller", "Which person has the greater height",
                "Who has greater height", "Who is taller",
            ),
            "min": (
                "Which person stands shorter", "Which person has the lesser height",
                "Who has less height", "Who is shorter",
            ),
            "roles": ("taller", "shorter"),
        },
        "zh": {
            "positive": (
                "{left} 比 {right} 更高",
                "{left} 的身高超过 {right}",
                "{left} 的身高大于 {right}",
                "{left} 比 {right} 更高",
            ),
            "max": ("谁更高", "谁的身高更大", "谁的身高更大", "谁更高"),
            "min": ("谁更矮", "谁的身高更小", "谁的身高更小", "谁更矮"),
            "roles": ("较高", "较矮"),
        },
    },
    "age": {
        "en": {
            "positive": (
                "{left} is older than {right}",
                "{left} exceeds {right} in age",
                "{left} has greater age than {right}",
                "{left} is older than {right}",
            ),
            "max": ("Who is older", "Who has the greater age", "Who has greater age", "Who is older"),
            "min": ("Who is younger", "Who has the lesser age", "Who has less age", "Who is younger"),
            "roles": ("older", "younger"),
        },
        "zh": {
            "positive": (
                "{left} 比 {right} 年长",
                "{left} 的年龄超过 {right}",
                "{left} 的年龄大于 {right}",
                "{left} 比 {right} 年长",
            ),
            "max": ("谁更年长", "谁的年龄更大", "谁的年龄更大", "谁更年长"),
            "min": ("谁更年轻", "谁的年龄更小", "谁的年龄更小", "谁更年轻"),
            "roles": ("年长", "年轻"),
        },
    },
    "weight": {
        "en": {
            "positive": (
                "{left} is heavier than {right}",
                "{left} exceeds {right} in weight",
                "{left} has greater weight than {right}",
                "{left} is heavier than {right}",
            ),
            "max": ("Who is heavier", "Who has the greater weight", "Who has greater weight", "Who is heavier"),
            "min": ("Who is lighter", "Who has the lesser weight", "Who has less weight", "Who is lighter"),
            "roles": ("heavier", "lighter"),
        },
        "zh": {
            "positive": (
                "{left} 比 {right} 更重",
                "{left} 的重量超过 {right}",
                "{left} 的重量大于 {right}",
                "{left} 比 {right} 更重",
            ),
            "max": ("谁更重", "谁的重量更大", "谁的重量更大", "谁更重"),
            "min": ("谁更轻", "谁的重量更小", "谁的重量更小", "谁更轻"),
            "roles": ("较重", "较轻"),
        },
    },
    "arrival": {
        "en": {
            "positive": (
                "{left} arrived before {right}",
                "{left} completed arrival earlier than {right}",
                "{left} arrived earlier than {right}",
                "{left} arrived before {right}",
            ),
            "max": ("Who arrived earlier", "Who completed arrival first", "Who arrived earlier", "Who arrived first"),
            "min": ("Who arrived later", "Who completed arrival last", "Who arrived later", "Who arrived last"),
            "roles": ("earlier", "later"),
        },
        "zh": {
            "positive": (
                "{left} 比 {right} 更早到达",
                "{left} 的到达时间早于 {right}",
                "{left} 到达得比 {right} 更早",
                "{left} 比 {right} 更早到达",
            ),
            "max": ("谁更早到达", "谁最先到达", "谁更早到达", "谁先到达"),
            "min": ("谁更晚到达", "谁最后到达", "谁更晚到达", "谁后到达"),
            "roles": ("较早", "较晚"),
        },
    },
    "score": {
        "en": {
            "positive": (
                "{left} scored higher than {right}",
                "{left} exceeds {right} in score",
                "{left} earned more points than {right}",
                "{left} scored higher than {right}",
            ),
            "max": ("Who scored higher", "Who has the greater score", "Who earned more points", "Who scored higher"),
            "min": ("Who scored lower", "Who has the lesser score", "Who earned fewer points", "Who scored lower"),
            "roles": ("higher-scoring", "lower-scoring"),
        },
        "zh": {
            "positive": (
                "{left} 比 {right} 得分更高",
                "{left} 的分数超过 {right}",
                "{left} 获得的分数多于 {right}",
                "{left} 比 {right} 得分更高",
            ),
            "max": ("谁得分更高", "谁的分数更大", "谁获得的分数更多", "谁得分更高"),
            "min": ("谁得分更低", "谁的分数更小", "谁获得的分数更少", "谁得分更低"),
            "roles": ("高分", "低分"),
        },
    },
}


SHELLS = {
    "en": {
        0: "Observed fact: {fact}. Unrelated display: {carrier}. Query: {question}? Return one person's name.",
        1: "Use this record: {fact}. Neutral order: {carrier}. Please decide: {question}? Answer with one name.",
        2: "A comparison states that {fact}. Incidental sequence: {carrier}. Determine: {question}? Write one person's name only.",
        3: "Consider the evidence that {fact}. Displayed list: {carrier}. Final query: {question}? Give exactly one name.",
    },
    "zh": {
        0: "观察事实： {fact}。无关列表： {carrier}。问题： {question}？只回答一个人的名字。",
        1: "使用这条记录： {fact}。中性顺序： {carrier}。请判断： {question}？用一个名字回答。",
        2: "比较记录表明： {fact}。附带顺序： {carrier}。请确定： {question}？只写一个人的名字。",
        3: "考虑这项证据： {fact}。显示列表： {carrier}。最终问题： {question}？仅给出一个名字。",
    },
}


BRANCH_MARKERS = {
    "en": {0: "Query:", 1: "Please decide:", 2: "Determine:", 3: "Final query:"},
    "zh": {0: "问题：", 1: "请判断：", 2: "请确定：", 3: "最终问题："},
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_for_template(template: int) -> str:
    return "discovery" if template < 2 else "confirmation"


def state_factors(state: str) -> tuple[str, str, int, int]:
    match = re.fullmatch(r"p(.+)_t(max|min)_o([01])_c([01])", state)
    if not match:
        raise ValueError(state)
    return match.group(1), match.group(2), int(match.group(3)), int(match.group(4))


def old_names() -> set[str]:
    values = set(phase1096.prior_names())
    prior = phase1097.OUT_ROOT / "protocol" / "preregistration.json"
    if prior.exists():
        values.update(read_json(prior).get("selected_names", []))
    return values


def select_names(tokenizers: dict[str, Any]) -> tuple[str, ...]:
    excluded = old_names()
    candidates = tuple(dict.fromkeys(
        phase1097.EXTRA_NAME_CANDIDATES
        + phase1096.ADDITIONAL_NAME_CANDIDATES
        + name_source.HELDOUT_NAME_CANDIDATES
    ))
    ranked = sorted(candidates, key=lambda value: sha256_text(f"phase1098|{value}"))
    used_ids = {model: set() for model in MODELS}
    selected: list[str] = []
    required = len(TEMPLATES) * ITEMS_PER_TEMPLATE * 2
    for name in ranked:
        if name in excluded:
            continue
        token_ids: dict[str, int] = {}
        for model, tokenizer in tokenizers.items():
            values = tokenizer.encode(" " + name, add_special_tokens=False)
            if len(values) != 1 or int(values[0]) in used_ids[model]:
                break
            token_ids[model] = int(values[0])
        if len(token_ids) != len(MODELS):
            continue
        selected.append(name)
        for model, token_id in token_ids.items():
            used_ids[model].add(token_id)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise RuntimeError(f"need {required} new one-token names, found {len(selected)}")
    return tuple(selected)


def name_pair(names: tuple[str, ...], template: int, item_index: int) -> tuple[str, str]:
    pair_index = template * ITEMS_PER_TEMPLATE + item_index
    return names[2 * pair_index], names[2 * pair_index + 1]


def mark(text: str, value: str, start: int = 0) -> tuple[int, int, str]:
    position = text.find(value, start)
    if position < 0:
        raise RuntimeError(f"missing marked value {value!r}")
    return position, position + len(value), value


def render_prompt(
    relation: str,
    surface: str,
    template: int,
    panel: str,
    task: str,
    orientation: int,
    order: int,
    names: tuple[str, str],
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, Any]]:
    entity0, entity1 = names
    high, low = (entity0, entity1) if orientation == 0 else (entity1, entity0)
    high_marked, low_marked = f"[ {high} ]", f"[ {low} ]"
    spec = RELATION_SPECS[relation][surface]
    if panel == "relational":
        fact = spec["positive"][template].format(left=high_marked, right=low_marked)
    else:
        max_role, min_role = spec["roles"]
        if surface == "en":
            fact = f"{high_marked} is the {max_role} endpoint while {low_marked} is the {min_role} endpoint"
        else:
            fact = f"{high_marked} 是{max_role}端，{low_marked} 是{min_role}端"
    first, second = (entity0, entity1) if order == 0 else (entity1, entity0)
    first_marked, second_marked = f"[ {first} ]", f"[ {second} ]"
    carrier = (
        f"person {first_marked}; person {second_marked}"
        if surface == "en" else f"人物 {first_marked}；人物 {second_marked}"
    )
    question = spec[task][template]
    raw_prompt = SHELLS[surface][template].format(fact=fact, carrier=carrier, question=question)
    fact_span = mark(raw_prompt, fact)
    carrier_span = mark(raw_prompt, carrier, fact_span[1])
    branch_span = mark(raw_prompt, BRANCH_MARKERS[surface][template], carrier_span[1])
    cue_span = mark(raw_prompt, question, branch_span[1])
    spans = {"branch_probe": branch_span, "query_end": cue_span}
    return raw_prompt, spans, {
        "entity0": entity0, "entity1": entity1, "high": high, "low": low,
        "fact": fact, "carrier": carrier, "cue": question,
    }


def continuation_ids(tokenizer, rendered: str, label: str) -> list[int]:
    base = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    extended = [int(value) for value in tokenizer.encode(rendered + CONTINUATION_PREFIX + label, add_special_tokens=False)]
    if extended[:len(base)] != base:
        raise RuntimeError(f"continuation retokenized prompt for {label!r}")
    suffix = extended[len(base):]
    if not suffix:
        raise RuntimeError(f"empty continuation for {label!r}")
    return suffix


def build_case(
    tokenizer,
    model_name: str,
    selected_names: tuple[str, ...],
    relation: str,
    surface: str,
    template: int,
    item_index: int,
    state: str,
    case_index: int,
) -> dict[str, Any]:
    panel, task, orientation, order = state_factors(state)
    names = name_pair(selected_names, template, item_index)
    raw_prompt, raw_spans, meta = render_prompt(
        relation, surface, template, panel, task, orientation, order, names
    )
    rendered = behavior_tools.render_native(tokenizer, model_name, raw_prompt, with_system=False) + ASSISTANT_PREFILL
    input_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    role_spans = offset_token_spans(tokenizer, rendered, raw_prompt, raw_spans)
    role_positions = {role: int(span[1]) for role, span in role_spans.items()}
    role_positions["answer_boundary"] = len(input_ids) - 1
    role_spans["answer_boundary"] = (len(input_ids) - 1, len(input_ids) - 1)
    expected_entity = meta["high"] if task == "max" else meta["low"]
    expected_class = "e0" if expected_entity == meta["entity0"] else "e1"
    candidate_labels = {"e0": meta["entity0"], "e1": meta["entity1"]}
    candidate_token_ids = {
        key: continuation_ids(tokenizer, rendered, label)
        for key, label in candidate_labels.items()
    }
    unit_id = f"phase1098.{model_name}.{relation}.{surface}.t{template}.i{item_index:02d}"
    return {
        "schema_version": "phase1098_relation_geometry_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{unit_id}.{state}",
        "unit_id": unit_id,
        "superunit_id": f"phase1098.{model_name}.{surface}.t{template}.i{item_index:02d}",
        "relation": relation,
        "surface": surface,
        "split": split_for_template(template),
        "template": template,
        "item_index": item_index,
        "state": state,
        "panel": panel,
        "task": task,
        "orientation": orientation,
        "carrier_order": order,
        "entity0": meta["entity0"],
        "entity1": meta["entity1"],
        "high": meta["high"],
        "low": meta["low"],
        "expected_entity": expected_entity,
        "expected_class": expected_class,
        "candidate_labels": candidate_labels,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {key: [int(values[0])] for key, values in candidate_token_ids.items()},
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_spans": {key: list(value) for key, value in role_spans.items()},
        "role_positions": role_positions,
        "fact_text": meta["fact"],
        "carrier_text": meta["carrier"],
        "task_cue_text": meta["cue"],
        "continuation_prefix": CONTINUATION_PREFIX,
        "prompt_digest": sha256_text(raw_prompt),
    }


def build_model_cases(tokenizer, model_name: str, selected_names: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relation in RELATIONS:
        for surface in SURFACES:
            for template in TEMPLATES:
                for item_index in range(ITEMS_PER_TEMPLATE):
                    for state in STATES:
                        rows.append(build_case(
                            tokenizer, model_name, selected_names, relation, surface,
                            template, item_index, state, len(rows),
                        ))
    return rows


def select_row(rows: list[dict[str, Any]], panel: str, task: str, orientation: int, order: int) -> dict[str, Any]:
    return next(
        row for row in rows
        if row["panel"] == panel and row["task"] == task
        and int(row["orientation"]) == orientation and int(row["carrier_order"]) == order
    )


def audit_model(model_name: str, rows: list[dict[str, Any]], selected_names: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    supergrouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["unit_id"])].append(row)
        supergrouped[str(row["superunit_id"])].append(row)
    checks: dict[str, bool] = {}
    checks["case_count"] = len(rows) == len(RELATIONS) * len(SURFACES) * len(TEMPLATES) * ITEMS_PER_TEMPLATE * len(STATES)
    checks["unit_count"] = len(grouped) == len(RELATIONS) * len(SURFACES) * len(TEMPLATES) * ITEMS_PER_TEMPLATE
    checks["superunit_count"] = len(supergrouped) == len(SURFACES) * len(TEMPLATES) * ITEMS_PER_TEMPLATE
    checks["complete_factorial_states"] = all(len(values) == len(STATES) and {row["state"] for row in values} == set(STATES) for values in grouped.values())
    checks["complete_relations_per_superunit"] = all({row["relation"] for row in values} == set(RELATIONS) for values in supergrouped.values())
    checks["same_entities_across_relations"] = all(len({(row["entity0"], row["entity1"]) for row in values}) == 1 for values in supergrouped.values())
    checks["candidate_continuations_one_token"] = all(all(len(values) == 1 for values in row["candidate_token_ids"].values()) for row in rows)
    checks["candidate_first_tokens_distinct"] = all(row["candidate_first_token_ids"]["e0"] != row["candidate_first_token_ids"]["e1"] for row in rows)
    checks["panels_have_identical_answers"] = all(
        select_row(values, "relational", task, orientation, order)["expected_class"]
        == select_row(values, "role_lookup", task, orientation, order)["expected_class"]
        for values in grouped.values() for task in TASKS for orientation in ORIENTATIONS for order in CARRIER_ORDERS
    )
    checks["carrier_has_no_answer_consequence"] = all(
        select_row(values, panel, task, orientation, 0)["expected_class"]
        == select_row(values, panel, task, orientation, 1)["expected_class"]
        for values in grouped.values() for panel in PANELS for task in TASKS for orientation in ORIENTATIONS
    )
    checks["answer_identity_balanced"] = all(
        Counter(select_row(values, panel, task, orientation, order)["expected_class"] for task in TASKS for orientation in ORIENTATIONS)
        == Counter({"e0": 2, "e1": 2})
        for values in grouped.values() for panel in PANELS for order in CARRIER_ORDERS
    )
    checks["task_prefix_exact_through_branch"] = all(
        select_row(values, panel, "max", orientation, order)["input_ids"][:select_row(values, panel, "max", orientation, order)["role_positions"]["branch_probe"] + 1]
        == select_row(values, panel, "min", orientation, order)["input_ids"][:select_row(values, panel, "min", orientation, order)["role_positions"]["branch_probe"] + 1]
        for values in grouped.values() for panel in PANELS for orientation in ORIENTATIONS for order in CARRIER_ORDERS
    )
    checks["orientation_preserves_token_multiset"] = all(
        Counter(select_row(values, panel, task, 0, order)["input_ids"])
        == Counter(select_row(values, panel, task, 1, order)["input_ids"])
        for values in grouped.values() for panel in PANELS for task in TASKS for order in CARRIER_ORDERS
    )
    checks["carrier_order_preserves_token_multiset"] = all(
        Counter(select_row(values, panel, task, orientation, 0)["input_ids"])
        == Counter(select_row(values, panel, task, orientation, 1)["input_ids"])
        for values in grouped.values() for panel in PANELS for task in TASKS for orientation in ORIENTATIONS
    )
    discovery_names = set(selected_names[:4 * ITEMS_PER_TEMPLATE])
    confirmation_names = set(selected_names[4 * ITEMS_PER_TEMPLATE:])
    checks["name_splits_disjoint"] = not (discovery_names & confirmation_names)
    checks["names_held_out_from_prior_phases"] = not (set(selected_names) & old_names())
    checks["roles_complete_and_ordered"] = all(
        set(row["role_positions"]) == set(CAPTURE_ROLES)
        and row["role_positions"]["branch_probe"] < row["role_positions"]["query_end"] < row["role_positions"]["answer_boundary"]
        for row in rows
    )
    checks["natural_question_has_no_candidate_list"] = all("Candidates:" not in row["raw_prompt"] and "候选" not in row["raw_prompt"] for row in rows)
    result = {
        "schema_version": "phase1098_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(rows),
        "unit_count": len(grouped),
        "superunit_count": len(supergrouped),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(rows),
    }
    result["audit_digest"] = digest(result)
    return result


def main() -> None:
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    selected_names = select_names(tokenizers)
    protocol_root = OUT_ROOT / "protocol"
    model_audits: dict[str, Any] = {}
    case_digests: dict[str, str] = {}
    row_count = 0
    for model_name in MODELS:
        rows = build_model_cases(tokenizers[model_name], model_name, selected_names)
        row_count = len(rows)
        audit = audit_model(model_name, rows, selected_names)
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
        case_digests[model_name] = audit["case_digest"]
        print({"phase": PHASE, "model": model_name, "cases": len(rows), "units": audit["unit_count"], "superunits": audit["superunit_count"], "audit_passed": audit["all_checks_passed"]})
    source_summary = read_json(SOURCE_PHASE1097) if SOURCE_PHASE1097.exists() else {}
    source_audit = read_json(SOURCE_PHASE1097_AUDIT) if SOURCE_PHASE1097_AUDIT.exists() else {}
    block_audit = read_json(SOURCE_BLOCK_AUDIT) if SOURCE_BLOCK_AUDIT.exists() else {}
    revision1_authorization_path = REVISION1_ROOT / "behavior_authorization.json"
    revision1_authorization = read_json(revision1_authorization_path) if revision1_authorization_path.exists() else {}
    prereg = {
        "schema_version": "phase1098_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "relations": list(RELATIONS),
        "surfaces": list(SURFACES),
        "splits": list(SPLITS),
        "templates": list(TEMPLATES),
        "templates_by_split": {key: list(value) for key, value in TEMPLATES_BY_SPLIT.items()},
        "panels": list(PANELS),
        "states": list(STATES),
        "items_per_template": ITEMS_PER_TEMPLATE,
        "case_count_per_model": row_count,
        "unit_count_per_model": len(RELATIONS) * len(SURFACES) * len(TEMPLATES) * ITEMS_PER_TEMPLATE,
        "superunit_count_per_model": len(SURFACES) * len(TEMPLATES) * ITEMS_PER_TEMPLATE,
        "selected_names": list(selected_names),
        "capture_roles": list(CAPTURE_ROLES),
        "fields": list(FIELDS),
        "primary_field": PRIMARY_FIELD,
        "control_fields": list(CONTROL_FIELDS),
        "generation_steps": GENERATION_STEPS,
        "generation_items_per_cell": GENERATION_ITEMS_PER_CELL,
        "geometry_measurement": {
            "unit_of_analysis": "five relations sharing model, language, template, names, and factorial state design",
            "primary_object": "eventwise signed relation-centered 5x5 Gram graph",
            "shared_energy": "squared norm of the five-relation mean divided by mean squared relation norm",
            "differential_energy": "mean squared centered norm divided by mean squared relation norm",
            "primary_band": "relative depth 0.25 through 0.75, residual plus attention and MLP outputs",
            "primary_role": "answer_boundary",
            "forbidden_primary_inputs": ["local logits", "candidate margin", "generation score", "PCA", "learned probe"],
            "permutation_test": "all 5! relation-label permutations",
        },
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "automatic_next_rule": "Only if P1-P6 all pass may Phase1099 test a minimal causal transport map. Otherwise stop automatic continuation and redesign the observed object.",
        "source_phase1097_summary_digest": source_summary.get("summary_digest"),
        "source_phase1097_result_audit_digest": source_audit.get("audit_digest"),
        "source_signature_block_audit_present": bool(block_audit),
        "revision1_behavior_failure": {
            "archived": revision1_authorization_path.exists(),
            "authorization_digest": revision1_authorization.get("authorization_digest"),
            "passing_models": revision1_authorization.get("passing_models", []),
            "reason_for_revision": "The abstract ordering wording introduced an additional relation-to-ordering translation task, concentrated in confirmation template 3 and role_lookup.",
            "unchanged_between_revisions": ["names", "item_count", "factorial states", "token-boundary markers", "thresholds", "geometry gates"],
        },
        "model_case_digests": case_digests,
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    aggregate = {
        "schema_version": "phase1098_protocol_audit.v1",
        "phase": PHASE,
        "model_audits": model_audits,
        "source_phase1097_present": SOURCE_PHASE1097.exists(),
        "source_phase1097_audit_present": SOURCE_PHASE1097_AUDIT.exists(),
        "source_signature_block_audit_present": SOURCE_BLOCK_AUDIT.exists(),
        "revision1_behavior_failure_archived": (REVISION1_ROOT / "behavior_authorization.json").exists(),
    }
    aggregate["all_checks_passed"] = (
        all(audit["all_checks_passed"] for audit in model_audits.values())
        and aggregate["source_phase1097_present"]
        and aggregate["source_phase1097_audit_present"]
        and aggregate["source_signature_block_audit_present"]
        and aggregate["revision1_behavior_failure_archived"]
    )
    aggregate["audit_digest"] = digest(aggregate)
    write_json(protocol_root / "audit.json", aggregate)
    print({"phase": PHASE, "protocol_digest": prereg["protocol_digest"], "all_checks_passed": aggregate["all_checks_passed"], "selected_names": len(selected_names)})


if __name__ == "__main__":
    main()
