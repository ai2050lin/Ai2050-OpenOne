#!/usr/bin/env python3
"""Freeze Phase1097 natural conditional-transition atlas protocol.

The phase does not assume that a language operation is a fixed vector.  It
measures per-item full-depth transition invariants before aggregation.  A
relational panel and a direct role-lookup panel share the answer mapping; the
lookup panel is a comparison trajectory rather than a quantity subtracted from
the relational trajectory.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior_tools
import phase1075_relation_polarity_protocol as name_source
import phase1096_comparison_dynamics_protocol as phase1096


PHASE = 1097
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
RELATIONS = ("height", "age", "weight", "arrival", "score")
SURFACES = ("en", "zh")
SPLITS = ("discovery", "confirmation")
TEMPLATES = (0, 1, 2, 3)
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
PANELS = ("relational", "role_lookup")
TASKS = ("max", "min")
ORIENTATIONS = (0, 1)
CARRIER_ORDERS = (0, 1)
ITEMS_PER_TEMPLATE = 6
ASSISTANT_PREFILL = "Answer:"
CONTINUATION_PREFIX = " "
GENERATION_STEPS = 6
GENERATION_ITEMS_PER_CELL = 2
CAPTURE_ROLES = (
    "fact_end",
    "carrier_end",
    "branch_probe",
    "task_cue",
    "query_end",
    "answer_boundary",
)
PRE_TASK_ROLES = ("fact_end", "carrier_end", "branch_probe")
DYNAMIC_ROLES = ("task_cue", "query_end", "answer_boundary")
FIELDS = (
    "relational_representation",
    "lookup_representation",
    "relational_control",
    "lookup_control",
    "relational_execution",
    "lookup_execution",
    "relational_carrier",
    "lookup_carrier",
)
TRAJECTORY_FIELDS = (
    "relational_execution",
    "lookup_execution",
    "relational_carrier",
    "lookup_carrier",
)
DEPTH_ANCHORS = tuple(index / 12.0 for index in range(13))
STATES = tuple(
    f"p{panel}_t{task}_o{orientation}_c{order}"
    for panel in PANELS
    for task in TASKS
    for orientation in ORIENTATIONS
    for order in CARRIER_ORDERS
)
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1097_conditional_transition_atlas"
)
SOURCE_PHASE1096 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1096_comparison_primitive_dynamics"
    / "analysis" / "final_summary.json"
)
SOURCE_PHASE1096_PREREG = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1096_comparison_primitive_dynamics"
    / "protocol" / "preregistration.json"
)


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


RELATION_SPECS = {
    "height": {
        "en": {
            "positive": ("{left} is taller than {right}", "{left} has greater height than {right}"),
            "max": ("Who is taller", "Who has greater height"),
            "min": ("Who is shorter", "Who has less height"),
            "roles": ("taller", "shorter"),
        },
        "zh": {
            "positive": ("{left}比{right}更高", "{left}的身高大于{right}"),
            "max": ("谁更高", "谁的身高更大"),
            "min": ("谁更矮", "谁的身高更小"),
            "roles": ("较高", "较矮"),
        },
    },
    "age": {
        "en": {
            "positive": ("{left} is older than {right}", "{left} has greater age than {right}"),
            "max": ("Who is older", "Who has greater age"),
            "min": ("Who is younger", "Who has less age"),
            "roles": ("older", "younger"),
        },
        "zh": {
            "positive": ("{left}比{right}年长", "{left}的年龄大于{right}"),
            "max": ("谁更年长", "谁的年龄更大"),
            "min": ("谁更年轻", "谁的年龄更小"),
            "roles": ("年长", "年轻"),
        },
    },
    "weight": {
        "en": {
            "positive": ("{left} is heavier than {right}", "{left} has greater weight than {right}"),
            "max": ("Who is heavier", "Who has greater weight"),
            "min": ("Who is lighter", "Who has less weight"),
            "roles": ("heavier", "lighter"),
        },
        "zh": {
            "positive": ("{left}比{right}更重", "{left}的重量大于{right}"),
            "max": ("谁更重", "谁的重量更大"),
            "min": ("谁更轻", "谁的重量更小"),
            "roles": ("较重", "较轻"),
        },
    },
    "arrival": {
        "en": {
            "positive": ("{left} arrived before {right}", "{left} completed arrival earlier than {right}"),
            "max": ("Who arrived earlier", "Who completed arrival first"),
            "min": ("Who arrived later", "Who completed arrival last"),
            "roles": ("earlier", "later"),
        },
        "zh": {
            "positive": ("{left}比{right}更早到达", "{left}的到达时间早于{right}"),
            "max": ("谁更早到达", "谁最先到达"),
            "min": ("谁更晚到达", "谁最后到达"),
            "roles": ("较早", "较晚"),
        },
    },
    "score": {
        "en": {
            "positive": ("{left} scored higher than {right}", "{left} earned more points than {right}"),
            "max": ("Who scored higher", "Who earned more points"),
            "min": ("Who scored lower", "Who earned fewer points"),
            "roles": ("higher-scoring", "lower-scoring"),
        },
        "zh": {
            "positive": ("{left}比{right}得分更高", "{left}获得的分数多于{right}"),
            "max": ("谁得分更高", "谁获得的分数更多"),
            "min": ("谁得分更低", "谁获得的分数更少"),
            "roles": ("高分", "低分"),
        },
    },
}


SHELLS = {
    "en": {
        0: "Evidence: {fact}. Neutral listing: {carrier}. Question: {question}? Reply with one person's name.",
        1: "Record: {fact}. Reference order: {carrier}. Now answer: {question}? Give only one name.",
        2: "The following relation was recorded: {fact}. Incidental listing: {carrier}. Decision: {question}? Write one name only.",
        3: "Review this statement: {fact}. Display order: {carrier}. Final question: {question}? Respond with one name.",
    },
    "zh": {
        0: "证据：{fact}。中性列举：{carrier}。问题：{question}？只回答一个人的姓名。",
        1: "记录：{fact}。参考顺序：{carrier}。现在回答：{question}？只给出一个姓名。",
        2: "已记录以下关系：{fact}。附带列举：{carrier}。判断：{question}？只写一个姓名。",
        3: "核对这项陈述：{fact}。显示顺序：{carrier}。最终问题：{question}？用一个姓名回答。",
    },
}


BRANCH_MARKERS = {
    "en": {0: "Question:", 1: "Now answer:", 2: "Decision:", 3: "Final question:"},
    "zh": {0: "问题：", 1: "现在回答：", 2: "判断：", 3: "最终问题："},
}


EVIDENCE_THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_candidate_accuracy": 0.85,
    "minimum_generation_accuracy": 0.70,
    "minimum_relations_per_model": 4,
    "minimum_behavior_models": 2,
    "minimum_hidden_finite_fraction": 0.97,
    "pre_task_tolerance": 1e-8,
    "local_readout_tolerance": 0.08,
    "minimum_split_trajectory_cosine": 0.70,
    "minimum_split_records": 16,
    "minimum_heldout_relations": 4,
    "minimum_content_over_carrier_advantage": 0.05,
    "minimum_behavior_anchor_cells": 16,
    "maximum_carrier_to_execution_ratio": 0.20,
    "minimum_panel_convergence_rise": 0.10,
    "minimum_panel_convergence_advantage": 0.05,
    "minimum_cross_language_directions": 1,
    "minimum_cross_model_profile_cosine": 0.70,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": "All name-holdout, factorial, natural-prefix, token, answer-balance, and source audits pass.",
    "P2": "At least two models pass four of five relations in both languages and both panels.",
    "P3": "At least two models pass hidden finite-state, exact pre-task-zero, identity, and local-readout audits.",
    "P4": "Per-item transition signatures repeat across independent names, phrases, and templates in at least 16 of 20 relation-surface-role-field records in two models.",
    "P5": "Relational execution transition signatures predict at least four held-out relations in both languages and beat matched carrier signatures by 0.05 in two models.",
    "P6": "The final local-logit relational execution interaction is behavior aligned while the carrier interaction is at most 20 percent as large in at least 16 of 20 cells in two models.",
    "P7": "Relational and lookup execution trajectories converge from early to late depth by 0.10 and exceed carrier convergence rise by 0.05 in at least 16 of 20 cells in two models.",
    "P8": "Transition signatures transfer between English and Chinese with carrier advantage in at least one direction in two models.",
    "P9": "Cross-model scalar phase profiles are descriptive only; no causal localization is authorized in Phase1097.",
}


EXTRA_NAME_CANDIDATES = (
    "Aaron", "Ada", "Adele", "Adrian", "Agnes", "Aidan", "Albert", "Alexis",
    "Alfred", "Alice", "Allen", "Amanda", "Amy", "Angela", "Angelo", "Anna",
    "Annie", "Arthur", "Audrey", "Barbara", "Barry", "Benjamin", "Bernard",
    "Bianca", "Brandon", "Brenda", "Brian", "Bruce", "Caleb", "Calvin",
    "Carla", "Carol", "Caroline", "Catherine", "Charles", "Charlotte", "Claudia",
    "Colin", "Crystal", "Daniel", "Deborah", "Dennis", "Derek", "Diana", "Dolores",
    "Donna", "Doris", "Dorothy", "Douglas", "Edward", "Elaine", "Elizabeth",
    "Emily", "Emma", "Eric", "Ethan", "Eugene", "Evelyn", "Fiona", "Florence",
    "Frank", "Frederick", "Gavin", "George", "Georgia", "Gloria", "Grace", "Gregory",
    "Harold", "Harry", "Hazel", "Helen", "Henry", "Hugh", "Irene", "Isaac",
    "Jack", "Jacob", "Janet", "Jason", "Jean", "Jennifer", "Jessica", "Joanna",
    "John", "Jonah", "Joseph", "Joyce", "Judith", "Julia", "Justin", "Karen",
    "Kevin", "Laura", "Lauren", "Leo", "Leonard", "Leslie", "Linda", "Lisa",
    "Louise", "Lucy", "Luke", "Maria", "Marie", "Martin", "Mary", "Matthew",
    "Maureen", "Melissa", "Michael", "Michelle", "Molly", "Natalie", "Nicholas",
    "Nicole", "Noah", "Norman", "Pamela", "Patricia", "Paul", "Peter", "Philip",
    "Rachel", "Raymond", "Rebecca", "Richard", "Robert", "Roger", "Rose", "Roy",
    "Russell", "Ruth", "Sarah", "Scott", "Sharon", "Simon", "Sophie", "Stephen",
    "Susan", "Teresa", "Thomas", "Timothy", "Tracy", "Victor", "Vincent", "Wendy",
    "William", "Yvonne",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_for_template(template: int) -> str:
    return "discovery" if template < 2 else "confirmation"


def phrase_index(template: int) -> int:
    return template % 2


def state_factors(state: str) -> tuple[str, str, int, int]:
    match = re.fullmatch(r"p(.+)_t(max|min)_o([01])_c([01])", state)
    if not match:
        raise ValueError(state)
    return match.group(1), match.group(2), int(match.group(3)), int(match.group(4))


def excluded_names() -> set[str]:
    excluded = set(phase1096.prior_names())
    if SOURCE_PHASE1096_PREREG.exists():
        excluded.update(read_json(SOURCE_PHASE1096_PREREG).get("selected_names", []))
    return excluded


def select_names(tokenizers: dict[str, Any]) -> tuple[str, ...]:
    excluded = excluded_names()
    eligible: list[str] = []
    used_ids = {model: set() for model in MODELS}
    candidates = tuple(dict.fromkeys(
        EXTRA_NAME_CANDIDATES
        + phase1096.ADDITIONAL_NAME_CANDIDATES
        + name_source.HELDOUT_NAME_CANDIDATES
    ))
    ranked = sorted(candidates, key=lambda value: sha256_text(f"phase1097|{value}"))
    required = len(TEMPLATES) * ITEMS_PER_TEMPLATE * 2
    for name in ranked:
        if name in excluded:
            continue
        token_ids: dict[str, int] = {}
        valid = True
        for model, tokenizer in tokenizers.items():
            values = tokenizer.encode(" " + name, add_special_tokens=False)
            if len(values) != 1 or int(values[0]) in used_ids[model]:
                valid = False
                break
            token_ids[model] = int(values[0])
        if not valid:
            continue
        eligible.append(name)
        for model, token_id in token_ids.items():
            used_ids[model].add(token_id)
        if len(eligible) == required:
            break
    if len(eligible) != required:
        raise RuntimeError(f"need {required} new one-token names, found {len(eligible)}")
    return tuple(eligible)


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
    spec = RELATION_SPECS[relation][surface]
    pindex = phrase_index(template)
    if panel == "relational":
        fact = spec["positive"][pindex].format(left=high, right=low)
    else:
        max_role, min_role = spec["roles"]
        if surface == "en":
            fact = f"{high} is the {max_role} endpoint and {low} is the {min_role} endpoint"
        else:
            fact = f"{high}是{max_role}端，{low}是{min_role}端"
    first, second = (entity0, entity1) if order == 0 else (entity1, entity0)
    carrier = (
        f"person {first}; person {second}"
        if surface == "en"
        else f"人物 {first}；人物 {second}"
    )
    question = spec[task][pindex]
    raw_prompt = SHELLS[surface][template].format(
        fact=fact, carrier=carrier, question=question
    )
    fact_span = mark(raw_prompt, fact)
    carrier_span = mark(raw_prompt, carrier, fact_span[1])
    branch_span = mark(raw_prompt, BRANCH_MARKERS[surface][template], carrier_span[1])
    cue_span = mark(raw_prompt, question, branch_span[1])
    spans = {
        "fact_end": fact_span,
        "carrier_end": carrier_span,
        "branch_probe": branch_span,
        "task_cue": cue_span,
        "query_end": cue_span,
    }
    return raw_prompt, spans, {
        "entity0": entity0,
        "entity1": entity1,
        "high": high,
        "low": low,
        "fact": fact,
        "carrier": carrier,
        "cue": question,
    }


def continuation_ids(tokenizer, rendered: str, label: str) -> list[int]:
    base = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    extended = [int(value) for value in tokenizer.encode(
        rendered + CONTINUATION_PREFIX + label, add_special_tokens=False
    )]
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
    split = split_for_template(template)
    panel, task, orientation, order = state_factors(state)
    names = name_pair(selected_names, template, item_index)
    raw_prompt, raw_spans, meta = render_prompt(
        relation, surface, template, panel, task, orientation, order, names
    )
    rendered = behavior_tools.render_native(
        tokenizer, model_name, raw_prompt, with_system=False
    ) + ASSISTANT_PREFILL
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
    unit_id = f"phase1097.{model_name}.{relation}.{surface}.t{template}.i{item_index:02d}"
    return {
        "schema_version": "phase1097_transition_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{unit_id}.{state}",
        "unit_id": unit_id,
        "relation": relation,
        "surface": surface,
        "split": split,
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
        "candidate_first_token_ids": {
            key: [int(values[0])] for key, values in candidate_token_ids.items()
        },
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
    case_index = 0
    for relation in RELATIONS:
        for surface in SURFACES:
            for template in TEMPLATES:
                for item_index in range(ITEMS_PER_TEMPLATE):
                    for state in STATES:
                        rows.append(build_case(
                            tokenizer, model_name, selected_names, relation, surface,
                            template, item_index, state, case_index,
                        ))
                        case_index += 1
    return rows


def select_row(rows: list[dict[str, Any]], panel: str, task: str, orientation: int, order: int) -> dict[str, Any]:
    return next(
        row for row in rows
        if row["panel"] == panel
        and row["task"] == task
        and int(row["orientation"]) == orientation
        and int(row["carrier_order"]) == order
    )


def audit_model(model_name: str, rows: list[dict[str, Any]], selected_names: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["unit_id"])].append(row)
    checks: dict[str, bool] = {}
    checks["case_count"] = len(rows) == len(RELATIONS) * len(SURFACES) * len(TEMPLATES) * ITEMS_PER_TEMPLATE * len(STATES)
    checks["unit_count"] = len(grouped) == len(RELATIONS) * len(SURFACES) * len(TEMPLATES) * ITEMS_PER_TEMPLATE
    checks["complete_factorial_states"] = all(
        len(unit_rows) == len(STATES) and {row["state"] for row in unit_rows} == set(STATES)
        for unit_rows in grouped.values()
    )
    checks["candidate_continuations_one_token"] = all(
        all(len(values) == 1 for values in row["candidate_token_ids"].values()) for row in rows
    )
    checks["candidate_first_tokens_distinct"] = all(
        row["candidate_first_token_ids"]["e0"] != row["candidate_first_token_ids"]["e1"] for row in rows
    )
    checks["panels_have_identical_answers"] = all(
        select_row(unit_rows, "relational", task, orientation, order)["expected_class"]
        == select_row(unit_rows, "role_lookup", task, orientation, order)["expected_class"]
        for unit_rows in grouped.values() for task in TASKS for orientation in ORIENTATIONS for order in CARRIER_ORDERS
    )
    checks["carrier_has_no_answer_consequence"] = all(
        select_row(unit_rows, panel, task, orientation, 0)["expected_class"]
        == select_row(unit_rows, panel, task, orientation, 1)["expected_class"]
        for unit_rows in grouped.values() for panel in PANELS for task in TASKS for orientation in ORIENTATIONS
    )
    checks["task_orientation_balances_answer_identity"] = all(
        Counter(
            select_row(unit_rows, panel, task, orientation, order)["expected_class"]
            for task in TASKS for orientation in ORIENTATIONS
        ) == Counter({"e0": 2, "e1": 2})
        for unit_rows in grouped.values() for panel in PANELS for order in CARRIER_ORDERS
    )
    checks["task_prefix_exact_through_branch"] = all(
        select_row(unit_rows, panel, "max", orientation, order)["input_ids"][:
            select_row(unit_rows, panel, "max", orientation, order)["role_positions"]["branch_probe"] + 1]
        == select_row(unit_rows, panel, "min", orientation, order)["input_ids"][:
            select_row(unit_rows, panel, "min", orientation, order)["role_positions"]["branch_probe"] + 1]
        for unit_rows in grouped.values() for panel in PANELS for orientation in ORIENTATIONS for order in CARRIER_ORDERS
    )
    checks["orientation_preserves_token_multiset"] = all(
        Counter(select_row(unit_rows, panel, task, 0, order)["input_ids"])
        == Counter(select_row(unit_rows, panel, task, 1, order)["input_ids"])
        for unit_rows in grouped.values() for panel in PANELS for task in TASKS for order in CARRIER_ORDERS
    )
    checks["carrier_order_preserves_token_multiset"] = all(
        Counter(select_row(unit_rows, panel, task, orientation, 0)["input_ids"])
        == Counter(select_row(unit_rows, panel, task, orientation, 1)["input_ids"])
        for unit_rows in grouped.values() for panel in PANELS for task in TASKS for orientation in ORIENTATIONS
    )
    discovery_names = set(selected_names[:2 * 2 * ITEMS_PER_TEMPLATE])
    confirmation_names = set(selected_names[2 * 2 * ITEMS_PER_TEMPLATE:])
    checks["name_splits_disjoint"] = not (discovery_names & confirmation_names)
    checks["names_held_out_from_prior_phases"] = not (set(selected_names) & excluded_names())
    checks["roles_complete_and_ordered"] = all(
        set(row["role_positions"]) == set(CAPTURE_ROLES)
        and row["role_positions"]["fact_end"] < row["role_positions"]["carrier_end"]
        < row["role_positions"]["branch_probe"] < row["role_positions"]["task_cue"]
        <= row["role_positions"]["query_end"] < row["role_positions"]["answer_boundary"]
        for row in rows
    )
    checks["natural_question_has_no_candidate_list"] = all(
        "Candidates:" not in row["raw_prompt"] and "候选" not in row["raw_prompt"] for row in rows
    )
    result = {
        "schema_version": "phase1097_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(rows),
        "unit_count": len(grouped),
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
    for model_name in MODELS:
        rows = build_model_cases(tokenizers[model_name], model_name, selected_names)
        audit = audit_model(model_name, rows, selected_names)
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
        case_digests[model_name] = audit["case_digest"]
        print({"phase": PHASE, "model": model_name, "cases": len(rows), "units": audit["unit_count"], "audit_passed": audit["all_checks_passed"]})

    source_summary = read_json(SOURCE_PHASE1096) if SOURCE_PHASE1096.exists() else {}
    prereg = {
        "schema_version": "phase1097_preregistration.v1",
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
        "tasks": list(TASKS),
        "orientations": list(ORIENTATIONS),
        "carrier_orders": list(CARRIER_ORDERS),
        "states": list(STATES),
        "items_per_template": ITEMS_PER_TEMPLATE,
        "case_count_per_model": len(rows),
        "unit_count_per_model": len(RELATIONS) * len(SURFACES) * len(TEMPLATES) * ITEMS_PER_TEMPLATE,
        "selected_names": list(selected_names),
        "capture_roles": list(CAPTURE_ROLES),
        "pre_task_roles": list(PRE_TASK_ROLES),
        "dynamic_roles": list(DYNAMIC_ROLES),
        "fields": list(FIELDS),
        "trajectory_fields": list(TRAJECTORY_FIELDS),
        "depth_anchors": list(DEPTH_ANCHORS),
        "trajectory_measurement": {
            "aggregation_order": "compute full-dimensional invariants per item, then aggregate",
            "stored_invariants": ["relative_amplitude", "depth_gram", "step_retention", "panel_alignment", "local_candidate_margin"],
            "forbidden_primary_methods": ["fixed-vector averaging", "PCA", "UMAP", "random-projection identity retrieval"],
            "role_lookup_use": "comparison trajectory, not a globally subtracted target",
        },
        "generation_steps": GENERATION_STEPS,
        "generation_items_per_cell": GENERATION_ITEMS_PER_CELL,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "automatic_next_rule": "No causal localization in Phase1097. Independent replication is allowed only if P5-P8 pass in two formal models.",
        "source_phase1096_summary_digest": source_summary.get("summary_digest"),
        "source_phase1096_auto_next": source_summary.get("automatic_next_required"),
        "model_case_digests": case_digests,
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    aggregate = {
        "schema_version": "phase1097_protocol_audit.v1",
        "phase": PHASE,
        "model_audits": model_audits,
        "source_phase1096_present": SOURCE_PHASE1096.exists(),
        "source_phase1096_auto_next_was_false": source_summary.get("automatic_next_required") is False,
    }
    aggregate["all_checks_passed"] = (
        all(audit["all_checks_passed"] for audit in model_audits.values())
        and aggregate["source_phase1096_present"]
        and aggregate["source_phase1096_auto_next_was_false"]
    )
    aggregate["audit_digest"] = digest(aggregate)
    write_json(protocol_root / "audit.json", aggregate)
    print({"phase": PHASE, "protocol_digest": prereg["protocol_digest"], "all_checks_passed": aggregate["all_checks_passed"], "selected_names": len(selected_names)})


if __name__ == "__main__":
    main()
