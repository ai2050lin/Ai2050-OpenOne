#!/usr/bin/env python3
"""Freeze Phase1096 comparison-primitive state-transition protocol.

The protocol separates three evidence ledgers on the same natural task:
relation representation, late max/min control, and answer-selecting execution.
Five relation families share names, factors, output contract, and templates.
A direct role-lookup panel has the same answer mapping without requiring a
relational comparison. Candidate-order interactions provide a matched carrier
control with no semantic answer consequence.
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


PHASE = 1096
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
CANDIDATE_ORDERS = (0, 1)
ITEMS_PER_TEMPLATE = 6
ASSISTANT_PREFILL = "Answer:"
CONTINUATION_PREFIX = " "
GENERATION_STEPS = 6
GENERATION_ITEMS_PER_CELL = 2
CAPTURE_ROLES = (
    "fact_end",
    "branch_probe",
    "task_cue",
    "query_end",
    "answer_boundary",
)
PRE_TASK_ROLES = ("fact_end", "branch_probe")
DYNAMIC_ROLES = ("task_cue", "query_end", "answer_boundary")
SIGNED_PROJECTION_DIM = 96
SIGNED_PROJECTION_REPLICATES = 2
SIGNED_PROJECTION_SEED = 1096001
SIGNED_FIELDS = (
    "relational_representation",
    "lookup_representation",
    "relational_control",
    "lookup_control",
    "comparison_control",
    "relational_execution",
    "lookup_execution",
    "comparison_execution",
    "relational_carrier",
    "lookup_carrier",
    "comparison_carrier",
)
STATES = tuple(
    f"p{panel}_t{task}_o{orientation}_c{order}"
    for panel in PANELS
    for task in TASKS
    for orientation in ORIENTATIONS
    for order in CANDIDATE_ORDERS
)
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1096_comparison_primitive_dynamics"
)
SOURCE_PHASE1095 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1095_query_antisymmetric_transport"
    / "analysis" / "final_summary.json"
)
SOURCE_PHASE1075 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1075_heldout_relation_polarity"
    / "protocol" / "cases.qwen3.jsonl"
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
            "max": ("which candidate is taller", "which candidate has greater height"),
            "min": ("which candidate is shorter", "which candidate has less height"),
            "roles": ("taller", "shorter"),
        },
        "zh": {
            "positive": ("{left}比{right}更高", "{left}的身高大于{right}"),
            "max": ("哪位候选者更高", "哪位候选者身高更大"),
            "min": ("哪位候选者更矮", "哪位候选者身高更小"),
            "roles": ("较高", "较矮"),
        },
    },
    "age": {
        "en": {
            "positive": ("{left} is older than {right}", "{left} has greater age than {right}"),
            "max": ("which candidate is older", "which candidate has greater age"),
            "min": ("which candidate is younger", "which candidate has less age"),
            "roles": ("older", "younger"),
        },
        "zh": {
            "positive": ("{left}比{right}年长", "{left}的年龄大于{right}"),
            "max": ("哪位候选者更年长", "哪位候选者年龄更大"),
            "min": ("哪位候选者更年轻", "哪位候选者年龄更小"),
            "roles": ("年长", "年轻"),
        },
    },
    "weight": {
        "en": {
            "positive": ("{left} is heavier than {right}", "{left} has greater weight than {right}"),
            "max": ("which candidate is heavier", "which candidate has greater weight"),
            "min": ("which candidate is lighter", "which candidate has less weight"),
            "roles": ("heavier", "lighter"),
        },
        "zh": {
            "positive": ("{left}比{right}更重", "{left}的重量大于{right}"),
            "max": ("哪位候选者更重", "哪位候选者重量更大"),
            "min": ("哪位候选者更轻", "哪位候选者重量更小"),
            "roles": ("较重", "较轻"),
        },
    },
    "arrival": {
        "en": {
            "positive": ("{left} arrived before {right}", "{left} completed arrival earlier than {right}"),
            "max": ("which candidate arrived earlier", "which candidate completed arrival first"),
            "min": ("which candidate arrived later", "which candidate completed arrival last"),
            "roles": ("earlier", "later"),
        },
        "zh": {
            "positive": ("{left}比{right}更早到达", "{left}的到达时间早于{right}"),
            "max": ("哪位候选者更早到达", "哪位候选者最先到达"),
            "min": ("哪位候选者更晚到达", "哪位候选者最后到达"),
            "roles": ("较早", "较晚"),
        },
    },
    "score": {
        "en": {
            "positive": ("{left} scored higher than {right}", "{left} earned more points than {right}"),
            "max": ("which candidate scored higher", "which candidate earned more points"),
            "min": ("which candidate scored lower", "which candidate earned fewer points"),
            "roles": ("higher-scoring", "lower-scoring"),
        },
        "zh": {
            "positive": ("{left}比{right}得分更高", "{left}获得的分数多于{right}"),
            "max": ("哪位候选者得分更高", "哪位候选者分数更多"),
            "min": ("哪位候选者得分更低", "哪位候选者分数更少"),
            "roles": ("高分", "低分"),
        },
    },
}


SHELLS = {
    "en": {
        0: "Evidence: {fact}. Candidates: {candidates}. Decision marker: {question}? Return exactly one candidate name.",
        1: "Use this record: {fact}. Choose between {candidates}. Final task: {question}? Give one name only.",
        2: "A log contains this statement: {fact}. Consider {candidates}. Now decide: {question}? Write only the chosen name.",
        3: "Review the following fact: {fact}. The options are {candidates}. Selection request: {question}? Respond with one candidate name.",
    },
    "zh": {
        0: "证据：{fact}。候选者：{candidates}。判定标记：{question}？只返回一个候选者姓名。",
        1: "使用这条记录：{fact}。请在{candidates}之间选择。最终任务：{question}？只给出一个姓名。",
        2: "日志中有这条陈述：{fact}。考虑{candidates}。现在判断：{question}？只写被选中的姓名。",
        3: "核对以下事实：{fact}。选项是{candidates}。选择请求：{question}？用一个候选者姓名回答。",
    },
}


EVIDENCE_THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_candidate_accuracy": 0.85,
    "minimum_generation_accuracy": 0.70,
    "minimum_relations_per_model": 4,
    "minimum_behavior_models": 2,
    "minimum_hidden_finite_fraction": 0.97,
    "maximum_projection_median_abs_norm_error": 0.08,
    "maximum_projection_p95_abs_norm_error": 0.20,
    "pre_task_tolerance": 1e-8,
    "minimum_split_cosine": 0.25,
    "minimum_heldout_relations": 4,
    "minimum_content_over_carrier_advantage": 0.10,
    "minimum_cross_language_directions": 1,
    "minimum_functional_profile_cosine": 0.50,
    "minimum_functional_profile_advantage": 0.10,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": "All source, name-holdout, factorial, prefix, token, answer-balance, and FP16 protocol audits pass.",
    "P2": "At least two models pass four of five relation families in both languages and both panels.",
    "P3": "At least two models pass hidden finite-state, exact pre-task-zero, identity, and dual-projection audits.",
    "P4": "Relational input-orientation fields repeat across independent names, phrases, and templates; this is representation evidence only.",
    "P5": "Answer-balanced max/min control fields appear only after the late task cue and repeat across content families.",
    "P6": "The relational-minus-lookup execution field predicts at least four held-out relation families and beats the no-answer-consequence carrier interaction by 0.10 in at least two models.",
    "P7": "The execution prediction transfers between English and Chinese and retains its carrier advantage in at least one direction in two models.",
    "P8": "A normalized component/depth/role execution profile repeats across at least two directed model pairs and beats the carrier profile.",
    "P9": "Only if P6-P8 pass may the repeated physical band be promoted for independent causal validation; otherwise it remains descriptive.",
}


# Phase1075 consumed its original held-out pool across the five relations.
# This independent pool is deliberately broader; the protocol still filters
# every name through all three tokenizers and rejects every Phase1075 name.
ADDITIONAL_NAME_CANDIDATES = (
    "Abigail", "Adriana", "Aimee", "Alana", "Alberto", "Alessandra",
    "Alfonso", "Alicia", "Amara", "Amelia", "Amira", "Andre",
    "Andreas", "Anita", "Annabel", "Arianna", "Arnold", "Ashton",
    "Aurora", "Beatrice", "Belinda", "Blake", "Bonnie", "Bradley",
    "Brett", "Brooke", "Byron", "Camille", "Carmen", "Cassandra",
    "Cedric", "Cesar", "Chloe", "Clara", "Clayton", "Cody", "Cole",
    "Conrad", "Daisy", "Damian", "Darius", "Darren", "Dean", "Diego",
    "Dominic", "Donovan", "Edgar", "Edith", "Edmund", "Eleanor",
    "Elena", "Eliza", "Elliot", "Eloise", "Elsa", "Emilio", "Erin",
    "Esther", "Eva", "Evan", "Frances", "Francesca", "Gabriel",
    "Gabriela", "Gemma", "Gerald", "Giselle", "Gordon", "Graham",
    "Grant", "Greta", "Haley", "Hector", "Heidi", "Holly", "Ian",
    "Imogen", "Isabel", "Jasmine", "Jasper", "Javier", "Jeremy",
    "Jocelyn", "Joel", "Jorge", "Julian", "Katrina", "Kayla",
    "Kelsey", "Lara", "Leila", "Lillian", "Lionel", "Logan", "Lucas",
    "Lydia", "Malcolm", "Marcus", "Marina", "Megan", "Melanie",
    "Meredith", "Monica", "Morgan", "Nadia", "Nathan", "Neil",
    "Olivia", "Pablo", "Paige", "Pedro", "Phoebe", "Quentin",
    "Regina", "Renee", "Ricardo", "Riley", "Robin", "Rosa", "Ross",
    "Ruby", "Sabrina", "Samuel", "Sandra", "Sebastian", "Serena",
    "Seth", "Stella", "Sylvia", "Tara", "Tobias", "Trevor", "Valerie",
    "Vanessa", "Victoria", "Vivian", "Wesley", "Whitney", "Zachary",
    "Zoe", "Akira", "Hana", "Haru", "Kenji", "Mei", "Mina", "Ren",
    "Sora", "Yuki",
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


def prior_names() -> set[str]:
    if not SOURCE_PHASE1075.exists():
        return set()
    result: set[str] = set()
    for row in read_jsonl(SOURCE_PHASE1075):
        result.update(str(value) for value in row.get("cell_names", []))
    return result


def select_names(tokenizers: dict[str, Any]) -> tuple[str, ...]:
    excluded = prior_names()
    eligible = []
    used_ids = {model: set() for model in MODELS}
    candidates = tuple(dict.fromkeys(
        ADDITIONAL_NAME_CANDIDATES + name_source.HELDOUT_NAME_CANDIDATES
    ))
    ranked = sorted(
        candidates,
        key=lambda value: sha256_text(f"phase1096|{value}"),
    )
    for name in ranked:
        if name in excluded:
            continue
        ids = {}
        valid = True
        for model, tokenizer in tokenizers.items():
            values = tokenizer.encode(" " + name, add_special_tokens=False)
            if len(values) != 1 or int(values[0]) in used_ids[model]:
                valid = False
                break
            ids[model] = int(values[0])
        if not valid:
            continue
        eligible.append(name)
        for model, token_id in ids.items():
            used_ids[model].add(token_id)
        if len(eligible) == len(TEMPLATES) * ITEMS_PER_TEMPLATE * 2:
            break
    required = len(TEMPLATES) * ITEMS_PER_TEMPLATE * 2
    if len(eligible) != required:
        raise RuntimeError(f"need {required} held-out names, found {len(eligible)}")
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
            fact = (
                f"For this decision, {high} is the {max_role} endpoint and "
                f"{low} is the {min_role} endpoint"
            )
        else:
            fact = f"在这次判定中，{high}是{max_role}端，{low}是{min_role}端"
    cue = spec[task][pindex]
    question = cue
    candidates = (
        f"{entity0} and {entity1}" if order == 0 else f"{entity1} and {entity0}"
    ) if surface == "en" else (
        f"{entity0}和{entity1}" if order == 0 else f"{entity1}和{entity0}"
    )
    raw_prompt = SHELLS[surface][template].format(
        fact=fact, candidates=candidates, question=question
    )
    fact_span = mark(raw_prompt, fact)
    if surface == "en":
        branch_text = {
            0: "Decision marker:", 1: "Final task:",
            2: "Now decide:", 3: "Selection request:",
        }[template]
    else:
        branch_text = {
            0: "判定标记：", 1: "最终任务：",
            2: "现在判断：", 3: "选择请求：",
        }[template]
    branch_span = mark(raw_prompt, branch_text, fact_span[1])
    cue_span = mark(raw_prompt, cue, branch_span[1])
    question_span = cue_span
    spans = {
        "fact_end": fact_span,
        "branch_probe": branch_span,
        "task_cue": cue_span,
        "query_end": question_span,
    }
    return raw_prompt, spans, {
        "entity0": entity0,
        "entity1": entity1,
        "high": high,
        "low": low,
        "fact": fact,
        "cue": cue,
        "candidates": candidates,
    }


def continuation_ids(tokenizer, rendered: str, label: str) -> list[int]:
    base = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    extended = [
        int(value) for value in tokenizer.encode(
            rendered + CONTINUATION_PREFIX + label,
            add_special_tokens=False,
        )
    ]
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
    input_ids = [
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
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
    unit_id = (
        f"phase1096.{model_name}.{relation}.{surface}."
        f"t{template}.i{item_index:02d}"
    )
    return {
        "schema_version": "phase1096_comparison_case.v1",
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
        "candidate_order": order,
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
        "task_cue_text": meta["cue"],
        "continuation_prefix": CONTINUATION_PREFIX,
        "prompt_digest": sha256_text(raw_prompt),
    }


def build_model_cases(
    tokenizer,
    model_name: str,
    selected_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    case_index = 0
    for relation in RELATIONS:
        for surface in SURFACES:
            for template in TEMPLATES:
                for item_index in range(ITEMS_PER_TEMPLATE):
                    for state in STATES:
                        rows.append(build_case(
                            tokenizer, model_name, selected_names,
                            relation, surface, template, item_index,
                            state, case_index,
                        ))
                        case_index += 1
    return rows


def select_row(rows: list[dict[str, Any]], panel: str, task: str, orientation: int, order: int) -> dict[str, Any]:
    return next(
        row for row in rows
        if row["panel"] == panel
        and row["task"] == task
        and int(row["orientation"]) == orientation
        and int(row["candidate_order"]) == order
    )


def audit_model(
    model_name: str,
    rows: list[dict[str, Any]],
    selected_names: tuple[str, ...],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["unit_id"])].append(row)
    checks: dict[str, bool] = {}
    checks["case_count"] = len(rows) == (
        len(RELATIONS) * len(SURFACES) * len(TEMPLATES)
        * ITEMS_PER_TEMPLATE * len(STATES)
    )
    checks["unit_count"] = len(grouped) == (
        len(RELATIONS) * len(SURFACES) * len(TEMPLATES)
        * ITEMS_PER_TEMPLATE
    )
    checks["complete_factorial_states"] = all(
        {row["state"] for row in unit_rows} == set(STATES)
        and len(unit_rows) == len(STATES)
        for unit_rows in grouped.values()
    )
    checks["candidate_continuations_one_token"] = all(
        all(len(values) == 1 for values in row["candidate_token_ids"].values())
        for row in rows
    )
    checks["candidate_first_tokens_distinct"] = all(
        row["candidate_first_token_ids"]["e0"]
        != row["candidate_first_token_ids"]["e1"]
        for row in rows
    )
    checks["panels_have_identical_answers"] = all(
        select_row(unit_rows, "relational", task, orientation, order)["expected_class"]
        == select_row(unit_rows, "role_lookup", task, orientation, order)["expected_class"]
        for unit_rows in grouped.values()
        for task in TASKS
        for orientation in ORIENTATIONS
        for order in CANDIDATE_ORDERS
    )
    checks["candidate_order_has_no_answer_consequence"] = all(
        select_row(unit_rows, panel, task, orientation, 0)["expected_class"]
        == select_row(unit_rows, panel, task, orientation, 1)["expected_class"]
        for unit_rows in grouped.values()
        for panel in PANELS
        for task in TASKS
        for orientation in ORIENTATIONS
    )
    checks["task_orientation_balances_answer_identity"] = all(
        Counter(
            select_row(unit_rows, panel, task, orientation, order)["expected_class"]
            for task in TASKS
            for orientation in ORIENTATIONS
        ) == Counter({"e0": 2, "e1": 2})
        for unit_rows in grouped.values()
        for panel in PANELS
        for order in CANDIDATE_ORDERS
    )
    checks["task_prefix_exact_through_branch"] = all(
        select_row(unit_rows, panel, "max", orientation, order)["input_ids"][:
            select_row(unit_rows, panel, "max", orientation, order)["role_positions"]["branch_probe"] + 1
        ]
        == select_row(unit_rows, panel, "min", orientation, order)["input_ids"][:
            select_row(unit_rows, panel, "min", orientation, order)["role_positions"]["branch_probe"] + 1
        ]
        for unit_rows in grouped.values()
        for panel in PANELS
        for orientation in ORIENTATIONS
        for order in CANDIDATE_ORDERS
    )
    checks["orientation_preserves_token_multiset"] = all(
        Counter(select_row(unit_rows, panel, task, 0, order)["input_ids"])
        == Counter(select_row(unit_rows, panel, task, 1, order)["input_ids"])
        for unit_rows in grouped.values()
        for panel in PANELS
        for task in TASKS
        for order in CANDIDATE_ORDERS
    )
    checks["candidate_order_preserves_token_multiset"] = all(
        Counter(select_row(unit_rows, panel, task, orientation, 0)["input_ids"])
        == Counter(select_row(unit_rows, panel, task, orientation, 1)["input_ids"])
        for unit_rows in grouped.values()
        for panel in PANELS
        for task in TASKS
        for orientation in ORIENTATIONS
    )
    discovery_names = set(selected_names[:2 * 2 * ITEMS_PER_TEMPLATE])
    confirmation_names = set(selected_names[2 * 2 * ITEMS_PER_TEMPLATE:])
    checks["name_splits_disjoint"] = not (discovery_names & confirmation_names)
    checks["names_held_out_from_phase1075"] = not (set(selected_names) & prior_names())
    checks["roles_complete_and_ordered"] = all(
        set(row["role_positions"]) == set(CAPTURE_ROLES)
        and row["role_positions"]["fact_end"] <= row["role_positions"]["branch_probe"]
        < row["role_positions"]["task_cue"] <= row["role_positions"]["query_end"]
        < row["role_positions"]["answer_boundary"]
        for row in rows
    )
    checks["all_checks_boolean"] = all(isinstance(value, bool) for value in checks.values())
    result = {
        "schema_version": "phase1096_protocol_model_audit.v1",
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
    model_audits = {}
    case_digests = {}
    for model_name in MODELS:
        rows = build_model_cases(tokenizers[model_name], model_name, selected_names)
        audit = audit_model(model_name, rows, selected_names)
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
        case_digests[model_name] = audit["case_digest"]
        print({
            "phase": PHASE,
            "model": model_name,
            "cases": len(rows),
            "units": audit["unit_count"],
            "audit_passed": audit["all_checks_passed"],
        })
    source_summary = read_json(SOURCE_PHASE1095) if SOURCE_PHASE1095.exists() else {}
    prereg = {
        "schema_version": "phase1096_preregistration.v1",
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
        "candidate_orders": list(CANDIDATE_ORDERS),
        "states": list(STATES),
        "items_per_template": ITEMS_PER_TEMPLATE,
        "case_count_per_model": len(rows),
        "unit_count_per_model": len(rows) // len(STATES),
        "selected_names": list(selected_names),
        "capture_roles": list(CAPTURE_ROLES),
        "pre_task_roles": list(PRE_TASK_ROLES),
        "dynamic_roles": list(DYNAMIC_ROLES),
        "signed_fields": list(SIGNED_FIELDS),
        "projection": {
            "type": "deterministic_rademacher",
            "dimension_per_replicate": SIGNED_PROJECTION_DIM,
            "replicates": SIGNED_PROJECTION_REPLICATES,
            "seed": SIGNED_PROJECTION_SEED,
            "cross_model_rule": "Compare normalized component-depth-role profiles, never raw projected coordinates.",
        },
        "measurement_definitions": {
            "representation": "orientation main effect averaged over task and candidate order",
            "control": "max-minus-min main effect averaged over orientation and candidate order",
            "execution": "task-by-orientation interaction averaged over candidate order",
            "comparison_execution": "0.5*(relational_execution-role_lookup_execution)",
            "carrier": "orientation-by-candidate-order interaction averaged over task; candidate order has no answer consequence",
            "comparison_carrier": "0.5*(relational_carrier-role_lookup_carrier)",
        },
        "generation_steps": GENERATION_STEPS,
        "generation_items_per_cell": GENERATION_ITEMS_PER_CELL,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_phase1095_summary_digest": source_summary.get("summary_digest"),
        "model_case_digests": case_digests,
        "model_audits": model_audits,
        "interpretation_limits": [
            "The panel subtraction isolates relational inference relative to direct role lookup in this protocol; it is not a universal comparison operator by definition.",
            "Factorial contrasts are measurement operators and do not assert that the network is globally linear or additive.",
            "Cross-content prediction must exceed the candidate-order carrier interaction before any computational primitive claim.",
            "Cross-model raw directions are incomparable because dimensions and random projections are model-specific.",
            "All physical bands remain descriptive until a separate independent causal phase is authorized.",
        ],
        "automatic_next": {
            "hidden_scan_if": "P2 behavior authorization passes in at least two models.",
            "independent_replication_if": "P6 and P7 pass prospectively in at least two models.",
            "causal_if": "Never in Phase1096; require independent replication and P8 first.",
            "otherwise": "Freeze the three ledgers and do not run a nearby contrast variant automatically.",
        },
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    checks = {
        "all_model_audits_passed": all(row["all_checks_passed"] for row in model_audits.values()),
        "model_order_frozen": tuple(prereg["sequential_model_order"]) == MODELS,
        "fp16_no_quantization": PRECISION == "fp16" and QUANTIZATION == "none",
        "large_case_count": int(prereg["case_count_per_model"]) >= 3000,
        "five_relations_two_languages_two_panels": (
            len(RELATIONS) == 5 and len(SURFACES) == 2 and len(PANELS) == 2
        ),
        "source_phase1095_bound": bool(prereg["source_phase1095_summary_digest"]),
    }
    audit = {
        "schema_version": "phase1096_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    audit["audit_digest"] = digest(audit)
    write_json(protocol_root / "audit.json", audit)
    print({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "audit_passed": audit["all_checks_passed"],
        "selected_name_count": len(selected_names),
    })


if __name__ == "__main__":
    main()
