#!/usr/bin/env python3
"""Freeze Phase1103 natural relation-address and causal-transport protocol."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1098_relative_relation_geometry_protocol as tools
import phase1101_relation_identity_routing_protocol as phase1101


PHASE = 1103
PROTOCOL_REVISION = 2
MODELS = ("qwen3", "glm4", "deepseek7b")
FORMAL_MODELS = MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SURFACES = ("en", "zh")
TEMPLATES = (0, 1, 2, 3)
TEMPLATES_BY_SPLIT = {
    "qualification": (0, 1),
    "confirmation": (2, 3),
}
SPLITS = tuple(TEMPLATES_BY_SPLIT)
ITEMS_PER_TEMPLATE = 3
ROUTE_TYPES = ("exact", "paraphrase", "ordinal")
CONGRUENCES = ("conflict", "congruent")
TARGET_RELATIONS = (0, 1)
RELATION_ORDERS = (0, 1)
ORIENTATIONS = (0, 1)
ASSISTANT_PREFILL = "Answer:"
CONTINUATION_PREFIX = " "
GENERATION_STEPS = 6
GENERATION_ITEMS_PER_CELL = 1
CAPTURE_ROLES = ("facts_end", "selector_end", "query_end", "answer_boundary")
CAUSAL_DEPTH_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
CAUSAL_DISCOVERY_ITEMS = (0,)
CAUSAL_CONFIRMATION_ITEMS = (0, 1, 2)
CAUSAL_RELATION_ORDERS = (0,)
PATCH_ALPHA = 1.0
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1103_natural_relation_route"
SOURCE_PHASE1102_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1102_relation_identity_routing_replication"
)
SOURCE_PHASE1102_AUTHORIZATION = (
    SOURCE_PHASE1102_ROOT / "analysis" / "behavior_authorization.json"
)
SOURCE_PHASE1102_AUDIT = SOURCE_PHASE1102_ROOT / "audit" / "result_audit.json"


write_json = tools.write_json
write_jsonl = tools.write_jsonl
read_json = tools.read_json
read_jsonl = tools.read_jsonl
digest = tools.digest
sha256_text = tools.sha256_text


RELATIONS = phase1101.RELATIONS
RELATION_FAMILY = phase1101.RELATION_FAMILY
RELATION_PAIRS = phase1101.RELATION_PAIRS
PAIR_RELATIONS = phase1101.PAIR_RELATIONS
PAIR_FAMILY = phase1101.PAIR_FAMILY
FAMILIES = phase1101.FAMILIES


# Four natural noun-phrase realizations per relation. Templates 0/1 use only
# aliases 0/1, while templates 2/3 use only aliases 2/3. This freezes a
# lexical holdout between qualification and causal confirmation.
EN_ALIASES = {
    "height": ("height", "vertical stature", "tallness", "body height"),
    "weight": ("weight", "body mass", "heaviness", "physical weight"),
    "speed": ("speed", "movement rate", "quickness", "movement pace"),
    "brightness": ("brightness", "light intensity", "luminosity", "light level"),
    "temperature": ("temperature", "heat level", "warmth", "degree of heat"),
    "price": ("price", "monetary cost", "expense level", "purchase cost"),
    "arrival_time": ("arrival time", "time of arrival", "arrival timing", "when arrival occurred"),
    "departure_time": ("departure time", "time of departure", "departure timing", "when departure occurred"),
    "start_time": ("start time", "starting time", "commencement time", "when the start occurred"),
    "finish_time": ("finish time", "completion time", "ending time", "when completion occurred"),
    "registration_time": ("registration time", "signup time", "enrollment timing", "when registration occurred"),
    "publication_time": ("publication time", "release time", "publication timing", "when publication occurred"),
    "north_position": ("northward position", "northerly location", "northing", "position toward the north"),
    "east_position": ("eastward position", "easterly location", "easting", "position toward the east"),
    "elevation": ("elevation", "altitude", "vertical level", "height above sea level"),
    "forward_position": ("forward position", "frontward location", "position ahead", "progress position"),
    "distance": ("distance", "remoteness", "separation", "spatial distance"),
    "depth": ("depth", "downward extent", "deepness", "vertical depth"),
    "authority": ("authority", "decision power", "command power", "level of authority"),
    "seniority": ("seniority", "length of service", "tenure rank", "level of seniority"),
    "influence": ("influence", "persuasive power", "sway", "degree of influence"),
    "popularity": ("popularity", "public favor", "audience appeal", "degree of popularity"),
    "responsibility": ("responsibility", "duty level", "accountability", "degree of responsibility"),
    "leadership_rank": ("leadership rank", "position in leadership", "command rank", "leadership level"),
    "causal_influence": ("causal influence", "causal impact", "causal force", "effect on the outcome"),
    "evidence_strength": ("evidence strength", "evidential support", "support strength", "strength of evidence"),
    "likelihood": ("likelihood", "probability", "chance", "degree of likelihood"),
    "certainty": ("certainty", "confidence level", "degree of assurance", "level of certainty"),
    "explanatory_power": ("explanatory power", "ability to explain", "explanatory strength", "power of explanation"),
    "dependency_strength": ("dependency strength", "degree of dependence", "dependence intensity", "strength of dependency"),
}

ZH_ALIASES = {
    "height": ("身高", "垂直身材", "高矮程度", "身体高度"),
    "weight": ("体重", "身体质量", "轻重程度", "身体重量"),
    "speed": ("速度", "移动速率", "快慢程度", "运动节奏"),
    "brightness": ("亮度", "光照强度", "明亮程度", "光亮水平"),
    "temperature": ("温度", "热度", "冷热程度", "温热水平"),
    "price": ("价格", "货币成本", "昂贵程度", "购买成本"),
    "arrival_time": ("到达时间", "抵达时刻", "到达先后", "抵达时间点"),
    "departure_time": ("出发时间", "离开时刻", "出发先后", "启程时间点"),
    "start_time": ("开始时间", "起始时刻", "开始先后", "启动时间点"),
    "finish_time": ("完成时间", "结束时刻", "完成先后", "完工时间点"),
    "registration_time": ("登记时间", "注册时刻", "报名先后", "登记时间点"),
    "publication_time": ("发布时间", "发表时刻", "发布先后", "出版时间点"),
    "north_position": ("南北位置", "向北程度", "北向坐标", "北侧位置"),
    "east_position": ("东西位置", "向东程度", "东向坐标", "东侧位置"),
    "elevation": ("海拔", "高度", "垂直高程", "海平面以上高度"),
    "forward_position": ("前后位置", "向前程度", "前方位置", "前进位置"),
    "distance": ("距离", "远近程度", "空间间隔", "相隔距离"),
    "depth": ("深度", "下探程度", "纵深", "深浅程度"),
    "authority": ("权威程度", "决策权", "支配力", "权力水平"),
    "seniority": ("资历", "任职年限", "年资等级", "资深程度"),
    "influence": ("影响力", "影响程度", "左右能力", "作用强度"),
    "popularity": ("受欢迎程度", "公众喜爱度", "大众吸引力", "人气水平"),
    "responsibility": ("责任程度", "职责大小", "问责程度", "责任水平"),
    "leadership_rank": ("领导级别", "领导层级", "指挥等级", "领导职位高低"),
    "causal_influence": ("因果影响", "因果作用", "因果效力", "对结果的影响程度"),
    "evidence_strength": ("证据强度", "证据支持力", "支持强度", "证据有力程度"),
    "likelihood": ("可能性", "概率", "机会大小", "发生可能程度"),
    "certainty": ("确定性", "把握程度", "确信度", "确定程度"),
    "explanatory_power": ("解释力", "解释能力", "说明力", "解释结果的程度"),
    "dependency_strength": ("依赖强度", "依赖程度", "依存强度", "相互依赖力度"),
}

RELATION_ALIASES = {
    relation: {"en": EN_ALIASES[relation], "zh": ZH_ALIASES[relation]}
    for relation in RELATIONS
}
PARAPHRASE_INDEX = {0: 1, 1: 0, 2: 3, 3: 2}


SHELLS = {
    "en": (
        "A committee assessed the same two people in two ways. {fact1}. {fact2}. {question}? Return exactly one person name.",
        "Two criteria were considered for the same pair of people. {fact1}. Also, {fact2}. {question}? Give one person name only.",
        "The final choice follows one of two independent comparisons. {fact1}. Separately, {fact2}. {question}? Respond with exactly one person name.",
        "A decision uses one of two assessments of the same people. {fact1}. In addition, {fact2}. {question}? Write only the selected person's name.",
    ),
    "zh": (
        "委员会从两个方面评估同一组人。{fact1}。{fact2}。{question}？只回答一个人名。",
        "同一组人接受了两项标准的比较。{fact1}。另外，{fact2}。{question}？仅给出一个人名。",
        "最终选择依据两项独立比较中的一项。{fact1}。另一项是，{fact2}。{question}？请准确回答一个人名。",
        "一项决定会使用对同一组人的两种评估之一。{fact1}。此外，{fact2}。{question}？只写被选中的人名。",
    ),
}

ORDINAL_SELECTORS = {
    "en": ("the first comparison", "the second comparison"),
    "zh": ("第一项比较", "第二项比较"),
}


ROUTE_CODES = {"exact": "e", "paraphrase": "p", "ordinal": "o"}


def state_name(
    route_type: str,
    congruence: str,
    target_relation: int,
    relation_order: int,
    orientation: int,
) -> str:
    return (
        f"r{ROUTE_CODES[route_type]}"
        f"_c{0 if congruence == 'conflict' else 1}"
        f"_q{target_relation}_o{relation_order}_b{orientation}"
    )


STATES = tuple(
    state_name(route, congruence, target, order, orientation)
    for route, congruence, target, order, orientation in itertools.product(
        ROUTE_TYPES, CONGRUENCES, TARGET_RELATIONS,
        RELATION_ORDERS, ORIENTATIONS,
    )
)


def state_factors(state: str) -> tuple[str, str, int, int, int]:
    for factors in itertools.product(
        ROUTE_TYPES, CONGRUENCES, TARGET_RELATIONS,
        RELATION_ORDERS, ORIENTATIONS,
    ):
        if state == state_name(*factors):
            return factors
    raise ValueError(f"unknown Phase1103 state: {state}")


THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_pair_cell_accuracy": 0.80,
    "minimum_route_accuracy": 0.90,
    "minimum_congruent_accuracy": 0.90,
    "minimum_generation_accuracy": 0.75,
    "minimum_models_per_shared_pair": 2,
    "minimum_model_passing_pairs_for_wrong_control": 2,
    "minimum_causal_finite_fraction": 0.97,
    "minimum_causal_behavior_valid_fraction": 0.80,
    "minimum_causal_median_recovery": 0.10,
    "minimum_causal_positive_fraction": 0.60,
    "minimum_each_direction_median_recovery": 0.05,
    "minimum_each_direction_positive_fraction": 0.55,
    "minimum_causal_flip_rate": 0.10,
    "minimum_causal_specificity_advantage": 0.05,
    "maximum_congruent_collateral_flip_rate": 0.10,
    "minimum_models_per_confirmed_causal_cell": 2,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": "All source, name-holdout, lexical-split, token, factorial, prefix, answer-balance, and behavior-necessity audits pass.",
    "P2": "At least one relation pair passes exact, nonidentical-paraphrase, ordinal, congruent, and natural-generation gates in both splits for at least two models.",
    "P3": "Only prospectively shared behavior-passing pairs enter the causal scan; all conclusions remain pair-and-surface specific rather than family-wide.",
    "P4": "A qualification-selected residual depth yields median cross-expression recovery of at least 0.10 on independent confirmation prompts.",
    "P5": "Confirmation recovery exceeds ordinal, behavior-passing wrong-pair, and equal-norm random controls by at least 0.05, with positive recovery in at least 60 percent of cells.",
    "P6": "The same intervention flips at least 10 percent of conflict targets while flipping at most 10 percent of congruent controls.",
    "P7": "At least two models independently pass P4-P6 for the same pair-surface cell before any component or neuron localization is authorized.",
    "P8": "Failure of P2 stops hidden-state access; failure of P4-P7 preserves the physical response map but does not count as relation-semantic transport.",
}


def split_for_template(template: int) -> str:
    for split, templates in TEMPLATES_BY_SPLIT.items():
        if template in templates:
            return split
    raise ValueError(template)


def mark(text: str, value: str, start: int = 0) -> tuple[int, int, str]:
    position = text.find(value, start)
    if position < 0:
        raise RuntimeError(f"missing marked value {value!r}")
    return position, position + len(value), value


def prior_names() -> set[str]:
    names = set(phase1101.old_names())
    for path in (
        phase1101.OUT_ROOT / "protocol" / "preregistration.json",
        SOURCE_PHASE1102_ROOT / "protocol" / "preregistration.json",
    ):
        if path.exists():
            names.update(read_json(path).get("selected_names", []))
    return names


def select_names(tokenizers: dict[str, Any]) -> tuple[str, ...]:
    phase1101_names = tuple(read_json(
        phase1101.OUT_ROOT / "protocol" / "preregistration.json"
    )["selected_names"])
    phase1102_names = tuple(read_json(
        SOURCE_PHASE1102_ROOT / "protocol" / "preregistration.json"
    )["selected_names"])
    per_split = 2 * ITEMS_PER_TEMPLATE * 2
    selected = phase1101_names[:per_split] + phase1102_names[:per_split]
    required = len(TEMPLATES) * ITEMS_PER_TEMPLATE * 2
    if len(selected) != required:
        raise RuntimeError(
            f"need {required} frozen one-token names, found {len(selected)}"
        )
    if set(selected[:per_split]) & set(selected[per_split:]):
        raise RuntimeError("qualification and confirmation name worlds overlap")
    for model, tokenizer in tokenizers.items():
        ids = [
            tokenizer.encode(" " + name, add_special_tokens=False)
            for name in selected
        ]
        if any(len(values) != 1 for values in ids):
            raise RuntimeError(f"frozen name tokenization drift for {model}")
        if len({int(values[0]) for values in ids}) != len(ids):
            raise RuntimeError(f"frozen name token collision for {model}")
    return selected


def name_pair(
    selected_names: tuple[str, ...], template: int, item_index: int
) -> tuple[str, str]:
    index = template * ITEMS_PER_TEMPLATE + item_index
    return selected_names[2 * index], selected_names[2 * index + 1]


def relation_fact(
    surface: str, alias: str, high: str, low: str
) -> str:
    if surface == "en":
        return f"For {alias}, [ {high} ] was ranked ahead of [ {low} ]"
    return f"按{alias}比较，[ {high} ] 排在 [ {low} ] 前面"


def render_prompt(
    relation_pair: str,
    surface: str,
    template: int,
    route_type: str,
    congruence: str,
    target_relation: int,
    relation_order: int,
    orientation: int,
    names: tuple[str, str],
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, Any]]:
    entity0, entity1 = names
    relation0, relation1 = PAIR_RELATIONS[relation_pair]
    winner0, loser0 = (
        (entity0, entity1) if orientation == 0 else (entity1, entity0)
    )
    if congruence == "conflict":
        winner1, loser1 = loser0, winner0
    else:
        winner1, loser1 = winner0, loser0
    alias_index = template
    aliases = {
        relation0: RELATION_ALIASES[relation0][surface][alias_index],
        relation1: RELATION_ALIASES[relation1][surface][alias_index],
    }
    facts = {
        relation0: relation_fact(surface, aliases[relation0], winner0, loser0),
        relation1: relation_fact(surface, aliases[relation1], winner1, loser1),
    }
    displayed = (
        (relation0, relation1) if relation_order == 0
        else (relation1, relation0)
    )
    fact1, fact2 = facts[displayed[0]], facts[displayed[1]]
    target_name = (relation0, relation1)[target_relation]
    if route_type == "exact":
        selector = aliases[target_name]
    elif route_type == "paraphrase":
        selector = RELATION_ALIASES[target_name][surface][
            PARAPHRASE_INDEX[template]
        ]
    else:
        selector = ORDINAL_SELECTORS[surface][displayed.index(target_name)]
    if surface == "en":
        selector_clause = (
            selector if route_type == "ordinal" else f"the {selector} criterion"
        )
        question = f"Using {selector_clause}, which person should be selected"
    else:
        question = f"以{selector}为准，应该选择哪个人"
    raw_prompt = SHELLS[surface][template].format(
        fact1=fact1, fact2=fact2, question=question
    )
    fact1_span = mark(raw_prompt, fact1)
    fact2_span = mark(raw_prompt, fact2, fact1_span[1])
    question_span = mark(raw_prompt, question, fact2_span[1])
    selector_span = mark(raw_prompt, selector, question_span[0])
    expected = winner0 if target_relation == 0 else winner1
    return raw_prompt, {
        "facts_end": fact2_span,
        "selector_end": selector_span,
        "query_end": question_span,
    }, {
        "entity0": entity0,
        "entity1": entity1,
        "winner0": winner0,
        "winner1": winner1,
        "expected": expected,
        "relation0": relation0,
        "relation1": relation1,
        "displayed_relations": displayed,
        "fact1": fact1,
        "fact2": fact2,
        "question": question,
        "selector": selector,
        "visible_target_alias": aliases[target_name],
        "alias_index": alias_index,
    }


def build_case(
    tokenizer,
    model_name: str,
    selected_names: tuple[str, ...],
    relation_pair: str,
    surface: str,
    template: int,
    item_index: int,
    state: str,
    case_index: int,
) -> dict[str, Any]:
    route, congruence, target, order, orientation = state_factors(state)
    names = name_pair(selected_names, template, item_index)
    raw_prompt, raw_spans, meta = render_prompt(
        relation_pair, surface, template, route, congruence,
        target, order, orientation, names,
    )
    rendered = (
        phase1101.base.behavior_tools.render_native(
            tokenizer, model_name, raw_prompt, with_system=False
        )
        + ASSISTANT_PREFILL
    )
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_spans = offset_token_spans(
        tokenizer, rendered, raw_prompt, raw_spans
    )
    role_positions = {
        role: int(span[1]) for role, span in role_spans.items()
    }
    role_positions["answer_boundary"] = len(input_ids) - 1
    role_spans["answer_boundary"] = (
        len(input_ids) - 1, len(input_ids) - 1
    )
    expected_class = (
        "e0" if meta["expected"] == meta["entity0"] else "e1"
    )
    candidate_labels = {
        "e0": meta["entity0"], "e1": meta["entity1"]
    }
    candidate_token_ids = {
        key: phase1101.base.continuation_ids(tokenizer, rendered, label)
        for key, label in candidate_labels.items()
    }
    unit_id = (
        f"phase1103.{model_name}.{relation_pair}.{surface}"
        f".t{template}.i{item_index:02d}"
    )
    selector_token_ids = [
        int(value) for value in tokenizer.encode(
            meta["selector"], add_special_tokens=False
        )
    ]
    visible_alias_token_ids = [
        int(value) for value in tokenizer.encode(
            meta["visible_target_alias"], add_special_tokens=False
        )
    ]
    return {
        "schema_version": "phase1103_natural_relation_route_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{unit_id}.{state}",
        "unit_id": unit_id,
        "superunit_id": (
            f"phase1103.{model_name}.{surface}"
            f".t{template}.i{item_index:02d}"
        ),
        "relation_pair": relation_pair,
        "relation0": meta["relation0"],
        "relation1": meta["relation1"],
        "family": PAIR_FAMILY[relation_pair],
        "surface": surface,
        "split": split_for_template(template),
        "template": template,
        "item_index": item_index,
        "state": state,
        "route_type": route,
        "congruence": congruence,
        "target_relation": target,
        "relation_order": order,
        "orientation": orientation,
        "entity0": meta["entity0"],
        "entity1": meta["entity1"],
        "winner0": meta["winner0"],
        "winner1": meta["winner1"],
        "expected_entity": meta["expected"],
        "expected_class": expected_class,
        "candidate_labels": candidate_labels,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {
            key: [int(values[0])]
            for key, values in candidate_token_ids.items()
        },
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_spans": {
            key: list(value) for key, value in role_spans.items()
        },
        "role_positions": role_positions,
        "fact1_text": meta["fact1"],
        "fact2_text": meta["fact2"],
        "question_text": meta["question"],
        "selector_text": meta["selector"],
        "visible_target_alias": meta["visible_target_alias"],
        "selector_token_ids": selector_token_ids,
        "visible_target_alias_token_ids": visible_alias_token_ids,
        "displayed_relations": list(meta["displayed_relations"]),
        "continuation_prefix": CONTINUATION_PREFIX,
        "prompt_digest": sha256_text(raw_prompt),
    }


def build_model_cases(
    tokenizer, model_name: str, selected_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in RELATION_PAIRS:
        for surface in SURFACES:
            for template in TEMPLATES:
                for item_index in range(ITEMS_PER_TEMPLATE):
                    for state in STATES:
                        rows.append(build_case(
                            tokenizer, model_name, selected_names, pair,
                            surface, template, item_index, state, len(rows),
                        ))
    return rows


def audit_model(
    model_name: str,
    rows: list[dict[str, Any]],
    selected_names: tuple[str, ...],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    expected_count = (
        len(RELATION_PAIRS) * len(SURFACES) * len(TEMPLATES)
        * ITEMS_PER_TEMPLATE * len(STATES)
    )
    checks["case_count"] = len(rows) == expected_count
    checks["state_count"] = len(STATES) == 48
    checks["record_ids_unique"] = (
        len({row["record_id"] for row in rows}) == len(rows)
    )
    checks["candidate_names_one_token"] = all(
        len(row["candidate_token_ids"][key]) == 1
        for row in rows for key in ("e0", "e1")
    )
    checks["candidate_first_tokens_distinct"] = all(
        row["candidate_first_token_ids"]["e0"]
        != row["candidate_first_token_ids"]["e1"]
        for row in rows
    )
    checks["roles_complete_and_ordered"] = all(
        set(row["role_positions"]) == set(CAPTURE_ROLES)
        and row["role_positions"]["facts_end"]
        < row["role_positions"]["selector_end"]
        <= row["role_positions"]["query_end"]
        < row["role_positions"]["answer_boundary"]
        for row in rows
    )
    checks["name_worlds_frozen_before_phase1103"] = all(
        name in prior_names() for name in selected_names
    )
    qualification_names = set(
        selected_names[:2 * 2 * ITEMS_PER_TEMPLATE]
    )
    confirmation_names = set(
        selected_names[2 * 2 * ITEMS_PER_TEMPLATE:]
    )
    checks["name_splits_disjoint"] = not (
        qualification_names & confirmation_names
    )
    checks["lexical_alias_splits_disjoint"] = all(
        not (
            set(RELATION_ALIASES[relation][surface][:2])
            & set(RELATION_ALIASES[relation][surface][2:])
        )
        for relation in RELATIONS for surface in SURFACES
    )
    checks["exact_and_paraphrase_selectors_nonidentical"] = all(
        row["selector_text"] != row["visible_target_alias"]
        for row in rows if row["route_type"] == "paraphrase"
    )
    checks["exact_selector_repeats_visible_alias"] = all(
        row["selector_text"] == row["visible_target_alias"]
        for row in rows if row["route_type"] == "exact"
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["unit_id"])].append(row)
    checks["unit_count"] = len(grouped) == (
        len(RELATION_PAIRS) * len(SURFACES) * len(TEMPLATES)
        * ITEMS_PER_TEMPLATE
    )
    checks["complete_factorial"] = all(
        {row["state"] for row in unit} == set(STATES)
        for unit in grouped.values()
    )
    prefix_ok = True
    necessity_ok = True
    same_facts_answers = True
    answer_balance = True
    for unit in grouped.values():
        index = {
            (
                row["route_type"], row["congruence"],
                int(row["target_relation"]), int(row["relation_order"]),
                int(row["orientation"]),
            ): row
            for row in unit
        }
        for congruence, order, orientation in itertools.product(
            CONGRUENCES, RELATION_ORDERS, ORIENTATIONS
        ):
            reference = index[("exact", congruence, 0, order, orientation)]
            stop = int(reference["role_positions"]["facts_end"]) + 1
            for route, target in itertools.product(
                ROUTE_TYPES, TARGET_RELATIONS
            ):
                row = index[(route, congruence, target, order, orientation)]
                prefix_ok &= (
                    int(row["role_positions"]["facts_end"]) + 1 == stop
                    and row["input_ids"][:stop] == reference["input_ids"][:stop]
                )
            for route in ROUTE_TYPES:
                left = index[(route, congruence, 0, order, orientation)]
                right = index[(route, congruence, 1, order, orientation)]
                necessity_ok &= (
                    left["expected_class"] != right["expected_class"]
                    if congruence == "conflict"
                    else left["expected_class"] == right["expected_class"]
                )
                for other_route in ROUTE_TYPES:
                    other = index[(
                        other_route, congruence, 0, order, orientation
                    )]
                    same_facts_answers &= (
                        left["fact1_text"] == other["fact1_text"]
                        and left["fact2_text"] == other["fact2_text"]
                        and left["expected_class"] == other["expected_class"]
                    )
        for route, congruence, target in itertools.product(
            ROUTE_TYPES, CONGRUENCES, TARGET_RELATIONS
        ):
            counts = Counter(
                row["expected_class"] for row in unit
                if row["route_type"] == route
                and row["congruence"] == congruence
                and int(row["target_relation"]) == target
            )
            answer_balance &= counts == Counter({"e0": 2, "e1": 2})
    checks["causal_prefix_exact_through_facts"] = prefix_ok
    checks["relation_identity_behaviorally_required"] = necessity_ok
    checks["routes_share_facts_and_answers"] = same_facts_answers
    checks["answer_identity_balanced"] = answer_balance
    checks["pair_family_balance"] = (
        Counter(PAIR_FAMILY.values())
        == Counter({family: 3 for family in FAMILIES})
    )
    counts = Counter(
        (row["relation_pair"], row["surface"], row["split"])
        for row in rows
    )
    checks["cell_balance"] = len(set(counts.values())) == 1
    paraphrase_overlaps = []
    for row in rows:
        if row["route_type"] != "paraphrase":
            continue
        left = set(row["selector_token_ids"])
        right = set(row["visible_target_alias_token_ids"])
        union = left | right
        paraphrase_overlaps.append(
            len(left & right) / len(union) if union else 0.0
        )
    return {
        "schema_version": "phase1103_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(rows),
        "unit_count": len(grouped),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "paraphrase_token_jaccard": {
            "minimum": min(paraphrase_overlaps),
            "median": sorted(paraphrase_overlaps)[
                len(paraphrase_overlaps) // 2
            ],
            "maximum": max(paraphrase_overlaps),
            "interpretation": (
                "Descriptive tokenizer overlap only; no token-disjointness "
                "claim is made."
            ),
        },
        "case_digest": digest(rows),
    }


def main() -> None:
    if not SOURCE_PHASE1102_AUTHORIZATION.exists():
        raise RuntimeError("Phase1102 authorization is missing")
    if not SOURCE_PHASE1102_AUDIT.exists():
        raise RuntimeError("Phase1102 result audit is missing")
    source_authorization = read_json(SOURCE_PHASE1102_AUTHORIZATION)
    source_audit = read_json(SOURCE_PHASE1102_AUDIT)
    if source_authorization.get("hidden_scan_authorized", True):
        raise RuntimeError("Phase1103 requires the frozen Phase1102 behavior stop")
    if not source_audit.get("all_checks_passed", False):
        raise RuntimeError("Phase1102 result audit did not pass")
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    selected_names = select_names(tokenizers)
    model_case_digests: dict[str, str] = {}
    model_audits: dict[str, Any] = {}
    for model in MODELS:
        rows = build_model_cases(tokenizers[model], model, selected_names)
        audit = audit_model(model, rows, selected_names)
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"Phase1103 protocol audit failed for {model}: {audit}"
            )
        write_jsonl(
            OUT_ROOT / "protocol" / f"cases.{model}.jsonl", rows
        )
        write_json(
            OUT_ROOT / "protocol" / f"audit.{model}.json", audit
        )
        model_case_digests[model] = audit["case_digest"]
        model_audits[model] = audit
    prereg = {
        "schema_version": "phase1103_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "formal_models": list(FORMAL_MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "families": list(FAMILIES),
        "relations": list(RELATIONS),
        "relation_pairs": list(RELATION_PAIRS),
        "pair_relations": {
            pair: list(relations)
            for pair, relations in PAIR_RELATIONS.items()
        },
        "pair_family": PAIR_FAMILY,
        "surfaces": list(SURFACES),
        "templates": list(TEMPLATES),
        "templates_by_split": {
            split: list(templates)
            for split, templates in TEMPLATES_BY_SPLIT.items()
        },
        "items_per_template": ITEMS_PER_TEMPLATE,
        "states": list(STATES),
        "selected_names": list(selected_names),
        "protocol_revision_history": {
            "revision1": (
                "Stopped before case generation or model execution because "
                "only four unused names remained single-token in all three "
                "tokenizers; 24 were required."
            ),
            "revision2": (
                "Uses the first 12 previously frozen Phase1101 names only "
                "for qualification and the first 12 disjoint Phase1102 "
                "names only for confirmation. No Phase1103 effect was seen."
            ),
        },
        "case_count_per_model": len(read_jsonl(
            OUT_ROOT / "protocol" / f"cases.{MODELS[0]}.jsonl"
        )),
        "generation_steps": GENERATION_STEPS,
        "generation_items_per_cell": GENERATION_ITEMS_PER_CELL,
        "capture_roles": list(CAPTURE_ROLES),
        "causal_design": {
            "component": "residual_stream_after_sampled_layer",
            "depth_fractions": list(CAUSAL_DEPTH_FRACTIONS),
            "patch_role": "query_end",
            "patch_alpha": PATCH_ALPHA,
            "qualification_items": list(CAUSAL_DISCOVERY_ITEMS),
            "confirmation_items": list(CAUSAL_CONFIRMATION_ITEMS),
            "relation_orders": list(CAUSAL_RELATION_ORDERS),
            "primary_transport": [
                "exact_delta_to_paraphrase_target",
                "paraphrase_delta_to_exact_target",
            ],
            "matched_controls": [
                "ordinal_delta", "behavior_passing_wrong_pair_delta",
                "equal_norm_random_delta", "congruent_target_collateral",
            ],
            "depth_selection": (
                "Per model, pair, and surface, choose one depth using only "
                "qualification prompts; apply it unchanged to confirmation."
            ),
        },
        "behavior_claim_scope": (
            "Pair-specific prospective authorization. A passing subset cannot "
            "support a 15-pair family-wide claim."
        ),
        "primary_object": (
            "Whether a late natural relation address is stable across "
            "nonidentical paraphrases and whether its signed query-end "
            "difference causally transports relation selection."
        ),
        "explicit_nonclaims": [
            "The prompts are controlled naturalistic assessments, not an unlabeled natural corpus.",
            "A relation word may still act as an instruction/address rather than exposing a pretrained knowledge coordinate.",
            "Residual transport does not by itself identify a head, MLP, neuron, or globally conserved coordinate.",
            "Passing pairs do not establish a complete relation-family mechanism.",
            "Compression, minimality, optimality, and unique lexical niches are not assumed by the formulas.",
        ],
        "forbidden_escalations": [
            "post-hoc behavior threshold changes",
            "post-hoc pair inclusion after hidden effects are observed",
            "family-wide claims from pair-specific authorization",
            "component or neuron localization before independent causal confirmation",
            "mixing API or quantized large-model behavior with FP16 internal-mechanism evidence",
        ],
        "evidence_thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "automatic_next_rule": (
            "Behavior P2 automatically authorizes signed residual transport "
            "for only the frozen shared pairs. Only P4-P7 automatically "
            "authorize a later component-level Phase1104."
        ),
        "source_phase1102_authorization_digest": source_authorization[
            "authorization_digest"
        ],
        "source_phase1102_audit_digest": source_audit["audit_digest"],
        "model_case_digests": model_case_digests,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    aggregate_audit = {
        "schema_version": "phase1103_protocol_audit.v1",
        "phase": PHASE,
        "models": model_audits,
        "checks": {
            "all_model_audits_passed": all(
                audit["all_checks_passed"]
                for audit in model_audits.values()
            ),
            "model_case_counts_equal": len({
                audit["case_count"] for audit in model_audits.values()
            }) == 1,
            "name_worlds_frozen_before_phase1103": all(
                name in prior_names() for name in selected_names
            ),
            "qualification_confirmation_names_disjoint": not (
                set(selected_names[:2 * 2 * ITEMS_PER_TEMPLATE])
                & set(selected_names[2 * 2 * ITEMS_PER_TEMPLATE:])
            ),
            "source_phase1102_stopped_before_hidden": not source_authorization[
                "hidden_scan_authorized"
            ],
            "source_phase1102_audit_passed": source_audit[
                "all_checks_passed"
            ],
        },
        "model_case_digests": model_case_digests,
        "protocol_digest": prereg["protocol_digest"],
    }
    aggregate_audit["all_checks_passed"] = all(
        aggregate_audit["checks"].values()
    )
    aggregate_audit["audit_digest"] = digest(aggregate_audit)
    write_json(OUT_ROOT / "protocol" / "audit.json", aggregate_audit)
    print(json.dumps({
        "phase": PHASE,
        "case_count_per_model": prereg["case_count_per_model"],
        "selected_names": list(selected_names),
        "protocol_digest": prereg["protocol_digest"],
        "audit_digest": aggregate_audit["audit_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
