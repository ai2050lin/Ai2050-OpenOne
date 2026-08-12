#!/usr/bin/env python3
"""Freeze the Phase1101 behavior-necessary relation-identity routing protocol."""

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
import phase1098_relative_relation_geometry_protocol as base
import phase1099_relation_family_atlas_protocol as phase1099


PHASE = 1101
PROTOCOL_REVISION = 2
MODELS = ("qwen3", "glm4", "deepseek7b")
FORMAL_MODELS = ("qwen3", "glm4")
PRECISION = "fp16"
QUANTIZATION = "none"
SURFACES = ("en", "zh")
TEMPLATES = (0, 1, 2, 3)
TEMPLATES_BY_SPLIT = {"discovery": (0, 1), "confirmation": (2, 3)}
SPLITS = tuple(TEMPLATES_BY_SPLIT)
ITEMS_PER_TEMPLATE = 3
ROUTE_TYPES = ("semantic", "ordinal")
CONGRUENCES = ("conflict", "congruent")
TARGET_RELATIONS = (0, 1)
RELATION_ORDERS = (0, 1)
ORIENTATIONS = (0, 1)
ASSISTANT_PREFILL = "Answer:"
CONTINUATION_PREFIX = " "
GENERATION_STEPS = 6
GENERATION_ITEMS_PER_CELL = 1
CAPTURE_ROLES = ("facts_end", "selector_end", "query_end", "answer_boundary")
FIELDS = (
    "semantic_routing",
    "ordinal_routing",
    "semantic_selector",
    "ordinal_selector",
)
PRIMARY_FIELD = "semantic_routing"
MATCHED_CONTROLS = ("ordinal_routing", "semantic_selector", "ordinal_selector")
PRIMARY_ROLE = "answer_boundary"
DEPTH_FRACTIONS = tuple(value / 10.0 for value in range(11))
COMPONENTS = ("residual", "attention_output", "mlp_output")
REVISION1_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1101_relation_identity_routing"
REVISION1_AUTHORIZATION = REVISION1_ROOT / "analysis" / "behavior_authorization.json"
OUT_ROOT = REVISION1_ROOT / "revision2"
SOURCE_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1100_relation_graph_inheritance"
SOURCE_PHASE1100 = SOURCE_ROOT / "analysis" / "final_summary.json"
SOURCE_PHASE1100_AUDIT = SOURCE_ROOT / "audit" / "result_audit.json"


write_json = base.write_json
write_jsonl = base.write_jsonl
read_json = base.read_json
read_jsonl = base.read_jsonl
digest = base.digest
sha256_text = base.sha256_text


RELATION_ROWS = (
    ("height", "physical_magnitude", "height", "身高"),
    ("weight", "physical_magnitude", "weight", "体重"),
    ("speed", "physical_magnitude", "speed", "速度"),
    ("brightness", "physical_magnitude", "brightness", "亮度"),
    ("temperature", "physical_magnitude", "temperature", "温度"),
    ("price", "physical_magnitude", "price", "价格"),
    ("arrival_time", "temporal_order", "arrival time", "到达时间"),
    ("departure_time", "temporal_order", "departure time", "出发时间"),
    ("start_time", "temporal_order", "start time", "开始时间"),
    ("finish_time", "temporal_order", "finish time", "完成时间"),
    ("registration_time", "temporal_order", "registration time", "登记时间"),
    ("publication_time", "temporal_order", "publication time", "发布时间"),
    ("north_position", "spatial_order", "northward position", "南北位置"),
    ("east_position", "spatial_order", "eastward position", "东西位置"),
    ("elevation", "spatial_order", "elevation", "海拔"),
    ("forward_position", "spatial_order", "forward position", "前后位置"),
    ("distance", "spatial_order", "distance", "距离"),
    ("depth", "spatial_order", "depth", "深度"),
    ("authority", "social_status", "authority", "权威程度"),
    ("seniority", "social_status", "seniority", "资历"),
    ("influence", "social_status", "influence", "影响力"),
    ("popularity", "social_status", "popularity", "受欢迎程度"),
    ("responsibility", "social_status", "responsibility", "责任程度"),
    ("leadership_rank", "social_status", "leadership rank", "领导级别"),
    ("causal_influence", "epistemic_causal", "causal influence", "因果影响"),
    ("evidence_strength", "epistemic_causal", "evidence strength", "证据强度"),
    ("likelihood", "epistemic_causal", "likelihood", "可能性"),
    ("certainty", "epistemic_causal", "certainty", "确定性"),
    ("explanatory_power", "epistemic_causal", "explanatory power", "解释力"),
    ("dependency_strength", "epistemic_causal", "dependency strength", "依赖强度"),
)
RELATIONS = tuple(row[0] for row in RELATION_ROWS)
RELATION_FAMILY = {row[0]: row[1] for row in RELATION_ROWS}
RELATION_LABELS = {row[0]: {"en": row[2], "zh": row[3]} for row in RELATION_ROWS}
RELATION_PAIRS = tuple(
    f"{RELATIONS[index]}__{RELATIONS[index + 1]}"
    for index in range(0, len(RELATIONS), 2)
)
PAIR_RELATIONS = {
    pair: tuple(pair.split("__", 1)) for pair in RELATION_PAIRS
}
PAIR_FAMILY = {
    pair: RELATION_FAMILY[PAIR_RELATIONS[pair][0]] for pair in RELATION_PAIRS
}
FAMILIES = tuple(dict.fromkeys(PAIR_FAMILY[pair] for pair in RELATION_PAIRS))


def state_name(
    route_type: str,
    congruence: str,
    target_relation: int,
    relation_order: int,
    orientation: int,
) -> str:
    return (
        f"r{route_type[0]}_c{0 if congruence == 'conflict' else 1}"
        f"_q{target_relation}_o{relation_order}_b{orientation}"
    )


STATES = tuple(
    state_name(route_type, congruence, target, order, orientation)
    for route_type in ROUTE_TYPES
    for congruence in CONGRUENCES
    for target in TARGET_RELATIONS
    for order in RELATION_ORDERS
    for orientation in ORIENTATIONS
)


def state_factors(state: str) -> tuple[str, str, int, int, int]:
    for route_type, congruence, target, order, orientation in itertools.product(
        ROUTE_TYPES, CONGRUENCES, TARGET_RELATIONS, RELATION_ORDERS, ORIENTATIONS
    ):
        if state == state_name(route_type, congruence, target, order, orientation):
            return route_type, congruence, target, order, orientation
    raise ValueError(f"unknown Phase1101 state: {state}")


THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_candidate_accuracy": 0.90,
    "minimum_conflict_accuracy": 0.90,
    "minimum_congruent_accuracy": 0.90,
    "minimum_pair_cell_accuracy": 0.80,
    "minimum_passing_pairs": 13,
    "minimum_generation_accuracy": 0.75,
    "minimum_behavior_models": 2,
    "minimum_hidden_finite_fraction": 0.97,
    "pre_query_tolerance": 1e-8,
    "minimum_graph_finite_fraction": 0.99,
    "minimum_inheritance_cosine": 0.50,
    "minimum_family_permutation_margin": 0.02,
    "minimum_within_family_permutation_margin": 0.02,
    "minimum_specificity_advantage": 0.05,
    "minimum_confirmation_cells_per_surface": 1,
    "minimum_surface_passes_per_formal_model": 2,
    "minimum_cross_model_curve_cosine": 0.75,
    "maximum_cross_model_curve_mean_absolute_error": 0.18,
    "minimum_cross_model_curve_cells": 4,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": "All protocol, source-order, tokenization, causal-prefix, answer-balance, and factorial audits pass.",
    "P2": "At least two models solve both semantic and ordinal conflict routing for at least 13 of 15 relation pairs, with candidate and natural-generation behavior above frozen thresholds.",
    "P3": "At least two behavior-authorized models pass hidden finiteness and exact pre-query-zero audits.",
    "P4": "The Phase1100 lexical pair-difference graph predicts the behavior-necessary semantic-routing graph on independent templates in both surfaces of Qwen3 and GLM4, beating all 120 family and 7775 nonidentity within-family permutations.",
    "P5": "Lexical inheritance is specific to semantic relation-address routing and exceeds ordinal routing and selector-form controls by at least 0.05 in both surfaces of Qwen3 and GLM4.",
    "P6": "The normalized semantic-routing inheritance trajectory repeats across Qwen3 and GLM4 in at least four of six surface-component cells.",
    "P7": "The primary target excludes logits, generated scores, PCA, learned probes, and post-hoc component selection; DS7B remains exploratory.",
    "P8": "Only P1-P7 jointly authorize a later causal interface test; failure stops automatic escalation and cannot be rewritten as semantic-interface success.",
}


SHELLS = {
    "en": (
        "Two independent rankings are given and they may disagree; use only the one requested in the question. Record 1: {fact1}. Record 2: {fact2}. Query: {question}? Return exactly one person name.",
        "Use the two separate ranking records. They may conflict, so use only the requested one. First record: {fact1}. Second record: {fact2}. Final question: {question}? Answer with one person name only.",
        "Here are two unrelated comparison files. They may disagree; use only the requested file. File one states: {fact1}. File two states: {fact2}. Determine: {question}? Write exactly one person name.",
        "Consider both ranking entries independently. They may conflict; use only the requested entry. Entry one says: {fact1}. Entry two says: {fact2}. Please decide: {question}? Give one person name only.",
    ),
    "zh": (
        "给出两个相互独立且可能冲突的排序；请只使用问题指定的一项。记录一：{fact1}。记录二：{fact2}。问题：{question}？只回答一个人名。",
        "请使用两条彼此独立的排序记录。它们可能相反，因此只使用被询问的一条。第一条：{fact1}。第二条：{fact2}。最终问题：{question}？仅回答一个人名。",
        "这里有两个互不相关且可能冲突的比较档案；只使用问题指定的档案。档案一写明：{fact1}。档案二写明：{fact2}。请判断：{question}？只写一个人名。",
        "分别考虑下面两项可能相反的排序；只使用问题指定的条目。条目一：{fact1}。条目二：{fact2}。请回答：{question}？仅给出一个人名。",
    ),
}


ORDINAL_SELECTORS = {
    "en": {
        0: ("first record", "second record"),
        1: ("first record", "second record"),
        2: ("first file", "second file"),
        3: ("first entry", "second entry"),
    },
    "zh": {
        0: ("第一条记录", "第二条记录"),
        1: ("第一条记录", "第二条记录"),
        2: ("第一个档案", "第二个档案"),
        3: ("第一个条目", "第二个条目"),
    },
}


def split_for_template(template: int) -> str:
    for split, templates in TEMPLATES_BY_SPLIT.items():
        if template in templates:
            return split
    raise ValueError(f"unknown template: {template}")


def mark(text: str, value: str, start: int = 0) -> tuple[int, int, str]:
    position = text.find(value, start)
    if position < 0:
        raise RuntimeError(f"missing marked value {value!r}")
    return position, position + len(value), value


def old_names() -> set[str]:
    values = set(phase1099.old_names())
    prereg = phase1099.OUT_ROOT / "protocol" / "preregistration.json"
    if prereg.exists():
        values.update(read_json(prereg).get("selected_names", []))
    return values


def select_names(tokenizers: dict[str, Any]) -> tuple[str, ...]:
    revision1_prereg = REVISION1_ROOT / "protocol" / "preregistration.json"
    if revision1_prereg.exists():
        names = tuple(read_json(revision1_prereg).get("selected_names", []))
        if len(names) == len(TEMPLATES) * ITEMS_PER_TEMPLATE * 2:
            for model, tokenizer in tokenizers.items():
                ids = [tokenizer.encode(" " + name, add_special_tokens=False) for name in names]
                if any(len(values) != 1 for values in ids):
                    raise RuntimeError(f"Revision1 name tokenization drift for {model}")
                if len({int(values[0]) for values in ids}) != len(ids):
                    raise RuntimeError(f"Revision1 name token collision for {model}")
            return names
    candidates = tuple(dict.fromkeys(
        phase1099.base.phase1097.EXTRA_NAME_CANDIDATES
        + phase1099.base.phase1096.ADDITIONAL_NAME_CANDIDATES
        + phase1099.base.name_source.HELDOUT_NAME_CANDIDATES
    ))
    ranked = sorted(
        candidates,
        key=lambda value: hashlib.sha256(
            f"phase1101|{value}".encode("utf-8")
        ).hexdigest(),
    )
    excluded = old_names()
    used_ids = {model: set() for model in MODELS}
    selected: list[str] = []
    required = len(TEMPLATES) * ITEMS_PER_TEMPLATE * 2
    for name in ranked:
        if name in excluded:
            continue
        ids: dict[str, int] = {}
        for model, tokenizer in tokenizers.items():
            values = tokenizer.encode(" " + name, add_special_tokens=False)
            if len(values) != 1 or int(values[0]) in used_ids[model]:
                break
            ids[model] = int(values[0])
        if len(ids) != len(MODELS):
            continue
        selected.append(name)
        for model, token_id in ids.items():
            used_ids[model].add(token_id)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise RuntimeError(f"need {required} new one-token names, found {len(selected)}")
    return tuple(selected)


def name_pair(names: tuple[str, ...], template: int, item_index: int) -> tuple[str, str]:
    index = template * ITEMS_PER_TEMPLATE + item_index
    return names[2 * index], names[2 * index + 1]


def relation_fact(surface: str, relation: str, high: str, low: str) -> str:
    label = RELATION_LABELS[relation][surface]
    if surface == "en":
        return f"On {label}, [ {high} ] ranks above [ {low} ]"
    return f"按{label}比较，[ {high} ] 排在 [ {low} ] 前面"


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
    winner0, loser0 = (entity0, entity1) if orientation == 0 else (entity1, entity0)
    if congruence == "conflict":
        winner1, loser1 = loser0, winner0
    else:
        winner1, loser1 = winner0, loser0
    facts_by_relation = {
        relation0: relation_fact(surface, relation0, winner0, loser0),
        relation1: relation_fact(surface, relation1, winner1, loser1),
    }
    displayed_relations = (
        (relation0, relation1) if relation_order == 0 else (relation1, relation0)
    )
    fact1 = facts_by_relation[displayed_relations[0]]
    fact2 = facts_by_relation[displayed_relations[1]]
    target_name = (relation0, relation1)[target_relation]
    if route_type == "semantic":
        selector = RELATION_LABELS[target_name][surface]
        if surface == "en":
            question = f"under the {selector} ranking, which person ranks above the other"
        else:
            question = f"按照{selector}排序，哪个人排在另一个人前面"
    else:
        display_index = displayed_relations.index(target_name)
        selector = ORDINAL_SELECTORS[surface][template][display_index]
        if surface == "en":
            question = f"according to the {selector}, which person ranks above the other"
        else:
            question = f"按照{selector}，哪个人排在另一个人前面"
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
        "displayed_relations": displayed_relations,
        "fact1": fact1,
        "fact2": fact2,
        "question": question,
        "selector": selector,
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
    route_type, congruence, target, order, orientation = state_factors(state)
    names = name_pair(selected_names, template, item_index)
    raw_prompt, raw_spans, meta = render_prompt(
        relation_pair, surface, template, route_type, congruence,
        target, order, orientation, names,
    )
    rendered = (
        base.behavior_tools.render_native(
            tokenizer, model_name, raw_prompt, with_system=False
        )
        + ASSISTANT_PREFILL
    )
    input_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    role_spans = base.offset_token_spans(tokenizer, rendered, raw_prompt, raw_spans)
    role_positions = {role: int(span[1]) for role, span in role_spans.items()}
    role_positions["answer_boundary"] = len(input_ids) - 1
    role_spans["answer_boundary"] = (len(input_ids) - 1, len(input_ids) - 1)
    expected_class = "e0" if meta["expected"] == meta["entity0"] else "e1"
    candidate_labels = {"e0": meta["entity0"], "e1": meta["entity1"]}
    candidate_token_ids = {
        key: base.continuation_ids(tokenizer, rendered, label)
        for key, label in candidate_labels.items()
    }
    unit_id = (
        f"phase1101.{model_name}.{relation_pair}.{surface}.t{template}.i{item_index:02d}"
    )
    return {
        "schema_version": "phase1101_relation_identity_routing_case.v2",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{unit_id}.{state}",
        "unit_id": unit_id,
        "superunit_id": f"phase1101.{model_name}.{surface}.t{template}.i{item_index:02d}",
        "relation_pair": relation_pair,
        "relation0": meta["relation0"],
        "relation1": meta["relation1"],
        "family": PAIR_FAMILY[relation_pair],
        "surface": surface,
        "split": split_for_template(template),
        "template": template,
        "item_index": item_index,
        "state": state,
        "route_type": route_type,
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
            key: [int(values[0])] for key, values in candidate_token_ids.items()
        },
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_spans": {key: list(value) for key, value in role_spans.items()},
        "role_positions": role_positions,
        "fact1_text": meta["fact1"],
        "fact2_text": meta["fact2"],
        "question_text": meta["question"],
        "selector_text": meta["selector"],
        "displayed_relations": list(meta["displayed_relations"]),
        "continuation_prefix": CONTINUATION_PREFIX,
        "prompt_digest": sha256_text(raw_prompt),
    }


def build_model_cases(tokenizer, model_name: str, selected_names: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in RELATION_PAIRS:
        for surface in SURFACES:
            for template in TEMPLATES:
                for item_index in range(ITEMS_PER_TEMPLATE):
                    for state in STATES:
                        rows.append(build_case(
                            tokenizer, model_name, selected_names, pair, surface,
                            template, item_index, state, len(rows),
                        ))
    return rows


def audit_model(model_name: str, rows: list[dict[str, Any]], selected_names: tuple[str, ...]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    expected_count = (
        len(RELATION_PAIRS) * len(SURFACES) * len(TEMPLATES)
        * ITEMS_PER_TEMPLATE * len(STATES)
    )
    checks["case_count"] = len(rows) == expected_count
    checks["record_ids_unique"] = len({row["record_id"] for row in rows}) == len(rows)
    checks["state_count"] = len(STATES) == 32
    checks["candidate_first_tokens_distinct"] = all(
        row["candidate_first_token_ids"]["e0"]
        != row["candidate_first_token_ids"]["e1"] for row in rows
    )
    checks["candidate_names_one_token"] = all(
        len(row["candidate_token_ids"][key]) == 1
        for row in rows for key in ("e0", "e1")
    )
    checks["roles_in_range"] = all(
        0 <= int(row["role_positions"][role]) < len(row["input_ids"])
        for row in rows for role in CAPTURE_ROLES
    )
    checks["new_names"] = not (set(selected_names) & old_names())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["unit_id"])].append(row)
    checks["unit_count"] = len(grouped) == (
        len(RELATION_PAIRS) * len(SURFACES) * len(TEMPLATES) * ITEMS_PER_TEMPLATE
    )
    checks["complete_factorial"] = all(
        {row["state"] for row in unit} == set(STATES) for unit in grouped.values()
    )
    prefix_ok = True
    answer_balance_ok = True
    identity_required_ok = True
    same_facts_across_routes = True
    for unit in grouped.values():
        index = {
            (row["route_type"], row["congruence"], int(row["target_relation"]),
             int(row["relation_order"]), int(row["orientation"])): row
            for row in unit
        }
        for route, congruence, order, orientation in itertools.product(
            ROUTE_TYPES, CONGRUENCES, RELATION_ORDERS, ORIENTATIONS
        ):
            left = index[(route, congruence, 0, order, orientation)]
            right = index[(route, congruence, 1, order, orientation)]
            stop = int(left["role_positions"]["facts_end"]) + 1
            prefix_ok &= (
                stop == int(right["role_positions"]["facts_end"]) + 1
                and left["input_ids"][:stop] == right["input_ids"][:stop]
            )
            if congruence == "conflict":
                identity_required_ok &= left["expected_class"] != right["expected_class"]
            else:
                identity_required_ok &= left["expected_class"] == right["expected_class"]
        for congruence, target, order, orientation in itertools.product(
            CONGRUENCES, TARGET_RELATIONS, RELATION_ORDERS, ORIENTATIONS
        ):
            semantic = index[("semantic", congruence, target, order, orientation)]
            ordinal = index[("ordinal", congruence, target, order, orientation)]
            same_facts_across_routes &= (
                semantic["fact1_text"] == ordinal["fact1_text"]
                and semantic["fact2_text"] == ordinal["fact2_text"]
                and semantic["expected_class"] == ordinal["expected_class"]
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
            answer_balance_ok &= counts == Counter({"e0": 2, "e1": 2})
    checks["causal_prefix_exact"] = prefix_ok
    checks["answer_identity_balanced"] = answer_balance_ok
    checks["relation_identity_behaviorally_required"] = identity_required_ok
    checks["semantic_ordinal_facts_and_answers_matched"] = same_facts_across_routes
    counts = Counter((row["relation_pair"], row["surface"], row["split"]) for row in rows)
    checks["cell_balance"] = len(set(counts.values())) == 1
    checks["pair_family_balance"] = Counter(PAIR_FAMILY.values()) == Counter({family: 3 for family in FAMILIES})
    return {
        "schema_version": "phase1101_protocol_model_audit.v1",
        "model": model_name,
        "case_count": len(rows),
        "unit_count": len(grouped),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(rows),
    }


def main() -> None:
    if not SOURCE_PHASE1100.exists() or not SOURCE_PHASE1100_AUDIT.exists():
        raise RuntimeError("Phase1100 frozen source artifacts are missing")
    source_final = read_json(SOURCE_PHASE1100)
    source_audit = read_json(SOURCE_PHASE1100_AUDIT)
    if not REVISION1_AUTHORIZATION.exists():
        raise RuntimeError("Phase1101 Revision1 behavior authorization is missing")
    revision1_authorization = read_json(REVISION1_AUTHORIZATION)
    if revision1_authorization.get("hidden_scan_authorized", True):
        raise RuntimeError("Revision2 is allowed only after the frozen Revision1 behavior stop")
    if not source_audit.get("all_checks_passed", False):
        raise RuntimeError("Phase1100 result audit did not pass")
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    selected_names = select_names(tokenizers)
    model_digests: dict[str, str] = {}
    model_audits: dict[str, Any] = {}
    for model in MODELS:
        rows = build_model_cases(tokenizers[model], model, selected_names)
        audit = audit_model(model, rows, selected_names)
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"Phase1101 protocol audit failed for {model}: {audit}")
        write_jsonl(OUT_ROOT / "protocol" / f"cases.{model}.jsonl", rows)
        write_json(OUT_ROOT / "protocol" / f"audit.{model}.json", audit)
        model_digests[model] = audit["case_digest"]
        model_audits[model] = audit
    source_relations = tuple(phase1099.RELATIONS)
    if source_relations != RELATIONS:
        raise RuntimeError("Phase1100 relation order does not match Phase1101")
    preregistration = {
        "schema_version": "phase1101_preregistration.v2",
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
        "pair_relations": {key: list(value) for key, value in PAIR_RELATIONS.items()},
        "pair_family": PAIR_FAMILY,
        "surfaces": list(SURFACES),
        "templates": list(TEMPLATES),
        "templates_by_split": {key: list(value) for key, value in TEMPLATES_BY_SPLIT.items()},
        "items_per_template": ITEMS_PER_TEMPLATE,
        "states": list(STATES),
        "selected_names": list(selected_names),
        "case_count_per_model": len(read_jsonl(OUT_ROOT / "protocol" / f"cases.{MODELS[0]}.jsonl")),
        "generation_steps": GENERATION_STEPS,
        "generation_items_per_cell": GENERATION_ITEMS_PER_CELL,
        "capture_roles": list(CAPTURE_ROLES),
        "fields": list(FIELDS),
        "primary_field": PRIMARY_FIELD,
        "matched_controls": list(MATCHED_CONTROLS),
        "primary_role": PRIMARY_ROLE,
        "sampled_event_grid": {
            "components": list(COMPONENTS),
            "relative_depths": list(DEPTH_FRACTIONS),
        },
        "primary_object": "Exact centered 15-pair Gram geometry of answer-balanced, congruent-subtracted semantic relation-address routing.",
        "behavioral_necessity": "Within every conflict state, switching the late target relation flips the correct entity while facts, candidates, output protocol, relation order, and entity multiset remain fixed.",
        "matched_route_control": "Ordinal routing selects the same fact and answer by first/second record rather than by relation identity.",
        "source_object": "Phase1100 input-query-polarity pair differences, r1 minus r0, for the same 15 frozen relation pairs.",
        "forbidden_primary_inputs": [
            "candidate logits", "output margins", "generation scores", "PCA",
            "learned probes", "post-hoc components", "post-hoc roles",
        ],
        "evidence_thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "automatic_next_rule": "Only P1-P7 jointly authorize an independent causal interface phase.",
        "revision_history": {
            "revision1_authorization_digest": revision1_authorization["authorization_digest"],
            "revision1_hidden_scan_authorized": revision1_authorization["hidden_scan_authorized"],
            "revision1_failure": "Qwen3 and DS7B failed the frozen conflict-routing and pair-coverage behavior gates; no hidden states were scanned.",
            "revision2_single_change": "Keep relations, names, factorial states, outputs, sample size, and thresholds fixed; align ordinal selector nouns with record/file/entry carrier nouns and explicitly state that conflicting rankings must be handled by using only the requested one.",
            "further_behavior_revisions_authorized": False,
        },
        "source_phase1100_final_digest": source_final["final_digest"],
        "source_phase1100_audit_digest": source_audit["audit_digest"],
        "model_case_digests": model_digests,
        "model_audits": model_audits,
    }
    preregistration["protocol_digest"] = digest(preregistration)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", preregistration)
    common_audit = {
        "schema_version": "phase1101_protocol_audit.v1",
        "phase": PHASE,
        "checks": {
            "all_model_audits_pass": all(row["all_checks_passed"] for row in model_audits.values()),
            "source_phase1100_audit_passed": bool(source_audit["all_checks_passed"]),
            "source_relation_order_exact": source_relations == RELATIONS,
            "fifteen_pairs_three_per_family": (
                len(RELATION_PAIRS) == 15
                and Counter(PAIR_FAMILY.values()) == Counter({family: 3 for family in FAMILIES})
            ),
            "formal_models_frozen": FORMAL_MODELS == ("qwen3", "glm4"),
            "fp16_no_quantization": PRECISION == "fp16" and QUANTIZATION == "none",
        },
        "model_case_digests": model_digests,
        "protocol_digest": preregistration["protocol_digest"],
    }
    common_audit["all_checks_passed"] = all(common_audit["checks"].values())
    common_audit["audit_digest"] = digest(common_audit)
    write_json(OUT_ROOT / "protocol" / "audit.json", common_audit)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "audit_digest": common_audit["audit_digest"],
        "case_count_per_model": preregistration["case_count_per_model"],
        "selected_names": selected_names,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
