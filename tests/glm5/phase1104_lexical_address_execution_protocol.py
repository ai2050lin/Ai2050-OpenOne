#!/usr/bin/env python3
"""Freeze Phase1104 lexical-address execution and causal-routing protocol."""

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
import phase1103_natural_relation_route_protocol as source


PHASE = 1104
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
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
LABEL_REGIMES = ("relation_label", "neutral_label")
ROUTE_TYPES = ("exact", "ordinal")
CONGRUENCES = ("conflict", "congruent")
TARGET_RELATIONS = (0, 1)
RELATION_ORDERS = (0, 1)
ORIENTATIONS = (0, 1)
ASSISTANT_PREFILL = "Answer:"
CONTINUATION_PREFIX = " "
CAPTURE_ROLES = ("facts_end", "selector_end", "query_end", "answer_boundary")
CAUSAL_DEPTH_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
CAUSAL_DISCOVERY_ITEMS = (0,)
CAUSAL_CONFIRMATION_ITEMS = (0, 1, 2)
CAUSAL_RELATION_ORDERS = (0,)
MAX_CAUSAL_PAIRS_PER_MODEL = 3
PATCH_ALPHA = 1.0
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1104_lexical_address_execution"
SOURCE_ROOT = source.OUT_ROOT
SOURCE_PREREG = SOURCE_ROOT / "protocol" / "preregistration.json"
SOURCE_DIAGNOSTIC = SOURCE_ROOT / "analysis" / "failure_diagnostic.json"
SOURCE_AUDIT = SOURCE_ROOT / "audit" / "result_audit.json"


write_json = tools.write_json
write_jsonl = tools.write_jsonl
read_json = tools.read_json
read_jsonl = tools.read_jsonl
digest = tools.digest


# Frozen from the descriptive exact-route subgate in Phase1103. These are
# prior hypotheses for a new phase, not a retroactive Phase1103 authorization.
CANDIDATE_PAIRS = (
    "temperature__price",
    "start_time__finish_time",
    "registration_time__publication_time",
    "influence__popularity",
    "responsibility__leadership_rank",
    "causal_influence__evidence_strength",
    "likelihood__certainty",
    "explanatory_power__dependency_strength",
)
PAIR_RELATIONS = phase1101.PAIR_RELATIONS
PAIR_FAMILY = phase1101.PAIR_FAMILY


NATURAL_SUFFIXES = {
    "en": ("index", "measure", "score", "rating"),
    "zh": ("指标", "度量", "分数", "等级"),
}
NEUTRAL_LABELS = {
    "en": (
        ("amber file", "cobalt file"),
        ("maple file", "cedar file"),
        ("silver record", "gold record"),
        ("river record", "harbor record"),
    ),
    "zh": (
        ("琥珀档案", "钴蓝档案"),
        ("枫叶档案", "雪松档案"),
        ("银色记录", "金色记录"),
        ("河流记录", "港口记录"),
    ),
}
SHELLS = {
    "en": (
        "A registry contains two independent keyed rankings. {fact1}. {fact2}. {question}? Return exactly one person name.",
        "Two keyed records compare the same people. {fact1}. Separately, {fact2}. {question}? Give one person name only.",
        "The archive lists two independent comparisons. {fact1}. In another entry, {fact2}. {question}? Respond with exactly one person name.",
        "A decision file has two keyed ranking entries. {fact1}. The other entry says, {fact2}. {question}? Write only the selected person's name.",
    ),
    "zh": (
        "登记表包含两项带键名的独立排名。{fact1}。{fact2}。{question}？只回答一个人名。",
        "两项带键名的记录比较同一组人。{fact1}。另一项是，{fact2}。{question}？仅给出一个人名。",
        "档案列出两项独立比较。{fact1}。另一条记录为，{fact2}。{question}？请准确回答一个人名。",
        "决策档案含有两项带键名的排名。{fact1}。另一项写着，{fact2}。{question}？只写被选中的人名。",
    ),
}
ORDINAL_SELECTORS = {
    "en": ("the first listed ranking", "the second listed ranking"),
    "zh": ("第一项列出的排名", "第二项列出的排名"),
}
ROUTE_CODES = {"exact": "e", "ordinal": "o"}
REGIME_CODES = {"relation_label": "r", "neutral_label": "n"}


THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_conflict_cell_accuracy": 0.80,
    "minimum_regime_accuracy": 0.90,
    "minimum_congruent_accuracy": 0.90,
    "minimum_models_for_cross_model_upgrade": 2,
    "minimum_causal_finite_fraction": 0.97,
    "minimum_causal_behavior_valid_fraction": 0.80,
    "minimum_raw_median_recovery": 0.10,
    "minimum_interaction_median_recovery": 0.10,
    "minimum_positive_fraction": 0.60,
    "minimum_each_direction_median_recovery": 0.05,
    "minimum_each_direction_positive_fraction": 0.55,
    "minimum_flip_rate": 0.10,
    "minimum_specificity_advantage": 0.05,
    "maximum_congruent_collateral_flip_rate": 0.10,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": "All source-chain, candidate-freeze, token, prefix, factorial, answer-balance, label-assignment, and fresh-pairing audits pass.",
    "P2": "At least one model-pair passes exact lexical routing in relation-label and neutral-label regimes across both independent splits; this alone authorizes only a model-specific lexical execution scan.",
    "P3": "The behavior gate excludes free generation and paraphrase. Those are separate interface and semantic-equivalence ledgers, not prerequisites for candidate-logit causal analysis.",
    "P4": "Within-regime raw selector replacement is an instrument check; it is not sufficient evidence for content-conditioned execution.",
    "P5": "A qualification-selected cross-regime residual depth transports relation choice on independent confirmation prompts with median recovery at least 0.10.",
    "P6": "The active-minus-congruent interaction delta exceeds congruent-selector, ordinal, wrong-pair, and equal-norm random controls by at least 0.05.",
    "P7": "A passing interaction cell has positive recovery in at least 60 percent of valid cases, at least 10 percent flips, and at most 10 percent congruent collateral flips.",
    "P8": "One-model confirmation is a model-specific mechanism entry; two-model confirmation is required before any cross-model or family-wide claim.",
    "P9": "Failure of P2 stops hidden-state access. Failure of P5-P7 retains a causal response map but does not establish a lexical execution mechanism.",
}


def split_for_template(template: int) -> str:
    for split, templates in TEMPLATES_BY_SPLIT.items():
        if template in templates:
            return split
    raise ValueError(template)


def state_name(
    regime: str,
    route: str,
    congruence: str,
    target: int,
    order: int,
    orientation: int,
) -> str:
    return (
        f"g{REGIME_CODES[regime]}_r{ROUTE_CODES[route]}"
        f"_c{0 if congruence == 'conflict' else 1}"
        f"_q{target}_o{order}_b{orientation}"
    )


STATES = tuple(
    state_name(*factors)
    for factors in itertools.product(
        LABEL_REGIMES, ROUTE_TYPES, CONGRUENCES,
        TARGET_RELATIONS, RELATION_ORDERS, ORIENTATIONS,
    )
)


def state_factors(state: str) -> tuple[str, str, str, int, int, int]:
    for factors in itertools.product(
        LABEL_REGIMES, ROUTE_TYPES, CONGRUENCES,
        TARGET_RELATIONS, RELATION_ORDERS, ORIENTATIONS,
    ):
        if state == state_name(*factors):
            return factors
    raise ValueError(state)


def mark(text: str, value: str, start: int = 0) -> tuple[int, int, str]:
    position = text.find(value, start)
    if position < 0:
        raise RuntimeError(f"missing marked value {value!r}")
    return position, position + len(value), value


def selected_names() -> tuple[str, ...]:
    names = tuple(read_json(SOURCE_PREREG)["selected_names"])
    if len(names) != 24:
        raise RuntimeError(f"expected 24 frozen names, found {len(names)}")
    return names


PAIRING_INDEX = {
    0: ((0, 5), (2, 7), (4, 9)),
    1: ((1, 6), (3, 10), (8, 11)),
}


def name_pair(names: tuple[str, ...], template: int, item: int) -> tuple[str, str]:
    split = split_for_template(template)
    pool = names[:12] if split == "qualification" else names[12:]
    local_template = TEMPLATES_BY_SPLIT[split].index(template)
    left, right = PAIRING_INDEX[local_template][item]
    return pool[left], pool[right]


def relation_label(relation: str, surface: str, template: int) -> str:
    if surface == "en":
        base = relation.replace("_", " ")
    else:
        base = source.RELATION_ALIASES[relation][surface][0]
    return f"{base} {NATURAL_SUFFIXES[surface][template]}"


def labels_for(
    pair: str, surface: str, template: int, item: int, regime: str
) -> dict[str, str]:
    relation0, relation1 = PAIR_RELATIONS[pair]
    if regime == "relation_label":
        return {
            relation0: relation_label(relation0, surface, template),
            relation1: relation_label(relation1, surface, template),
        }
    first, second = NEUTRAL_LABELS[surface][template]
    if (template + item) % 2:
        first, second = second, first
    return {relation0: first, relation1: second}


def fact(
    surface: str, label: str, relation: str, winner: str, loser: str
) -> str:
    if surface == "en":
        payload = relation.replace("_", " ")
        return (
            f"Under the key {label}, the {payload} comparison places "
            f"[ {winner} ] ahead of [ {loser} ]"
        )
    payload = source.RELATION_ALIASES[relation][surface][0]
    return (
        f"在键名{label}下，{payload}比较把 [ {winner} ] 排在 "
        f"[ {loser} ] 前面"
    )


def render_prompt(
    pair: str,
    surface: str,
    template: int,
    item: int,
    regime: str,
    route: str,
    congruence: str,
    target: int,
    order: int,
    orientation: int,
    names: tuple[str, str],
) -> tuple[str, dict[str, tuple[int, int, str]], dict[str, Any]]:
    entity0, entity1 = names
    relation0, relation1 = PAIR_RELATIONS[pair]
    winner0, loser0 = (
        (entity0, entity1) if orientation == 0 else (entity1, entity0)
    )
    if congruence == "conflict":
        winner1, loser1 = loser0, winner0
    else:
        winner1, loser1 = winner0, loser0
    labels = labels_for(pair, surface, template, item, regime)
    facts = {
        relation0: fact(
            surface, labels[relation0], relation0, winner0, loser0
        ),
        relation1: fact(
            surface, labels[relation1], relation1, winner1, loser1
        ),
    }
    displayed = (relation0, relation1) if order == 0 else (relation1, relation0)
    fact1, fact2 = facts[displayed[0]], facts[displayed[1]]
    target_relation = (relation0, relation1)[target]
    if route == "exact":
        selector = labels[target_relation]
    else:
        selector = ORDINAL_SELECTORS[surface][displayed.index(target_relation)]
    if surface == "en":
        question = f"Using {selector}, which person ranks ahead"
    else:
        question = f"使用{selector}时，哪个人排在前面"
    raw_prompt = SHELLS[surface][template].format(
        fact1=fact1, fact2=fact2, question=question
    )
    fact1_span = mark(raw_prompt, fact1)
    fact2_span = mark(raw_prompt, fact2, fact1_span[1])
    question_span = mark(raw_prompt, question, fact2_span[1])
    selector_span = mark(raw_prompt, selector, question_span[0])
    expected = winner0 if target == 0 else winner1
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
        "labels": labels,
        "neutral_key_swap": (template + item) % 2,
    }


def build_case(
    tokenizer,
    model_name: str,
    names: tuple[str, ...],
    pair: str,
    surface: str,
    template: int,
    item: int,
    state: str,
    case_index: int,
) -> dict[str, Any]:
    regime, route, congruence, target, order, orientation = state_factors(state)
    raw_prompt, raw_spans, meta = render_prompt(
        pair, surface, template, item, regime, route, congruence,
        target, order, orientation, name_pair(names, template, item),
    )
    rendered = (
        phase1101.base.behavior_tools.render_native(
            tokenizer, model_name, raw_prompt, with_system=False
        )
        + ASSISTANT_PREFILL
    )
    input_ids = [
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_spans = offset_token_spans(tokenizer, rendered, raw_prompt, raw_spans)
    role_positions = {role: int(span[1]) for role, span in role_spans.items()}
    role_positions["answer_boundary"] = len(input_ids) - 1
    role_spans["answer_boundary"] = (len(input_ids) - 1, len(input_ids) - 1)
    expected_class = "e0" if meta["expected"] == meta["entity0"] else "e1"
    candidate_labels = {"e0": meta["entity0"], "e1": meta["entity1"]}
    candidate_token_ids = {
        key: phase1101.base.continuation_ids(tokenizer, rendered, label)
        for key, label in candidate_labels.items()
    }
    unit_id = f"phase1104.{model_name}.{pair}.{surface}.t{template}.i{item:02d}"
    selector_token_ids = [
        int(value) for value in tokenizer.encode(meta["selector"], add_special_tokens=False)
    ]
    return {
        "schema_version": "phase1104_lexical_address_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{unit_id}.{state}",
        "unit_id": unit_id,
        "relation_pair": pair,
        "family": PAIR_FAMILY[pair],
        "surface": surface,
        "split": split_for_template(template),
        "template": template,
        "item_index": item,
        "state": state,
        "label_regime": regime,
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
        "selector_token_ids": selector_token_ids,
        "relation_labels": meta["labels"],
        "neutral_key_swap": meta["neutral_key_swap"],
        "displayed_relations": list(meta["displayed_relations"]),
        "continuation_prefix": CONTINUATION_PREFIX,
        "prompt_digest": hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
    }


def build_model_cases(tokenizer, model_name: str, names: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for pair in CANDIDATE_PAIRS:
        for surface in SURFACES:
            for template in TEMPLATES:
                for item in range(ITEMS_PER_TEMPLATE):
                    for state in STATES:
                        rows.append(build_case(
                            tokenizer, model_name, names, pair, surface,
                            template, item, state, len(rows),
                        ))
    return rows


def source_name_pairs() -> set[frozenset[str]]:
    prereg = read_json(SOURCE_PREREG)
    names = tuple(prereg["selected_names"])
    return {
        frozenset(source.name_pair(names, template, item))
        for template in source.TEMPLATES
        for item in range(source.ITEMS_PER_TEMPLATE)
    }


def audit_model(model_name: str, rows: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    expected = (
        len(CANDIDATE_PAIRS) * len(SURFACES) * len(TEMPLATES)
        * ITEMS_PER_TEMPLATE * len(STATES)
    )
    checks["case_count"] = len(rows) == expected
    checks["state_count"] = len(STATES) == 64
    checks["record_ids_unique"] = len({row["record_id"] for row in rows}) == len(rows)
    checks["candidate_names_one_token"] = all(
        len(row["candidate_token_ids"][key]) == 1
        for row in rows for key in ("e0", "e1")
    )
    checks["candidate_first_tokens_distinct"] = all(
        row["candidate_first_token_ids"]["e0"] != row["candidate_first_token_ids"]["e1"]
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
    qualification_names = set(names[:12])
    confirmation_names = set(names[12:])
    checks["name_splits_disjoint"] = not (qualification_names & confirmation_names)
    current_pairs = {
        frozenset(name_pair(names, template, item))
        for template in TEMPLATES for item in range(ITEMS_PER_TEMPLATE)
    }
    checks["name_pairings_fresh_vs_phase1103"] = not (current_pairs & source_name_pairs())
    checks["relation_label_splits_disjoint"] = all(
        not (
            {relation_label(rel, surface, t) for t in (0, 1)}
            & {relation_label(rel, surface, t) for t in (2, 3)}
        )
        for pair in CANDIDATE_PAIRS
        for rel in PAIR_RELATIONS[pair]
        for surface in SURFACES
    )
    checks["neutral_label_splits_disjoint"] = all(
        not (
            set(sum((NEUTRAL_LABELS[surface][t] for t in (0, 1)), ()))
            & set(sum((NEUTRAL_LABELS[surface][t] for t in (2, 3)), ()))
        )
        for surface in SURFACES
    )
    checks["exact_selector_repeated"] = all(
        row["raw_prompt"].count(row["selector_text"]) >= 2
        for row in rows if row["route_type"] == "exact"
    )
    checks["neutral_key_assignment_balanced"] = all(
        Counter(
            row["neutral_key_swap"] for row in rows
            if row["label_regime"] == "neutral_label"
            and row["relation_pair"] == pair and row["surface"] == surface
        )[0]
        == Counter(
            row["neutral_key_swap"] for row in rows
            if row["label_regime"] == "neutral_label"
            and row["relation_pair"] == pair and row["surface"] == surface
        )[1]
        for pair in CANDIDATE_PAIRS for surface in SURFACES
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["unit_id"]].append(row)
    checks["complete_factorial"] = all(
        {row["state"] for row in unit} == set(STATES) for unit in grouped.values()
    )
    prefix_ok = True
    necessity_ok = True
    answer_balance = True
    for unit in grouped.values():
        index = {
            (
                row["label_regime"], row["route_type"], row["congruence"],
                int(row["target_relation"]), int(row["relation_order"]),
                int(row["orientation"]),
            ): row
            for row in unit
        }
        for regime, congruence, order, orientation in itertools.product(
            LABEL_REGIMES, CONGRUENCES, RELATION_ORDERS, ORIENTATIONS
        ):
            reference = index[(regime, "exact", congruence, 0, order, orientation)]
            stop = int(reference["role_positions"]["facts_end"]) + 1
            for route, target in itertools.product(ROUTE_TYPES, TARGET_RELATIONS):
                row = index[(regime, route, congruence, target, order, orientation)]
                prefix_ok &= (
                    int(row["role_positions"]["facts_end"]) + 1 == stop
                    and row["input_ids"][:stop] == reference["input_ids"][:stop]
                )
            for route in ROUTE_TYPES:
                left = index[(regime, route, congruence, 0, order, orientation)]
                right = index[(regime, route, congruence, 1, order, orientation)]
                necessity_ok &= (
                    left["expected_class"] != right["expected_class"]
                    if congruence == "conflict"
                    else left["expected_class"] == right["expected_class"]
                )
        for regime, route, congruence, target in itertools.product(
            LABEL_REGIMES, ROUTE_TYPES, CONGRUENCES, TARGET_RELATIONS
        ):
            counts = Counter(
                row["expected_class"] for row in unit
                if row["label_regime"] == regime
                and row["route_type"] == route
                and row["congruence"] == congruence
                and int(row["target_relation"]) == target
            )
            answer_balance &= counts == Counter({"e0": 2, "e1": 2})
    checks["causal_prefix_exact_through_facts"] = prefix_ok
    checks["relation_choice_behaviorally_required"] = necessity_ok
    checks["answer_identity_balanced"] = answer_balance
    checks["prompt_digests_unique_within_model"] = (
        len({row["prompt_digest"] for row in rows}) == len(rows)
    )
    return {
        "schema_version": "phase1104_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(rows),
        "unit_count": len(grouped),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(rows),
    }


def main() -> None:
    source_audit = read_json(SOURCE_AUDIT)
    source_prereg = read_json(SOURCE_PREREG)
    source_diagnostic = read_json(SOURCE_DIAGNOSTIC)
    if not source_audit["all_checks_passed"]:
        raise RuntimeError("Phase1103 result audit did not pass")
    frozen_exact = tuple(
        source_diagnostic["models"]["glm4"]
        ["route_pairs_passing_both_splits"]["exact"]
    )
    if frozen_exact != CANDIDATE_PAIRS:
        raise RuntimeError(
            f"Phase1103 exact candidate drift: {frozen_exact!r}"
        )
    names = selected_names()
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    model_case_digests = {}
    model_audits = {}
    for model, tokenizer in tokenizers.items():
        rows = build_model_cases(tokenizer, model, names)
        audit = audit_model(model, rows, names)
        if not audit["all_checks_passed"]:
            failed = [key for key, value in audit["checks"].items() if not value]
            raise RuntimeError(f"Phase1104 {model} protocol audit failed: {failed}")
        write_jsonl(OUT_ROOT / "protocol" / f"cases.{model}.jsonl", rows)
        write_json(OUT_ROOT / "protocol" / f"audit.{model}.json", audit)
        model_case_digests[model] = audit["case_digest"]
        model_audits[model] = audit
        print(json.dumps({
            "phase": PHASE,
            "model": model,
            "case_count": len(rows),
            "case_digest": audit["case_digest"],
        }), flush=True)
    prereg = {
        "schema_version": "phase1104_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "candidate_pairs": list(CANDIDATE_PAIRS),
        "candidate_selection_rule": (
            "Frozen Phase1103 GLM4 exact-route descriptive subgate; reused only "
            "as prior hypotheses in this independent phase."
        ),
        "surfaces": list(SURFACES),
        "templates": list(TEMPLATES),
        "templates_by_split": {
            key: list(value) for key, value in TEMPLATES_BY_SPLIT.items()
        },
        "items_per_template": ITEMS_PER_TEMPLATE,
        "label_regimes": list(LABEL_REGIMES),
        "route_types": list(ROUTE_TYPES),
        "states": list(STATES),
        "selected_names": list(names),
        "name_policy": (
            "The shared one-token pool is exhausted. Frozen names are rotated "
            "into pairings not used in Phase1103, with disjoint split worlds."
        ),
        "capture_roles": list(CAPTURE_ROLES),
        "causal_design": {
            "depth_fractions": list(CAUSAL_DEPTH_FRACTIONS),
            "discovery_items": list(CAUSAL_DISCOVERY_ITEMS),
            "confirmation_items": list(CAUSAL_CONFIRMATION_ITEMS),
            "relation_orders": list(CAUSAL_RELATION_ORDERS),
            "maximum_pairs_per_model": MAX_CAUSAL_PAIRS_PER_MODEL,
            "pair_selection": (
                "Among pairs passing both splits, rank only by the minimum "
                "qualification exact-conflict cell accuracy; break ties by "
                "the frozen candidate order. Confirmation never selects."
            ),
            "patch_role": "query_end",
            "patch_alpha": PATCH_ALPHA,
            "primary_deltas": [
                "raw selector difference",
                "active-minus-congruent selector interaction",
            ],
            "primary_transport": "cross label-regime, both target directions",
            "controls": [
                "congruent selector difference",
                "ordinal difference",
                "frozen cyclic wrong-pair difference",
                "equal-norm deterministic Rademacher",
                "congruent collateral",
            ],
        },
        "behavior_authorization": {
            "model_specific_scan": "one model-pair passing both splits",
            "cross_model_upgrade": "same pair passing in at least two models",
            "generation_is_gate": False,
            "paraphrase_is_gate": False,
        },
        "claim_scope": (
            "Lexical key routing and its content-conditioned execution within "
            "the tested model, pair, surface, and interface."
        ),
        "explicit_nonclaims": [
            "No Phase1103 frozen decision is changed.",
            "No paraphrase-semantic, family-wide, cross-model, compression, optimality, or full-language claim follows from a one-model pass.",
            "Raw within-regime replacement alone is an instrument check, not a mechanism closure.",
        ],
        "thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_phase1103_protocol_digest": source_prereg["protocol_digest"],
        "source_phase1103_result_audit_digest": source_audit["audit_digest"],
        "source_phase1103_failure_diagnostic_digest": source_diagnostic["diagnostic_digest"],
        "model_case_digests": model_case_digests,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    combined_checks = {
        "source_result_audit_passed": source_audit["all_checks_passed"],
        "candidate_list_matches_frozen_phase1103_exact_subgate": frozen_exact == CANDIDATE_PAIRS,
        "all_model_protocol_audits_passed": all(
            row["all_checks_passed"] for row in model_audits.values()
        ),
        "models_exact": set(model_audits) == set(MODELS),
    }
    combined = {
        "schema_version": "phase1104_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": combined_checks,
        "model_audit_digests": {
            model: digest(row) for model, row in model_audits.items()
        },
        "all_checks_passed": all(combined_checks.values()),
    }
    combined["audit_digest"] = digest(combined)
    write_json(OUT_ROOT / "protocol" / "audit.json", combined)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "audit_digest": combined["audit_digest"],
        "all_checks_passed": combined["all_checks_passed"],
    }), flush=True)


if __name__ == "__main__":
    main()
