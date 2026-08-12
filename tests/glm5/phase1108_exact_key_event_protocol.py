#!/usr/bin/env python3
"""Freeze Phase1108 exact-key routing event-map protocol."""

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
import phase1104_lexical_address_execution_protocol as source


PHASE = 1108
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
SURFACES = ("formal", "plain")
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
CAPTURE_ROLES = (
    "fact1_end",
    "facts_end",
    "selector_start",
    "selector_end",
    "query_end",
    "answer_boundary",
)
COMPONENTS = ("residual", "attention_output", "mlp_output")
DEPTH_FRACTIONS = tuple(value / 20.0 for value in range(21))
SIGNED_PROJECTION_DIM = 96
SIGNED_PROJECTION_REPLICATES = 2
SIGNED_PROJECTION_SEED = 11080031
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1108_exact_key_event_map"
SOURCE_ROOT = source.OUT_ROOT
SOURCE_PREREG = SOURCE_ROOT / "protocol" / "preregistration.json"
SOURCE_FINAL = SOURCE_ROOT / "analysis" / "final_summary.json"
SOURCE_AUDIT = SOURCE_ROOT / "audit" / "result_audit.json"


write_json = tools.write_json
write_jsonl = tools.write_jsonl
read_json = tools.read_json
read_jsonl = tools.read_jsonl
digest = tools.digest


# Frozen before any Phase1108 model run: the four pairs that passed Phase1104
# behavior in both Qwen3 and GLM4. No Phase1104 model-specific top-three list
# is used, which repairs the earlier cross-model mismatch.
RELATION_PAIRS = (
    "responsibility__leadership_rank",
    "causal_influence__evidence_strength",
    "likelihood__certainty",
    "explanatory_power__dependency_strength",
)
PAIR_RELATIONS = phase1101.PAIR_RELATIONS
PAIR_FAMILY = phase1101.PAIR_FAMILY


LABEL_SUFFIXES = {
    "formal": ("axis", "measure", "channel", "register"),
    "plain": ("index", "scale", "marker", "track"),
}
NEUTRAL_LABELS = {
    "formal": (
        ("amber docket", "cobalt docket"),
        ("maple ledger", "cedar ledger"),
        ("silver archive", "golden archive"),
        ("river register", "harbor register"),
    ),
    "plain": (
        ("north card", "south card"),
        ("open note", "closed note"),
        ("morning file", "evening file"),
        ("stone tag", "glass tag"),
    ),
}
SHELLS = {
    "formal": (
        "A registry contains two independent keyed rankings. {fact1}. {fact2}. {question}? Return exactly one person name.",
        "Two keyed comparisons are stored in a ledger. {fact1}. Separately, {fact2}. {question}? Give one person name only.",
        "An archive holds two independent keyed rankings. {fact1}. In another record, {fact2}. {question}? Respond with exactly one person name.",
        "A decision register contains two keyed comparisons. {fact1}. The remaining record states, {fact2}. {question}? Write only the selected person's name.",
    ),
    "plain": (
        "Here are two separately keyed comparisons. {fact1}. {fact2}. {question}? Answer with one person name.",
        "Two different keys organize these comparisons. {fact1}. Also, {fact2}. {question}? Reply with one name only.",
        "The notes give two independent keyed results. {fact1}. A separate note says, {fact2}. {question}? Provide exactly one person name.",
        "Two keyed results appear below. {fact1}. The other result is, {fact2}. {question}? State only the chosen person's name.",
    ),
}
ORDINAL_SELECTORS = (
    "the first displayed entry",
    "the second displayed entry",
)
ROUTE_CODES = {"exact": "e", "ordinal": "o"}
REGIME_CODES = {"relation_label": "r", "neutral_label": "n"}


THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_conflict_cell_accuracy": 0.80,
    "minimum_regime_accuracy": 0.90,
    "minimum_congruent_accuracy": 0.90,
    "minimum_models_per_pair": 2,
    "minimum_hidden_finite_fraction": 0.97,
    "pre_query_tolerance": 1e-8,
    "minimum_projection_finite_fraction": 0.99,
    "minimum_projection_replicate_cosine": 0.90,
    "minimum_confirmation_cross_regime_cosine": 0.25,
    "minimum_confirmation_control_advantage": 0.05,
    "minimum_pair_retrieval_count": 3,
    "minimum_pair_retrieval_margin": 0.05,
    "minimum_cross_model_curve_cosine": 0.75,
    "maximum_cross_model_curve_mae": 0.18,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": "Source, freeze, token, prefix, factorial, split, key-assignment, and answer-balance audits all pass.",
    "P2": "At least two models independently pass both splits for at least one of the four frozen exact-key relation pairs.",
    "P3": "All authorized signed scans pass FP16, no-quantization, finiteness, deterministic-identity, projection, and pre-query-zero audits.",
    "P4": "A qualification-selected exact-key lexical-address event repeats on confirmation with cross-regime cosine at least 0.25 and at least 0.05 advantage over ordinal and selector controls in two models.",
    "P5": "At the frozen event, centered pair-differential directions retrieve at least three of four pair identities with mean margin at least 0.05 in two models.",
    "P6": "The normalized event-response curve repeats across Qwen3 and GLM4 with cosine at least 0.75 and mean absolute error at most 0.18.",
    "P7": "Only P2-P6 can authorize a separately preregistered causal staircase. A descriptive event-map pass is not itself causal closure.",
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
        LABEL_REGIMES,
        ROUTE_TYPES,
        CONGRUENCES,
        TARGET_RELATIONS,
        RELATION_ORDERS,
        ORIENTATIONS,
    )
)


def state_factors(state: str) -> tuple[str, str, str, int, int, int]:
    for factors in itertools.product(
        LABEL_REGIMES,
        ROUTE_TYPES,
        CONGRUENCES,
        TARGET_RELATIONS,
        RELATION_ORDERS,
        ORIENTATIONS,
    ):
        if state == state_name(*factors):
            return factors
    raise ValueError(state)


def selected_names() -> tuple[str, ...]:
    names = tuple(read_json(SOURCE_PREREG)["selected_names"])
    if len(names) != 24:
        raise RuntimeError(f"expected 24 frozen one-token names, found {len(names)}")
    return names


# Pairings are disjoint from Phase1104 and are frozen across the two split
# worlds. The names themselves are reused because the cross-model one-token
# pool was already exhausted; no novelty claim is made for lexical identity.
PAIRING_INDEX = {
    0: ((0, 7), (1, 9), (3, 11)),
    1: ((2, 8), (4, 10), (5, 6)),
}


def name_pair(names: tuple[str, ...], template: int, item: int) -> tuple[str, str]:
    split = split_for_template(template)
    pool = names[:12] if split == "qualification" else names[12:]
    local_template = TEMPLATES_BY_SPLIT[split].index(template)
    left, right = PAIRING_INDEX[local_template][item]
    return pool[left], pool[right]


def relation_label(relation: str, surface: str, template: int) -> str:
    return f"{relation.replace('_', ' ')} {LABEL_SUFFIXES[surface][template]}"


def labels_for(
    pair: str, surface: str, template: int, item: int, regime: str,
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
    surface: str, label: str, relation: str, winner: str, loser: str,
) -> str:
    payload = relation.replace("_", " ")
    if surface == "formal":
        return (
            f"Under the key {label}, the {payload} comparison ranks "
            f"[ {winner} ] ahead of [ {loser} ]"
        )
    return (
        f"For {label}, the {payload} result puts "
        f"[ {winner} ] before [ {loser} ]"
    )


def mark(text: str, value: str, start: int = 0) -> tuple[int, int, str]:
    position = text.find(value, start)
    if position < 0:
        raise RuntimeError(f"missing marked value {value!r}")
    return position, position + len(value), value


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
        relation0: fact(surface, labels[relation0], relation0, winner0, loser0),
        relation1: fact(surface, labels[relation1], relation1, winner1, loser1),
    }
    displayed = (relation0, relation1) if order == 0 else (relation1, relation0)
    fact1, fact2 = facts[displayed[0]], facts[displayed[1]]
    target_relation = (relation0, relation1)[target]
    selector = (
        labels[target_relation]
        if route == "exact"
        else ORDINAL_SELECTORS[displayed.index(target_relation)]
    )
    if surface == "formal":
        question = f"According to {selector}, which person ranks ahead"
    else:
        question = f"Using {selector}, who comes first"
    raw_prompt = SHELLS[surface][template].format(
        fact1=fact1, fact2=fact2, question=question
    )
    fact1_span = mark(raw_prompt, fact1)
    fact2_span = mark(raw_prompt, fact2, fact1_span[1])
    question_span = mark(raw_prompt, question, fact2_span[1])
    selector_span = mark(raw_prompt, selector, question_span[0])
    expected = winner0 if target == 0 else winner1
    return raw_prompt, {
        "fact1_end": fact1_span,
        "facts_end": fact2_span,
        "selector": selector_span,
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
        pair,
        surface,
        template,
        item,
        regime,
        route,
        congruence,
        target,
        order,
        orientation,
        name_pair(names, template, item),
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
    role_positions = {
        "fact1_end": int(role_spans["fact1_end"][1]),
        "facts_end": int(role_spans["facts_end"][1]),
        "selector_start": int(role_spans["selector"][0]),
        "selector_end": int(role_spans["selector"][1]),
        "query_end": int(role_spans["query_end"][1]),
        "answer_boundary": len(input_ids) - 1,
    }
    stored_spans = {
        "fact1_end": list(role_spans["fact1_end"]),
        "facts_end": list(role_spans["facts_end"]),
        "selector_start": [role_positions["selector_start"], role_positions["selector_start"]],
        "selector_end": list(role_spans["selector"]),
        "query_end": list(role_spans["query_end"]),
        "answer_boundary": [len(input_ids) - 1, len(input_ids) - 1],
    }
    expected_class = "e0" if meta["expected"] == meta["entity0"] else "e1"
    candidate_labels = {"e0": meta["entity0"], "e1": meta["entity1"]}
    candidate_token_ids = {
        key: phase1101.base.continuation_ids(tokenizer, rendered, label)
        for key, label in candidate_labels.items()
    }
    unit_id = f"phase1108.{model_name}.{pair}.{surface}.t{template}.i{item:02d}"
    return {
        "schema_version": "phase1108_exact_key_event_case.v1",
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
        "role_spans": stored_spans,
        "role_positions": role_positions,
        "fact1_text": meta["fact1"],
        "fact2_text": meta["fact2"],
        "question_text": meta["question"],
        "selector_text": meta["selector"],
        "relation_labels": meta["labels"],
        "neutral_key_swap": meta["neutral_key_swap"],
        "displayed_relations": list(meta["displayed_relations"]),
        "continuation_prefix": CONTINUATION_PREFIX,
        "prompt_digest": hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
    }


def build_model_cases(tokenizer, model_name: str, names: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for pair in RELATION_PAIRS:
        for surface in SURFACES:
            for template in TEMPLATES:
                for item in range(ITEMS_PER_TEMPLATE):
                    for state in STATES:
                        rows.append(build_case(
                            tokenizer,
                            model_name,
                            names,
                            pair,
                            surface,
                            template,
                            item,
                            state,
                            len(rows),
                        ))
    return rows


def source_name_pairs() -> set[frozenset[str]]:
    names = tuple(read_json(SOURCE_PREREG)["selected_names"])
    return {
        frozenset(source.name_pair(names, template, item))
        for template in source.TEMPLATES
        for item in range(source.ITEMS_PER_TEMPLATE)
    }


def audit_model(
    model_name: str, rows: list[dict[str, Any]], names: tuple[str, ...],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    expected = (
        len(RELATION_PAIRS)
        * len(SURFACES)
        * len(TEMPLATES)
        * ITEMS_PER_TEMPLATE
        * len(STATES)
    )
    checks["case_count"] = len(rows) == expected
    checks["state_count"] = len(STATES) == 64
    checks["record_ids_unique"] = len({row["record_id"] for row in rows}) == len(rows)
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
        and row["role_positions"]["fact1_end"]
        < row["role_positions"]["facts_end"]
        < row["role_positions"]["selector_start"]
        <= row["role_positions"]["selector_end"]
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
    checks["name_pairings_fresh_vs_phase1104"] = not (
        current_pairs & source_name_pairs()
    )
    checks["relation_label_splits_disjoint"] = all(
        not (
            {relation_label(relation, surface, template) for template in (0, 1)}
            & {relation_label(relation, surface, template) for template in (2, 3)}
        )
        for pair in RELATION_PAIRS
        for relation in PAIR_RELATIONS[pair]
        for surface in SURFACES
    )
    checks["neutral_label_splits_disjoint"] = all(
        not (
            set(sum((NEUTRAL_LABELS[surface][template] for template in (0, 1)), ()))
            & set(sum((NEUTRAL_LABELS[surface][template] for template in (2, 3)), ()))
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
            and row["relation_pair"] == pair
            and row["surface"] == surface
        )[0]
        == Counter(
            row["neutral_key_swap"] for row in rows
            if row["label_regime"] == "neutral_label"
            and row["relation_pair"] == pair
            and row["surface"] == surface
        )[1]
        for pair in RELATION_PAIRS for surface in SURFACES
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["unit_id"]].append(row)
    checks["complete_factorial"] = all(
        {row["state"] for row in unit} == set(STATES)
        for unit in grouped.values()
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
        "schema_version": "phase1108_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(rows),
        "unit_count": len(grouped),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(rows),
    }


def main() -> None:
    source_final = read_json(SOURCE_FINAL)
    source_audit = read_json(SOURCE_AUDIT)
    source_prereg = read_json(SOURCE_PREREG)
    if not source_audit["all_checks_passed"]:
        raise RuntimeError("Phase1104 result audit did not pass")
    frozen_pairs = tuple(source_final["behavior"]["cross_model_behavior_pairs"])
    if frozen_pairs != RELATION_PAIRS:
        raise RuntimeError(f"Phase1104 common-pair drift: {frozen_pairs!r}")
    names = selected_names()
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    model_case_digests = {}
    model_audits = {}
    for model, tokenizer in tokenizers.items():
        rows = build_model_cases(tokenizer, model, names)
        audit = audit_model(model, rows, names)
        if not audit["all_checks_passed"]:
            failed = [key for key, value in audit["checks"].items() if not value]
            raise RuntimeError(f"Phase1108 {model} protocol audit failed: {failed}")
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
        "schema_version": "phase1108_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "relation_pairs": list(RELATION_PAIRS),
        "pair_freeze_rule": "Exactly the four Phase1104 Qwen3-GLM4 common behavior pairs; no model-specific top-three selection.",
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
        "name_policy": "Reuse the exhausted shared one-token pool with split-disjoint worlds and pairings unused in Phase1104.",
        "capture_roles": list(CAPTURE_ROLES),
        "components": list(COMPONENTS),
        "depth_fractions": list(DEPTH_FRACTIONS),
        "signed_projection": {
            "dimension": SIGNED_PROJECTION_DIM,
            "replicates": SIGNED_PROJECTION_REPLICATES,
            "seed": SIGNED_PROJECTION_SEED,
            "type": "equal-norm deterministic Rademacher",
        },
        "primary_objects": {
            "routing": "one half of conflict selector difference minus congruent selector difference",
            "selector": "one half of conflict selector difference plus congruent selector difference",
            "lexical_address": "exact routing contrast minus ordinal routing contrast within a key regime",
            "reuse": "shared signed event direction across pairs, surfaces, and key regimes",
            "differential": "pair-centered signed direction and split-to-split pair retrieval",
        },
        "selection_rule": "Qualification selects one event coordinate globally from authorized models and pairs; confirmation alone judges P4-P6.",
        "automatic_next_rule": "Only simultaneous P2-P6 authorization starts a separately frozen Phase1109 causal staircase; otherwise stop without component selection.",
        "claim_scope": "Exact structural-key routing event topology in the tested local models and relation pairs.",
        "explicit_nonclaims": [
            "No semantic-paraphrase transfer is claimed.",
            "A signed event map is descriptive, not causal closure.",
            "Failure at this interface does not prove that distributed transport is absent.",
            "No compression optimality, brain equivalence, or full-language mechanism is tested.",
        ],
        "thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_phase1104_protocol_digest": source_prereg["protocol_digest"],
        "source_phase1104_result_audit_digest": source_audit["audit_digest"],
        "source_phase1104_final_summary_digest": source_final["final_summary_digest"],
        "model_case_digests": model_case_digests,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    checks = {
        "source_result_audit_passed": source_audit["all_checks_passed"],
        "pair_freeze_matches_phase1104": frozen_pairs == RELATION_PAIRS,
        "all_model_protocol_audits_passed": all(
            row["all_checks_passed"] for row in model_audits.values()
        ),
        "models_exact": set(model_audits) == set(MODELS),
    }
    combined = {
        "schema_version": "phase1108_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "model_audit_digests": {
            model: digest(row) for model, row in model_audits.items()
        },
        "all_checks_passed": all(checks.values()),
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
