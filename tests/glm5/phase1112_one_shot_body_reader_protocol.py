#!/usr/bin/env python3
"""Freeze the Phase1112 one-shot exact-key body-reader protocol."""

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
import phase1101_relation_identity_routing_protocol as relation_source
import phase1104_lexical_address_execution_protocol as phase1104
import phase1108_exact_key_event_protocol as phase1108


PHASE = 1112
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
SURFACES = ("formal", "plain")
TEMPLATES = (0, 1, 2, 3)
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
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
SOURCE_NAMES = ("key0", "body0", "key1", "body1", "outside")
QUERY_ROLE = "answer_boundary"
ASSISTANT_PREFILL = "Answer:"
CONTINUATION_PREFIX = " "
MAX_SELECTED_EVENTS = 4
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1112_one_shot_body_reader"
SOURCE_BEHAVIOR = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1102_relation_identity_routing_replication"
    / "analysis"
    / "behavior_authorization.json"
)


write_json = tools.write_json
write_jsonl = tools.write_jsonl
read_json = tools.read_json
read_jsonl = tools.read_jsonl
digest = tools.digest


# All four pairs passed Phase1102 in Qwen3, GLM4, and DS7B, and none appeared
# in the Phase1108 exact-key event map.
RELATION_PAIRS = (
    "arrival_time__departure_time",
    "registration_time__publication_time",
    "distance__depth",
    "influence__popularity",
)
PAIR_RELATIONS = relation_source.PAIR_RELATIONS
PAIR_FAMILY = relation_source.PAIR_FAMILY


LABEL_SUFFIXES = {
    "formal": ("folio", "catalog", "index", "register"),
    "plain": ("tab", "card", "note", "file"),
}
NEUTRAL_LABELS = {
    "formal": (
        ("ivory folio", "indigo folio"),
        ("copper catalog", "jade catalog"),
        ("lunar index", "solar index"),
        ("pine register", "birch register"),
    ),
    "plain": (
        ("red tab", "blue tab"),
        ("quiet card", "loud card"),
        ("round note", "square note"),
        ("cloud file", "field file"),
    ),
}
SHELLS = {
    "formal": (
        "A reference desk stores two keyed comparison records. {fact1}. {fact2}. {question}? Return exactly one person name.",
        "Two independent keyed entries appear in a catalog. {fact1}. The other entry states, {fact2}. {question}? Give one person name only.",
        "A lookup register contains two separate comparisons. {fact1}. In a distinct entry, {fact2}. {question}? Respond with exactly one person name.",
        "Two keyed records are listed for a decision. {fact1}. Separately, {fact2}. {question}? Write only the selected person's name.",
    ),
    "plain": (
        "Two labels organize the following results. {fact1}. {fact2}. {question}? Answer with one person name.",
        "Here are two separately labeled comparisons. {fact1}. A different label gives, {fact2}. {question}? Reply with one name only.",
        "The list has two independent labeled results. {fact1}. The second result is, {fact2}. {question}? Provide exactly one person name.",
        "Two labels point to different comparisons. {fact1}. Also, {fact2}. {question}? State only the chosen person's name.",
    ),
}
ORDINAL_SELECTORS = (
    "the first displayed entry",
    "the second displayed entry",
)
ROUTE_CODES = {"exact": "e", "ordinal": "o"}
REGIME_CODES = {"relation_label": "r", "neutral_label": "n"}


# These pairings are disjoint from the Phase1104 and Phase1108 pairing sets.
# The one-token name identities are necessarily reused because the shared pool
# is exhausted; no lexical-novelty claim is made.
PAIRING_INDEX = {
    0: ((0, 2), (1, 3), (4, 6)),
    1: ((5, 7), (8, 10), (9, 11)),
}


THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_conflict_cell_accuracy": 0.80,
    "minimum_regime_accuracy": 0.90,
    "minimum_congruent_accuracy": 0.90,
    "minimum_models_per_pair": 2,
    "minimum_cross_model_pairs": 3,
    "minimum_attention_finite_fraction": 0.999,
    "maximum_attention_partition_error": 0.005,
    "minimum_relative_depth": 0.30,
    "maximum_relative_depth": 0.85,
    "minimum_target_body_attention_mass": 0.03,
    "minimum_exact_body_following": 0.08,
    "minimum_body_over_key_attention": 0.03,
    "minimum_exact_over_ordinal_attention": 0.03,
    "minimum_positive_relation_pairs": 3,
    "maximum_selected_events": MAX_SELECTED_EVENTS,
    "minimum_value_finite_fraction": 0.999,
    "maximum_head_reconstruction_relative_error": 0.005,
    "maximum_key_value_matched_distance": 1e-5,
    "minimum_body_value_matched_distance": 0.02,
    "minimum_selected_body_av_distance_advantage": 0.03,
    "minimum_exact_over_ordinal_av_advantage": 0.02,
    "minimum_body_over_key_av_advantage": 0.02,
    "minimum_body_over_carrier_av_advantage": 0.02,
    "minimum_models": 2,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "The new-pair source, unused name combinations, new shells and keys, "
        "token spans, source partitions, factorial balance, and protocol digests pass."
    ),
    "P2": (
        "At least two FP16/no-quantization models pass both behavior splits on "
        "at least three of four relation pairs."
    ),
    "P3": (
        "All behavior-authorized all-head answer-boundary scans return finite "
        "attention masses whose disjoint source partition sums to one."
    ),
    "P4": (
        "Discovery selects at most four mid/late answer-boundary heads per model "
        "that follow target body over distractor and key, with exact-over-ordinal specificity."
    ),
    "P5": (
        "Frozen discovery heads repeat all body-attention gates on independent "
        "confirmation names, templates, and relation-key assignments in two models."
    ),
    "P6": (
        "At a frozen confirmed head, changed-body A-times-V is larger when selected "
        "than when distractor, exceeds ordinal, key, and unchanged-body carrier controls, "
        "and repeats across at least three relation pairs in two models."
    ),
    "P7": (
        "Only P1-P6 together establish a descriptive one-shot second-hop candidate. "
        "The exact-key registry is then closed to further hotspot search regardless of outcome."
    ),
    "P8": (
        "No causal intervention is automatic. A P7 pass only authorizes a separately "
        "preregistered intervention on independent material."
    ),
}


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def split_for_template(template: int) -> str:
    for split, templates in TEMPLATES_BY_SPLIT.items():
        if template in templates:
            return split
    raise ValueError(template)


def selected_names() -> tuple[str, ...]:
    names = tuple(phase1108.selected_names())
    if len(names) != 24:
        raise RuntimeError(f"expected 24 frozen one-token names, found {len(names)}")
    return names


def name_pair(names: tuple[str, ...], template: int, item: int) -> tuple[str, str]:
    split = split_for_template(template)
    pool = names[:12] if split == "discovery" else names[12:]
    local_template = TEMPLATES_BY_SPLIT[split].index(template)
    left, right = PAIRING_INDEX[local_template][item]
    return pool[left], pool[right]


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


def fact(surface: str, label: str, relation: str, winner: str, loser: str) -> str:
    payload = relation.replace("_", " ")
    if surface == "formal":
        return (
            f"Under the label {label}, the {payload} comparison places "
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
    winner1, loser1 = (
        (loser0, winner0) if congruence == "conflict" else (winner0, loser0)
    )
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
    question = (
        f"According to {selector}, which person is ahead"
        if surface == "formal"
        else f"Using {selector}, who comes first"
    )
    raw_prompt = SHELLS[surface][template].format(
        fact1=fact1, fact2=fact2, question=question
    )
    fact1_span = mark(raw_prompt, fact1)
    fact2_span = mark(raw_prompt, fact2, fact1_span[1])
    question_span = mark(raw_prompt, question, fact2_span[1])
    selector_span = mark(raw_prompt, selector, question_span[0])
    raw_spans = {
        "fact1": fact1_span,
        "fact2": fact2_span,
        "selector": selector_span,
        "question": question_span,
    }
    key_char_spans = {}
    for relation_index, relation in enumerate((relation0, relation1)):
        display_index = displayed.index(relation)
        fact_text = (fact1, fact2)[display_index]
        fact_span = (fact1_span, fact2_span)[display_index]
        local = fact_text.find(labels[relation])
        if local < 0:
            raise RuntimeError("key label missing from fact")
        start = fact_span[0] + local
        key_char_spans[f"key{relation_index}"] = (
            start,
            start + len(labels[relation]),
            labels[relation],
        )
    expected = winner0 if target == 0 else winner1
    return raw_prompt, raw_spans, {
        "entity0": entity0,
        "entity1": entity1,
        "winner0": winner0,
        "winner1": winner1,
        "expected": expected,
        "relation0": relation0,
        "relation1": relation1,
        "displayed": displayed,
        "fact1": fact1,
        "fact2": fact2,
        "selector": selector,
        "labels": labels,
        "key_char_spans": key_char_spans,
    }


def inclusive(span: tuple[int, int] | list[int]) -> list[int]:
    start, end = (int(value) for value in span)
    return list(range(start, end + 1))


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
        relation_source.base.behavior_tools.render_native(
            tokenizer, model_name, raw_prompt, with_system=False
        )
        + ASSISTANT_PREFILL
    )
    input_ids = [
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    token_spans = offset_token_spans(tokenizer, rendered, raw_prompt, raw_spans)
    key_spans = offset_token_spans(
        tokenizer, rendered, raw_prompt, meta["key_char_spans"]
    )
    displayed = list(meta["displayed"])
    relation0, relation1 = PAIR_RELATIONS[pair]
    fact_spans = (token_spans["fact1"], token_spans["fact2"])
    record_spans = {
        "record0": fact_spans[displayed.index(relation0)],
        "record1": fact_spans[displayed.index(relation1)],
    }
    records = {index: set(inclusive(record_spans[f"record{index}"])) for index in (0, 1)}
    keys = {index: set(inclusive(key_spans[f"key{index}"])) for index in (0, 1)}
    bodies = {index: records[index] - keys[index] for index in (0, 1)}
    occupied = records[0] | records[1]
    source_positions = {
        "key0": sorted(keys[0]),
        "body0": sorted(bodies[0]),
        "key1": sorted(keys[1]),
        "body1": sorted(bodies[1]),
        "outside": sorted(set(range(len(input_ids))) - occupied),
    }
    candidate_labels = {"e0": meta["entity0"], "e1": meta["entity1"]}
    candidate_token_ids = {
        key: relation_source.base.continuation_ids(tokenizer, rendered, label)
        for key, label in candidate_labels.items()
    }
    expected_class = "e0" if meta["expected"] == meta["entity0"] else "e1"
    unit_id = f"phase1112.{model_name}.{pair}.{surface}.t{template}.i{item:02d}"
    return {
        "schema_version": "phase1112_body_reader_case.v1",
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
        "expected_class": expected_class,
        "candidate_labels": candidate_labels,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {
            key: [int(values[0])] for key, values in candidate_token_ids.items()
        },
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "query_position": len(input_ids) - 1,
        "record_spans": {key: list(value) for key, value in record_spans.items()},
        "key_spans": {key: list(value) for key, value in key_spans.items()},
        "source_positions": source_positions,
        "selector_text": meta["selector"],
        "relation_labels": meta["labels"],
        "displayed_relations": displayed,
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


def pairing_set(module, names: tuple[str, ...]) -> set[frozenset[str]]:
    return {
        frozenset(module.name_pair(names, template, item))
        for template in module.TEMPLATES
        for item in range(module.ITEMS_PER_TEMPLATE)
    }


def audit_model(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[row["unit_id"]].append(row)
    factor_counts = {
        field: Counter(row[field] for row in rows)
        for field in (
            "label_regime",
            "route_type",
            "congruence",
            "target_relation",
            "relation_order",
            "orientation",
        )
    }
    partition_ok = []
    query_after = []
    key_before_payload = []
    for row in rows:
        groups = [set(row["source_positions"][name]) for name in SOURCE_NAMES]
        partition_ok.append(
            all(groups[i].isdisjoint(groups[j]) for i in range(5) for j in range(i + 1, 5))
            and set().union(*groups) == set(range(len(row["input_ids"])))
            and all(groups)
        )
        query_after.append(
            int(row["query_position"])
            > max(row["source_positions"]["body0"] + row["source_positions"]["body1"])
        )
        candidate_ids = {
            int(values[0]) for values in row["candidate_first_token_ids"].values()
        }
        for index in (0, 1):
            entities = [
                position
                for position in row["source_positions"][f"body{index}"]
                if int(row["input_ids"][position]) in candidate_ids
            ]
            key_before_payload.append(
                bool(entities)
                and max(row["source_positions"][f"key{index}"]) < min(entities)
            )
    expected_cases = (
        len(RELATION_PAIRS)
        * len(SURFACES)
        * len(TEMPLATES)
        * ITEMS_PER_TEMPLATE
        * len(STATES)
    )
    checks = {
        "case_count": len(rows) == expected_cases == 6144,
        "unit_count": len(by_unit) == 96,
        "units_have_exact_state_cube": all(
            {row["state"] for row in unit_rows} == set(STATES)
            for unit_rows in by_unit.values()
        ),
        "record_ids_unique": len({row["record_id"] for row in rows}) == len(rows),
        "source_partition_disjoint_exhaustive": all(partition_ok),
        "query_after_bodies": all(query_after),
        "keys_precede_entity_payload": all(key_before_payload),
        "candidate_continuations_nonempty": all(
            all(values for values in row["candidate_token_ids"].values()) for row in rows
        ),
        "factor_balance": all(
            len(set(counts.values())) == 1 for counts in factor_counts.values()
        ),
        "split_name_pools_disjoint": not (
            {row["entity0"] for row in rows if row["split"] == "discovery"}
            | {row["entity1"] for row in rows if row["split"] == "discovery"}
        ) & (
            {row["entity0"] for row in rows if row["split"] == "confirmation"}
            | {row["entity1"] for row in rows if row["split"] == "confirmation"}
        ),
    }
    return {
        "model": model,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(rows),
        "case_count": len(rows),
        "unit_count": len(by_unit),
    }


def main() -> None:
    names = selected_names()
    source_behavior = read_json(SOURCE_BEHAVIOR)
    source_pass = {
        model: {
            pair
            for pair, row in source_behavior["models"][model]["pair_results"].items()
            if row["passed"]
        }
        for model in MODELS
    }
    old_pairs_1104 = pairing_set(phase1104, names)
    old_pairs_1108 = pairing_set(phase1108, names)
    new_pairs = {
        frozenset(name_pair(names, template, item))
        for template in TEMPLATES
        for item in range(ITEMS_PER_TEMPLATE)
    }
    global_checks = {
        "state_count": len(STATES) == 64,
        "relation_pairs_new_vs_phase1108": not (set(RELATION_PAIRS) & set(phase1108.RELATION_PAIRS)),
        "relation_pairs_passed_phase1102_all_models": all(
            pair in source_pass[model] for model in MODELS for pair in RELATION_PAIRS
        ),
        "name_pairings_new_vs_phase1104": not (new_pairs & old_pairs_1104),
        "name_pairings_new_vs_phase1108": not (new_pairs & old_pairs_1108),
        "discovery_confirmation_templates_disjoint": not (
            set(TEMPLATES_BY_SPLIT["discovery"])
            & set(TEMPLATES_BY_SPLIT["confirmation"])
        ),
        "neutral_labels_split_disjoint": all(
            not (
                set(sum((NEUTRAL_LABELS[surface][index] for index in (0, 1)), ()))
                & set(sum((NEUTRAL_LABELS[surface][index] for index in (2, 3)), ()))
            )
            for surface in SURFACES
        ),
    }
    if not all(global_checks.values()):
        raise RuntimeError(f"global protocol checks failed: {global_checks}")

    protocol_root = OUT_ROOT / "protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)
    model_audits = {}
    case_digests = {}
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        rows = build_model_cases(tokenizer, model, names)
        audit = audit_model(rows, model)
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"{model} protocol checks failed: {audit['checks']}")
        write_jsonl(protocol_root / f"cases.{model}.jsonl", rows)
        model_audits[model] = audit
        case_digests[model] = audit["case_digest"]

    prereg = {
        "schema_version": "phase1112_body_reader_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "relation_pairs": list(RELATION_PAIRS),
        "pair_relations": {pair: list(PAIR_RELATIONS[pair]) for pair in RELATION_PAIRS},
        "surfaces": list(SURFACES),
        "templates_by_split": {key: list(value) for key, value in TEMPLATES_BY_SPLIT.items()},
        "items_per_template": ITEMS_PER_TEMPLATE,
        "states": list(STATES),
        "source_names": list(SOURCE_NAMES),
        "query_role": QUERY_ROLE,
        "name_policy": (
            "Reuse the exhausted 24-name one-token pool with split-disjoint worlds "
            "and pairings unused by Phase1104 and Phase1108."
        ),
        "one_shot_rule": (
            "This is the final hotspot search in the exact-key registry. Positive or "
            "negative results close the registry to further head reselection."
        ),
        "thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "case_digests": case_digests,
        "source": {
            "phase1102_behavior_authorization_digest": source_behavior["authorization_digest"],
            "phase1102_behavior_file_sha256": file_sha256(SOURCE_BEHAVIOR),
            "phase1108_protocol_digest": read_json(phase1108.OUT_ROOT / "protocol" / "preregistration.json")["protocol_digest"],
        },
        "interpretive_limits": [
            "The task closes an artificial exact-key retrieval primitive, not natural semantic addressing.",
            "Ordinal is a specificity control for the exact-key second-hop hypothesis, not a universal requirement for content readers.",
            "Attention following, A-times-V transport, output use, and causal necessity remain separate claims.",
            "Discovery may select at most four heads per model; thresholds and candidate count cannot be relaxed after model runs.",
            "A descriptive pass only authorizes an independently preregistered causal study.",
        ],
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    audit = {
        "schema_version": "phase1112_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "global_checks": global_checks,
        "model_audits": model_audits,
        "all_checks_passed": all(global_checks.values()) and all(
            row["all_checks_passed"] for row in model_audits.values()
        ),
    }
    audit["audit_digest"] = digest(audit)
    write_json(protocol_root / "audit.json", audit)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "case_count_per_model": 6144,
        "unit_count_per_model": 96,
        "relation_pairs": list(RELATION_PAIRS),
        "all_checks_passed": audit["all_checks_passed"],
        "audit_digest": audit["audit_digest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
