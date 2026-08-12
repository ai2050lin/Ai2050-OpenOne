#!/usr/bin/env python3
"""Freeze Phase1105 natural synonym-address behavior calibration."""

from __future__ import annotations

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
import phase1104_lexical_address_execution_protocol as source


PHASE = 1105
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
SURFACES = ("en",)
TEMPLATES = (0, 1, 2, 3)
TEMPLATES_BY_SPLIT = {
    "qualification": (0, 1),
    "confirmation": (2, 3),
}
ITEMS_PER_TEMPLATE = 6
ROUTE_TYPES = ("exact", "close_synonym", "natural_definition", "ordinal")
CONGRUENCES = ("conflict", "congruent")
TARGET_RELATIONS = (0, 1)
RELATION_ORDERS = (0, 1)
ORIENTATIONS = (0, 1)
ASSISTANT_PREFILL = "Answer:"
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1105_natural_synonym_address"
SOURCE_ROOT = source.OUT_ROOT
SOURCE_FINAL = SOURCE_ROOT / "analysis" / "final_summary.json"
SOURCE_AUDIT = SOURCE_ROOT / "audit" / "result_audit.json"

write_json = source.write_json
write_jsonl = source.write_jsonl
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest
sha256_text = source.source.sha256_text


RELATION_PAIRS = (
    "responsibility__leadership_rank",
    "causal_influence__evidence_strength",
    "likelihood__certainty",
    "explanatory_power__dependency_strength",
)
PAIR_RELATIONS = {
    pair: tuple(pair.split("__", 1)) for pair in RELATION_PAIRS
}
PAIR_CONTEXT = {
    "responsibility__leadership_rank": {
        "singular": "team member",
        "plural": "team members",
        "intro": "An incident review made two independent judgments about the same team members.",
    },
    "causal_influence__evidence_strength": {
        "singular": "proposed factor",
        "plural": "proposed factors",
        "intro": "An investigation compared two proposed factors from two independent perspectives.",
    },
    "likelihood__certainty": {
        "singular": "forecast",
        "plural": "forecasts",
        "intro": "A forecasting report evaluated two forecasts in two distinct ways.",
    },
    "explanatory_power__dependency_strength": {
        "singular": "model",
        "plural": "models",
        "intro": "A research review assessed two models on two separate properties.",
    },
}


# Each template has an independently worded fact and two nonidentical queries.
# Templates 0/1 and 2/3 are disjoint lexical splits.
FORMS = {
    "responsibility": (
        ("was held more responsible for the outcome than", "was held more responsible for the outcome", "was more accountable for what happened", "carried more of the duty for the result"),
        ("bore more responsibility for what happened than", "bore more responsibility for what happened", "had greater accountability for the result", "was more answerable for the outcome"),
        ("was assigned a larger share of responsibility for the result than", "was assigned a larger share of responsibility for the result", "carried greater accountability for the outcome", "bore more of the duty for what happened"),
        ("was judged more responsible for the final outcome than", "was judged more responsible for the final outcome", "was more answerable for the result", "held the greater share of duty for what happened"),
    ),
    "leadership_rank": (
        ("held a higher position in the leadership hierarchy than", "held a higher position in the leadership hierarchy", "had the more senior leadership rank", "stood higher in the chain of command"),
        ("occupied a more senior leadership post than", "occupied a more senior leadership post", "ranked higher among the leaders", "stood above the other in the command hierarchy"),
        ("held the higher rank in the leadership structure than", "held the higher rank in the leadership structure", "had the more senior command position", "stood higher in the leadership chain"),
        ("was placed at a more senior level of leadership than", "was placed at a more senior level of leadership", "occupied the higher command rank", "stood above the other in the leadership hierarchy"),
    ),
    "causal_influence": (
        ("had a greater causal effect on the outcome than", "had a greater causal effect on the outcome", "exerted more causal influence over the result", "made the larger causal difference to what happened"),
        ("contributed more strongly to producing the result than", "contributed more strongly to producing the result", "had the larger causal impact on the outcome", "played the greater role in bringing about the result"),
        ("exerted a stronger causal influence on the final outcome than", "exerted a stronger causal influence on the final outcome", "had the greater causal impact on the result", "contributed more to making the outcome occur"),
        ("played a larger causal role in producing the result than", "played a larger causal role in producing the result", "affected the outcome more strongly as a cause", "made more of the causal difference to the result"),
    ),
    "evidence_strength": (
        ("was supported by stronger evidence than", "was supported by stronger evidence", "had the more compelling evidential support", "had the better-supported case"),
        ("had more convincing evidence behind it than", "had more convincing evidence behind it", "was backed by stronger supporting data", "rested on the more persuasive body of evidence"),
        ("received stronger support from the available evidence than", "received stronger support from the available evidence", "had the more convincing evidential basis", "was better supported by the observed data"),
        ("was backed by a more powerful body of evidence than", "was backed by a more powerful body of evidence", "had stronger empirical support", "rested on the more convincing evidence"),
    ),
    "likelihood": (
        ("described an outcome more likely to occur than", "described an outcome more likely to occur", "had the higher probability of happening", "gave the outcome with the greater chance of occurring"),
        ("assigned a greater chance of occurrence than", "assigned a greater chance of occurrence", "judged the outcome more likely to happen", "gave the event the higher probability"),
        ("placed a higher probability on its outcome than", "placed a higher probability on its outcome", "described the event as more likely to occur", "gave its outcome the greater chance of happening"),
        ("rated its predicted outcome as more likely than", "rated its predicted outcome as more likely", "assigned the higher chance of occurrence", "treated its event as having the greater probability"),
    ),
    "certainty": (
        ("was stated with greater certainty than", "was stated with greater certainty", "was expressed with more confidence", "was presented with less doubt"),
        ("was held with a higher level of confidence than", "was held with a higher level of confidence", "carried the greater degree of certainty", "was regarded as the less doubtful forecast"),
        ("was reported with a greater degree of certainty than", "was reported with a greater degree of certainty", "was held with more confidence", "was presented with the smaller amount of doubt"),
        ("carried a higher confidence assessment than", "carried a higher confidence assessment", "was expressed with greater certainty", "was treated as the less uncertain forecast"),
    ),
    "explanatory_power": (
        ("had greater explanatory power than", "had greater explanatory power", "explained the observations better", "accounted for more of what was observed"),
        ("provided a stronger explanation of the findings than", "provided a stronger explanation of the findings", "had the better ability to explain the data", "made more of the observations understandable"),
        ("explained a larger share of the observed results than", "explained a larger share of the observed results", "had greater power to account for the findings", "made sense of more of the observed pattern"),
        ("offered the more powerful explanation of the data than", "offered the more powerful explanation of the data", "accounted for the observations more successfully", "explained more of what the researchers found"),
    ),
    "dependency_strength": (
        ("depended more strongly on the background assumption than", "depended more strongly on the background assumption", "relied more heavily on the assumption", "was more contingent on the assumption being true"),
        ("showed a stronger dependency on the initial condition than", "showed a stronger dependency on the initial condition", "relied more on the starting condition", "was more sensitive to whether the condition held"),
        ("relied more heavily on the supporting assumption than", "relied more heavily on the supporting assumption", "had the stronger dependence on the assumption", "was more contingent on the assumption holding"),
        ("was more dependent on the initial condition than", "was more dependent on the initial condition", "relied more strongly on the starting condition", "was more affected by whether the condition remained true"),
    ),
}

SHELLS = (
    "{intro} {fact1}. Separately, {fact2}. {question}?",
    "{intro} First, {fact1}. The report also concluded: {fact2}. {question}?",
    "{intro} One part of the review found: {fact1}. A separate analysis found: {fact2}. {question}?",
    "{intro} Investigators first recorded: {fact1}. Independently, they recorded: {fact2}. {question}?",
)
ORDINALS = ("first", "second")
PAIRING_INDEX = {
    0: ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)),
    1: ((0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11)),
}
THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_exact_template_accuracy": 0.80,
    "minimum_exact_aggregate_accuracy": 0.90,
    "minimum_semantic_template_accuracy": 0.70,
    "minimum_semantic_aggregate_accuracy": 0.82,
    "minimum_ordinal_template_accuracy": 0.80,
    "minimum_ordinal_aggregate_accuracy": 0.90,
    "minimum_congruent_accuracy": 0.90,
    "minimum_models_for_shared_semantic_pair": 2,
}


def split_for_template(template: int) -> str:
    for split, templates in TEMPLATES_BY_SPLIT.items():
        if template in templates:
            return split
    raise ValueError(template)


def selected_names() -> tuple[str, ...]:
    return source.selected_names()


def name_pair(names: tuple[str, ...], template: int, item: int) -> tuple[str, str]:
    split = split_for_template(template)
    pool = names[:12] if split == "qualification" else names[12:]
    local_template = TEMPLATES_BY_SPLIT[split].index(template)
    left, right = PAIRING_INDEX[local_template][item]
    return pool[left], pool[right]


def state_name(route: str, congruence: str, target: int, order: int, orientation: int) -> str:
    return f"r{ROUTE_TYPES.index(route)}_c{CONGRUENCES.index(congruence)}_q{target}_o{order}_b{orientation}"


STATES = tuple(
    state_name(*factors)
    for factors in itertools.product(
        ROUTE_TYPES, CONGRUENCES, TARGET_RELATIONS, RELATION_ORDERS, ORIENTATIONS
    )
)


def state_factors(state: str) -> tuple[str, str, int, int, int]:
    for factors in itertools.product(
        ROUTE_TYPES, CONGRUENCES, TARGET_RELATIONS, RELATION_ORDERS, ORIENTATIONS
    ):
        if state == state_name(*factors):
            return factors
    raise ValueError(state)


def render_prompt(
    pair: str,
    template: int,
    route: str,
    congruence: str,
    target: int,
    order: int,
    orientation: int,
    names: tuple[str, str],
) -> tuple[str, dict[str, Any]]:
    entity0, entity1 = names
    relation0, relation1 = PAIR_RELATIONS[pair]
    winner0, loser0 = (entity0, entity1) if orientation == 0 else (entity1, entity0)
    winner1, loser1 = (loser0, winner0) if congruence == "conflict" else (winner0, loser0)
    context = PAIR_CONTEXT[pair]
    winners = {relation0: winner0, relation1: winner1}
    losers = {relation0: loser0, relation1: loser1}
    facts = {}
    for relation in (relation0, relation1):
        fact_phrase = FORMS[relation][template][0]
        facts[relation] = (
            f"The {context['singular']} [ {winners[relation]} ] {fact_phrase} "
            f"the {context['singular']} [ {losers[relation]} ]"
        )
    displayed = (relation0, relation1) if order == 0 else (relation1, relation0)
    target_relation = (relation0, relation1)[target]
    if route == "exact":
        query_predicate = FORMS[target_relation][template][1]
    elif route == "close_synonym":
        query_predicate = FORMS[target_relation][template][2]
    elif route == "natural_definition":
        query_predicate = FORMS[target_relation][template][3]
    else:
        ordinal = ORDINALS[displayed.index(target_relation)]
        query_predicate = f"came out ahead in the {ordinal} finding"
    question = f"Which {context['singular']} {query_predicate}"
    raw_prompt = SHELLS[template].format(
        intro=context["intro"],
        fact1=facts[displayed[0]],
        fact2=facts[displayed[1]],
        question=question,
    )
    expected = winners[target_relation]
    return raw_prompt, {
        "entity0": entity0,
        "entity1": entity1,
        "winner0": winner0,
        "winner1": winner1,
        "expected": expected,
        "relation0": relation0,
        "relation1": relation1,
        "target_relation_name": target_relation,
        "displayed_relations": displayed,
        "fact1": facts[displayed[0]],
        "fact2": facts[displayed[1]],
        "target_fact": facts[target_relation],
        "query_predicate": query_predicate,
        "question": question,
    }


def build_case(
    tokenizer,
    model_name: str,
    names: tuple[str, ...],
    pair: str,
    template: int,
    item: int,
    state: str,
    case_index: int,
) -> dict[str, Any]:
    route, congruence, target, order, orientation = state_factors(state)
    raw_prompt, meta = render_prompt(
        pair, template, route, congruence, target, order, orientation,
        name_pair(names, template, item),
    )
    rendered = (
        source.phase1101.base.behavior_tools.render_native(
            tokenizer, model_name, raw_prompt, with_system=False
        )
        + ASSISTANT_PREFILL
    )
    input_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    expected_class = "e0" if meta["expected"] == meta["entity0"] else "e1"
    candidate_labels = {"e0": meta["entity0"], "e1": meta["entity1"]}
    candidate_token_ids = {
        key: source.phase1101.base.continuation_ids(tokenizer, rendered, label)
        for key, label in candidate_labels.items()
    }
    unit_id = f"phase1105.{model_name}.{pair}.t{template}.i{item:02d}"
    return {
        "schema_version": "phase1105_natural_synonym_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{unit_id}.{state}",
        "unit_id": unit_id,
        "relation_pair": pair,
        "surface": "en",
        "template": template,
        "split": split_for_template(template),
        "item_index": item,
        "route_type": route,
        "congruence": congruence,
        "target_relation": target,
        "target_relation_name": meta["target_relation_name"],
        "relation_order": order,
        "orientation": orientation,
        "state": state,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "input_length": len(input_ids),
        "candidate_labels": candidate_labels,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {key: [int(ids[0])] for key, ids in candidate_token_ids.items()},
        "expected_class": expected_class,
        "query_predicate": meta["query_predicate"],
        "target_fact": meta["target_fact"],
        "question": meta["question"],
        "prompt_digest": sha256_text(raw_prompt),
    }


def generate_cases(model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = tokenizer_for(model_name)
    names = selected_names()
    rows = []
    for pair, template, item, state in itertools.product(
        RELATION_PAIRS, TEMPLATES, range(ITEMS_PER_TEMPLATE), STATES
    ):
        rows.append(build_case(
            tokenizer, model_name, names, pair, template, item, state, len(rows)
        ))
    return rows, audit_cases(model_name, rows)


def audit_cases(model_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_count = (
        len(RELATION_PAIRS) * len(TEMPLATES) * ITEMS_PER_TEMPLATE * len(STATES)
    )
    checks: dict[str, bool] = {
        "case_count": len(rows) == expected_count,
        "unique_record_ids": len({row["record_id"] for row in rows}) == len(rows),
        "candidate_names_one_token": all(
            all(len(ids) == 1 for ids in row["candidate_token_ids"].values()) for row in rows
        ),
        "candidate_tokens_distinct": all(
            row["candidate_first_token_ids"]["e0"] != row["candidate_first_token_ids"]["e1"]
            for row in rows
        ),
        "exact_query_repeats_fact_language": all(
            row["query_predicate"].lower() in row["target_fact"].lower()
            for row in rows if row["route_type"] == "exact"
        ),
        "semantic_queries_nonidentical_to_fact": all(
            row["query_predicate"].lower() not in row["target_fact"].lower()
            for row in rows if row["route_type"] in {"close_synonym", "natural_definition"}
        ),
        "no_artificial_key_scaffold": all(
            not any(term in row["raw_prompt"].lower() for term in (
                "under the key", "criterion", "candidate a", "candidate b",
                "return exactly", "using the", "lookup table",
            )) for row in rows
        ),
        "answer_balanced": all(
            count == len(rows) // 2
            for count in Counter(row["expected_class"] for row in rows).values()
        ),
    }
    split_names: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_names[row["split"]].update(row["candidate_labels"].values())
    checks["split_name_pools_disjoint"] = not (
        split_names["qualification"] & split_names["confirmation"]
    )
    lexical_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row["item_index"] == 0 and row["target_relation"] == 0:
            lexical_sets[(row["target_relation_name"], row["split"], row["route_type"])].add(
                row["query_predicate"].lower()
            )
    checks["query_lexicon_split_disjoint"] = all(
        not (
            lexical_sets[(relation, "qualification", route)]
            & lexical_sets[(relation, "confirmation", route)]
        )
        for relation in FORMS
        for route in ("exact", "close_synonym", "natural_definition")
    )
    grouped: dict[tuple[Any, ...], dict[int, str]] = defaultdict(dict)
    for row in rows:
        key = (
            row["relation_pair"], row["template"], row["item_index"],
            row["route_type"], row["congruence"], row["relation_order"],
            row["orientation"],
        )
        grouped[key][int(row["target_relation"])] = row["expected_class"]
    checks["conflict_requires_relation_choice"] = all(
        values[0] != values[1]
        for key, values in grouped.items() if key[4] == "conflict"
    )
    checks["congruent_removes_relation_choice"] = all(
        values[0] == values[1]
        for key, values in grouped.items() if key[4] == "congruent"
    )
    source_final = read_json(SOURCE_FINAL)
    source_audit = read_json(SOURCE_AUDIT)
    checks["source_audit_passed"] = source_audit["all_checks_passed"]
    checks["pairs_equal_phase1104_cross_model_behavior_pairs"] = (
        tuple(source_final["behavior"]["cross_model_behavior_pairs"]) == RELATION_PAIRS
    )
    factor_counts = Counter(
        (row["relation_pair"], row["split"], row["route_type"], row["congruence"])
        for row in rows
    )
    checks["factorial_balance"] = len(set(factor_counts.values())) == 1
    result = {
        "schema_version": "phase1105_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "checks": checks,
        "case_count": len(rows),
        "expected_case_count": expected_count,
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = digest(result)
    return result


def main() -> None:
    protocol_root = OUT_ROOT / "protocol"
    model_digests = {}
    model_audits = {}
    for model_name in MODELS:
        rows, audit = generate_cases(model_name)
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"Phase1105 protocol audit failed for {model_name}: {audit}")
        path = protocol_root / f"cases.{model_name}.jsonl"
        write_jsonl(path, rows)
        model_digests[model_name] = digest(rows)
        model_audits[model_name] = audit
        write_json(protocol_root / f"audit.{model_name}.json", audit)
    prereg = {
        "schema_version": "phase1105_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase": 1104,
        "source_final_digest": read_json(SOURCE_FINAL)["final_summary_digest"],
        "source_audit_digest": read_json(SOURCE_AUDIT)["audit_digest"],
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "relation_pairs": list(RELATION_PAIRS),
        "templates_by_split": {key: list(value) for key, value in TEMPLATES_BY_SPLIT.items()},
        "items_per_template": ITEMS_PER_TEMPLATE,
        "route_types": list(ROUTE_TYPES),
        "congruences": list(CONGRUENCES),
        "model_case_digests": model_digests,
        "thresholds": THRESHOLDS,
        "claim_boundary": (
            "A pass authorizes a relation-pair-specific natural synonym behavior ledger. "
            "It does not establish a neural coordinate, causal transport, or family-wide semantics."
        ),
        "automatic_next_rule": (
            "If the same pair passes both splits in at least two models, Phase1106 may map the "
            "query-triggered semantic routing event without causal localization. Otherwise Phase1106 "
            "must repair natural paraphrase quality or interface behavior before hidden-state access."
        ),
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    combined = {
        "schema_version": "phase1105_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_audit_digests": {key: value["audit_digest"] for key, value in model_audits.items()},
        "checks": {
            "all_model_audits_passed": all(value["all_checks_passed"] for value in model_audits.values()),
            "all_case_digests_distinct": len(set(model_digests.values())) == len(MODELS),
            "source_result_audit_passed": read_json(SOURCE_AUDIT)["all_checks_passed"],
        },
    }
    combined["all_checks_passed"] = all(combined["checks"].values())
    combined["audit_digest"] = digest(combined)
    write_json(protocol_root / "audit.json", combined)
    print(json.dumps({
        "phase": PHASE,
        "cases_per_model": {model: model_audits[model]["case_count"] for model in MODELS},
        "protocol_digest": prereg["protocol_digest"],
        "audit_digest": combined["audit_digest"],
        "all_checks_passed": combined["all_checks_passed"],
    }))


if __name__ == "__main__":
    main()
