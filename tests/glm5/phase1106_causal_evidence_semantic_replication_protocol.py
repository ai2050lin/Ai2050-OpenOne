#!/usr/bin/env python3
"""Freeze Phase1106 claim-aligned natural semantic-routing replication."""

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
import phase1105_natural_synonym_address_protocol as source


PHASE = 1106
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
RELATION_PAIR = "causal_influence__evidence_strength"
RELATIONS = ("causal_influence", "evidence_strength")
TEMPLATES = tuple(range(8))
TEMPLATES_BY_SPLIT = {
    "qualification": (0, 1, 2, 3),
    "confirmation": (4, 5, 6, 7),
}
ITEMS_PER_TEMPLATE = 6
ROUTE_TYPES = source.ROUTE_TYPES
CONGRUENCES = source.CONGRUENCES
TARGET_RELATIONS = source.TARGET_RELATIONS
RELATION_ORDERS = source.RELATION_ORDERS
ORIENTATIONS = source.ORIENTATIONS
ASSISTANT_PREFILL = source.ASSISTANT_PREFILL
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1106_causal_evidence_semantic_replication"
SOURCE_ROOT = source.OUT_ROOT
SOURCE_AUTHORIZATION = SOURCE_ROOT / "analysis" / "behavior_authorization.json"
SOURCE_AUDIT = SOURCE_ROOT / "audit" / "result_audit.json"

write_json = source.write_json
write_jsonl = source.write_jsonl
read_json = source.read_json
read_jsonl = source.read_jsonl
digest = source.digest
sha256_text = source.sha256_text


SELECTED_NAMES = (
    "Aaron", "Ada", "Adrian", "Albert", "Alexis", "Alfred",
    "Alice", "Allen", "Amanda", "Amy", "Angela", "Angelo",
    "Anna", "Annie", "Arthur", "Audrey", "Barbara", "Barry",
    "Benjamin", "Bernard", "Brandon", "Brenda", "Brian", "Bruce",
)

FORMS = {
    "causal_influence": (
        ("was a stronger driver of the outcome than", "was a stronger driver of the outcome", "had more influence in causing the result", "did more to bring the outcome about"),
        ("was more causally important to the result than", "was more causally important to the result", "had the stronger effect in producing the outcome", "contributed more to why the result occurred"),
        ("changed the outcome more through its causal role than", "changed the outcome more through its causal role", "exerted the larger effect on what happened", "played more of the role that made the result occur"),
        ("was the more powerful cause of the observed result than", "was the more powerful cause of the observed result", "had greater influence over the outcome's occurrence", "did more to produce what was observed"),
        ("had the stronger role in causing the final result than", "had the stronger role in causing the final result", "made the greater causal contribution to the outcome", "was more important in bringing the result about"),
        ("produced a larger change in the outcome than", "produced a larger change in the outcome", "had the greater effect as a cause", "contributed more to the result happening"),
        ("was the more influential cause of the result than", "was the more influential cause of the result", "had the larger causal role in what occurred", "did more to make the outcome happen"),
        ("contributed more to the outcome's occurrence than", "contributed more to the outcome's occurrence", "exerted the stronger causal effect", "played more of the role that produced the result"),
    ),
    "evidence_strength": (
        ("had a stronger evidential basis than", "had a stronger evidential basis", "was supported by more persuasive evidence", "had the observations more firmly on its side"),
        ("was backed by more compelling data than", "was backed by more compelling data", "had the stronger body of evidence", "was better supported by what investigators observed"),
        ("rested on more convincing empirical support than", "rested on more convincing empirical support", "had the firmer evidential foundation", "fit the available evidence more convincingly"),
        ("had the better-supported empirical case than", "had the better-supported empirical case", "was backed by stronger observed evidence", "had more convincing support from the data"),
        ("received more persuasive support from the data than", "received more persuasive support from the data", "had the stronger evidential case", "was more firmly backed by the observations"),
        ("was grounded in stronger empirical evidence than", "was grounded in stronger empirical evidence", "had more convincing support from the findings", "had the data more clearly in its favor"),
        ("had a more robust body of supporting observations than", "had a more robust body of supporting observations", "rested on stronger evidence", "was better backed by what was measured"),
        ("was supported by a stronger set of findings than", "was supported by a stronger set of findings", "had the more convincing empirical basis", "matched the observed data with better support"),
    ),
}

SHELLS = (
    "An investigation examined two proposed factors. {fact1}. Independently, {fact2}. {question}?",
    "A review compared two possible factors in two different respects. First, {fact1}. Separately, {fact2}. {question}?",
    "Researchers reported two independent findings about the same proposed factors. {fact1}. They also found that {fact2}. {question}?",
    "An inquiry evaluated two possible explanations from two angles. One finding said that {fact1}. Another said that {fact2}. {question}?",
    "A new investigation compared two candidate factors. Its first analysis found that {fact1}. A separate analysis found that {fact2}. {question}?",
    "Two proposed causes were assessed in independent parts of a report. The report found that {fact1}. It separately concluded that {fact2}. {question}?",
    "A replication study made two distinct comparisons of the same factors. According to one comparison, {fact1}. According to the other, {fact2}. {question}?",
    "Investigators evaluated two competing factors without combining the assessments. They found that {fact1}. In another analysis, {fact2}. {question}?",
)

THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_semantic_template_accuracy": 0.75,
    "minimum_semantic_aggregate_accuracy": 0.88,
    "minimum_semantic_target_accuracy": 0.82,
    "minimum_models_for_shared_replication": 2,
}


def split_for_template(template: int) -> str:
    for split, templates in TEMPLATES_BY_SPLIT.items():
        if template in templates:
            return split
    raise ValueError(template)


def pairing(round_index: int) -> tuple[tuple[int, int], ...]:
    # Four edge-disjoint rounds of the 12-name round-robin schedule.
    fixed = 11
    rotating = 11
    result = [(fixed, round_index % rotating)]
    for offset in range(1, 6):
        result.append(((round_index - offset) % rotating, (round_index + offset) % rotating))
    return tuple(result)


def name_pair(template: int, item: int) -> tuple[str, str]:
    split = split_for_template(template)
    pool = SELECTED_NAMES[:12] if split == "qualification" else SELECTED_NAMES[12:]
    local_template = TEMPLATES_BY_SPLIT[split].index(template)
    left, right = pairing(local_template)[item]
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
    template: int,
    route: str,
    congruence: str,
    target: int,
    order: int,
    orientation: int,
    names: tuple[str, str],
) -> tuple[str, dict[str, Any]]:
    entity0, entity1 = names
    winner0, loser0 = (entity0, entity1) if orientation == 0 else (entity1, entity0)
    winner1, loser1 = (loser0, winner0) if congruence == "conflict" else (winner0, loser0)
    winners = {RELATIONS[0]: winner0, RELATIONS[1]: winner1}
    losers = {RELATIONS[0]: loser0, RELATIONS[1]: loser1}
    facts = {
        relation: (
            f"The proposed factor [ {winners[relation]} ] {FORMS[relation][template][0]} "
            f"the proposed factor [ {losers[relation]} ]"
        )
        for relation in RELATIONS
    }
    displayed = RELATIONS if order == 0 else tuple(reversed(RELATIONS))
    target_relation = RELATIONS[target]
    if route == "exact":
        query_predicate = FORMS[target_relation][template][1]
    elif route == "close_synonym":
        query_predicate = FORMS[target_relation][template][2]
    elif route == "natural_definition":
        query_predicate = FORMS[target_relation][template][3]
    else:
        ordinal = ("first", "second")[displayed.index(target_relation)]
        query_predicate = f"came out ahead in the {ordinal} comparison"
    question = f"Which proposed factor {query_predicate}"
    raw_prompt = SHELLS[template].format(
        fact1=facts[displayed[0]], fact2=facts[displayed[1]], question=question
    )
    expected = winners[target_relation]
    return raw_prompt, {
        "entity0": entity0,
        "entity1": entity1,
        "expected": expected,
        "target_relation_name": target_relation,
        "target_fact": facts[target_relation],
        "query_predicate": query_predicate,
        "question": question,
    }


def build_case(tokenizer, model_name: str, template: int, item: int, state: str, case_index: int) -> dict[str, Any]:
    route, congruence, target, order, orientation = state_factors(state)
    raw_prompt, meta = render_prompt(
        template, route, congruence, target, order, orientation, name_pair(template, item)
    )
    rendered = (
        source.source.phase1101.base.behavior_tools.render_native(
            tokenizer, model_name, raw_prompt, with_system=False
        )
        + ASSISTANT_PREFILL
    )
    input_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    expected_class = "e0" if meta["expected"] == meta["entity0"] else "e1"
    labels = {"e0": meta["entity0"], "e1": meta["entity1"]}
    candidate_token_ids = {
        key: source.source.phase1101.base.continuation_ids(tokenizer, rendered, value)
        for key, value in labels.items()
    }
    unit_id = f"phase1106.{model_name}.{RELATION_PAIR}.t{template}.i{item:02d}"
    return {
        "schema_version": "phase1106_semantic_replication_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{unit_id}.{state}",
        "unit_id": unit_id,
        "relation_pair": RELATION_PAIR,
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
        "candidate_labels": labels,
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {key: [int(ids[0])] for key, ids in candidate_token_ids.items()},
        "expected_class": expected_class,
        "query_predicate": meta["query_predicate"],
        "target_fact": meta["target_fact"],
        "question": meta["question"],
        "prompt_digest": sha256_text(raw_prompt),
    }


def source_semantic_candidates() -> list[str]:
    authorization = read_json(SOURCE_AUTHORIZATION)
    candidates = []
    for pair in source.RELATION_PAIRS:
        passed = True
        for model in ("qwen3", "glm4"):
            pair_row = authorization["models"][model]["pair_results"][pair]
            for split in source.TEMPLATES_BY_SPLIT:
                gates = pair_row["splits"][split]["gates"]
                passed &= all(gates[key] for key in (
                    "finite", "template_close_synonym", "template_natural_definition",
                    "close_synonym", "natural_definition",
                ))
        if passed:
            candidates.append(pair)
    return candidates


def generate_cases(model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = tokenizer_for(model_name)
    rows = [
        build_case(tokenizer, model_name, template, item, state, index)
        for index, (template, item, state) in enumerate(
            itertools.product(TEMPLATES, range(ITEMS_PER_TEMPLATE), STATES)
        )
    ]
    return rows, audit_cases(model_name, rows)


def audit_cases(model_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_count = len(TEMPLATES) * ITEMS_PER_TEMPLATE * len(STATES)
    old_names = set(source.selected_names())
    source_queries = {
        row["query_predicate"].lower()
        for row in read_jsonl(SOURCE_ROOT / "protocol" / f"cases.{model_name}.jsonl")
        if row["route_type"] in {"exact", "close_synonym", "natural_definition"}
    }
    current_queries = {
        row["query_predicate"].lower()
        for row in rows if row["route_type"] in {"exact", "close_synonym", "natural_definition"}
    }
    checks: dict[str, bool] = {
        "case_count": len(rows) == expected_count,
        "unique_record_ids": len({row["record_id"] for row in rows}) == len(rows),
        "new_names_vs_phase1105": not (set(SELECTED_NAMES) & old_names),
        "candidate_names_one_token": all(all(len(ids) == 1 for ids in row["candidate_token_ids"].values()) for row in rows),
        "candidate_tokens_distinct": all(row["candidate_first_token_ids"]["e0"] != row["candidate_first_token_ids"]["e1"] for row in rows),
        "exact_query_repeats_fact_language": all(row["query_predicate"].lower() in row["target_fact"].lower() for row in rows if row["route_type"] == "exact"),
        "semantic_queries_nonidentical_to_fact": all(row["query_predicate"].lower() not in row["target_fact"].lower() for row in rows if row["route_type"] in {"close_synonym", "natural_definition"}),
        "new_query_phrases_vs_phase1105": not (source_queries & current_queries),
        "no_artificial_key_scaffold": all(not any(term in row["raw_prompt"].lower() for term in ("under the key", "using the key", "using the criterion", "candidate a", "candidate b", "return exactly", "lookup table")) for row in rows),
        "answer_balanced": set(Counter(row["expected_class"] for row in rows).values()) == {len(rows) // 2},
        "source_audit_passed": read_json(SOURCE_AUDIT)["all_checks_passed"],
        "unique_source_semantic_candidate": source_semantic_candidates() == [RELATION_PAIR],
    }
    split_names: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_names[row["split"]].update(row["candidate_labels"].values())
    checks["split_name_pools_disjoint"] = not (split_names["qualification"] & split_names["confirmation"])
    lexical_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row["route_type"] in {"exact", "close_synonym", "natural_definition"}:
            lexical_sets[(row["split"], row["route_type"])].add(row["query_predicate"].lower())
    checks["query_lexicon_split_disjoint"] = all(
        not (lexical_sets[("qualification", route)] & lexical_sets[("confirmation", route)])
        for route in ("exact", "close_synonym", "natural_definition")
    )
    grouped: dict[tuple[Any, ...], dict[int, str]] = defaultdict(dict)
    for row in rows:
        key = (row["template"], row["item_index"], row["route_type"], row["congruence"], row["relation_order"], row["orientation"])
        grouped[key][int(row["target_relation"])] = row["expected_class"]
    checks["conflict_requires_relation_choice"] = all(values[0] != values[1] for key, values in grouped.items() if key[3] == "conflict")
    checks["congruent_removes_relation_choice"] = all(values[0] == values[1] for key, values in grouped.items() if key[3] == "congruent")
    result = {
        "schema_version": "phase1106_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "checks": checks,
        "case_count": len(rows),
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
            raise RuntimeError(f"Phase1106 protocol audit failed for {model_name}: {audit}")
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_digests[model_name] = digest(rows)
        model_audits[model_name] = audit
    prereg = {
        "schema_version": "phase1106_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase": 1105,
        "source_authorization_digest": read_json(SOURCE_AUTHORIZATION)["authorization_digest"],
        "source_audit_digest": read_json(SOURCE_AUDIT)["audit_digest"],
        "candidate_selection_rule": "The unique relation pair whose close-synonym and natural-definition gates passed both splits in both Qwen3 and GLM4 in Phase1105.",
        "relation_pair": RELATION_PAIR,
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "selected_names": list(SELECTED_NAMES),
        "templates_by_split": {key: list(value) for key, value in TEMPLATES_BY_SPLIT.items()},
        "items_per_template": ITEMS_PER_TEMPLATE,
        "route_types": list(ROUTE_TYPES),
        "model_case_digests": model_digests,
        "thresholds": THRESHOLDS,
        "claim_aligned_gate": "Only close-synonym and natural-definition conflict routes authorize semantic replication. Exact, ordinal, and congruent routes are frozen diagnostics and cannot veto this narrower claim.",
        "automatic_next_rule": "A shared pass in at least two models authorizes Phase1107 signed event mapping without causal or component claims. Otherwise repair behavior before hidden-state access.",
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    combined = {
        "schema_version": "phase1106_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_audit_digests": {model: row["audit_digest"] for model, row in model_audits.items()},
        "checks": {
            "all_model_audits_passed": all(row["all_checks_passed"] for row in model_audits.values()),
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
