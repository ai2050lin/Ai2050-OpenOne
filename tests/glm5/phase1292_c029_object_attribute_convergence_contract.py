#!/usr/bin/env python3
"""Phase 1292: freeze the C029 externally grounded binding contract.

This phase is zero-model by construction.  It selects one historical object,
builds fresh typed materials, freezes gates and stop rules, and permits only a
tokenizer load.  No model weights or hidden states are touched.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

PHASE = 1292
CAMPAIGN = "C029"
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1292_c029_object_attribute_convergence_contract_audit.py"
OUT = TEST_ROOT / "result/phase1292_c029_object_attribute_convergence_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_object_attribute_cases.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE_AUDIT = OUT / "audit/tokenizer_semantic_program_audit.json"
INDEPENDENT_AUDIT = OUT / "audit/independent_final_audit.json"
FINAL = OUT / "analysis/final.json"

MODEL_NAME = "qwen3"
SYSTEM_PROMPT = "Use only the supplied catalog. Reply exactly as requested and do not explain."
PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRIBUTES = ("color", "material", "location", "size", "shape", "status")
PANELS = ("active", "matched_null", "surface_only", "semantic_neighbor")
SURFACES = ("catalog_prose", "inventory_ledger")
CANDIDATE_ORDERS = (0, 1, 2)
BINDING_STATES = (0, 1)
PROFILES_PER_PARTITION = 8

ATTRIBUTE_LEXEME = {
    "color": "color",
    "material": "material",
    "location": "storage location",
    "size": "size",
    "shape": "shape",
    "status": "status",
}

VALUE_BANKS: dict[str, dict[str, tuple[str, str, str]]] = {
    "discovery": {
        "color": ("crimson", "azure", "emerald"),
        "material": ("timber", "ceramic", "marble"),
        "location": ("atrium", "basement", "rooftop"),
        "size": ("tiny", "moderate", "giant"),
        "shape": ("triangular", "rectangular", "spherical"),
        "status": ("available", "reserved", "pending"),
    },
    "confirmation": {
        "color": ("scarlet", "navy", "lime"),
        "material": ("bronze", "fabric", "concrete"),
        "location": ("hangar", "lobby", "vault"),
        "size": ("compact", "standard", "oversized"),
        "shape": ("hexagonal", "cylindrical", "conical"),
        "status": ("approved", "denied", "queued"),
    },
    "holdout": {
        "color": ("violet", "amber", "ivory"),
        "material": ("leather", "granite", "porcelain"),
        "location": ("annex", "courtyard", "workshop"),
        "size": ("miniature", "typical", "massive"),
        "shape": ("spiral", "concave", "convex"),
        "status": ("enabled", "disabled", "delayed"),
    },
}

NAME_POOL = (
    "Abigail", "Adrian", "Albert", "Alice", "Amanda", "Amber", "Andrea", "Andrew",
    "Angela", "Anita", "Anthony", "Ashley", "Austin", "Barbara", "Barry", "Betty",
    "Beverly", "Brandon", "Brenda", "Brittany", "Caleb", "Cameron", "Carl", "Carmen",
    "Catherine", "Charles", "Charlotte", "Cheryl", "Christian", "Christine", "Clara", "Craig",
    "Crystal", "Daisy", "David", "Deborah", "Diana", "Donald", "Donna", "Dorothy",
    "Douglas", "Dylan", "Edgar", "Edith", "Eleanor", "Elijah", "Emily", "Ethan",
    "Eugene", "Evelyn", "Faith", "Fiona", "Florence", "Gabriel", "Gary", "Gavin",
    "Gloria", "Grace", "Hannah", "Harold", "Heather", "Helen", "Howard", "Ian",
    "Irene", "Isaac", "Isabel", "Ivan", "Janet", "Jasmine", "Jeffrey", "Jennifer",
    "Jeremy", "Jessica", "Joan", "Jonathan", "Jordan", "Joyce", "Judith", "Julia",
    "Justin", "Keith", "Kenneth", "Kevin", "Kimberly", "Kyle", "Laura", "Lauren",
    "Lawrence", "Leonard", "Leslie", "Lillian", "Louis", "Lucas", "Margaret", "Maria",
    "Martha", "Matthew", "Megan", "Melissa", "Michelle", "Nathan", "Nicholas", "Nicole",
    "Noah", "Olivia", "Pamela", "Patricia", "Patrick", "Peter", "Philip", "Raymond",
    "Richard", "Ronald", "Rose", "Russell", "Samuel", "Sandra", "Sharon", "Sophia",
    "Stephanie", "Stephen", "Susan", "Teresa", "Thomas", "Timothy", "Victoria", "Vincent",
    "Walter", "Wayne", "Wendy", "William",
)

THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "overall_candidate_accuracy_min": 0.95,
    "partition_candidate_accuracy_min": 0.94,
    "panel_candidate_accuracy_min": 0.93,
    "surface_candidate_accuracy_min": 0.93,
    "base_side_accuracy_min": 0.93,
    "active_pair_success_min": 0.90,
    "matched_null_pair_success_min": 0.90,
    "surface_only_pair_success_min": 0.90,
    "semantic_neighbor_pair_success_min": 0.90,
    "candidate_order_triple_success_min": 0.90,
    "cross_surface_pair_success_min": 0.90,
    "generation_coverage_min": 0.95,
    "generation_accuracy_min": 0.90,
    "generation_pair_success_min": 0.85,
    "shortcut_program_accuracy_max": 0.70,
}

OBJECT_REGISTRY = (
    {
        "object": "object_attribute_inverse_lookup",
        "historical_basis": ["K184", "K185", "K186", "K187"],
        "criteria": {
            "external_world_gold": 1,
            "stable_input_output_type": 1,
            "qwen_behavior_pass": 1,
            "independent_hidden_repeat": 1,
            "causal_sufficiency": 1,
            "natural_generation_pass": 0,
        },
    },
    {
        "object": "query_object_marker_lookup",
        "historical_basis": ["K199", "K209"],
        "criteria": {
            "external_world_gold": 1,
            "stable_input_output_type": 1,
            "qwen_behavior_pass": 1,
            "independent_hidden_repeat": 0,
            "causal_sufficiency": 0,
            "natural_generation_pass": 0,
        },
    },
    {
        "object": "typed_binary_complement",
        "historical_basis": ["K259"],
        "criteria": {
            "external_world_gold": 0,
            "stable_input_output_type": 0,
            "qwen_behavior_pass": 0,
            "independent_hidden_repeat": 0,
            "causal_sufficiency": 0,
            "natural_generation_pass": 0,
        },
    },
    {
        "object": "expectation_response_signature",
        "historical_basis": ["K247", "K252", "K257"],
        "criteria": {
            "external_world_gold": 0,
            "stable_input_output_type": 1,
            "qwen_behavior_pass": 0,
            "independent_hidden_repeat": 0,
            "causal_sufficiency": 0,
            "natural_generation_pass": 0,
        },
    },
)

PERMUTATIONS = tuple(itertools.permutations(range(3)))
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[^\w\s]", re.UNICODE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def lexical_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def token_multiset_digest(text: str) -> str:
    return digest(sorted(lexical_tokens(text)))


def rotate(values: tuple[str, str, str], offset: int) -> tuple[str, str, str]:
    return values[offset:] + values[:offset]


def selected_object() -> dict[str, Any]:
    ranked = []
    for item in OBJECT_REGISTRY:
        score = sum(int(value) for value in item["criteria"].values())
        ranked.append((score, item["object"], item))
    ranked.sort(key=lambda value: (-value[0], value[1]))
    return ranked[0][2]


def select_names(tokenizer: Any) -> tuple[str, ...]:
    eligible = []
    for name in NAME_POOL:
        token_ids = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(token_ids) == 1:
            eligible.append(name)
    needed = len(PARTITIONS) * PROFILES_PER_PARTITION * 3
    if len(eligible) < needed:
        raise RuntimeError(f"only {len(eligible)} single-token names; need {needed}")
    return tuple(eligible[:needed])


def base_assignments(partition: str, profile_index: int, entities: tuple[str, str, str]) -> dict[str, dict[str, str]]:
    assignments = {entity: {} for entity in entities}
    partition_index = PARTITIONS.index(partition)
    for attribute_index, attribute in enumerate(ATTRIBUTES):
        values = VALUE_BANKS[partition][attribute]
        permutation = PERMUTATIONS[(partition_index + 2 * profile_index + attribute_index) % len(PERMUTATIONS)]
        for entity_index, entity in enumerate(entities):
            assignments[entity][attribute] = values[permutation[entity_index]]
    return assignments


def clone_assignments(value: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    return {entity: dict(fields) for entity, fields in value.items()}


def swap(assignments: dict[str, dict[str, str]], left: str, right: str, attribute: str) -> None:
    assignments[left][attribute], assignments[right][attribute] = assignments[right][attribute], assignments[left][attribute]


def record_clause(entity: str, fields: dict[str, str], surface: str) -> str:
    if surface == "catalog_prose":
        return (
            f"The sample named {entity} has a {fields['color']} color, is made of {fields['material']}, "
            f"is stored in the {fields['location']}, is {fields['size']} in size, has a {fields['shape']} "
            f"shape, and is marked {fields['status']}."
        )
    return (
        f"{entity} - color: {fields['color']}; material: {fields['material']}; "
        f"storage location: {fields['location']}; size: {fields['size']}; "
        f"shape: {fields['shape']}; status: {fields['status']}."
    )


def query_clause(attribute: str, value: str, surface: str) -> str:
    if surface == "inventory_ledger":
        return f"Which listed sample has {ATTRIBUTE_LEXEME[attribute]}: {value}?"
    if attribute == "color":
        return f"According to the catalog, which sample has a {value} color?"
    if attribute == "material":
        return f"According to the catalog, which sample is made of {value}?"
    if attribute == "location":
        return f"According to the catalog, which sample is stored in the {value}?"
    if attribute == "size":
        return f"According to the catalog, which sample is {value} in size?"
    if attribute == "shape":
        return f"According to the catalog, which sample has a {value} shape?"
    return f"According to the catalog, which sample is marked {value}?"


def all_spans(text: str, needle: str) -> list[list[int]]:
    return [[match.start(), match.end()] for match in re.finditer(re.escape(needle), text)]


def render_prompts(
    assignments: dict[str, dict[str, str]],
    record_order: tuple[str, str, str],
    attribute: str,
    target_value: str,
    candidates: tuple[str, str, str],
    surface: str,
) -> tuple[str, str, dict[str, Any]]:
    records = [record_clause(entity, assignments[entity], surface) for entity in record_order]
    query = query_clause(attribute, target_value, surface)
    prefix = "Catalog entries:" if surface == "catalog_prose" else "Inventory ledger:"
    base = " ".join([prefix, *records, query])
    candidate_instruction = f"Choose exactly one name from {', '.join(candidates)}. Answer:"
    generation_instruction = "Reply with only the sample name. Answer:"
    candidate_prompt = " ".join([base, candidate_instruction])
    generation_prompt = " ".join([base, generation_instruction])
    spans = {
        "records": [],
        "query": all_spans(candidate_prompt, query),
        "query_value": all_spans(candidate_prompt, target_value),
        "answer_boundary": all_spans(candidate_prompt, "Answer:"),
    }
    cursor = len(prefix) + 1
    for entity, record in zip(record_order, records):
        start = candidate_prompt.find(record, cursor)
        end = start + len(record)
        spans["records"].append({
            "entity": entity,
            "record": [start, end],
            "entity_spans": [[start + a, start + b] for a, b in all_spans(record, entity)],
            "queried_attribute_value_spans": [
                [start + a, start + b] for a, b in all_spans(record, assignments[entity][attribute])
            ],
        })
        cursor = end
    return candidate_prompt, generation_prompt, spans


def build_cases(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = select_names(tokenizer)
    rows: list[dict[str, Any]] = []
    name_cursor = 0
    for partition in PARTITIONS:
        for profile_index in range(PROFILES_PER_PARTITION):
            entities = tuple(names[name_cursor:name_cursor + 3])
            name_cursor += 3
            base = base_assignments(partition, profile_index, entities)
            for attribute_index, attribute in enumerate(ATTRIBUTES):
                focus_index = (profile_index + attribute_index) % 2
                neighbor = ATTRIBUTES[(attribute_index + 1) % len(ATTRIBUTES)]
                base_target = base[entities[focus_index]][attribute]
                for panel in PANELS:
                    for surface in SURFACES:
                        for candidate_order in CANDIDATE_ORDERS:
                            candidates = rotate(entities, candidate_order)
                            for state in BINDING_STATES:
                                assignments = clone_assignments(base)
                                record_order = entities
                                target_value = base_target
                                if panel in {"active", "matched_null"} and state == 1:
                                    swap(assignments, entities[0], entities[1], attribute)
                                if panel == "matched_null":
                                    target_value = base[entities[2]][attribute]
                                elif panel == "surface_only" and state == 1:
                                    record_order = (entities[1], entities[0], entities[2])
                                elif panel == "semantic_neighbor" and state == 1:
                                    swap(assignments, entities[0], entities[1], neighbor)
                                matches = [entity for entity in entities if assignments[entity][attribute] == target_value]
                                if len(matches) != 1:
                                    raise AssertionError((partition, profile_index, attribute, panel, state, matches))
                                gold = matches[0]
                                prompt, generation_prompt, spans = render_prompts(
                                    assignments, record_order, attribute, target_value, candidates, surface
                                )
                                group = f"{partition}|p{profile_index:02d}|{attribute}|{panel}|{surface}|o{candidate_order}"
                                case_key = f"{group}|s{state}"
                                rows.append({
                                    "schema_version": "phase1292.c029.case.v1",
                                    "case_id": "c029-" + digest(case_key)[:20],
                                    "group_id": group,
                                    "partition": partition,
                                    "profile_index": profile_index,
                                    "attribute": attribute,
                                    "neighbor_attribute": neighbor,
                                    "panel": panel,
                                    "surface": surface,
                                    "candidate_order": candidate_order,
                                    "binding_state": state,
                                    "entities": list(entities),
                                    "record_order": list(record_order),
                                    "candidates": list(candidates),
                                    "assignments": assignments,
                                    "target_value": target_value,
                                    "gold_candidate": gold,
                                    "gold_position": candidates.index(gold),
                                    "candidate_prompt": prompt,
                                    "generation_prompt": generation_prompt,
                                    "prompt_token_multiset_digest": token_multiset_digest(prompt),
                                    "typed_spans": spans,
                                })
    expected = len(PARTITIONS) * PROFILES_PER_PARTITION * len(ATTRIBUTES) * len(PANELS) * len(SURFACES) * len(CANDIDATE_ORDERS) * len(BINDING_STATES)
    if len(rows) != expected:
        raise AssertionError((len(rows), expected))

    rendered_lengths: list[int] = []
    candidate_lengths: list[int] = []
    for row in rows:
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["candidate_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rendered_ids = tokenizer.encode(rendered, add_special_tokens=False)
        rendered_lengths.append(len(rendered_ids))
        lengths = []
        for candidate in row["candidates"]:
            full_ids = tokenizer.encode(rendered + " " + candidate, add_special_tokens=False)
            suffix = full_ids[len(rendered_ids):]
            if full_ids[:len(rendered_ids)] != rendered_ids or not suffix:
                raise RuntimeError(f"candidate retokenized context: {row['case_id']} {candidate}")
            lengths.append(len(suffix))
            candidate_lengths.append(len(suffix))
        if len(set(lengths)) != 1 or lengths[0] != 1:
            raise RuntimeError(f"candidate lengths are not one matched token: {row['case_id']} {lengths}")

    return rows, {
        "tokenizer": "qwen3-fast-local",
        "selected_names": list(names),
        "eligible_name_count": sum(len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1 for name in NAME_POOL),
        "context_token_length_min": min(rendered_lengths),
        "context_token_length_max": max(rendered_lengths),
        "candidate_token_lengths": sorted(set(candidate_lengths)),
        "all_candidate_lengths_equal_within_case": True,
        "all_candidates_single_token": True,
    }


def program_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    programs = {
        "candidate_first": lambda row: row["candidates"][0],
        "candidate_second": lambda row: row["candidates"][1],
        "candidate_third": lambda row: row["candidates"][2],
        "record_first": lambda row: row["record_order"][0],
        "record_last": lambda row: row["record_order"][-1],
        "entity_first": lambda row: row["entities"][0],
        "entity_second": lambda row: row["entities"][1],
        "entity_third": lambda row: row["entities"][2],
    }
    accuracy = {}
    for name, fn in programs.items():
        accuracy[name] = sum(fn(row) == row["gold_candidate"] for row in rows) / len(rows)
    bag_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bag_groups[(row["partition"], row["attribute"], row["panel"], row["candidate_order"])].append(row)
    active_pairs = defaultdict(list)
    for row in rows:
        if row["panel"] == "active":
            active_pairs[row["group_id"]].append(row)
    collision_count = sum(
        len(pair) == 2
        and pair[0]["prompt_token_multiset_digest"] == pair[1]["prompt_token_multiset_digest"]
        and pair[0]["gold_candidate"] != pair[1]["gold_candidate"]
        for pair in active_pairs.values()
    )
    return {
        "program_accuracy": accuracy,
        "shortcut_ceiling": max(accuracy.values()),
        "active_same_bag_different_gold_pairs": collision_count,
        "active_pair_count": len(active_pairs),
        "bag_group_count": len(bag_groups),
    }


def naturalness_review(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prototypes = {}
    for row in rows:
        key = f"{row['surface']}|{row['attribute']}"
        prototypes.setdefault(key, {
            "candidate_prompt": row["candidate_prompt"],
            "generation_prompt": row["generation_prompt"],
        })
    forbidden = (
        "does not apply the", "fails to describe that first result", "unassigned alternative the",
        "does not belong to that intermediate label", "other state of that result",
    )
    flags = []
    for row in rows:
        text = row["candidate_prompt"]
        if any(fragment in text.lower() for fragment in forbidden):
            flags.append({"case_id": row["case_id"], "reason": "forbidden_malformed_phrase"})
        if "  " in text or text.count("?") != 1 or not text.endswith("Answer:"):
            flags.append({"case_id": row["case_id"], "reason": "surface_form"})
        if len(row["typed_spans"]["query"]) != 1 or len(row["typed_spans"]["answer_boundary"]) != 1:
            flags.append({"case_id": row["case_id"], "reason": "typed_span"})
    return {
        "reviewed_before_any_c029_weight_load": True,
        "reviewer_type": "researcher prototype review plus deterministic full-material grammar/type audit",
        "independent_human_panel": False,
        "type_signature": "(WorldState, Attribute, Value) -> Entity",
        "operation_requested_from_model": False,
        "semantic_gold_source": "unique lookup in the explicit external world-state table",
        "prototype_count": len(prototypes),
        "prototypes": prototypes,
        "forbidden_phrase_inventory": list(forbidden),
        "flags": flags,
        "semantic_unique": not flags,
        "limitation": (
            "The templates are grammatical controlled English and all cases are mechanically type checked, "
            "but no independent human rating panel was available; naturalness is therefore audited, not proven."
        ),
    }


def environment_snapshot() -> dict[str, Any]:
    import platform

    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "model_weights_loaded": False,
        "tokenizer_only": True,
    }


def build_protocol(rows: list[dict[str, Any]], token_audit: dict[str, Any], program: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    selected = selected_object()
    registry = []
    for item in OBJECT_REGISTRY:
        registry.append({**item, "score": sum(int(value) for value in item["criteria"].values())})
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "experiment_id": "EXP-C029-WP00-001",
        "schema_version": "phase1292.c029.preregistration.v1",
        "purpose": "select one stable externally grounded function and freeze its full behavior-to-causal branch before Qwen3 weights load",
        "historical_object_registry": registry,
        "selection_rule": "maximize the six frozen binary evidence criteria; break ties by object identifier",
        "selected_object": selected["object"],
        "selected_historical_basis": selected["historical_basis"],
        "construct": {
            "world_state": "a finite explicit map Entity x Attribute -> Value",
            "query": "retrieve the unique Entity for a supplied (Attribute, Value)",
            "type_signature": "(WorldState, Attribute, Value) -> Entity",
            "generator_only_perturbations": ["binding swap", "record-order swap", "neighbor-attribute swap"],
            "model_is_not_asked_to_execute": ["swap", "complement", "negate", "compose a metalinguistic operator"],
            "gold": "the unique entity satisfying the explicit world-state mapping",
        },
        "material": {
            "partitions": list(PARTITIONS),
            "profiles_per_partition": PROFILES_PER_PARTITION,
            "attributes": list(ATTRIBUTES),
            "panels": list(PANELS),
            "surfaces": list(SURFACES),
            "candidate_orders": list(CANDIDATE_ORDERS),
            "binding_states": list(BINDING_STATES),
            "case_count": len(rows),
            "independent_profile_count": len(PARTITIONS) * PROFILES_PER_PARTITION,
            "typed_query_count": len(PARTITIONS) * PROFILES_PER_PARTITION * len(ATTRIBUTES),
            "candidate_sequences": len(rows) * 3,
            "generation_cases": len(PARTITIONS[1:]) * PROFILES_PER_PARTITION * len(ATTRIBUTES) * len(PANELS) * len(SURFACES) * len(BINDING_STATES),
            "partition_entity_and_value_vocabularies_disjoint": True,
            "material_sha256": file_sha256(MATERIAL),
            "naturalness_sha256": file_sha256(NATURALNESS),
        },
        "model": {
            "behavior": ["qwen3-4b-fp16-cuda-no-quantization"],
            "other_models_authorized": False,
            "formal_behavior_runs": 1,
            "native_chat_template": True,
            "enable_thinking": False,
            "system_prompt": SYSTEM_PROMPT,
        },
        "zero_models": program,
        "tokenizer_audit": token_audit,
        "semantic_naturalness_review": {
            "semantic_unique": review["semantic_unique"],
            "flags": review["flags"],
            "independent_human_panel": review["independent_human_panel"],
            "limitation": review["limitation"],
        },
        "thresholds": THRESHOLDS,
        "behavior_gate": (
            "all finite, overall, partition, panel, surface, base-side, paired invariance, candidate-order, "
            "cross-surface, generation, and shortcut ledgers must pass"
        ),
        "hidden_contract_if_behavior_passes": {
            "object": "typed multi-event future-response path for the same C029 lookup",
            "events": ["record entity", "record queried value", "query attribute", "query value", "answer boundary"],
            "discovery_only_selection": "earliest adjacent residual-depth band whose active transfer exceeds every matched control",
            "confirmation_and_holdout": "frozen event/depth only",
            "forbidden": ["largest-activation search", "head/neuron fishing", "threshold repair", "surface deletion"],
        },
        "failure_and_stop_branches": {
            "phase1292_audit_pass": "authorize_phase1293_qwen3_behavior_only",
            "any_phase1293_behavior_or_generation_ledger_fails": "close_c029_without_hidden",
            "all_phase1293_ledgers_pass": "authorize_phase1294_multievent_future_response",
            "phase1294_response_or_confirmation_fails": "close_c029_without_causal_rescue",
            "phase1294_passes": "authorize_phase1295_path_cut_and_independent_rescue",
            "phase1295_necessity_or_rescue_fails": "close_c029_with_bounded_sufficiency_only",
            "all_qwen_ledgers_pass": "complete_c029_qwen_closure_and_require_new_cross_model_contract",
        },
        "freeze_rules": [
            "No C029 model weight may load before this contract and an independent replay audit pass.",
            "No object, material, split, model, zero model, threshold, parser, or stop branch may change after creation.",
            "Behavior and list-free generation must both pass before hidden states are measured.",
            "After unblinding, only the preregistered branch may run; failed branches close C029.",
            "No prompt repair, threshold relaxation, surface deletion, seed rerun, or other-model vote is permitted.",
            "A pass establishes only this finite typed lookup domain, not a universal semantic module or cross-model invariant.",
        ],
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "auditor": file_sha256(AUDIT_SCRIPT),
        },
        "model_weights_loaded": False,
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def build(force: bool) -> None:
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists; refusing to overwrite a frozen phase")
    if force and OUT.exists():
        import shutil
        shutil.rmtree(OUT)

    from model_utils import MODEL_CONFIGS
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[MODEL_NAME]["path"], trust_remote_code=True, local_files_only=True)
    rows, token_audit = build_cases(tokenizer)
    write_jsonl(MATERIAL, rows)
    review = naturalness_review(rows)
    atomic_json(NATURALNESS, review)
    program = program_audit(rows)
    machine = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": utc_now(),
        "token_audit": token_audit,
        "program_audit": program,
        "semantic_unique": review["semantic_unique"],
        "naturalness_flags": review["flags"],
        "all_machine_checks_passed": (
            review["semantic_unique"]
            and token_audit["all_candidates_single_token"]
            and program["shortcut_ceiling"] <= THRESHOLDS["shortcut_program_accuracy_max"]
            and program["active_same_bag_different_gold_pairs"] == program["active_pair_count"]
        ),
    }
    atomic_json(MACHINE_AUDIT, machine)
    atomic_json(ENVIRONMENT, environment_snapshot())
    protocol = build_protocol(rows, token_audit, program, review)
    atomic_json(PROTOCOL, protocol)
    print(canonical_json({
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "cases": len(rows),
        "selected_object": protocol["selected_object"],
        "shortcut_ceiling": program["shortcut_ceiling"],
        "protocol_digest": protocol["protocol_digest"],
    }))


def finalize() -> None:
    protocol = read_json(PROTOCOL)
    audit = read_json(INDEPENDENT_AUDIT)
    if not audit.get("all_checks_passed"):
        raise RuntimeError("independent Phase1292 audit did not pass")
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "verdict": "object_converged_contract_frozen_and_independently_audited",
        "selected_object": protocol["selected_object"],
        "protocol_digest": protocol["protocol_digest"],
        "material_sha256": protocol["material"]["material_sha256"],
        "model_weights_loaded": False,
        "audit_passed": True,
        "authorization": "phase1293_qwen3_behavior_only",
    }
    atomic_json(FINAL, final)
    print(canonical_json(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "finalize"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        build(args.force)
    else:
        finalize()


if __name__ == "__main__":
    main()
