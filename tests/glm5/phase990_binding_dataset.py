#!/usr/bin/env python3
"""Build the CPU-only Phase 990 two-hop binding corpus.

The generator first creates an abstract graph and then renders a complete
4 semantic transforms x 2 paraphrases x 2 orders x 2 horizons factor grid.
No tokenizer, model weight, torch, or CUDA runtime is used in this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import itertools
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping

import phase990_binding_core as core


_PERMUTATIONS = tuple(itertools.permutations(range(4)))
_SPLIT_OFFSETS: dict[str, int] = {}
_offset = 0
for _split, _count in core.SPLIT_COUNTS.items():
    _SPLIT_OFFSETS[_split] = _offset
    _offset += _count

_STRUCTURAL_REP_OFFSETS = {
    "discovery": 0,
    "confirmation": 3,
    "adversarial": 6,
    "sealed_holdout": 8,
}

_FILLERS = (
    ("filler_0", "The archive clock showed noon."),
    ("filler_1", "A clerk checked the window latch."),
    ("filler_2", "The reading room remained quiet."),
    ("filler_3", "A blank form rested on the desk."),
    ("filler_4", "The hallway lamp stayed on."),
    ("filler_5", "A visitor returned an empty folder."),
    ("filler_6", "The wall calendar showed no appointment."),
    ("filler_7", "A clean cup stood beside the door."),
)
_FILL_PERMUTATIONS = tuple(itertools.permutations(range(5)))
_ENTITY_LEXICON = core.PEOPLE[:8]
_OBJECT_LEXICON = core.OBJECTS[:4]
# Frozen entity-name assignments, indexed by split ordinal and entity slot.
# The query-name constraint is analytic; the remaining names were selected by
# an exact finite assignment solver before sealing.  The dataset audit does not
# trust this table: it recomputes every name x relation, name x answer, and
# owner-name x object x answer cell for all four semantic transforms.
_ENTITY_ASSIGNMENT_CODES = {
    "discovery": (
        "0631 1264 0356 1472 1257 0264 1257 0364 3217 1325 5230 2346 "
        "4370 0237 1374 3245 1247 4153 2741 7053 6354 1740 0456 0542 "
        "1426 4567 0316 6027 2507 2416 0157 4716 0635 1342 0632 1237 "
        "1674 0723 1645 0563 7215 5370 0247 1305 7350 7261 0314 1206 "
        "2643 2053 1743 0457 2654 6741 1256 2741 1046 4207 7016 5127 "
        "1207 7106 4637 0156 0573 1635 0627 1246 1650 0534 1354 0517 "
        "3201 7356 1263 6320 0361 6215 0357 7243 3745 7053 3246 6351 "
        "3254 5140 2054 6543 5046 3067 7256 4157 6427 5246 5247 0736"
    ).split(),
    "confirmation": (
        "0165 1457 0653 1724 1274 0361 1075 0713 3260 4327 6210 7316 "
        "7364 5206 6340 1254 6540 0253 5742 4657 7653 6045 0452 5742 "
        "2706 4257 0316 4307 2307 2306 4317 2046 0153 1475 0652 1240 "
        "1043 0316 1724 0356 7254 6370 1267 5306 7304 4210 0345 7206 "
        "1643 7251 6241 0652 2051 1046 2057 1743 0316 2457 5316 0217 "
        "0517 5146 2507 4536 0715 1635 0427 1452 1642 0576 1635 0156 "
        "1256 6372 1274 7365 5342 7213 6307 4263 5642 4251 3741 2357 "
        "1752 0743 3154 0143 7436 1347 3526 2317 3427 0326 3647 5106"
    ).split(),
    "adversarial": (
        "0375 1724 0637 1274 1524 0135 1254 0263 5270 4316 6257 4361 "
        "6301 3201 0316 4250 2746 6350 1247 3750 3156 0741 0156 1742 "
        "1526 0247 7436 5067 3407 7546 3027 2516 0572 1436 0572 1364 "
        "1674 0275 1543 0723 7243 4365 3241 5306 4365 6271 6350 7241 "
        "6143 2750 1346 7250 0152 0346 2057 3146 3026 5217 5406 4517 "
        "0147 3526 1427 3506"
    ).split(),
    "sealed_holdout": (
        "0536 1574 0456 1467 1724 0713 1274 0534 1275 1356 7210 1362 "
        "4310 7205 4361 5276 3047 0653 2143 0356 3056 2741 0356 2741 "
        "2346 6027 3746 4207 0257 3416 0537 3546 0527 1634 0257 1043 "
        "1643 0275 1564 0275 1276 2356 1275 5306 2316 0234 2306 7214 "
        "6541 4052 1643 1352 2750 6741 2750 6741 0346 4127 5306 2107 "
        "1457 5036 2417 0536"
    ).split(),
}
_ENTITY_SPLIT_PERMUTATIONS = {
    "discovery": (0, 1, 2, 3, 4, 5, 6, 7),
    "confirmation": (0, 1, 2, 3, 4, 5, 6, 7),
    # Frozen split permutation removes four otherwise model-visible semantic
    # collisions without changing any exact lexical balance marginal.
    "adversarial": (0, 1, 2, 3, 4, 6, 7, 5),
    "sealed_holdout": (0, 1, 2, 3, 4, 5, 6, 7),
}
_ORDER_FILL_INDEX = {
    "discovery": {
        "order_a": [100,7,95,18,12,52,50,4,81,82,73,85,33,106,46,31,20,63,63,69,56,89,47,109,104,37,91,37,2,114,66,100,80,73,23,114,82,92,105,75,14,115,69,1,17,33,24,112,94,70,70,10,74,61,106,50,11,92,110,60,70,91,106,29,5,36,0,10,64,104,100,34,40,51,68,5,19,6,109,89,59,32,101,32,102,63,20,95,67,109,73,48,28,6,8,54],
        "order_b": [101,28,12,32,113,101,82,51,8,23,90,75,19,44,23,116,73,21,61,52,27,93,8,87,45,104,38,4,32,28,10,83,17,68,38,95,108,6,57,22,27,48,53,78,33,37,56,31,1,59,81,46,8,68,44,44,21,13,50,96,21,60,22,34,76,106,43,85,67,30,69,42,62,27,26,10,41,40,20,78,93,107,56,2,42,41,16,75,92,76,26,48,36,99,40,52],
    },
    "confirmation": {
        "order_a": [44,72,85,7,113,113,27,110,49,15,66,7,25,110,37,94,74,91,4,19,112,68,61,66,6,66,54,60,89,24,15,70,103,108,3,116,61,108,85,11,21,116,119,9,40,54,116,9,85,14,105,64,15,39,48,0,16,48,33,107,1,27,51,90,91,106,72,14,14,28,52,7,6,111,6,80,105,30,18,76,69,29,46,105,74,114,59,71,113,72,54,51,110,31,27,67],
        "order_b": [57,27,34,95,1,59,31,63,92,76,58,9,37,68,32,73,98,23,12,4,117,76,4,107,31,66,24,11,35,41,46,21,29,25,109,75,84,13,51,3,61,3,73,23,93,32,58,66,117,1,7,68,43,31,27,3,16,95,99,16,85,22,27,28,18,101,23,32,43,21,32,43,117,74,47,82,49,14,86,47,73,32,82,65,56,41,116,19,95,63,73,75,97,48,37,8],
    },
    "adversarial": {
        "order_a": [39,111,63,108,44,14,88,0,53,16,44,96,27,2,33,90,28,68,117,1,56,84,71,13,31,102,22,28,78,119,21,69,30,102,97,81,66,106,37,52,61,55,83,38,68,28,94,6,109,75,71,51,94,40,94,115,90,3,37,89,81,27,10,92],
        "order_b": [53,87,58,3,49,47,115,21,110,100,58,41,33,76,67,74,108,88,54,22,84,32,0,13,3,10,41,55,41,25,40,3,27,29,11,114,42,102,75,53,88,97,99,118,9,84,101,26,34,83,37,119,97,53,46,65,22,6,74,1,23,99,104,68],
    },
    "sealed_holdout": {
        "order_a": [100,102,111,69,100,55,7,87,6,62,98,102,7,61,47,23,119,3,8,19,78,102,31,4,96,79,16,42,28,109,19,30,48,42,15,84,14,42,3,57,25,57,53,43,118,30,76,93,73,23,17,113,54,52,73,102,35,7,58,95,37,26,87,112],
        "order_b": [82,101,0,100,5,80,44,25,32,87,53,37,0,114,85,37,37,113,18,82,58,12,17,92,79,32,45,44,22,10,89,20,68,90,59,97,54,89,69,116,76,62,97,32,46,96,40,26,23,11,72,9,0,47,1,35,47,20,87,91,91,27,55,74],
    },
}
_WORD_RE = re.compile(r"[A-Za-z]+|\d+|[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def variant_id(
    semantic_transform: str,
    paraphrase_id: str,
    fact_order_id: str,
    horizon_id: str,
) -> str:
    value = (
        f"{semantic_transform}__{paraphrase_id}__"
        f"{fact_order_id}__{horizon_id}"
    )
    core.require(value in core.VARIANTS, f"invalid variant id: {value}")
    return value


def parse_variant(value: str) -> dict[str, str]:
    parts = value.split("__")
    core.require(len(parts) == 4, f"invalid variant: {value}")
    semantic, paraphrase, order, horizon = parts
    core.require(semantic in core.SEMANTIC_TRANSFORMS, "bad semantic transform")
    core.require(paraphrase in core.PARAPHRASE_IDS, "bad paraphrase id")
    core.require(order in core.FACT_ORDER_IDS, "bad order id")
    core.require(horizon in core.HORIZON_IDS, "bad horizon id")
    return {
        "semantic_transform": semantic,
        "paraphrase_id": paraphrase,
        "fact_order_id": order,
        "horizon_id": horizon,
    }


def _world_id(split: str, ordinal: int) -> str:
    return f"p990_w_{core.opaque_id('world-v2', split, ordinal)}"


def _item_id(world_id: str, variant: str) -> str:
    return f"p990_i_{core.opaque_id('item-v2', world_id, variant)}"


def _root_family_id(split: str, ordinal: int) -> str:
    return f"p990_f_{core.opaque_id('family-v2', split, ordinal)}"


def _abstract_attribute_slots(
    query_relation_slot: int,
    answer_value_slot: int,
    target_object_slot: int,
) -> list[list[int]]:
    values: list[list[int]] = [[0] * 4 for _ in range(2)]
    for object_slot in range(4):
        relative = (object_slot - target_object_slot) % 4
        values[query_relation_slot][object_slot] = (
            answer_value_slot + relative
        ) % 4
        values[1 - query_relation_slot][object_slot] = (
            answer_value_slot + relative + 1
        ) % 4
    return values


def _abstract_closure_hashes(
    query_entity_slot: int,
    query_relation_slot: int,
    target_object_slot: int,
    owner_permutation: tuple[int, ...],
    attribute_value_slots: list[list[int]],
) -> frozenset[str]:
    results: set[str] = set()
    for semantic_transform in core.SEMANTIC_TRANSFORMS:
        owner = list(owner_permutation)
        attributes = [row[:] for row in attribute_value_slots]
        relation = query_relation_slot
        if semantic_transform == "value_swap":
            partner = (target_object_slot + 2) % 4
            attributes[relation][target_object_slot], attributes[relation][partner] = (
                attributes[relation][partner],
                attributes[relation][target_object_slot],
            )
        elif semantic_transform == "binding_swap":
            partner_entity = owner.index((target_object_slot + 3) % 4)
            owner[query_entity_slot], owner[partner_entity] = (
                owner[partner_entity], owner[query_entity_slot]
            )
        elif semantic_transform == "relation_swap":
            relation = 1 - relation
        results.add(core.sha256_json({
            "owner_permutation": owner,
            "attribute_value_slots": attributes,
            "query_entity_slot": query_entity_slot,
            "query_relation_slot": relation,
        }))
    core.require(
        len(results) == len(core.SEMANTIC_TRANSFORMS),
        "abstract counterfactual closure collapsed",
    )
    return frozenset(results)


@lru_cache(maxsize=1)
def _structural_schedule() -> dict[
    tuple[str, int, int, int, int], tuple[int, tuple[int, ...], str]
]:
    """Select disjoint abstract counterfactual closures for every split.

    Candidate closures form eight connected components for each q/r pair.
    Two whole components are assigned to each split.  A small exact
    backtracking step then chooses the required repetitions without reusing
    any abstract semantic state inside that split.  Because components are
    never shared, abstract closure leakage across splits is impossible by
    construction and is independently rechecked by the dataset audit.
    """
    schedule: dict[
        tuple[str, int, int, int, int], tuple[int, tuple[int, ...], str]
    ] = {}
    split_index = {name: index for index, name in enumerate(core.SPLIT_ORDER)}

    for query_entity_slot in range(4):
        for query_relation_slot in range(2):
            candidates: list[dict[str, Any]] = []
            for answer_value_slot in range(4):
                for target_object_slot in range(4):
                    for owner_permutation in _PERMUTATIONS:
                        if owner_permutation[query_entity_slot] != target_object_slot:
                            continue
                        attributes = _abstract_attribute_slots(
                            query_relation_slot,
                            answer_value_slot,
                            target_object_slot,
                        )
                        closure = _abstract_closure_hashes(
                            query_entity_slot,
                            query_relation_slot,
                            target_object_slot,
                            owner_permutation,
                            attributes,
                        )
                        candidates.append({
                            "answer": answer_value_slot,
                            "target": target_object_slot,
                            "owner": owner_permutation,
                            "owner_index": _PERMUTATIONS.index(owner_permutation),
                            "closure": closure,
                        })

            parents = list(range(len(candidates)))

            def find(index: int) -> int:
                while parents[index] != index:
                    parents[index] = parents[parents[index]]
                    index = parents[index]
                return index

            def union(left: int, right: int) -> None:
                left_root = find(left)
                right_root = find(right)
                if left_root != right_root:
                    parents[right_root] = left_root

            state_owner: dict[str, int] = {}
            for index, candidate in enumerate(candidates):
                for state_hash in candidate["closure"]:
                    if state_hash in state_owner:
                        union(index, state_owner[state_hash])
                    else:
                        state_owner[state_hash] = index

            components_by_root: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for index, candidate in enumerate(candidates):
                components_by_root[find(index)].append(candidate)
            components = list(components_by_root.values())
            components.sort(key=lambda component: core.sha256_json(sorted(
                state_hash
                for candidate in component
                for state_hash in candidate["closure"]
            )))
            core.require(len(components) == 8, "abstract component count changed")

            for split in core.SPLIT_ORDER:
                repetitions = core.SPLIT_COUNTS[split] // 32
                component_offset = 2 * split_index[split]
                selected_components = components[
                    component_offset:component_offset + 2
                ]
                component_id = core.sha256_json(sorted(
                    state_hash
                    for component in selected_components
                    for candidate in component
                    for state_hash in candidate["closure"]
                ))
                pool = [
                    candidate
                    for component in selected_components
                    for candidate in component
                ]
                by_answer = {
                    answer: sorted(
                        [candidate for candidate in pool
                         if candidate["answer"] == answer],
                        key=lambda candidate: (
                            candidate["target"], candidate["owner_index"]
                        ),
                    )
                    for answer in range(4)
                }
                slots = [
                    answer
                    for answer in range(4)
                    for _ in range(repetitions)
                ]

                def choose(
                    slot_index: int,
                    used_hashes: frozenset[str],
                    chosen: list[dict[str, Any]],
                ) -> list[dict[str, Any]] | None:
                    if slot_index == len(slots):
                        return list(chosen)
                    answer = slots[slot_index]
                    already = [
                        candidate for candidate in chosen
                        if candidate["answer"] == answer
                    ]
                    for candidate in by_answer[answer]:
                        if candidate in chosen:
                            continue
                        if candidate["closure"] & used_hashes:
                            continue
                        if already and (
                            candidate["target"], candidate["owner_index"]
                        ) <= (
                            already[-1]["target"], already[-1]["owner_index"]
                        ):
                            continue
                        chosen.append(candidate)
                        result = choose(
                            slot_index + 1,
                            used_hashes | candidate["closure"],
                            chosen,
                        )
                        if result is not None:
                            return result
                        chosen.pop()
                    return None

                chosen = choose(0, frozenset(), [])
                core.require(chosen is not None, "abstract schedule has no solution")
                for answer in range(4):
                    answer_candidates = [
                        candidate for candidate in chosen
                        if candidate["answer"] == answer
                    ]
                    core.require(
                        len(answer_candidates) == repetitions,
                        "abstract answer repetition count changed",
                    )
                    for local_rep, candidate in enumerate(answer_candidates):
                        schedule[(
                            split,
                            query_entity_slot,
                            query_relation_slot,
                            answer,
                            local_rep,
                        )] = (
                            int(candidate["target"]),
                            tuple(candidate["owner"]),
                            component_id,
                        )

    core.require(
        len(schedule) == core.EXPECTED_WORLD_COUNT,
        "abstract structural schedule size changed",
    )
    return schedule


def build_base_world(split: str, ordinal: int) -> dict[str, Any]:
    core.require(split in core.SPLIT_COUNTS, f"unknown split: {split}")
    core.require(
        isinstance(ordinal, int)
        and not isinstance(ordinal, bool)
        and 0 <= ordinal < core.SPLIT_COUNTS[split],
        f"invalid ordinal for {split}: {ordinal}",
    )
    global_index = _SPLIT_OFFSETS[split] + ordinal
    local_rep = ordinal // 32
    structural_rep = _STRUCTURAL_REP_OFFSETS[split] + local_rep
    balance_cell = ordinal % 32
    answer_value_slot = balance_cell % 4
    query_relation_slot = (balance_cell // 4) % 2
    query_entity_slot = (balance_cell // 8) % 4

    target_object_slot, selected_owner, structural_component_id = (
        _structural_schedule()[
            (
                split,
                query_entity_slot,
                query_relation_slot,
                answer_value_slot,
                local_rep,
            )
        ]
    )
    owner_permutation = list(selected_owner)

    assignment_code = _ENTITY_ASSIGNMENT_CODES[split][ordinal]
    entity_name_indices = [int(character) for character in assignment_code]
    raw_expected_query_name_index = (
        2 * query_entity_slot
        + (query_relation_slot + answer_value_slot) % 2
    )
    permutation = _ENTITY_SPLIT_PERMUTATIONS[split]
    entity_name_indices = [permutation[index] for index in entity_name_indices]
    expected_query_name_index = permutation[raw_expected_query_name_index]
    core.require(
        len(entity_name_indices) == 4
        and len(set(entity_name_indices)) == 4
        and all(0 <= index < len(_ENTITY_LEXICON)
                for index in entity_name_indices),
        "entity assignment code",
    )
    core.require(
        entity_name_indices[query_entity_slot] == expected_query_name_index,
        "query entity lexical orthogonality",
    )
    entities = [_ENTITY_LEXICON[index] for index in entity_name_indices]

    target_object_name_index = (
        query_entity_slot
        + 2 * query_relation_slot
        + local_rep
        + 5 * core.SPLIT_ORDER.index(split)
    ) % len(_OBJECT_LEXICON)
    objects: list[str | None] = [None] * 4
    for relative_slot in range(4):
        objects[(target_object_slot + relative_slot) % 4] = _OBJECT_LEXICON[
            (target_object_name_index + relative_slot)
            % len(_OBJECT_LEXICON)
        ]
    core.require(all(isinstance(value, str) for value in objects), "object schedule")

    attribute_value_slots = _abstract_attribute_slots(
        query_relation_slot,
        answer_value_slot,
        target_object_slot,
    )

    owner_edges = [
        {
            "fact_id": f"owner_e{entity_slot}",
            "entity_slot": entity_slot,
            "entity": entities[entity_slot],
            "relation": core.OWNER_RELATION,
            "object_slot": owner_permutation[entity_slot],
            "object": objects[owner_permutation[entity_slot]],
        }
        for entity_slot in range(4)
    ]
    attribute_edges = [
        {
            "fact_id": f"attribute_r{relation_slot}_o{object_slot}",
            "object_slot": object_slot,
            "object": objects[object_slot],
            "relation_slot": relation_slot,
            "relation": core.ATTRIBUTE_RELATIONS[relation_slot],
            "value_slot": attribute_value_slots[relation_slot][object_slot],
            "value": core.VALUES[
                attribute_value_slots[relation_slot][object_slot]
            ],
        }
        for relation_slot in range(2)
        for object_slot in range(4)
    ]
    value_partner_object_slot = (target_object_slot + 2) % 4
    binding_partner_object_slot = (target_object_slot + 3) % 4
    binding_partner_entity_slot = owner_permutation.index(
        binding_partner_object_slot
    )
    world = {
        "root_family_id": _root_family_id(split, ordinal),
        "semantic_world_id": _world_id(split, ordinal),
        "split": split,
        "split_ordinal": ordinal,
        "global_ordinal": global_index,
        "local_rep": local_rep,
        "structural_rep": structural_rep,
        "structural_component_id": structural_component_id,
        "seed_key": f"p990:{split}:{ordinal}:abstract-v2",
        "entities": entities,
        "objects": objects,
        "values": list(core.VALUES),
        "owner_edges": owner_edges,
        "attribute_edges": attribute_edges,
        "query": {
            "entity_slot": query_entity_slot,
            "entity": entities[query_entity_slot],
            "relation_slot": query_relation_slot,
            "relation": core.ATTRIBUTE_RELATIONS[query_relation_slot],
        },
        "counterfactual_partners": {
            "value_partner_object_slot": value_partner_object_slot,
            "binding_partner_entity_slot": binding_partner_entity_slot,
            "binding_partner_object_slot": binding_partner_object_slot,
        },
        "base_target_object_slot": target_object_slot,
        "base_query_entity_slot": query_entity_slot,
        "base_query_relation_slot": query_relation_slot,
        "base_answer_value_slot": answer_value_slot,
        "independent_unit": True,
        "factor_grid_rows_are_paired": True,
    }
    _validate_graph(world)
    solution = solve_state(world)
    core.require(
        solution["answer_value_slot"] == answer_value_slot,
        "constructed answer does not match balance cell",
    )
    return world


def build_worlds() -> list[dict[str, Any]]:
    return [
        build_base_world(split, ordinal)
        for split in core.SPLIT_ORDER
        for ordinal in range(core.SPLIT_COUNTS[split])
    ]


def _validate_graph(state: Mapping[str, Any]) -> None:
    entities = state["entities"]
    objects = state["objects"]
    values = state["values"]
    core.require(
        isinstance(entities, list) and len(entities) == 4,
        "ORV_GRAPH_001 entities",
    )
    core.require(
        isinstance(objects, list) and len(objects) == 4,
        "ORV_GRAPH_001 objects",
    )
    core.require(
        isinstance(values, list) and values == list(core.VALUES),
        "ORV_GRAPH_001 values",
    )
    core.require(all(isinstance(value, str) and value for value in entities),
                 "ORV_GRAPH_001 entity strings")
    core.require(all(isinstance(value, str) and value for value in objects),
                 "ORV_GRAPH_001 object strings")
    core.require(len(set(entities)) == 4, "ORV_GRAPH_001 entity uniqueness")
    core.require(len(set(objects)) == 4, "ORV_GRAPH_001 object uniqueness")
    core.require(len(set(values)) == 4, "ORV_GRAPH_001 value uniqueness")
    core.require(
        not (set(entities) & set(objects) or set(entities) & set(values)
             or set(objects) & set(values)),
        "ORV_GRAPH_001 vocabularies overlap",
    )

    owner_edges = state["owner_edges"]
    core.require(len(owner_edges) == 4, "ORV_GRAPH_002 owner edge count")
    core.require(
        sorted(int(edge["entity_slot"]) for edge in owner_edges)
        == list(range(4)),
        "ORV_GRAPH_002 entity completeness",
    )
    owner_slots = [int(edge["object_slot"]) for edge in owner_edges]
    core.require(sorted(owner_slots) == list(range(4)), "ORV_GRAPH_002 owner bijection")
    for edge in owner_edges:
        entity_slot = int(edge["entity_slot"])
        object_slot = int(edge["object_slot"])
        core.require(edge["fact_id"] == f"owner_e{entity_slot}",
                     "ORV_GRAPH_002 owner fact id")
        core.require(edge["entity"] == entities[entity_slot],
                     "ORV_GRAPH_002 entity text/slot mismatch")
        core.require(edge["object"] == objects[object_slot],
                     "ORV_GRAPH_002 object text/slot mismatch")
        core.require(edge["relation"] == core.OWNER_RELATION,
                     "ORV_GRAPH_002 owner relation changed")

    attribute_edges = state["attribute_edges"]
    core.require(len(attribute_edges) == 8, "ORV_GRAPH_003 attribute edge count")
    for relation_slot in range(2):
        selected = [
            edge for edge in attribute_edges
            if int(edge["relation_slot"]) == relation_slot
        ]
        core.require(len(selected) == 4, "ORV_GRAPH_003 relation completeness")
        core.require(
            sorted(edge["object_slot"] for edge in selected) == list(range(4)),
            "ORV_GRAPH_003 object completeness",
        )
        core.require(
            sorted(edge["value_slot"] for edge in selected) == list(range(4)),
            "ORV_GRAPH_003 value bijection",
        )
        for edge in selected:
            object_slot = int(edge["object_slot"])
            value_slot = int(edge["value_slot"])
            core.require(
                edge["fact_id"]
                == f"attribute_r{relation_slot}_o{object_slot}",
                "ORV_GRAPH_003 attribute fact id",
            )
            core.require(edge["object"] == objects[object_slot],
                         "ORV_GRAPH_003 object text/slot mismatch")
            core.require(edge["relation"] == core.ATTRIBUTE_RELATIONS[relation_slot],
                         "ORV_GRAPH_003 relation text/slot mismatch")
            core.require(edge["value"] == values[value_slot],
                         "ORV_GRAPH_003 value text/slot mismatch")
    lookup = {
        (int(edge["relation_slot"]), int(edge["object_slot"])):
        int(edge["value_slot"])
        for edge in attribute_edges
    }
    for object_slot in range(4):
        core.require(
            lookup[(0, object_slot)] != lookup[(1, object_slot)],
            "ORV_GRAPH_004 relations must disagree per object",
        )
    query = state["query"]
    query_entity = int(query["entity_slot"])
    query_relation = int(query["relation_slot"])
    core.require(query_entity in range(4) and query_relation in range(2),
                 "ORV_GRAPH_006 query slot")
    core.require(query["entity"] == entities[query_entity],
                 "ORV_GRAPH_006 query entity mismatch")
    core.require(query["relation"] == core.ATTRIBUTE_RELATIONS[query_relation],
                 "ORV_GRAPH_006 query relation mismatch")
    fact_ids = [edge["fact_id"] for edge in [*owner_edges, *attribute_edges]]
    core.require(len(fact_ids) == len(set(fact_ids)) == 12,
                 "ORV_GRAPH_006 fact ids are not unique")


def solve_state(state: Mapping[str, Any]) -> dict[str, Any]:
    query_entity_slot = int(state["query"]["entity_slot"])
    query_relation_slot = int(state["query"]["relation_slot"])
    owner_matches = [
        edge for edge in state["owner_edges"]
        if int(edge["entity_slot"]) == query_entity_slot
    ]
    core.require(len(owner_matches) == 1, "ORV_GRAPH_006 owner path not unique")
    owner_edge = owner_matches[0]
    object_slot = int(owner_edge["object_slot"])
    attribute_matches = [
        edge for edge in state["attribute_edges"]
        if int(edge["object_slot"]) == object_slot
        and int(edge["relation_slot"]) == query_relation_slot
    ]
    core.require(
        len(attribute_matches) == 1,
        "ORV_GRAPH_006 attribute path not unique",
    )
    attribute_edge = attribute_matches[0]
    return {
        "answer_object_slot": object_slot,
        "answer_object": str(owner_edge["object"]),
        "answer_value_slot": int(attribute_edge["value_slot"]),
        "answer_value": str(attribute_edge["value"]),
        "owner_fact_id": str(owner_edge["fact_id"]),
        "attribute_fact_id": str(attribute_edge["fact_id"]),
    }


def _gold_path_deletion_recoverability(
    state: Mapping[str, Any],
) -> dict[str, bool]:
    """Measure structural recovery shortcuts after deleting gold path facts."""
    solution = solve_state(state)
    query_entity = int(state["query"]["entity_slot"])
    query_relation = int(state["query"]["relation_slot"])
    target_object = int(solution["answer_object_slot"])
    target_value = int(solution["answer_value_slot"])

    remaining_owner = [
        edge for edge in state["owner_edges"]
        if edge["fact_id"] != solution["owner_fact_id"]
    ]
    remaining_attributes = [
        edge for edge in state["attribute_edges"]
        if edge["fact_id"] != solution["attribute_fact_id"]
        and int(edge["relation_slot"]) == query_relation
    ]
    missing_entities = set(range(4)) - {
        int(edge["entity_slot"]) for edge in remaining_owner
    }
    missing_objects_from_owner = set(range(4)) - {
        int(edge["object_slot"]) for edge in remaining_owner
    }
    missing_objects_from_relation = set(range(4)) - {
        int(edge["object_slot"]) for edge in remaining_attributes
    }
    missing_values_from_relation = set(range(4)) - {
        int(edge["value_slot"]) for edge in remaining_attributes
    }
    owner_recoverable = (
        missing_entities == {query_entity}
        and missing_objects_from_owner == {target_object}
    )
    attribute_recoverable = (
        missing_objects_from_relation == {target_object}
        and missing_values_from_relation == {target_value}
    )
    return {
        "owner_fact_deleted_recoverable": owner_recoverable,
        "attribute_fact_deleted_recoverable": attribute_recoverable,
        "both_gold_facts_deleted_recoverable": (
            owner_recoverable and attribute_recoverable
        ),
    }


def transform_state(
    base_world: Mapping[str, Any],
    semantic_transform: str,
) -> dict[str, Any]:
    core.require(
        semantic_transform in core.SEMANTIC_TRANSFORMS,
        f"invalid semantic transform: {semantic_transform}",
    )
    state = deepcopy(dict(base_world))
    solution = solve_state(state)
    query_relation_slot = int(state["query"]["relation_slot"])

    if semantic_transform == "value_swap":
        target_object_slot = int(solution["answer_object_slot"])
        partner_object_slot = int(
            state["counterfactual_partners"]["value_partner_object_slot"]
        )
        selected = {
            int(edge["object_slot"]): edge
            for edge in state["attribute_edges"]
            if int(edge["relation_slot"]) == query_relation_slot
            and int(edge["object_slot"]) in {
                target_object_slot, partner_object_slot
            }
        }
        core.require(len(selected) == 2, "value swap endpoints missing")
        left = selected[target_object_slot]
        right = selected[partner_object_slot]
        left["value_slot"], right["value_slot"] = (
            right["value_slot"], left["value_slot"]
        )
        left["value"], right["value"] = right["value"], left["value"]
    elif semantic_transform == "binding_swap":
        query_entity_slot = int(state["query"]["entity_slot"])
        partner_entity_slot = int(
            state["counterfactual_partners"]["binding_partner_entity_slot"]
        )
        selected = {
            int(edge["entity_slot"]): edge
            for edge in state["owner_edges"]
            if int(edge["entity_slot"]) in {
                query_entity_slot, partner_entity_slot
            }
        }
        core.require(len(selected) == 2, "binding swap endpoints missing")
        left = selected[query_entity_slot]
        right = selected[partner_entity_slot]
        left["object_slot"], right["object_slot"] = (
            right["object_slot"], left["object_slot"]
        )
        left["object"], right["object"] = right["object"], left["object"]
    elif semantic_transform == "relation_swap":
        state["query"]["relation_slot"] = 1 - query_relation_slot
        state["query"]["relation"] = core.ATTRIBUTE_RELATIONS[
            int(state["query"]["relation_slot"])
        ]

    _validate_graph(state)
    return state


def _state_semantics(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entities": list(state["entities"]),
        "objects": list(state["objects"]),
        "values": list(state["values"]),
        "owner_edges": sorted(
            [
                {
                    "entity_slot": edge["entity_slot"],
                    "entity": edge["entity"],
                    "object_slot": edge["object_slot"],
                    "object": edge["object"],
                }
                for edge in state["owner_edges"]
            ],
            key=lambda value: value["entity_slot"],
        ),
        "attribute_edges": sorted(
            [
                {
                    "object_slot": edge["object_slot"],
                    "object": edge["object"],
                    "relation_slot": edge["relation_slot"],
                    "relation": edge["relation"],
                    "value_slot": edge["value_slot"],
                    "value": edge["value"],
                }
                for edge in state["attribute_edges"]
            ],
            key=lambda value: (value["relation_slot"], value["object_slot"]),
        ),
        "query": {
            "entity_slot": state["query"]["entity_slot"],
            "entity": state["query"]["entity"],
            "relation_slot": state["query"]["relation_slot"],
            "relation": state["query"]["relation"],
        },
    }


def _observable_semantics(state: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical semantic content available in the rendered task text."""
    return {
        "owner_edges": sorted(
            (str(edge["entity"]), str(edge["object"]))
            for edge in state["owner_edges"]
        ),
        "attribute_edges": sorted(
            (str(edge["object"]), str(edge["relation"]), str(edge["value"]))
            for edge in state["attribute_edges"]
        ),
        "query": (
            str(state["query"]["entity"]),
            str(state["query"]["relation"]),
        ),
    }


def _slot_semantics(state: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical abstract state with every lexical realization removed."""
    return {
        "owner_permutation": [
            int(edge["object_slot"])
            for edge in sorted(
                state["owner_edges"], key=lambda edge: int(edge["entity_slot"])
            )
        ],
        "attribute_value_slots": [
            [
                int(next(
                    edge["value_slot"] for edge in state["attribute_edges"]
                    if int(edge["relation_slot"]) == relation
                    and int(edge["object_slot"]) == obj
                ))
                for obj in range(4)
            ]
            for relation in range(2)
        ],
        "query_entity_slot": int(state["query"]["entity_slot"]),
        "query_relation_slot": int(state["query"]["relation_slot"]),
    }


def _apply_transform_twice(
    world: Mapping[str, Any],
    semantic_transform: str,
) -> bool:
    first = transform_state(world, semantic_transform)
    second = transform_state(first, semantic_transform)
    return _state_semantics(second) == _state_semantics(world)


def _owner_text(edge: Mapping[str, Any], paraphrase: bool) -> str:
    if not paraphrase:
        return f"{edge['entity']} is paired with the {edge['object']}."
    return f"The {edge['object']} is assigned to {edge['entity']}."


def _attribute_text(edge: Mapping[str, Any], paraphrase: bool) -> str:
    if not paraphrase:
        return (
            f"The {edge['object']}'s {edge['relation']} marker "
            f"is {edge['value']}."
        )
    return (
        f"{edge['value'].capitalize()} marks the {edge['relation']} side "
        f"of the {edge['object']}."
    )


def _query_text(query: Mapping[str, Any], paraphrase: bool) -> str:
    if not paraphrase:
        return (
            f"What is the {query['relation']} marker of the object "
            f"paired with {query['entity']}?"
        )
    return (
        f"Identify the marker on the {query['relation']} side of "
        f"{query['entity']}'s assigned object."
    )


def _core_fact_records(
    state: Mapping[str, Any],
    paraphrase_id: str,
) -> list[dict[str, str]]:
    paraphrase = paraphrase_id == "paraphrase"
    records = [
        {
            "fact_id": str(edge["fact_id"]),
            "kind": "owner",
            "text": _owner_text(edge, paraphrase),
        }
        for edge in state["owner_edges"]
    ]
    records.extend(
        {
            "fact_id": str(edge["fact_id"]),
            "kind": "attribute",
            "text": _attribute_text(edge, paraphrase),
        }
        for edge in state["attribute_edges"]
    )
    return records


def _order_core_facts(
    records: list[dict[str, str]],
    state: Mapping[str, Any],
    fact_order_id: str,
) -> list[dict[str, str]]:
    core.require(fact_order_id in core.FACT_ORDER_IDS, "unknown fact order")
    owners = sorted(
        [record for record in records if record["kind"] == "owner"],
        key=lambda record: record["fact_id"],
    )
    attributes = {
        record["fact_id"]: record
        for record in records if record["kind"] == "attribute"
    }
    core.require(len(owners) == 4 and len(attributes) == 8, "fact inventory")

    query_entity = int(state["base_query_entity_slot"])
    query_relation = int(state["base_query_relation_slot"])
    target_object = int(state["base_target_object_slot"])
    local_rep = int(state["local_rep"])
    base_position = (
        2 * query_entity + query_relation + 4 * local_rep
    ) % 8
    source_fact_ids = (
        f"attribute_r{query_relation}_o{target_object}",
        f"attribute_r{query_relation}_o{(target_object + 3) % 4}",
        f"attribute_r{1 - query_relation}_o{target_object}",
    )
    offsets = (0, 1, 2) if fact_order_id == "order_a" else (3, 5, 7)
    attribute_order: list[dict[str, str] | None] = [None] * 8
    for fact_id, offset in zip(source_fact_ids, offsets, strict=True):
        position = (base_position + offset) % 8
        core.require(attribute_order[position] is None, "source position collision")
        attribute_order[position] = attributes[fact_id]
    remaining = [
        attributes[fact_id] for fact_id in sorted(attributes)
        if fact_id not in source_fact_ids
    ]
    split = str(state["split"])
    split_ordinal = int(state["split_ordinal"])
    fill_index = _ORDER_FILL_INDEX[split][fact_order_id][split_ordinal]
    remaining = [
        remaining[index] for index in _FILL_PERMUTATIONS[fill_index]
    ]
    remaining_iter = iter(remaining)
    for position, value in enumerate(attribute_order):
        if value is None:
            attribute_order[position] = next(remaining_iter)
    ordered_attributes = [
        value for value in attribute_order if value is not None
    ]
    core.require(len(ordered_attributes) == 8, "attribute order incomplete")
    if fact_order_id == "order_b":
        owners = [owners[index] for index in (2, 0, 3, 1)]
    return [*owners, *ordered_attributes]


def _render_prompt(
    state: Mapping[str, Any],
    paraphrase_id: str,
    fact_order_id: str,
    horizon_id: str,
    solution: Mapping[str, Any],
) -> tuple[str, dict[str, list[int]], list[str]]:
    core_records = _order_core_facts(
        _core_fact_records(state, paraphrase_id),
        state,
        fact_order_id,
    )
    filler_records = [
        {"fact_id": fact_id, "kind": "filler", "text": text}
        for fact_id, text in _FILLERS
    ]
    records = (
        [*filler_records, *core_records]
        if horizon_id == "near"
        else [*core_records, *filler_records]
    )

    chunks = [
        "Read the temporary registry. Treat every registry statement as authoritative.\n",
        "Registry:\n",
    ]
    cursor = sum(len(chunk) for chunk in chunks)
    spans: dict[str, list[int]] = {}
    fact_order: list[str] = []
    for index, fact in enumerate(records, start=1):
        prefix = f"{index}. "
        line = f"{prefix}{fact['text']}\n"
        start = cursor + len(prefix)
        end = start + len(fact["text"])
        spans[fact["fact_id"]] = [start, end]
        fact_order.append(fact["fact_id"])
        chunks.append(line)
        cursor += len(line)

    prefix = "Question: "
    query_text = _query_text(
        state["query"],
        paraphrase_id == "paraphrase",
    )
    chunks.append(prefix)
    query_start = cursor + len(prefix)
    chunks.append(query_text)
    cursor = query_start + len(query_text)
    spans["query"] = [query_start, cursor]
    chunks.append(
        "\nAnswer in one short sentence. Do not answer with only a marker word."
    )
    prompt = "".join(chunks)

    for fact_id, (start, end) in spans.items():
        if fact_id == "query":
            continue
        core.require(start < end and prompt[start:end], "empty fact span")
    core.require(
        solution["owner_fact_id"] in spans
        and solution["attribute_fact_id"] in spans,
        "target anchor missing from prompt",
    )
    return prompt, spans, fact_order


def _surface_token_multiset(text: str) -> dict[str, int]:
    return dict(sorted(Counter(_WORD_RE.findall(text.casefold())).items()))


def _normalized_prompt(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip().casefold()


def build_item(world: Mapping[str, Any], variant: str) -> dict[str, Any]:
    factors = parse_variant(variant)
    state = transform_state(world, factors["semantic_transform"])
    solution = solve_state(state)
    prompt, spans, fact_order = _render_prompt(
        state,
        factors["paraphrase_id"],
        factors["fact_order_id"],
        factors["horizon_id"],
        solution,
    )
    original_solution = solve_state(world)
    original_variant = variant_id(
        "original",
        factors["paraphrase_id"],
        factors["fact_order_id"],
        factors["horizon_id"],
    )
    item = {
        "record_id": _item_id(str(world["semantic_world_id"]), variant),
        "root_family_id": str(world["root_family_id"]),
        "semantic_world_id": str(world["semantic_world_id"]),
        "split": str(world["split"]),
        "split_ordinal": int(world["split_ordinal"]),
        "seed_key": str(world["seed_key"]),
        "variant_id": variant,
        **factors,
        "paired_not_independent": True,
        "semantic_state": state,
        "canonical_semantic_variant_sha256": core.sha256_json(
            _state_semantics(state)
        ),
        "observable_semantic_variant_sha256": core.sha256_json(
            _observable_semantics(state)
        ),
        "slot_canonical_semantic_sha256": core.sha256_json(
            _slot_semantics(state)
        ),
        "prompt": prompt,
        "normalized_surface_sha256": core.sha256_json(
            _normalized_prompt(prompt)
        ),
        "surface_token_multiset": _surface_token_multiset(prompt),
        "fact_ids_in_render_order": fact_order,
        "fact_char_spans": spans,
        "answer_source_position": fact_order.index(
            solution["attribute_fact_id"]
        ),
        "answer_source_core_position": [
            fact_id for fact_id in fact_order
            if not fact_id.startswith("filler_")
        ].index(solution["attribute_fact_id"]),
        "answer_source_attribute_position": [
            fact_id for fact_id in fact_order
            if fact_id.startswith("attribute_")
        ].index(solution["attribute_fact_id"]),
        "physical_anchor_chars": {
            "owner_fact": spans[solution["owner_fact_id"]],
            "attribute_fact": spans[solution["attribute_fact_id"]],
            "query": spans["query"],
        },
        "gold": {
            **solution,
            "foil_values": [
                value for value in core.VALUES
                if value != solution["answer_value"]
            ],
            "original_answer_value": original_solution["answer_value"],
            "answer_changed_from_original": (
                solution["answer_value"] != original_solution["answer_value"]
            ),
        },
        "pair_links": {
            "original_surface_peer_record_id": _item_id(
                str(world["semantic_world_id"]), original_variant
            ),
            "semantic_peer_record_ids": {
                semantic: _item_id(
                    str(world["semantic_world_id"]),
                    variant_id(
                        semantic,
                        factors["paraphrase_id"],
                        factors["fact_order_id"],
                        factors["horizon_id"],
                    ),
                )
                for semantic in core.SEMANTIC_TRANSFORMS
            },
            "semantic_transform_is_involution": True,
            "root_family_stays_in_one_split": True,
        },
        "teacher_forced_answer_prefix": "The retrieved marker is",
        "teacher_forced_context_joiner": "\n",
        "teacher_forced_candidate_continuations": {
            value: f" {value}" for value in core.VALUES
        },
        "natural_reference_answer": (
            f"The retrieved marker is {solution['answer_value']}."
        ),
        "scoring_contract": {
            "semantic_content_primary": True,
            "compare_all_three_foils": True,
            "strict_final_contract_required": False,
            "format_scored_separately": True,
            "eos_scored_separately": True,
            "budget_truncation_scored_separately": True,
            "target_not_first_natural_reference_token": True,
            "target_is_first_scored_continuation_after_teacher_prefix": True,
            "input_assembly": (
                "tokenize(prompt + joiner + prefix + continuation) and "
                "require full_ids == context_ids + continuation_ids"
            ),
        },
    }
    return item


def build_items(
    worlds: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    world_rows = list(worlds) if worlds is not None else build_worlds()
    return [
        build_item(world, variant)
        for world in world_rows
        for variant in core.VARIANTS
    ]


def _error(
    codes: list[str],
    messages: list[str],
    code: str,
    message: str,
) -> None:
    codes.append(code)
    messages.append(f"{code}: {message}")


def _context_without_query(prompt: str) -> str:
    return prompt.split("Question:", 1)[0]


def _semantic_diff(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, int]:
    left_owner = {
        int(edge["entity_slot"]): (
            int(edge["object_slot"]), str(edge["object"])
        )
        for edge in left["owner_edges"]
    }
    right_owner = {
        int(edge["entity_slot"]): (
            int(edge["object_slot"]), str(edge["object"])
        )
        for edge in right["owner_edges"]
    }
    left_attribute = {
        (int(edge["relation_slot"]), int(edge["object_slot"])): (
            int(edge["value_slot"]), str(edge["value"])
        )
        for edge in left["attribute_edges"]
    }
    right_attribute = {
        (int(edge["relation_slot"]), int(edge["object_slot"])): (
            int(edge["value_slot"]), str(edge["value"])
        )
        for edge in right["attribute_edges"]
    }
    return {
        "owner_edge_changes": sum(
            left_owner[key] != right_owner[key] for key in left_owner
        ),
        "attribute_edge_changes": sum(
            left_attribute[key] != right_attribute[key]
            for key in left_attribute
        ),
        "query_changes": int(left["query"] != right["query"]),
    }


def _first_last_attribute_values(item: Mapping[str, Any]) -> tuple[str, str]:
    state = item["semantic_state"]
    value_by_fact = {
        edge["fact_id"]: edge["value"]
        for edge in state["attribute_edges"]
    }
    sequence = [
        value_by_fact[fact_id]
        for fact_id in item["fact_ids_in_render_order"]
        if fact_id in value_by_fact
    ]
    core.require(len(sequence) == 8, "attribute value sequence changed")
    return str(sequence[0]), str(sequence[-1])


def _answer_recency_rank(item: Mapping[str, Any]) -> int:
    state = item["semantic_state"]
    value_by_fact = {
        edge["fact_id"]: edge["value"]
        for edge in state["attribute_edges"]
    }
    sequence = [
        value_by_fact[fact_id]
        for fact_id in item["fact_ids_in_render_order"]
        if fact_id in value_by_fact
    ]
    answer = item["gold"]["answer_value"]
    last_position = {
        value: max(index for index, observed in enumerate(sequence)
                   if observed == value)
        for value in core.VALUES
    }
    return 1 + sum(
        last_position[value] > last_position[answer]
        for value in core.VALUES
    )


def _counter_exact(
    counter: Counter,
    universe: Iterable[Any],
    expected: int,
) -> bool:
    cells = list(universe)
    return set(counter) == set(cells) and all(
        counter[cell] == expected for cell in cells
    )


def _majority_lookup_baseline(
    counter: Counter,
) -> dict[str, int | float]:
    by_feature: dict[tuple[Any, ...], Counter] = defaultdict(Counter)
    for cell, count in counter.items():
        core.require(isinstance(cell, tuple) and len(cell) >= 2, "baseline cell")
        by_feature[cell[:-1]][cell[-1]] += count
    numerator = sum(max(values.values()) for values in by_feature.values())
    denominator = sum(counter.values())
    core.require(denominator > 0, "empty lookup baseline")
    return {
        "majority_correct": numerator,
        "denominator": denominator,
        "accuracy_percent": 100.0 * numerator / denominator,
        "nonempty_feature_cells": len(by_feature),
    }


def audit_dataset(
    worlds: Iterable[Mapping[str, Any]] | None = None,
    items: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    world_rows = [deepcopy(dict(value)) for value in (
        list(worlds) if worlds is not None else build_worlds()
    )]
    item_rows = [deepcopy(dict(value)) for value in (
        list(items) if items is not None else build_items(world_rows)
    )]
    error_codes: list[str] = []
    errors: list[str] = []

    if set(_ORDER_FILL_INDEX) != set(core.SPLIT_ORDER):
        _error(error_codes, errors, "ORV_ORDER_000", "split order schedule changed")
    else:
        for split, expected_count in core.SPLIT_COUNTS.items():
            schedules = _ORDER_FILL_INDEX.get(split, {})
            if set(schedules) != set(core.FACT_ORDER_IDS):
                _error(
                    error_codes, errors, "ORV_ORDER_000",
                    f"{split} fact-order registry changed",
                )
                continue
            for order_id, values in schedules.items():
                if (
                    len(values) != expected_count
                    or any(not isinstance(value, int)
                           or isinstance(value, bool)
                           or value not in range(len(_FILL_PERMUTATIONS))
                           for value in values)
                ):
                    _error(
                        error_codes, errors, "ORV_ORDER_000",
                        f"{split}/{order_id} fill schedule invalid",
                    )

    if (
        set(_ENTITY_ASSIGNMENT_CODES) != set(core.SPLIT_ORDER)
        or set(_ENTITY_SPLIT_PERMUTATIONS) != set(core.SPLIT_ORDER)
    ):
        _error(
            error_codes, errors, "ORV_LEXICAL_000",
            "entity lexical schedule split registry changed",
        )
    else:
        for split, expected_count in core.SPLIT_COUNTS.items():
            codes = _ENTITY_ASSIGNMENT_CODES[split]
            permutation = _ENTITY_SPLIT_PERMUTATIONS[split]
            if (
                len(codes) != expected_count
                or any(len(code) != 4 or len(set(code)) != 4
                       or any(character not in "01234567" for character in code)
                       for code in codes)
                or sorted(permutation) != list(range(len(_ENTITY_LEXICON)))
            ):
                _error(
                    error_codes, errors, "ORV_LEXICAL_000",
                    f"{split} entity lexical schedule is invalid",
                )

    expected_worlds = build_worlds()
    expected_items = build_items(expected_worlds)
    if world_rows != expected_worlds:
        _error(
            error_codes, errors, "ORV_REBUILD_001",
            "worlds differ from deterministic reconstruction",
        )
    if item_rows != expected_items:
        _error(
            error_codes, errors, "ORV_REBUILD_002",
            "records differ from deterministic reconstruction",
        )
    if len(world_rows) != core.EXPECTED_WORLD_COUNT:
        _error(error_codes, errors, "ORV_COUNT_001", "world count changed")
    if len(item_rows) != core.EXPECTED_ITEM_COUNT:
        _error(error_codes, errors, "ORV_COUNT_002", "record count changed")

    world_ids = [str(world.get("semantic_world_id", "")) for world in world_rows]
    family_ids = [str(world.get("root_family_id", "")) for world in world_rows]
    record_ids = [str(item.get("record_id", "")) for item in item_rows]
    seed_keys = [str(world.get("seed_key", "")) for world in world_rows]
    for code, values, label in (
        ("ORV_SPLIT_008", world_ids, "world ids"),
        ("ORV_SPLIT_008", family_ids, "family ids"),
        ("ORV_SPLIT_008", record_ids, "record ids"),
        ("ORV_SPLIT_007", seed_keys, "seed keys"),
    ):
        if len(set(values)) != len(values):
            _error(error_codes, errors, code, f"{label} are not unique")

    split_counts = Counter(str(world.get("split", "")) for world in world_rows)
    if dict(split_counts) != core.SPLIT_COUNTS:
        _error(error_codes, errors, "ORV_BAL_000", "split counts changed")

    world_by_id = {
        str(world["semantic_world_id"]): world for world in world_rows
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in item_rows:
        grouped[str(item.get("semantic_world_id", ""))].append(item)
    if set(grouped) != set(world_by_id):
        _error(error_codes, errors, "ORV_SPLIT_001", "world lineage changed")
    for world_id, rows in grouped.items():
        if [row.get("variant_id") for row in rows] != list(core.VARIANTS):
            _error(
                error_codes, errors, "ORV_GRID_001",
                f"{world_id} does not contain the exact 32-cell grid",
            )
        splits = {row.get("split") for row in rows}
        families = {row.get("root_family_id") for row in rows}
        if len(splits) != 1 or len(families) != 1:
            _error(
                error_codes, errors, "ORV_SPLIT_003",
                f"{world_id} counterfactual closure crossed split/family",
            )

    graph_checks = 0
    path_value_coverage_checks = 0
    involution_checks = 0
    for world in world_rows:
        try:
            _validate_graph(world)
            stored = solve_state(world)
            if stored["answer_value_slot"] != world["base_answer_value_slot"]:
                _error(
                    error_codes, errors, "ORV_GRAPH_005",
                    f"{world['semantic_world_id']} stored answer mismatch",
                )
            else:
                graph_checks += 1
            if (
                len({
                    edge["value_slot"] for edge in world["attribute_edges"]
                    if edge["relation_slot"] == world["query"]["relation_slot"]
                }) == 4
            ):
                path_value_coverage_checks += 1
            else:
                _error(
                    error_codes, errors, "ORV_GRAPH_007",
                    f"{world['semantic_world_id']} relation value coverage changed",
                )
            for semantic in core.SEMANTIC_TRANSFORMS:
                if _apply_transform_twice(world, semantic):
                    involution_checks += 1
                else:
                    _error(
                        error_codes, errors, "ORV_PAIR_003",
                        f"{world['semantic_world_id']}/{semantic} not involutive",
                    )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            _error(
                error_codes, errors, "ORV_GRAPH_001",
                f"{world.get('semantic_world_id', '?')}: {exc}",
            )

    prompt_hashes = [item.get("normalized_surface_sha256") for item in item_rows]
    if len(set(prompt_hashes)) != len(prompt_hashes):
        _error(
            error_codes, errors, "ORV_SPLIT_006",
            "normalized prompt hashes are not globally unique",
        )
    abstract_state_rows: list[tuple[str, str, str, str]] = []
    deletion_recoverability = Counter()
    for world in world_rows:
        try:
            for semantic in core.SEMANTIC_TRANSFORMS:
                state = transform_state(world, semantic)
                deletion_recoverability.update(
                    key for key, recoverable
                    in _gold_path_deletion_recoverability(state).items()
                    if recoverable
                )
                abstract_state_rows.append((
                    str(world["semantic_world_id"]),
                    semantic,
                    core.sha256_json(_slot_semantics(state)),
                    core.sha256_json(_observable_semantics(state)),
                ))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            _error(
                error_codes, errors, "ORV_GRAPH_001",
                f"{world.get('semantic_world_id', '?')} abstract closure: {exc}",
            )
    abstract_hashes = [row[2] for row in abstract_state_rows]
    observable_hashes = [row[3] for row in abstract_state_rows]
    if len(set(abstract_hashes)) != (
        core.EXPECTED_WORLD_COUNT * len(core.SEMANTIC_TRANSFORMS)
    ):
        _error(
            error_codes, errors, "ORV_SPLIT_005",
            "abstract semantic counterfactual closures are not globally unique",
        )
    if len(set(observable_hashes)) != (
        core.EXPECTED_WORLD_COUNT * len(core.SEMANTIC_TRANSFORMS)
    ):
        _error(
            error_codes, errors, "ORV_SPLIT_009",
            "model-visible semantic counterfactual states are not globally unique",
        )
    expected_semantic_states = (
        core.EXPECTED_WORLD_COUNT * len(core.SEMANTIC_TRANSFORMS)
    )
    if deletion_recoverability != Counter({
        "owner_fact_deleted_recoverable": expected_semantic_states,
        "attribute_fact_deleted_recoverable": expected_semantic_states,
        "both_gold_facts_deleted_recoverable": expected_semantic_states,
    }):
        _error(
            error_codes, errors, "ORV_CAUSAL_001",
            "gold-path deletion redundancy measurement changed",
        )
    if any("final:" in str(item.get("prompt", "")).casefold() for item in item_rows):
        _error(
            error_codes, errors, "ORV_SURFACE_009",
            "strict FINAL contract leaked into prompt",
        )

    surface_checks = 0
    mechanical_checks = 0
    for item in item_rows:
        try:
            _validate_graph(item["semantic_state"])
            solution = solve_state(item["semantic_state"])
            if (
                solution["answer_value"] != item["gold"]["answer_value"]
                or solution["answer_object"] != item["gold"]["answer_object"]
                or set(item["gold"]["foil_values"])
                != set(core.VALUES) - {solution["answer_value"]}
            ):
                _error(
                    error_codes, errors, "ORV_GRAPH_005",
                    f"{item.get('record_id')} gold/foil mismatch",
                )
            else:
                mechanical_checks += 1

            expected_lexical_hash = core.sha256_json(
                _state_semantics(item["semantic_state"])
            )
            expected_observable_hash = core.sha256_json(
                _observable_semantics(item["semantic_state"])
            )
            expected_slot_hash = core.sha256_json(
                _slot_semantics(item["semantic_state"])
            )
            if (
                item.get("canonical_semantic_variant_sha256")
                != expected_lexical_hash
                or item.get("observable_semantic_variant_sha256")
                != expected_observable_hash
                or item.get("slot_canonical_semantic_sha256")
                != expected_slot_hash
            ):
                _error(
                    error_codes, errors, "ORV_HASH_001",
                    f"{item.get('record_id')} semantic hash mismatch",
                )
            if item.get("normalized_surface_sha256") != core.sha256_json(
                _normalized_prompt(str(item["prompt"]))
            ):
                _error(
                    error_codes, errors, "ORV_HASH_002",
                    f"{item.get('record_id')} surface hash mismatch",
                )
            if item.get("surface_token_multiset") != _surface_token_multiset(
                str(item["prompt"])
            ):
                _error(
                    error_codes, errors, "ORV_HASH_003",
                    f"{item.get('record_id')} surface multiset mismatch",
                )

            expected_reference = (
                f"The retrieved marker is {solution['answer_value']}."
            )
            scoring = item.get("scoring_contract", {})
            if (
                item.get("teacher_forced_answer_prefix")
                != "The retrieved marker is"
                or item.get("teacher_forced_context_joiner") != "\n"
                or item.get("teacher_forced_candidate_continuations")
                != {value: f" {value}" for value in core.VALUES}
                or item.get("natural_reference_answer") != expected_reference
                or scoring.get(
                    "target_not_first_natural_reference_token"
                ) is not True
                or scoring.get(
                    "target_is_first_scored_continuation_after_teacher_prefix"
                ) is not True
                or scoring.get("input_assembly") != (
                    "tokenize(prompt + joiner + prefix + continuation) and "
                    "require full_ids == context_ids + continuation_ids"
                )
                or "target_not_first_teacher_forced_output_token" in scoring
            ):
                _error(
                    error_codes, errors, "ORV_SCORING_001",
                    f"{item.get('record_id')} scoring contract is inconsistent",
                )

            context = _context_without_query(str(item["prompt"])).casefold()
            value_counts = Counter(
                value for value in core.VALUES
                for _ in range(len(re.findall(
                    rf"\b{re.escape(value)}\b", context
                )))
            )
            if value_counts != Counter({value: 2 for value in core.VALUES}):
                _error(
                    error_codes, errors, "ORV_SURFACE_001",
                    f"{item.get('record_id')} values are not exactly twice each",
                )
            else:
                surface_checks += 1
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            _error(
                error_codes, errors, "ORV_RECORD_001",
                f"{item.get('record_id', '?')}: {exc}",
            )

    pair_checks = Counter()
    horizon_checks = 0
    for world_id, rows in grouped.items():
        by_variant = {str(row["variant_id"]): row for row in rows}
        world = world_by_id[world_id]
        base_solution = solve_state(world)
        for row in rows:
            expected_peers = {
                semantic: _item_id(
                    world_id,
                    variant_id(
                        semantic,
                        str(row["paraphrase_id"]),
                        str(row["fact_order_id"]),
                        str(row["horizon_id"]),
                    ),
                )
                for semantic in core.SEMANTIC_TRANSFORMS
            }
            links = row.get("pair_links", {})
            actual_peers = links.get("semantic_peer_record_ids")
            peer_records = {
                peer.get("record_id"): peer for peer in rows
            }
            if (
                actual_peers != expected_peers
                or any(peer_id not in peer_records
                       for peer_id in expected_peers.values())
                or any(peer_records[peer_id].get("split") != row.get("split")
                       for peer_id in expected_peers.values())
            ):
                _error(
                    error_codes, errors, "ORV_PAIR_001",
                    f"{row.get('record_id')} peer closure is not exact",
                )
        for paraphrase in core.PARAPHRASE_IDS:
            for order in core.FACT_ORDER_IDS:
                for horizon in core.HORIZON_IDS:
                    original_key = variant_id(
                        "original", paraphrase, order, horizon
                    )
                    original = by_variant[original_key]
                    for semantic in (
                        "value_swap", "binding_swap", "relation_swap"
                    ):
                        changed = by_variant[
                            variant_id(semantic, paraphrase, order, horizon)
                        ]
                        diff = _semantic_diff(
                            original["semantic_state"],
                            changed["semantic_state"],
                        )
                        expected_diff = {
                            "value_swap": {
                                "owner_edge_changes": 0,
                                "attribute_edge_changes": 2,
                                "query_changes": 0,
                            },
                            "binding_swap": {
                                "owner_edge_changes": 2,
                                "attribute_edge_changes": 0,
                                "query_changes": 0,
                            },
                            "relation_swap": {
                                "owner_edge_changes": 0,
                                "attribute_edge_changes": 0,
                                "query_changes": 1,
                            },
                        }[semantic]
                        if diff != expected_diff:
                            _error(
                                error_codes, errors, "ORV_PAIR_002",
                                f"{world_id}/{changed['variant_id']} edge diff {diff}",
                            )
                        if changed["gold"]["answer_value"] == base_solution["answer_value"]:
                            _error(
                                error_codes, errors, "ORV_PAIR_007",
                                f"{world_id}/{semantic} answer did not change",
                            )
                        if semantic in {"value_swap", "binding_swap"}:
                            if (
                                original["surface_token_multiset"]
                                != changed["surface_token_multiset"]
                            ):
                                _error(
                                    error_codes, errors, "ORV_PAIR_004",
                                    f"{world_id}/{semantic} token multiset changed",
                                )
                            else:
                                pair_checks[semantic] += 1
                        else:
                            if (
                                _context_without_query(original["prompt"])
                                != _context_without_query(changed["prompt"])
                            ):
                                _error(
                                    error_codes, errors, "ORV_PAIR_005",
                                    f"{world_id}/relation context changed",
                                )
                            else:
                                pair_checks[semantic] += 1

        for semantic in core.SEMANTIC_TRANSFORMS:
            for paraphrase in core.PARAPHRASE_IDS:
                for order in core.FACT_ORDER_IDS:
                    near = by_variant[
                        variant_id(semantic, paraphrase, order, "near")
                    ]
                    far = by_variant[
                        variant_id(semantic, paraphrase, order, "far")
                    ]
                    if near["surface_token_multiset"] != far["surface_token_multiset"]:
                        _error(
                            error_codes, errors, "ORV_HORIZON_001",
                            f"{world_id} near/far sentence multiset changed",
                        )
                        continue
                    core_ids_near = [
                        fact_id for fact_id in near["fact_ids_in_render_order"]
                        if not fact_id.startswith("filler_")
                    ]
                    core_ids_far = [
                        fact_id for fact_id in far["fact_ids_in_render_order"]
                        if not fact_id.startswith("filler_")
                    ]
                    if core_ids_near != core_ids_far:
                        _error(
                            error_codes, errors, "ORV_HORIZON_003",
                            f"{world_id} only part of the core moved",
                        )
                        continue
                    exact_sentence_shift = all(
                        near["fact_ids_in_render_order"].index(fact_id)
                        - far["fact_ids_in_render_order"].index(fact_id)
                        == len(_FILLERS)
                        for fact_id in core_ids_near
                    )
                    if not exact_sentence_shift:
                        _error(
                            error_codes, errors, "ORV_HORIZON_002",
                            f"{world_id} core shift is not exactly {len(_FILLERS)}",
                        )
                    query_start_near = near["physical_anchor_chars"]["query"][0]
                    query_start_far = far["physical_anchor_chars"]["query"][0]
                    all_farther = True
                    for fact_id in core_ids_near:
                        near_end = near["fact_char_spans"][fact_id][1]
                        far_end = far["fact_char_spans"][fact_id][1]
                        if (
                            query_start_far - far_end
                            <= query_start_near - near_end
                        ):
                            all_farther = False
                    if not all_farther:
                        _error(
                            error_codes, errors, "ORV_HORIZON_002",
                            f"{world_id} not all core facts moved farther",
                        )
                    elif exact_sentence_shift:
                        horizon_checks += 1

    balance: dict[str, Any] = {}
    shortcut_checks = 0
    for split, split_world_count in core.SPLIT_COUNTS.items():
        balance[split] = {}
        for variant in core.VARIANTS:
            rows = [
                item for item in item_rows
                if item.get("split") == split
                and item.get("variant_id") == variant
            ]
            cell_counts = Counter((
                int(item["semantic_state"]["query"]["entity_slot"]),
                int(item["semantic_state"]["query"]["relation_slot"]),
                int(item["gold"]["answer_value_slot"]),
            ) for item in rows)
            expected_per_cell = split_world_count // 32
            if cell_counts != Counter({
                (entity, relation, answer): expected_per_cell
                for entity in range(4)
                for relation in range(2)
                for answer in range(4)
            }):
                _error(
                    error_codes, errors, "ORV_BAL_002",
                    f"{split}/{variant} q-r-answer grid unbalanced",
                )
            answer_counts = Counter(
                item["gold"]["answer_value"] for item in rows
            )
            if answer_counts != Counter({
                value: split_world_count // 4 for value in core.VALUES
            }):
                _error(
                    error_codes, errors, "ORV_BAL_001",
                    f"{split}/{variant} answer balance changed",
                )
            first_hits_by_answer = Counter()
            last_hits_by_answer = Counter()
            recency_by_answer = Counter()
            source_position_by_answer = Counter()
            for item in rows:
                first_value, last_value = _first_last_attribute_values(item)
                answer = item["gold"]["answer_value"]
                first_hits_by_answer[answer] += int(first_value == answer)
                last_hits_by_answer[answer] += int(last_value == answer)
                recency_by_answer[(answer, _answer_recency_rank(item))] += 1
                source_position_by_answer[(
                    answer, int(item["answer_source_attribute_position"])
                )] += 1
            first_last_expected = split_world_count // 16
            source_expected = split_world_count // 32
            first_last_ok = (
                _counter_exact(
                    first_hits_by_answer, core.VALUES, first_last_expected
                )
                and _counter_exact(
                    last_hits_by_answer, core.VALUES, first_last_expected
                )
            )
            recency_ok = _counter_exact(
                recency_by_answer,
                ((answer, rank) for answer in core.VALUES
                 for rank in range(1, 5)),
                first_last_expected,
            )
            source_position_ok = _counter_exact(
                source_position_by_answer,
                ((answer, position) for answer in core.VALUES
                 for position in range(8)),
                source_expected,
            )
            if not first_last_ok:
                _error(
                    error_codes, errors, "ORV_SHORTCUT_001",
                    f"{split}/{variant} first/last conditional shortcut imbalance",
                )
            if not recency_ok:
                _error(
                    error_codes, errors, "ORV_SHORTCUT_002",
                    f"{split}/{variant} answer-recency rank grid unbalanced",
                )
            if not source_position_ok:
                _error(
                    error_codes, errors, "ORV_BAL_004",
                    f"{split}/{variant} answer-source position grid unbalanced",
                )
            if first_last_ok and recency_ok and source_position_ok:
                shortcut_checks += 1
            balance[split][variant] = {
                "q_relation_answer_cells": len(cell_counts),
                "count_per_cell": expected_per_cell,
                "answer_counts": dict(sorted(answer_counts.items())),
                "first_hits_by_answer": dict(sorted(first_hits_by_answer.items())),
                "last_hits_by_answer": dict(sorted(last_hits_by_answer.items())),
                "recency_rank_counts": {
                    f"{answer}|{rank}": recency_by_answer[(answer, rank)]
                    for answer in core.VALUES for rank in range(1, 5)
                },
                "source_position_counts": {
                    f"{answer}|{position}": source_position_by_answer[
                        (answer, position)
                    ]
                    for answer in core.VALUES for position in range(8)
                },
            }

    lexical_balance: dict[str, Any] = {}
    for split, split_world_count in core.SPLIT_COUNTS.items():
        lexical_balance[split] = {}
        for semantic in core.SEMANTIC_TRANSFORMS:
            rows = [
                item for item in item_rows
                if item.get("split") == split
                and item.get("semantic_transform") == semantic
                and item.get("paraphrase_id") == "standard"
                and item.get("fact_order_id") == "order_a"
                and item.get("horizon_id") == "near"
            ]
            query_name_answer = Counter((
                item["semantic_state"]["query"]["entity"],
                item["gold"]["answer_value"],
            ) for item in rows)
            query_name_relation = Counter((
                item["semantic_state"]["query"]["entity"],
                item["semantic_state"]["query"]["relation"],
            ) for item in rows)
            answer_object_relation_answer = Counter((
                item["gold"]["answer_object"],
                item["semantic_state"]["query"]["relation"],
                item["gold"]["answer_value"],
            ) for item in rows)
            attribute_lexical_edges = Counter(
                (edge["object"], edge["relation"], edge["value"])
                for item in rows
                for edge in item["semantic_state"]["attribute_edges"]
            )
            owner_lexical_edges = Counter(
                (edge["entity"], edge["object"])
                for item in rows
                for edge in item["semantic_state"]["owner_edges"]
            )
            owner_lexical_answer = Counter(
                (edge["entity"], edge["object"], item["gold"]["answer_value"])
                for item in rows
                for edge in item["semantic_state"]["owner_edges"]
            )
            selected_owner_answer = Counter(
                (
                    item["semantic_state"]["query"]["entity"],
                    item["gold"]["answer_object"],
                    item["gold"]["answer_value"],
                )
                for item in rows
            )
            selected_owner_relation_answer = Counter(
                (
                    item["semantic_state"]["query"]["entity"],
                    item["gold"]["answer_object"],
                    item["semantic_state"]["query"]["relation"],
                    item["gold"]["answer_value"],
                )
                for item in rows
            )
            query_ok = _counter_exact(
                query_name_answer,
                ((entity, answer) for entity in _ENTITY_LEXICON
                 for answer in core.VALUES),
                split_world_count // 32,
            )
            query_relation_ok = _counter_exact(
                query_name_relation,
                ((entity, relation) for entity in _ENTITY_LEXICON
                 for relation in core.ATTRIBUTE_RELATIONS),
                split_world_count // 16,
            )
            answer_object_ok = _counter_exact(
                answer_object_relation_answer,
                ((obj, relation, answer) for obj in _OBJECT_LEXICON
                 for relation in core.ATTRIBUTE_RELATIONS
                 for answer in core.VALUES),
                split_world_count // 32,
            )
            attribute_ok = _counter_exact(
                attribute_lexical_edges,
                ((obj, relation, value) for obj in _OBJECT_LEXICON
                 for relation in core.ATTRIBUTE_RELATIONS
                 for value in core.VALUES),
                split_world_count // 4,
            )
            owner_pair_ok = _counter_exact(
                owner_lexical_edges,
                ((entity, obj) for entity in _ENTITY_LEXICON
                 for obj in _OBJECT_LEXICON),
                split_world_count // 8,
            )
            owner_answer_ok = _counter_exact(
                owner_lexical_answer,
                ((entity, obj, answer) for entity in _ENTITY_LEXICON
                 for obj in _OBJECT_LEXICON for answer in core.VALUES),
                split_world_count // 32,
            )
            selected_pair_baseline = _majority_lookup_baseline(
                selected_owner_answer
            )
            selected_relation_baseline = _majority_lookup_baseline(
                selected_owner_relation_answer
            )
            frozen_selected_baseline_ok = (
                selected_pair_baseline["majority_correct"]
                == 32
                and selected_pair_baseline["denominator"] == split_world_count
                and selected_relation_baseline["majority_correct"]
                == split_world_count // 2
                and selected_relation_baseline["denominator"]
                == split_world_count
            )
            if not query_ok:
                _error(
                    error_codes, errors, "ORV_LEXICAL_001",
                    f"{split}/{semantic} query-name x answer is unbalanced",
                )
            if not answer_object_ok:
                _error(
                    error_codes, errors, "ORV_LEXICAL_002",
                    f"{split}/{semantic} object-relation-answer is unbalanced",
                )
            if not query_relation_ok:
                _error(
                    error_codes, errors, "ORV_LEXICAL_004",
                    f"{split}/{semantic} query-name x relation is unbalanced",
                )
            if not owner_pair_ok:
                _error(
                    error_codes, errors, "ORV_LEXICAL_005",
                    f"{split}/{semantic} owner name x object is unbalanced",
                )
            if not owner_answer_ok:
                _error(
                    error_codes, errors, "ORV_LEXICAL_006",
                    f"{split}/{semantic} owner name-object-answer is unbalanced",
                )
            if not frozen_selected_baseline_ok:
                _error(
                    error_codes, errors, "ORV_LEXICAL_007",
                    f"{split}/{semantic} selected-owner lookup baseline drifted",
                )
            if not attribute_ok:
                _error(
                    error_codes, errors, "ORV_LEXICAL_003",
                    f"{split}/{semantic} object-relation-value facts unbalanced",
                )
            lexical_balance[split][semantic] = {
                "passed": (
                    query_ok and query_relation_ok and answer_object_ok
                    and attribute_ok and owner_pair_ok and owner_answer_ok
                    and frozen_selected_baseline_ok
                ),
                "query_entity_count": len(_ENTITY_LEXICON),
                "object_count": len(_OBJECT_LEXICON),
                "query_name_answer_count_per_cell": split_world_count // 32,
                "query_name_relation_count_per_cell": split_world_count // 16,
                "object_relation_answer_count_per_cell": split_world_count // 32,
                "attribute_object_relation_value_count_per_cell": (
                    split_world_count // 4
                ),
                "fixed_value_accuracy_percent": 25,
                "query_name_lookup_accuracy_percent": 25,
                "answer_object_relation_lookup_accuracy_percent": 25,
                "owner_name_object_count_per_cell": split_world_count // 8,
                "owner_name_object_answer_count_per_cell": (
                    split_world_count // 32
                ),
                "owner_pair_is_a_hard_balance_gate": True,
                "selected_owner_name_object_lookup_baseline": (
                    selected_pair_baseline
                ),
                "selected_owner_name_object_relation_lookup_baseline": (
                    selected_relation_baseline
                ),
                "selected_owner_conjunction_is_not_exactly_balanced": True,
                "selected_owner_conjunction_must_remain_a_registered_baseline": True,
            }

    pooled_selected_owner_baselines: dict[str, Any] = {}
    for split in core.SPLIT_ORDER:
        rows = [
            item for item in item_rows
            if item.get("split") == split
            and item.get("paraphrase_id") == "standard"
            and item.get("fact_order_id") == "order_a"
            and item.get("horizon_id") == "near"
        ]
        pooled_counter = Counter(
            (
                item["semantic_state"]["query"]["entity"],
                item["gold"]["answer_object"],
                item["semantic_state"]["query"]["relation"],
                item["gold"]["answer_value"],
            )
            for item in rows
        )
        pooled_baseline = _majority_lookup_baseline(pooled_counter)
        pooled_selected_owner_baselines[split] = pooled_baseline
        if not (
            pooled_baseline["majority_correct"]
            == 3 * core.SPLIT_COUNTS[split] // 2
            and pooled_baseline["denominator"]
            == core.SPLIT_COUNTS[split] * len(core.SEMANTIC_TRANSFORMS)
            and pooled_baseline["nonempty_feature_cells"] == 64
        ):
            _error(
                error_codes, errors, "ORV_LEXICAL_008",
                f"{split} pooled selected-owner baseline drifted",
            )

    split_overlap: dict[str, dict[str, dict[str, int]]] = {}
    for left_index, left in enumerate(core.SPLIT_ORDER):
        split_overlap[left] = {}
        left_world_hashes = {
            core.sha256_json(_state_semantics(world))
            for world in world_rows if world.get("split") == left
        }
        left_semantic_hashes = {
            item["canonical_semantic_variant_sha256"]
            for item in item_rows if item.get("split") == left
        }
        left_observable_hashes = {
            item["observable_semantic_variant_sha256"]
            for item in item_rows if item.get("split") == left
        }
        left_slot_hashes = {
            item["slot_canonical_semantic_sha256"]
            for item in item_rows if item.get("split") == left
        }
        left_prompt_hashes = {
            item["normalized_surface_sha256"]
            for item in item_rows if item.get("split") == left
        }
        for right in core.SPLIT_ORDER[left_index + 1:]:
            right_world_hashes = {
                core.sha256_json(_state_semantics(world))
                for world in world_rows if world.get("split") == right
            }
            right_semantic_hashes = {
                item["canonical_semantic_variant_sha256"]
                for item in item_rows if item.get("split") == right
            }
            right_observable_hashes = {
                item["observable_semantic_variant_sha256"]
                for item in item_rows if item.get("split") == right
            }
            right_slot_hashes = {
                item["slot_canonical_semantic_sha256"]
                for item in item_rows if item.get("split") == right
            }
            right_prompt_hashes = {
                item["normalized_surface_sha256"]
                for item in item_rows if item.get("split") == right
            }
            overlaps = {
                "base_world": len(left_world_hashes & right_world_hashes),
                "semantic_variant": len(
                    left_semantic_hashes & right_semantic_hashes
                ),
                "observable_semantic_variant": len(
                    left_observable_hashes & right_observable_hashes
                ),
                "slot_semantic_variant": len(
                    left_slot_hashes & right_slot_hashes
                ),
                "normalized_prompt": len(
                    left_prompt_hashes & right_prompt_hashes
                ),
            }
            split_overlap[left][right] = overlaps
            if any(overlaps.values()):
                _error(
                    error_codes, errors, "ORV_SPLIT_004",
                    f"{left}/{right} leakage {overlaps}",
                )

    identity_body = {
        "world_count": len(world_rows),
        "record_count": len(item_rows),
        "worlds_sha256": core.sha256_json(world_rows),
        "records_sha256": core.sha256_json(item_rows),
        "world_split_counts": dict(split_counts),
        "variant_counts": dict(Counter(
            str(item.get("variant_id", "")) for item in item_rows
        )),
        "normalized_prompts_sha256": core.sha256_json(prompt_hashes),
    }
    identity = {
        **identity_body,
        "identity_sha256": core.sha256_json(identity_body),
    }
    report = {
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "cpu_only_dataset_audit",
        "passed": not error_codes,
        "error_codes": sorted(set(error_codes)),
        "errors": errors,
        "independent_unit": "semantic_world_id",
        "factor_grid_rows_are_paired": True,
        "world_count": len(world_rows),
        "record_count": len(item_rows),
        "split_counts": dict(split_counts),
        "graph_checks": graph_checks,
        "path_value_coverage_checks": path_value_coverage_checks,
        "prompt_fact_deletion_is_not_a_registered_causal_test": True,
        "prompt_fact_deletion_limitation": (
            "owner edges are a complete bijection and each relation is a "
            "complete value permutation; deleting the gold owner fact, the "
            "gold attribute fact, or both exposes unique missing-object and/or "
            "missing-value shortcuts in every semantic state"
        ),
        "gold_path_deletion_recoverability_counts": dict(sorted(
            deletion_recoverability.items()
        )),
        "mechanical_gold_checks": mechanical_checks,
        "surface_equal_frequency_checks": surface_checks,
        "involution_checks": involution_checks,
        "pair_checks": dict(pair_checks),
        "horizon_checks": horizon_checks,
        "shortcut_balance_checks": shortcut_checks,
        "balance": balance,
        "lexical_balance": lexical_balance,
        "selected_owner_conjunction_baselines_pooled_across_transforms": (
            pooled_selected_owner_baselines
        ),
        "selected_owner_conjunction_limitation": (
            "the query-selected name-object-relation conjunction is not an "
            "exact answer-balanced grid at 96/64 worlds; its measured majority "
            "lookup accuracy must be reported as a baseline, and success cannot "
            "by itself establish second-hop computation"
        ),
        "split_overlap": split_overlap,
        "abstract_semantic_state_count": len(set(abstract_hashes)),
        "observable_semantic_state_count": len(set(observable_hashes)),
        "model_weights_loaded": False,
        "cuda_used": False,
        "holdout_semantics": "preregistered_immutable_not_blind",
        "holdout_used_for_model_decision": False,
        "identity": identity,
    }
    return report


def dataset_payload(
    worlds: list[dict[str, Any]],
    items: list[dict[str, Any]],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    core.require(audit.get("passed") is True, "cannot seal failed audit")
    return {
        "phase": core.PHASE,
        "schema_version": core.SCHEMA_VERSION,
        "experiment": core.EXPERIMENT,
        "role": "cpu_only_complete_factor_grid",
        "independent_unit": "semantic_world_id",
        "factor_grid_rows_are_paired": True,
        "lexical_scope": {
            "entity_lexicon": list(_ENTITY_LEXICON),
            "object_lexicon": list(_OBJECT_LEXICON),
            "value_lexicon": list(core.VALUES),
            "reason": (
                "registered marginal and all-owner-edge grids are exact; "
                "the query-selected conjunction is measured but not exact"
            ),
            "lexical_generalization_is_not_tested": True,
        },
        "counts": {
            "worlds": len(worlds),
            "records": len(items),
            "split_worlds": dict(core.SPLIT_COUNTS),
            "records_per_world": len(core.VARIANTS),
            "semantic_transforms": len(core.SEMANTIC_TRANSFORMS),
            "paraphrases": len(core.PARAPHRASE_IDS),
            "fact_orders": len(core.FACT_ORDER_IDS),
            "horizons": len(core.HORIZON_IDS),
        },
        "worlds": worlds,
        "records": items,
        "identity": deepcopy(dict(audit["identity"])),
        "runtime_boundary": {
            "cpu_only": True,
            "tokenizers_loaded": False,
            "model_weights_loaded": False,
            "cuda_used": False,
            "holdout_semantics": "preregistered_immutable_not_blind",
            "holdout_used_for_model_decision": False,
        },
    }


def _with_existing_timestamp(
    path: Any,
    payload: Mapping[str, Any],
    hash_field: str,
) -> dict[str, Any]:
    timestamp = None
    if path.is_file():
        existing = core.load_json(path, path.name)
        timestamp = existing.get("created_at_utc")
        timestamp = core.validate_utc_timestamp(timestamp, path.name)
    return core.sealed_document(payload, hash_field, timestamp)


def _publication_install_self_test() -> dict[str, bool]:
    """Exercise no-overwrite publication only in an isolated temp directory."""
    temporary_root = core.ROOT / "tests" / "glm5_temp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="phase990-install-probe-", dir=temporary_root
    ) as directory:
        target = Path(directory) / "sealed.json"
        payload = b'{"probe":1}\n'
        first_installed = core.install_exact(target, payload)
        identical_installed = core.install_exact(target, payload)
        different_rejected = False
        try:
            core.install_exact(target, b'{"probe":2}\n')
        except RuntimeError:
            different_rejected = True
        return {
            "isolated_first_install_succeeds": first_installed is True,
            "isolated_identical_install_is_idempotent": (
                identical_installed is False
            ),
            "isolated_different_bytes_rejected": (
                different_rejected and target.read_bytes() == payload
            ),
        }


def self_test() -> dict[str, Any]:
    worlds = build_worlds()
    records = build_items(worlds)
    audit = audit_dataset(worlds, records)

    probes: dict[str, bool] = {}

    def run_probe(
        name: str,
        candidate_worlds: list[dict[str, Any]],
        candidate_records: list[dict[str, Any]],
        expected_code: str,
    ) -> None:
        report = audit_dataset(candidate_worlds, candidate_records)
        probes[name] = (
            report["passed"] is False
            and expected_code in report["error_codes"]
        )

    wrong_gold = deepcopy(records)
    wrong_gold[0]["gold"]["answer_value"] = core.VALUES[1]
    run_probe("wrong_gold_rejected", worlds, wrong_gold, "ORV_GRAPH_005")

    crossed_split = deepcopy(records)
    crossed_split[0]["split"] = "confirmation"
    run_probe(
        "counterfactual_split_tamper_rejected",
        worlds,
        crossed_split,
        "ORV_SPLIT_003",
    )

    prompt_tamper = deepcopy(records)
    prompt_tamper[0]["prompt"] += " direct shortcut red"
    run_probe(
        "surface_tamper_rejected", worlds, prompt_tamper, "ORV_HASH_002"
    )

    graph_tamper = deepcopy(worlds)
    graph_tamper[0]["owner_edges"][0]["object_slot"] = (
        graph_tamper[0]["owner_edges"][1]["object_slot"]
    )
    run_probe(
        "non_bijection_rejected", graph_tamper, records, "ORV_GRAPH_001"
    )

    relation_tamper = deepcopy(worlds)
    for object_slot in range(4):
        left = next(
            edge for edge in relation_tamper[0]["attribute_edges"]
            if edge["relation_slot"] == 0
            and edge["object_slot"] == object_slot
        )
        right = next(
            edge for edge in relation_tamper[0]["attribute_edges"]
            if edge["relation_slot"] == 1
            and edge["object_slot"] == object_slot
        )
        right["value_slot"] = left["value_slot"]
        right["value"] = left["value"]
    run_probe(
        "same_relation_values_rejected",
        relation_tamper,
        records,
        "ORV_GRAPH_001",
    )

    pair_tamper = deepcopy(records)
    pair_tamper[0]["pair_links"]["semantic_peer_record_ids"]["value_swap"] = (
        "nonexistent"
    )
    run_probe("pair_link_tamper_rejected", worlds, pair_tamper, "ORV_PAIR_001")

    semantic_hash_tamper = deepcopy(records)
    semantic_hash_tamper[0]["slot_canonical_semantic_sha256"] = "0" * 64
    run_probe(
        "semantic_hash_tamper_rejected",
        worlds,
        semantic_hash_tamper,
        "ORV_HASH_001",
    )

    source_position_tamper = deepcopy(records)
    source_position_tamper[0]["answer_source_attribute_position"] = (
        source_position_tamper[0]["answer_source_attribute_position"] + 1
    ) % 8
    run_probe(
        "source_position_tamper_rejected",
        worlds,
        source_position_tamper,
        "ORV_BAL_004",
    )

    checks = {
        "generation_deterministic": (
            worlds == build_worlds() and records == build_items(build_worlds())
        ),
        "formal_audit_passed": audit["passed"],
        "exact_world_count": len(worlds) == core.EXPECTED_WORLD_COUNT,
        "exact_record_count": len(records) == core.EXPECTED_ITEM_COUNT,
        "all_graphs_checked": audit["graph_checks"] == core.EXPECTED_WORLD_COUNT,
        "all_records_mechanically_checked": (
            audit["mechanical_gold_checks"] == core.EXPECTED_ITEM_COUNT
        ),
        "all_records_value_frequency_checked": (
            audit["surface_equal_frequency_checks"] == core.EXPECTED_ITEM_COUNT
        ),
        "all_semantic_transforms_involutive": (
            audit["involution_checks"]
            == core.EXPECTED_WORLD_COUNT * len(core.SEMANTIC_TRANSFORMS)
        ),
        "all_horizon_pairs_checked": (
            audit["horizon_checks"]
            == core.EXPECTED_WORLD_COUNT
            * len(core.SEMANTIC_TRANSFORMS)
            * len(core.PARAPHRASE_IDS)
            * len(core.FACT_ORDER_IDS)
        ),
        "all_abstract_semantic_states_unique": (
            audit["abstract_semantic_state_count"]
            == core.EXPECTED_WORLD_COUNT * len(core.SEMANTIC_TRANSFORMS)
        ),
        "all_observable_semantic_states_unique": (
            audit["observable_semantic_state_count"]
            == core.EXPECTED_WORLD_COUNT * len(core.SEMANTIC_TRANSFORMS)
        ),
        "all_gold_path_deletion_shortcuts_measured": all(
            count == core.EXPECTED_WORLD_COUNT * len(core.SEMANTIC_TRANSFORMS)
            for count in audit[
                "gold_path_deletion_recoverability_counts"
            ].values()
        ) and len(audit["gold_path_deletion_recoverability_counts"]) == 3,
        "all_shortcut_grids_checked": (
            audit["shortcut_balance_checks"]
            == len(core.SPLIT_COUNTS) * len(core.VARIANTS)
        ),
        "all_lexical_grids_checked": all(
            row["passed"]
            for split_rows in audit["lexical_balance"].values()
            for row in split_rows.values()
        ),
        **probes,
    }
    core.require(all(checks.values()), f"dataset self-test failed: {checks}")
    return {"passed": True, "checks": checks, "identity": audit["identity"]}


def write_artifacts() -> dict[str, Any]:
    core.self_test()
    negative_tests = self_test()
    publication_tests = _publication_install_self_test()
    core.require(
        all(publication_tests.values()),
        f"isolated publication self-test failed: {publication_tests}",
    )
    worlds = build_worlds()
    records = build_items(worlds)
    audit = audit_dataset(worlds, records)
    core.require(audit["passed"], f"dataset audit failed: {audit['errors'][:20]}")

    definitions = _with_existing_timestamp(
        core.DEFINITIONS_PATH,
        core.definitions_payload(),
        "definitions_sha256",
    )
    dataset = _with_existing_timestamp(
        core.DATASET_PATH,
        dataset_payload(worlds, records, audit),
        "dataset_sha256",
    )
    audit_payload = {
        **audit,
        "definitions_sha256": definitions["definitions_sha256"],
        "dataset_sha256": dataset["dataset_sha256"],
        "source_script_seals": core.file_seals({
            "core": core.SCRIPT_PATHS["core"],
            "dataset": core.SCRIPT_PATHS["dataset"],
        }),
        "negative_tests": negative_tests["checks"],
        "publication_negative_tests": publication_tests,
    }
    audit_document = _with_existing_timestamp(
        core.DATASET_AUDIT_PATH,
        audit_payload,
        "dataset_audit_sha256",
    )
    installed = {
        "definitions": core.install_exact(
            core.DEFINITIONS_PATH, core.json_bytes(definitions)
        ),
        "dataset": core.install_exact(
            core.DATASET_PATH, core.json_bytes(dataset)
        ),
        "dataset_audit": core.install_exact(
            core.DATASET_AUDIT_PATH, core.json_bytes(audit_document)
        ),
    }
    verified = verify_artifacts()
    return {
        "passed": True,
        "installed": installed,
        "post_install_verification": verified["passed"],
        "definitions_sha256": definitions["definitions_sha256"],
        "dataset_sha256": dataset["dataset_sha256"],
        "dataset_audit_sha256": audit_document["dataset_audit_sha256"],
        "dataset_file_sha256": core.sha256_file(core.DATASET_PATH),
        "world_count": len(worlds),
        "record_count": len(records),
    }


def verify_artifacts() -> dict[str, Any]:
    definitions = core.load_json(core.DEFINITIONS_PATH, "definitions")
    dataset = core.load_json(core.DATASET_PATH, "dataset")
    dataset_audit = core.load_json(core.DATASET_AUDIT_PATH, "dataset audit")
    core.verify_self_hash(definitions, "definitions_sha256", "definitions")
    core.verify_self_hash(dataset, "dataset_sha256", "dataset")
    core.verify_self_hash(
        dataset_audit, "dataset_audit_sha256", "dataset audit"
    )
    expected_definitions = core.sealed_document(
        core.definitions_payload(),
        "definitions_sha256",
        str(definitions["created_at_utc"]),
    )
    core.verify_exact_document(
        definitions,
        expected_definitions,
        "definitions_sha256",
        "definitions",
    )
    core.require(
        core.DEFINITIONS_PATH.read_bytes() == core.json_bytes(expected_definitions),
        "definitions bytes are not canonical",
    )
    worlds = build_worlds()
    records = build_items(worlds)
    audit = audit_dataset(worlds, records)
    expected_dataset = core.sealed_document(
        dataset_payload(worlds, records, audit),
        "dataset_sha256",
        str(dataset["created_at_utc"]),
    )
    core.verify_exact_document(
        dataset,
        expected_dataset,
        "dataset_sha256",
        "dataset",
    )
    core.require(
        core.DATASET_PATH.read_bytes() == core.json_bytes(expected_dataset),
        "dataset bytes are not canonical",
    )
    negative_tests = self_test()
    expected_audit_payload = {
        **audit,
        "definitions_sha256": definitions["definitions_sha256"],
        "dataset_sha256": dataset["dataset_sha256"],
        "source_script_seals": core.file_seals({
            "core": core.SCRIPT_PATHS["core"],
            "dataset": core.SCRIPT_PATHS["dataset"],
        }),
        "negative_tests": negative_tests["checks"],
        "publication_negative_tests": {
            "isolated_first_install_succeeds": True,
            "isolated_identical_install_is_idempotent": True,
            "isolated_different_bytes_rejected": True,
        },
    }
    expected_audit = core.sealed_document(
        expected_audit_payload,
        "dataset_audit_sha256",
        str(dataset_audit["created_at_utc"]),
    )
    core.verify_exact_document(
        dataset_audit,
        expected_audit,
        "dataset_audit_sha256",
        "dataset audit",
    )
    core.require(
        core.DATASET_AUDIT_PATH.read_bytes() == core.json_bytes(expected_audit),
        "dataset audit bytes are not canonical",
    )
    core.verify_file_seals(
        dataset_audit.get("source_script_seals"),
        {
            "core": core.SCRIPT_PATHS["core"],
            "dataset": core.SCRIPT_PATHS["dataset"],
        },
        "dataset audit",
    )
    return {
        "passed": True,
        "files_written": False,
        "definitions_sha256": definitions["definitions_sha256"],
        "dataset_sha256": dataset["dataset_sha256"],
        "dataset_audit_sha256": dataset_audit["dataset_audit_sha256"],
        "world_count": len(worlds),
        "record_count": len(records),
    }


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--self-test"]:
        result = self_test()
    elif arguments == ["--write"]:
        result = write_artifacts()
    elif arguments == ["--verify"]:
        result = verify_artifacts()
    else:
        raise SystemExit(
            "usage: phase990_binding_dataset.py "
            "[--self-test|--write|--verify]"
        )
    print(core.canonical_json(result))


if __name__ == "__main__":
    main()
