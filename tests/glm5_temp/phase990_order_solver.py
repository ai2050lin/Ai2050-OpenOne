#!/usr/bin/env python3
"""CPU-only deterministic search for Phase990 shortcut-balanced fact orders."""
from __future__ import annotations

import itertools
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase990_binding_core as core
import phase990_binding_dataset as dataset


ATTRIBUTE_KEYS = tuple(
    (relation, obj) for relation in range(2) for obj in range(4)
)
FILL_PERMUTATIONS = tuple(itertools.permutations(range(5)))


def source_layout(world: dict, order_id: str) -> tuple[list[tuple[int, int] | None], list[tuple[int, int]]]:
    q = world["base_query_entity_slot"]
    relation = world["base_query_relation_slot"]
    target = world["base_target_object_slot"]
    base_position = (2 * q + relation + 4 * world["local_rep"]) % 8
    sources = (
        (relation, target),
        (relation, (target + 3) % 4),
        (1 - relation, target),
    )
    offsets = (0, 1, 2) if order_id == "order_a" else (3, 5, 7)
    layout: list[tuple[int, int] | None] = [None] * 8
    for source, offset in zip(sources, offsets, strict=True):
        layout[(base_position + offset) % 8] = source
    remaining = [key for key in ATTRIBUTE_KEYS if key not in sources]
    return layout, remaining


def render_order(world: dict, order_id: str, fill_index: int) -> list[tuple[int, int]]:
    layout, remaining = source_layout(world, order_id)
    selected = [remaining[index] for index in FILL_PERMUTATIONS[fill_index]]
    iterator = iter(selected)
    return [next(iterator) if value is None else value for value in layout]


def contribution(world: dict, order: list[tuple[int, int]], key_index: dict[tuple, int]) -> np.ndarray:
    vector = np.zeros(len(key_index), dtype=np.int16)
    for semantic in core.SEMANTIC_TRANSFORMS:
        state = dataset.transform_state(world, semantic)
        solution = dataset.solve_state(state)
        answer = solution["answer_value_slot"]
        values = {
            (edge["relation_slot"], edge["object_slot"]): edge["value_slot"]
            for edge in state["attribute_edges"]
        }
        sequence = [values[key] for key in order]
        if sequence[0] == answer:
            vector[key_index[("first", semantic, answer)]] += 1
        if sequence[-1] == answer:
            vector[key_index[("last", semantic, answer)]] += 1
        last = {
            value: max(i for i, observed in enumerate(sequence) if observed == value)
            for value in range(4)
        }
        rank = 1 + sum(last[value] > last[answer] for value in range(4))
        vector[key_index[("rank", semantic, answer, rank)]] += 1
    return vector


def solve(split: str, order_id: str) -> list[int]:
    worlds = [world for world in dataset.build_worlds() if world["split"] == split]
    keys = [
        *((kind, semantic, answer)
          for kind in ("first", "last")
          for semantic in core.SEMANTIC_TRANSFORMS
          for answer in range(4)),
        *(("rank", semantic, answer, rank)
          for semantic in core.SEMANTIC_TRANSFORMS
          for answer in range(4)
          for rank in range(1, 5)),
    ]
    key_index = {key: index for index, key in enumerate(keys)}
    target = np.array([
        len(worlds) // 16 for _ in keys
    ], dtype=np.int32)
    options = np.stack([
        np.stack([
            contribution(world, render_order(world, order_id, fill_index), key_index)
            for fill_index in range(len(FILL_PERMUTATIONS))
        ])
        for world in worlds
    ]).astype(np.int32)
    rng = random.Random(f"phase990-order:{split}:{order_id}")
    best_score = 10**9
    best_assignment: list[int] = []

    for restart in range(100):
        assignment = np.array([
            rng.randrange(len(FILL_PERMUTATIONS)) for _ in worlds
        ], dtype=np.int32)
        counts = options[np.arange(len(worlds)), assignment].sum(axis=0)
        score = int(np.square(counts - target).sum())
        temperature = 4.0
        for iteration in range(200_000):
            world_index = rng.randrange(len(worlds))
            current = int(assignment[world_index])
            delta = options[world_index] - options[world_index, current]
            residual = counts - target
            candidate_delta = (
                2 * (delta @ residual) + np.square(delta).sum(axis=1)
            )
            minimum = int(candidate_delta.min())
            if minimum <= 0:
                choices = np.flatnonzero(candidate_delta == minimum)
                selected = int(choices[rng.randrange(len(choices))])
            else:
                selected = rng.randrange(len(FILL_PERMUTATIONS))
                minimum = int(candidate_delta[selected])
                if rng.random() >= np.exp(-minimum / max(temperature, 0.05)):
                    temperature *= 0.9999
                    continue
            if selected != current:
                counts += delta[selected]
                assignment[world_index] = selected
                score += minimum
            temperature *= 0.9999
            if score == 0:
                return assignment.tolist()
        if score < best_score:
            best_score = score
            best_assignment = assignment.tolist()
            print(split, order_id, "best", best_score, "restart", restart, flush=True)
    raise RuntimeError(
        f"no exact order for {split}/{order_id}; best={best_score}; "
        f"assignment={best_assignment}"
    )


def main() -> None:
    result: dict[str, dict[str, list[int]]] = {}
    for split in core.SPLIT_ORDER:
        result[split] = {
            order_id: solve(split, order_id)
            for order_id in core.FACT_ORDER_IDS
        }
    print(core.canonical_json(result))


if __name__ == "__main__":
    main()
