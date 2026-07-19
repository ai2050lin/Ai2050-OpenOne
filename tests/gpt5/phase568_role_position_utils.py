#!/usr/bin/env python3
"""Role-coordinate utilities for the Phase568 coarse residual audit."""

from __future__ import annotations

from typing import Any


ROLE_GROUPS = (
    "target_fact_object",
    "target_fact_value",
    "target_fact_relation",
    "same_relation_other_fact_ends",
    "other_relation_fact_ends",
    "query_relation",
    "query_object",
    "answer_boundary",
)


def char_end_to_token(offsets: list[tuple[int, int]], char_end: int) -> int:
    for index, (start, end) in enumerate(offsets):
        if start < char_end <= end:
            return index
    candidates = [index for index, (_start, end) in enumerate(offsets) if 0 < end <= char_end]
    if not candidates:
        raise ValueError(f"No token covers character boundary {char_end}")
    return max(candidates)


def span_in_parent(
    prompt: str, parent: str, child: str, *, last_child: bool = False
) -> tuple[int, int]:
    parent_start = prompt.index(parent)
    child_offset = parent.rfind(child) if last_child else parent.find(child)
    if child_offset < 0:
        raise ValueError(f"{child!r} is absent from {parent!r}")
    start = parent_start + child_offset
    return start, start + len(child)


def token_indices_for_span(
    offsets: list[tuple[int, int]], start: int, end: int
) -> list[int]:
    indices = [
        index for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > start and token_start < end
    ]
    if not indices:
        raise ValueError(f"No token overlaps character span [{start}, {end})")
    return indices


def role_positions(tokenizer: Any, row: dict[str, Any]) -> tuple[list[int], dict[str, list[int]]]:
    prompt = row["prompt"]
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    fragments = row["semantic_fragments"]
    target_fact = fragments["target_fact"]
    object_span = span_in_parent(prompt, target_fact, fragments["target_fact_object"])
    value_span = span_in_parent(prompt, target_fact, fragments["target_fact_value"])
    relation_span = span_in_parent(prompt, target_fact, fragments["target_fact_relation"])
    query_relation_span = span_in_parent(
        prompt, row["question"], fragments["query_relation"]
    )
    query_object_span = span_in_parent(
        prompt, row["question"], fragments["query_object"], last_child=True
    )
    groups = {
        "target_fact_object": token_indices_for_span(offsets, *object_span),
        "target_fact_value": token_indices_for_span(offsets, *value_span),
        "target_fact_relation": token_indices_for_span(offsets, *relation_span),
        "same_relation_other_fact_ends": [
            char_end_to_token(offsets, prompt.index(fact) + len(fact))
            for fact in fragments["same_relation_other_facts"]
        ],
        "other_relation_fact_ends": [
            char_end_to_token(offsets, prompt.index(fact) + len(fact))
            for fact in fragments["other_relation_facts"]
        ],
        "query_relation": token_indices_for_span(offsets, *query_relation_span),
        "query_object": token_indices_for_span(offsets, *query_object_span),
        "answer_boundary": [len(ids) - 1],
    }
    if tuple(groups) != ROLE_GROUPS:
        raise RuntimeError("Phase568 role group order drift")
    if any(not positions for positions in groups.values()):
        raise RuntimeError("Phase568 produced an empty role group")
    seen: dict[int, str] = {}
    overlaps = []
    for role, positions in groups.items():
        if len(positions) != len(set(positions)):
            raise RuntimeError(f"Phase568 duplicate coordinate within {role}")
        for position in positions:
            if position in seen:
                overlaps.append((position, seen[position], role))
            seen[position] = role
    if overlaps:
        raise RuntimeError(f"Phase568 role coordinates overlap: {overlaps}")
    return ids, groups


def typed_union(groups: dict[str, list[int]]) -> list[int]:
    return sorted(position for role in ROLE_GROUPS for position in groups[role])


def role_coordinate_count(groups: dict[str, list[int]]) -> int:
    return sum(len(groups[role]) for role in ROLE_GROUPS)
