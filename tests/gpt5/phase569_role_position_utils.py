#!/usr/bin/env python3
"""Token-coordinate utilities for the Phase569 coarse event trace."""

from __future__ import annotations

from typing import Any


ROLE_GROUPS = (
    "target_fact_object",
    "target_fact_relation",
    "target_fact_value",
    "other_fact_object",
    "other_fact_relation",
    "other_fact_value",
    "query_relation",
    "query_object",
    "query_terminal",
    "answer_boundary",
)


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
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > start and token_start < end
    ]
    if not indices:
        raise ValueError(f"No token overlaps character span [{start}, {end})")
    return indices


def role_positions(
    tokenizer: Any, prompt: str, row: dict[str, Any]
) -> tuple[list[int], dict[str, list[int]]]:
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    fragments = row["semantic_fragments"]
    target_fact = fragments["target_fact"]
    other_fact = fragments["other_fact"]
    question = row["question"]
    terminal_index = len(question.rstrip()) - 1
    if terminal_index < 0:
        raise ValueError("Phase569 question is empty")
    question_start = prompt.index(question)
    groups = {
        "target_fact_object": token_indices_for_span(
            offsets,
            *span_in_parent(prompt, target_fact, fragments["target_fact_object"]),
        ),
        "target_fact_relation": token_indices_for_span(
            offsets,
            *span_in_parent(prompt, target_fact, fragments["target_fact_relation"]),
        ),
        "target_fact_value": token_indices_for_span(
            offsets,
            *span_in_parent(prompt, target_fact, fragments["target_fact_value"]),
        ),
        "other_fact_object": token_indices_for_span(
            offsets,
            *span_in_parent(prompt, other_fact, fragments["other_fact_object"]),
        ),
        "other_fact_relation": token_indices_for_span(
            offsets,
            *span_in_parent(prompt, other_fact, fragments["other_fact_relation"]),
        ),
        "other_fact_value": token_indices_for_span(
            offsets,
            *span_in_parent(prompt, other_fact, fragments["other_fact_value"]),
        ),
        "query_relation": token_indices_for_span(
            offsets,
            *span_in_parent(prompt, question, fragments["query_relation"]),
        ),
        "query_object": token_indices_for_span(
            offsets,
            *span_in_parent(
                prompt, question, fragments["query_object"], last_child=True
            ),
        ),
        "query_terminal": token_indices_for_span(
            offsets,
            question_start + terminal_index,
            question_start + terminal_index + 1,
        ),
        "answer_boundary": [len(ids) - 1],
    }
    if tuple(groups) != ROLE_GROUPS:
        raise RuntimeError("Phase569 role group order drift")
    if any(not positions for positions in groups.values()):
        raise RuntimeError("Phase569 produced an empty role group")
    seen: dict[int, str] = {}
    overlaps = []
    for role, positions in groups.items():
        if len(positions) != len(set(positions)):
            raise RuntimeError(f"Phase569 duplicate coordinate within {role}")
        for position in positions:
            if position in seen:
                overlaps.append((position, seen[position], role))
            seen[position] = role
    if overlaps:
        raise RuntimeError(f"Phase569 role coordinates overlap: {overlaps}")
    return ids, groups


def typed_union(groups: dict[str, list[int]]) -> list[int]:
    return sorted(position for role in ROLE_GROUPS for position in groups[role])


def role_coordinate_count(groups: dict[str, list[int]]) -> int:
    return sum(len(groups[role]) for role in ROLE_GROUPS)
