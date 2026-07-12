"""Exact prompt-token role mapping used by Phase390 analysis and interventions."""

from __future__ import annotations

from typing import Any


REGISTERED_ROLES = (
    "entities",
    "attributes_items",
    "relations",
    "query_keywords",
    "query_window",
    "other_causal_prefix",
)


def prompt_token_ids(tokenizer: Any, case: dict[str, Any]) -> list[int]:
    return [
        int(value)
        for value in tokenizer(
            case["prompt"],
            add_special_tokens=bool(case["tokenization_add_special_tokens"]),
            truncation=True,
            max_length=256,
        )["input_ids"]
    ]


def occurrences(sequence: list[int], pattern: list[int]) -> list[list[int]]:
    if not pattern or len(pattern) > len(sequence):
        return []
    return [
        list(range(start, start + len(pattern)))
        for start in range(len(sequence) - len(pattern) + 1)
        if sequence[start : start + len(pattern)] == pattern
    ]


def fragment_positions(
    tokenizer: Any, prompt_ids: list[int], fragment: str
) -> list[int]:
    pieces = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in prompt_ids
    ]
    rendered = "".join(pieces)
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = rendered.find(fragment, cursor)
        if start < 0:
            break
        spans.append((start, start + len(fragment)))
        cursor = start + max(1, len(fragment))
    if not spans:
        return []
    positions: set[int] = set()
    offset = 0
    for position, piece in enumerate(pieces):
        piece_start, piece_end = offset, offset + len(piece)
        if any(piece_start < end and piece_end > start for start, end in spans):
            positions.add(position)
        offset = piece_end
    return sorted(positions)


def semantic_role_indices(
    tokenizer: Any,
    case: dict[str, Any],
    receiver_position: int,
    total_sequence_length: int | None = None,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    ids = prompt_token_ids(tokenizer, case)
    sequence_length = total_sequence_length or len(ids)
    if receiver_position < 0 or receiver_position >= sequence_length:
        raise RuntimeError(
            f"Invalid receiver {receiver_position}/{sequence_length} for {case['blind_case_id']}"
        )
    fragments = case["semantic_role_fragments_private"]
    raw: dict[str, set[int]] = {}
    missing: dict[str, list[str]] = {}
    for role in ("entities", "attributes_items", "relations", "query_keywords"):
        role_positions: set[int] = set()
        missing_fragments: list[str] = []
        for fragment in fragments[role]:
            mapped = fragment_positions(tokenizer, ids, fragment)
            if not mapped:
                missing_fragments.append(fragment)
            role_positions.update(
                position for position in mapped if position <= receiver_position
            )
        raw[role] = role_positions
        if missing_fragments:
            missing[role] = missing_fragments

    query_positions = fragment_positions(tokenizer, ids, case["query_fragment"])
    if not query_positions:
        missing["query_window"] = [case["query_fragment"]]
    raw["query_window"] = {
        position for position in query_positions if position <= receiver_position
    }

    assigned: set[int] = set()
    partition: dict[str, list[int]] = {}
    for role in REGISTERED_ROLES[:-1]:
        positions = sorted(raw[role] - assigned)
        partition[role] = positions
        assigned.update(positions)
    causal_prefix = set(range(receiver_position + 1))
    partition["other_causal_prefix"] = sorted(causal_prefix - assigned)
    flattened = [position for role in REGISTERED_ROLES for position in partition[role]]
    duplicate_count = len(flattened) - len(set(flattened))
    missing_prefix_positions = sorted(causal_prefix - set(flattened))
    outside_positions = sorted(set(flattened) - causal_prefix)
    audit = {
        "prompt_token_count": len(ids),
        "total_sequence_length": sequence_length,
        "receiver_position": receiver_position,
        "registered_role_count": len(REGISTERED_ROLES),
        "mapped_fragment_count": sum(
            len(fragments[role]) - len(missing.get(role, []))
            for role in ("entities", "attributes_items", "relations", "query_keywords")
        )
        + (0 if "query_window" in missing else 1),
        "missing_fragments": missing,
        "duplicate_partition_position_count": duplicate_count,
        "missing_prefix_positions": missing_prefix_positions,
        "outside_prefix_positions": outside_positions,
        "partition_conserved": (
            not duplicate_count and not missing_prefix_positions and not outside_positions
        ),
    }
    return partition, audit
