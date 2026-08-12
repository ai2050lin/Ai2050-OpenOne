#!/usr/bin/env python3
"""Validate a deterministic refinement of the broad other_prompt region."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase548_shared_attention_compute_protocol import tokenizer_for
from phase1002_multitoken_protocol import MODELS, OUT_ROOT


REFINED = (
    "user_header",
    "fact_scaffold",
    "query_scaffold",
    "instruction",
    "assistant_protocol",
)


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prefix_boundary(tokenizer, text: str, char_index: int, ids: list[int]):
    prefix = tokenizer.encode(
        text[:char_index], add_special_tokens=False
    )
    if list(ids[:len(prefix)]) != list(prefix):
        raise RuntimeError(
            f"prefix tokenization drift at char {char_index}"
        )
    return len(prefix)


def regions(tokenizer, case: dict):
    rendered = case["rendered_prompt"]
    raw = case["raw_prompt"]
    raw_start = rendered.index(raw)
    lines = raw.splitlines()
    if len(lines) != 3:
        raise RuntimeError(f"line drift: {lines}")
    query_char = raw_start + len(lines[0]) + 1
    instruction_char = query_char + len(lines[1]) + 1
    assistant_char = raw_start + len(raw)
    ids = case["input_ids"]
    query_start = prefix_boundary(
        tokenizer, rendered, query_char, ids
    )
    instruction_start = prefix_boundary(
        tokenizer, rendered, instruction_char, ids
    )
    assistant_start = prefix_boundary(
        tokenizer, rendered, assistant_char, ids
    )
    roles = case["role_positions"]
    prompt_length = case["input_token_count"]
    excluded = {
        int(roles["slot0_entity"]),
        int(roles["slot1_entity"]),
        int(roles["slot0_color"]),
        int(roles["slot1_color"]),
        int(roles["query_name"]),
        prompt_length - 1,
    }
    first_fact = min(
        int(roles[name])
        for name in (
            "slot0_entity", "slot1_entity",
            "slot0_color", "slot1_color",
        )
    )
    bounds = {
        "user_header": (0, first_fact),
        "fact_scaffold": (first_fact, query_start),
        "query_scaffold": (query_start, instruction_start),
        "instruction": (instruction_start, assistant_start),
        "assistant_protocol": (assistant_start, prompt_length),
    }
    output = {
        name: [
            position
            for position in range(start, end)
            if position not in excluded
        ]
        for name, (start, end) in bounds.items()
    }
    broad_other = {
        position
        for position in range(prompt_length)
        if position not in excluded
    }
    flattened = [
        position for name in REFINED for position in output[name]
    ]
    if set(flattened) != broad_other or len(flattened) != len(set(flattened)):
        raise RuntimeError("refined region partition drift")
    return {
        "counts": {name: len(output[name]) for name in REFINED},
        "boundaries": {
            "first_fact": first_fact,
            "query_start": query_start,
            "instruction_start": instruction_start,
            "assistant_start": assistant_start,
            "prompt_length": prompt_length,
        },
    }


def main():
    payload = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        cases = read_jsonl(
            OUT_ROOT / "protocol" / model_name / "cases.jsonl"
        )
        rows = [regions(tokenizer, case) for case in cases]
        payload[model_name] = {
            "n": len(rows),
            "all_valid": True,
            "count_ranges": {
                name: [
                    min(row["counts"][name] for row in rows),
                    max(row["counts"][name] for row in rows),
                ]
                for name in REFINED
            },
            "boundary_examples": rows[:4],
        }
    path = (
        ROOT
        / "tests"
        / "glm5_temp"
        / "phase1002_region_boundary_probe.json"
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
