#!/usr/bin/env python3
"""Probe tokenizer compatibility for the Phase1001 cross-model topology test."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for


MODELS = ("qwen3", "glm4", "deepseek7b")
NAMES = (
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Jack", "Kelly", "Paul", "Ruby", "Sam", "Blake", "Leo", "Will",
    "Iris", "Liam", "Maya", "Nora", "Oscar", "Quinn", "Tina", "Uma",
)
COLORS = ("red", "blue", "green", "yellow")


def one_token(tokenizer, text: str) -> int | None:
    values = tokenizer.encode(text, add_special_tokens=False)
    return int(values[0]) if len(values) == 1 else None


def main() -> None:
    result = {}
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        prompt = (
            "Records: Alice carries the red marker. Bob carries the blue marker.\n"
            "Question: What color marker does Alice carry?\n"
            "Answer with exactly one color word."
        )
        rendered = render_chat(tokenizer, model, prompt)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        name_ids = {name: one_token(tokenizer, " " + name) for name in NAMES}
        prompt_color_ids = {
            color: one_token(tokenizer, " " + color) for color in COLORS
        }
        candidate_ids = {color: one_token(tokenizer, color) for color in COLORS}
        alice_positions = [
            index for index, value in enumerate(ids)
            if value == name_ids["Alice"]
        ]
        bob_positions = [
            index for index, value in enumerate(ids)
            if value == name_ids["Bob"]
        ]
        result[model] = {
            "prompt_token_count": len(ids),
            "one_token_name_count": sum(value is not None for value in name_ids.values()),
            "one_token_names": [
                name for name, value in name_ids.items() if value is not None
            ],
            "prompt_color_ids": prompt_color_ids,
            "candidate_ids": candidate_ids,
            "alice_positions": alice_positions,
            "bob_positions": bob_positions,
            "answer_boundary": len(ids) - 1,
            "rendered_suffix": rendered[-120:],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
