#!/usr/bin/env python3
"""Audit candidate Phase1015 role tokens across all local tokenizers."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import tokenizer_for


WORDS = (
    "above",
    "leads",
    "below",
    "under",
    "present",
    "missing",
    "helped",
    "guided",
    "thanked",
    "called",
    "red",
    "blue",
    "left",
    "right",
    "highest",
    "lowest",
    "tallest",
    "shortest",
    "upper",
    "lower",
    "listed",
    "absent",
    "available",
    "agent",
    "patient",
    "actor",
    "recipient",
    "source",
    "receiver",
    "ruby",
    "azure",
    "crimson",
    "navy",
    "western",
    "eastern",
    "west",
    "east",
    "first",
    "second",
    "former",
    "latter",
    "clear",
    "quartz",
)


def main() -> None:
    models = ("qwen3", "glm4", "deepseek7b")
    widths = {}
    for model in models:
        tokenizer = tokenizer_for(model)
        widths[model] = {
            word: tokenizer.encode(
                " " + word,
                add_special_tokens=False,
            )
            for word in WORDS
        }
    for word in WORDS:
        cells = " ".join(
            f"{model}={widths[model][word]}"
            for model in models
        )
        status = "PASS" if all(
            len(widths[model][word]) == 1 for model in models
        ) else "FAIL"
        print(f"{status:4s} {word:12s} {cells}")


if __name__ == "__main__":
    main()
