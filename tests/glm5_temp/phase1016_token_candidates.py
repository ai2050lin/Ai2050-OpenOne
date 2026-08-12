#!/usr/bin/env python3
"""Audit Phase1016 factorial-query vocabulary across local tokenizers."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import tokenizer_for


WORDS = (
    "highest",
    "lowest",
    "tallest",
    "shortest",
    "upper",
    "lower",
    "top",
    "bottom",
    "maximum",
    "minimum",
    "greater",
    "lesser",
    "present",
    "missing",
    "listed",
    "absent",
    "available",
    "unavailable",
    "included",
    "excluded",
    "known",
    "unknown",
    "agent",
    "patient",
    "actor",
    "recipient",
    "source",
    "receiver",
    "sender",
    "target",
    "leader",
    "follower",
    "red",
    "blue",
    "crimson",
    "azure",
    "ruby",
    "navy",
    "rose",
    "cyan",
    "scarlet",
    "indigo",
    "left",
    "right",
    "western",
    "eastern",
    "west",
    "east",
    "port",
    "starboard",
    "leftward",
    "rightward",
    "leftmost",
    "rightmost",
    "westward",
    "eastward",
    "westerly",
    "easterly",
    "occidental",
    "oriental",
    "sinistral",
    "dextral",
    "larboard",
)


def main() -> None:
    models = ("qwen3", "glm4", "deepseek7b")
    widths = {}
    for model in models:
        tokenizer = tokenizer_for(model)
        widths[model] = {
            word: tokenizer.encode(" " + word, add_special_tokens=False)
            for word in WORDS
        }
    for word in WORDS:
        status = "PASS" if all(
            len(widths[model][word]) == 1 for model in models
        ) else "FAIL"
        detail = " ".join(
            f"{model}={widths[model][word]}" for model in models
        )
        print(f"{status:4s} {word:12s} {detail}")


if __name__ == "__main__":
    main()
