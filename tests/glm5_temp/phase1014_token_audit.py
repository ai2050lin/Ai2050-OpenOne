"""Audit reserve vocabulary for the Phase1014 balanced-factor protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase548_shared_attention_compute_protocol import tokenizer_for


MODELS = ("qwen3", "glm4", "deepseek7b")
NAMES = (
    "Paul",
    "Peter",
    "Philip",
    "Ray",
    "Robert",
    "Ryan",
    "Sam",
    "Scott",
    "Sean",
    "Simon",
    "Stephen",
    "Steve",
    "Thomas",
    "Tim",
    "Victor",
    "Walter",
    "William",
    "Zach",
    "Adam",
    "Brian",
    "Colin",
    "Daniel",
    "Eric",
    "Frank",
    "George",
    "Henry",
    "Ian",
    "Jack",
    "James",
    "Jason",
    "Jeff",
    "John",
    "Kevin",
    "Mark",
    "Martin",
    "Mike",
    "Nathan",
    "Noah",
    "Patrick",
    "Richard",
)
WORDS = (
    "clear",
    "quartz",
    "highest",
    "lowest",
    "present",
    "missing",
    "agent",
    "patient",
    "red",
    "blue",
    "left",
    "right",
    "yes",
    "no",
)


def main() -> None:
    result = {"models": {}, "common_single_token_names": []}
    safe_by_model = {}
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        names = {
            name: tokenizer.encode(
                " " + name, add_special_tokens=False
            )
            for name in NAMES
        }
        words = {
            word: tokenizer.encode(
                " " + word, add_special_tokens=False
            )
            for word in WORDS
        }
        safe = sorted(
            name for name, ids in names.items() if len(ids) == 1
        )
        safe_by_model[model] = set(safe)
        result["models"][model] = {
            "single_token_names": safe,
            "unsafe_names": {
                name: ids for name, ids in names.items() if len(ids) != 1
            },
            "words": words,
            "all_words_single_token": all(
                len(ids) == 1 for ids in words.values()
            ),
        }
    result["common_single_token_names"] = sorted(
        set.intersection(*safe_by_model.values())
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
