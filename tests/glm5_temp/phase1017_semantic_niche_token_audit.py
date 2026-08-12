#!/usr/bin/env python3
"""Audit candidate Phase1017 words across all local tokenizers."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase548_shared_attention_compute_protocol import tokenizer_for


MODELS = ("qwen3", "glm4", "deepseek7b")
WORDS = {
    "bank": {
        "labels": ("river", "finance"),
        "cues": (
            "water", "shore", "stream", "coast",
            "money", "loan", "credit", "lender",
        ),
    },
    "bat": {
        "labels": ("animal", "sports"),
        "cues": (
            "cave", "wing", "night", "mammal",
            "baseball", "pitcher", "stadium", "inning",
        ),
    },
    "crane": {
        "labels": ("bird", "machine"),
        "cues": (
            "feather", "marsh", "nest", "beak",
            "building", "lifting", "steel", "hoist",
        ),
    },
    "seal": {
        "labels": ("animal", "stamp"),
        "cues": (
            "ocean", "flipper", "marine", "whisker",
            "wax", "document", "letter", "signature",
        ),
    },
    "spring": {
        "labels": ("season", "coil"),
        "cues": (
            "flower", "April", "warm", "bloom",
            "metal", "tension", "compress", "spiral",
        ),
    },
    "club": {
        "labels": ("group", "stick"),
        "cues": (
            "member", "meeting", "society", "join",
            "swing", "hit", "golf", "wooden",
        ),
    },
    "match": {
        "labels": ("contest", "flame"),
        "cues": (
            "game", "score", "tournament", "opponent",
            "fire", "candle", "ignite", "spark",
        ),
    },
    "port": {
        "labels": ("harbor", "computer"),
        "cues": (
            "ship", "dock", "vessel", "coast",
            "network", "socket", "server", "cable",
        ),
    },
    "file": {
        "labels": ("document", "tool"),
        "cues": (
            "folder", "paper", "record", "office",
            "metal", "teeth", "workshop", "smooth",
        ),
    },
    "mouse": {
        "labels": ("animal", "device"),
        "cues": (
            "cheese", "tail", "rodent", "cage",
            "cursor", "click", "computer", "button",
        ),
    },
    "jam": {
        "labels": ("food", "traffic"),
        "cues": (
            "bread", "berry", "toast", "sweet",
            "cars", "road", "highway", "congestion",
        ),
    },
    "pitch": {
        "labels": ("throw", "sound"),
        "cues": (
            "baseball", "ball", "pitcher", "field",
            "music", "tone", "frequency", "voice",
        ),
    },
}


def main() -> None:
    result = {}
    all_tokens = sorted({
        value
        for word, spec in WORDS.items()
        for value in (word, *spec["labels"], *spec["cues"])
    })
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        rows = {}
        for value in all_tokens:
            plain = tokenizer.encode(value, add_special_tokens=False)
            leading = tokenizer.encode(" " + value, add_special_tokens=False)
            rows[value] = {
                "plain_ids": [int(item) for item in plain],
                "leading_space_ids": [int(item) for item in leading],
                "plain_single": len(plain) == 1,
                "leading_space_single": len(leading) == 1,
            }
        result[model_name] = rows

    viable = {}
    for word, spec in WORDS.items():
        tokens = (word, *spec["labels"], *spec["cues"])
        viable[word] = {
            model_name: all(
                result[model_name][token]["leading_space_single"]
                for token in tokens
            )
            for model_name in MODELS
        }

    output = {
        "models": MODELS,
        "words": WORDS,
        "viable": viable,
        "tokens": result,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
