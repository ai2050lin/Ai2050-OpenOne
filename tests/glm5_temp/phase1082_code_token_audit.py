#!/usr/bin/env python3
"""Behavior-free token audit for Phase1082 shared output codes."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
import phase1051_natural_behavior_protocol as behavior


MODELS = ("qwen3", "glm4", "deepseek7b")
CANDIDATES = (
    "alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "zeta",
    "amber", "bronze", "coral", "denim", "emerald", "gold", "hazel", "ivory",
    "atlas", "beacon", "crown", "drum", "eagle", "flame", "harbor", "island",
)
PROMPT = (
    "Codebook: alpha means supported; beta means contradicted.\n"
    "Case A: Evidence and claim.\n"
    "Request: Give the status code for case A.\n"
    "Return only the code.\nCompletion:"
)


def main() -> None:
    output = {}
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        rendered = behavior.render_native(
            tokenizer, model, PROMPT, with_system=False
        )
        rendered += "Completion:"
        rows = {}
        for candidate in CANDIDATES:
            ids = behavior.continuation_ids(
                tokenizer, rendered, " ", candidate
            )
            rows[candidate] = {
                "ids": [int(value) for value in ids],
                "width": len(ids),
                "first": int(ids[0]),
            }
        output[model] = rows
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
