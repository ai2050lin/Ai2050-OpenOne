#!/usr/bin/env python3
"""Independent lexical pilot for the redesigned Phase400 field prompt."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402
from phase371c_behavior_qualification import generate_batch  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase400_partial_order/field_contract_pilot"
MODELS = ("qwen3", "glm4", "deepseek7b")
PILOT_GROUPS = (
    ("Alden", "Brisa", "cobalt", "willow", "marble", "velvet"),
    ("Cato", "Della", "lantern", "meadow", "quartz", "ribbon"),
    ("Eamon", "Fara", "saffron", "timber", "violet", "wicker"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def raw_prompt(entity_a: str, entity_b: str, value_a: str, value_b: str, relation: int, order: int, query: int) -> tuple[str, str, str]:
    entities = [entity_a, entity_b]
    values = [value_a, value_b]
    if relation:
        entities = [entity_b, entity_a]
    if order:
        entities = list(reversed(entities))
        values = list(reversed(values))
    rows = [f"- record {entities[i]}: item = {values[i]}" for i in range(2)]
    queried = entity_b if query else entity_a
    binding = dict(zip(entities, values))
    target = binding[queried]
    distractor = next(value for value in values if value != target)
    prompt = (
        "Reference table:\n"
        + "\n".join(rows)
        + f"\nQuestion: What item value is stored for record {queried}?\n"
        + "Return exactly the lowercase item value and nothing else.\nAnswer:"
    )
    return prompt, target, distractor


def cases_for(loaded: Any, model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(PILOT_GROUPS):
        entity_a, entity_b, x1, x2, y1, y2 = group
        for axis, values in (("X", (x1, x2)), ("Y", (y1, y2))):
            for relation in (0, 1):
                for order in (0, 1):
                    for query in (0, 1):
                        raw, target, distractor = raw_prompt(
                            entity_a, entity_b, *values, relation, order, query
                        )
                        prompt, _add_special, _phase = interface_prompt(
                            loaded.tokenizer, model, raw, "answer_aligned_chat"
                        )
                        rows.append(
                            {
                                "prompt": prompt,
                                "target": target,
                                "target_aliases": [target],
                                "distractors": [distractor],
                                "group_index": group_index,
                                "condition": f"{axis}_R{relation}_O{order}_Q{query}",
                            }
                        )
    return rows


@torch.inference_mode()
def run(model: str) -> dict[str, Any]:
    loaded = None
    try:
        loaded = load_probe_model(model)
        cases = cases_for(loaded, model)
        generated = []
        for start in range(0, len(cases), 8):
            batch = cases[start : start + 8]
            results = generate_batch(loaded, batch, 12)
            generated.extend({**case, **result} for case, result in zip(batch, results, strict=True))
        groups = []
        for group_index in range(len(PILOT_GROUPS)):
            selected = [row for row in generated if row["group_index"] == group_index]
            groups.append(
                {
                    "group_index": group_index,
                    "case_count": len(selected),
                    "strict_correct_count": sum(row["strict_behavior_correct"] for row in selected),
                    "complete": all(row["strict_behavior_correct"] for row in selected),
                }
            )
        payload = {
            "schema_version": "74.0.0-pilot",
            "phase_id": "Phase400-FieldContractPilot",
            "created_at": now(),
            "model": model,
            "case_count": len(generated),
            "strict_correct_count": sum(row["strict_behavior_correct"] for row in generated),
            "complete_group_count": sum(row["complete"] for row in groups),
            "group_count": len(groups),
            "groups": groups,
            "formal_denominator_overlap": False,
            "valid": len(generated) == 48,
        }
        write_json(OUT / model / "complete.json", payload)
        write_json(
            OUT / model / "private_rows.json",
            {"rows": generated, "private": True},
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    run(args.model)
