#!/usr/bin/env python3
"""Calibration-only comparison for the ambiguous Phase 1002 color-first template."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase548_shared_attention_compute_protocol import render_chat


MODELS = ("qwen3", "glm4", "deepseek7b")
CALIBRATION = (
    ("Alice", "Bob", "red", "blue", "Alice", "red"),
    ("Alice", "Bob", "red", "blue", "Bob", "blue"),
    ("Carol", "David", "green", "yellow", "Carol", "green"),
    ("Carol", "David", "green", "yellow", "David", "yellow"),
    ("Emma", "Frank", "blue", "green", "Emma", "blue"),
    ("Emma", "Frank", "blue", "green", "Frank", "green"),
    ("Grace", "Henry", "yellow", "red", "Grace", "yellow"),
    ("Grace", "Henry", "yellow", "red", "Henry", "red"),
)
TEMPLATES = {
    "explicit_assigned": (
        "The {c0} marker is assigned to {e0}. "
        "The {c1} marker is assigned to {e1}.\n"
        "Which marker color is assigned to {query}?"
    ),
    "explicit_color_for": (
        "Color-first record: {c0} is the marker color for {e0}; "
        "{c1} is the marker color for {e1}.\n"
        "Report the marker color for {query}."
    ),
    "explicit_belongs": (
        "Pairing record: marker color {c0} belongs to {e0}; "
        "marker color {c1} belongs to {e1}.\n"
        "Look up the marker color belonging to {query}."
    ),
}
INSTRUCTION = (
    "Answer with exactly four words in this form: "
    "The marker is [color]. Replace [color] with the answer."
)


def run_model(model_name: str) -> dict:
    model = tokenizer = None
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        effective_eos = eos_ids(model, tokenizer)
        rows = []
        for template_name, template in TEMPLATES.items():
            for e0, e1, c0, c1, query, gold in CALIBRATION:
                raw = (
                    template.format(
                        e0=e0, e1=e1, c0=c0, c1=c1, query=query
                    )
                    + "\n"
                    + INSTRUCTION
                )
                rendered = render_chat(tokenizer, model_name, raw)
                ids = tokenizer.encode(rendered, add_special_tokens=False)
                input_ids = torch.tensor(
                    [ids], dtype=torch.long, device=device
                )
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        max_new_tokens=12,
                        eos_token_id=effective_eos,
                        pad_token_id=int(tokenizer.pad_token_id),
                        return_dict_in_generate=True,
                    )
                suffix = [
                    int(value)
                    for value in output.sequences[0, len(ids):]
                    .detach()
                    .cpu()
                ]
                text = tokenizer.decode(
                    suffix,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                expected = f"The marker is {gold}."
                rows.append({
                    "template": template_name,
                    "query": query,
                    "gold": gold,
                    "expected": expected,
                    "text": text,
                    "exact": text == expected,
                })
        summaries = {}
        for template_name in TEMPLATES:
            subset = [
                row for row in rows if row["template"] == template_name
            ]
            summaries[template_name] = {
                "n": len(subset),
                "exact_rate": sum(row["exact"] for row in subset)
                / len(subset),
            }
        return {
            "model": model_name,
            "summaries": summaries,
            "rows": rows,
        }
    finally:
        if model is not None:
            release_model(model)


def main() -> None:
    payload = {
        "purpose": "calibration_only_template2_repair",
        "selection_rule": (
            "Choose the template with the highest worst-model exact rate; "
            "break ties by the listed order."
        ),
        "models": [run_model(name) for name in MODELS],
    }
    path = (
        ROOT
        / "tests"
        / "glm5_temp"
        / "phase1002_template2_repair_probe.json"
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
