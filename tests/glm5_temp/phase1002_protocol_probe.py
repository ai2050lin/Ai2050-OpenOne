#!/usr/bin/env python3
"""Calibration-only probe for a fixed multi-token answer surface."""
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
    ("Carol", "David", "green", "yellow", "David", "yellow"),
    ("Emma", "Frank", "blue", "green", "Emma", "blue"),
    ("Grace", "Henry", "yellow", "red", "Henry", "red"),
)


def prompt(e0: str, e1: str, c0: str, c1: str, query: str) -> str:
    return (
        f"Marker ledger: {e0} is linked to {c0}; "
        f"{e1} is linked to {c1}.\n"
        f"What marker color is linked to {query}?\n"
        "Answer with exactly four words in this form: "
        "The marker is [color]. Replace [color] with the answer."
    )


def run_model(model_name: str) -> dict:
    model = tokenizer = None
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        tokenizer.padding_side = "left"
        effective_eos = eos_ids(model, tokenizer)
        rows = []
        for e0, e1, c0, c1, query, gold in CALIBRATION:
            raw = prompt(e0, e1, c0, c1, query)
            rendered = render_chat(tokenizer, model_name, raw)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention = torch.ones_like(input_ids)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention,
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
                for value in generated.sequences[0, len(ids):].detach().cpu()
            ]
            rows.append({
                "gold": gold,
                "suffix_ids": suffix,
                "tokens": [
                    tokenizer.decode(
                        [value],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    for value in suffix
                ],
                "text": tokenizer.decode(
                    suffix,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "expected_text_ids": [
                    int(value)
                    for value in tokenizer.encode(
                        f"The marker is {gold}.",
                        add_special_tokens=False,
                    )
                ],
            })
        return {
            "model": model_name,
            "rows": rows,
            "eos_ids": effective_eos,
        }
    finally:
        if model is not None:
            release_model(model)


def main() -> None:
    payload = [run_model(name) for name in MODELS]
    path = ROOT / "tests" / "glm5_temp" / "phase1002_protocol_probe.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
