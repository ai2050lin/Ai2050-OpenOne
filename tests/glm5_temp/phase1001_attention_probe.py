#!/usr/bin/env python3
"""Temporary Qwen3 attention-shape probe for Phase 1001."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model


def main() -> None:
    case_path = (
        ROOT
        / "tests"
        / "glm5"
        / "result"
        / "phase1000_factorial_binding_scpg"
        / "protocol"
        / "cases.jsonl"
    )
    case = json.loads(case_path.read_text(encoding="utf-8").splitlines()[0])
    model = tokenizer = None
    handles = []
    captured = {}
    try:
        model, tokenizer, device = load_model(
            "qwen3", dtype=torch.bfloat16, use_8bit=False
        )
        layers = get_layers(model)
        layer = layers[24]

        def o_pre(module, args):
            captured["o_input"] = tuple(args[0].shape)

        def v_hook(module, args, output):
            captured["v_output"] = tuple(output.shape)

        def attn_hook(module, args, output):
            captured["attn_type"] = type(output).__name__
            captured["attn_len"] = len(output) if isinstance(output, tuple) else None
            if isinstance(output, tuple):
                captured["attn_shapes"] = [
                    tuple(item.shape) if torch.is_tensor(item) else str(type(item))
                    for item in output
                ]
            else:
                captured["attn_shapes"] = [tuple(output.shape)]

        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(o_pre))
        handles.append(layer.self_attn.v_proj.register_forward_hook(v_hook))
        handles.append(layer.self_attn.register_forward_hook(attn_hook))
        input_ids = torch.tensor([case["input_ids"]], dtype=torch.long, device=device)
        attention = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        captured["model_attention_shape"] = tuple(output.attentions[24].shape)
        captured["config"] = {
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "head_dim": model.config.head_dim,
            "hidden_size": model.config.hidden_size,
        }
        print(json.dumps(captured, ensure_ascii=False, indent=2))
    finally:
        for handle in reversed(handles):
            handle.remove()
        if model is not None:
            release_model(model)


if __name__ == "__main__":
    main()
