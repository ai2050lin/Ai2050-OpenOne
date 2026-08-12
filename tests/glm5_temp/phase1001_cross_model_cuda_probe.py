#!/usr/bin/env python3
"""Smoke-test 8-bit CUDA loading and component hooks for one local model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, load_model, release_model
from phase548_shared_attention_compute_protocol import render_chat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("qwen3", "glm4", "deepseek7b"))
    args = parser.parse_args()
    model = tokenizer = None
    captures = []
    try:
        model, tokenizer, device = load_model(
            args.model, dtype=torch.bfloat16, use_8bit=True
        )
        layers = get_layers(model)
        text = render_chat(
            tokenizer,
            args.model,
            "Records: Alice carries the red marker. Bob carries the blue marker.\n"
            "Question: What color marker does Alice carry?\n"
            "Answer with exactly one color word.",
        )
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        def hook(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            captures.append(
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
            )
            return output

        handle = layers[-1].self_attn.register_forward_hook(hook)
        with torch.inference_mode():
            result = model(**encoded, use_cache=False, return_dict=True)
        handle.remove()
        print(json.dumps({
            "model": args.model,
            "class": type(model).__name__,
            "layer_count": len(layers),
            "hidden_size": int(model.config.hidden_size),
            "quantized_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "input_device": str(device),
            "logits_shape": list(result.logits.shape),
            "capture": captures,
            "cuda_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        }, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_model(model)


if __name__ == "__main__":
    main()
