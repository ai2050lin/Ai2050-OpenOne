#!/usr/bin/env python3
"""BF16, non-quantized, sequential model loading for C043."""
from __future__ import annotations

import gc
from typing import Any

import torch

from model_utils import MODEL_CONFIGS, load_model, release_model


MODELS = ("qwen3", "glm4", "deepseek7b")


def parameter_dtype_counts(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype).replace("torch.", "")
        counts[key] = counts.get(key, 0) + int(parameter.numel())
    return counts


def quantization_audit(model) -> dict[str, Any]:
    module_names = [type(module).__name__.lower() for module in model.modules()]
    suspicious = sorted({
        name for name in module_names
        if "8bit" in name or "4bit" in name or "bitsandbytes" in name
    })
    dtypes = parameter_dtype_counts(model)
    return {
        "parameter_dtypes": dtypes,
        "suspicious_quantized_module_classes": suspicious,
        "has_bf16_parameters": dtypes.get("bfloat16", 0) > 0,
        "has_quantized_modules": bool(suspicious),
    }


def load_bf16(model_name: str):
    """Load exactly one local model in BF16 without weight quantization."""
    if model_name == "qwen3":
        model, tokenizer, device = load_model(
            model_name,
            dtype=torch.bfloat16,
            use_8bit=False,
        )
        return model, tokenizer, device, {
            "placement": "full_cuda",
            "max_memory": None,
            "parameter_dtypes": parameter_dtype_counts(model),
            "quantization": "none",
        }

    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = MODEL_CONFIGS[model_name]["path"]
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_memory = {0: "11GiB", "cpu": "24GiB"}
    print(f"[phase1332-bf16] loading {model_name} with {max_memory}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    placement = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    return model, tokenizer, device, {
        "placement": "accelerate_auto_cpu_gpu",
        "max_memory": {"cuda:0": "11GiB", "cpu": "24GiB"},
        "device_map": placement,
        "parameter_dtypes": parameter_dtype_counts(model),
        "quantization": "none",
    }


def release_bf16(model) -> None:
    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
