from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_registry import ModelSpec, get_model_spec


@dataclass
class LoadedModel:
    key: str
    spec: ModelSpec
    model: Any
    tokenizer: Any
    input_device: torch.device


def load_probe_model(model_key: str) -> LoadedModel:
    spec = get_model_spec(model_key)
    if not spec.local_dir.exists():
        raise FileNotFoundError(
            f"Missing local model dir: {spec.local_dir}. "
            f"Run: python tests/gpt5/download_models.py {model_key}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    default_dtype_by_model = {
        "qwen3": "float16",
        "glm4": "float16",
        "deepseek7b": "bfloat16",
    }
    dtype_name = os.environ.get(
        "PROBE_TORCH_DTYPE", default_dtype_by_model.get(model_key, "bfloat16")
    ).strip().lower()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in dtype_map:
        valid = ", ".join(sorted(dtype_map))
        raise ValueError(f"Unsupported PROBE_TORCH_DTYPE={dtype_name!r}. Valid: {valid}")

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[dtype_name],
        "trust_remote_code": spec.trust_remote_code,
        "local_files_only": True,
        "attn_implementation": spec.attn_implementation,
    }
    if spec.load_strategy == "cuda":
        kwargs["device_map"] = "cpu"
    else:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: "22GiB", "cpu": "96GiB"}

    model = AutoModelForCausalLM.from_pretrained(str(spec.local_dir), **kwargs)
    if spec.load_strategy == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    input_device = next(model.parameters()).device
    return LoadedModel(
        key=model_key,
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        input_device=input_device,
    )


def release_loaded(loaded: LoadedModel | None) -> None:
    if loaded is not None:
        del loaded.model
        del loaded.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_layers(model: Any) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise TypeError(f"Cannot locate transformer layers for {type(model).__name__}")


def encode(loaded: LoadedModel, prompt: str) -> dict[str, torch.Tensor]:
    batch = loaded.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96)
    return {k: v.to(loaded.input_device) for k, v in batch.items()}


def first_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Could not tokenize target text: {text!r}")
    return int(ids[0])


def vram_gb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9


def local_model_status() -> dict[str, dict[str, str | bool]]:
    from model_registry import MODEL_SPECS

    status = {}
    for key, spec in MODEL_SPECS.items():
        status[key] = {
            "repo_id": spec.repo_id,
            "local_dir": str(spec.local_dir),
            "exists": spec.local_dir.exists(),
            "has_config": (spec.local_dir / "config.json").exists(),
            "has_safetensors": any(Path(spec.local_dir).glob("*.safetensors")),
        }
    return status
