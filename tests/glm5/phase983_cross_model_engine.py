#!/usr/bin/env python3
"""Shared fail-closed CUDA generation engine for future cross-model studies.

This module is engineering infrastructure only.  It does not define a Phase983
dataset, scientific gate, preregistration, or admission.  CPU inspection never
loads model weights.  CUDA is reachable only through the explicit
``--gpu-smoke MODEL`` command or by a future runner calling
``load_model_adapter`` directly.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DynamicCache,
    GenerationConfig,
)
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


ROOT = Path(__file__).resolve().parents[2]
GPT5 = ROOT / "tests" / "gpt5"
if str(GPT5) not in sys.path:
    sys.path.insert(0, str(GPT5))

from model_registry import get_model_spec  # noqa: E402


ENGINE_SCHEMA_VERSION = 1
ENGINE_NAMESPACE = "phase983-cross-model-engine-v1"
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
DEFAULT_BATCH_SIZE = 8
GPU_SMOKE_MAX_NEW_TOKENS = 24
SAMPLING_CONTRACT = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
}
QUANTIZATION_CONTRACT = {
    "backend": "bitsandbytes",
    "load_in_8bit": True,
    "llm_int8_enable_fp32_cpu_offload": False,
    "non_quantized_dtype": "torch.bfloat16",
    "device_map": "auto",
    "attn_implementation": "sdpa",
    "local_files_only": True,
}
SMALL_IDENTITY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_eos_values(value: Any, label: str) -> list[int]:
    """Normalize one EOS source without accepting bools or nested junk."""
    if value is None:
        return []
    if isinstance(value, bool):
        raise RuntimeError(f"{label} EOS cannot be bool")
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        raise RuntimeError(f"{label} EOS has unsupported type: {type(value).__name__}")
    output: list[int] = []
    for item in values:
        require(isinstance(item, int) and not isinstance(item, bool),
                f"{label} EOS contains a non-integer")
        require(item >= 0, f"{label} EOS contains a negative ID")
        output.append(int(item))
    return output


def eos_identity_from_sources(sources: dict[str, Any]) -> dict[str, Any]:
    require(bool(sources), "EOS source registry is empty")
    normalized = {
        label: sorted(set(_plain_eos_values(value, label)))
        for label, value in sources.items()
    }
    effective = sorted({item for values in normalized.values() for item in values})
    require(bool(effective), "effective EOS union is empty")
    return {
        "sources": normalized,
        "effective_eos_token_ids": effective,
        "multiple_effective_eos": len(effective) > 1,
    }


def tokenizer_pad_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            "tokenizer must provide a non-negative integer pad_token_id")
    return int(value)


def stable_pair_seed(
    dataset_namespace: str, model_key: str, seed_key: str, stream: int,
    arm: str | None = None,
) -> int:
    """Return a per-model pair seed; ``arm`` is validated but excluded.

    A/B arms for the same model/item/stream therefore receive the same random
    stream.  The model key is included because token vocabularies and sampling
    distributions are not a meaningful common-random-number system across
    different models.
    """
    require(isinstance(dataset_namespace, str) and dataset_namespace.strip(),
            "dataset namespace must be non-empty")
    require(model_key in MODEL_ORDER, f"unsupported model key: {model_key}")
    require(isinstance(seed_key, str) and seed_key.strip(), "seed_key must be non-empty")
    require(isinstance(stream, int) and not isinstance(stream, bool) and stream >= 0,
            "stream must be a non-negative integer")
    if arm is not None:
        require(isinstance(arm, str) and arm.strip(), "arm must be non-empty")
    payload = {
        "dataset_namespace": dataset_namespace,
        "engine_namespace": ENGINE_NAMESPACE,
        # The field name is retained in the frozen engine namespace for
        # compatibility; the value is the dataset-provided semantic twin
        # seed key, so original/swapped and A/B form one 2x2 CRN block.
        "item_id": seed_key,
        "model_key": model_key,
        "stream": stream,
    }
    value = int.from_bytes(
        hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()[:8], "big",
    )
    return int(value % (2**31 - 1))


@dataclass(frozen=True)
class RenderedPrefix:
    user_text: str
    rendered_text: str
    input_ids: tuple[int, ...]
    rendered_sha256: str


@dataclass(frozen=True)
class SamplingRequest:
    item_id: str
    stream: int
    arm: str
    user_text: str
    seed_key: str | None = None


@dataclass(frozen=True)
class SampledRow:
    item_id: str
    seed_key: str
    stream: int
    arm: str
    model_key: str
    pair_seed: int
    rendered_prefix_sha256: str
    input_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    first_eos_token_id: int | None
    first_eos_absorbing: bool


@dataclass
class _InspectionBundle:
    model_key: str
    tokenizer: Any
    config: Any
    generation_config: Any
    identity: dict[str, Any]


@dataclass
class ModelAdapter:
    model_key: str
    tokenizer: Any
    config: Any
    generation_config: Any
    model: Any
    input_device: torch.device
    eos_identity: dict[str, Any]
    pad_token_id: int
    identity: dict[str, Any]

    def render_user(self, user_text: str) -> RenderedPrefix:
        return render_native_user(self.tokenizer, user_text)


def render_native_user(tokenizer: Any, user_text: str) -> RenderedPrefix:
    """Render exactly one native user message without a thinking-mode kwarg."""
    require(isinstance(user_text, str) and user_text.strip(),
            "native user text must be non-empty")
    messages = [{"role": "user", "content": user_text}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    require(isinstance(rendered, str) and rendered,
            "native chat template returned empty/non-string text")
    encoded = tokenizer(
        rendered, add_special_tokens=False, return_attention_mask=False,
    )
    ids = getattr(encoded, "input_ids", None)
    require(isinstance(ids, (list, tuple)) and ids,
            "native rendered prefix produced no token IDs")
    require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in ids), "rendered prefix contains invalid token IDs")
    return RenderedPrefix(
        user_text=user_text,
        rendered_text=rendered,
        input_ids=tuple(int(value) for value in ids),
        rendered_sha256=sha256_bytes(rendered.encode("utf-8")),
    )


def native_generation_prefill_identity(tokenizer: Any) -> dict[str, Any]:
    """Describe the model-native assistant prefill without enabling controls."""
    probe = "PHASE983_NATIVE_GENERATION_PREFILL_PROBE"
    messages = [{"role": "user", "content": probe}]
    without = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    with_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    require(isinstance(without, str) and isinstance(with_prompt, str)
            and with_prompt.startswith(without),
            "native generation prompt is not a strict textual suffix")
    suffix = with_prompt[len(without):]
    require(bool(suffix), "native generation prefill suffix is empty")
    without_ids = list(tokenizer(
        without, add_special_tokens=False, return_attention_mask=False,
    ).input_ids)
    with_ids = list(tokenizer(
        with_prompt, add_special_tokens=False, return_attention_mask=False,
    ).input_ids)
    suffix_ids = list(tokenizer(
        suffix, add_special_tokens=False, return_attention_mask=False,
    ).input_ids)
    require(with_ids[:len(without_ids)] == without_ids
            and with_ids[len(without_ids):] == suffix_ids and bool(suffix_ids),
            "native generation prefill is not a strict token suffix")
    return {
        "probe_text": probe,
        "without_generation_prompt_sha256": sha256_bytes(without.encode("utf-8")),
        "with_generation_prompt_sha256": sha256_bytes(
            with_prompt.encode("utf-8")),
        "assistant_prefill_text": suffix,
        "assistant_prefill_text_sha256": sha256_bytes(suffix.encode("utf-8")),
        "assistant_prefill_token_ids": [int(value) for value in suffix_ids],
        "assistant_prefill_token_ids_sha256": sha256_bytes(
            canonical_json(suffix_ids).encode("utf-8")),
    }


def _model_class_for_config(config: Any) -> type[Any]:
    try:
        model_class = AutoModelForCausalLM._model_mapping[type(config)]
    except Exception as exc:  # pragma: no cover - depends on installed transformers
        raise RuntimeError(
            f"no AutoModelForCausalLM mapping for {type(config).__name__}"
        ) from exc
    require(bool(getattr(model_class, "_supports_sdpa", False)),
            f"{model_class.__name__} does not declare SDPA support")
    signature = inspect.signature(model_class.forward)
    require("logits_to_keep" in signature.parameters,
            f"{model_class.__name__}.forward lacks logits_to_keep")
    return model_class


def _artifact_identity(model_key: str, local_dir: Path) -> dict[str, Any]:
    spec = get_model_spec(model_key)
    require(local_dir.resolve() == Path(spec.local_dir).resolve(),
            "model path differs from canonical registry")
    require(local_dir.is_dir(), f"missing local model directory: {local_dir}")
    small_files: dict[str, Any] = {}
    for name in SMALL_IDENTITY_FILES:
        path = local_dir / name
        if path.is_file():
            small_files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    require("config.json" in small_files and "tokenizer_config.json" in small_files,
            f"{model_key} lacks required config/tokenizer identity files")
    weight_files = sorted(local_dir.glob("*.safetensors"), key=lambda path: path.name)
    require(bool(weight_files), f"{model_key} has no local safetensors weights")
    weight_registry = [
        {"name": path.name, "size_bytes": path.stat().st_size}
        for path in weight_files
    ]
    payload = {
        "logical_name": model_key,
        "repo_id": spec.repo_id,
        "local_dir": str(local_dir.resolve()),
        "small_files": small_files,
        "weight_file_registry": weight_registry,
        "weight_file_count": len(weight_registry),
        "weight_total_bytes": sum(item["size_bytes"] for item in weight_registry),
        "weight_note": (
            "Shard names/sizes are engineering identity only; a future formal "
            "protocol must seal full artifact hashes independently."
        ),
    }
    return {**payload, "engineering_identity_sha256": sha256_bytes(
        canonical_json(payload).encode("utf-8"))}


def _load_inspection_bundle(model_key: str) -> _InspectionBundle:
    require(model_key in MODEL_ORDER, f"unsupported model key: {model_key}")
    spec = get_model_spec(model_key)
    local_dir = Path(spec.local_dir)
    artifact_identity = _artifact_identity(model_key, local_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_dir), trust_remote_code=spec.trust_remote_code,
        local_files_only=True, use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        str(local_dir), trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
    )
    generation_config = GenerationConfig.from_pretrained(
        str(local_dir), local_files_only=True,
    )
    model_class = _model_class_for_config(config)
    pad_id = tokenizer_pad_id(tokenizer)
    eos_identity = eos_identity_from_sources({
        "tokenizer": getattr(tokenizer, "eos_token_id", None),
        "config": getattr(config, "eos_token_id", None),
        "generation_config": getattr(generation_config, "eos_token_id", None),
    })
    require(pad_id < len(tokenizer), f"{model_key} pad ID is outside tokenizer")
    require(all(token_id < len(tokenizer) for token_id in
                eos_identity["effective_eos_token_ids"]),
            f"{model_key} effective EOS ID is outside tokenizer")
    template = getattr(tokenizer, "chat_template", None)
    require(isinstance(template, str) and template,
            f"{model_key} tokenizer lacks a native chat template")
    rendered_probe = render_native_user(tokenizer, "ENGINEERING INSPECTION ONLY: return A.")
    raw_special_ids = list(getattr(tokenizer, "all_special_ids", []) or [])
    require(raw_special_ids
            and all(isinstance(value, int) and not isinstance(value, bool)
                    and 0 <= value < len(tokenizer) for value in raw_special_ids),
            f"{model_key} tokenizer special-token registry is invalid")
    all_special_ids = sorted(set(int(value) for value in raw_special_ids))
    native_prefill = native_generation_prefill_identity(tokenizer)
    identity = {
        "schema_version": ENGINE_SCHEMA_VERSION,
        "model_key": model_key,
        "model_order_index": MODEL_ORDER.index(model_key),
        "artifact_identity": artifact_identity,
        "architecture": list(getattr(config, "architectures", []) or []),
        "model_type": getattr(config, "model_type", None),
        "model_class": model_class.__name__,
        "model_class_declares_sdpa": True,
        "model_forward_has_logits_to_keep": True,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_length": len(tokenizer),
        "chat_template_sha256": sha256_bytes(template.encode("utf-8")),
        "all_special_ids": all_special_ids,
        "native_generation_prefill": native_prefill,
        "native_single_user_probe": {
            "rendered_sha256": rendered_probe.rendered_sha256,
            "input_token_count": len(rendered_probe.input_ids),
        },
        "eos_identity": eos_identity,
        "pad_token_id": pad_id,
        "planned_quantization": dict(QUANTIZATION_CONTRACT),
        "weights_loaded": False,
        "gpu_used": False,
    }
    return _InspectionBundle(
        model_key=model_key, tokenizer=tokenizer, config=config,
        generation_config=generation_config, identity=identity,
    )


def inspect_model(model_key: str) -> dict[str, Any]:
    """Inspect local config/tokenizer only; model weights are never loaded."""
    return _load_inspection_bundle(model_key).identity


def inspect_all_models() -> dict[str, Any]:
    inspections = [inspect_model(model_key) for model_key in MODEL_ORDER]
    return {
        "schema_version": ENGINE_SCHEMA_VERSION,
        "model_order": list(MODEL_ORDER),
        "models": inspections,
        "weights_loaded": False,
        "gpu_used": False,
        "files_written": False,
    }


def _actual_quantization_identity(model: Any) -> dict[str, Any]:
    quantizer = getattr(model, "hf_quantizer", None)
    quant_config = getattr(quantizer, "quantization_config", None)
    config_flag = bool(getattr(quant_config, "load_in_8bit", False))
    model_flag = bool(getattr(model, "is_loaded_in_8bit", False))
    int8_module_count = sum(
        1 for module in model.modules()
        if type(module).__module__.startswith("bitsandbytes")
        and type(module).__name__ == "Linear8bitLt"
    )
    require(config_flag or model_flag,
            "loaded model does not report bitsandbytes 8-bit quantization")
    require(int8_module_count > 0, "loaded model has no Linear8bitLt modules")
    floating_dtypes = sorted({
        str(parameter.dtype) for parameter in model.parameters()
        if bool(getattr(parameter, "is_floating_point", lambda: False)())
    })
    require(floating_dtypes == ["torch.bfloat16"],
            f"non-quantized floating parameter dtypes changed: {floating_dtypes}")
    return {
        **QUANTIZATION_CONTRACT,
        "model_reports_loaded_in_8bit": model_flag,
        "quantizer_reports_load_in_8bit": config_flag,
        "linear8bitlt_module_count": int8_module_count,
        "floating_parameter_dtypes": floating_dtypes,
    }


def _validate_cuda_only_device_map(model: Any) -> dict[str, str]:
    """Reject offload using published placement or direct tensor residency.

    Transformers 5.12 may omit ``hf_device_map`` when ``device_map='auto'``
    resolves the complete model to one GPU.  In that case an empty optional
    metadata map is not evidence of offload; every parameter and non-empty
    buffer is inspected directly and must reside on one concrete CUDA device.
    """
    raw = getattr(model, "hf_device_map", None)
    if raw is None:
        raw = {}
    require(isinstance(raw, dict), "hf_device_map has an invalid type")
    if not raw:
        named_parameters = getattr(model, "named_parameters", None)
        named_buffers = getattr(model, "named_buffers", None)
        require(callable(named_parameters) and callable(named_buffers),
                "empty hf_device_map cannot be replaced by direct tensor audit")
        tensors = [
            (f"parameter:{name}", value)
            for name, value in named_parameters()
        ] + [
            (f"buffer:{name}", value)
            for name, value in named_buffers()
            if int(value.numel()) > 0
        ]
        require(bool(tensors), "direct tensor residency audit found no tensors")
        devices: set[str] = set()
        for name, tensor in tensors:
            require(not bool(getattr(tensor, "is_meta", False)),
                    f"meta tensor forbidden: {name}")
            device = torch.device(tensor.device)
            require(device.type == "cuda" and device.index is not None,
                    f"CPU/disk/meta/offload tensor forbidden: {name} -> {device}")
            devices.add(str(device))
        require(len(devices) == 1,
                f"direct tensor residency spans CUDA devices: {sorted(devices)}")
        return {"<direct_parameter_and_buffer_audit>": next(iter(devices))}
    normalized: dict[str, str] = {}
    for module_name, placement in raw.items():
        if isinstance(placement, bool):
            raise RuntimeError(
                f"invalid bool device placement for {module_name!r}: {placement!r}")
        if isinstance(placement, int):
            require(placement >= 0,
                    f"negative CUDA device index for {module_name!r}")
            device = torch.device(f"cuda:{placement}")
        else:
            try:
                device = torch.device(placement)
            except (TypeError, RuntimeError) as exc:
                raise RuntimeError(
                    f"unsupported device placement for {module_name!r}: {placement!r}"
                ) from exc
        require(device.type == "cuda" and device.index is not None,
                f"CPU/disk/meta/offload placement forbidden: "
                f"{module_name!r} -> {placement!r}")
        normalized[str(module_name)] = str(device)
    return normalized


def _input_device(model: Any) -> torch.device:
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    require(weight is not None and not bool(getattr(weight, "is_meta", False)),
            "input embedding weight is missing or meta")
    device = torch.device(weight.device)
    require(device.type == "cuda",
            f"input embeddings are not on CUDA under device_map=auto: {device}")
    return device


def load_model_adapter(model_key: str) -> ModelAdapter:
    """Explicitly load one int8/BF16/SDPA model; never retries or downgrades."""
    require(model_key in MODEL_ORDER, f"unsupported model key: {model_key}")
    require(torch.cuda.is_available(), "CUDA is required for model loading")
    bundle = _load_inspection_bundle(model_key)
    spec = get_model_spec(model_key)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    # There is deliberately no exception fallback to eager attention, another
    # dtype, CPU offload, 4-bit, or a smaller batch.
    model = AutoModelForCausalLM.from_pretrained(
        str(spec.local_dir),
        config=bundle.config,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    try:
        model.eval()
        actual_attn = getattr(model.config, "_attn_implementation", None)
        require(actual_attn == "sdpa",
                f"model silently changed attention implementation: {actual_attn!r}")
        quant_identity = _actual_quantization_identity(model)
        cuda_device_map = _validate_cuda_only_device_map(model)
        device = _input_device(model)
        require(set(cuda_device_map.values()) == {str(device)},
                "model was split across CUDA devices; compact cache indices "
                "must stay on one device")
        loaded_eos = eos_identity_from_sources({
            "tokenizer": getattr(bundle.tokenizer, "eos_token_id", None),
            "inspected_config": getattr(bundle.config, "eos_token_id", None),
            "inspected_generation_config": getattr(
                bundle.generation_config, "eos_token_id", None),
            "loaded_model_config": getattr(model.config, "eos_token_id", None),
            "loaded_model_generation_config": getattr(
                getattr(model, "generation_config", None), "eos_token_id", None),
        })
        require(
            loaded_eos["effective_eos_token_ids"]
            == bundle.identity["eos_identity"]["effective_eos_token_ids"],
            "loaded model changed the inspected EOS union",
        )
        loaded_identity = {
            **bundle.identity,
            "loaded_model_class": type(model).__name__,
            "loaded_attn_implementation": actual_attn,
            "loaded_quantization": quant_identity,
            "input_device": str(device),
            "hf_device_map": cuda_device_map,
            "cuda_only_no_cpu_or_disk_offload": True,
            "eos_identity": loaded_eos,
            "weights_loaded": True,
            "gpu_used": True,
        }
        return ModelAdapter(
            model_key=model_key,
            tokenizer=bundle.tokenizer,
            config=model.config,
            generation_config=getattr(model, "generation_config", None),
            model=model,
            input_device=device,
            eos_identity=loaded_eos,
            pad_token_id=bundle.identity["pad_token_id"],
            identity=loaded_identity,
        )
    except Exception:
        # Validation failures are qualification failures, not retry signals.
        # Release any partially loaded CUDA object before propagating unchanged.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        raise


def release_model_adapter(adapter: ModelAdapter | None) -> None:
    """Remove caller-visible model references before CUDA allocator cleanup."""
    if adapter is not None:
        model = adapter.model
        adapter.model = None
        adapter.config = None
        adapter.generation_config = None
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            # ipc_collect may be unavailable in a restricted CUDA runtime; it
            # is cleanup-only and never changes sampling or loading policy.
            pass
    gc.collect()


def _make_logits_warpers() -> tuple[Any, ...]:
    return (
        TemperatureLogitsWarper(SAMPLING_CONTRACT["temperature"]),
        TopKLogitsWarper(SAMPLING_CONTRACT["top_k"]),
        TopPLogitsWarper(SAMPLING_CONTRACT["top_p"]),
        MinPLogitsWarper(SAMPLING_CONTRACT["min_p"]),
    )


def _apply_logits_warpers(
    logits: torch.Tensor, warpers: tuple[Any, ...],
) -> torch.Tensor:
    require(logits.ndim == 2, "sampler logits must be rank two")
    scores = logits.float()
    dummy_ids = torch.zeros(
        (scores.shape[0], 1), dtype=torch.long, device=scores.device,
    )
    for warper in warpers:
        scores = warper(dummy_ids, scores)
    return scores


def _left_pad(
    prompts: Sequence[Sequence[int]], pad_id: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require(bool(prompts), "cannot left-pad an empty batch")
    require(all(bool(prompt) for prompt in prompts), "batch contains an empty prefix")
    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full(
        (len(prompts), width), pad_id, dtype=torch.long, device=device,
    )
    attention = torch.zeros_like(input_ids)
    for index, prompt in enumerate(prompts):
        values = torch.tensor(prompt, dtype=torch.long, device=device)
        input_ids[index, width - len(prompt):] = values
        attention[index, width - len(prompt):] = 1
    positions = attention.cumsum(dim=-1) - 1
    positions.masked_fill_(attention == 0, 0)
    return input_ids, attention, positions


def _require_dynamic_cache(cache: Any) -> DynamicCache:
    require(isinstance(cache, DynamicCache),
            f"compact sampler requires DynamicCache, got {type(cache).__name__}")
    method = getattr(cache, "batch_select_indices", None)
    require(callable(method), "DynamicCache lacks batch_select_indices")
    return cache


def _advance_active_mapping(
    active_global: Sequence[int], sampled_ids: Sequence[int], eos_ids: set[int],
) -> tuple[list[int], list[int], list[int]]:
    """Return survivor local indices, survivor globals, and finished globals."""
    require(len(active_global) == len(sampled_ids),
            "active mapping and sampled token lengths differ")
    require(len(set(active_global)) == len(active_global),
            "active mapping contains duplicate global rows")
    require(bool(eos_ids), "active mapping received an empty EOS set")
    survivor_local: list[int] = []
    survivor_global: list[int] = []
    finished_global: list[int] = []
    for local_index, (global_index, token_id) in enumerate(
        zip(active_global, sampled_ids, strict=True)
    ):
        require(isinstance(global_index, int) and global_index >= 0,
                "active mapping has invalid global index")
        require(isinstance(token_id, int) and token_id >= 0,
                "active mapping has invalid sampled token")
        if token_id in eos_ids:
            finished_global.append(global_index)
        else:
            survivor_local.append(local_index)
            survivor_global.append(global_index)
    return survivor_local, survivor_global, finished_global


def _validate_loaded_adapter_contract(adapter: Any) -> None:
    require(getattr(adapter, "model_key", None) in MODEL_ORDER,
            "sampler adapter has an unsupported model key")
    require(getattr(adapter, "model", None) is not None,
            "adapter model has been released")
    device = getattr(adapter, "input_device", None)
    require(isinstance(device, torch.device) and device.type == "cuda"
            and device.index is not None,
            "sampler requires a concrete CUDA input device")
    identity = getattr(adapter, "identity", None)
    require(isinstance(identity, dict)
            and identity.get("weights_loaded") is True
            and identity.get("gpu_used") is True
            and identity.get("loaded_attn_implementation") == "sdpa"
            and identity.get("cuda_only_no_cpu_or_disk_offload") is True,
            "sampler adapter identity is not an admitted CUDA/SDPA load")
    quant = identity.get("loaded_quantization")
    require(isinstance(quant, dict)
            and quant.get("load_in_8bit") is True
            and quant.get("non_quantized_dtype") == "torch.bfloat16"
            and quant.get("device_map") == "auto",
            "sampler adapter identity is not int8/BF16/device_map-auto")
    eos_identity = getattr(adapter, "eos_identity", None)
    require(isinstance(eos_identity, dict)
            and bool(eos_identity.get("effective_eos_token_ids")),
            "sampler adapter EOS identity is empty")
    observed_pad = tokenizer_pad_id(getattr(adapter, "tokenizer", None))
    require(getattr(adapter, "pad_token_id", None) == observed_pad,
            "adapter pad_token_id differs from tokenizer")


def _validate_requests(
    adapter: ModelAdapter, requests: Sequence[SamplingRequest], batch_size: int,
) -> None:
    _validate_loaded_adapter_contract(adapter)
    require(isinstance(batch_size, int) and not isinstance(batch_size, bool)
            and batch_size > 0, "batch_size must be a positive integer")
    require(len(requests) == batch_size,
            f"batch has {len(requests)} rows, expected exactly {batch_size}")
    keys: list[tuple[str, str, int]] = []
    for request in requests:
        require(isinstance(request, SamplingRequest), "invalid sampling request type")
        require(request.item_id.strip() and request.arm.strip() and request.user_text.strip(),
                "sampling request contains an empty field")
        require(isinstance(request.stream, int) and not isinstance(request.stream, bool)
                and request.stream >= 0, "request stream is invalid")
        if request.seed_key is not None:
            require(isinstance(request.seed_key, str) and request.seed_key.strip(),
                    "request seed_key is invalid")
        keys.append((request.item_id, request.arm, request.stream))
    require(len(set(keys)) == len(keys), "batch contains duplicate item/arm/stream keys")


def _prepare_batch(
    adapter: ModelAdapter, requests: Sequence[SamplingRequest],
    dataset_namespace: str, batch_size: int,
) -> tuple[list[RenderedPrefix], list[int], list[torch.Generator],
           torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_requests(adapter, requests, batch_size)
    prefixes = [adapter.render_user(request.user_text) for request in requests]
    seeds = [
        stable_pair_seed(
            dataset_namespace, adapter.model_key,
            request.seed_key if request.seed_key is not None else request.item_id,
            request.stream, request.arm,
        )
        for request in requests
    ]
    generators: list[torch.Generator] = []
    for seed in seeds:
        generator = torch.Generator(device=adapter.input_device)
        generator.manual_seed(seed)
        generators.append(generator)
    input_ids, attention, positions = _left_pad(
        [prefix.input_ids for prefix in prefixes],
        adapter.pad_token_id, adapter.input_device,
    )
    return prefixes, seeds, generators, input_ids, attention, positions


def _initial_forward(
    adapter: ModelAdapter, input_ids: torch.Tensor,
    attention: torch.Tensor, positions: torch.Tensor,
) -> tuple[torch.Tensor, DynamicCache]:
    outputs = adapter.model(
        input_ids=input_ids,
        attention_mask=attention,
        position_ids=positions,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    require(outputs.logits.ndim == 3 and outputs.logits.shape[1] == 1,
            f"logits_to_keep=1 contract changed: {tuple(outputs.logits.shape)}")
    logits = outputs.logits[:, -1, :]
    cache = _require_dynamic_cache(outputs.past_key_values)
    return logits, cache


def _finalize_rows(
    adapter: ModelAdapter, requests: Sequence[SamplingRequest],
    prefixes: Sequence[RenderedPrefix], seeds: Sequence[int],
    generated: Sequence[Sequence[int]], eos_set: set[int],
) -> list[SampledRow]:
    rows: list[SampledRow] = []
    for request, prefix, seed, token_ids in zip(
        requests, prefixes, seeds, generated, strict=True,
    ):
        require(bool(token_ids), "sampler produced an empty trajectory")
        eos_positions = [
            index for index, token_id in enumerate(token_ids)
            if token_id in eos_set
        ]
        require(not eos_positions or eos_positions == [len(token_ids) - 1],
                "tokens were retained after the first EOS")
        first_eos = token_ids[-1] if eos_positions else None
        rows.append(SampledRow(
            item_id=request.item_id,
            seed_key=(request.seed_key if request.seed_key is not None
                      else request.item_id),
            stream=request.stream,
            arm=request.arm,
            model_key=adapter.model_key,
            pair_seed=seed,
            rendered_prefix_sha256=prefix.rendered_sha256,
            input_ids=prefix.input_ids,
            generated_ids=tuple(int(value) for value in token_ids),
            first_eos_token_id=int(first_eos) if first_eos is not None else None,
            first_eos_absorbing=True,
        ))
    return rows


def sample_batch(
    adapter: ModelAdapter,
    requests: Sequence[SamplingRequest],
    dataset_namespace: str,
    max_new_tokens: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[SampledRow]:
    """Sample one compacting batch; this is the only formal sampler API."""
    require(isinstance(max_new_tokens, int) and not isinstance(max_new_tokens, bool)
            and max_new_tokens > 0, "max_new_tokens must be a positive integer")
    (prefixes, seeds, generators, input_ids, attention,
     positions) = _prepare_batch(
        adapter, requests, dataset_namespace, batch_size,
    )
    eos_set = set(adapter.eos_identity["effective_eos_token_ids"])
    require(bool(eos_set), "adapter has no effective EOS IDs")
    generated: list[list[int]] = [[] for _ in requests]
    active_global = list(range(len(requests)))
    warpers = _make_logits_warpers()
    with torch.inference_mode():
        logits, cache = _initial_forward(
            adapter, input_ids, attention, positions,
        )
        del input_ids, positions
        for step in range(max_new_tokens):
            require(logits.shape[0] == len(active_global),
                    "compact logits/active mapping batch mismatch")
            probabilities = torch.softmax(
                _apply_logits_warpers(logits, warpers), dim=-1,
            )
            sampled_tensor = torch.stack([
                torch.multinomial(
                    probabilities[local_index], 1, replacement=True,
                    generator=generators[global_index],
                ).squeeze(0)
                for local_index, global_index in enumerate(active_global)
            ]).long()
            sampled_ids = [int(value) for value in sampled_tensor.tolist()]
            for global_index, token_id in zip(
                active_global, sampled_ids, strict=True,
            ):
                generated[global_index].append(token_id)
            survivor_local, survivor_global, _finished = _advance_active_mapping(
                active_global, sampled_ids, eos_set,
            )
            if not survivor_global or step + 1 == max_new_tokens:
                break
            if len(survivor_global) != len(active_global):
                select = torch.tensor(
                    survivor_local, dtype=torch.long, device=adapter.input_device,
                )
                cache.batch_select_indices(select)
                attention = attention.index_select(0, select)
                sampled_tensor = sampled_tensor.index_select(0, select)
            active_global = survivor_global
            attention = torch.cat((
                attention,
                torch.ones(
                    (len(active_global), 1), dtype=attention.dtype,
                    device=attention.device,
                ),
            ), dim=1)
            step_ids = sampled_tensor.unsqueeze(1)
            step_positions = attention.sum(dim=-1, keepdim=True) - 1
            step_positions.clamp_min_(0)
            outputs = adapter.model(
                input_ids=step_ids,
                attention_mask=attention,
                position_ids=step_positions,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            require(outputs.logits.ndim == 3 and outputs.logits.shape[1] == 1,
                    "incremental logits_to_keep=1 contract changed")
            cache = _require_dynamic_cache(outputs.past_key_values)
            logits = outputs.logits[:, -1, :]
            del (outputs, step_ids, step_positions, sampled_tensor,
                 probabilities)
    del logits, cache, attention
    return _finalize_rows(
        adapter, requests, prefixes, seeds, generated, eos_set,
    )


def _sample_batch_dense_reference(
    adapter: ModelAdapter,
    requests: Sequence[SamplingRequest],
    dataset_namespace: str,
    max_new_tokens: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[SampledRow]:
    """Dense engineering reference; future formal runners must not call it."""
    require(isinstance(max_new_tokens, int) and not isinstance(max_new_tokens, bool)
            and max_new_tokens > 0, "max_new_tokens must be a positive integer")
    (prefixes, seeds, generators, input_ids, attention,
     positions) = _prepare_batch(
        adapter, requests, dataset_namespace, batch_size,
    )
    eos_set = set(adapter.eos_identity["effective_eos_token_ids"])
    generated: list[list[int]] = [[] for _ in requests]
    active = [True] * len(requests)
    warpers = _make_logits_warpers()
    with torch.inference_mode():
        logits, cache = _initial_forward(
            adapter, input_ids, attention, positions,
        )
        del input_ids, positions
        for step in range(max_new_tokens):
            require(logits.shape[0] == len(requests),
                    "dense logits batch size changed")
            probabilities = torch.softmax(
                _apply_logits_warpers(logits, warpers), dim=-1,
            )
            sampled: list[torch.Tensor] = []
            for index in range(len(requests)):
                if active[index]:
                    sampled.append(torch.multinomial(
                        probabilities[index], 1, replacement=True,
                        generator=generators[index],
                    ).squeeze(0))
                else:
                    sampled.append(torch.tensor(
                        adapter.pad_token_id, dtype=torch.long,
                        device=adapter.input_device,
                    ))
            sampled_tensor = torch.stack(sampled).long()
            sampled_ids = [int(value) for value in sampled_tensor.tolist()]
            next_active: list[bool] = []
            for index, token_id in enumerate(sampled_ids):
                if active[index]:
                    generated[index].append(token_id)
                    next_active.append(token_id not in eos_set)
                else:
                    next_active.append(False)
            active = next_active
            if not any(active) or step + 1 == max_new_tokens:
                break
            step_mask = torch.tensor(
                active, dtype=attention.dtype, device=attention.device,
            ).unsqueeze(1)
            attention = torch.cat((attention, step_mask), dim=1)
            step_ids = sampled_tensor.unsqueeze(1)
            step_positions = attention.sum(dim=-1, keepdim=True) - 1
            step_positions.clamp_min_(0)
            outputs = adapter.model(
                input_ids=step_ids,
                attention_mask=attention,
                position_ids=step_positions,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            require(outputs.logits.ndim == 3 and outputs.logits.shape[1] == 1,
                    "dense incremental logits_to_keep=1 contract changed")
            cache = _require_dynamic_cache(outputs.past_key_values)
            logits = outputs.logits[:, -1, :]
            del (outputs, step_ids, step_positions, step_mask,
                 sampled_tensor, probabilities)
    del logits, cache, attention
    return _finalize_rows(
        adapter, requests, prefixes, seeds, generated, eos_set,
    )


def compare_compact_to_dense(
    adapter: ModelAdapter,
    requests: Sequence[SamplingRequest],
    dataset_namespace: str,
    max_new_tokens: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Run both engines for a pre-freeze smoke and compare every token."""
    compact = sample_batch(
        adapter, requests, dataset_namespace, max_new_tokens, batch_size,
    )
    dense = _sample_batch_dense_reference(
        adapter, requests, dataset_namespace, max_new_tokens, batch_size,
    )
    require(len(compact) == len(dense) == len(requests),
            "compact/dense row denominator changed")
    comparisons: list[dict[str, Any]] = []
    exact = True
    for left, right in zip(compact, dense, strict=True):
        require((left.item_id, left.seed_key, left.arm, left.stream, left.pair_seed)
                == (right.item_id, right.seed_key, right.arm, right.stream,
                    right.pair_seed),
                "compact/dense row identity mismatch")
        left_ids = list(left.generated_ids)
        right_ids = list(right.generated_ids)
        first_mismatch = None
        for index, (left_id, right_id) in enumerate(
            zip(left_ids, right_ids, strict=False)
        ):
            if left_id != right_id:
                first_mismatch = index
                break
        if first_mismatch is None and len(left_ids) != len(right_ids):
            first_mismatch = min(len(left_ids), len(right_ids))
        row_exact = first_mismatch is None
        exact = exact and row_exact
        comparisons.append({
            "item_id": left.item_id,
            "seed_key": left.seed_key,
            "arm": left.arm,
            "stream": left.stream,
            "exact_token_match": row_exact,
            "first_mismatch_index": first_mismatch,
            "compact_token_count": len(left_ids),
            "dense_token_count": len(right_ids),
            "compact_tokens_sha256": sha256_bytes(
                canonical_json(left_ids).encode("utf-8")),
            "dense_tokens_sha256": sha256_bytes(
                canonical_json(right_ids).encode("utf-8")),
        })
    return {
        "model_key": adapter.model_key,
        "exact_token_match": exact,
        "row_count": len(comparisons),
        "max_new_tokens": max_new_tokens,
        "sampling_contract": dict(SAMPLING_CONTRACT),
        "rows": comparisons,
        "engineering_smoke_only": True,
    }


def _rejected(callable_value: Any) -> bool:
    try:
        callable_value()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


class _FakeTokenizer:
    eos_token_id = 7
    pad_token_id = 9
    chat_template = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.calls.append((messages, kwargs))
        return f"USER:{messages[0]['content']}|ASSISTANT:"

    def __call__(self, text: str, **_kwargs: Any) -> Any:
        return SimpleNamespace(input_ids=[ord(char) % 97 for char in text])


def self_test() -> dict[str, Any]:
    require(MODEL_ORDER == ("qwen3", "glm4", "deepseek7b"),
            "cross-model execution order changed")
    require(tuple(get_model_spec(key).key for key in MODEL_ORDER) == MODEL_ORDER,
            "model registry order/support changed")
    require(DEFAULT_BATCH_SIZE == 8, "default batch size changed")
    require(SAMPLING_CONTRACT == {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    }, "sampling contract changed")
    require(QUANTIZATION_CONTRACT == {
        "backend": "bitsandbytes",
        "load_in_8bit": True,
        "llm_int8_enable_fp32_cpu_offload": False,
        "non_quantized_dtype": "torch.bfloat16",
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "local_files_only": True,
    }, "quantization contract changed")

    fake = _FakeTokenizer()
    rendered = render_native_user(fake, "return A")
    require(rendered.user_text == "return A" and bool(rendered.input_ids),
            "native single-user rendering failed")
    require(len(fake.calls) == 1, "native renderer called template more than once")
    messages, kwargs = fake.calls[0]
    require(messages == [{"role": "user", "content": "return A"}],
            "native renderer did not serialize exactly one user message")
    require(kwargs == {"tokenize": False, "add_generation_prompt": True},
            "native renderer passed a non-native chat control")
    require("enable_thinking" not in kwargs,
            "Qwen thinking switch leaked into native renderer")

    protocol_sha256 = "a" * 64
    seed_a = stable_pair_seed(protocol_sha256, "qwen3", "item-1", 0, "A")
    seed_b = stable_pair_seed(protocol_sha256, "qwen3", "item-1", 0, "B")
    seed_other_model = stable_pair_seed(
        protocol_sha256, "glm4", "item-1", 0, "A",
    )
    original_request = SamplingRequest(
        item_id="surface-original", seed_key="semantic-seed-1", stream=0,
        arm="A", user_text="A",
    )
    swapped_request = SamplingRequest(
        item_id="surface-swapped", seed_key="semantic-seed-1", stream=0,
        arm="B", user_text="B",
    )
    seed_twin_original = stable_pair_seed(
        protocol_sha256, "qwen3", str(original_request.seed_key),
        original_request.stream, original_request.arm,
    )
    seed_twin_swapped = stable_pair_seed(
        protocol_sha256, "qwen3", str(swapped_request.seed_key),
        swapped_request.stream, swapped_request.arm,
    )
    require(seed_a == seed_b, "arm changed the pair seed")
    require(original_request.item_id != swapped_request.item_id
            and seed_twin_original == seed_twin_swapped,
            "shared semantic seed key changed the 2x2 CRN block seed")
    require(seed_a != seed_other_model, "model namespace is absent from pair seed")
    require(_validate_cuda_only_device_map(SimpleNamespace(
        hf_device_map={"": 0, "lm_head": "cuda:0"},
    )) == {"": "cuda:0", "lm_head": "cuda:0"},
            "valid CUDA-only device map was not normalized exactly")

    multi_eos = eos_identity_from_sources({
        "tokenizer": 5,
        "config": [5, 7],
        "generation_config": (8, 7),
    })
    require(multi_eos["effective_eos_token_ids"] == [5, 7, 8]
            and multi_eos["multiple_effective_eos"] is True,
            "multi-EOS union failed")
    survivor_local, survivor_global, finished_global = _advance_active_mapping(
        [4, 9, 2, 7], [11, 7, 8, 12], {7, 8},
    )
    require(survivor_local == [0, 3] and survivor_global == [4, 7]
            and finished_global == [9, 2],
            "dynamic active-row mapping failed")

    negative_tests = {
        "unknown_model_seed_rejected": _rejected(
            lambda: stable_pair_seed("d", "unknown", "i", 0, "A")),
        "empty_dataset_namespace_rejected": _rejected(
            lambda: stable_pair_seed("", "qwen3", "i", 0, "A")),
        "negative_stream_rejected": _rejected(
            lambda: stable_pair_seed("d", "qwen3", "i", -1, "A")),
        "bool_EOS_rejected": _rejected(
            lambda: eos_identity_from_sources({"tokenizer": True})),
        "empty_EOS_union_rejected": _rejected(
            lambda: eos_identity_from_sources({"tokenizer": None})),
        "negative_EOS_rejected": _rejected(
            lambda: eos_identity_from_sources({"tokenizer": -1})),
        "active_mapping_length_mismatch_rejected": _rejected(
            lambda: _advance_active_mapping([0, 1], [3], {3})),
        "duplicate_active_global_rejected": _rejected(
            lambda: _advance_active_mapping([0, 0], [1, 2], {2})),
        "empty_active_EOS_set_rejected": _rejected(
            lambda: _advance_active_mapping([0], [1], set())),
        "unsupported_cache_rejected": _rejected(
            lambda: _require_dynamic_cache(object())),
        "CPU_sampler_adapter_rejected": _rejected(
            lambda: _validate_loaded_adapter_contract(SimpleNamespace(
                model_key="qwen3", model=object(), input_device=torch.device("cpu"),
                identity={}, eos_identity={}, tokenizer=fake,
            ))),
        "cpu_device_map_rejected": _rejected(
            lambda: _validate_cuda_only_device_map(SimpleNamespace(
                hf_device_map={"model.layers.0": 0, "model.layers.1": "cpu"},
            ))),
        "disk_device_map_rejected": _rejected(
            lambda: _validate_cuda_only_device_map(SimpleNamespace(
                hf_device_map={"model.layers.0": "disk"},
            ))),
        "empty_device_map_rejected": _rejected(
            lambda: _validate_cuda_only_device_map(SimpleNamespace(
                hf_device_map={},
            ))),
        "missing_pad_rejected": _rejected(
            lambda: tokenizer_pad_id(SimpleNamespace(pad_token_id=None))),
        "multi_message_or_control_not_exposed": (
            tuple(inspect.signature(render_native_user).parameters)
            == ("tokenizer", "user_text")
        ),
    }
    require(all(negative_tests.values()),
            f"fail-closed engine self-test failed: {negative_tests}")
    return {
        "schema_version": ENGINE_SCHEMA_VERSION,
        "model_order": list(MODEL_ORDER),
        "default_batch_size": DEFAULT_BATCH_SIZE,
        "sampling_contract": dict(SAMPLING_CONTRACT),
        "quantization_contract": dict(QUANTIZATION_CONTRACT),
        "arm_excluded_from_pair_seed": True,
        "semantic_twin_2x2_crn_seed_supported": True,
        "model_namespace_in_pair_seed": True,
        "protocol_sha256_accepted_as_dataset_namespace": True,
        "native_single_user_only": True,
        "enable_thinking_passed": False,
        "multi_eos_union": multi_eos,
        "active_mapping_test": {
            "survivor_local": survivor_local,
            "survivor_global": survivor_global,
            "finished_global": finished_global,
        },
        "negative_tests": negative_tests,
        "gpu_used": False,
        "model_weights_loaded": False,
        "files_written": False,
    }


def _gpu_smoke_requests() -> list[SamplingRequest]:
    prompts = (
        "Return exactly the single capital letter A and then stop.",
        "Reply with A only.",
        "The required answer is A. Output only A.",
        "Choose A and return no explanation.",
        "Write exactly A.",
        "Output the one-character answer A.",
        "Answer this engineering check with A only.",
        "For this cache smoke, return A and stop.",
    )
    require(len(prompts) == DEFAULT_BATCH_SIZE,
            "GPU smoke prompt denominator changed")
    return [
        SamplingRequest(
            item_id=f"engineering_smoke_{index:02d}",
            stream=index,
            arm="engineering_smoke",
            user_text=prompt,
        )
        for index, prompt in enumerate(prompts)
    ]


def gpu_smoke(model_key: str) -> dict[str, Any]:
    """Explicit one-model, no-write engineering qualification."""
    require(model_key in MODEL_ORDER, f"unsupported smoke model: {model_key}")
    self_test()
    adapter: ModelAdapter | None = None
    try:
        adapter = load_model_adapter(model_key)
        comparison = compare_compact_to_dense(
            adapter,
            _gpu_smoke_requests(),
            dataset_namespace="phase983-engineering-smoke-v1",
            max_new_tokens=GPU_SMOKE_MAX_NEW_TOKENS,
            batch_size=DEFAULT_BATCH_SIZE,
        )
        require(comparison["exact_token_match"] is True,
                "compact and dense GPU smoke trajectories differ")
        return {
            "schema_version": ENGINE_SCHEMA_VERSION,
            "model_key": model_key,
            "model_identity": adapter.identity,
            "comparison": comparison,
            "engineering_smoke_only": True,
            "formal_result": False,
            "files_written": False,
            "gpu_used": True,
        }
    finally:
        release_model_adapter(adapter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--self-test", action="store_true",
        help="CPU-only invariant and fail-closed tests",
    )
    modes.add_argument(
        "--inspect", action="store_true",
        help="CPU-only local tokenizer/config inspection for all three models",
    )
    modes.add_argument(
        "--gpu-smoke", choices=MODEL_ORDER, metavar="MODEL",
        help="explicit single-model CUDA engineering smoke; writes no files",
    )
    args = parser.parse_args()
    if args.self_test:
        output = self_test()
    elif args.inspect:
        output = inspect_all_models()
    else:
        output = gpu_smoke(str(args.gpu_smoke))
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
