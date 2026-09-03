#!/usr/bin/env python3
"""Preflight and engineering smoke tests for local Qwen3.8-27B NF4.

This entry point deliberately stays separate from the existing FP16 probe
loader.  Qwen3.8-27B cannot fit on this machine through that loader: its
language stack must be quantized, while the embedding, LM head and vision
modules remain on CPU.

Examples (use the prepared Python 3.11 runtime):

    python tests/glm5/qwen38_27b_nf4_smoke.py preflight
    python tests/glm5/qwen38_27b_nf4_smoke.py load
    python tests/glm5/qwen38_27b_nf4_smoke.py hidden --prompt "The capital of France is"

The ``hidden`` command stores the complete token-by-feature embedding matrix
and all 64 complete token-by-feature layer outputs.  It does not generate a
scientific result; it only verifies the loading and hidden-state capture path.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open as native_safe_open
from safetensors.torch import save_file as save_safetensors
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models" / "hf" / "Qwen3.8-27B"
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "qwen38_27b_nf4_smoke"
OFFLOAD_ROOT = OUT_ROOT / "offload"

REPO_ID = "Qwen/Qwen3.8-27B"
EXPECTED_COMMIT = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECTED_TOTAL_BYTES = 55_586_114_863
EXPECTED_WEIGHT_BYTES = 55_563_006_776
EXPECTED_PARAMETER_COUNT = 27_356_728_560
EXPECTED_LANGUAGE_LINEAR_PARAMETERS = 24_350_556_160
CUDA_LAYER_COUNT = 48
CPU_LAYER_COUNT = 16

EXPECTED_FILES = {
    ".gitattributes": 1_570,
    "LICENSE": 11_544,
    "README.md": 65_012,
    "chat_template.jinja": 8_952,
    "config.json": 4_312,
    "crc32.txt": 238,
    "generation_config.json": 202,
    "merges.txt": 3_353_259,
    "model-00001-of-00018.safetensors": 3_966_730_552,
    "model-00002-of-00018.safetensors": 3_043_080_328,
    "model-00003-of-00018.safetensors": 2_542_796_952,
    "model-00004-of-00018.safetensors": 3_988_973_152,
    "model-00005-of-00018.safetensors": 2_099_339_864,
    "model-00006-of-00018.safetensors": 3_979_553_696,
    "model-00007-of-00018.safetensors": 2_108_759_344,
    "model-00008-of-00018.safetensors": 3_979_553_696,
    "model-00009-of-00018.safetensors": 2_108_759_344,
    "model-00010-of-00018.safetensors": 3_979_553_696,
    "model-00011-of-00018.safetensors": 2_108_759_344,
    "model-00012-of-00018.safetensors": 3_979_553_696,
    "model-00013-of-00018.safetensors": 2_108_759_344,
    "model-00014-of-00018.safetensors": 3_979_553_696,
    "model-00015-of-00018.safetensors": 2_108_759_344,
    "model-00016-of-00018.safetensors": 3_979_564_040,
    "model-00017-of-00018.safetensors": 2_108_759_344,
    "model-00018-of-00018.safetensors": 3_392_197_344,
    "model.safetensors.index.json": 112_216,
    "preprocessor_config.json": 390,
    "tokenizer.json": 12_809_320,
    "tokenizer_config.json": 17_928,
    "video_preprocessor_config.json": 385,
    "vocab.json": 6_722_759,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def local_commit() -> str | None:
    metadata = MODEL_ROOT / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if not metadata.exists():
        return None
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else None


def file_manifest() -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    for name, expected_size in EXPECTED_FILES.items():
        path = MODEL_ROOT / name
        actual_size = path.stat().st_size if path.is_file() else None
        rows.append(
            {
                "name": name,
                "expected_bytes": expected_size,
                "actual_bytes": actual_size,
                "ready": actual_size == expected_size,
            }
        )
    return rows, all(row["ready"] for row in rows)


def nf4_cuda_operator_smoke() -> dict[str, Any]:
    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(
        64,
        64,
        bias=False,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
    )
    layer.load_state_dict({"weight": torch.randn(64, 64)})
    layer = layer.to("cuda:0")
    sample = torch.randn(2, 64, device="cuda:0", dtype=torch.bfloat16)
    with torch.inference_mode():
        output = layer(sample)
    torch.cuda.synchronize()
    result = {
        "shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all().item()),
        "quant_type": str(layer.weight.quant_state.quant_type),
    }
    del output, sample, layer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def preflight() -> dict[str, Any]:
    rows, files_ready = file_manifest()
    config_path = MODEL_ROOT / "config.json"
    config = (
        AutoConfig.from_pretrained(MODEL_ROOT, local_files_only=True)
        if config_path.exists()
        else None
    )
    text_config = getattr(config, "text_config", None)
    layer_types = list(getattr(text_config, "layer_types", []))
    meta_parameter_count = None
    meta_language_linear_parameters = None
    meta_model_error = None
    if config is not None:
        try:
            from accelerate import init_empty_weights
            from transformers import Qwen3_5ForConditionalGeneration

            with init_empty_weights():
                meta_model = Qwen3_5ForConditionalGeneration(config)
            meta_parameter_count = sum(parameter.numel() for parameter in meta_model.parameters())
            meta_language_linear_parameters = sum(
                parameter.numel()
                for module in meta_model.model.language_model.layers.modules()
                if isinstance(module, torch.nn.Linear)
                for parameter in module.parameters(recurse=False)
            )
            del meta_model
        except Exception as error:  # recorded as an engineering gate
            meta_model_error = f"{type(error).__name__}: {error}"
    cuda_ready = torch.cuda.is_available()
    gpu = None
    if cuda_ready:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    operator = None
    operator_error = None
    if cuda_ready and package_version("bitsandbytes") is not None:
        try:
            operator = nf4_cuda_operator_smoke()
        except Exception as error:  # recorded as an engineering gate
            operator_error = f"{type(error).__name__}: {error}"

    checks = {
        "official_files_complete": files_ready,
        "official_total_bytes": sum(EXPECTED_FILES.values()) == EXPECTED_TOTAL_BYTES,
        "official_weight_bytes": sum(
            size for name, size in EXPECTED_FILES.items() if name.endswith(".safetensors")
        )
        == EXPECTED_WEIGHT_BYTES,
        "pinned_commit": local_commit() == EXPECTED_COMMIT,
        "qwen35_architecture_available": package_version("transformers") is not None
        and hasattr(__import__("transformers"), "Qwen3_5ForConditionalGeneration"),
        "architecture": config is not None
        and list(getattr(config, "architectures", [])) == ["Qwen3_5ForConditionalGeneration"],
        "layer_count_64": text_config is not None
        and int(getattr(text_config, "num_hidden_layers", -1)) == 64,
        "hidden_size_5120": text_config is not None
        and int(getattr(text_config, "hidden_size", -1)) == 5_120,
        "vocabulary_248320": text_config is not None
        and int(getattr(text_config, "vocab_size", -1)) == 248_320,
        "hybrid_layer_schedule": len(layer_types) == 64
        and layer_types.count("linear_attention") == 48
        and layer_types.count("full_attention") == 16,
        "parameter_count": meta_parameter_count == EXPECTED_PARAMETER_COUNT,
        "language_linear_parameter_count": meta_language_linear_parameters
        == EXPECTED_LANGUAGE_LINEAR_PARAMETERS,
        "cuda_available": cuda_ready,
        "gpu_headroom_for_planned_map": gpu is not None
        and gpu["free_bytes"] >= 14 * 1024**3,
        "bitsandbytes_0492": package_version("bitsandbytes") == "0.49.2",
        "nf4_cuda_operator": operator is not None
        and operator["finite"]
        and operator["quant_type"] == "nf4",
    }
    result = {
        "schema_version": "qwen38_27b_nf4_preflight.v1",
        "scope": "engineering readiness only; no model behavior or scientific claim",
        "repo_id": REPO_ID,
        "expected_commit": EXPECTED_COMMIT,
        "local_commit": local_commit(),
        "model_root": str(MODEL_ROOT),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "bitsandbytes": package_version("bitsandbytes"),
            "safetensors": package_version("safetensors"),
        },
        "gpu": gpu,
        "architecture": {
            "class": "Qwen3_5ForConditionalGeneration",
            "parameter_count": meta_parameter_count,
            "language_linear_parameter_count": meta_language_linear_parameters,
            "meta_model_error": meta_model_error,
            "layers": len(layer_types),
            "hidden_size": getattr(text_config, "hidden_size", None),
            "vocab_size": getattr(text_config, "vocab_size", None),
            "linear_attention_layers": layer_types.count("linear_attention"),
            "full_attention_layers": layer_types.count("full_attention"),
        },
        "planned_placement": {
            "cpu": [
                "model.visual",
                "model.language_model.embed_tokens",
                f"model.language_model.layers.{CUDA_LAYER_COUNT}-63",
                "model.language_model.norm",
                "lm_head",
            ],
            "cuda_0": [
                f"model.language_model.layers.0-{CUDA_LAYER_COUNT - 1}",
                "model.language_model.rotary_emb",
            ],
            "cuda_language_linear_nf4_payload_estimate_bytes": int(
                EXPECTED_LANGUAGE_LINEAR_PARAMETERS
                * CUDA_LAYER_COUNT
                / (CUDA_LAYER_COUNT + CPU_LAYER_COUNT)
                * 0.5
            ),
            "warning": (
                "The first 48 layers use NF4 on CUDA; the last 16 stay BF16 on CPU. "
                "The payload estimate excludes quantization metadata and CUDA runtime buffers."
            ),
        },
        "nf4_operator": operator,
        "nf4_operator_error": operator_error,
        "file_manifest": rows,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    write_json(OUT_ROOT / "preflight.json", result)
    return result


def fixed_device_map() -> dict[str, int | str]:
    device_map: dict[str, int | str] = {
        "model.visual": "cpu",
        "model.language_model.embed_tokens": "cpu",
        "model.language_model.norm": "cpu",
        "model.language_model.rotary_emb": 0,
        "lm_head": "cpu",
    }
    device_map.update(
        {
            f"model.language_model.layers.{index}": (
                0 if index < CUDA_LAYER_COUNT else "cpu"
            )
            for index in range(CUDA_LAYER_COUNT + CPU_LAYER_COUNT)
        }
    )
    return device_map


@contextmanager
def windows_shard_streaming_loader(model_class: type[Any]) -> Any:
    """Load one safetensors shard at a time under Transformers 5.12 on Windows.

    The upstream loader keeps all 18 shard readers alive simultaneously.  That
    consumes the Windows commit limit before this 55.6 GB checkpoint can be
    quantized.  Qwen3.8 uses only independent key-renaming conversions, so the
    same conversion routine can safely run once per shard while accumulating a
    single final loading report.
    """

    if platform.system() != "Windows":
        yield
        return

    from transformers.core_model_loading import convert_and_load_state_dict_in_model
    from transformers.modeling_utils import caching_allocator_warmup, expand_device_map
    from transformers.utils.loading_report import LoadStateDictInfo

    original_loader = model_class._load_pretrained_model
    had_local_override = "_load_pretrained_model" in model_class.__dict__
    original_descriptor = model_class.__dict__.get("_load_pretrained_model")

    def load_shards(
        model: Any,
        state_dict: dict[str, Any] | None,
        checkpoint_files: list[str] | None,
        load_config: Any,
        expected_keys: list[str] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        can_stream = (
            state_dict is None
            and checkpoint_files is not None
            and len(checkpoint_files) > 1
            and all(str(path).endswith(".safetensors") for path in checkpoint_files)
            and not (
                load_config.device_map is not None
                and "disk" in load_config.device_map.values()
            )
        )
        if not can_stream:
            return original_loader(
                model, state_dict, checkpoint_files, load_config, expected_keys
            )

        expected = list(model.state_dict().keys()) if expected_keys is None else expected_keys
        if load_config.device_map is not None:
            expanded_device_map = expand_device_map(load_config.device_map, expected)
            caching_allocator_warmup(model, expanded_device_map, load_config.hf_quantizer)

        loading_info = LoadStateDictInfo(
            missing_keys=set(expected),
            unexpected_keys=set(),
            mismatched_keys=set(),
            conversion_errors={},
            error_msgs=[],
        )
        disk_offload_index = None
        for checkpoint_file in checkpoint_files:
            with native_safe_open(
                checkpoint_file, framework="pt", device="cpu"
            ) as reader:
                shard_state_dict = {
                    key: reader.get_slice(key) for key in reader.keys()
                }
                shard_info, disk_offload_index = convert_and_load_state_dict_in_model(
                    model=model,
                    state_dict=shard_state_dict,
                    load_config=load_config,
                    tp_plan=model.tp_plan,
                    disk_offload_index=disk_offload_index,
                )

            loading_info.missing_keys.intersection_update(shard_info.missing_keys)
            loading_info.unexpected_keys.update(shard_info.unexpected_keys)
            loading_info.mismatched_keys.update(shard_info.mismatched_keys)
            loading_info.conversion_errors.update(shard_info.conversion_errors)
            loading_info.error_msgs.extend(shard_info.error_msgs)
            del shard_state_dict, shard_info
            gc.collect()

        return loading_info, disk_offload_index

    model_class._load_pretrained_model = staticmethod(load_shards)
    try:
        yield
    finally:
        if had_local_override:
            model_class._load_pretrained_model = original_descriptor
        else:
            delattr(model_class, "_load_pretrained_model")


def load_nf4_model() -> Any:
    from transformers import Qwen3_5ForConditionalGeneration

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    OFFLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    with windows_shard_streaming_loader(Qwen3_5ForConditionalGeneration):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ROOT,
            local_files_only=True,
            dtype=torch.bfloat16,
            quantization_config=quantization,
            device_map=fixed_device_map(),
            low_cpu_mem_usage=True,
            offload_folder=OFFLOAD_ROOT,
            attn_implementation="eager",
        )
    model.eval()
    return model


def module_device(parameter: torch.nn.Parameter) -> str:
    return str(parameter.device)


def load_command() -> dict[str, Any]:
    readiness = preflight()
    if not readiness["all_checks_passed"]:
        raise RuntimeError("preflight failed; refusing the 27B weight load")

    import bitsandbytes as bnb

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model = load_nf4_model()
    linear4bit_count = sum(
        isinstance(module, bnb.nn.Linear4bit) for module in model.modules()
    )
    language_model = model.model.language_model
    layer_devices = [module_device(next(layer.parameters())) for layer in language_model.layers]
    hf_device_map = {key: str(value) for key, value in model.hf_device_map.items()}
    embedding_hook = type(getattr(language_model.embed_tokens, "_hf_hook", None)).__name__
    last_layer_hook = type(getattr(language_model.layers[-1], "_hf_hook", None)).__name__
    lm_head_hook = type(getattr(model.lm_head, "_hf_hook", None)).__name__
    result = {
        "schema_version": "qwen38_27b_nf4_load_smoke.v1",
        "scope": "engineering weight-load verification only",
        "repo_id": REPO_ID,
        "commit": EXPECTED_COMMIT,
        "checkpoint_reader": "Windows sequential safetensors shard streaming",
        "elapsed_seconds": time.time() - started,
        "logical_parameter_count": EXPECTED_PARAMETER_COUNT,
        "stored_parameter_elements_after_quantization": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "linear4bit_module_count": linear4bit_count,
        "embedding_device": module_device(language_model.embed_tokens.weight),
        "layer_devices": layer_devices,
        "planned_cuda_layer_count": CUDA_LAYER_COUNT,
        "planned_cpu_layer_count": CPU_LAYER_COUNT,
        "resident_cuda_layer_count": sum(
            device.startswith("cuda") for device in layer_devices
        ),
        "resident_cpu_layer_count": sum(device == "cpu" for device in layer_devices),
        "offloaded_meta_layer_count": sum(device == "meta" for device in layer_devices),
        "lm_head_device": module_device(model.lm_head.weight),
        "offload_hooks": {
            "embedding": embedding_hook,
            "last_layer": last_layer_hook,
            "lm_head": lm_head_hook,
        },
        "hf_device_map": hf_device_map,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
    }
    result["checks"] = {
        "logical_parameter_count_preflight_verified": bool(
            readiness["checks"]["parameter_count"]
        ),
        "nf4_modules_present": linear4bit_count > 0,
        "embedding_cpu_offload_ready": hf_device_map.get(
            "model.language_model.embed_tokens"
        )
        == "cpu"
        and result["embedding_device"] in {"cpu", "meta"}
        and embedding_hook != "NoneType",
        "first_48_language_layers_on_cuda": all(
            device.startswith("cuda")
            and hf_device_map.get(f"model.language_model.layers.{index}") == "0"
            for index, device in enumerate(layer_devices[:CUDA_LAYER_COUNT])
        ),
        "last_16_language_layers_cpu_offload_ready": all(
            device in {"cpu", "meta"}
            and hf_device_map.get(f"model.language_model.layers.{index}") == "cpu"
            for index, device in enumerate(
                layer_devices[CUDA_LAYER_COUNT:], start=CUDA_LAYER_COUNT
            )
        ),
        "lm_head_cpu_offload_ready": hf_device_map.get("lm_head") == "cpu"
        and result["lm_head_device"] in {"cpu", "meta"}
        and lm_head_hook != "NoneType",
    }
    result["all_checks_passed"] = all(result["checks"].values())
    write_json(OUT_ROOT / "load.json", result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def hidden_command(prompt: str, max_length: int) -> dict[str, Any]:
    readiness = preflight()
    if not readiness["all_checks_passed"]:
        raise RuntimeError("preflight failed; refusing the 27B hidden-state smoke")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT, local_files_only=True)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    if encoded["input_ids"].shape[1] == 0:
        raise ValueError("prompt produced no tokens")

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model = load_nf4_model()
    language_model = model.model.language_model
    captures: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def capture(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captures[name] = hidden[0].detach().to("cpu", dtype=torch.bfloat16).contiguous()

        return hook

    handles.append(language_model.embed_tokens.register_forward_hook(capture("embedding")))
    for index, layer in enumerate(language_model.layers):
        handles.append(layer.register_forward_hook(capture(f"layer_{index:03d}")))

    try:
        with torch.inference_mode():
            output = language_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        del output
    finally:
        for handle in handles:
            handle.remove()

    expected_names = {"embedding", *(f"layer_{index:03d}" for index in range(64))}
    capture_path = OUT_ROOT / "hidden" / "complete_vectors.safetensors"
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(captures, capture_path)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
    finite = all(bool(torch.isfinite(tensor.float()).all().item()) for tensor in captures.values())
    shapes = {name: list(tensor.shape) for name, tensor in captures.items()}
    result = {
        "schema_version": "qwen38_27b_nf4_hidden_smoke.v1",
        "scope": "engineering hidden-state capture only; not scientific evidence",
        "repo_id": REPO_ID,
        "commit": EXPECTED_COMMIT,
        "quantization": "bitsandbytes NF4 with double quantization",
        "prompt": prompt,
        "token_ids": encoded["input_ids"][0].tolist(),
        "tokens": tokens,
        "capture_file": str(capture_path),
        "capture_names": sorted(captures),
        "capture_shapes": shapes,
        "elapsed_seconds": time.time() - started,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "checks": {
            "complete_embedding": shapes.get("embedding") == [len(tokens), 5_120],
            "all_64_layers": set(captures) == expected_names,
            "complete_layer_shapes": all(
                shapes.get(f"layer_{index:03d}") == [len(tokens), 5_120]
                for index in range(64)
            ),
            "all_values_finite": finite,
        },
    }
    result["all_checks_passed"] = all(result["checks"].values())
    write_json(OUT_ROOT / "hidden" / "summary.json", result)
    del model, captures
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="verify files, architecture, CUDA and NF4")
    subparsers.add_parser("load", help="load all weights with the fixed CPU/CUDA NF4 map")
    hidden = subparsers.add_parser(
        "hidden", help="capture complete embeddings and all 64 complete hidden-state matrices"
    )
    hidden.add_argument("--prompt", required=True)
    hidden.add_argument("--max-length", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "preflight":
            result = preflight()
        elif args.command == "load":
            result = load_command()
        else:
            result = hidden_command(args.prompt, args.max_length)
    except Exception as error:
        failure = {
            "schema_version": "qwen38_27b_nf4_failure.v1",
            "command": args.command,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "all_checks_passed": False,
        }
        write_json(OUT_ROOT / f"{args.command}_failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        raise SystemExit(1) from error

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)
    (OUT_ROOT / f"{args.command}_failure.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
