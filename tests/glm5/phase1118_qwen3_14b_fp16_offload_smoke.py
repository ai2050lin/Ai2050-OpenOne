#!/usr/bin/env python3
"""Actual FP16 GPU+CPU load and forward smoke test for Qwen3-14B."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import psutil
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import quantization_audit


PHASE = 1118
REPO = "Qwen/Qwen3-14B"
EXPECTED_COMMIT = "40c069824f4251a91eefaf281ebe4c544efd3e18"
MODEL_ROOT = ROOT / "models" / "hf" / "Qwen3-14B"
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1118_qwen3_14b_fp16_offload_smoke"
SOURCE_AUDIT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1117_pythia_training_dynamics_verified_safetensors_v4"
    / "resource"
    / "qwen3_14b_feasibility.json"
)
PREFETCH_AUDIT = OUT_ROOT / "download" / "prefetch_audit.json"
PROMPTS = (
    "The capital of France is",
    "Two plus two equals",
    "An apple is a kind of",
    "Water freezes at zero degrees",
    "The opposite of hot is",
    "A triangle has three",
    "The color of fresh grass is",
    "The largest planet in the Solar System is",
)
GPU_WEIGHT_BYTES = 13_447_054_336
CPU_WEIGHT_BYTES = 16_089_560_064
GIB = 1024**3
OFFLOAD_ROOT = OUT_ROOT / "disk_offload_revision5"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    source = read_json(SOURCE_AUDIT)
    prefetch = read_json(PREFETCH_AUDIT)
    if not source["all_static_checks_passed"] or source["repo_commit"] != EXPECTED_COMMIT:
        raise RuntimeError("Phase1117 Qwen3-14B static feasibility gate did not pass")
    if not prefetch["all_checks_passed"] or prefetch["repo_commit"] != EXPECTED_COMMIT:
        raise RuntimeError("Phase1118 verified weight prefetch gate did not pass")
    if not all((MODEL_ROOT / row["name"]).exists() and row["passed"] for row in prefetch["rows"]):
        raise RuntimeError("Phase1118 verified weight files are incomplete")
    device_map = {
        key: (int(value) if value.isdigit() else value)
        for key, value in source["device_map"].items()
    }
    for layer in range(13, 18):
        device_map[f"model.layers.{layer}"] = 0
    for key, value in tuple(device_map.items()):
        if value == "cpu" and key != "model.rotary_emb":
            device_map[key] = "disk"
    available_ram_before_load = int(psutil.virtual_memory().available)
    free_gpu_before_load = int(torch.cuda.mem_get_info()[0])
    free_disk_before_load = int(psutil.disk_usage(str(OUT_ROOT)).free)
    protocol_core = {
        "schema_version": "phase1118_qwen3_14b_fp16_smoke_protocol.v5",
        "phase": PHASE,
        "repo": REPO,
        "expected_commit": EXPECTED_COMMIT,
        "source_static_audit_digest": source["audit_digest"],
        "source_prefetch_audit_digest": prefetch["audit_digest"],
        "precision": "fp16",
        "quantization": "none",
        "device_map": {str(key): str(value) for key, value in device_map.items()},
        "prompts": list(PROMPTS),
        "thresholds": {
            "parameter_count": 14_768_307_200,
            "finite_forward_fraction": 1.0,
            "minimum_gpu_layers": 18,
            "minimum_disk_layers": 22,
            "disk_offload_required": True,
            "minimum_gpu_runtime_headroom_bytes": int(1.5 * GIB),
            "minimum_ram_runtime_headroom_bytes": 8 * GIB,
            "minimum_disk_runtime_headroom_bytes": 20 * GIB,
        },
        "planned_allocation_bytes": {"0": GPU_WEIGHT_BYTES, "disk": CPU_WEIGHT_BYTES},
        "resources_before_load": {
            "gpu_free_bytes": free_gpu_before_load,
            "ram_available_bytes": available_ram_before_load,
            "disk_free_bytes": free_disk_before_load,
        },
        "engineering_revision": {
            "revision": 5,
            "reason": "bypass the from_pretrained conversion buffer with an empty meta model and Accelerate checkpoint dispatch while preserving the revision-4 execution claim and gates",
            "load_strategy": "accelerate_empty_model_indexed_checkpoint_dispatch_with_disk_offload",
            "revision_1_status": "aborted during download before any weight load or model output",
            "revision_2_status": "terminated by Windows Resource-Exhaustion-Detector event 2004 at weight 1/443 before any model output",
            "revision_2_private_commit_bytes": 49_734_443_008,
            "revision_2_weight_or_forward_result_exists": False,
            "revision_3_status": "terminated by Windows Resource-Exhaustion-Detector event 2004 before any model output; a separate redirected launcher also failed before load",
            "revision_3_private_commit_bytes": 49_326_837_760,
            "revision_3_weight_or_forward_result_exists": False,
            "revision_4_status": "from_pretrained disk-offload conversion failed before load while requesting a 1555824640-byte CPU buffer",
            "revision_4_weight_or_forward_result_exists": False,
        },
        "scientific_behavior_claim_authorized": False,
        "model_outputs_read_before_protocol": False,
    }
    protocol = dict(protocol_core)
    protocol["protocol_digest"] = digest(protocol_core)
    write_json(OUT_ROOT / "protocol" / "protocol.json", protocol)

    started = time.time()
    model = None
    weights_loaded = False
    result: dict[str, Any]
    try:
        if free_gpu_before_load - GPU_WEIGHT_BYTES < protocol["thresholds"]["minimum_gpu_runtime_headroom_bytes"]:
            raise RuntimeError("insufficient current GPU headroom for the frozen v3 map")
        if available_ram_before_load < protocol["thresholds"]["minimum_ram_runtime_headroom_bytes"]:
            raise RuntimeError("insufficient current RAM headroom for the frozen v4 disk-offload map")
        if free_disk_before_load - CPU_WEIGHT_BYTES < protocol["thresholds"]["minimum_disk_runtime_headroom_bytes"]:
            raise RuntimeError("insufficient current disk headroom for the frozen v4 disk-offload map")
        OFFLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        local_path = str(MODEL_ROOT)
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(local_path, local_files_only=True, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                dtype=torch.float16,
                trust_remote_code=True,
            )
        model.tie_weights()
        torch.cuda.reset_peak_memory_stats()
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=local_path,
            device_map=device_map,
            no_split_module_classes=list(model._no_split_modules),
            offload_folder=str(OFFLOAD_ROOT),
            offload_buffers=False,
            dtype=torch.float16,
            offload_state_dict=True,
            force_hooks=True,
            strict=True,
        )
        weights_loaded = True
        model.eval()
        precision = quantization_audit(model)
        actual_map = {str(key): str(value) for key, value in model.hf_device_map.items()}
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            encoded = tokenizer(list(PROMPTS), return_tensors="pt", add_special_tokens=False, padding=True)
            input_ids = encoded["input_ids"].to("cuda:0")
            attention_mask = encoded["attention_mask"].to("cuda:0")
            before = time.time()
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            logits = output.logits[:, -1, :].float()
            batch_seconds = time.time() - before
            for index, prompt in enumerate(PROMPTS):
                row_logits = logits[index]
                top_id = int(torch.argmax(row_logits, dim=-1).item())
                top_logit = float(row_logits[top_id].item())
                finite = bool(torch.isfinite(row_logits).all().item()) and math.isfinite(top_logit)
                rows.append(
                    {
                        "index": index,
                        "prompt": prompt,
                        "input_length": int(attention_mask[index].sum().item()),
                        "finite": finite,
                        "top_token_id": top_id,
                        "top_token_text": tokenizer.decode([top_id]),
                        "top_logit": top_logit,
                        "batched_forward_seconds": batch_seconds,
                    }
                )
            del output, logits, input_ids, attention_mask

        finite_fraction = sum(row["finite"] for row in rows) / len(rows)
        gpu_layer_count = sum(key.startswith("model.layers.") and value == "0" for key, value in actual_map.items())
        disk_layer_count = sum(key.startswith("model.layers.") and value == "disk" for key, value in actual_map.items())
        checks = {
            "parameter_count": parameter_count == protocol["thresholds"]["parameter_count"],
            "fp16_parameters": precision["has_fp16_parameters"] and not precision["has_bf16_parameters"],
            "not_quantized": not precision["has_quantized_modules"],
            "finite_forward_fraction": finite_fraction == protocol["thresholds"]["finite_forward_fraction"],
            "gpu_layer_count": gpu_layer_count >= protocol["thresholds"]["minimum_gpu_layers"],
            "disk_layer_count": disk_layer_count >= protocol["thresholds"]["minimum_disk_layers"],
            "disk_offload_used": any(value == "disk" for value in actual_map.values()),
            "protocol_device_map_preserved": actual_map == protocol["device_map"],
        }
        core = {
            "schema_version": "phase1118_qwen3_14b_fp16_smoke_result.v5",
            "phase": PHASE,
            "protocol_digest": protocol["protocol_digest"],
            "repo": REPO,
            "repo_commit": EXPECTED_COMMIT,
            "precision": precision,
            "parameter_count": parameter_count,
            "actual_device_map": actual_map,
            "finite_forward_fraction": finite_fraction,
            "rows": rows,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "process_rss_bytes": int(psutil.Process().memory_info().rss),
            "elapsed_seconds": time.time() - started,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "actual_weight_load_verified": True,
            "actual_forward_verified": True,
            "scientific_scale_effect_identified": False,
            "decision": (
                "local FP16 Qwen3-14B GPU-plus-disk execution is feasible; GPU-plus-CPU execution remains rejected on this Windows commit limit"
                if all(checks.values())
                else "do not authorize a Qwen3-14B scale experiment on this load path"
            ),
        }
        result = dict(core)
        result["result_digest"] = digest(core)
    except Exception as error:
        core = {
            "schema_version": "phase1118_qwen3_14b_fp16_smoke_result.v5",
            "phase": PHASE,
            "protocol_digest": protocol["protocol_digest"],
            "repo": REPO,
            "repo_commit": EXPECTED_COMMIT,
            "elapsed_seconds": time.time() - started,
            "all_checks_passed": False,
            "actual_weight_load_verified": weights_loaded,
            "actual_forward_verified": False,
            "scientific_scale_effect_identified": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "decision": "do not authorize a Qwen3-14B scale experiment on this load path",
        }
        result = dict(core)
        result["result_digest"] = digest(core)
    finally:
        write_json(OUT_ROOT / "result" / "smoke_result.json", result)
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
