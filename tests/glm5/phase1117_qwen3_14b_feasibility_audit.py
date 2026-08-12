#!/usr/bin/env python3
"""Static, no-weight feasibility audit for a local FP16 Qwen3-14B scale arm."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import psutil
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1117_pythia_training_dynamics_verified_safetensors_v4"
    / "resource"
    / "qwen3_14b_feasibility.json"
)
CONFIG_ROOT = ROOT / "models" / "hf" / "qwen3-14b-feasibility"
REPO = "Qwen/Qwen3-14B"
GPU_BUDGET_GIB = 11
CPU_BUDGET_GIB = 22


def digest(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assigned_device(name: str, device_map: dict[str, Any]) -> str:
    matches = [prefix for prefix in device_map if prefix == "" or name == prefix or name.startswith(prefix + ".")]
    if not matches:
        raise RuntimeError(f"no device assignment for {name}")
    prefix = max(matches, key=len)
    return str(device_map[prefix])


def main() -> None:
    snapshot_download(
        REPO,
        local_dir=CONFIG_ROOT,
        allow_patterns=["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    )
    info = HfApi().model_info(REPO, files_metadata=True)
    weight_files = [
        {"path": sibling.rfilename, "size": int(sibling.size or 0)}
        for sibling in info.siblings
        if sibling.rfilename.endswith(".safetensors")
    ]
    config = AutoConfig.from_pretrained(CONFIG_ROOT, local_files_only=True, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.tie_weights()
    device_map = infer_auto_device_map(
        model,
        max_memory={0: f"{GPU_BUDGET_GIB}GiB", "cpu": f"{CPU_BUDGET_GIB}GiB"},
        dtype=torch.float16,
        no_split_module_classes=["Qwen3DecoderLayer"],
    )
    unique_parameters: dict[int, torch.nn.Parameter] = {}
    for parameter in model.parameters():
        unique_parameters[id(parameter)] = parameter
    total_parameters = sum(parameter.numel() for parameter in unique_parameters.values())
    allocation_bytes: dict[str, int] = {}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        device = assigned_device(name, device_map)
        allocation_bytes[device] = allocation_bytes.get(device, 0) + parameter.numel() * 2

    gpu_total = int(torch.cuda.get_device_properties(0).total_memory)
    ram_total = int(psutil.virtual_memory().total)
    disk_free = int(shutil.disk_usage(ROOT).free)
    weight_bytes = sum(entry["size"] for entry in weight_files)
    disk_assignments = [name for name, device in device_map.items() if str(device) == "disk"]
    gpu_assigned = allocation_bytes.get("0", allocation_bytes.get("cuda:0", 0))
    cpu_assigned = allocation_bytes.get("cpu", 0)
    gib = 1024**3
    checks = {
        "official_config_loaded": total_parameters > 14_000_000_000,
        "fp16_weight_size_matches_parameters": abs(weight_bytes - total_parameters * 2) / max(total_parameters * 2, 1) < 0.02,
        "no_disk_offload_in_empty_map": not disk_assignments,
        "gpu_assignment_within_budget": gpu_assigned <= GPU_BUDGET_GIB * gib,
        "cpu_assignment_within_budget": cpu_assigned <= CPU_BUDGET_GIB * gib,
        "gpu_runtime_headroom_at_least_3_gib": gpu_total - gpu_assigned >= 3 * gib,
        "ram_unassigned_headroom_at_least_5_gib": ram_total - cpu_assigned >= 5 * gib,
        "disk_headroom_after_download_at_least_40_gib": disk_free - weight_bytes >= 40 * gib,
        "no_model_weights_downloaded_or_loaded": not any(CONFIG_ROOT.glob("*.safetensors")),
    }
    core = {
        "schema_version": "phase1117_qwen3_14b_feasibility_audit.v1",
        "phase": 1117,
        "repo": REPO,
        "repo_commit": str(info.sha),
        "precision_target": "fp16",
        "quantization_target": "none",
        "parameter_count": total_parameters,
        "official_weight_files": weight_files,
        "official_weight_bytes": weight_bytes,
        "device_map": {str(key): str(value) for key, value in device_map.items()},
        "allocation_bytes": allocation_bytes,
        "resources": {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_bytes": gpu_total,
            "ram_total_bytes": ram_total,
            "workspace_disk_free_bytes_before_weights": disk_free,
            "gpu_budget_gib": GPU_BUDGET_GIB,
            "cpu_budget_gib": CPU_BUDGET_GIB,
        },
        "checks": checks,
        "all_static_checks_passed": all(checks.values()),
        "actual_weight_load_verified": False,
        "actual_forward_verified": False,
        "scientific_scale_effect_identified": False,
        "decision": (
            "authorize a separately preregistered FP16 GPU+CPU load smoke test"
            if all(checks.values())
            else "do not download weights on this machine"
        ),
        "limits": [
            "An empty-weight device map does not prove that real loading or forward execution succeeds.",
            "GPU+CPU placement will be slower than full-GPU inference.",
            "A 14B result alone cannot identify parameter-count causality; the same frozen protocol must also run on Qwen3-4B.",
        ],
    }
    result = dict(core)
    result["audit_digest"] = digest(core)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_static_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
