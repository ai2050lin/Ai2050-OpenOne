from __future__ import annotations

import json
from pathlib import Path

import psutil
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "models" / "hf" / "qwen3-14b-feasibility"


def assigned_device(name: str, device_map: dict[str, object]) -> str:
    prefixes = [key for key in device_map if key == "" or name == key or name.startswith(key + ".")]
    return str(device_map[max(prefixes, key=len)])


config = AutoConfig.from_pretrained(CONFIG_ROOT, local_files_only=True, trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.tie_weights()

rows = []
for gpu_budget in (12, 13, 14):
    device_map = infer_auto_device_map(
        model,
        max_memory={0: f"{gpu_budget}GiB", "cpu": "20GiB"},
        dtype=torch.float16,
        no_split_module_classes=["Qwen3DecoderLayer"],
    )
    allocation: dict[str, int] = {}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        device = assigned_device(name, device_map)
        allocation[device] = allocation.get(device, 0) + parameter.numel() * 2
    rows.append(
        {
            "gpu_budget_gib": gpu_budget,
            "device_map": {str(key): str(value) for key, value in device_map.items()},
            "allocation_bytes": allocation,
            "available_ram_bytes": int(psutil.virtual_memory().available),
            "gpu_free_bytes": int(torch.cuda.mem_get_info()[0]),
        }
    )
print(json.dumps(rows, indent=2))
