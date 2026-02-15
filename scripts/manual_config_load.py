
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

snapshot_path = "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

print(f"Loading config manually from {snapshot_path}...")
try:
    # 1. Load config
    config = AutoConfig.from_pretrained(snapshot_path, local_files_only=True, trust_remote_code=True)
    print("Config loaded. Attempting to load model with this config...")
    
    # 2. Load model using the config object (bypassing path-to-config lookup)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        config=config,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("SUCCESS: Model loaded with manual config object.")
except Exception as e:
    print(f"FAIL: {e}")
