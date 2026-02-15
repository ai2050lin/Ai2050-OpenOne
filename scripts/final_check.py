
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

path = r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"
print(f"PATH: {path}")

try:
    print("Loading with AutoConfig.from_pretrained(path)...")
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    print("Config SUCCESS")
    
    print("Loading with AutoModelForCausalLM.from_pretrained(path, config=config)...")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("MODEL SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
