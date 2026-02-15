
import os
import sys

import torch
import transformers

print(f"--- ENV TEST ---")
print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CWD: {os.getcwd()}")
print(f"PYTHONPATH: {sys.path}")

snapshot_path = "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
config_file = os.path.join(snapshot_path, "config.json")
print(f"Config path: {config_file}")
print(f"Config exists via os.path: {os.path.exists(config_file)}")

try:
    with open(config_file, "r") as f:
        print("Config readable via open()")
except Exception as e:
    print(f"Config NOT readable via open(): {e}")

try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(snapshot_path, local_files_only=True)
    print("AutoConfig SUCCESS")
except Exception as e:
    print(f"AutoConfig FAIL: {e}")
