
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

snapshot_path = "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

print(f"--- PATH DIAGNOSTICS ---")
print(f"Input path: {snapshot_path}")
print(f"Exists? {os.path.exists(snapshot_path)}")
print(f"Is Directory? {os.path.isdir(snapshot_path)}")

config_path = os.path.join(snapshot_path, "config.json")
print(f"Config exists? {os.path.exists(config_path)}")

print("\n--- ATTEMPTING LOAD ---")
try:
    # Try with forward slashes
    print("Test 1: Forward slashes...")
    m = AutoModelForCausalLM.from_pretrained(snapshot_path, local_files_only=True, trust_remote_code=True)
    print("SUCCESS Test 1")
except Exception as e:
    print(f"FAIL Test 1: {e}")

try:
    # Try with backslashes
    print("\nTest 2: Backslashes...")
    bs_path = snapshot_path.replace("/", "\\")
    m = AutoModelForCausalLM.from_pretrained(bs_path, local_files_only=True, trust_remote_code=True)
    print("SUCCESS Test 2")
except Exception as e:
    print(f"FAIL Test 2: {e}")

try:
    # Try with normpath
    print("\nTest 3: os.path.normpath...")
    n_path = os.path.normpath(snapshot_path)
    m = AutoModelForCausalLM.from_pretrained(n_path, local_files_only=True, trust_remote_code=True)
    print("SUCCESS Test 3")
except Exception as e:
    print(f"FAIL Test 3: {e}")
