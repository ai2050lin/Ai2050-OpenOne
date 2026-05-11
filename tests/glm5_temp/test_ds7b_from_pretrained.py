"""测试DS7B from_pretrained - 设置环境变量 + device_map=auto"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
# 设置环境变量优化内存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DISABLE_CACHING_ALLOCATOR_WARMUP"] = "1"

import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

print("="*60)
print("DS7B from_pretrained + device_map=auto")
print("="*60)

# 检查内存
import psutil
m = psutil.virtual_memory()
print(f"RAM: {m.total/1e9:.1f}GB, available: {m.available/1e9:.1f}GB")
print(f"GPU: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Step 1: Tokenizer
print("\n--- Step 1: Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
print("Tokenizer OK")

# Step 2: from_pretrained with device_map=auto
print("\n--- Step 2: from_pretrained (device_map=auto) ---")
t0 = time.time()
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "11GiB", "cpu": "28GiB"},
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    t1 = time.time()
    print(f"Model loaded in {t1-t0:.1f}s")
    if hasattr(model, "hf_device_map"):
        dm = model.hf_device_map
        gpu_count = sum(1 for v in dm.values() if v == 0 or v == "cuda:0")
        cpu_count = sum(1 for v in dm.values() if v == "cpu")
        print(f"Device map: GPU={gpu_count}, CPU={cpu_count}")
except Exception as e:
    print(f"from_pretrained failed: {e}")
    import traceback; traceback.print_exc()

    # 尝试方案B: 手动加载
    print("\n--- Step 2b: 手动加载 (safetensors + load_state_dict) ---")
    from safetensors.torch import load_file
    from transformers import Qwen2ForCausalLM, Qwen2Config
    import json

    state1 = load_file(os.path.join(MODEL_PATH, "model-00001-of-000002.safetensors"))
    state2 = load_file(os.path.join(MODEL_PATH, "model-00002-of-000002.safetensors"))
    state_dict = {**state1, **state2}
    del state1, state2
    print(f"State dict loaded: {len(state_dict)} keys")

    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        config_dict = json.load(f)
    config = Qwen2Config(**config_dict)

    # 直接在CPU上创建模型（不用meta device）
    print("Creating model on CPU...")
    model = Qwen2ForCausalLM(config)
    print("Loading state_dict (bf16)...")
    state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    del state_dict
    model.load_state_dict(state_dict_bf16, strict=False)
    del state_dict_bf16
    model.eval()
    print("Model on CPU OK")

    # 使用accelerate dispatch分配GPU+CPU
    from accelerate import infer_auto_device_map, dispatch_model
    print("Computing device map...")
    device_map = infer_auto_device_map(
        model,
        max_memory={0: "11GiB", "cpu": "28GiB"},
        no_split_module_classes=["Qwen2DecoderLayer"]
    )
    gpu_count = sum(1 for v in device_map.values() if v == 0 or v == "cuda:0")
    cpu_count = sum(1 for v in device_map.values() if v == "cpu")
    print(f"Device map: GPU={gpu_count}, CPU={cpu_count}")

    print("Dispatching model...")
    model = dispatch_model(model, device_map=device_map)
    print("Dispatch OK")

gpu_mem = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {gpu_mem:.2f}GB")

# Step 3: 推理
print("\n--- Step 3: 推理 ---")
device = next(model.parameters()).device
prompt = "The scientist discovered a new"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

t0 = time.time()
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
t_fwd = time.time() - t0
print(f"Forward: {t_fwd:.2f}s, layers={len(out.hidden_states)}")

import numpy as np
logits = out.logits[0, -1].float().cpu().numpy()
top5_ids = np.argsort(logits)[-5:][::-1]
top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
print(f"Top-5: {top5}")

# 释放
print("\n--- 释放 ---")
del model; gc.collect(); torch.cuda.empty_cache()
print(f"GPU after release: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print("\nDone!")
