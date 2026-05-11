"""手动加载DS7B - 绕过from_pretrained卡死问题"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os
import gc
import time
import json
import torch
import psutil

path = "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

print("="*60)
print("手动加载DS7B - 分步诊断")
print("="*60)

m = psutil.virtual_memory()
print(f"初始RAM: 可用{m.available/1e9:.1f}GB")
print(f"GPU: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Step 1: 加载tokenizer
print("\n--- Step 1: Tokenizer ---")
from transformers import AutoTokenizer
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
print(f"Tokenizer: {time.time()-t0:.1f}s")

# Step 2: 用torch直接加载safetensors
print("\n--- Step 2: 手动加载safetensors ---")
from safetensors.torch import load_file

shard1 = os.path.join(path, "model-00001-of-000002.safetensors")
shard2 = os.path.join(path, "model-00002-of-000002.safetensors")

t0 = time.time()
print(f"加载分片1: {os.path.getsize(shard1)/1e9:.2f}GB ...")
state1 = load_file(shard1)
print(f"分片1加载: {time.time()-t0:.1f}s, keys={len(state1)}")
m = psutil.virtual_memory()
print(f"RAM: 可用{m.available/1e9:.1f}GB")

t0 = time.time()
print(f"加载分片2: {os.path.getsize(shard2)/1e9:.2f}GB ...")
state2 = load_file(shard2)
print(f"分片2加载: {time.time()-t0:.1f}s, keys={len(state2)}")
m = psutil.virtual_memory()
print(f"RAM: 可用{m.available/1e9:.1f}GB")

# 合并state dict
state_dict = {**state1, **state2}
del state1, state2
print(f"合并后state_dict: {len(state_dict)} keys")

# Step 3: 创建模型结构并加载权重
print("\n--- Step 3: 创建模型结构 ---")
from transformers import Qwen2ForCausalLM, Qwen2Config

with open(os.path.join(path, "config.json")) as f:
    config_dict = json.load(f)

config = Qwen2Config(**config_dict)
print(f"Config: layers={config.num_hidden_layers}, d={config.hidden_size}")

t0 = time.time()
model = Qwen2ForCausalLM(config)
print(f"模型结构创建: {time.time()-t0:.1f}s")

# Step 4: 加载state dict
print("\n--- Step 4: 加载权重 ---")
t0 = time.time()
# 转换dtype
state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
del state_dict
model.load_state_dict(state_dict_bf16, strict=False)
del state_dict_bf16
print(f"权重加载: {time.time()-t0:.1f}s")

m = psutil.virtual_memory()
print(f"RAM: 可用{m.available/1e9:.1f}GB")

# Step 5: 移到CUDA
print("\n--- Step 5: 移到CUDA ---")
model.eval()
t0 = time.time()
model = model.to("cuda")
print(f"移到CUDA: {time.time()-t0:.1f}s")
gpu_mem = torch.cuda.memory_allocated() / 1e9
print(f"GPU: {gpu_mem:.2f}GB")

# Step 6: 简短推理
print("\n--- Step 6: 推理 ---")
inputs = tokenizer("Hello world", return_tensors="pt", truncation=True, max_length=16)
input_ids = inputs["input_ids"].to("cuda:0")
attention_mask = inputs["attention_mask"].to("cuda:0")

t0 = time.time()
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
print(f"推理: {time.time()-t0:.2f}s, 层数={len(out.hidden_states)}")

# Step 7: 释放
print("\n--- Step 7: 释放 ---")
del model; gc.collect(); torch.cuda.empty_cache()
print(f"释放后GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print("\n手动加载DS7B成功!")
