"""手动加载DS7B - 使用accelerate dispatch分配CPU/GPU
关键: DS7B ~15GB > GPU 12GB, 必须用device_map自动分配
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os
import gc
import time
import json
import torch
import numpy as np
from safetensors.torch import load_file
from transformers import Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
from accelerate import infer_auto_device_map, dispatch_model

MODEL_PATH = "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

print("="*60)
print("手动加载DS7B (accelerate dispatch)")
print("="*60)

# Step 1: Tokenizer
print("\n--- Step 1: Tokenizer ---")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer: {time.time()-t0:.1f}s")

# Step 2: 加载safetensors权重
print("\n--- Step 2: 加载safetensors权重 (mmap) ---")
t0 = time.time()
state1 = load_file(os.path.join(MODEL_PATH, "model-00001-of-000002.safetensors"))
state2 = load_file(os.path.join(MODEL_PATH, "model-00002-of-000002.safetensors"))
state_dict = {**state1, **state2}
del state1, state2
print(f"State dict: {len(state_dict)} keys, {time.time()-t0:.1f}s")

# Step 3: 创建模型 (meta device)
print("\n--- Step 3: 创建模型结构 (meta) ---")
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    config_dict = json.load(f)
config = Qwen2Config(**config_dict)
print(f"Config: layers={config.num_hidden_layers}, d={config.hidden_size}")

with torch.device("meta"):
    model = Qwen2ForCausalLM(config)
print(f"模型结构创建 OK")

# Step 4: 加载权重
print("\n--- Step 4: 加载权重 ---")
t0 = time.time()
state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
del state_dict
missing, unexpected = model.load_state_dict(state_dict_bf16, strict=False, assign=True)
del state_dict_bf16
print(f"权重加载: {time.time()-t0:.1f}s, missing={len(missing)}, unexpected={len(unexpected)}")

# Step 5: 自动分配设备
print("\n--- Step 5: 自动分配设备 (GPU+CPU) ---")
model.eval()
max_gpu_mem = "11GiB"  # 留一些余量
max_cpu_mem = "28GiB"
device_map = infer_auto_device_map(
    model,
    max_memory={0: max_gpu_mem, "cpu": max_cpu_mem},
    no_split_module_classes=["Qwen2DecoderLayer"]
)
# 统计分配
gpu_layers = sum(1 for v in device_map.values() if v == 0 or v == "cuda:0")
cpu_layers = sum(1 for v in device_map.values() if v == "cpu")
print(f"Device map: GPU layers={gpu_layers}, CPU layers={cpu_layers}")
# 打印前10个分配
for i, (k, v) in enumerate(list(device_map.items())[:10]):
    print(f"  {k}: {v}")
print(f"  ... (total {len(device_map)} entries)")

t0 = time.time()
model = dispatch_model(model, device_map=device_map)
print(f"Dispatch: {time.time()-t0:.1f}s")
gpu_mem = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {gpu_mem:.2f}GB")

# Step 6: 前向推理
print("\n--- Step 6: 前向推理 ---")
prompt = "The scientist discovered a new"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
# 移到模型第一个设备
first_device = next(model.parameters()).device
input_ids = inputs["input_ids"].to(first_device)
attention_mask = inputs["attention_mask"].to(first_device)

t0 = time.time()
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
t_fwd = time.time() - t0
print(f"推理: {t_fwd:.2f}s, 层数={len(out.hidden_states)}")

logits = out.logits[0, -1].float().cpu().numpy()
top5_ids = np.argsort(logits)[-5:][::-1]
top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
print(f"Top-5: {top5}")

# Step 7: Hook提取中间层输出
print("\n--- Step 7: Hook提取 ---")
from model_utils import get_layers
layers = get_layers(model)
captured = {}
def make_hook(key):
    def hook(module, input, output):
        if isinstance(output, tuple):
            captured[key] = output[0].detach().float().cpu()
        else:
            captured[key] = output.detach().float().cpu()
    return hook

hook_indices = [0, 14, 27]
hooks = [layers[li].register_forward_hook(make_hook(f"L{li}")) for li in hook_indices]

with torch.no_grad():
    model(input_ids=input_ids, attention_mask=attention_mask)
for h in hooks:
    h.remove()

for key in sorted(captured.keys()):
    t = captured[key]
    print(f"  Hook {key}: shape={t.shape}, norm={t.float().norm():.2f}")

# Step 8: 释放
print("\n--- Step 8: 释放 ---")
del model; gc.collect(); torch.cuda.empty_cache()
print(f"释放后GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print("\n手动加载DS7B成功!")
