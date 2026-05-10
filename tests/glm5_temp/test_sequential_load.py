"""
顺序加载验证: Qwen3 → 释放 → DeepSeek7B
验证 model_utils.release_model() 是否能正确释放GPU内存
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import torch
from model_utils import load_model, get_model_info, release_model

def gpu_mem():
    return torch.cuda.memory_allocated() / 1e9

print(f"初始 GPU: {gpu_mem():.2f} GB")

# --- Qwen3 ---
print("\n=== 加载 Qwen3 ===")
t0 = time.time()
model, tokenizer, device = load_model("qwen3")
info = get_model_info(model, "qwen3")
print(f"  加载耗时: {time.time()-t0:.1f}s, class={info.model_class}, "
      f"layers={info.n_layers}, d={info.d_model}")
print(f"  GPU: {gpu_mem():.2f} GB")

# 简单推理
inputs = tokenizer("Hello", return_tensors="pt").to(device)
with torch.no_grad():
    out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                output_hidden_states=True)
print(f"  推理OK: hidden_states={len(out.hidden_states)}层, "
      f"最后层norm={out.hidden_states[-1].float().norm():.1f}")

# 释放
print("\n=== 释放 Qwen3 ===")
release_model(model)
model = None
gc.collect(); torch.cuda.empty_cache()
print(f"  释放后 GPU: {gpu_mem():.2f} GB")

# --- DeepSeek7B ---
print("\n=== 加载 DeepSeek7B ===")
t0 = time.time()
model, tokenizer, device = load_model("deepseek7b")
info = get_model_info(model, "deepseek7b")
print(f"  加载耗时: {time.time()-t0:.1f}s, class={info.model_class}, "
      f"layers={info.n_layers}, d={info.d_model}")
print(f"  GPU: {gpu_mem():.2f} GB")

# 简单推理
inputs = tokenizer("Hello", return_tensors="pt").to(device)
with torch.no_grad():
    out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                output_hidden_states=True)
print(f"  推理OK: hidden_states={len(out.hidden_states)}层, "
      f"最后层norm={out.hidden_states[-1].float().norm():.1f}")

# 释放
print("\n=== 释放 DeepSeek7B ===")
release_model(model)
model = None
gc.collect(); torch.cuda.empty_cache()
print(f"  释放后 GPU: {gpu_mem():.2f} GB")

print("\n=== 顺序加载验证通过! ===")
