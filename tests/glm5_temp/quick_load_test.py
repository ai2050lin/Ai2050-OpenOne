"""
临时脚本: 快速测试DS7B是否能加载和前向推理
使用device_map="auto"让transformers自动分配
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os
os.environ["HF_DISABLE_CACHING_ALLOCATOR_WARMUP"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
configs = {
    "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
}
path = configs[model_name]
print(f"=== Testing {model_name} ===")

# Step 1: Try bfloat16 with device_map="auto"
print("\n[1] Trying bfloat16 with device_map=auto...")
try:
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s, device_map={model.hf_device_map}")
    
    # Forward
    prompt = "Translate the word cat into Chinese."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    attention_mask = inputs["attention_mask"].to(next(model.parameters()).device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
    n_layers = len(out.hidden_states)
    d_model = out.hidden_states[0].shape[-1]
    print(f"  Forward OK: {n_layers} layers, d_model={d_model}")
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory: {gpu_mem:.2f} GB")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"  SUCCESS with bfloat16!")
    
except Exception as e:
    print(f"  FAILED (bfloat16): {type(e).__name__}: {str(e)[:200]}")
    try:
        del model; gc.collect(); torch.cuda.empty_cache()
    except:
        pass

# Step 2: Try 8-bit
print("\n[2] Trying 8-bit...")
try:
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        path,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s")
    
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"].to(model.device), 
                    attention_mask=inputs["attention_mask"].to(model.device),
                    output_hidden_states=True)
    
    n_layers = len(out.hidden_states)
    d_model = out.hidden_states[0].shape[-1]
    print(f"  Forward OK: {n_layers} layers, d_model={d_model}")
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory: {gpu_mem:.2f} GB")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"  SUCCESS with 8-bit!")
    
except Exception as e:
    print(f"  FAILED (8-bit): {type(e).__name__}: {str(e)[:200]}")
    try:
        del model; gc.collect(); torch.cuda.empty_cache()
    except:
        pass

print(f"\nDone testing {model_name}")
