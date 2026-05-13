"""DS7B加载测试 — 用CPU加载+手动移动到GPU"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import time
import gc
import torch
from model_utils import MODEL_CONFIGS

print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

from transformers import AutoModelForCausalLM, AutoTokenizer

cfg = MODEL_CONFIGS["deepseek7b"]

# 方法: 先CPU加载, 再8bit量化到GPU
print("\n方法: CPU加载 → 8bit量化移动到GPU...")
t0 = time.time()
try:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 先CPU加载
    print("CPU加载中...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    t1 = time.time()
    print(f"CPU加载耗时: {t1-t0:.1f}s")
    
    # 移到CUDA
    print("移动到CUDA...")
    model = model.to("cuda")
    t2 = time.time()
    print(f"CUDA移动耗时: {t2-t1:.1f}s")
    model.eval()
    
    print(f"device: {next(model.parameters()).device}")
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # 简单推理
    inputs = tokenizer("Hello world", return_tensors="pt", truncation=True, max_length=32)
    input_ids = inputs["input_ids"].to("cuda")
    attn_mask = inputs["attention_mask"].to("cuda")
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    
    print(f"推理成功! {len(out.hidden_states)}层, last_norm={out.hidden_states[-1].float().norm():.2f}")
    
    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    print("完成!")
    
except Exception as e:
    print(f"失败: {e}")
    import traceback; traceback.print_exc()
