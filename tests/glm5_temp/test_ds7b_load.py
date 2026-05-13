"""快速测试DS7B加载 — 用bfloat16+device_map=auto"""
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

print("\n尝试 bfloat16 + device_map=auto...")
t0 = time.time()
try:
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    t1 = time.time()
    print(f"bfloat16加载耗时: {t1-t0:.1f}s")
    print(f"device: {next(model.parameters()).device}")
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # 简单推理
    inputs = tokenizer("Hello world", return_tensors="pt", truncation=True, max_length=32)
    input_device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs["attention_mask"].to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    
    print(f"推理成功! {len(out.hidden_states)}层, last_norm={out.hidden_states[-1].float().norm():.2f}")
    
    del model, out
    gc.collect(); torch.cuda.empty_cache()
    print("完成!")
    
except Exception as e:
    print(f"bfloat16加载失败: {e}")
    import traceback; traceback.print_exc()
