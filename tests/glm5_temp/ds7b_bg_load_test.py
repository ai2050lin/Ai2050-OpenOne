"""
后台测试: DS7B加载 — 结果写入日志文件
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["HF_DISABLE_CACHING_ALLOCATOR_WARMUP"] = "1"

import gc
import time
import torch
from datetime import datetime

LOG_FILE = "tests/glm5_temp/ds7b_load_test.log"

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

log(f"=== DS7B Load Test Start ===")

path = "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

try:
    log("Loading tokenizer...")
    from transformers import AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer loaded in {time.time()-t0:.1f}s, vocab={len(tokenizer)}")
    
    log("Loading model (bfloat16, device_map=auto)...")
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    log(f"Model loaded in {time.time()-t0:.1f}s")
    log(f"Device map: {model.hf_device_map}")
    log(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    log("Running forward pass...")
    prompt = "Translate the word cat into Chinese."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    log(f"Forward pass in {time.time()-t0:.1f}s")
    log(f"Hidden states: {len(out.hidden_states)} layers, d_model={out.hidden_states[0].shape[-1]}")
    
    log("Releasing model...")
    del model; gc.collect(); torch.cuda.empty_cache()
    log("SUCCESS! DS7B can be loaded with bfloat16 + device_map=auto")
    
except Exception as e:
    log(f"FAILED: {type(e).__name__}: {str(e)[:300]}")
    import traceback
    log(traceback.format_exc()[:500])
    try:
        del model; gc.collect(); torch.cuda.empty_cache()
    except:
        pass

log("=== DS7B Load Test End ===")
