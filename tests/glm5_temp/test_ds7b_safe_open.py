"""
DS7B 加载 - 用safe_open逐tensor加载，避免mmap冲突
关键：safe_open读取每个tensor时显式clone()，确保真实内存副本
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import gc
import time
import os
import json
import torch
import numpy as np
from safetensors import safe_open

MODEL_PATH = "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def fix_meta_tensors(model):
    """修复meta device上残留的buffer"""
    fixed = []
    for name, module in model.named_modules():
        for buf_name, buf in list(module.named_buffers(recurse=False)):
            if buf.is_meta:
                if 'inv_freq' in buf_name:
                    dim = buf.shape[-1]
                    base = getattr(module, 'base', 10000.0)
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim * 2, 2, dtype=torch.float32) / (dim * 2)))
                    new_buf = inv_freq.to(buf.dtype)
                    setattr(module, buf_name, new_buf)
                    fixed.append(f"{name}.{buf_name}")
                else:
                    new_buf = torch.zeros(buf.shape, dtype=buf.dtype)
                    setattr(module, buf_name, new_buf)
                    fixed.append(f"{name}.{buf_name}")
    return fixed


def load_shard_safe(shard_path):
    """用safe_open逐tensor加载，clone确保真实副本"""
    state_dict = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for i, key in enumerate(keys):
            tensor = f.get_tensor(key)
            # clone()确保是真实内存副本，不依赖mmap
            state_dict[key] = tensor.clone()
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(keys)} tensors loaded", flush=True)
    return state_dict


def load_ds7b():
    """手动加载DS7B"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # 1. Tokenizer
    print("[1] Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"    OK, vocab={len(tokenizer)}", flush=True)

    # 2. Config
    print("[2] Config...", flush=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    print(f"    {config.model_type}, {config.num_hidden_layers} layers, d={config.hidden_size}", flush=True)

    # 3. Meta device空模型
    print("[3] Meta model...", flush=True)
    t0 = time.time()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to(torch.bfloat16)
    print(f"    Created in {time.time()-t0:.1f}s", flush=True)

    # 4. 逐分片加载权重
    print("[4] Loading weights (safe_open + clone)...", flush=True)
    t0 = time.time()
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"    {len(shard_files)} shards", flush=True)
    
    total_loaded = 0
    for shard_file in shard_files:
        shard_path = os.path.join(MODEL_PATH, shard_file)
        sz = os.path.getsize(shard_path) / 1e9
        print(f"    {shard_file} ({sz:.2f}GB)...", flush=True)
        
        t1 = time.time()
        shard_state = load_shard_safe(shard_path)
        t_load = time.time() - t1
        print(f"      Loaded {len(shard_state)} tensors in {t_load:.1f}s", flush=True)
        
        # 加载到模型
        t1 = time.time()
        model.load_state_dict(shard_state, strict=False, assign=True)
        t_assign = time.time() - t1
        total_loaded += len(shard_state)
        print(f"      Assigned in {t_assign:.1f}s, total={total_loaded}", flush=True)
        
        del shard_state
        gc.collect()
    
    print(f"    All weights: {time.time()-t0:.1f}s", flush=True)

    # 5. 修复meta tensors
    print("[5] Fixing meta tensors...", flush=True)
    fixed = fix_meta_tensors(model)
    print(f"    Fixed {len(fixed)}: {fixed[:3]}", flush=True)

    # 检查
    meta_params = sum(1 for p in model.parameters() if p.is_meta)
    meta_bufs = sum(1 for b in model.buffers() if b.is_meta)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Meta: {meta_params} params, {meta_bufs} bufs", flush=True)
    print(f"    Total: {total_params/1e9:.2f}B params", flush=True)

    model.eval()
    return model, tokenizer


def test_inference(model, tokenizer):
    """CPU推理测试"""
    print("\n[Test] CPU forward...", flush=True)
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], 
                    output_hidden_states=True)
    t_fwd = time.time() - t0
    hs = out.hidden_states
    print(f"    Forward: {t_fwd:.1f}s, {len(hs)} layers, last norm={hs[-1].float().norm():.2f}", flush=True)

    logits = out.logits[0, -1].float().numpy()
    top5_ids = np.argsort(logits)[-5:][::-1]
    top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
    print(f"    Top-5: {top5}", flush=True)

    return {"fwd_time": round(t_fwd, 1), "top5": top5, "n_layers": len(hs)}


if __name__ == "__main__":
    t0 = time.time()
    model, tokenizer = load_ds7b()
    t_load = time.time() - t0
    print(f"\nLoad time: {t_load:.1f}s", flush=True)

    result = test_inference(model, tokenizer)
    print(f"\nResult: {result}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Done!", flush=True)
