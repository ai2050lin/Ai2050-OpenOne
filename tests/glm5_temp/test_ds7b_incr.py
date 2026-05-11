"""
绕过 from_pretrained 卡死 - DS7B 逐分片加载
关键：加载一个分片后立即load进模型，释放dict内存
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import gc
import time
import os
import json
import torch
import numpy as np
from safetensors.torch import load_file

MODEL_PATH = "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def fix_meta_tensors(model):
    """修复meta device上残留的tensor"""
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


def load_ds7b_incremental():
    """逐分片加载DS7B，避免内存峰值"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # 1. Tokenizer
    print("[1] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"    OK, vocab={len(tokenizer)}", flush=True)

    # 2. Config
    print("[2] Loading config...", flush=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    print(f"    {config.model_type}, {config.num_hidden_layers} layers, d={config.hidden_size}", flush=True)

    # 3. 直接在CPU上创建模型（不用meta device）
    # 先估算内存需求
    import psutil
    avail_mem = psutil.virtual_memory().available
    print(f"    Available RAM: {avail_mem/1e9:.1f}GB", flush=True)
    
    # 方案：用from_pretrained加载到CPU，但分步处理
    # 尝试直接用from_pretrained看是否能加载
    print("[3] Attempting from_pretrained with device_map=cpu...", flush=True)
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        print(f"    from_pretrained succeeded in {time.time()-t0:.1f}s!", flush=True)
    except Exception as e:
        print(f"    from_pretrained failed: {e}", flush=True)
        print("    Falling back to manual loading...", flush=True)
        model = None
    
    if model is not None:
        model.eval()
        return model, tokenizer, "cpu"

    # Fallback: meta device + 逐分片加载
    print("[3b] Creating meta model...", flush=True)
    t0 = time.time()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to(torch.bfloat16)
    print(f"    Created in {time.time()-t0:.1f}s", flush=True)

    # 4. 逐分片加载权重
    print("[4] Loading weights shard by shard...", flush=True)
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    print(f"    {len(shard_files)} shards: {shard_files}", flush=True)
    
    # 为每个分片创建key列表
    shard_keys = {}
    for key, shard in weight_map.items():
        if shard not in shard_keys:
            shard_keys[shard] = []
        shard_keys[shard].append(key)
    
    total_loaded = 0
    for shard_file in shard_files:
        shard_path = os.path.join(MODEL_PATH, shard_file)
        sz = os.path.getsize(shard_path) / 1e9
        print(f"    Loading {shard_file} ({sz:.2f}GB)...", end=" ", flush=True)
        
        t0 = time.time()
        shard_state = load_file(shard_path, device="cpu")
        load_time = time.time() - t0
        print(f"{load_time:.1f}s, {len(shard_state)} tensors", flush=True)
        
        # 立即加载到模型（部分加载）
        print(f"      Loading into model...", end=" ", flush=True)
        t0 = time.time()
        # 只加载这个分片的权重
        model.load_state_dict(shard_state, strict=False, assign=True)
        load_model_time = time.time() - t0
        print(f"{load_model_time:.1f}s", flush=True)
        
        total_loaded += len(shard_state)
        del shard_state
        gc.collect()
        
        avail = psutil.virtual_memory().available / 1e9
        print(f"      RAM available: {avail:.1f}GB, total loaded: {total_loaded} tensors", flush=True)
    
    # 5. 修复meta tensors
    print("[5] Fixing meta tensors...", flush=True)
    fixed = fix_meta_tensors(model)
    print(f"    Fixed {len(fixed)} meta tensors", flush=True)

    # 检查
    meta_count = sum(1 for p in model.parameters() if p.is_meta)
    meta_count += sum(1 for b in model.buffers() if b.is_meta)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Meta tensors remaining: {meta_count}, params: {total_params/1e9:.2f}B", flush=True)

    model.eval()
    return model, tokenizer, "cpu"


def test_inference(model, tokenizer):
    """测试推理"""
    print("\n[Test] CPU forward pass...", flush=True)
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    t_fwd = time.time() - t0
    hs = out.hidden_states
    print(f"    Forward: {t_fwd:.1f}s (CPU), {len(hs)} layers, last norm={hs[-1].float().norm():.2f}", flush=True)

    logits = out.logits[0, -1].float().numpy()
    top5_ids = np.argsort(logits)[-5:][::-1]
    top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
    print(f"    Top-5: {top5}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=15, do_sample=False)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t_gen = time.time() - t0
    print(f"    Generate ({t_gen:.1f}s): '{gen_text}'", flush=True)

    return top5


if __name__ == "__main__":
    t0 = time.time()
    model, tokenizer, device = load_ds7b_incremental()
    t_load = time.time() - t0
    print(f"\nTotal load time: {t_load:.1f}s", flush=True)

    result = test_inference(model, tokenizer)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n=== Done! Top-5: {result} ===", flush=True)
