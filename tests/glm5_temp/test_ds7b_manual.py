"""
DS7B 加载 - 纯手动方式，完全绕过 from_pretrained
Meta device + safetensors.load_file + load_state_dict(assign=True)
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
    """修复meta device上残留的buffer（如inv_freq）"""
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


def load_ds7b_manual():
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

    # 4. 逐分片加载权重到模型
    print("[4] Loading weights...", flush=True)
    t0 = time.time()
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    print(f"    {len(shard_files)} shards", flush=True)
    
    total_loaded = 0
    for shard_file in shard_files:
        shard_path = os.path.join(MODEL_PATH, shard_file)
        sz = os.path.getsize(shard_path) / 1e9
        print(f"    {shard_file} ({sz:.2f}GB)...", end=" ", flush=True)
        
        t1 = time.time()
        shard_state = load_file(shard_path, device="cpu")
        t_load = time.time() - t1
        print(f"load={t_load:.1f}s,", end=" ", flush=True)
        
        # 立即加载到模型
        t1 = time.time()
        missing, unexpected = model.load_state_dict(shard_state, strict=False, assign=True)
        t_assign = time.time() - t1
        total_loaded += len(shard_state)
        print(f"assign={t_assign:.1f}s, total={total_loaded} tensors", flush=True)
        
        del shard_state
        gc.collect()
    
    print(f"    All weights loaded in {time.time()-t0:.1f}s", flush=True)

    # 5. 修复meta tensors
    print("[5] Fixing meta tensors...", flush=True)
    fixed = fix_meta_tensors(model)
    print(f"    Fixed {len(fixed)}: {fixed[:3]}", flush=True)

    # 检查
    meta_params = sum(1 for p in model.parameters() if p.is_meta)
    meta_bufs = sum(1 for b in model.buffers() if b.is_meta)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Meta remaining: {meta_params} params, {meta_bufs} bufs", flush=True)
    print(f"    Total params: {total_params/1e9:.2f}B", flush=True)

    model.eval()
    return model, tokenizer


def test_inference(model, tokenizer):
    """CPU推理测试"""
    print("\n[Test] Forward pass...", flush=True)
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    t_fwd = time.time() - t0
    hs = out.hidden_states
    print(f"    Forward: {t_fwd:.1f}s, {len(hs)} layers, last norm={hs[-1].float().norm():.2f}", flush=True)

    logits = out.logits[0, -1].float().numpy()
    top5_ids = np.argsort(logits)[-5:][::-1]
    top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
    print(f"    Top-5: {top5}", flush=True)

    # 生成
    t0 = time.time()
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=15, do_sample=False)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t_gen = time.time() - t0
    print(f"    Generate ({t_gen:.1f}s): '{gen_text}'", flush=True)

    return {"fwd_time": t_fwd, "gen_time": t_gen, "top5": top5, "n_layers": len(hs)}


def test_gpu(model, tokenizer):
    """测试部分层移到GPU"""
    print("\n[Test2] Partial GPU...", flush=True)
    
    layers = model.model.layers
    n_layers = len(layers)
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    layer_size = sum(p.numel() * p.element_size() for p in layers[0].parameters())
    max_gpu = int((gpu_total * 0.70) / layer_size)
    print(f"    Layer: {layer_size/1e6:.0f}MB, max GPU: {max_gpu}/{n_layers}", flush=True)
    
    # 移动embedding + 前N层 + norm + lm_head 到GPU
    model.model.embed_tokens = model.model.embed_tokens.to("cuda")
    for i in range(min(max_gpu, n_layers)):
        layers[i] = layers[i].to("cuda")
    model.model.norm = model.model.norm.to("cuda")
    model.lm_head = model.lm_head.to("cuda")
    
    gpu_alloc = torch.cuda.memory_allocated() / 1e9
    print(f"    GPU: {gpu_alloc:.2f}GB, {max_gpu} layers", flush=True)
    
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    try:
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        t_fwd = time.time() - t0
        hs = out.hidden_states
        print(f"    Forward: {t_fwd:.2f}s (mixed), {len(hs)} layers", flush=True)
        
        logits = out.logits[0, -1].float().cpu().numpy()
        top5_ids = np.argsort(logits)[-5:][::-1]
        top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
        print(f"    Top-5: {top5}", flush=True)
        return True, top5
    except Exception as e:
        print(f"    Failed: {e}", flush=True)
        return False, str(e)


if __name__ == "__main__":
    t0 = time.time()
    model, tokenizer = load_ds7b_manual()
    t_load = time.time() - t0
    print(f"\nLoad time: {t_load:.1f}s", flush=True)

    # CPU测试
    cpu_result = test_inference(model, tokenizer)

    # GPU测试
    gpu_ok, gpu_result = test_gpu(model, tokenizer)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n=== Summary ===", flush=True)
    print(f"CPU: {cpu_result}", flush=True)
    print(f"GPU: ok={gpu_ok}, {gpu_result}", flush=True)
    print("Done!", flush=True)
