"""
绕过 from_pretrained 卡死 - DS7B完整加载测试
修复：meta device残留的buffer需要手动初始化
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
    """修复meta device上残留的tensor（如inv_freq等buffer）"""
    fixed = []
    for name, module in model.named_modules():
        # 修复register_buffer创建的meta tensor
        for buf_name, buf in list(module.named_buffers(recurse=False)):
            if buf.is_meta:
                # inv_freq: RoPE的频率向量，需要重新计算
                if 'inv_freq' in buf_name:
                    dim = buf.shape[-1]
                    base = getattr(module, 'base', 10000.0)
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim * 2, 2, dtype=torch.float32) / (dim * 2)))
                    new_buf = inv_freq.to(buf.dtype)
                    # 直接设置属性，不用register_buffer避免名字冲突
                    setattr(module, buf_name, new_buf)
                    fixed.append(f"{name}.{buf_name}")
                else:
                    # 其他meta buffer：尝试zero初始化
                    new_buf = torch.zeros(buf.shape, dtype=buf.dtype)
                    setattr(module, buf_name, new_buf)
                    fixed.append(f"{name}.{buf_name}")
    
    # 修复parameter中的meta tensor
    for name, param in model.named_parameters():
        if param.is_meta:
            print(f"  WARNING: parameter still meta: {name}")
    
    return fixed


def load_ds7b_bypass():
    """手动加载DS7B"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # 1. Tokenizer
    print("[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"    OK, vocab={len(tokenizer)}")

    # 2. Config
    print("[2] Loading config...")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    print(f"    {config.model_type}, {config.num_hidden_layers} layers, d={config.hidden_size}")

    # 3. Meta device空模型
    print("[3] Creating meta model...")
    t0 = time.time()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to(torch.bfloat16)
    print(f"    Created in {time.time()-t0:.1f}s")

    # 4. safetensors加载权重
    print("[4] Loading weights from safetensors...")
    t0 = time.time()
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"    {len(shard_files)} shards: {shard_files}")

    state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(MODEL_PATH, shard_file)
        sz = os.path.getsize(shard_path) / 1e9
        print(f"    {shard_file} ({sz:.2f}GB)...", end=" ", flush=True)
        shard_state = load_file(shard_path, device="cpu")
        for k, v in shard_state.items():
            state_dict[k] = v.to(torch.bfloat16)
        print(f"OK, {len(shard_state)} tensors")
    print(f"    Loaded in {time.time()-t0:.1f}s, total={len(state_dict)} tensors")

    # 5. 加载state_dict
    print("[5] Loading state dict (assign=True)...")
    t0 = time.time()
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"    Loaded in {time.time()-t0:.1f}s")
    if missing:
        print(f"    Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"    Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    del state_dict
    gc.collect()

    # 6. 修复meta tensors
    print("[6] Fixing meta tensors...")
    fixed = fix_meta_tensors(model)
    print(f"    Fixed {len(fixed)} meta tensors: {fixed[:3]}...")

    # 7. 确认所有tensor都在CPU上
    meta_count = 0
    for name, param in model.named_parameters():
        if param.is_meta:
            meta_count += 1
    for name, buf in model.named_buffers():
        if buf.is_meta:
            meta_count += 1
            print(f"    STILL META: {name}")
    
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    dev = next(model.parameters()).device
    print(f"    Device: {dev}, params={total_params/1e9:.2f}B, size={total_size/1e9:.2f}GB, meta={meta_count}")

    model.eval()
    return model, tokenizer


def test_cpu_inference(model, tokenizer):
    """CPU推理测试"""
    print("\n[Test] CPU forward pass...")
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    t_fwd = time.time() - t0
    hs = out.hidden_states
    print(f"    Forward: {t_fwd:.1f}s (CPU), {len(hs)} layers, last norm={hs[-1].float().norm():.2f}")

    logits = out.logits[0, -1].float().numpy()
    top5_ids = np.argsort(logits)[-5:][::-1]
    top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
    print(f"    Top-5: {top5}")

    # 生成
    t0 = time.time()
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=15, do_sample=False)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t_gen = time.time() - t0
    print(f"    Generate ({t_gen:.1f}s): '{gen_text}'")

    return top5


def test_gpu_partial(model, tokenizer):
    """将模型部分移到GPU进行推理"""
    print("\n[Test2] Moving model to GPU (partial)...")
    
    # 方法：逐层移到GPU，直到GPU接近满
    layers = model.model.layers
    n_layers = len(layers)
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    
    # 每层大小
    layer_size = sum(p.numel() * p.element_size() for p in layers[0].parameters())
    max_gpu_layers = int((gpu_total * 0.75) / layer_size)
    print(f"    Layer size: {layer_size/1e6:.0f}MB, max GPU layers: {max_gpu_layers}/{n_layers}")
    
    # 移动embedding和前N层到GPU
    model.model.embed_tokens = model.model.embed_tokens.to("cuda")
    for i in range(min(max_gpu_layers, n_layers)):
        layers[i] = layers[i].to("cuda")
    
    # 移动最后norm和lm_head
    model.model.norm = model.model.norm.to("cuda")
    model.lm_head = model.lm_head.to("cuda")
    
    gpu_alloc = torch.cuda.memory_allocated() / 1e9
    print(f"    GPU allocated: {gpu_alloc:.2f} GB, {max_gpu_layers} layers on GPU")
    
    # 推理测试
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
        print(f"    Forward: {t_fwd:.2f}s (mixed), {len(hs)} layers, last norm={hs[-1].float().norm():.2f}")
        
        logits = out.logits[0, -1].float().cpu().numpy()
        top5_ids = np.argsort(logits)[-5:][::-1]
        top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
        print(f"    Top-5: {top5}")
        return True, top5
    except Exception as e:
        print(f"    Failed: {e}")
        return False, None


if __name__ == "__main__":
    t0 = time.time()
    model, tokenizer = load_ds7b_bypass()
    t_load = time.time() - t0
    print(f"\nTotal CPU load time: {t_load:.1f}s")

    # Test 1: CPU推理
    cpu_top5 = test_cpu_inference(model, tokenizer)

    # Test 2: 部分GPU
    success, gpu_top5 = test_gpu_partial(model, tokenizer)

    # 清理
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n=== Summary ===")
    print(f"CPU Top-5: {cpu_top5}")
    if success:
        print(f"GPU Top-5: {gpu_top5}")
    print("Done!")
