"""
DS7B 8bit加载 - safe_open逐tensor + load_state_dict(assign=True)
完全绕过 from_pretrained 的卡死问题
然后用8bit方式量化关键层，放入GPU
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
                    setattr(module, buf_name, inv_freq.to(buf.dtype))
                    fixed.append(f"{name}.{buf_name}")
                else:
                    setattr(module, buf_name, torch.zeros(buf.shape, dtype=buf.dtype))
                    fixed.append(f"{name}.{buf_name}")
    return fixed


def load_shard_tensors(shard_path):
    """用safe_open逐tensor加载，clone确保真实副本"""
    state_dict = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for i, key in enumerate(keys):
            tensor = f.get_tensor(key)
            state_dict[key] = tensor.clone()  # clone脱离mmap
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(keys)}", end=" ", flush=True)
    print()
    return state_dict


def load_ds7b_8bit():
    """手动加载DS7B，8bit量化"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    print("[4] Loading weights (safe_open)...", flush=True)
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
        print(f"    {shard_file} ({sz:.2f}GB):", flush=True)
        
        t1 = time.time()
        # 加载分片权重
        shard_state = load_shard_tensors(shard_path)
        t_load = time.time() - t1
        print(f"      Load: {t_load:.1f}s, {len(shard_state)} tensors", flush=True)
        
        # assign到模型
        t1 = time.time()
        model.load_state_dict(shard_state, strict=False, assign=True)
        t_assign = time.time() - t1
        total_loaded += len(shard_state)
        print(f"      Assign: {t_assign:.1f}s, total={total_loaded}", flush=True)
        
        del shard_state
        gc.collect()
    
    print(f"    All weights: {time.time()-t0:.1f}s", flush=True)

    # 5. 修复meta tensors
    print("[5] Fixing meta tensors...", flush=True)
    fixed = fix_meta_tensors(model)
    print(f"    Fixed {len(fixed)}", flush=True)

    # 检查
    meta_count = sum(1 for p in model.parameters() if p.is_meta)
    meta_count += sum(1 for b in model.buffers() if b.is_meta)
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"    Meta: {meta_count}, params: {total_params/1e9:.2f}B, size: {total_size/1e9:.2f}GB", flush=True)

    # 6. 使用from_pretrained的8bit重新加载方式
    # 直接将CPU模型保存到临时目录，然后用8bit from_pretrained加载
    # 但这也会卡死...所以改为：手动将关键层量化并移到GPU
    
    # 简化方案：直接用model.to("cuda")但需要GPU能放下
    # DS7B bf16 = 15GB > 12GB GPU
    # 改用float16量化MLP层权重来节省内存
    
    print("[6] Quantizing MLP layers to int8...", flush=True)
    t0 = time.time()
    import bitsandbytes as bnb
    
    quant_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 只量化大的Linear层
            if module.weight.shape[0] * module.weight.shape[1] > 1024 * 1024:
                weight_data = module.weight.data
                if not weight_data.is_meta and weight_data.device.type == 'cpu':
                    # 替换为Int8Linear
                    new_linear = bnb.nn.Int8Linear(
                        module.in_features, module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False
                    )
                    # 将权重转为int8
                    new_linear.weight = bnb.nn.Int8Params(
                        weight_data, requires_grad=False, has_fp16_weights=False
                    )
                    if module.bias is not None:
                        new_linear.bias.data = module.bias.data
                    
                    # 替换模块
                    parts = name.split('.')
                    parent = model
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], new_linear)
                    quant_count += 1
    
    print(f"    Quantized {quant_count} layers in {time.time()-t0:.1f}s", flush=True)
    
    # 检查量化后大小
    total_size_q = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"    Size after quant: {total_size_q/1e9:.2f}GB", flush=True)

    # 7. Dispatch到GPU/CPU
    print("[7] Dispatching...", flush=True)
    t0 = time.time()
    from accelerate import infer_auto_device_map, dispatch_model
    
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    max_gpu = int(gpu_total * 0.80)
    
    try:
        device_map = infer_auto_device_map(
            model,
            max_memory={0: max_gpu, "cpu": "24GiB"},
            no_split_module_classes=["Qwen2DecoderLayer"]
        )
        gpu_mods = sum(1 for v in device_map.values() if v == 0 or v == 'cuda:0')
        cpu_mods = sum(1 for v in device_map.values() if v == 'cpu')
        print(f"    {gpu_mods} GPU, {cpu_mods} CPU", flush=True)
        
        model = dispatch_model(model, device_map=device_map)
    except Exception as e:
        print(f"    Dispatch failed: {e}", flush=True)
        print(f"    Trying partial GPU...", flush=True)
        # 回退：逐层移到GPU
        layers = model.model.layers
        n = len(layers)
        model.model.embed_tokens = model.model.embed_tokens.to("cuda")
        for i in range(n):
            try:
                layers[i] = layers[i].to("cuda")
            except RuntimeError:
                print(f"    GPU full at layer {i}/{n}", flush=True)
                break
        model.model.norm = model.model.norm.to("cuda")
        model.lm_head = model.lm_head.to("cuda")
    
    model.eval()
    device = next(model.parameters()).device
    gpu_alloc = torch.cuda.memory_allocated() / 1e9
    print(f"    Device: {device}, GPU: {gpu_alloc:.2f}GB, time: {time.time()-t0:.1f}s", flush=True)

    return model, tokenizer, device


def test_inference(model, tokenizer, device):
    """推理测试"""
    print("\n[Test] Forward pass...", flush=True)
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    t_fwd = time.time() - t0
    hs = out.hidden_states
    print(f"    Forward: {t_fwd:.2f}s, {len(hs)} layers, last norm={hs[-1].float().norm():.2f}", flush=True)

    logits = out.logits[0, -1].float().cpu().numpy()
    top5_ids = np.argsort(logits)[-5:][::-1]
    top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
    print(f"    Top-5: {top5}", flush=True)

    # 生成
    t0 = time.time()
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20, do_sample=False)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t_gen = time.time() - t0
    print(f"    Generate ({t_gen:.1f}s): '{gen_text}'", flush=True)

    return {"fwd_time": round(t_fwd, 2), "gen_time": round(t_gen, 1), "top5": top5, "n_layers": len(hs)}


if __name__ == "__main__":
    t0 = time.time()
    model, tokenizer, device = load_ds7b_8bit()
    t_load = time.time() - t0
    print(f"\nTotal load time: {t_load:.1f}s", flush=True)

    result = test_inference(model, tokenizer, device)
    print(f"\nResult: {result}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU after release: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    print("Done!", flush=True)
