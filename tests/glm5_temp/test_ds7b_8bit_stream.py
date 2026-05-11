"""
DS7B 8bit加载 - 逐tensor流式加载，避免内存峰值
关键改进：每加载一个tensor就立即set到模型参数，不累积dict
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


def set_param_by_name(model, name, tensor):
    """通过名字设置模型参数"""
    parts = name.split('.')
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    # 如果是parameter，直接设置data
    attr = getattr(obj, parts[-1])
    if isinstance(attr, torch.nn.Parameter):
        attr.data = tensor
    else:
        setattr(obj, parts[-1], tensor)


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


def load_ds7b_stream():
    """流式加载DS7B，逐tensor直接设置到模型"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import psutil

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

    # 3. 先创建CPU上的空模型（不用meta device）
    print("[3] Creating model on CPU (empty)...", flush=True)
    t0 = time.time()
    # 用meta device创建，然后materialize到CPU
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # 将所有meta tensor替换为空tensor（在CPU上）
    for name, param in list(model.named_parameters()):
        if param.is_meta:
            new_param = torch.nn.Parameter(
                torch.empty(param.shape, dtype=torch.bfloat16, device='cpu'),
                requires_grad=False
            )
            set_param_by_name(model, name, new_param)
    # 修复buffers
    fixed = fix_meta_tensors(model)
    print(f"    Created in {time.time()-t0:.1f}s, fixed {len(fixed)} buffers", flush=True)

    # 4. 逐tensor流式加载权重
    print("[4] Streaming weights...", flush=True)
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
        count = 0
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                tensor = f.get_tensor(key)
                # 直接设置到模型参数
                set_param_by_name(model, key, tensor.to(torch.bfloat16))
                del tensor
                count += 1
                if count % 50 == 0:
                    print(f"      {count}/{len(keys)}", end=" ", flush=True)
        
        total_loaded += count
        avail = psutil.virtual_memory().available / 1e9
        print(f"\n      Done: {count} tensors, {time.time()-t1:.1f}s, RAM: {avail:.1f}GB", flush=True)
    
    print(f"    All weights: {time.time()-t0:.1f}s, total={total_loaded}", flush=True)

    # 5. 验证
    meta_count = sum(1 for p in model.parameters() if p.is_meta)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Meta: {meta_count}, params: {total_params/1e9:.2f}B", flush=True)

    model.eval()
    return model, tokenizer


def quantize_and_dispatch(model):
    """8bit量化MLP层并分配到GPU"""
    import bitsandbytes as bnb
    from accelerate import infer_auto_device_map, dispatch_model
    
    print("\n[5] 8bit quantizing MLP layers...", flush=True)
    t0 = time.time()
    
    quant_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            # 只量化MLP层（gate/up/down_proj）
            if any(k in name for k in ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
                try:
                    weight_data = module.weight.data.cpu()
                    new_linear = bnb.nn.Int8Linear(in_f, out_f, bias=module.bias is not None, has_fp16_weights=False)
                    new_linear.weight = bnb.nn.Int8Params(weight_data, requires_grad=False, has_fp16_weights=False)
                    if module.bias is not None:
                        new_linear.bias.data = module.bias.data
                    
                    parts = name.split('.')
                    parent = model
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], new_linear)
                    quant_count += 1
                except Exception as e:
                    print(f"      Fail: {name}: {e}", flush=True)
    
    print(f"    Quantized {quant_count} layers in {time.time()-t0:.1f}s", flush=True)
    
    # 6. Dispatch
    print("[6] Dispatching...", flush=True)
    t0 = time.time()
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    max_gpu = int(gpu_total * 0.80)
    
    try:
        device_map = infer_auto_device_map(
            model, max_memory={0: max_gpu, "cpu": "24GiB"},
            no_split_module_classes=["Qwen2DecoderLayer"]
        )
        gpu_mods = sum(1 for v in device_map.values() if v == 0 or v == 'cuda:0')
        cpu_mods = sum(1 for v in device_map.values() if v == 'cpu')
        print(f"    {gpu_mods} GPU, {cpu_mods} CPU", flush=True)
        model = dispatch_model(model, device_map=device_map)
    except Exception as e:
        print(f"    Dispatch failed: {e}", flush=True)
        # 回退：部分层移到GPU
        layers = model.model.layers
        n = len(layers)
        layer_size = sum(p.numel() * p.element_size() for p in layers[0].parameters())
        max_layers = int(max_gpu / layer_size * 0.8)
        print(f"    Putting {max_layers}/{n} layers on GPU", flush=True)
        model.model.embed_tokens = model.model.embed_tokens.to("cuda")
        for i in range(min(max_layers, n)):
            layers[i] = layers[i].to("cuda")
        model.model.norm = model.model.norm.to("cuda")
        model.lm_head = model.lm_head.to("cuda")
    
    model.eval()
    device = next(model.parameters()).device
    gpu_alloc = torch.cuda.memory_allocated() / 1e9
    print(f"    Device: {device}, GPU: {gpu_alloc:.2f}GB, time: {time.time()-t0:.1f}s", flush=True)
    return model, device


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

    t0 = time.time()
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20, do_sample=False)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t_gen = time.time() - t0
    print(f"    Generate ({t_gen:.1f}s): '{gen_text}'", flush=True)

    return {"fwd_time": round(t_fwd, 2), "gen_time": round(t_gen, 1), "top5": top5}


if __name__ == "__main__":
    t0 = time.time()
    model, tokenizer = load_ds7b_stream()
    t_load = time.time() - t0
    print(f"\nWeight loading time: {t_load:.1f}s", flush=True)

    model, device = quantize_and_dispatch(model)
    t_total = time.time() - t0
    print(f"Total setup time: {t_total:.1f}s", flush=True)

    result = test_inference(model, tokenizer, device)
    print(f"\nResult: {result}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Done!", flush=True)
