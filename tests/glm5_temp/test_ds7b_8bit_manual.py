"""
DS7B 8bit加载 - 手动方式绕过from_pretrained卡死
步骤:
1. meta device创建模型
2. safetensors逐分片加载权重到CPU
3. 修复meta buffer
4. 8bit量化后移到GPU
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


def load_shard_incremental(shard_path, keys_to_load=None):
    """逐tensor加载safetensors分片，立即clone释放mmap"""
    state_dict = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        if keys_to_load:
            all_keys = [k for k in all_keys if k in keys_to_load]
        for i, key in enumerate(all_keys):
            tensor = f.get_tensor(key)
            state_dict[key] = tensor.clone()
            if (i + 1) % 20 == 0:
                print(f"        {i+1}/{len(all_keys)}", end="", flush=True)
                # 每加载20个tensor就assign到模型，释放内存
                if len(state_dict) >= 20:
                    yield state_dict
                    state_dict = {}
        if state_dict:
            yield state_dict
    print()


def load_ds7b_8bit():
    """手动加载DS7B并8bit量化"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import bitsandbytes as bnb

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
    # 直接用float16创建（8bit量化需要float输入）
    model = model.to(torch.float16)
    print(f"    Created in {time.time()-t0:.1f}s", flush=True)

    # 4. 逐分片+逐batch加载权重
    print("[4] Loading weights incrementally...", flush=True)
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
        for batch_state in load_shard_incremental(shard_path):
            # 加载到模型
            model.load_state_dict(batch_state, strict=False, assign=True)
            total_loaded += len(batch_state)
            del batch_state
            gc.collect()
        
        print(f"      {time.time()-t1:.1f}s, total={total_loaded}", flush=True)
    
    print(f"    All weights: {time.time()-t0:.1f}s, total={total_loaded} tensors", flush=True)

    # 5. 修复meta tensors
    print("[5] Fixing meta tensors...", flush=True)
    fixed = fix_meta_tensors(model)
    print(f"    Fixed {len(fixed)}", flush=True)

    # 检查
    meta_count = sum(1 for p in model.parameters() if p.is_meta)
    meta_count += sum(1 for b in model.buffers() if b.is_meta)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Meta remaining: {meta_count}, params: {total_params/1e9:.2f}B", flush=True)

    # 6. 8bit量化并移到GPU
    print("[6] 8bit quantization & GPU dispatch...", flush=True)
    t0 = time.time()
    
    # 手动8bit量化关键大层
    from accelerate import infer_auto_device_map, dispatch_model
    import psutil
    
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    print(f"    GPU: {gpu_total/1e9:.1f}GB", flush=True)
    
    # 8bit量化MLP层（最大的权重）
    quantized_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            in_f = module.in_features
            out_f = module.out_features
            # 量化大矩阵（MLP和attention的qkv/o）
            if in_f * out_f > 1024 * 1024:  # >1M params
                try:
                    # 用bnb量化
                    weight = module.weight.data
                    if not weight.is_meta and weight.device.type == 'cpu':
                        # 替换为Int8Linear
                        int8_linear = bnb.nn.Int8Linear(
                            in_f, out_f, bias=module.bias is not None,
                            has_fp16_weights=False  # 纯8bit
                        )
                        # 复制权重
                        int8_linear.weight = bnb.nn.Int8Params(
                            weight.data, requires_grad=False, has_fp16_weights=False
                        )
                        if module.bias is not None:
                            int8_linear.bias = module.bias
                        # 替换
                        parts = name.split('.')
                        parent = model
                        for p in parts[:-1]:
                            parent = getattr(parent, p)
                        setattr(parent, parts[-1], int8_linear)
                        quantized_count += 1
                        del weight
                except Exception as e:
                    print(f"      Quantize {name} failed: {e}", flush=True)
    
    print(f"    Quantized {quantized_count} linear layers", flush=True)
    print(f"    Quantization took {time.time()-t0:.1f}s", flush=True)
    
    # 7. 分配到GPU/CPU
    print("[7] Dispatching to GPU/CPU...", flush=True)
    t0 = time.time()
    
    max_gpu = int(gpu_total * 0.80)
    try:
        device_map = infer_auto_device_map(
            model,
            max_memory={0: max_gpu, "cpu": "24GiB"},
            no_split_module_classes=["Qwen2DecoderLayer"]
        )
        gpu_mods = sum(1 for v in device_map.values() if v == 0 or v == 'cuda:0')
        cpu_mods = sum(1 for v in device_map.values() if v == 'cpu')
        print(f"    {gpu_mods} on GPU, {cpu_mods} on CPU", flush=True)
        
        model = dispatch_model(model, device_map=device_map)
    except Exception as e:
        print(f"    Dispatch failed: {e}", flush=True)
        print(f"    Trying direct .to('cuda')...", flush=True)
        # 回退：不dispatch，直接逐层移动
        try:
            model = model.to("cuda")
        except Exception as e2:
            print(f"    .to('cuda') also failed: {e2}", flush=True)
            print(f"    Will run on CPU", flush=True)
    
    model.eval()
    device = next(model.parameters()).device
    gpu_alloc = torch.cuda.memory_allocated() / 1e9
    print(f"    Device: {device}, GPU: {gpu_alloc:.2f}GB, dispatch: {time.time()-t0:.1f}s", flush=True)

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

    return {"fwd_time": round(t_fwd, 2), "top5": top5, "n_layers": len(hs)}


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
    print("Done!", flush=True)
