"""
绕过 from_pretrained 卡死问题，手动加载DS7B
策略：全部权重先在CPU上，然后用accelerate dispatch分配GPU/CPU
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

def load_ds7b_bypass():
    """手动加载DS7B，绕过from_pretrained卡死"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import infer_auto_device_map, dispatch_model

    # 1. 加载tokenizer
    print("[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"    Tokenizer OK, vocab={len(tokenizer)}")

    # 2. 加载配置
    print("[2] Loading config...")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    print(f"    Config: {config.model_type}, {config.num_hidden_layers} layers, d={config.hidden_size}")

    # 3. 在meta device上创建空模型
    print("[3] Creating model on meta device...")
    t0 = time.time()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to(torch.bfloat16)
    print(f"    Meta model created in {time.time()-t0:.1f}s")

    # 4. 用safetensors直接加载权重到CPU
    print("[4] Loading weights from safetensors...")
    t0 = time.time()
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"    Found {len(shard_files)} shards: {shard_files}")

    state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(MODEL_PATH, shard_file)
        sz = os.path.getsize(shard_path) / 1e9
        print(f"    Loading {shard_file} ({sz:.2f}GB)...", end=" ", flush=True)
        shard_state = load_file(shard_path, device="cpu")
        for k, v in shard_state.items():
            state_dict[k] = v.to(torch.bfloat16)
        print(f"OK, {len(shard_state)} tensors")
    print(f"    All weights loaded in {time.time()-t0:.1f}s, total={len(state_dict)} tensors")

    # 5. 将权重加载到meta模型 (assign=True 直接替换tensor，不复制)
    print("[5] Loading state dict into model (assign=True)...")
    t0 = time.time()
    model.load_state_dict(state_dict, strict=True, assign=True)
    print(f"    State dict loaded in {time.time()-t0:.1f}s")
    del state_dict
    gc.collect()

    # 6. 先确认所有权重都在CPU上
    print("[6] Verifying all weights on CPU...")
    for name, param in list(model.named_parameters())[:3]:
        print(f"    {name}: device={param.device}, shape={param.shape}")
    
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"    Total params: {total_params/1e9:.2f}B, size: {total_size/1e9:.2f}GB")

    # 7. 使用accelerate dispatch分配GPU/CPU
    # GPU总内存11.94GiB，模型~15GB，需要部分在CPU
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    # 保守设置：只用70%的GPU内存给模型权重
    max_gpu_mem = int(gpu_total * 0.70)
    print(f"[7] Setting up dispatch: GPU max={max_gpu_mem/1e9:.1f}GB, CPU max=24GiB")
    
    device_map = infer_auto_device_map(
        model,
        max_memory={0: max_gpu_mem, "cpu": "24GiB"},
        no_split_module_classes=["Qwen2DecoderLayer"]
    )
    
    # 统计分布
    gpu_count = sum(1 for v in device_map.values() if v == 0)
    cpu_count = sum(1 for v in device_map.values() if v == "cpu")
    print(f"    Device map: {gpu_count} modules on GPU, {cpu_count} on CPU")
    
    # 8. 逐步dispatch，避免OOM
    print("[8] Dispatching model...")
    t0 = time.time()
    model = dispatch_model(model, device_map=device_map)
    model.eval()
    print(f"    Dispatched in {time.time()-t0:.1f}s")

    device = next(model.parameters()).device
    gpu_alloc = torch.cuda.memory_allocated() / 1e9
    print(f"    Primary device: {device}, GPU allocated: {gpu_alloc:.2f} GB")

    return model, tokenizer, device


def test_forward(model, tokenizer, device):
    """测试前向推理"""
    print("\n[Test] Forward pass...")
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 将输入移到模型第一个设备
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)
    attention_mask = attention_mask.to(first_device)

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    t_fwd = time.time() - t0
    hs = out.hidden_states
    print(f"    Forward: {t_fwd:.2f}s, {len(hs)} layers, last norm={hs[-1].float().norm():.2f}")

    # Top-5预测
    logits = out.logits[0, -1].float().cpu().numpy()
    top5_ids = np.argsort(logits)[-5:][::-1]
    top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
    print(f"    Top-5: {top5}")

    # 生成
    t0 = time.time()
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20, do_sample=False)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t_gen = time.time() - t0
    print(f"    Generate ({t_gen:.1f}s): '{gen_text}'")

    return {
        "fwd_time": round(t_fwd, 2),
        "gen_time": round(t_gen, 1),
        "n_layers": len(hs),
        "top5": top5,
        "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    }


if __name__ == "__main__":
    t0 = time.time()
    model, tokenizer, device = load_ds7b_bypass()
    t_load = time.time() - t0
    print(f"\nTotal load time: {t_load:.1f}s")

    result = test_forward(model, tokenizer, device)
    print(f"\nResult: {result}")

    # 释放
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU after release: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("Done!")
