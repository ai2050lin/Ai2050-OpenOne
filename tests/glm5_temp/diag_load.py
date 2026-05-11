"""诊断DS7B/GLM4加载问题"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import torch
import psutil

def diag_load(model_name):
    from model_utils import MODEL_CONFIGS
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np

    cfg = MODEL_CONFIGS[model_name]
    path = cfg["path"]

    print(f"\n{'='*60}")
    print(f"诊断加载: {model_name}")
    print(f"路径: {path}")
    print(f"{'='*60}")

    # 检查路径
    import os
    if not os.path.exists(path):
        print(f"!!! 路径不存在: {path}")
        return
    files = os.listdir(path)
    print(f"目录文件数: {len(files)}")
    print(f"safetensors文件: {[f for f in files if 'safetensors' in f]}")

    # 系统内存
    m = psutil.virtual_memory()
    print(f"系统RAM: {m.total/1e9:.1f}GB, 可用: {m.available/1e9:.1f}GB ({m.percent}%)")
    print(f"GPU: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # Step 1: 加载tokenizer
    print(f"\n--- Step 1: 加载tokenizer ---")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    print(f"Tokenizer加载: {time.time()-t0:.1f}s")

    # Step 2: 加载模型到CPU
    print(f"\n--- Step 2: 加载模型到CPU ---")
    m = psutil.virtual_memory()
    print(f"加载前RAM: 可用{m.available/1e9:.1f}GB")
    t0 = time.time()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        t_load = time.time() - t0
        print(f"CPU加载成功: {t_load:.1f}s")
    except Exception as e:
        print(f"CPU加载失败: {e}")
        import traceback; traceback.print_exc()
        return

    m = psutil.virtual_memory()
    print(f"加载后RAM: 可用{m.available/1e9:.1f}GB")

    # 检查模型大小
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"模型参数: {param_count:.2f}B, bfloat16大小: {model_size_gb:.2f}GB")

    # Step 3: 移到CUDA
    print(f"\n--- Step 3: 移到CUDA ---")
    gpu_before = torch.cuda.memory_allocated() / 1e9
    print(f"移前GPU: {gpu_before:.2f}GB")

    t0 = time.time()
    try:
        model = model.to("cuda")
        t_move = time.time() - t0
        gpu_after = torch.cuda.memory_allocated() / 1e9
        print(f"移到CUDA成功: {t_move:.1f}s, GPU: {gpu_after:.2f}GB")
    except Exception as e:
        print(f"移到CUDA失败: {e}")
        import traceback; traceback.print_exc()
        # 释放
        del model; gc.collect(); torch.cuda.empty_cache()
        return

    # Step 4: 前向推理
    print(f"\n--- Step 4: 简短前向推理 ---")
    model.eval()
    inputs = tokenizer("Hello", return_tensors="pt", truncation=True, max_length=16)
    input_ids = inputs["input_ids"].to("cuda:0")
    attention_mask = inputs["attention_mask"].to("cuda:0")

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    t_fwd = time.time() - t0
    print(f"前向推理成功: {t_fwd:.2f}s, 层数: {len(out.hidden_states)}")

    # Step 5: 释放
    print(f"\n--- Step 5: 释放 ---")
    del model; gc.collect(); torch.cuda.empty_cache()
    gpu_after_release = torch.cuda.memory_allocated() / 1e9
    print(f"释放后GPU: {gpu_after_release:.2f}GB")

    print(f"\n诊断完成: {model_name} 一切正常!")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    diag_load(model_name)
