"""
多模型加载与测试 — BF16优先版
=============================

用法:
  python tests/model_demo_bf16.py qwen3       # 只测Qwen3
  python tests/model_demo_bf16.py deepseek7b   # 只测DeepSeek7B (BF16+CPU/GPU混合)
  python tests/model_demo_bf16.py glm4         # 只测GLM4 (BF16+CPU/GPU混合)
  python tests/model_demo_bf16.py all          # 顺序测试所有模型(逐个加载,避免OOM)

=== BF16 vs 8bit 加载策略 ===
| 模型       | BF16大小 | 8bit大小 | 加载方式(12GB GPU)              |
|-----------|---------|---------|-------------------------------|
| qwen3     | ~8 GB   | N/A     | bfloat16 全GPU (fit)          |
| deepseek7b| ~14 GB  | ~8.7 GB | bfloat16 + device_map="auto"  |
| glm4      | ~18 GB  | ~10 GB  | bfloat16 + device_map="auto"  |

=== device_map="auto" 原理 ===
accelerate自动将模型层分配到GPU/CPU:
- GPU优先: 尽可能多的层放GPU (直到显存接近上限)
- CPU兜底: 剩余层放CPU内存
- 前向传播: 自动处理跨设备数据搬运, Hook完全兼容
- GPU占用: DS7B约9GB, GLM4约9-10GB, 剩余层在CPU

=== BF16对机制解释研究的意义 ===
1. 激活值无损: 8bit权重dequantize有离散化误差, BF16反映真实计算
2. Hook兼容: device_map="auto"下register_forward_hook完全正常
3. 速度: BF16混合 ≈ 1.4 tok/s, 8bit ≈ 1.0 tok/s (实测DS7B)
4. 结论增强: 若BF16和8bit下结构信号一致, 说明信号鲁棒非量化噪声

=== 已验证的测试结果 (2026-05-14) ===
| 模型       | 加载方式          | GPU内存  | 推理速度      | Hook | 状态 |
|-----------|-----------------|---------|-------------|------|------|
| qwen3     | bfloat16全GPU   | ~8GB    | ~3 tok/s    | OK   | OK   |
| deepseek7b| bf16+auto混合   | ~9GB    | ~1.4 tok/s  | OK   | OK   |
| glm4      | bf16+auto混合   | ~9-10GB | ~0.8 tok/s(估) | OK | 待验证 |
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import numpy as np
import torch
from model_utils import (get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)


def load_model_bf16(model_name: str):
    """
    BF16优先加载模型 — 所有模型均用bfloat16 + device_map="auto"
    
    与8bit版本的区别:
    - 不使用BitsAndBytesConfig量化
    - torch_dtype=torch.bfloat16 保持原始精度
    - device_map="auto" 自动GPU/CPU分配 (accelerate)
    - 激活值完全无损, Hook兼容
    
    Returns:
        (model, tokenizer, device)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name} (bfloat16 + device_map=auto)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # BF16 + device_map="auto": GPU放不下时自动offload到CPU
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    
    # 显示层分配情况
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        print(f"[bf16] {model_name} loaded: GPU={gpu_count} components, CPU={cpu_count} components, "
              f"class={type(model).__name__}, GPU mem={gpu_mem:.2f}GB")
    else:
        print(f"[bf16] {model_name} loaded: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    
    return model, tokenizer, device


def get_device_for_input(model) -> torch.device:
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(model_name: str):
    """完整测试一个模型的加载、前向推理、隐藏状态提取、权重访问"""
    print(f"\n{'='*60}")
    print(f"测试模型 (BF16): {model_name}")
    print(f"{'='*60}")

    # ---- 0. GPU信息 ----
    if torch.cuda.is_available():
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[0] GPU总VRAM: {gpu_total:.1f}GB")

    # ---- 1. 加载 (BF16) ----
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    print(f"[1] 加载耗时: {t_load:.1f}s, device={device}")

    # ---- 2. 基本信息 ----
    info = get_model_info(model, model_name)
    print(f"[2] class={info.model_class}, n_layers={info.n_layers}, "
          f"d_model={info.d_model}, vocab={info.vocab_size}, mlp_type={info.mlp_type}")

    # ---- 3. 层分配详情 ----
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        # 统计每层在哪个设备
        layer_devices = {}
        for k, v in dmap.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in layer_devices:
                    layer_devices[lid] = str(v)
        gpu_layers = sum(1 for v in layer_devices.values() if 'cuda' in v)
        cpu_layers = sum(1 for v in layer_devices.values() if 'cpu' in v)
        print(f"[3] 层分配: {gpu_layers}层GPU + {cpu_layers}层CPU (共{info.n_layers}层)")
        # 显示前3层和后3层的设备
        for lid in sorted(layer_devices.keys(), key=int)[:3]:
            print(f"    Layer {lid}: {layer_devices[lid]}")
        if len(layer_devices) > 6:
            print(f"    ...")
        for lid in sorted(layer_devices.keys(), key=int)[-3:]:
            print(f"    Layer {lid}: {layer_devices[lid]}")

    # ---- 4. 前向推理 ----
    input_device = get_device_for_input(model)
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    t_fwd = time.time() - t0
    hs = out.hidden_states
    logits = out.logits[0, -1].float().cpu().numpy()
    print(f"[4] 前向推理 ({t_fwd:.2f}s): {len(hs)}层隐藏状态, "
          f"最后层norm={hs[-1].float().norm():.2f}")

    # ---- 5. Top-5 预测 ----
    top5_ids = np.argsort(logits)[-5:][::-1]
    top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
    print(f"[5] Top-5: {top5}")

    # ---- 6. 生成 ----
    t0 = time.time()
    gen_kwargs = dict(
        max_new_tokens=20,
        do_sample=False,
        repetition_penalty=1.2,
    )
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask,
                                  **gen_kwargs)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t_gen = time.time() - t0
    print(f"[6] 生成 ({t_gen:.1f}s): '{gen_text}'")

    # ---- 7. W_U 权重 ----
    t0_wu = time.time()
    W_U = get_W_U(model, model_name)
    t_wu = time.time() - t0_wu
    print(f"[7] W_U: shape={W_U.shape}, dtype={W_U.dtype}, norm={np.linalg.norm(W_U):.2f} ({t_wu:.1f}s)")

    # ---- 8. Hook 提取中间层输出 ----
    captured = {}
    layers = get_layers(model)

    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook

    hook_layers = [0, info.n_layers // 2, info.n_layers - 1]
    hooks = [layers[li].register_forward_hook(make_hook(f"L{li}")) for li in hook_layers]

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()

    for key in sorted(captured.keys()):
        t = captured[key]
        print(f"    Hook {key}: shape={t.shape}, norm={t.float().norm():.2f}, "
              f"mean={t.float().mean():.4f}, std={t.float().std():.4f}")

    # ---- 9. 层权重提取测试 (BF16权重直接转fp32, 无dequantize) ----
    layer0 = layers[0]
    sa = layer0.self_attn
    W_q = sa.q_proj.weight.detach().cpu().float().numpy()
    print(f"[9] L0 W_q: shape={W_q.shape}, dtype={W_q.dtype}, norm={np.linalg.norm(W_q):.2f}")

    # ---- 10. 激活值精度验证 (BF16 vs 8bit关键差异) ----
    # 提取所有层的residual stream, 检查统计分布
    all_activations = {}
    def make_full_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                all_activations[f'L{layer_idx}'] = {
                    'mean': float(input[0].float().mean()),
                    'std': float(input[0].float().std()),
                    'norm': float(input[0].float().norm()),
                }
        return hook

    hooks2 = [layers[i].register_forward_hook(make_full_hook(i)) for i in range(info.n_layers)]
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks2:
        h.remove()

    print(f"[10] Residual stream统计 (BF16无损):")
    for key in sorted(all_activations.keys(), key=lambda x: int(x[1:])):
        a = all_activations[key]
        print(f"    {key}: mean={a['mean']:.4f}, std={a['std']:.4f}, norm={a['norm']:.2f}")

    # ---- 11. GPU 内存 & 释放 ----
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    gpu_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[11] GPU内存: allocated={gpu_mem:.2f}GB, reserved={gpu_reserved:.2f}GB")

    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    gpu_after = torch.cuda.memory_allocated() / 1e9
    print(f"[12] 释放后 GPU: {gpu_after:.2f} GB")

    return {
        "model": model_name,
        "load_mode": "bfloat16+auto",
        "load_time": round(t_load, 1),
        "fwd_time": round(t_fwd, 2),
        "gen_time": round(t_gen, 1),
        "gpu_mem_gb": round(gpu_mem, 2),
        "top5": top5,
        "info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "vocab_size": info.vocab_size,
        }
    }


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        results = {}
        for name in ["qwen3", "deepseek7b", "glm4"]:
            try:
                r = test_model(name)
                results[name] = r
            except Exception as e:
                print(f"!!! {name} 测试失败: {e}")
                import traceback; traceback.print_exc()
                results[name] = {"error": str(e)}

            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

        print(f"\n{'='*60}")
        print("汇总 (BF16版本)")
        print(f"{'='*60}")
        for name, r in results.items():
            if "error" in r:
                print(f"  {name}: FAILED - {r['error']}")
            else:
                print(f"  {name} ({r['load_mode']}): load={r['load_time']}s, "
                      f"fwd={r['fwd_time']}s, gen={r['gen_time']}s, "
                      f"gpu={r['gpu_mem_gb']}GB, top={r['top5'][0]}")
    else:
        test_model(model_name)

    print("\nBF16 Demo 完成!")


if __name__ == "__main__":
    main()
