"""
多模型加载与测试标准 Demo
=========================

用法:
  python tests/model_demo.py qwen3       # 只测Qwen3
  python tests/model_demo.py deepseek7b   # 只测DeepSeek7B
  python tests/model_demo.py glm4         # 只测GLM4
  python tests/model_demo.py all          # 顺序测试所有模型(逐个加载,避免OOM)

=== 为什么DS7B/GLM4必须用8bit ===
1. 硬件限制: RTX 5070 只有12GB显存
   - DS7B bfloat16需要~15GB, 8bit只需~8.7GB
   - GLM4 bfloat16需要~18GB, 8bit只需~10GB
2. PyTorch 2.10.0+cu130 兼容性问题:
   - from_pretrained + bfloat16 对DS7B大分片(8.61GB)卡在0%
   - 8bit模式的from_pretrained进度正常(50%→100%)
   - Qwen3分片较小(~4GB each), bfloat16不受影响
3. 8bit量化通过BitsAndBytesConfig实现:
   - load_in_8bit=True: 权重从fp16/bf16量化为int8
   - llm_int8_enable_fp32_cpu_offload=True: 允许CPU offload
   - device_map="auto": 自动分配GPU/CPU, 无需手动.to("cuda")

=== 8bit对实验的影响 ===
- 隐藏状态: 8bit权重经dequantize后推理, 输出仍为fp16/bf16, 影响较小
- 权重矩阵: .weight返回Int8Params, .detach().cpu().float()可正常转为fp32
- 精度损失: 理论误差<0.1%, 对PCA/SVD等分析影响可忽略
- 生成质量: 可能出现重复模式, 调整temperature/repetition_penalty可缓解

=== 模型参数 ===
| 模型       | 类名                | 层数 | d_model | vocab  | GPU内存       | 加载方式 |
|-----------|--------------------|------|---------|--------|--------------|---------|
| qwen3     | Qwen3ForCausalLM   | 36   | 2560    | 151936 | ~8 GB        | bfloat16|
| deepseek7b| Qwen2ForCausalLM   | 28   | 3584    | 152064 | ~8.7 GB      | 8bit    |
| glm4      | GlmForCausalLM     | 40   | 4096    | 151552 | ~10 GB(预估)  | 8bit    |

=== 已验证的测试结果 (2026-05-11) ===
| 模型       | 加载时间 | 前向推理 | GPU内存 | 状态 |
|-----------|---------|---------|---------|------|
| qwen3     | ~8s     | ~0.3s   | ~8GB    | OK   |
| deepseek7b| 11.6s   | 0.54s   | 8.71GB  | OK   |
| glm4      | TBD     | TBD     | TBD     | 待测  |
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import numpy as np
import torch
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)


def get_device_for_input(model, model_name: str) -> torch.device:
    """获取输入tensor应放的设备 (8bit模型参数可能分散在GPU/CPU)"""
    # 8bit模型用device_map="auto", 第一个参数的设备就是输入设备
    try:
        first_param = next(model.parameters())
        return first_param.device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(model_name: str):
    """完整测试一个模型的加载、前向推理、隐藏状态提取、权重访问"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")

    # ---- 0. 确定加载方式 ----
    cfg = MODEL_CONFIGS[model_name]
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    load_mode = "8bit量化" if use_8bit else "bfloat16"
    print(f"[0] 加载方式: {load_mode} (GPU={gpu_mem_gb:.1f}GB)")

    # ---- 1. 加载 ----
    t0 = time.time()
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    t_load = time.time() - t0
    print(f"[1] 加载耗时: {t_load:.1f}s, device={device}")

    # ---- 2. 基本信息 ----
    info = get_model_info(model, model_name)
    print(f"[2] class={info.model_class}, n_layers={info.n_layers}, "
          f"d_model={info.d_model}, vocab={info.vocab_size}, mlp_type={info.mlp_type}")

    # ---- 3. 前向推理 ----
    # 注意: 8bit模型参数在device_map="auto"管理的设备上
    # input_ids和attention_mask需要放到模型第一个参数的设备上
    input_device = get_device_for_input(model, model_name)
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    t_fwd = time.time() - t0
    hs = out.hidden_states  # tuple of (n_layers+1,) 每个 [1, seq_len, d_model]
    logits = out.logits[0, -1].float().cpu().numpy()
    print(f"[3] 前向推理 ({t_fwd:.2f}s): {len(hs)}层隐藏状态, "
          f"最后层norm={hs[-1].float().norm():.2f}")

    # ---- 4. Top-5 预测 ----
    top5_ids = np.argsort(logits)[-5:][::-1]
    top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
    print(f"[4] Top-5: {top5}")

    # ---- 5. 生成 ----
    # 8bit模型可能产生重复输出, 增加repetition_penalty缓解
    t0 = time.time()
    gen_kwargs = dict(
        max_new_tokens=20,
        do_sample=False,
        repetition_penalty=1.2,  # 缓解8bit重复问题
    )
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask,
                                  **gen_kwargs)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t_gen = time.time() - t0
    print(f"[5] 生成 ({t_gen:.1f}s): '{gen_text}'")

    # ---- 6. W_U 权重 ----
    # 8bit模型的lm_head.weight是Int8Params, .detach().cpu().float()自动dequantize
    t0_wu = time.time()
    W_U = get_W_U(model)
    t_wu = time.time() - t0_wu
    print(f"[6] W_U: shape={W_U.shape}, norm={np.linalg.norm(W_U):.2f} ({t_wu:.1f}s)")

    # ---- 7. Hook 提取中间层输出 ----
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
        print(f"    Hook {key}: shape={t.shape}, norm={t.float().norm():.2f}")

    # ---- 8. 层权重提取测试 (验证8bit权重可正确转为fp32) ----
    layer0 = layers[0]
    sa = layer0.self_attn
    W_q = sa.q_proj.weight.detach().cpu().float().numpy()
    print(f"[8] L0 W_q: shape={W_q.shape}, dtype={W_q.dtype}, norm={np.linalg.norm(W_q):.2f}")

    # ---- 9. GPU 内存 & 释放 ----
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"[9] GPU 内存: {gpu_mem:.2f} GB")

    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    gpu_after = torch.cuda.memory_allocated() / 1e9
    print(f"[10] 释放后 GPU: {gpu_after:.2f} GB")

    return {
        "model": model_name,
        "load_mode": load_mode,
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
        # 逐个测试, 确保前一个模型完全释放后再加载下一个
        for name in ["qwen3", "deepseek7b", "glm4"]:
            try:
                r = test_model(name)
                results[name] = r
            except Exception as e:
                print(f"!!! {name} 测试失败: {e}")
                import traceback; traceback.print_exc()
                results[name] = {"error": str(e)}

            # 确保GPU完全释放
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

        print(f"\n{'='*60}")
        print("汇总")
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

    print("\nDemo 完成!")


if __name__ == "__main__":
    main()
