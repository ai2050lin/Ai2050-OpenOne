"""
多模型加载与测试标准 Demo
=========================
基于 2026-05-05 model_utils.py 的标准加载方式

用法:
  python tests/glm5/model_demo.py qwen3       # 只测Qwen3
  python tests/glm5/model_demo.py deepseek7b   # 只测DeepSeek7B
  python tests/glm5/model_demo.py glm4         # 只测GLM4
  python tests/glm5/model_demo.py all          # 顺序测试所有模型

=== 加载方式要点 (来自5月5日验证通过的方案) ===
1. torch_dtype=torch.bfloat16  (不用8bit量化, 8bit导致加载极慢/卡住)
2. device_map="cpu" 先加载, 再 model.to("cuda") 整体移动
3. trust_remote_code=True, local_files_only=True
4. low_cpu_mem_usage=True
5. output_hidden_states=True 只在 forward() 调用时传, 不传给 from_pretrained()
6. 释放时: release_model(model); model=None; gc.collect(); torch.cuda.empty_cache()

=== 模型参数 ===
| 模型       | 类名                | 层数 | d_model | vocab  | GPU内存 |
|-----------|--------------------|------|---------|--------|---------|
| qwen3     | Qwen3ForCausalLM   | 36   | 2560    | 151936 | ~8 GB   |
| deepseek7b| Qwen2ForCausalLM   | 28   | 3584    | 152064 | ~15 GB  |
| glm4      | GlmForCausalLM     | 40   | 4096    | 151552 | ~18 GB  |
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


def test_model(model_name: str):
    """完整测试一个模型的加载、前向推理、隐藏状态提取、权重访问"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")

    # ---- 1. 加载 ----
    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    t_load = time.time() - t0
    print(f"[1] 加载耗时: {t_load:.1f}s, device={device}")

    # ---- 2. 基本信息 ----
    info = get_model_info(model, model_name)
    print(f"[2] class={info.model_class}, n_layers={info.n_layers}, "
          f"d_model={info.d_model}, vocab={info.vocab_size}, mlp_type={info.mlp_type}")

    # ---- 3. 前向推理 (output_hidden_states=True 传给 forward, 不传给 from_pretrained) ----
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    t_fwd = time.time() - t0
    hs = out.hidden_states  # tuple of (n_layers+1,) 每个 [1, seq_len, d_model]
    logits = out.logits[0, -1].float().cpu().numpy()
    print(f"[3] 前向推理 ({t_fwd:.1f}s): {len(hs)}层隐藏状态, "
          f"最后层norm={hs[-1].float().norm():.2f}")

    # ---- 4. Top-5 预测 ----
    top5_ids = np.argsort(logits)[-5:][::-1]
    top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
    print(f"[4] Top-5: {top5}")

    # ---- 5. 生成 ----
    t0 = time.time()
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask,
                                  max_new_tokens=20, do_sample=False)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t_gen = time.time() - t0
    print(f"[5] 生成 ({t_gen:.1f}s): '{gen_text}'")

    # ---- 6. W_U 权重 ----
    W_U = get_W_U(model)
    print(f"[6] W_U: shape={W_U.shape}, norm={np.linalg.norm(W_U):.2f}")

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

    # ---- 8. GPU 内存 & 释放 ----
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"[8] GPU 内存: {gpu_mem:.2f} GB")

    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    gpu_after = torch.cuda.memory_allocated() / 1e9
    print(f"[9] 释放后 GPU: {gpu_after:.2f} GB")

    return {
        "model": model_name,
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

        print(f"\n{'='*60}")
        print("汇总")
        print(f"{'='*60}")
        for name, r in results.items():
            if "error" in r:
                print(f"  {name}: FAILED - {r['error']}")
            else:
                print(f"  {name}: load={r['load_time']}s, fwd={r['fwd_time']}s, "
                      f"gen={r['gen_time']}s, gpu={r['gpu_mem_gb']}GB, top={r['top5'][0]}")
    else:
        test_model(model_name)

    print("\nDemo 完成!")


if __name__ == "__main__":
    main()
