"""Phase 149b: EPS敏感性测试 — 验证方向丢失是真实现象还是大扰动假象
=================================================================
Phase 149 Exp2发现: cos(δ_injected, δ_propagated) 在5层内从1.0降到0.71
Phase 147的cos≈0.999可能是cos(h_clean, h_perturbed), 不是扰动方向保持

本测试:
  1. 变化eps: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
  2. 测量两种cos:
     a. cos(h_clean, h_perturbed) — Phase 147的度量
     b. cos(δ_injected, δ_propagated) — Phase 149的度量
  3. 在L5和L18两个测量点

如果: 小eps时cos(δ_in, δ_out)≈1.0 → 方向保持是eps依赖的线性区现象
如果: 小eps时cos(δ_in, δ_out)仍然低 → 方向丢失是真实的非线性效应
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc, json, numpy as np, torch
from datetime import datetime
from pathlib import Path
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)

TEST_PROMPTS = [
    "The scientist discovered that the",
    "In the morning, she decided to",
    "The book on the table was about",
    "After the rain stopped, the children",
    "The most important thing about science is",
    "When the sun sets over the ocean,",
    "She walked into the room and saw",
    "The professor explained that the theory",
]

OUTPUT_DIR = Path("tests/glm5_temp")


def get_row_null_bases(W_U, n_components=200):
    d = W_U.shape[1]
    k = min(n_components, d - 2)
    np.random.seed(42)
    n_samples = k + 10
    Omega = np.random.randn(W_U.shape[0], n_samples).astype(np.float32)
    batch_size = 50000
    Y = np.zeros((d, n_samples), dtype=np.float32)
    for i in range(0, W_U.shape[0], batch_size):
        end = min(i + batch_size, W_U.shape[0])
        Y += W_U[i:end].T.astype(np.float32) @ Omega[i:end]
    del Omega
    Q, R = np.linalg.qr(Y)
    del Y, R
    effective_k = min(k, Q.shape[1])
    row_basis = Q[:, :effective_k].T
    return row_basis, effective_k


def project_to_null(vec, row_basis):
    row_component = row_basis.T @ (row_basis @ vec)
    return vec - row_component


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"Phase 149b: EPS Sensitivity Test")
    print(f"Model: {model_name}, Time: {timestamp}")

    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16

    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model

    W_U = get_W_U(model, model_name)
    row_basis, k = get_row_null_bases(W_U)

    epsilons = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    measure_layers = [5, 18, n_layers - 1]
    inject_layer = 0
    n_sents = len(TEST_PROMPTS)

    results = {}

    for eps in epsilons:
        eps_results = {}
        for sent_idx in range(n_sents):
            prompt = TEST_PROMPTS[sent_idx]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Clean forward
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=True)
            clean_hs = out_clean.hidden_states

            # 生成null-space扰动
            np.random.seed(sent_idx * 100 + 7)
            raw_vec = np.random.randn(d_model)
            delta = project_to_null(raw_vec, row_basis)
            norm = np.linalg.norm(delta)
            if norm > 1e-8:
                delta = delta / norm * eps
            else:
                continue

            # 扰动forward (注入在L0, last token position)
            def make_inject_hook(delta_tensor):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        out[0, -1, :] += delta_tensor.to(out.dtype).to(out.device)
                        return (out,) + output[1:]
                    else:
                        out = output.clone()
                        out[0, -1, :] += delta_tensor.to(out.dtype).to(out.device)
                        return out
                return hook

            hooks = [layers[inject_layer].register_forward_hook(
                make_inject_hook(torch.tensor(delta, dtype=torch.float32)))]

            try:
                with torch.no_grad():
                    out_perturbed = model(input_ids=input_ids, attention_mask=attention_mask,
                                          output_hidden_states=True)
            finally:
                for h in hooks:
                    h.remove()

            # 在测量层分析
            for ml in measure_layers:
                if ml > n_layers:
                    continue

                h_clean = clean_hs[ml][0, -1, :].float().cpu().numpy()
                h_perturbed = out_perturbed.hidden_states[ml][0, -1, :].float().cpu().numpy()
                delta_prop = h_perturbed - h_clean

                # Metric A: cos(h_clean, h_perturbed) — Phase 147的度量
                cos_state = float(np.dot(h_clean, h_perturbed) /
                                   (np.linalg.norm(h_clean) * np.linalg.norm(h_perturbed) + 1e-10))

                # Metric B: cos(δ_injected, δ_propagated) — Phase 149的度量
                cos_delta = float(np.dot(delta, delta_prop) /
                                   (np.linalg.norm(delta) * np.linalg.norm(delta_prop) + 1e-10))

                delta_prop_norm = float(np.linalg.norm(delta_prop))
                growth_ratio = delta_prop_norm / (eps + 1e-10)

                key = f"eps{eps}_L{ml}"
                if key not in eps_results:
                    eps_results[key] = {
                        'cos_state': [], 'cos_delta': [],
                        'delta_norm': [], 'growth_ratio': []
                    }
                eps_results[key]['cos_state'].append(cos_state)
                eps_results[key]['cos_delta'].append(cos_delta)
                eps_results[key]['delta_norm'].append(delta_prop_norm)
                eps_results[key]['growth_ratio'].append(growth_ratio)

        for key, data in eps_results.items():
            results[key] = {
                'eps': eps,
                'measure_layer': int(key.split('_L')[1]),
                'avg_cos_state': float(np.mean(data['cos_state'])),
                'avg_cos_delta': float(np.mean(data['cos_delta'])),
                'avg_delta_norm': float(np.mean(data['delta_norm'])),
                'avg_growth': float(np.mean(data['growth_ratio'])),
            }

    # 打印结果
    print("\n" + "="*80)
    print("EPS Sensitivity: Two Metrics Comparison")
    print("="*80)
    print(f"  {'eps':>6}  {'Layer':>5}  {'cos(h,h\')':>10}  {'cos(δ,δ\')':>10}  {'||δ_out||':>10}  {'growth':>8}")
    print(f"  {'':>6}  {'':>5}  {'Phase147':>10}  {'Phase149':>10}  {'':>10}  {'':>8}")

    for eps in epsilons:
        for ml in measure_layers:
            key = f"eps{eps}_L{ml}"
            if key in results:
                r = results[key]
                print(f"  {eps:>6.2f}  L{ml:>4d}  {r['avg_cos_state']:>10.6f}  "
                      f"{r['avg_cos_delta']:>10.4f}  {r['avg_delta_norm']:>10.4f}  "
                      f"{r['avg_growth']:>8.2f}")
        print()

    # 关键分析
    print("\n=== KEY ANALYSIS ===")
    # 找到cos_delta从>0.9降到<0.9的eps阈值
    for ml in measure_layers:
        print(f"\n  At L{ml}:")
        for eps in epsilons:
            key = f"eps{eps}_L{ml}"
            if key in results:
                r = results[key]
                print(f"    eps={eps:.2f}: cos_state={r['avg_cos_state']:.6f}, cos_delta={r['avg_cos_delta']:.4f}, growth={r['avg_growth']:.2f}x")

    # 保存结果
    result_file = OUTPUT_DIR / f"phase149b_{model_name}_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {result_file}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("Model released.")


if __name__ == "__main__":
    main()
