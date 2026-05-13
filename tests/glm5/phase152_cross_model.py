"""
Phase 152 跨模型验证: DS7B + GLM4
=====================================

核心修复:
1. hidden_states[0]=embedding(不受L0 hook影响), 用hs[1]作为参考
2. sample_layers 动态适配不同模型层数
3. 依次运行DS7B和GLM4(避免GPU内存溢出)

关键实验(修正版):
- Exp 1: MI Flow (线性R² + 扰动协方差)
- Exp 2: Logit Boundary Geometry
- Exp 3: Second-Order Propagation (修正: 用hs[1]作为参考)

用法:
  python tests/glm5/phase152_cross_model.py deepseek7b
  python tests/glm5/phase152_cross_model.py glm4
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)

OUTPUT_DIR = Path("tests/glm5_temp")

TEST_PROMPTS = [
    "The scientist discovered that the",
    "In the morning, she decided to",
    "The book on the table was about",
    "After the rain stopped, the children",
    "The most important thing about science is",
    "When the sun sets over the ocean,",
    "She walked into the room and saw",
    "The professor explained that the theory",
    "Despite the challenges, the team managed",
    "The ancient city was known for its",
    "He realized that the answer was",
    "The relationship between language and thought",
    "Every morning she would read the",
    "The experiment showed that the results",
    "Music has the power to change how",
    "The government announced that the new policy",
    "In the future, artificial intelligence will",
    "The philosopher argued that consciousness is",
    "After years of research, they found that",
    "The key difference between the two approaches is",
    "The cat sat on the windowsill and watched",
    "Through the telescope, they observed a new",
    "The river flowed gently through the valley",
    "She opened the letter and read the",
    "The painting on the wall depicted a",
    "During the concert, the audience was",
    "The invention changed the way people",
    "He wrote a letter to his friend about",
    "The students in the classroom were learning",
    "The old building at the corner had",
]

EPSILON = 1.0
N_SENTENCES = 30


def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def get_sample_layers(n_layers):
    """动态生成采样层列表 — 适配不同模型"""
    if n_layers <= 12:
        return list(range(0, n_layers + 1))
    # 均匀采样约10个点 + 首尾
    step = max(1, n_layers // 9)
    layers = list(range(0, n_layers + 1, step))
    layers = sorted(set(layers + [1, n_layers]))
    return layers


# ============================================================
# Exp 1: MI Flow (修正版)
# ============================================================
def exp1_mi_flow(model, tokenizer, model_name):
    print("\n" + "="*60)
    print("Exp 1: Mutual Information Flow (FIXED)")
    print("="*60)

    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model

    sample_layers = get_sample_layers(n_layers)

    # === Method A: 线性预测性 R² ===
    print("\n  --- Method A: Linear Predictability R² ---")

    all_hs = {}
    for li in sample_layers:
        all_hs[li] = []

    for sent_idx in range(min(N_SENTENCES, 30)):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)

        for li in sample_layers:
            vec = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            all_hs[li].append(vec)

    # 对每层, 用h_1(L0 output)预测h_ℓ
    # 修正: 不用h_0(embedding), 用h_1(L0 output)作为起点
    ref_layer = min(1, n_layers)
    h_ref_matrix = np.array(all_hs[ref_layer])
    h_ref_centered = h_ref_matrix - h_ref_matrix.mean(axis=0)

    linear_r2 = {}
    for li in sample_layers:
        hl_matrix = np.array(all_hs[li])
        hl_centered = hl_matrix - hl_matrix.mean(axis=0)

        n_pca = min(50, d_model, h_ref_centered.shape[0] - 1)

        try:
            cov_ref = (h_ref_centered.T @ h_ref_centered) / (h_ref_centered.shape[0] - 1)
            eigvals_ref, eigvecs_ref = np.linalg.eigh(cov_ref)
            idx = np.argsort(-eigvals_ref)[:n_pca]
            h_ref_pca = h_ref_centered @ eigvecs_ref[:, idx]
        except:
            h_ref_pca = h_ref_centered[:, :n_pca]

        try:
            cov_hl = (hl_centered.T @ hl_centered) / (hl_centered.shape[0] - 1)
            eigvals_hl, eigvecs_hl = np.linalg.eigh(cov_hl)
            idx_hl = np.argsort(-eigvals_hl)[:n_pca]
            hl_pca = hl_centered @ eigvecs_hl[:, idx_hl]
        except:
            hl_pca = hl_centered[:, :n_pca]

        N_samples = h_ref_pca.shape[0]
        if N_samples < 5:
            linear_r2[li] = 0.0
            continue

        n_train = max(5, N_samples * 2 // 3)
        h_ref_train = h_ref_pca[:n_train]
        h_ref_test = h_ref_pca[n_train:]
        hl_train = hl_pca[:n_train]
        hl_test = hl_pca[n_train:]

        if h_ref_test.shape[0] < 2:
            linear_r2[li] = 0.0
            continue

        r2_per_component = []
        for comp_j in range(min(10, hl_pca.shape[1])):
            y_train = hl_train[:, comp_j]
            y_test = hl_test[:, comp_j]

            try:
                W, residuals, rank, sv = np.linalg.lstsq(
                    np.column_stack([h_ref_train, np.ones(n_train)]),
                    y_train, rcond=None)
                y_pred = np.column_stack([h_ref_test, np.ones(h_ref_test.shape[0])]) @ W

                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                r2 = 1 - ss_res / max(ss_tot, 1e-10)
                r2_per_component.append(max(0, r2))
            except:
                r2_per_component.append(0.0)

        avg_r2 = np.mean(r2_per_component) if r2_per_component else 0
        linear_r2[li] = float(avg_r2)

    print("\n  Linear Predictability R² (h₁ → hℓ):")
    for li in sample_layers:
        r2 = linear_r2.get(li, 0)
        status = "STRONG" if r2 > 0.3 else "MODERATE" if r2 > 0.1 else "WEAK" if r2 > 0.01 else "~ZERO"
        print(f"    hs[{li:>3d}]: R²={r2:.4f} [{status}]")

    # === Method B: 扰动协方差结构 (修正: 用hs[1]作为参考) ===
    print("\n  --- Method B: Perturbation Covariance Structure (FIXED) ---")

    prompt = TEST_PROMPTS[0]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
    clean_hs = out_clean.hidden_states

    n_perturbations = 100

    delta_at_layer = {}
    for li in sample_layers:
        delta_at_layer[li] = []

    input_deltas = []

    for p_idx in range(n_perturbations):
        np.random.seed(42 + p_idx)
        delta = np.random.randn(d_model)
        delta = delta / np.linalg.norm(delta) * EPSILON
        input_deltas.append(delta.copy())

        layers = get_layers(model)
        delta_tensor = torch.tensor(delta, dtype=torch.float32)

        last_pos = input_ids.shape[1] - 1

        def make_hook(pos, delta_t):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                    return (out,) + output[1:]
                else:
                    out = output.clone()
                    out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                    return out
            return hook

        hooks = [layers[0].register_forward_hook(make_hook(last_pos, delta_tensor))]

        try:
            with torch.no_grad():
                out_perturbed = model(input_ids=input_ids, attention_mask=attention_mask,
                                      output_hidden_states=True)
        except:
            for h in hooks:
                h.remove()
            continue

        for h in hooks:
            h.remove()

        for li in sample_layers:
            perturbed_vec = out_perturbed.hidden_states[li][0, last_pos, :].float().cpu().numpy()
            clean_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
            delta_prop = perturbed_vec - clean_vec
            delta_at_layer[li].append(delta_prop)

    # 修正: 用hs[1]的差作为参考
    ref_hs = min(1, n_layers)
    delta_ref = np.array(delta_at_layer[ref_hs])
    input_deltas_arr = np.array(input_deltas)

    cov_rank = {}
    cov_top_eigenvalue_ratio = {}
    delta_corr_with_input = {}  # 修正: 与原始注入delta的相关
    delta_corr_with_ref = {}    # 与hs[1]的delta相关

    for li in sample_layers:
        delta_matrix = np.array(delta_at_layer[li])

        if delta_matrix.shape[0] < 5:
            cov_rank[li] = 0
            cov_top_eigenvalue_ratio[li] = 0
            delta_corr_with_input[li] = 0
            delta_corr_with_ref[li] = 0
            continue

        try:
            U_d, s_d, Vt_d = np.linalg.svd(delta_matrix, full_matrices=False)
            total_energy = np.sum(s_d ** 2)
            cumulative = np.cumsum(s_d ** 2)
            k95 = np.searchsorted(cumulative, 0.95 * total_energy) + 1
            cov_rank[li] = int(k95)
            top_ratio = (s_d[0] ** 2) / total_energy if total_energy > 0 else 0
            cov_top_eigenvalue_ratio[li] = float(top_ratio)
        except:
            cov_rank[li] = 0
            cov_top_eigenvalue_ratio[li] = 0

        # cos(δ_ℓ, δ_input) — 修正: 用原始注入的delta作为参考
        cos_with_input = []
        for p_idx in range(min(50, delta_matrix.shape[0])):
            d_l = delta_matrix[p_idx]
            d_in = input_deltas_arr[p_idx]
            norm_l = np.linalg.norm(d_l)
            norm_in = np.linalg.norm(d_in)
            if norm_l > 1e-10 and norm_in > 1e-10:
                cos_with_input.append(float(np.dot(d_l, d_in) / (norm_l * norm_in)))
        delta_corr_with_input[li] = float(np.mean(cos_with_input)) if cos_with_input else 0

        # cos(δ_ℓ, δ_ref) — 与hs[1]的delta相关
        cos_with_ref = []
        for p_idx in range(min(50, delta_matrix.shape[0], delta_ref.shape[0])):
            d_l = delta_matrix[p_idx]
            d_r = delta_ref[p_idx]
            norm_l = np.linalg.norm(d_l)
            norm_r = np.linalg.norm(d_r)
            if norm_l > 1e-10 and norm_r > 1e-10:
                cos_with_ref.append(float(np.dot(d_l, d_r) / (norm_l * norm_r)))
        delta_corr_with_ref[li] = float(np.mean(cos_with_ref)) if cos_with_ref else 0

    print("\n  Perturbation Covariance Structure (FIXED):")
    for li in sample_layers:
        rank = cov_rank.get(li, 0)
        top_ratio = cov_top_eigenvalue_ratio.get(li, 0)
        corr_in = delta_corr_with_input.get(li, 0)
        corr_ref = delta_corr_with_ref.get(li, 0)
        print(f"    hs[{li:>3d}]: rank={rank}, top_ratio={top_ratio:.4f}, "
              f"cos(δ_ℓ,δ_input)={corr_in:.4f}, cos(δ_ℓ,δ_ref)={corr_ref:.4f}")

    return {
        'linear_r2': linear_r2,
        'cov_rank': cov_rank,
        'cov_top_eigenvalue_ratio': cov_top_eigenvalue_ratio,
        'delta_corr_with_input': delta_corr_with_input,
        'delta_corr_with_ref': delta_corr_with_ref,
    }


# ============================================================
# Exp 2: Logit Boundary Geometry
# ============================================================
def exp2_logit_boundary(model, tokenizer, model_name):
    print("\n" + "="*60)
    print("Exp 2: Logit Boundary Geometry")
    print("="*60)

    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model

    results = []

    for sent_idx in range(min(N_SENTENCES, 25)):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = out.logits[0, -1, :].float().cpu().numpy()
        probs = softmax(logits)

        sorted_ids = np.argsort(-logits)
        top1_id = sorted_ids[0]
        top2_id = sorted_ids[1]

        logit_margin = logits[top1_id] - logits[top2_id]
        prob_margin = probs[top1_id] - probs[top2_id]

        inject_layer = n_layers // 2
        last_pos = input_ids.shape[1] - 1

        np.random.seed(sent_idx * 100)
        delta = np.random.randn(d_model)
        delta = delta / np.linalg.norm(delta)

        eps_scan = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        switching_eps = None

        for eps in eps_scan:
            delta_scaled = delta * eps

            layers = get_layers(model)
            delta_tensor = torch.tensor(delta_scaled, dtype=torch.float32)

            def make_hook(pos, delta_t):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                        return (out,) + output[1:]
                    else:
                        out = output.clone()
                        out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                        return out
                return hook

            hooks = [layers[inject_layer].register_forward_hook(make_hook(last_pos, delta_tensor))]

            try:
                with torch.no_grad():
                    out_p = model(input_ids=input_ids, attention_mask=attention_mask)

                perturbed_logits = out_p.logits[0, -1, :].float().cpu().numpy()
                perturbed_top1 = int(np.argmax(perturbed_logits))

                if perturbed_top1 != top1_id and switching_eps is None:
                    switching_eps = eps
            except:
                pass

            for h in hooks:
                h.remove()

        results.append({
            'sent_idx': sent_idx,
            'logit_margin': float(logit_margin),
            'prob_margin': float(prob_margin),
            'switching_eps': switching_eps,
        })

    # 汇总
    margins = [r['logit_margin'] for r in results]
    switching = [r['switching_eps'] for r in results if r['switching_eps'] is not None]

    print(f"\n  Logit margin: mean={np.mean(margins):.3f}, std={np.std(margins):.3f}, "
          f"range=[{np.min(margins):.3f}, {np.max(margins):.3f}]")
    print(f"  Switching rate: {len(switching)}/{len(results)} = {len(switching)/len(results):.1%}")
    if switching:
        print(f"  Switching eps: mean={np.mean(switching):.3f}, range=[{np.min(switching):.3f}, {np.max(switching):.3f}]")

    # 按 margin 分组
    print("\n  --- By Logit Margin ---")
    for label, lo, hi in [("Narrow(<1)", 0, 1), ("Medium(1-3)", 1, 3), ("Wide(>3)", 3, 1000)]:
        subset = [r for r in results if lo <= r['logit_margin'] < hi]
        if subset:
            switch_rate = sum(1 for r in subset if r['switching_eps'] is not None) / len(subset)
            switch_vals = [r['switching_eps'] for r in subset if r['switching_eps'] is not None]
            if switch_vals:
                print(f"  {label}: N={len(subset)}, switch_rate={switch_rate:.1%}, "
                      f"avg_eps={np.mean(switch_vals):.2f}")
            else:
                print(f"  {label}: N={len(subset)}, switch_rate={switch_rate:.1%}")

    # 相关性
    if len(results) > 5:
        margins_switch = [r['logit_margin'] for r in results if r['switching_eps'] is not None]
        margins_no_switch = [r['logit_margin'] for r in results if r['switching_eps'] is None]
        if margins_switch and margins_no_switch:
            print(f"\n  Margin for switching: mean={np.mean(margins_switch):.3f}")
            print(f"  Margin for no-switch: mean={np.mean(margins_no_switch):.3f}")

    return results


# ============================================================
# Exp 3: Second-Order Propagation (修正版)
# ============================================================
def exp3_second_order(model, tokenizer, model_name):
    print("\n" + "="*60)
    print("Exp 3: Second-Order Perturbation Propagation (FIXED)")
    print("="*60)

    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    d_model = info.d_model

    # 动态采样层
    sample_layers = get_sample_layers(n_layers)
    # 确保包含hs[1]
    if 1 not in sample_layers:
        sample_layers = sorted([1] + sample_layers)

    prompt = TEST_PROMPTS[0]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    last_pos = input_ids.shape[1] - 1

    with torch.no_grad():
        out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
    clean_hs = out_clean.hidden_states

    n_perturbations = 200

    delta_at_layer = {}
    for li in sample_layers:
        delta_at_layer[li] = []

    input_deltas = []

    layers = get_layers(model)

    for p_idx in range(n_perturbations):
        np.random.seed(200 + p_idx)
        delta = np.random.randn(d_model)
        delta = delta / np.linalg.norm(delta) * EPSILON
        input_deltas.append(delta.copy())

        delta_tensor = torch.tensor(delta, dtype=torch.float32)

        def make_hook(pos, delta_t):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                    return (out,) + output[1:]
                else:
                    out = output.clone()
                    out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                    return out
            return hook

        hooks = [layers[0].register_forward_hook(make_hook(last_pos, delta_tensor))]

        try:
            with torch.no_grad():
                out_p = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)

            for li in sample_layers:
                p_vec = out_p.hidden_states[li][0, last_pos, :].float().cpu().numpy()
                c_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
                delta_at_layer[li].append(p_vec - c_vec)
        except:
            pass

        for h in hooks:
            h.remove()

        if p_idx % 50 == 0:
            print(f"  Progress: {p_idx}/{n_perturbations}")

    print(f"  Progress: {n_perturbations}/{n_perturbations}")

    # === 分析 ===
    # 参考层: hs[1] = L0 output
    ref_layer = 1
    delta_ref = np.array(delta_at_layer[ref_layer])
    input_deltas_arr = np.array(input_deltas)

    # 1. 一阶cos: cos(δ_ℓ, δ_input)
    print("\n  --- First-Order: cos(δ_ℓ, δ_input) ---")
    first_order_cos = {}
    for li in sample_layers:
        cos_values = []
        delta_l = np.array(delta_at_layer[li])
        for p in range(min(100, delta_l.shape[0])):
            nl = np.linalg.norm(delta_l[p])
            ni = np.linalg.norm(input_deltas_arr[p])
            if nl > 1e-10 and ni > 1e-10:
                cos_values.append(float(np.dot(delta_l[p], input_deltas_arr[p]) / (nl * ni)))
        avg_cos = np.mean(cos_values) if cos_values else 0
        first_order_cos[li] = avg_cos
        print(f"    hs[{li:>3d}]: cos(δ_ℓ, δ_input)={avg_cos:.6f}")

    # 2. 二阶: PCA + subspace overlap
    print(f"\n  --- Second-Order: PCA Subspace Overlap [ref=hs[{ref_layer}]] ---")

    delta_ref_centered = delta_ref - delta_ref.mean(axis=0)
    try:
        U_ref, s_ref, Vt_ref = np.linalg.svd(delta_ref_centered, full_matrices=False)
        pcs_ref = Vt_ref[:min(10, Vt_ref.shape[0]), :]
    except:
        print("  ERROR: SVD failed for reference layer")
        return {
            'first_order_cos': first_order_cos,
            'second_order_overlap': {},
            'pc1_correlation': {},
        }

    second_order_overlap = {}
    pc1_corr = {}
    rank_90 = {}
    top_ratio = {}

    for li in sample_layers:
        delta_l = np.array(delta_at_layer[li])
        if delta_l.shape[0] < 10:
            continue

        delta_l_centered = delta_l - delta_l.mean(axis=0)
        try:
            U_l, s_l, Vt_l = np.linalg.svd(delta_l_centered, full_matrices=False)
            pcs_l = Vt_l[:min(10, Vt_l.shape[0]), :]

            # PC1 correlation
            pc1_c = abs(float(np.dot(pcs_l[0], pcs_ref[0])))
            pc1_corr[li] = pc1_c

            # Subspace overlap (top 5 PCs)
            n_sub = min(5, pcs_ref.shape[0], pcs_l.shape[0])
            if n_sub > 0:
                Q_ref = pcs_ref[:n_sub].T @ pcs_ref[:n_sub]
                Q_l = pcs_l[:n_sub].T @ pcs_l[:n_sub]
                overlap = np.trace(Q_ref @ Q_l) / n_sub
            else:
                overlap = 0
            second_order_overlap[li] = overlap

            # Effective rank
            total_e = np.sum(s_l ** 2)
            cumul = np.cumsum(s_l ** 2)
            k90 = np.searchsorted(cumul, 0.90 * total_e) + 1
            rank_90[li] = int(k90)
            top_ratio[li] = float(s_l[0] ** 2 / total_e) if total_e > 0 else 0

            print(f"    hs[{li:>3d}]: PC1_corr={pc1_c:.4f}, overlap={overlap:.4f}, "
                  f"rank(90%)={k90}, top_ratio={top_ratio[li]:.4f}")
        except:
            pass

    # 3. 关键对比: 一阶 vs 二阶
    print(f"\n  === CRITICAL: First-Order vs Second-Order ===")
    print(f"  {'Layer':>6} | {'cos(1st)':>10} | {'overlap(2nd)':>12} | {'Diagnosis'}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*25}")

    for li in sample_layers:
        cos_1st = first_order_cos.get(li, 0)
        overlap_2nd = second_order_overlap.get(li, 0)

        if cos_1st < 0.1 and overlap_2nd > 0.2:
            diagnosis = "2ND-ORDER PRESERVED ★"
        elif cos_1st < 0.1 and overlap_2nd < 0.05:
            diagnosis = "1st&2nd both decayed"
        elif cos_1st >= 0.1:
            diagnosis = "1st-order present"
        else:
            diagnosis = "mixed/weak 2nd-order"

        print(f"  hs[{li:>3d}] | {cos_1st:>10.6f} | {overlap_2nd:>12.4f} | {diagnosis}")

    return {
        'first_order_cos': first_order_cos,
        'second_order_overlap': second_order_overlap,
        'pc1_correlation': pc1_corr,
        'rank_90': rank_90,
        'top_ratio': top_ratio,
    }


# ============================================================
# 主函数
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"Phase 152 Cross-Model: Statistical Language Dynamics (FIXED)")
    print(f"Model: {model_name}")
    print(f"Time: {timestamp}")

    # 加载模型
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    print(f"Mode: {'8bit' if use_8bit else 'bfloat16'}")

    # 清理GPU内存
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    # 使用自定义加载以处理DS7B的attention问题
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cfg = MODEL_CONFIGS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        # DS7B 使用 sdpa 而非 eager (sliding window attention 不支持 eager)
        attn_impl = "sdpa" if model_name == "deepseek7b" else "eager"
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    model.eval()
    device = next(model.parameters()).device
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    print(f"Load time: {time.time()-t0:.1f}s")

    # 运行实验
    exp1_results = exp1_mi_flow(model, tokenizer, model_name)
    exp2_results = exp2_logit_boundary(model, tokenizer, model_name)
    exp3_results = exp3_second_order(model, tokenizer, model_name)

    # 保存结果
    all_results = {
        "phase": "152_cross_model",
        "model": model_name,
        "timestamp": timestamp,
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
        "exp1_mi_flow": {
            "linear_r2": exp1_results['linear_r2'],
            "cov_rank": exp1_results['cov_rank'],
            "cov_top_eigenvalue_ratio": exp1_results['cov_top_eigenvalue_ratio'],
            "delta_corr_with_input": exp1_results['delta_corr_with_input'],
            "delta_corr_with_ref": exp1_results['delta_corr_with_ref'],
        },
        "exp2_logit_boundary": exp2_results,
        "exp3_second_order": exp3_results,
    }

    result_file = OUTPUT_DIR / f"phase152_cross_{model_name}_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)

    print(f"\nResults saved to: {result_file}")

    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("Model released.")


if __name__ == "__main__":
    main()
