"""
Phase 71: 梯度流架构分析 — 理解梯度如何跨层变换
===================================================

Phase 70的核心发现:
1. eff_rank = 20(L0) → 1.4(L11): 梯度空间逐层压缩
2. CKA(sing,plur) = 0.05(L0) → 0.99(L11): 梯度从语法特异→通用
3. v_grad比v_mean有效4-22倍: 表征几何 ≠ 控制几何

Phase 71要回答:
1. 梯度如何跨层变换? — 逐层梯度流追踪
2. 20个奇异向量对应什么? — 语义分析
3. 通用vs特异的分解 — 漏斗机制的量化

实验设计:
A) 全层梯度流: 12层梯度向量，追踪变换轨迹
B) 奇异向量语义: 早期层top-k方向的注入测试
C) 通用-特异分解: 梯度中多少是输入无关的?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse, time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

GPT2_PATH = "D:/develop/model/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

NVA_PAIRS = [
    ("cat","cats","runs","run"),
    ("dog","dogs","walks","walk"),
    ("bird","birds","flies","fly"),
    ("girl","girls","reads","read"),
    ("boy","boys","sings","sing"),
    ("horse","horses","jumps","jump"),
    ("bear","bears","sleeps","sleep"),
    ("snake","snakes","crawls","crawl"),
    ("fish","fishes","swims","swim"),
    ("frog","frogs","hops","hop"),
    ("star","stars","shines","shine"),
    ("tree","trees","grows","grow"),
    ("clock","clocks","ticks","tick"),
    ("bell","bells","rings","ring"),
    ("fox","foxes","hunts","hunt"),
]


def load_gpt2_float32():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[Phase71] Loading GPT-2 Small float32...")
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_PATH, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(
        GPT2_PATH,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    if torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.n_layer
    d_model = model.config.n_embd
    print(f"[Phase71] GPT-2 loaded: {n_layers} layers, d_model={d_model}, dtype=float32, device={device}")
    return model, tokenizer, device


def compute_autograd_gradient(model, tokenizer, device, sentence, layer_idx,
                               sv_id, pv_id):
    """
    用autograd计算精确梯度 ∂logit_gap/∂h_l
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = toks.input_ids

    layers = model.transformer.h

    # 先no_grad forward捕获h_l
    h_l_captured = [None]
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            h_l_captured[0] = output[0].detach().clone()
        else:
            h_l_captured[0] = output.detach().clone()
        return output

    hook = layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    hook.remove()

    if h_l_captured[0] is None:
        return None, 0.0

    h_l = h_l_captured[0]

    # 用requires_grad的h_l做backward
    h_l_grad = h_l.detach().clone().requires_grad_(True)

    def replace_hook(module, input, output):
        if isinstance(output, tuple):
            return (h_l_grad,) + output[1:]
        return h_l_grad

    hook2 = layers[layer_idx].register_forward_hook(replace_hook)

    with torch.enable_grad():
        out = model(input_ids=input_ids)
        logits = out.logits[0, -1, :].float()
        logit_gap = logits[pv_id] - logits[sv_id]
        logit_gap.backward()

    hook2.remove()

    gradient = h_l_grad.grad[0, -1, :].detach().cpu().numpy()
    logit_gap_val = float(logit_gap.detach().cpu())

    return gradient, logit_gap_val


def compute_autograd_top_logits(model, tokenizer, device, sentence, layer_idx, top_k=10):
    """
    计算梯度w.r.t. top-k logits — 理解梯度不仅对logit_gap，而是对整个输出分布
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = toks.input_ids

    layers = model.transformer.h

    h_l_captured = [None]
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            h_l_captured[0] = output[0].detach().clone()
        else:
            h_l_captured[0] = output.detach().clone()
        return output

    hook = layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        out_base = model(input_ids=input_ids)
        logits_base = out_base.logits[0, -1, :].float().cpu().numpy()
    hook.remove()

    if h_l_captured[0] is None:
        return None, None, None

    h_l = h_l_captured[0]

    # 找到top-k token IDs
    top_k_ids = np.argsort(logits_base)[-top_k:][::-1]

    # 对每个top-k logit计算梯度
    gradients = {}
    for tid in top_k_ids:
        h_l_grad = h_l.detach().clone().requires_grad_(True)

        def replace_hook(module, input, output):
            if isinstance(output, tuple):
                return (h_l_grad,) + output[1:]
            return h_l_grad

        hook2 = layers[layer_idx].register_forward_hook(replace_hook)
        with torch.enable_grad():
            out = model(input_ids=input_ids)
            logits = out.logits[0, -1, :].float()
            logits[tid].backward()

        hook2.remove()
        gradients[tid] = h_l_grad.grad[0, -1, :].detach().cpu().numpy()

    return gradients, top_k_ids, logits_base


def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XY = np.trace((X.T @ X) @ (Y.T @ Y))
    XX = np.trace((X.T @ X) @ (X.T @ X))
    YY = np.trace((Y.T @ Y) @ (Y.T @ Y))
    denom = np.sqrt(max(XX * YY, 1e-30))
    if denom < 1e-30:
        return 0.0
    return XY / denom


def inject_and_test(model, tokenizer, device, sentence, layer_idx, direction, beta,
                    sv_id, pv_id):
    """注入方向并测量效果"""
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    layers = model.transformer.h

    # 基线
    with torch.no_grad():
        out_base = model(**toks)
        logits_base = out_base.logits[0, -1, :].float().cpu().numpy()

    # 捕获h_l
    h_captured = [None]
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            h_captured[0] = output[0].detach().clone()
        else:
            h_captured[0] = output.detach().clone()
        return output

    hook = layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(**toks)
    hook.remove()

    h_l = h_captured[0]
    if h_l is None:
        return None, None

    # 注入
    h_mod = h_l.detach().clone()
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    h_mod[0, -1, :] += beta * direction_t

    def replace_hook(module, input, output, h_replace=h_mod):
        if isinstance(output, tuple):
            return (h_replace,) + output[1:]
        return h_replace

    hook2 = layers[layer_idx].register_forward_hook(replace_hook)
    with torch.no_grad():
        out_mod = model(**toks)
        logits_mod = out_mod.logits[0, -1, :].float().cpu().numpy()
    hook2.remove()

    gap_base = logits_base[pv_id] - logits_base[sv_id]
    gap_mod = logits_mod[pv_id] - logits_mod[sv_id]

    return gap_mod - gap_base, logits_mod - logits_base


# ============================================================
# Experiment A: 全层梯度流
# ============================================================
def experiment_a(model, tokenizer, device, n_test=15):
    """
    ★★★ 全12层梯度流追踪

    对每个输入，计算所有12层的梯度向量:
    g_l = ∂logit_gap/∂h_l

    分析:
    1. 梯度范数跨层变化
    2. 相邻层梯度余弦相似度
    3. 全层梯度矩阵的SVD — 梯度空间维度
    4. 梯度"轨迹"的几何形状
    """
    print("\n" + "="*70)
    print("Experiment A: Full-Layer Gradient Flow")
    print("="*70)

    n_layers = model.config.n_layer
    sv_id = tokenizer.encode(" runs", add_special_tokens=False)[0]
    pv_id = tokenizer.encode(" run", add_special_tokens=False)[0]

    # 收集所有层的梯度
    # grad_matrix[condition][layer] = [n_test, d_model]
    grad_data = {'sing': defaultdict(list), 'plur': defaultdict(list)}

    for cond in ['sing', 'plur']:
        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:n_test]):
            noun = sn if cond == 'sing' else pn
            sentence = f"The {noun}"

            for li in range(n_layers):
                grad, gap = compute_autograd_gradient(
                    model, tokenizer, device, sentence, li, sv_id, pv_id
                )
                if grad is not None:
                    grad_data[cond][li].append(grad)

            if (i+1) % 5 == 0:
                print(f"  {cond}: {i+1}/{n_test} done")

    # 分析1: 梯度范数跨层变化
    print("\n--- Gradient Norm Across Layers ---")
    for cond in ['sing', 'plur']:
        norms = []
        for li in range(n_layers):
            if li in grad_data[cond] and grad_data[cond][li]:
                n = [np.linalg.norm(g) for g in grad_data[cond][li]]
                norms.append((li, np.mean(n), np.std(n)))
        print(f"  {cond}:")
        for li, mn, sd in norms:
            print(f"    L{li:2d}: norm={mn:.4f}±{sd:.4f}")

    # 分析2: 相邻层梯度余弦相似度
    print("\n--- Inter-Layer Gradient Cosine Similarity ---")
    for cond in ['sing', 'plur']:
        print(f"  {cond}:")
        for li in range(n_layers - 1):
            if li in grad_data[cond] and (li+1) in grad_data[cond]:
                cos_vals = []
                for g1, g2 in zip(grad_data[cond][li], grad_data[cond][li+1]):
                    n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_vals.append(np.dot(g1, g2) / (n1 * n2))
                if cos_vals:
                    print(f"    cos(L{li},L{li+1})={np.mean(cos_vals):.4f}±{np.std(cos_vals):.4f}")

    # 分析3: 全层梯度矩阵SVD (合并所有层)
    print("\n--- Full-Layer Gradient SVD ---")
    for cond in ['sing', 'plur']:
        all_grads = []
        for li in range(n_layers):
            if li in grad_data[cond]:
                all_grads.extend(grad_data[cond][li])

        if len(all_grads) < 5:
            continue

        G = np.array(all_grads)  # [n_total, d_model]
        U, S, Vt = np.linalg.svd(G, full_matrices=False)

        S_norm = S / S.sum()
        eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30)))

        print(f"  {cond} (n={len(all_grads)}): eff_rank={eff_rank:.2f}")
        print(f"    S[0:10]={S[:10].round(3)}")
        print(f"    S/S[0]={(S[:10]/S[0]).round(4)}")

    # 分析4: 逐层梯度空间维度 (每层单独SVD)
    print("\n--- Per-Layer Effective Rank ---")
    for cond in ['sing', 'plur']:
        ranks = []
        for li in range(n_layers):
            if li in grad_data[cond] and len(grad_data[cond][li]) >= 3:
                G_l = np.array(grad_data[cond][li])
                U, S, Vt = np.linalg.svd(G_l, full_matrices=False)
                S_norm = S / S.sum()
                eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30)))
                ranks.append((li, eff_rank, S[:3]))
        print(f"  {cond}:")
        for li, er, s3 in ranks:
            print(f"    L{li:2d}: eff_rank={er:.2f}, S[:3]={s3.round(3)}")

    # 分析5: 跨层CKA (sing vs plur)
    print("\n--- Cross-Condition CKA Per Layer ---")
    for li in range(n_layers):
        if li in grad_data['sing'] and li in grad_data['plur']:
            G_s = np.array(grad_data['sing'][li])
            G_p = np.array(grad_data['plur'][li])
            if len(G_s) >= 3 and len(G_p) >= 3:
                cka = linear_cka(G_s, G_p)
                print(f"  L{li:2d}: CKA(sing,plur)={cka:.4f}")

    return grad_data


# ============================================================
# Experiment B: 奇异向量语义分析
# ============================================================
def experiment_b(model, tokenizer, device, n_test=12):
    """
    ★★★ 早期层奇异向量的语义分析

    Phase 70发现L0有rank≈20。这20个方向是什么?
    
    方法:
    1. 在L0收集梯度矩阵 G [n, d_model]
    2. SVD: G = U S V^T → V的列是右奇异向量
    3. 对每个奇异向量v_k，注入β·v_k看对logit_gap的效果
    4. 对每个奇异向量v_k，注入β·v_k看对整个logit分布的效果
    """
    print("\n" + "="*70)
    print("Experiment B: Singular Vector Semantic Analysis")
    print("="*70)

    sv_id = tokenizer.encode(" runs", add_special_tokens=False)[0]
    pv_id = tokenizer.encode(" run", add_special_tokens=False)[0]

    # 选3个关键层: 早期(L0)，中期(L6)，晚期(L11)
    target_layers = [0, 6, 11]

    for li in target_layers:
        print(f"\n  === Layer {li} ===")

        # 收集梯度矩阵
        sing_grads = []
        plur_grads = []

        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:n_test]):
            grad_s, _ = compute_autograd_gradient(
                model, tokenizer, device, f"The {sn}", li, sv_id, pv_id
            )
            if grad_s is not None:
                sing_grads.append(grad_s)

            grad_p, _ = compute_autograd_gradient(
                model, tokenizer, device, f"The {pn}", li, sv_id, pv_id
            )
            if grad_p is not None:
                plur_grads.append(grad_p)

        if len(sing_grads) < 3:
            print("  Not enough data")
            continue

        G_all = np.array(sing_grads + plur_grads)  # [2n, d_model]
        U, S, Vt = np.linalg.svd(G_all, full_matrices=False)

        print(f"  SVD: S[:10]={S[:10].round(4)}")
        print(f"  S/S[0]: {(S[:10]/S[0]).round(4)}")

        # 对top-5奇异向量进行注入测试
        n_sv = min(5, len(S))
        betas = [1.0, 4.0]

        print(f"\n  Injection test (n_test={n_test}):")
        print(f"  {'SV':>4} {'σ':>8} {'β':>4} {'Δgap(sing)':>12} {'Δgap(plur)':>12} {'Δgap_diff':>12} {'flip_s':>7} {'flip_p':>7}")

        for k in range(n_sv):
            v_k = Vt[k, :]  # 右奇异向量
            sigma_k = S[k]

            for beta in betas:
                delta_sing = []
                delta_plur = []
                flip_s = 0
                flip_p = 0

                for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:n_test]):
                    # Sing sentence
                    ds, _ = inject_and_test(
                        model, tokenizer, device, f"The {sn}", li, v_k, beta,
                        sv_id, pv_id
                    )
                    if ds is not None:
                        delta_sing.append(ds)

                    # Plur sentence
                    dp, _ = inject_and_test(
                        model, tokenizer, device, f"The {pn}", li, v_k, beta,
                        sv_id, pv_id
                    )
                    if dp is not None:
                        delta_plur.append(dp)

                mean_ds = np.mean(delta_sing) if delta_sing else 0
                mean_dp = np.mean(delta_plur) if delta_plur else 0
                diff = mean_dp - mean_ds

                print(f"  {k:4d} {sigma_k:8.4f} {beta:4.1f} {mean_ds:12.4f} {mean_dp:12.4f} {diff:12.4f}")

        # 额外: 奇异向量之间的正交性验证
        print(f"\n  SV orthogonality (cosine between SVs):")
        for i in range(min(5, n_sv)):
            for j in range(i+1, min(5, n_sv)):
                cos = np.dot(Vt[i], Vt[j])
                print(f"    cos(v_{i}, v_{j}) = {cos:.6f}")


# ============================================================
# Experiment C: 通用-特异分解
# ============================================================
def experiment_c(model, tokenizer, device, n_test=15):
    """
    ★★★ 梯度分解: 通用成分 vs 输入特异成分

    g_l(x) = g_universal + g_specific(x)

    g_universal = E_x[g_l(x)]  (平均梯度)
    g_specific(x) = g_l(x) - g_universal

    问题:
    1. 通用成分的范数占比跨层如何变化?
    2. 特异成分是否语法相关?
    3. 通用成分是否就是第一奇异向量方向?
    """
    print("\n" + "="*70)
    print("Experiment C: Universal vs Specific Decomposition")
    print("="*70)

    n_layers = model.config.n_layer
    sv_id = tokenizer.encode(" runs", add_special_tokens=False)[0]
    pv_id = tokenizer.encode(" run", add_special_tokens=False)[0]

    # 收集所有梯度
    # grad_dict[layer][condition][sentence_idx] = gradient vector
    grad_dict = defaultdict(lambda: defaultdict(list))

    for cond in ['sing', 'plur']:
        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:n_test]):
            noun = sn if cond == 'sing' else pn
            sentence = f"The {noun}"

            for li in range(n_layers):
                grad, _ = compute_autograd_gradient(
                    model, tokenizer, device, sentence, li, sv_id, pv_id
                )
                if grad is not None:
                    grad_dict[li][cond].append(grad)

            if (i+1) % 5 == 0:
                print(f"  {cond}: {i+1}/{n_test} done")

    # 分析1: 通用成分占比
    print("\n--- Universal Component Ratio ---")
    print(f"  {'Layer':>5} {'||g_univ||':>10} {'||g_spec_s||':>12} {'||g_spec_p||':>12} {'ratio_univ':>11} {'cos(univ,SV0)':>14}")

    for li in range(n_layers):
        all_grads = grad_dict[li]['sing'] + grad_dict[li]['plur']
        if len(all_grads) < 5:
            continue

        G = np.array(all_grads)
        g_universal = G.mean(axis=0)
        g_universal_norm = np.linalg.norm(g_universal)

        # 特异成分范数
        spec_sing = [g - g_universal for g in grad_dict[li]['sing']]
        spec_plur = [g - g_universal for g in grad_dict[li]['plur']]
        spec_s_norm = np.mean([np.linalg.norm(s) for s in spec_sing]) if spec_sing else 0
        spec_p_norm = np.mean([np.linalg.norm(s) for s in spec_plur]) if spec_plur else 0

        total_norm = np.mean([np.linalg.norm(g) for g in all_grads])
        ratio_univ = g_universal_norm / max(total_norm, 1e-10)

        # 通用方向 vs SVD第一奇异向量
        U, S, Vt = np.linalg.svd(G, full_matrices=False)
        cos_univ_sv0 = np.dot(g_universal, Vt[0]) / max(np.linalg.norm(g_universal) * np.linalg.norm(Vt[0]), 1e-10)

        print(f"  L{li:2d}   {g_universal_norm:10.4f} {spec_s_norm:12.4f} {spec_p_norm:12.4f} {ratio_univ:11.4f} {cos_univ_sv0:14.6f}")

    # 分析2: 特异成分是否语法相关?
    print("\n--- Specific Component: Sing vs Plur Direction ---")
    print(f"  {'Layer':>5} {'cos(spec_s,spec_p)':>20} {'Δgap(univ)':>12} {'Δgap(spec_s)':>14} {'Δgap(spec_p)':>14}")

    for li in range(n_layers):
        all_grads = grad_dict[li]['sing'] + grad_dict[li]['plur']
        if len(all_grads) < 5:
            continue

        G = np.array(all_grads)
        g_universal = G.mean(axis=0)

        # 平均特异方向
        spec_sing = [g - g_universal for g in grad_dict[li]['sing']]
        spec_plur = [g - g_universal for g in grad_dict[li]['plur']]

        mean_spec_s = np.mean(spec_sing, axis=0) if spec_sing else np.zeros_like(g_universal)
        mean_spec_p = np.mean(spec_plur, axis=0) if spec_plur else np.zeros_like(g_universal)

        n_s, n_p = np.linalg.norm(mean_spec_s), np.linalg.norm(mean_spec_p)
        if n_s > 1e-10 and n_p > 1e-10:
            cos_spec = np.dot(mean_spec_s, mean_spec_p) / (n_s * n_p)
        else:
            cos_spec = 0.0

        # 测试注入效果
        beta = 2.0
        # Universal direction
        g_univ_dir = g_universal / max(np.linalg.norm(g_universal), 1e-10)
        delta_gap_univ = []
        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:min(6, n_test)]):
            dg, _ = inject_and_test(
                model, tokenizer, device, f"The {sn}", li, g_univ_dir, beta,
                sv_id, pv_id
            )
            if dg is not None:
                delta_gap_univ.append(dg)

        # Sing-specific direction
        if n_s > 1e-10:
            spec_s_dir = mean_spec_s / n_s
            delta_gap_spec_s = []
            for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:min(6, n_test)]):
                dg, _ = inject_and_test(
                    model, tokenizer, device, f"The {sn}", li, spec_s_dir, beta,
                    sv_id, pv_id
                )
                if dg is not None:
                    delta_gap_spec_s.append(dg)
        else:
            delta_gap_spec_s = [0]

        # Plur-specific direction
        if n_p > 1e-10:
            spec_p_dir = mean_spec_p / n_p
            delta_gap_spec_p = []
            for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:min(6, n_test)]):
                dg, _ = inject_and_test(
                    model, tokenizer, device, f"The {sn}", li, spec_p_dir, beta,
                    sv_id, pv_id
                )
                if dg is not None:
                    delta_gap_spec_p.append(dg)
        else:
            delta_gap_spec_p = [0]

        print(f"  L{li:2d}   {cos_spec:20.4f} {np.mean(delta_gap_univ):12.4f} {np.mean(delta_gap_spec_s):14.4f} {np.mean(delta_gap_spec_p):14.4f}")

    # 分析3: 通用方向与均值差方向的关系
    print("\n--- Universal Gradient vs Mean Difference Direction ---")
    for li in range(n_layers):
        all_grads = grad_dict[li]['sing'] + grad_dict[li]['plur']
        if len(all_grads) < 5:
            continue

        g_universal = np.mean(all_grads, axis=0)

        # Mean difference direction (从hidden states)
        layers = model.transformer.h
        sing_h = []
        plur_h = []
        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:min(8, n_test)]):
            for cond, noun in [('sing', sn), ('plur', pn)]:
                sent = f"The {noun}"
                toks = tokenizer(sent, return_tensors="pt").to(device)
                h_captured = [None]
                def capture_hook(module, input, output):
                    if isinstance(output, tuple):
                        h_captured[0] = output[0].detach().float().cpu().numpy()
                    else:
                        h_captured[0] = output.detach().float().cpu().numpy()
                    return output
                hook = layers[li].register_forward_hook(capture_hook)
                with torch.no_grad():
                    _ = model(**toks)
                hook.remove()
                if h_captured[0] is not None:
                    if cond == 'sing':
                        sing_h.append(h_captured[0][0, -1, :])
                    else:
                        plur_h.append(h_captured[0][0, -1, :])

        if sing_h and plur_h:
            v_mean = np.mean(plur_h, axis=0) - np.mean(sing_h, axis=0)
            v_mean_norm = np.linalg.norm(v_mean)
            g_univ_norm = np.linalg.norm(g_universal)
            if v_mean_norm > 1e-10 and g_univ_norm > 1e-10:
                cos = np.dot(v_mean, g_universal) / (v_mean_norm * g_univ_norm)
            else:
                cos = 0.0
            print(f"  L{li:2d}: cos(v_mean, g_universal) = {cos:.4f}, ||v_mean||={v_mean_norm:.4f}, ||g_univ||={g_univ_norm:.4f}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="a", choices=["a", "b", "c", "all"])
    parser.add_argument("--n_test", type=int, default=15)
    args = parser.parse_args()

    model, tokenizer, device = load_gpt2_float32()

    try:
        if args.exp in ["a", "all"]:
            experiment_a(model, tokenizer, device, n_test=args.n_test)

        if args.exp in ["b", "all"]:
            experiment_b(model, tokenizer, device, n_test=args.n_test)

        if args.exp in ["c", "all"]:
            experiment_c(model, tokenizer, device, n_test=args.n_test)
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print("\n[Phase71] Done. GPU memory released.")
