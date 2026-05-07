"""
Phase 72: 梯度分解深度分析 — LayerNorm瓶颈与全局语法轴
========================================================

Phase 71的核心发现:
1. 梯度 = 通用成分 + 特异成分，通用成分从46%→100%增长
2. 早期层SV0/SV1有"语法双通道"分工
3. 最后层梯度坍缩到秩1

Phase 72要回答:
1. LayerNorm+Unembed为什么导致秩1? — 理论分析+实验验证
2. g_spec(l)的方向是否跨层一致? — 是否存在全局语法控制轴?
3. 双通道结构是否泛化到其他语法特征? — 时态/语态测试

核心理论预测:
∂logit_gap/∂h_L = (W[pv]-W[sv]) · ∂LN(h)/∂h
∂LN(h)/∂h = (1/σ)(I - 1/d·11^T - LN(h)·LN(h)^T)
所以梯度 ≈ (1/σ)(W[pv]-W[sv]) — 一个固定方向! 
h的影响只是小的rank-1修正。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse
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

# 时态测试对
TENSE_PAIRS = [
    ("walked", "walks"),   # 过去 vs 现在
    ("played", "plays"),
    ("jumped", "jumps"),
    ("cooked", "cooks"),
    ("cleaned", "cleans"),
    ("worked", "works"),
    ("helped", "helps"),
    ("talked", "talks"),
    ("smiled", "smiles"),
    ("danced", "dances"),
    ("moved", "moves"),
    ("lived", "lives"),
]


def load_gpt2_float32():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[Phase72] Loading GPT-2 Small float32...")
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
    print(f"[Phase72] GPT-2 loaded: {n_layers} layers, d_model={d_model}")
    return model, tokenizer, device


def compute_autograd_gradient(model, tokenizer, device, sentence, layer_idx,
                               sv_id, pv_id):
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
        _ = model(input_ids=input_ids)
    hook.remove()

    if h_l_captured[0] is None:
        return None, 0.0

    h_l = h_l_captured[0]
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
    return gradient, float(logit_gap.detach().cpu())


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


# ============================================================
# Experiment A: LayerNorm+Unembed 梯度秩的理论分析
# ============================================================
def experiment_a(model, tokenizer, device, n_test=12):
    """
    ★★★ 理论预测: 最后层梯度是秩1的，方向≈W[pv]-W[sv]

    验证:
    1. 计算 W[pv]-W[sv] (unembedding差异向量)
    2. 计算 ∂logit_gap/∂h_11 的autograd梯度
    3. 比较两者方向是否一致
    4. 量化LayerNorm的影响(小的rank-1修正)
    """
    print("\n" + "="*70)
    print("Experiment A: LayerNorm+Unembed Gradient Analysis")
    print("="*70)

    sv_id = tokenizer.encode(" runs", add_special_tokens=False)[0]
    pv_id = tokenizer.encode(" run", add_special_tokens=False)[0]

    # 1. 获取Unembed矩阵
    W_unembed = model.lm_head.weight.detach().cpu().numpy()  # [vocab, d_model]
    print(f"  Unembed matrix shape: {W_unembed.shape}")
    print(f"  sv_id={sv_id} ('{tokenizer.decode([sv_id])}'), pv_id={pv_id} ('{tokenizer.decode([pv_id])}')")

    # 2. 理论预测的梯度方向: W[pv] - W[sv]
    w_diff = W_unembed[pv_id] - W_unembed[sv_id]  # [d_model]
    w_diff_norm = np.linalg.norm(w_diff)
    print(f"  ||W[pv]-W[sv]|| = {w_diff_norm:.4f}")

    # 3. 对多个输入计算L11的autograd梯度
    print("\n--- Autograd Gradient at L11 vs W[pv]-W[sv] ---")
    cos_vals_sing = []
    cos_vals_plur = []
    grad_norms = []

    for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:n_test]):
        for cond, noun in [('sing', sn), ('plur', pn)]:
            sentence = f"The {noun}"
            grad, gap = compute_autograd_gradient(
                model, tokenizer, device, sentence, 11, sv_id, pv_id
            )
            if grad is not None:
                grad_norm = np.linalg.norm(grad)
                cos = np.dot(grad, w_diff) / max(grad_norm * w_diff_norm, 1e-10)
                if cond == 'sing':
                    cos_vals_sing.append((cos, grad_norm, gap))
                else:
                    cos_vals_plur.append((cos, grad_norm, gap))
                grad_norms.append(grad_norm)

    print(f"  cos(grad, W[pv]-W[sv]):")
    print(f"    sing: mean={np.mean([c for c,_,_ in cos_vals_sing]):.6f}, "
          f"std={np.std([c for c,_,_ in cos_vals_sing]):.6f}")
    print(f"    plur: mean={np.mean([c for c,_,_ in cos_vals_plur]):.6f}, "
          f"std={np.std([c for c,_,_ in cos_vals_plur]):.6f}")

    # 4. 分析: 梯度与W_diff的偏差是否来自LayerNorm?
    # LayerNorm梯度: ∂LN(h)/∂h = (1/σ)(I - 1/d·11^T - LN(h)·LN(h)^T)
    # 所以梯度 = (1/σ)(W_diff - mean(W_diff)·1 - (W_diff·LN(h))·LN(h)^T)

    print("\n--- LayerNorm Effect on Gradient ---")

    # 计算GPT-2的LayerNorm参数
    # GPT-2的最后LayerNorm在model.transformer.ln_f
    ln_f = model.transformer.ln_f
    ln_weight = ln_f.weight.detach().cpu().numpy()  # [d_model]
    ln_bias = ln_f.bias.detach().cpu().numpy()  # [d_model]
    ln_eps = ln_f.eps
    print(f"  LayerNorm eps={ln_eps}, weight_norm={np.linalg.norm(ln_weight):.4f}")

    # 对一个具体输入，计算理论梯度和实际梯度的对比
    sentence = "The cat"
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = toks.input_ids

    layers = model.transformer.h

    # 捕获L11输出(LayerNorm之前)
    h11_captured = [None]
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            h11_captured[0] = output[0].detach().clone()
        else:
            h11_captured[0] = output.detach().clone()
        return output

    hook = layers[11].register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    hook.remove()

    h11 = h11_captured[0][0, -1, :].cpu().numpy()  # [d_model]

    # 计算LayerNorm的理论梯度
    # LN(x) = γ * (x - μ) / σ + β
    # ∂logit_gap/∂x = γ * (∂logit_gap/∂LN) * (I - 1/d·11^T - (x-μ)(x-μ)^T/(dσ²)) / σ

    # 实际autograd梯度
    grad_actual, _ = compute_autograd_gradient(
        model, tokenizer, device, sentence, 11, sv_id, pv_id
    )

    if grad_actual is not None:
        # 简化: 直接比较cos(grad, W_diff)
        # 如果LayerNorm只做缩放，则梯度≈γ/σ * (W_diff - projection)
        print(f"\n  Input: '{sentence}'")
        print(f"  ||h_11|| = {np.linalg.norm(h11):.4f}")
        print(f"  h_11 mean = {np.mean(h11):.4f}, std = {np.std(h11):.4f}")
        print(f"  ||grad_actual|| = {np.linalg.norm(grad_actual):.4f}")
        print(f"  cos(grad, W_diff) = {np.dot(grad_actual, w_diff)/max(np.linalg.norm(grad_actual)*w_diff_norm,1e-10):.6f}")

        # 尝试: 理论梯度 = γ/σ * W_diff (最简单的近似)
        h_mean = np.mean(h11)
        h_std = np.std(h11)
        theoretical_simple = ln_weight * w_diff / h_std
        cos_theory = np.dot(grad_actual, theoretical_simple) / max(
            np.linalg.norm(grad_actual) * np.linalg.norm(theoretical_simple), 1e-10)
        print(f"  Simple theory (γ/σ · W_diff): cos={cos_theory:.6f}")

        # 更精确: 减去均值分量
        w_diff_centered = w_diff - np.mean(w_diff)
        theoretical_centered = ln_weight * w_diff_centered / h_std
        cos_centered = np.dot(grad_actual, theoretical_centered) / max(
            np.linalg.norm(grad_actual) * np.linalg.norm(theoretical_centered), 1e-10)
        print(f"  Centered theory (γ/σ · (W_diff-mean)): cos={cos_centered:.6f}")

    # 5. 跨层验证: 在其他层，cos(grad, W_diff)是否递减?
    print("\n--- cos(grad, W_diff) Across All Layers ---")
    for li in range(12):
        cos_vals = []
        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:6]):
            for cond, noun in [('sing', sn), ('plur', pn)]:
                sentence = f"The {noun}"
                grad, _ = compute_autograd_gradient(
                    model, tokenizer, device, sentence, li, sv_id, pv_id
                )
                if grad is not None:
                    cos = np.dot(grad, w_diff) / max(np.linalg.norm(grad) * w_diff_norm, 1e-10)
                    cos_vals.append(cos)
        if cos_vals:
            print(f"  L{li:2d}: cos(grad, W_diff) = {np.mean(cos_vals):.4f}±{np.std(cos_vals):.4f}")


# ============================================================
# Experiment B: g_spec(l) 跨层一致性 — 是否存在全局语法轴?
# ============================================================
def experiment_b(model, tokenizer, device, n_test=15):
    """
    ★★★ 核心问题: 语法特异梯度方向g_spec(l)是否跨层一致?

    如果g_spec(l)在所有层指向相同方向 → 存在全局语法控制轴
    如果g_spec(l)逐层旋转 → 语法控制是层特异的

    方法:
    1. 对每层l，计算g_spec(l) = mean(grad_sing) - mean(grad_plur)
    2. 计算跨层CKA(cos)矩阵
    3. 分析g_spec(l)的旋转轨迹
    """
    print("\n" + "="*70)
    print("Experiment B: Cross-Layer Consistency of g_spec")
    print("="*70)

    sv_id = tokenizer.encode(" runs", add_special_tokens=False)[0]
    pv_id = tokenizer.encode(" run", add_special_tokens=False)[0]
    n_layers = model.config.n_layer

    # 收集每层的梯度
    grad_sing = defaultdict(list)
    grad_plur = defaultdict(list)

    for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:n_test]):
        for li in range(n_layers):
            grad_s, _ = compute_autograd_gradient(
                model, tokenizer, device, f"The {sn}", li, sv_id, pv_id
            )
            if grad_s is not None:
                grad_sing[li].append(grad_s)

            grad_p, _ = compute_autograd_gradient(
                model, tokenizer, device, f"The {pn}", li, sv_id, pv_id
            )
            if grad_p is not None:
                grad_plur[li].append(grad_p)

        if (i+1) % 5 == 0:
            print(f"  {i+1}/{n_test} done")

    # 计算g_spec(l) = mean(grad_sing) - mean(grad_plur)
    g_spec = {}
    g_univ = {}
    for li in range(n_layers):
        if li in grad_sing and li in grad_plur and len(grad_sing[li]) >= 3:
            mean_s = np.mean(grad_sing[li], axis=0)
            mean_p = np.mean(grad_plur[li], axis=0)
            g_spec[li] = mean_s - mean_p  # 特异方向: sing-plur差
            g_univ[li] = (mean_s + mean_p) / 2  # 通用方向

    # 1. g_spec(l)的跨层余弦相似度矩阵
    print("\n--- g_spec Cross-Layer Cosine Similarity ---")
    layer_list = sorted(g_spec.keys())
    n_l = len(layer_list)

    # 打印关键对
    print(f"  {'':>4}", end='')
    for li in layer_list:
        print(f"  L{li:2d}", end='')
    print()

    for li_i in layer_list:
        print(f"  L{li_i:2d}", end='')
        for li_j in layer_list:
            n_i = np.linalg.norm(g_spec[li_i])
            n_j = np.linalg.norm(g_spec[li_j])
            if n_i > 1e-10 and n_j > 1e-10:
                cos = np.dot(g_spec[li_i], g_spec[li_j]) / (n_i * n_j)
                print(f" {cos:5.2f}", end='')
            else:
                print(f"   N/A", end='')
        print()

    # 2. g_spec的跨层CKA
    print("\n--- g_spec Cross-Layer CKA ---")
    # 用梯度矩阵而不是单个方向来计算CKA
    for li_i in layer_list[:4]:
        for li_j in layer_list[-4:]:
            if li_i in grad_sing and li_j in grad_sing:
                G_i = np.array(grad_sing[li_i] + grad_plur[li_i])
                G_j = np.array(grad_sing[li_j] + grad_plur[li_j])
                if len(G_i) >= 3 and len(G_j) >= 3:
                    cka = linear_cka(G_i, G_j)
                    print(f"  CKA(L{li_i}, L{li_j}) = {cka:.4f}")

    # 3. g_spec(l)与v_mean(l)的关系
    print("\n--- g_spec vs v_mean (Representation Direction) ---")
    layers = model.transformer.h
    for li in layer_list:
        # v_mean
        sing_h = []
        plur_h = []
        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:8]):
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
            n_vm = np.linalg.norm(v_mean)
            n_gs = np.linalg.norm(g_spec[li])
            if n_vm > 1e-10 and n_gs > 1e-10:
                cos = np.dot(v_mean, g_spec[li]) / (n_vm * n_gs)
            else:
                cos = 0.0
            print(f"  L{li:2d}: cos(v_mean, g_spec) = {cos:.4f}, "
                  f"||v_mean||={n_vm:.2f}, ||g_spec||={n_gs:.4f}")

    # 4. g_spec(l)与g_univ(l)的关系
    print("\n--- g_spec vs g_univ ---")
    for li in layer_list:
        n_gs = np.linalg.norm(g_spec[li])
        n_gu = np.linalg.norm(g_univ[li])
        if n_gs > 1e-10 and n_gu > 1e-10:
            cos = np.dot(g_spec[li], g_univ[li]) / (n_gs * n_gu)
        else:
            cos = 0.0
        ratio = n_gs / max(n_gu, 1e-10)
        print(f"  L{li:2d}: cos(g_spec, g_univ) = {cos:.4f}, "
              f"||g_spec||/||g_univ|| = {ratio:.4f}")

    return g_spec, g_univ


# ============================================================
# Experiment C: 多语法特征测试 — 时态
# ============================================================
def experiment_c(model, tokenizer, device, n_test=12):
    """
    ★★★ 时态特征是否也有双通道结构?

    测试: "He walked" vs "He walks"
    - 过去时: walked
    - 现在时: walks
    
    问题:
    1. 时态的梯度是否也有通用+特异分解?
    2. 时态特异方向是否与数特异方向不同?
    3. 两个语法特征是否占据梯度空间的不同维度?
    """
    print("\n" + "="*70)
    print("Experiment C: Tense Feature Gradient Analysis")
    print("="*70)

    # 时态的token IDs
    past_id = tokenizer.encode(" walked", add_special_tokens=False)[0]
    pres_id = tokenizer.encode(" walks", add_special_tokens=False)[0]
    print(f"  past_id={past_id} ('{tokenizer.decode([past_id])}'), "
          f"pres_id={pres_id} ('{tokenizer.decode([pres_id])}')")

    n_layers = model.config.n_layer

    # 收集时态梯度
    grad_past = defaultdict(list)
    grad_pres = defaultdict(list)

    for i, (past_v, pres_v) in enumerate(TENSE_PAIRS[:n_test]):
        past_toks = tokenizer(f"He {past_v}", add_special_tokens=False)
        pres_toks = tokenizer(f"He {pres_v}", add_special_tokens=False)

        for li in [0, 3, 6, 9, 11]:
            # 过去时
            sent = f"He {past_v}"
            # 用autograd计算∂(logit_pres - logit_past)/∂h_l
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids

            layers = model.transformer.h
            h_captured = [None]
            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    h_captured[0] = output[0].detach().clone()
                else:
                    h_captured[0] = output.detach().clone()
                return output

            hook = layers[li].register_forward_hook(capture_hook)
            with torch.no_grad():
                _ = model(input_ids=input_ids)
            hook.remove()

            if h_captured[0] is None:
                continue

            h_l = h_captured[0]
            h_l_grad = h_l.detach().clone().requires_grad_(True)

            def replace_hook(module, input, output):
                if isinstance(output, tuple):
                    return (h_l_grad,) + output[1:]
                return h_l_grad

            hook2 = layers[li].register_forward_hook(replace_hook)
            with torch.enable_grad():
                out = model(input_ids=input_ids)
                logits = out.logits[0, -1, :].float()
                logit_gap = logits[pres_id] - logits[past_id]
                logit_gap.backward()
            hook2.remove()

            grad_past[li].append(h_l_grad.grad[0, -1, :].detach().cpu().numpy())

            # 现在时
            sent = f"He {pres_v}"
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids

            h_captured[0] = None
            hook = layers[li].register_forward_hook(capture_hook)
            with torch.no_grad():
                _ = model(input_ids=input_ids)
            hook.remove()

            if h_captured[0] is None:
                continue

            h_l = h_captured[0]
            h_l_grad = h_l.detach().clone().requires_grad_(True)

            def replace_hook2(module, input, output):
                if isinstance(output, tuple):
                    return (h_l_grad,) + output[1:]
                return h_l_grad

            hook2 = layers[li].register_forward_hook(replace_hook2)
            with torch.enable_grad():
                out = model(input_ids=input_ids)
                logits = out.logits[0, -1, :].float()
                logit_gap = logits[pres_id] - logits[past_id]
                logit_gap.backward()
            hook2.remove()

            grad_pres[li].append(h_l_grad.grad[0, -1, :].detach().cpu().numpy())

        if (i+1) % 4 == 0:
            print(f"  {i+1}/{n_test} done")

    # 分析时态梯度
    print("\n--- Tense Gradient Analysis ---")
    for li in [0, 3, 6, 9, 11]:
        if li not in grad_past or li not in grad_pres:
            continue
        if len(grad_past[li]) < 3 or len(grad_pres[li]) < 3:
            continue

        G_past = np.array(grad_past[li])
        G_pres = np.array(grad_pres[li])

        # 有效秩
        G_all = np.vstack([G_past, G_pres])
        U, S, Vt = np.linalg.svd(G_all, full_matrices=False)
        S_norm = S / S.sum()
        eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30)))

        # CKA
        cka = linear_cka(G_past, G_pres)

        # 特异方向
        g_spec_tense = np.mean(G_pres, axis=0) - np.mean(G_past, axis=0)
        g_univ_tense = (np.mean(G_pres, axis=0) + np.mean(G_past, axis=0)) / 2
        ratio_univ = np.linalg.norm(g_univ_tense) / max(
            np.mean([np.linalg.norm(g) for g in G_all]), 1e-10)

        print(f"  L{li:2d}: eff_rank={eff_rank:.2f}, CKA(past,pres)={cka:.4f}, "
              f"ratio_univ={ratio_univ:.4f}, ||g_spec||={np.linalg.norm(g_spec_tense):.4f}")

    # ★★★ 关键比较: 时态g_spec vs 数g_spec是否正交?
    print("\n--- Number g_spec vs Tense g_spec ---")
    sv_id_num = tokenizer.encode(" runs", add_special_tokens=False)[0]
    pv_id_num = tokenizer.encode(" run", add_special_tokens=False)[0]

    for li in [0, 3, 6, 9, 11]:
        # 数的g_spec
        sing_grads = []
        plur_grads = []
        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:8]):
            grad_s, _ = compute_autograd_gradient(
                model, tokenizer, device, f"The {sn}", li, sv_id_num, pv_id_num
            )
            if grad_s is not None:
                sing_grads.append(grad_s)
            grad_p, _ = compute_autograd_gradient(
                model, tokenizer, device, f"The {pn}", li, sv_id_num, pv_id_num
            )
            if grad_p is not None:
                plur_grads.append(grad_p)

        if len(sing_grads) < 3 or len(plur_grads) < 3:
            continue
        if li not in grad_past or li not in grad_pres:
            continue
        if len(grad_past[li]) < 3 or len(grad_pres[li]) < 3:
            continue

        g_spec_number = np.mean(sing_grads, axis=0) - np.mean(plur_grads, axis=0)
        g_spec_tense = np.mean(grad_pres[li], axis=0) - np.mean(grad_past[li], axis=0)

        n_num = np.linalg.norm(g_spec_number)
        n_tense = np.linalg.norm(g_spec_tense)

        if n_num > 1e-10 and n_tense > 1e-10:
            cos = np.dot(g_spec_number, g_spec_tense) / (n_num * n_tense)
        else:
            cos = 0.0

        print(f"  L{li:2d}: cos(g_spec_number, g_spec_tense) = {cos:.4f}, "
              f"||g_spec_num||={n_num:.4f}, ||g_spec_tense||={n_tense:.4f}")

    # ★★★ 合并两个特征的梯度矩阵
    print("\n--- Combined Number+Tense Gradient SVD ---")
    for li in [0, 6, 11]:
        # 收集4类梯度: sing+present, sing+past, plur+present, plur+past
        # 简化: 用已有的数据
        all_grads = []

        # 数特征梯度
        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:6]):
            for cond, noun in [('sing', sn), ('plur', pn)]:
                grad, _ = compute_autograd_gradient(
                    model, tokenizer, device, f"The {noun}", li, sv_id_num, pv_id_num
                )
                if grad is not None:
                    all_grads.append(grad)

        # 时态特征梯度
        if li in grad_past and li in grad_pres:
            all_grads.extend(grad_past[li])
            all_grads.extend(grad_pres[li])

        if len(all_grads) < 10:
            continue

        G = np.array(all_grads)
        U, S, Vt = np.linalg.svd(G, full_matrices=False)
        S_norm = S / S.sum()
        eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30)))

        print(f"  L{li:2d}: eff_rank={eff_rank:.2f}, S[:8]={S[:8].round(3)}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="a", choices=["a", "b", "c", "all"])
    parser.add_argument("--n_test", type=int, default=12)
    args = parser.parse_args()

    model, tokenizer, device = load_gpt2_float32()

    try:
        if args.exp in ["a", "all"]:
            experiment_a(model, tokenizer, device, n_test=args.n_test)

        if args.exp in ["b", "all"]:
            g_spec, g_univ = experiment_b(model, tokenizer, device, n_test=args.n_test)

        if args.exp in ["c", "all"]:
            experiment_c(model, tokenizer, device, n_test=args.n_test)
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print("\n[Phase72] Done. GPU memory released.")
