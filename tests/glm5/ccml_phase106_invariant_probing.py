"""
Phase 106: 规范不变量与约束传输分析
=====================================

Phase 105的硬伤 (用户批判):
  1. "margin被读出来 ≠ 信息在那里被释放"
     错误: margin下降 → 信息被压抑
     正确: margin下降可能只是基底旋转，信息仍在
     关键: 需要invariant probing区分"压抑"vs"基底旋转"

  2. "trajectory curvature ≠ computational complexity"
     高维空间中近正交增量更新本来就会产生长弯曲路径
     curvature/disp≈20不证明强非线性吸引子动力学
     只能证明"不是平行传输"，不能证明"混沌计算"

  3. "margin_dir太接近直接操纵输出层"
     W_U[en]-W_U[zh]本质是decoder读出方向
     α=0.5即可flip不证明内部用此机制
     只证明"decoder geometry matters"，不证明"internal uses margin coding"

  4. "三阶段模型是人为切分"
     数据是连续变化，没有sharp bifurcation
     更严格描述: "decoder alignment oscillates across depth"
     不是"three discrete stages"

Phase 106核心升级:
  从"信息在哪里"到"哪些不变量跨层保留"

关键实验:
  Exp 1: Probe Invariance — 线性/非线性探针在各层的翻译可解码性
    核心问题: L12-L30的margin下降是"压抑"还是"基底旋转"?
    - 线性探针(h@W_U): 可解码性在L12-L30下降 → 之前观察到的
    - MLP探针(2层): 可解码性是否也下降?
    - 如果MLP也能读出 → 基底旋转，信息仍在
    - 如果MLP也读不出 → 真正压抑

  Exp 2: Subspace Transport — 翻译相关子空间的跨层传输
    不追单方向，看子空间
    - 在每层计算翻译相关子空间(翻译vs中文的主成分差)
    - 测量相邻层的子空间principal angles
    - 如果principal angles小 → 子空间被保持(有结构的传输)
    - 如果principal angles大 → 子空间被重参数化

  Exp 3: Gauge-Invariant Quantities — CKA/CCA跨层分析
    哪些量在rotation/LN/basis change后仍保持
    - CKA between layers (translation-relevant directions)
    - Mutual information estimate
    - Fisher geometry: ∂logits/∂h的Fisher information matrix

  Exp 4: Attention-Mediated Transport — token间约束传输
    核心突破: 从单token状态几何 → token-token相互作用
    - 翻译prompt中，不同token位置的hidden state如何交互
    - 哪些attention head负责"翻译约束"的跨token传输
    - 翻译约束是从哪个token位置传播到最后位置的?

Run:
  python tests/glm5/ccml_phase106_invariant_probing.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase106_invariant_probing.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase106_invariant_probing.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase106_invariant_probing.py --model qwen3 --exp 4
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U


# ============================================================
# 测试数据
# ============================================================
TRANSLATION_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("水", "water"), ("火", "fire"),
    ("花", "flower"), ("鱼", "fish"), ("树", "tree"), ("鸟", "bird"),
    ("马", "horse"), ("铁", "iron"), ("金", "gold"), ("茶", "tea"),
    ("月", "moon"), ("日", "sun"), ("风", "wind"), ("雨", "rain"),
    ("雪", "snow"), ("石", "stone"), ("云", "cloud"), ("星", "star"),
    ("河", "river"), ("山", "mountain"), ("草", "grass"), ("光", "light"),
    ("血", "blood"), ("冰", "ice"), ("沙", "sand"), ("雾", "fog"),
    ("雷", "thunder"), ("霜", "frost"),
]

# 额外训练词对用于MLP探针训练
EXTRA_PAIRS = [
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("白", "white"),
    ("黑", "black"), ("大", "big"), ("小", "small"), ("长", "long"),
    ("短", "short"), ("新", "new"), ("旧", "old"), ("快", "fast"),
    ("慢", "slow"), ("高", "tall"), ("低", "low"), ("热", "hot"),
    ("冷", "cold"), ("甜", "sweet"), ("苦", "bitter"), ("酸", "sour"),
]


def get_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def apply_layer_norm(h, eps=1e-5):
    mean = np.mean(h)
    std = np.std(h)
    return (h - mean) / (std + eps)


# ============================================================
# Exp 1: Probe Invariance — 线性/非线性探针的翻译可解码性
# ============================================================
def exp1_probe_invariance(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 1: Probe Invariance — 区分压抑vs基底旋转")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # === Step 1: 收集各层hidden states ===
    # 需要大量样本训练MLP探针
    print(f"\n  === Step 1: 收集hidden states ===")

    all_pairs = TRANSLATION_PAIRS + EXTRA_PAIRS  # 50个词对
    n_pairs = len(all_pairs)

    # 每个词对，每个层，收集: zh_continue, trans_short, trans_instr
    # 标签: 0=中文续写, 1=翻译
    layer_states = defaultdict(lambda: {"zh": [], "trans": []})

    for zh, en in all_pairs:
        prompts = {
            "zh": f"{zh}是一种",
            "trans": f'"{zh}"的英文是',
        }
        for ptype, prompt in prompts.items():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                layer_states[l][ptype].append(h)

    # === Step 2: 线性探针 vs MLP探针 ===
    print(f"\n  === Step 2: 训练探针 ===")

    results = {}
    sample_layers = list(range(0, n_layers + 1, 3))  # 每3层测一次
    if n_layers not in sample_layers:
        sample_layers.append(n_layers)

    for l in sample_layers:
        # 数据
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float32)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float32)

        # 标签
        X = np.vstack([zh_data, trans_data])
        y = np.array([0]*len(zh_data) + [1]*len(trans_data))

        # LN后的数据
        X_ln = np.array([apply_layer_norm(x) for x in X], dtype=np.float32)

        # 1. 线性探针 (sklearn LogisticRegression)
        try:
            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(X_ln, y)
            linear_acc = accuracy_score(y, lr.predict(X_ln))
        except:
            linear_acc = 0.5

        # 2. MLP探针 (2层, 128隐藏单元)
        try:
            X_t = torch.tensor(X_ln, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.long)

            mlp = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )

            optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(200):
                optimizer.zero_grad()
                out = mlp(X_t)
                loss = criterion(out, y_t)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred = mlp(X_t).argmax(dim=1).numpy()
                mlp_acc = accuracy_score(y, pred)
        except:
            mlp_acc = 0.5

        # 3. 直接margin可解码性 (用W_U)
        # 在翻译prompt下: margin = h_ln @ (w_en - w_zh)
        margins_zh = []
        margins_trans = []
        for i, (zh, en) in enumerate(all_pairs):
            en_id = get_token_id(tokenizer, en)
            zh_id = get_token_id(tokenizer, zh)
            if en_id is None or zh_id is None:
                continue
            w_en = W_U[en_id]
            w_zh = W_U[zh_id]
            margin_dir = w_en - w_zh

            h_zh_ln = apply_layer_norm(layer_states[l]["zh"][i].astype(np.float64))
            h_tr_ln = apply_layer_norm(layer_states[l]["trans"][i].astype(np.float64))

            margins_zh.append(np.dot(h_zh_ln, margin_dir))
            margins_trans.append(np.dot(h_tr_ln, margin_dir))

        # margin分类: sign(margin)>0 → 英文
        margin_correct = sum(1 for m_zh, m_tr in zip(margins_zh, margins_trans)
                            if m_tr > m_zh)  # 翻译prompt的英文margin应该更高
        margin_acc = margin_correct / len(margins_zh) if margins_zh else 0.5

        # mean margin
        mean_margin_zh = np.mean(margins_zh) if margins_zh else 0
        mean_margin_trans = np.mean(margins_trans) if margins_trans else 0
        margin_diff = mean_margin_trans - mean_margin_zh

        results[l] = {
            "linear_probe_acc": float(linear_acc),
            "mlp_probe_acc": float(mlp_acc),
            "margin_classification_acc": float(margin_acc),
            "mean_margin_zh": float(mean_margin_zh),
            "mean_margin_trans": float(mean_margin_trans),
            "margin_diff": float(margin_diff),
        }

        print(f"  L{l:2d}: linear={linear_acc:.3f}, mlp={mlp_acc:.3f}, "
              f"margin_acc={margin_acc:.3f}, margin_diff={margin_diff:.3f}")

    # === Step 3: 分析压抑vs基底旋转 ===
    print(f"\n  === Step 3: 压抑vs基底旋转分析 ===")

    # 关键: 如果MLP准确率在"压抑阶段"仍然高 → 基底旋转
    # 如果MLP准确率也下降 → 真正压抑
    linear_accs = [results[l]["linear_probe_acc"] for l in sorted(results.keys())]
    mlp_accs = [results[l]["mlp_probe_acc"] for l in sorted(results.keys())]
    margin_accs = [results[l]["margin_classification_acc"] for l in sorted(results.keys())]

    print(f"\n  线性探针范围: {min(linear_accs):.3f} - {max(linear_accs):.3f}")
    print(f"  MLP探针范围: {min(mlp_accs):.3f} - {max(mlp_accs):.3f}")
    print(f"  Margin分类范围: {min(margin_accs):.3f} - {max(margin_accs):.3f}")

    # 在margin最低的层，MLP是否仍能读出?
    min_margin_layer = min(results.keys(), key=lambda l: results[l]["margin_diff"])
    min_margin_data = results[min_margin_layer]
    print(f"\n  Margin最低层L{min_margin_layer}: margin_diff={min_margin_data['margin_diff']:.3f}")
    print(f"    线性: {min_margin_data['linear_probe_acc']:.3f}")
    print(f"    MLP:  {min_margin_data['mlp_probe_acc']:.3f}")

    if min_margin_data['mlp_probe_acc'] > 0.8:
        print(f"    → MLP仍能读出翻译信息 → 基底旋转，不是压抑！")
    elif min_margin_data['mlp_probe_acc'] < 0.6:
        print(f"    → MLP也读不出 → 可能是真正压抑")
    else:
        print(f"    → MLP部分读出 → 部分压抑+部分旋转")

    out_path = f"tests/glm5_temp/phase106_exp1_{model_name}_probe_invariance.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 2: Subspace Transport — 翻译相关子空间的跨层传输
# ============================================================
def exp2_subspace_transport(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 2: Subspace Transport — 子空间跨层传输")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # 收集各层的翻译差分子空间
    print(f"\n  === Step 1: 收集翻译差分子空间 ===")

    # 对每个词，收集zh和trans的hidden state
    layer_zh_states = defaultdict(list)
    layer_trans_states = defaultdict(list)

    for zh, en in TRANSLATION_PAIRS:
        zh_prompt = f"{zh}是一种"
        trans_prompt = f'"{zh}"的英文是'

        for ptype, prompt in [("zh", zh_prompt), ("trans", trans_prompt)]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                if ptype == "zh":
                    layer_zh_states[l].append(h)
                else:
                    layer_trans_states[l].append(h)

    # === Step 2: 计算每层的翻译相关子空间 ===
    print(f"\n  === Step 2: 计算翻译相关子空间 ===")

    # 翻译差分矩阵: D = mean(trans_states) - mean(zh_states)
    # 然后SVD得到子空间
    subspaces = {}
    subspace_dims = [1, 3, 5, 10, 20]  # 不同维度

    for l in range(n_layers + 1):
        zh_mat = np.array(layer_zh_states[l], dtype=np.float64)  # (n_pairs, d_model)
        tr_mat = np.array(layer_trans_states[l], dtype=np.float64)

        # 差分矩阵
        diff_mat = tr_mat - zh_mat  # (n_pairs, d_model)

        # SVD
        U, S, Vt = np.linalg.svd(diff_mat, full_matrices=False)

        # 各维度的子空间基
        subspaces[l] = {
            "singular_values": S.tolist(),
            "basis": Vt,  # (min(n_pairs, d_model), d_model)
            "variance_explained": (S**2 / np.sum(S**2)).tolist() if np.sum(S**2) > 0 else [],
        }

    # === Step 3: 计算相邻层的principal angles ===
    print(f"\n  === Step 3: Principal angles between adjacent layers ===")

    results = {}

    for dim in subspace_dims:
        angle_data = []
        for l in range(n_layers):
            # 两个子空间的基
            A = subspaces[l]["basis"][:dim]   # (dim, d_model)
            B = subspaces[l+1]["basis"][:dim]

            # Principal angles via SVD
            # cos(θ_i) = singular values of A @ B.T
            M = A @ B.T  # (dim, dim)
            cos_vals = np.linalg.svd(M, compute_uv=False)
            cos_vals = np.clip(cos_vals, 0, 1)
            angles = np.arccos(cos_vals) * 180 / np.pi

            angle_data.append({
                "mean_angle": float(np.mean(angles)),
                "max_angle": float(np.max(angles)),
                "min_angle": float(np.min(angles)),
                "angles": angles.tolist(),
            })

        results[f"dim_{dim}"] = angle_data

        print(f"\n  Subspace dim={dim}:")
        for l_idx, l in enumerate(range(0, n_layers, 6)):
            if l_idx < len(angle_data):
                ad = angle_data[l]
                print(f"    L{l}→L{l+1}: mean={ad['mean_angle']:.1f}°, max={ad['max_angle']:.1f}°")

    # === Step 4: 全局子空间保持 ===
    print(f"\n  === Step 4: Global subspace preservation ===")

    for dim in subspace_dims:
        print(f"\n  Subspace dim={dim}:")
        for ref_l in [0, 6, 12, 21, 27, 33]:
            A = subspaces[ref_l]["basis"][:dim]
            B = subspaces[n_layers]["basis"][:dim]
            M = A @ B.T
            cos_vals = np.linalg.svd(M, compute_uv=False)
            cos_vals = np.clip(cos_vals, 0, 1)
            angles = np.arccos(cos_vals) * 180 / np.pi
            print(f"    L{ref_l}→L{n_layers}: mean={np.mean(angles):.1f}°, "
                  f"max={np.max(angles):.1f}°")

    # === Step 5: 差分子空间的奇异值衰减 ===
    print(f"\n  === Step 5: Singular value decay ===")
    for l in [0, 6, 12, 21, 27, 33, 35]:
        if l in subspaces:
            sv = subspaces[l]["singular_values"][:10]
            ve = subspaces[l]["variance_explained"][:5]
            print(f"    L{l}: top5 sv={[f'{s:.1f}' for s in sv[:5]]}, "
                  f"var_explained={[f'{v:.3f}' for v in ve[:5]]}")

    out_path = f"tests/glm5_temp/phase106_exp2_{model_name}_subspace_transport.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 3: Gauge-Invariant Quantities — CKA跨层分析
# ============================================================
def exp3_gauge_invariant(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 3: Gauge-Invariant Quantities — CKA/Fisher分析")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # === Step 1: 收集多词的hidden states ===
    print(f"\n  === Step 1: 收集hidden states ===")

    all_pairs = TRANSLATION_PAIRS + EXTRA_PAIRS

    # 每个词在3种prompt下的states
    layer_data = defaultdict(lambda: defaultdict(list))  # l → ptype → [h]

    for zh, en in all_pairs:
        prompts = {
            "zh": f"{zh}是一种",
            "trans_short": f'"{zh}"的英文是',
            "trans_instr": f'请把"{zh}"翻译成英文：',
        }
        for ptype, prompt in prompts.items():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                layer_data[l][ptype].append(h)

    # === Step 2: Linear CKA ===
    def linear_cka(X, Y):
        """Linear CKA between two matrices X, Y of shape (n, d)"""
        # Center
        X = X - np.mean(X, axis=0, keepdims=True)
        Y = Y - np.mean(Y, axis=0, keepdims=True)

        # HSIC
        def hsic(A, B):
            n = A.shape[0]
            K = A @ A.T
            L = B @ B.T
            H = np.eye(n) - np.ones((n, n)) / n
            return np.trace(K @ H @ L @ H) / (n - 1)**2

        hsic_xy = hsic(X, Y)
        hsic_xx = hsic(X, X)
        hsic_yy = hsic(Y, Y)

        if hsic_xx * hsic_yy == 0:
            return 0.0
        return hsic_xy / np.sqrt(hsic_xx * hsic_yy)

    # === Step 3: CKA分析 ===
    print(f"\n  === Step 3: CKA分析 ===")

    results = {}

    # 3.1: 翻译vs中文在同一层的CKA
    cka_zh_trans = {}
    for l in range(n_layers + 1):
        X_zh = np.array(layer_data[l]["zh"], dtype=np.float64)
        X_tr = np.array(layer_data[l]["trans_short"], dtype=np.float64)
        cka = linear_cka(X_zh, X_tr)
        cka_zh_trans[l] = float(cka)

    results["cka_zh_trans_same_layer"] = cka_zh_trans

    # 3.2: 同一prompt类型在相邻层的CKA (层间表示相似度)
    cka_interlayer = {}
    for ptype in ["zh", "trans_short"]:
        cka_chain = []
        for l in range(n_layers):
            X_l = np.array(layer_data[l][ptype], dtype=np.float64)
            X_l1 = np.array(layer_data[l+1][ptype], dtype=np.float64)
            cka = linear_cka(X_l, X_l1)
            cka_chain.append(float(cka))
        cka_interlayer[ptype] = cka_chain

    results["cka_interlayer"] = cka_interlayer

    # 3.3: 翻译差分表示的跨层CKA
    # 差分表示: trans_states - zh_states (每个词的差分)
    diff_data = defaultdict(list)
    for l in range(n_layers + 1):
        for i in range(len(all_pairs)):
            diff = layer_data[l]["trans_short"][i] - layer_data[l]["zh"][i]
            diff_data[l].append(diff)

    cka_diff_chain = []
    for l in range(n_layers):
        D_l = np.array(diff_data[l], dtype=np.float64)
        D_l1 = np.array(diff_data[l+1], dtype=np.float64)
        cka = linear_cka(D_l, D_l1)
        cka_diff_chain.append(float(cka))

    results["cka_diff_interlayer"] = cka_diff_chain

    # === Step 4: Fisher信息矩阵的简化估计 ===
    print(f"\n  === Step 4: Fisher信息估计 ===")

    # 简化: 计算∂logits/∂h的条件数
    # 用W_U作为∂logits/∂h的线性近似
    fisher_results = {}
    for l in range(0, n_layers + 1, 3):
        # 收集一些hidden states
        h_samples = np.array(layer_data[l]["zh"][:20] + layer_data[l]["trans_short"][:20],
                            dtype=np.float64)
        h_ln = np.array([apply_layer_norm(h) for h in h_samples], dtype=np.float64)

        # logits = h_ln @ W_U.T
        # ∂logits/∂h = W_U (对于logit空间)
        # Fisher ≈ E[h * h.T] @ W_U.T @ W_U
        HtH = h_ln.T @ h_ln / len(h_ln)
        WtW = W_U.T @ W_U / W_U.shape[0]
        Fisher = HtH @ WtW

        # 条件数和有效秩
        sv = np.linalg.svd(Fisher, compute_uv=False)
        if sv[0] > 0:
            condition_number = float(sv[0] / (sv[-1] + 1e-10))
            effective_rank = int(np.sum(sv > 0.01 * sv[0]))
        else:
            condition_number = float('inf')
            effective_rank = 0

        fisher_results[l] = {
            "condition_number": float(condition_number),
            "effective_rank": effective_rank,
            "top5_sv": sv[:5].tolist(),
        }

    results["fisher"] = fisher_results

    # === 输出 ===
    print(f"\n  === Summary ===")
    print(f"\n  CKA(zh, trans) same layer:")
    for l in [0, 6, 12, 18, 21, 24, 27, 30, 33, 35]:
        if l <= n_layers:
            print(f"    L{l}: {cka_zh_trans[l]:.4f}")

    print(f"\n  CKA interlayer (trans_short):")
    for l in [0, 5, 10, 15, 20, 25, 30, 33]:
        if l < len(cka_interlayer["trans_short"]):
            print(f"    L{l}→L{l+1}: {cka_interlayer['trans_short'][l]:.4f}")

    print(f"\n  CKA diff interlayer:")
    for l in [0, 5, 10, 15, 20, 25, 30, 33]:
        if l < len(cka_diff_chain):
            print(f"    L{l}→L{l+1}: {cka_diff_chain[l]:.4f}")

    print(f"\n  Fisher info:")
    for l in sorted(fisher_results.keys()):
        fr = fisher_results[l]
        print(f"    L{l}: cond={fr['condition_number']:.0f}, "
              f"eff_rank={fr['effective_rank']}, "
              f"top5_sv=[{', '.join(f'{s:.1f}' for s in fr['top5_sv'][:5])}]")

    out_path = f"tests/glm5_temp/phase106_exp3_{model_name}_gauge_invariant.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 4: Attention-Mediated Transport — token间约束传输
# ============================================================
def exp4_attention_transport(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 4: Attention-Mediated Transport — token间约束传输")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    n_heads = model_info.n_layers  # 假设每层一个head group, 需要实际获取
    # 尝试获取head数
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            n_heads = model.model.layers[0].self_attn.num_heads
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            n_heads = model.transformer.h[0].attn.num_heads
        else:
            n_heads = 20  # Qwen2默认
    except:
        n_heads = 20

    print(f"  Detected n_heads={n_heads}")

    # === Step 1: 收集多token位置的hidden states和attention ===
    print(f"\n  === Step 1: 收集多token信息 ===")

    test_pairs = TRANSLATION_PAIRS[:10]
    results = {}

    for zh, en in test_pairs:
        word_result = {"en": en, "zh": zh}

        # 翻译prompt
        trans_prompt = f'"{zh}"的英文是'
        inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)
        n_tokens = inputs["input_ids"].shape[1]

        # 获取hidden states (Qwen3不支持output_attentions)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)

        # 1. 各token位置的hidden states
        token_states = {}
        for l in range(n_layers + 1):
            # hidden_states[l] shape: (1, n_tokens, d_model)
            states_l = outputs.hidden_states[l][0].float().cpu().numpy()  # (n_tokens, d_model)
            token_states[l] = states_l

        # 2. Attention weights (如果模型支持)
        attn_weights = {}
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            for l in range(len(outputs.attentions)):
                attn_l = outputs.attentions[l][0].float().cpu().numpy()
                attn_weights[l] = attn_l

        # 3. 最后token对各source token的attention
        last_token_attn = {}
        if attn_weights:
            for l, attn in attn_weights.items():
                last_to_all = attn[:, -1, :]
                last_token_attn[l] = {
                    "mean": np.mean(last_to_all, axis=0).tolist(),
                    "max_head": np.max(last_to_all, axis=0).tolist(),
                    "per_head": last_to_all.tolist(),
                }

        word_result["n_tokens"] = n_tokens
        word_result["token_states"] = {l: s.tolist() for l, s in token_states.items()}
        word_result["last_token_attn"] = last_token_attn

        # 4. 跨token约束传播分析
        # 在每层，计算: 最后token的h与其他token的h的cosine相似度
        cross_token_sim = {}
        for l in range(n_layers + 1):
            h_last = token_states[l][-1]  # 最后token的hidden state
            sims = []
            for t in range(n_tokens - 1):
                h_t = token_states[l][t]
                cos = np.dot(h_last, h_t) / (np.linalg.norm(h_last) * np.linalg.norm(h_t) + 1e-10)
                sims.append(float(cos))
            cross_token_sim[l] = sims

        word_result["cross_token_cosine"] = cross_token_sim

        # 5. 关键分析: 哪个token位置最影响最后token的margin
        W_U = get_W_U(model).astype(np.float64)
        en_id = get_token_id(tokenizer, en)
        zh_id = get_token_id(tokenizer, zh)
        if en_id is not None and zh_id is not None:
            w_en = W_U[en_id]
            w_zh = W_U[zh_id]

            margin_per_token = {}
            for l in range(n_layers + 1):
                margins = []
                for t in range(n_tokens):
                    h_t_ln = apply_layer_norm(token_states[l][t].astype(np.float64))
                    margin = np.dot(h_t_ln, w_en) - np.dot(h_t_ln, w_zh)
                    margins.append(float(margin))
                margin_per_token[l] = margins

            word_result["margin_per_token"] = margin_per_token

        results[f"{zh}_{en}"] = word_result
        print(f"  {zh}→{en}: {n_tokens} tokens, attn layers={len(attn_weights)}")

    # === Step 2: 汇总 ===
    print(f"\n  === Summary ===")

    # 每个词的margin演化(最后token位置)
    print(f"\n  Last token margin (trans prompt):")
    for key, wr in results.items():
        if "margin_per_token" in wr:
            mpt = wr["margin_per_token"]
            l0_m = mpt.get(0, mpt.get("0", [0]))[-1] if mpt else 0
            l33_m = mpt.get(33, mpt.get("33", [0]))[-1] if mpt else 0
            l35_m = mpt.get(35, mpt.get("35", [0]))[-1] if mpt else 0
            print(f"    {key}: L0={l0_m:.3f}, L33={l33_m:.3f}, L35={l35_m:.3f}")

    # Attention汇聚: 最后token最关注哪个位置?
    print(f"\n  Attention to last token (which source positions matter):")
    for key, wr in results.items():
        if "last_token_attn" in wr and len(wr["last_token_attn"]) > 0:
            for l_key in [21, 27, 33, "21", "27", "33"]:
                if l_key in wr["last_token_attn"]:
                    mean_attn = wr["last_token_attn"][l_key]["mean"]
                    top_pos = np.argmax(mean_attn[:-1]) if len(mean_attn) > 1 else 0
                    print(f"    {key} L{l_key}: top_src_pos={top_pos}, attn={mean_attn[top_pos]:.3f}")
                    break

    out_path = f"tests/glm5_temp/phase106_exp4_{model_name}_attention_transport.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "ds7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    if args.exp == 1:
        exp1_probe_invariance(args)
    elif args.exp == 2:
        exp2_subspace_transport(args)
    elif args.exp == 3:
        exp3_gauge_invariant(args)
    elif args.exp == 4:
        exp4_attention_transport(args)

    gc.collect()
    torch.cuda.empty_cache()
