"""
Phase 203: Geometric Flow Analysis — Jacobian Spectrum & Output-Sensitivity Decomposition
============================================================================================

理论框架 (基于Phase 202关键修正):

  Phase 202的核心错误:
  1. 低参与比(d_eff)≠低维流形 — d_eff只衡量方差集中度(各向异性), 不能声称"几何维度低"
  2. Transformer不是经典动力系统 — 固定层数, 无"attractor" — 应用"transport corridor"
  3. 标点触发只是embedding+训练统计 — 不是"basin boundary"
  4. "协议投影"是优化结果不是目的论 — 训练目标强制最后层靠近rowspace(W_U)

  ★核心修正★:
  Transformer是有限深度非线性映射链 (finite-depth nonlinear transport system):
    h_{l+1} = h_l + F_l(h_l, p)
  其中 p 是 prompt-induced constraint field (约束场)

  ★真正该研究的数学对象★:
  1. Jacobian几何: J_l = ∂h_{l+1}/∂h_l — 最核心对象
     决定: 稳定性, 放大率, 轨迹压缩, 模式偏转
  2. 零空间动力学: h_⊥ ∈ ker(W_U) — 真正推理发生的地方
     但ker(W_U)≈{0}, 所以用"弱输出耦合方向"替代
  3. 曲率场: ||J_l|| 和特征值谱 — 轨迹压缩/扩展
  4. Attention核几何: K(x_i, x_j) = exp(q_i^T k_j) — 动态核系统

  ★术语修正★ (Phase 202 → Phase 203):
  "attractor basin" → "transport corridor"
  "basin boundary" → "regime transition"
  "bifurcation" → "sensitivity change"
  "protocol projection" → "language projection" (优化结果, 非目的论)
  "dark matter" → "weakly-coupled computation" (弱输出耦合计算)

Phase 203实验 (4个子实验):

Exp1: Cumulative Transport Spectrum (累计传输谱分析)
  核心: 从输入到每层, 扰动如何被传输/放大/压缩?
  方法: 输入层注入N_probe个随机扰动, 测量每层的扰动响应
  SVD给出"累计Jacobian" T_l = J_0×J_1×...×J_{l-1} 的近似奇异值谱
  关键指标:
  - 谱半径增长: σ_max(l) vs l — 扰动是被放大还是压缩?
  - 各向异性演化: participation ratio vs l — 传输是越来越"窄"还是"宽"?
  - 模式差异: CoT/normal/translation的谱半径不同吗?

Exp2: Output-Sensitivity Decomposition (输出-敏感度分解)
  核心: hidden state的"输出耦合"和"弱耦合"分量分布
  方法: SVD of W_U → 按奇异值大小分方向
  - 高σ方向: "强耦合" (输出敏感度高, 微小变化导致logits大变)
  - 低σ方向: "弱耦合" (输出敏感度低, 变化对logits影响小)
  分解 h_l = h_strong + h_weak, 追踪各层能量分布
  速度分解: Δh_l = Δh_strong + Δh_weak — 哪些层在"强耦合"方向贡献大?

Exp3: Per-layer Growth Rate (逐层增长率, 从Exp1数据推导)
  核心: 每层的J_l的谱半径估计
  方法: σ_max(J_l) ≈ σ_max(T_{l+1}) / σ_max(T_l)
  关键: 哪些层在"压缩"轨迹(σ<1), 哪些在"扩展"(σ>1)?

Exp4: Attention Kernel Geometry (注意力核几何)
  核心: attention是在做"硬路由"还是"软平滑"?
  方法: 提取attention pattern, 计算entropy, effective rank, concentration
  如果entropy低 → 硬路由 (少量position被关注)
  如果entropy高 → 软平滑 (所有position被均匀关注)

数据量 (加大关键实验):
  - Exp1 (最关键): Qwen3=20句×3mode×40probe, GLM4/DS7B=10句×3mode×40probe
  - Exp2: Qwen3=30句×8mode, GLM4/DS7B=15句×8mode
  - Exp3: 从Exp1数据推导, 无额外计算
  - Exp4: Qwen3=10句×3mode, GLM4/DS7B=5句×3mode

模型加载: bf16 + device_map="auto" + eager attention
"""

import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent / "tests"))

import gc, time, json, math, warnings
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from model_demo_bf16 import load_model_bf16
from model_utils import get_model_info, release_model, get_layers, get_W_U

warnings.filterwarnings('ignore')


# ========================================================================
# 基础句子
# ========================================================================
BASE_SENTENCES = [
    "The cat chases the dog",
    "The teacher helps the student",
    "The leader guides the team",
    "The doctor treats the patient",
    "The chef cooks the meal",
    "The writer drafts the letter",
    "The farmer plants the seed",
    "The artist paints the portrait",
    "The scientist discovers the element",
    "The engineer designs the bridge",
    "The judge delivers the verdict",
    "The soldier defends the fortress",
    "The musician composes the symphony",
    "The pilot flies the airplane",
    "The author writes the novel",
    "The builder constructs the house",
    "The driver operates the vehicle",
    "The guard protects the treasure",
    "The merchant trades the goods",
    "The hunter tracks the prey",
    "The baker prepares the bread",
    "The sailor navigates the ship",
    "The programmer writes the code",
    "The mechanic repairs the engine",
    "The librarian organizes the books",
    "The photographer captures the image",
    "The detective solves the mystery",
    "The architect plans the building",
    "The translator converts the text",
    "The analyst evaluates the data",
]

LITE_SENTENCES = BASE_SENTENCES[:15]

# Exp1专用: 较少句子但更多probes
EXP1_SENTENCES_Q = BASE_SENTENCES[:20]
EXP1_SENTENCES_L = BASE_SENTENCES[:10]


# ========================================================================
# Mode Prompts — 修正后术语
# ========================================================================
CORE_MODES = ["normal", "cot", "translation"]
ALL_MODES = ["normal", "qa", "cot", "conditional", "translation", "coding", "negation", "narrative"]

MODE_PROMPTS = {
    "normal": lambda b: b,
    "qa": lambda b: "Does " + b[0].lower() + b[1:] + "?",
    "cot": lambda b: "Problem: " + b[0].lower() + b[1:] + ". Think step by step.",
    "conditional": lambda b: "If " + b[0].lower() + b[1:] + ", then",
    "translation": lambda b: "Translate to Chinese: " + b,
    "coding": lambda b: "Write code for: " + b[0].lower() + b[1:],
    "negation": lambda b: "It is false that " + b[0].lower() + b[1:],
    "narrative": lambda b: "Once upon a time, " + b[0].lower() + b[1:],
}


# ========================================================================
# Utility Functions
# ========================================================================
def compute_cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def compute_cosine_dist(v1, v2):
    return 1.0 - compute_cosine_sim(v1, v2)


def log_progress(exp_name, current, total, t_start, extra=""):
    elapsed = time.time() - t_start
    gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  [{exp_name}] {current}/{total} ({elapsed:.0f}s) GPU={gpu:.2f}GB {extra}",
          flush=True)


def extract_full_trajectory(model, tokenizer, device, text, n_layers, max_len=96):
    """提取完整的层轨迹: 每层last token的hidden state"""
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
    with torch.no_grad():
        try:
            out = model(input_ids=ids, output_hidden_states=True)
        except Exception:
            attn_mask = torch.ones_like(ids)
            out = model(input_ids=ids, attention_mask=attn_mask, output_hidden_states=True)

    hs = out.hidden_states
    trajectory = []
    for li, h in enumerate(hs):
        if li > n_layers:
            break
        trajectory.append(h[0, -1, :].detach().float().cpu().numpy())

    del out, hs
    return np.array(trajectory)


# ========================================================================
# EXP1: Cumulative Transport Spectrum
# ========================================================================
def exp1_cumulative_transport(model, tokenizer, device, info, sentences, n_probe=40):
    """
    核心: 估计"累计Jacobian" T_l = J_0 × J_1 × ... × J_{l-1} 的奇异值谱

    方法:
    1. 运行基准前向 → 得到所有层的 h_l^0
    2. 对每个probe方向 δ_i:
       a. 加δ_i到input embedding
       b. 运行前向 → 得到所有层的 h_l^i
       c. Δ_l^i = h_l^i - h_l^0
    3. 在每层, 堆叠扰动响应: D_l = [Δ_l^1, ..., Δ_l^N]
    4. SVD of D_l → 近似top奇异值

    关键指标:
    - σ_max(l): 累计传输的谱半径 — 扰动被放大还是压缩?
    - participation_ratio(l): 各向异性 — 传输在多少维方向上发生?
    - growth_rate(l) = σ_max(l+1)/σ_max(l): 逐层谱半径
    """
    print("\n" + "="*60)
    print("Exp1: Cumulative Transport Spectrum")
    print("="*60)

    n_layers = info.n_layers
    d_model = info.d_model
    modes = CORE_MODES  # ["normal", "cot", "translation"]

    print(f"  Modes: {modes}")
    print(f"  n_layers: {n_layers}, d_model: {d_model}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Probes per sentence-mode: {n_probe}")

    # 扰动尺度: 相对于embedding norm
    epsilon = 0.01  # 1% of embedding norm

    # 存储结果: {mode: {layer_idx: singular_values, participation_ratios, ...}}
    results = {mode: {} for mode in modes}

    # 采样层 (不需要每层都分析, 节省计算)
    sample_layers = sorted(set(
        [0, 1] +
        list(range(0, n_layers, max(1, n_layers // 12))) +
        [n_layers - 2, n_layers - 1]
    ))

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        t_start_mode = time.time()

        # 收集所有句子的扰动响应
        # 对每层, D_l = (d_model, n_sents * n_probe) 的扰动响应矩阵
        all_delta_by_layer = defaultdict(list)  # {layer: [delta_vectors]}
        base_norm_avg = 0.0

        for si, base in enumerate(sentences):
            text = MODE_PROMPTS[mode](base)

            # Tokenize
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)

            # 基准前向
            with torch.no_grad():
                embed_layer = model.get_input_embeddings()
                base_embed = embed_layer(ids)  # (1, seq_len, d_model)

            base_norm = float(base_embed[0, -1, :].norm())
            base_norm_avg += base_norm

            # 基准hidden states
            with torch.no_grad():
                try:
                    out_base = model(input_ids=ids, output_hidden_states=True)
                except Exception:
                    attn_mask = torch.ones_like(ids)
                    out_base = model(input_ids=ids, attention_mask=attn_mask,
                                     output_hidden_states=True)

            base_states = {}
            for li in range(len(out_base.hidden_states)):
                if li <= n_layers:
                    base_states[li] = out_base.hidden_states[li][0, -1, :].detach().float().cpu().numpy()

            del out_base

            # N_probe个随机扰动
            for pi in range(n_probe):
                # 随机扰动方向 (在d_model空间)
                delta = torch.randn(d_model, device=base_embed.device, dtype=base_embed.dtype)
                delta = delta / delta.norm() * epsilon * base_norm  # ε * ||h_0||

                # 扰动embedding的last token位置
                pert_embed = base_embed.clone()
                pert_embed[0, -1, :] += delta

                # 扰动前向
                with torch.no_grad():
                    try:
                        out_pert = model(inputs_embeds=pert_embed,
                                         output_hidden_states=True)
                    except Exception:
                        continue

                # 收集每层的扰动响应
                for li in sample_layers:
                    if li < len(out_pert.hidden_states):
                        pert_h = out_pert.hidden_states[li][0, -1, :].detach().float().cpu().numpy()
                        if li in base_states:
                            delta_h = pert_h - base_states[li]
                            all_delta_by_layer[li].append(delta_h)

                del out_pert, pert_embed
                gc.collect()

            # 日志
            if (si + 1) % 5 == 0:
                elapsed = time.time() - t_start_mode
                gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"  [Exp1-{mode}] {si+1}/{len(sentences)} ({elapsed:.0f}s) GPU={gpu:.2f}GB",
                      flush=True)

        base_norm_avg /= max(len(sentences), 1)

        # === 分析每层的奇异值谱 ===
        print(f"\n  Cumulative Transport Spectrum for '{mode}':")
        print(f"  {'Layer':>6} {'σ_max':>10} {'σ_min':>10} {'σ_ratio':>10} "
              f"{'d_eff':>8} {'cond#':>10} {'Interpretation':>25}")

        for li in sample_layers:
            deltas = all_delta_by_layer[li]
            if len(deltas) < 5:
                continue

            D = np.array(deltas).T  # (d_model, n_probes_total)

            # SVD of D → 近似top奇异值
            try:
                # 经济型SVD
                n_svd = min(D.shape[1], D.shape[0], 200)
                U, S, Vt = np.linalg.svd(D[:, :n_svd], full_matrices=False)
            except Exception:
                continue

            sigma_max = float(S[0]) if len(S) > 0 else 0
            sigma_min = float(S[-1]) if len(S) > 0 else 0
            sigma_ratio = sigma_max / max(sigma_min, 1e-20)

            # 参与比 (各向异性度量)
            total_energy = np.sum(S ** 2)
            if total_energy > 1e-20:
                pr = (np.sum(S ** 2))**2 / np.sum(S ** 4)
            else:
                pr = 0

            # 条件数
            cond = sigma_max / max(sigma_min, 1e-20)

            # 解释
            if sigma_max > 1.0:
                interp = "EXPANSION (σ>1)"
            elif sigma_max > 0.1:
                interp = "TRANSPORT (σ~1)"
            else:
                interp = "COMPRESSION (σ<0.1)"

            print(f"  {li:6d} {sigma_max:10.6f} {sigma_min:10.6f} {sigma_ratio:10.2f} "
                  f"{pr:8.1f} {cond:10.2f} {interp:>25}")

            results[mode][li] = {
                'sigma_max': sigma_max,
                'sigma_min': sigma_min,
                'sigma_ratio': sigma_ratio,
                'participation_ratio': float(pr),
                'condition_number': float(cond),
                'top_singular_values': [float(s) for s in S[:20]],
            }

        # 清理
        del all_delta_by_layer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === 跨模式比较 ===
    print("\n--- Exp1 Summary: Cross-Mode Transport Comparison ---")

    # 1. 谱半径比较
    print(f"\n  Spectral Radius (σ_max) by Mode and Layer:")
    print(f"  {'Layer':>6}", end="")
    for mode in modes:
        print(f" {mode[:10]:>10}", end="")
    print()

    for li in sample_layers:
        print(f"  {li:6d}", end="")
        for mode in modes:
            if li in results[mode]:
                print(f" {results[mode][li]['sigma_max']:10.6f}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()

    # 2. 各向异性比较
    print(f"\n  Anisotropy (participation ratio) by Mode and Layer:")
    print(f"  {'Layer':>6}", end="")
    for mode in modes:
        print(f" {mode[:10]:>10}", end="")
    print()

    for li in sample_layers:
        print(f"  {li:6d}", end="")
        for mode in modes:
            if li in results[mode]:
                print(f" {results[mode][li]['participation_ratio']:10.1f}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()

    # 3. 关键判断: CoT是否降低各向异性?
    mid_layer = n_layers // 2
    if all(mid_layer in results[m] for m in modes):
        print(f"\n  >>> Anisotropy at mid-layer (L{mid_layer}):")
        for mode in modes:
            pr = results[mode][mid_layer]['participation_ratio']
            sigma = results[mode][mid_layer]['sigma_max']
            print(f"      {mode}: d_eff={pr:.1f}, σ_max={sigma:.6f}")

    return results


# ========================================================================
# EXP2: Output-Sensitivity Decomposition
# ========================================================================
def exp2_output_sensitivity(model, tokenizer, device, info, sentences):
    """
    核心: 分解hidden state的"强输出耦合"和"弱输出耦合"分量

    理论基础:
    W_U ∈ R^{V × d}, SVD: W_U = U Σ V^T
    V的列(右奇异向量)构成R^d的一组基
    σ_i = 第i个奇异值 = 方向v_i的"输出敏感度"
    大σ_i → 微小hidden state变化导致大logit变化
    小σ_i → 大hidden state变化导致小logit变化

    分解:
    h = h_strong + h_weak
    h_strong = V_high @ V_high^T @ h (在top-k高σ方向上的投影)
    h_weak = h - h_strong (在其余方向上的投影)

    注意: 这不是"nullspace分解"(ker(W_U)≈{0}), 而是"输出敏感度分层"
    """
    print("\n" + "="*60)
    print("Exp2: Output-Sensitivity Decomposition")
    print("="*60)

    n_layers = info.n_layers
    d_model = info.d_model
    mid_layer = n_layers // 2
    modes = ALL_MODES

    print(f"  Modes: {modes}")
    print(f"  n_layers: {n_layers}, d_model: {d_model}")
    print(f"  Sentences: {len(sentences)}")

    # 加载W_U并计算SVD
    print("  Loading W_U and computing SVD...")
    W_U = get_W_U(model, info.name if hasattr(info, 'name') else None)
    print(f"  W_U shape: {W_U.shape}")

    # SVD of W_U (经济型, 只需要右奇异向量V)
    print("  Computing SVD of W_U...")
    W_U_f32 = W_U.astype(np.float32)
    # 用截断SVD (比完整SVD快得多)
    from scipy.sparse.linalg import svds
    k_svd = min(500, min(W_U_f32.shape) - 2)
    U_wu, sigma_wu, Vt_wu = svds(W_U_f32, k=k_svd)
    # Vt_wu shape: (k, d_model) — 行是右奇异向量
    V_high = Vt_wu.T  # (d_model, k) — 列是右奇异向量

    print(f"  W_U SVD: top-5 σ = {sigma_wu[-5:][::-1].tolist()}")
    print(f"  W_U SVD: bottom-5 σ = {sigma_wu[:5].tolist()}")
    print(f"  Condition number (σ_max/σ_min) = {sigma_wu[-1]/max(sigma_wu[-1], 1e-20):.2f}")

    # 确定分层阈值: 用cumulative energy找"拐点"
    sigma_sorted = np.sort(sigma_wu)[::-1]
    cum_energy = np.cumsum(sigma_sorted ** 2) / np.sum(sigma_sorted ** 2)
    # 找90%能量对应的维度数
    n_90 = np.searchsorted(cum_energy, 0.9) + 1
    n_50 = np.searchsorted(cum_energy, 0.5) + 1

    print(f"  W_U energy: 50% in top-{n_50} directions, 90% in top-{n_90} directions")

    # 定义"强耦合"和"弱耦合"的分割点
    # 使用90%能量阈值
    n_strong = n_90
    V_strong = V_high[:, :n_strong]  # (d_model, n_strong) — top奇异向量
    V_weak = V_high[:, n_strong:]    # (d_model, k-n_strong) — 其余奇异向量

    print(f"  Strong-coupling directions: top-{n_strong} (capture 90% of output energy)")
    print(f"  Weak-coupling directions: remaining {k_svd - n_strong}")

    # 收集每层每个mode的分解结果
    # {mode: {layer: [strong_ratios, weak_ratios, velocity_strong_ratios, ...]}}
    decomp_data = {mode: {li: {'strong': [], 'weak': [],
                                'vel_strong': [], 'vel_weak': []}
                          for li in range(n_layers + 1)}
                   for mode in modes}

    t_start = time.time()
    total = len(sentences) * len(modes)

    for si, base in enumerate(sentences):
        for mode in modes:
            text = MODE_PROMPTS[mode](base)
            traj = extract_full_trajectory(model, tokenizer, device, text, n_layers)

            for li in range(min(traj.shape[0], n_layers + 1)):
                h = traj[li]

                # 分解 h = h_strong + h_weak
                proj_strong = V_strong @ (V_strong.T @ h)
                proj_weak = h - proj_strong

                energy_total = np.sum(h ** 2)
                energy_strong = np.sum(proj_strong ** 2)
                energy_weak = np.sum(proj_weak ** 2)

                strong_ratio = energy_strong / max(energy_total, 1e-20)
                weak_ratio = energy_weak / max(energy_total, 1e-20)

                decomp_data[mode][li]['strong'].append(strong_ratio)
                decomp_data[mode][li]['weak'].append(weak_ratio)

                # 速度分解: Δh = h_{l+1} - h_l
                if li > 0:
                    vel = traj[li] - traj[li - 1]
                    vel_strong = V_strong @ (V_strong.T @ vel)
                    vel_weak = vel - vel_strong

                    vel_total = np.sum(vel ** 2)
                    vel_strong_ratio = np.sum(vel_strong ** 2) / max(vel_total, 1e-20)
                    vel_weak_ratio = np.sum(vel_weak ** 2) / max(vel_total, 1e-20)

                    decomp_data[mode][li]['vel_strong'].append(vel_strong_ratio)
                    decomp_data[mode][li]['vel_weak'].append(vel_weak_ratio)

        if (si + 1) % 5 == 0:
            log_progress("Exp2", si + 1, len(sentences), t_start,
                         f"({(si + 1) * len(modes)}/{total})")

    # === 分析 ===
    print("\n--- Exp2A: Strong-Coupling Energy Ratio by Layer (all modes) ---")
    print("  (How much of hidden state energy is in 'output-important' directions?)")

    sample_layers = sorted(set(
        [0, n_layers // 4, mid_layer, 3 * n_layers // 4, n_layers - 1]
    ))

    print(f"  {'Layer':>6}", end="")
    for mode in modes:
        print(f" {mode[:7]:>7}", end="")
    print("  Note")

    for li in sample_layers:
        if li > n_layers:
            continue
        print(f"  {li:6d}", end="")
        for mode in modes:
            strongs = decomp_data[mode][li]['strong']
            avg = float(np.mean(strongs)) if strongs else 0
            print(f" {avg:7.4f}", end="")
        if li == 0:
            print("  (embedding)")
        elif li == mid_layer:
            print("  (mid)")
        elif li == n_layers - 1:
            print("  (final)")
        else:
            print()

    # === Exp2B: 速度的强/弱耦合分解 ===
    print("\n--- Exp2B: Velocity Strong-Coupling Ratio by Layer ---")
    print("  (How much of each layer's 'computation' is in output-important directions?)")

    print(f"  {'Layer':>6}", end="")
    for mode in modes:
        print(f" {mode[:7]:>7}", end="")
    print()

    for li in sample_layers:
        if li > n_layers or li == 0:
            continue
        print(f"  {li:6d}", end="")
        for mode in modes:
            vel_strongs = decomp_data[mode][li]['vel_strong']
            avg = float(np.mean(vel_strongs)) if vel_strongs else 0
            print(f" {avg:7.4f}", end="")
        print()

    # === Exp2C: 弱耦合能量的层分布 ===
    print("\n--- Exp2C: Weak-Coupling Energy Distribution ---")
    print("  (Where is the 'weakly-coupled computation' concentrated?)")

    for mode in ["normal", "cot", "translation"]:
        print(f"\n  Mode: {mode}")
        weak_profile = []
        for li in range(n_layers + 1):
            weaks = decomp_data[mode][li]['weak']
            avg = float(np.mean(weaks)) if weaks else 0
            weak_profile.append(avg)

        # 找峰值
        if weak_profile:
            peak_layer = np.argmax(weak_profile)
            peak_val = weak_profile[peak_layer]
            min_val = min(weak_profile)
            min_layer = weak_profile.index(min_val)
            print(f"    Peak weak-coupling at L{peak_layer} ({peak_val:.4f})")
            print(f"    Minimum at L{min_layer} ({min_val:.4f})")
            print(f"    Layer profile: ", end="")
            for li in range(0, n_layers + 1, max(1, n_layers // 8)):
                print(f"L{li}={weak_profile[li]:.3f} ", end="")
            print()

    # === 综合判断 ===
    print("\n--- Exp2 Overall: Output-Sensitivity Summary ---")

    # 计算"弱耦合峰值层"在哪个区域
    for mode in modes:
        weak_profile = []
        for li in range(n_layers + 1):
            weaks = decomp_data[mode][li]['weak']
            avg = float(np.mean(weaks)) if weaks else 0
            weak_profile.append(avg)

        if weak_profile:
            peak_layer = np.argmax(weak_profile)
            if peak_layer < n_layers // 3:
                region = "EARLY"
            elif peak_layer < 2 * n_layers // 3:
                region = "MIDDLE"
            else:
                region = "LATE"

            # 速度分解: 哪个层在"弱耦合"方向上贡献最大?
            vel_weak_profile = []
            for li in range(1, n_layers + 1):
                vel_weaks = decomp_data[mode][li]['vel_weak']
                avg = float(np.mean(vel_weaks)) if vel_weaks else 0
                vel_weak_profile.append(avg)

            vel_peak = np.argmax(vel_weak_profile) + 1 if vel_weak_profile else 0

            print(f"  {mode:<15} weak_energy_peak=L{peak_layer}({region}), "
                  f"vel_weak_peak=L{vel_peak}")

    # 关键判断
    print("\n  Key findings:")
    for mode in ["normal", "cot", "translation"]:
        mid_weaks = decomp_data[mode][mid_layer]['weak']
        late_weaks = decomp_data[mode][n_layers - 1]['weak']
        mid_avg = float(np.mean(mid_weaks)) if mid_weaks else 0
        late_avg = float(np.mean(late_weaks)) if late_weaks else 0

        if mid_avg > late_avg + 0.05:
            print(f"  {mode}: Mid layers have MORE weak-coupling than late layers "
                  f"({mid_avg:.4f} vs {late_avg:.4f})")
            print(f"    → Optimization pressure forces late layers toward output-coupling")
        else:
            print(f"  {mode}: Weak-coupling similar at mid and late "
                  f"({mid_avg:.4f} vs {late_avg:.4f})")

    return {
        'n_strong': n_strong,
        'n_90': n_90,
        'decomp_data': {mode: {
            li: {
                'strong_avg': float(np.mean(d['strong'])) if d['strong'] else 0,
                'weak_avg': float(np.mean(d['weak'])) if d['weak'] else 0,
                'vel_strong_avg': float(np.mean(d['vel_strong'])) if d['vel_strong'] else 0,
                'vel_weak_avg': float(np.mean(d['vel_weak'])) if d['vel_weak'] else 0,
            }
            for li, d in layer_data.items()
        } for mode, layer_data in decomp_data.items()},
    }


# ========================================================================
# EXP3: Per-layer Growth Rate (from Exp1 data)
# ========================================================================
def exp3_per_layer_growth(transport_results, n_layers, modes):
    """
    从Exp1的累计传输数据推导逐层增长率

    核心逻辑:
    T_l = J_0 × J_1 × ... × J_{l-1} (累计传输矩阵)
    σ_max(T_{l+1}) ≈ σ_max(T_l) × σ_max(J_l)
    所以: σ_max(J_l) ≈ σ_max(T_{l+1}) / σ_max(T_l)

    如果 σ_max(J_l) > 1 → 第l层在"扩展"扰动 (expansion)
    如果 σ_max(J_l) < 1 → 第l层在"压缩"扰动 (compression)
    如果 σ_max(J_l) ≈ 1 → 第l层在"等幅传输" (transport)

    类似地分析各向异性(参与比)的变化:
    Δpr(l) = pr(T_{l+1}) - pr(T_l) → 传输变得更窄(Δpr<0)还是更宽(Δpr>0)
    """
    print("\n" + "="*60)
    print("Exp3: Per-layer Growth Rate (from Exp1 data)")
    print("="*60)

    results = {}

    for mode in modes:
        if mode not in transport_results:
            continue

        mode_data = transport_results[mode]
        layers = sorted(mode_data.keys())

        print(f"\n  Mode: {mode}")
        print(f"  {'Layer':>6} {'σ_max(T_l)':>12} {'Growth Rate':>12} "
              f"{'pr(T_l)':>10} {'Δpr':>8} {'Type':>15}")

        prev_sigma = None
        prev_pr = None

        layer_analysis = []

        for li in layers:
            sigma_max = mode_data[li]['sigma_max']
            pr = mode_data[li]['participation_ratio']

            # 增长率
            growth_rate = sigma_max / prev_sigma if prev_sigma and prev_sigma > 1e-20 else None

            # 各向异性变化
            delta_pr = pr - prev_pr if prev_pr is not None else None

            # 分类
            if growth_rate is not None:
                if growth_rate > 1.5:
                    gtype = "EXPANSION"
                elif growth_rate > 0.7:
                    gtype = "TRANSPORT"
                else:
                    gtype = "COMPRESSION"
            else:
                gtype = "—"

            growth_str = f"{growth_rate:.4f}" if growth_rate is not None else "—"
            delta_pr_str = f"{delta_pr:+.2f}" if delta_pr is not None else "—"

            print(f"  {li:6d} {sigma_max:12.6f} {growth_str:>12} "
                  f"{pr:10.1f} {delta_pr_str:>8} {gtype:>15}")

            layer_analysis.append({
                'layer': li,
                'sigma_max': sigma_max,
                'growth_rate': growth_rate,
                'pr': pr,
                'delta_pr': delta_pr,
                'type': gtype,
            })

            prev_sigma = sigma_max
            prev_pr = pr

        results[mode] = layer_analysis

        # 总结
        if layer_analysis:
            growth_rates = [la['growth_rate'] for la in layer_analysis if la['growth_rate'] is not None]
            if growth_rates:
                avg_growth = np.mean(growth_rates)
                expansion_layers = sum(1 for g in growth_rates if g > 1.5)
                compression_layers = sum(1 for g in growth_rates if g < 0.7)

                print(f"\n  Summary for '{mode}':")
                print(f"    Avg growth rate: {avg_growth:.4f}")
                print(f"    Expansion layers (σ>1.5): {expansion_layers}")
                print(f"    Compression layers (σ<0.7): {compression_layers}")
                print(f"    Transport layers (0.7<σ<1.5): {len(growth_rates) - expansion_layers - compression_layers}")

    # === 跨模式比较 ===
    print("\n--- Exp3 Summary: Cross-Mode Growth Rate Comparison ---")
    print("  (Per-layer spectral radius of Jacobian J_l)")

    for mode in modes:
        if mode in results and results[mode]:
            growth_rates = [la['growth_rate'] for la in results[mode]
                           if la['growth_rate'] is not None]
            if growth_rates:
                print(f"  {mode}: avg growth = {np.mean(growth_rates):.4f}, "
                      f"range = [{min(growth_rates):.4f}, {max(growth_rates):.4f}]")

    return results


# ========================================================================
# EXP4: Attention Kernel Geometry
# ========================================================================
def exp4_attention_kernel(model, tokenizer, device, info, sentences):
    """
    核心: 分析attention pattern的核几何

    Attention = softmax(Q K^T / √d) = kernel matrix K(x_i, x_j) = exp(q_i^T k_j / √d)

    对每个head在采样层上:
    1. Entropy: H = -Σ_j A_{ij} log A_{ij} (平均化)
       低entropy → "硬路由" (少量position被关注)
       高entropy → "软平滑" (所有position被均匀关注)
    2. Effective rank: A的参与比
       低rank → 注意力集中在少量位置关系上
       高rank → 注意力分散在很多位置关系上
    3. Concentration: top-1和top-3位置的注意力占比
       高concentration → 类似"指针"行为
       低concentration → 类似"平均池化"行为

    关键问题: attention是"硬路由"还是"软平滑"?
    """
    print("\n" + "="*60)
    print("Exp4: Attention Kernel Geometry")
    print("="*60)

    n_layers = info.n_layers
    d_model = info.d_model
    modes = CORE_MODES  # ["normal", "cot", "translation"]

    print(f"  Modes: {modes}")
    print(f"  Sentences: {len(sentences)}")

    # 采样层
    sample_layers = sorted(set(
        [0, 1, 2] +
        list(range(0, n_layers, max(1, n_layers // 6))) +
        [n_layers - 3, n_layers - 2, n_layers - 1]
    ))
    sample_layers = [li for li in sample_layers if li < n_layers]

    # 收集数据: {mode: {layer: {head: [entropy, concentration, ...]}}}
    attn_data = {mode: {li: {} for li in sample_layers} for mode in modes}

    t_start = time.time()

    for si, base in enumerate(sentences):
        for mode in modes:
            text = MODE_PROMPTS[mode](base)
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=96).input_ids.to(device)

            with torch.no_grad():
                try:
                    out = model(input_ids=ids, output_attentions=True)
                except Exception:
                    continue

            attentions = out.attentions  # tuple of (1, n_heads, seq_len, seq_len)

            if attentions is None:
                continue

            for li_idx, li in enumerate(sample_layers):
                if li >= len(attentions):
                    continue

                attn = attentions[li]  # (1, n_heads, seq_len, seq_len)
                attn_np = attn[0].float().cpu().numpy()  # (n_heads, seq_len, seq_len)
                n_heads = attn_np.shape[0]
                seq_len = attn_np.shape[1]

                for hi in range(n_heads):
                    A = attn_np[hi]  # (seq_len, seq_len) — 注意力矩阵

                    # 1. Entropy (平均化, 每行是一个分布)
                    entropies = []
                    for row in A:
                        row_safe = row[row > 1e-10]
                        if len(row_safe) > 0:
                            h = -np.sum(row_safe * np.log(row_safe))
                            entropies.append(h)
                    avg_entropy = float(np.mean(entropies)) if entropies else 0

                    # 2. Concentration (top-1和top-3占比)
                    top1_fracs = []
                    top3_fracs = []
                    for row in A:
                        sorted_vals = np.sort(row)[::-1]
                        top1_fracs.append(float(sorted_vals[0]))
                        top3_fracs.append(float(np.sum(sorted_vals[:3])))
                    avg_top1 = float(np.mean(top1_fracs))
                    avg_top3 = float(np.mean(top3_fracs))

                    # 3. Effective rank (参与比 of A)
                    try:
                        U_a, S_a, Vt_a = np.linalg.svd(A, full_matrices=False)
                        total_sq = np.sum(S_a ** 2)
                        if total_sq > 1e-20:
                            eff_rank = (total_sq) ** 2 / np.sum(S_a ** 4)
                        else:
                            eff_rank = 0
                    except Exception:
                        eff_rank = 0

                    # 存储
                    if hi not in attn_data[mode][li]:
                        attn_data[mode][li][hi] = {
                            'entropy': [], 'top1': [], 'top3': [],
                            'eff_rank': []
                        }

                    attn_data[mode][li][hi]['entropy'].append(avg_entropy)
                    attn_data[mode][li][hi]['top1'].append(avg_top1)
                    attn_data[mode][li][hi]['top3'].append(avg_top3)
                    attn_data[mode][li][hi]['eff_rank'].append(float(eff_rank))

            del out, attentions
            gc.collect()

        if (si + 1) % 3 == 0:
            log_progress("Exp4", si + 1, len(sentences), t_start)

    # === 分析 ===
    print("\n--- Exp4A: Attention Entropy by Layer (averaged over heads) ---")
    print("  (Low entropy = hard routing, High entropy = soft smoothing)")

    for mode in modes:
        print(f"\n  Mode: {mode}")
        print(f"  {'Layer':>6} {'Avg Entropy':>12} {'Max Entropy':>12} "
              f"{'Avg Top1':>10} {'Avg Top3':>10} {'Classification':>20}")

        for li in sample_layers:
            heads_data = attn_data[mode][li]
            if not heads_data:
                continue

            # 所有head的平均
            all_entropy = []
            all_top1 = []
            all_top3 = []
            all_eff_rank = []
            for hi, hdata in heads_data.items():
                all_entropy.extend(hdata['entropy'])
                all_top1.extend(hdata['top1'])
                all_top3.extend(hdata['top3'])
                all_eff_rank.extend(hdata['eff_rank'])

            avg_ent = float(np.mean(all_entropy)) if all_entropy else 0
            max_ent = float(np.max(all_entropy)) if all_entropy else 0
            avg_t1 = float(np.mean(all_top1)) if all_top1 else 0
            avg_t3 = float(np.mean(all_top3)) if all_top3 else 0
            avg_er = float(np.mean(all_eff_rank)) if all_eff_rank else 0

            # 分类
            max_possible_entropy = np.log(seq_len) if seq_len > 1 else 1
            norm_entropy = avg_ent / max_possible_entropy

            if norm_entropy < 0.3:
                cls = "HARD ROUTING"
            elif norm_entropy < 0.6:
                cls = "MIXED"
            else:
                cls = "SOFT SMOOTHING"

            print(f"  {li:6d} {avg_ent:12.4f} {max_ent:12.4f} "
                  f"{avg_t1:10.4f} {avg_t3:10.4f} {cls:>20}")

    # === Exp4B: Head Diversity ===
    print("\n--- Exp4B: Head Diversity at Selected Layers ---")

    for li in [0, n_layers // 2, n_layers - 1]:
        if li not in sample_layers:
            continue

        print(f"\n  Layer {li}:")
        for mode in modes:
            heads_data = attn_data[mode][li]
            if not heads_data:
                continue

            # 计算head间的熵差异
            head_avg_entropies = []
            for hi, hdata in heads_data.items():
                head_avg_entropies.append(float(np.mean(hdata['entropy'])))

            if head_avg_entropies:
                ent_range = max(head_avg_entropies) - min(head_avg_entropies)
                ent_std = float(np.std(head_avg_entropies))

                # 找"最硬"和"最软"的head
                heads_sorted = sorted(heads_data.items(),
                                       key=lambda x: np.mean(x[1]['entropy']))
                hardest_head = heads_sorted[0][0]
                softest_head = heads_sorted[-1][0]

                print(f"    {mode}: entropy range={ent_range:.3f} (std={ent_std:.3f}), "
                      f"hardest=H{hardest_head}, softest=H{softest_head}")

    # === Exp4C: CoT vs Normal的attention差异 ===
    print("\n--- Exp4C: Attention Differences (CoT vs Normal vs Translation) ---")

    if "normal" in attn_data and "cot" in attn_data:
        print(f"  {'Layer':>6} {'Normal Ent':>11} {'CoT Ent':>11} {'Trans Ent':>11} "
              f"{'CoT-Normal':>11} {'Interpretation':>20}")

        for li in sample_layers:
            n_ent = [np.mean(hd['entropy']) for hd in attn_data['normal'][li].values()] if attn_data['normal'][li] else []
            c_ent = [np.mean(hd['entropy']) for hd in attn_data['cot'][li].values()] if attn_data['cot'][li] else []
            t_ent = [np.mean(hd['entropy']) for hd in attn_data['translation'][li].values()] if attn_data['translation'][li] else []

            if n_ent and c_ent:
                n_avg = float(np.mean(n_ent))
                c_avg = float(np.mean(c_ent))
                t_avg = float(np.mean(t_ent)) if t_ent else 0
                diff = c_avg - n_avg

                if diff > 0.1:
                    interp = "CoT MORE smoothing"
                elif diff < -0.1:
                    interp = "CoT MORE routing"
                else:
                    interp = "Similar"

                print(f"  {li:6d} {n_avg:11.4f} {c_avg:11.4f} {t_avg:11.4f} "
                      f"{diff:+11.4f} {interp:>20}")

    # === 综合判断 ===
    print("\n--- Exp4 Overall: Attention Kernel Classification ---")

    for mode in modes:
        # 计算各层的平均归一化熵
        layer_entropies = {}
        for li in sample_layers:
            heads_data = attn_data[mode][li]
            if not heads_data:
                continue
            all_ent = []
            for hi, hdata in heads_data.items():
                all_ent.extend(hdata['entropy'])
            if all_ent:
                layer_entropies[li] = float(np.mean(all_ent))

        if layer_entropies:
            early_ent = np.mean([v for k, v in layer_entropies.items() if k < n_layers // 3])
            mid_ent = np.mean([v for k, v in layer_entropies.items()
                              if n_layers // 3 <= k < 2 * n_layers // 3])
            late_ent = np.mean([v for k, v in layer_entropies.items() if k >= 2 * n_layers // 3])

            print(f"  {mode}: Early ent={early_ent:.4f}, Mid ent={mid_ent:.4f}, Late ent={late_ent:.4f}")

    return attn_data


# ========================================================================
# MAIN
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    is_lite = model_name != "qwen3"

    t0 = time.time()
    print(f"[Phase 203] Geometric Flow Analysis — {model_name}")
    print(f"[Phase 203] Time: {datetime.now()}")
    print(f"[Phase 203] Lite mode: {is_lite}")
    print(f"[Phase 203] Theory: LLM = finite-depth nonlinear transport system")
    print(f"[Phase 203] Key shift: 'conceptual explanation' → 'differentiable mathematical objects'")
    print(f"[Phase 203] h_{{l+1}} = h_l + F_l(h_l, p)")
    print(f"[Phase 203] Core objects: J_l (Jacobian), W_U sensitivity, attention kernel K(x_i,x_j)")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"[load] {model_name}: {info.model_class}, {info.n_layers}L, d={info.d_model}")

    # Config
    if is_lite:
        exp1_sents = EXP1_SENTENCES_L
        exp2_sents = LITE_SENTENCES
        exp4_sents = LITE_SENTENCES[:5]
        n_probe = 40
    else:
        exp1_sents = EXP1_SENTENCES_Q
        exp2_sents = BASE_SENTENCES
        exp4_sents = BASE_SENTENCES[:10]
        n_probe = 40

    print(f"  Exp1 sentences: {len(exp1_sents)}, probes: {n_probe}")
    print(f"  Exp2 sentences: {len(exp2_sents)}")
    print(f"  Exp4 sentences: {len(exp4_sents)}")

    # Run experiments
    print(f"\n{'='*60}")
    print("Starting Phase 203 experiments...")
    print(f"{'='*60}")

    exp1_results = exp1_cumulative_transport(model, tokenizer, device, info,
                                              exp1_sents, n_probe=n_probe)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    exp2_results = exp2_output_sensitivity(model, tokenizer, device, info, exp2_sents)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    exp3_results = exp3_per_layer_growth(exp1_results, info.n_layers, CORE_MODES)

    exp4_results = exp4_attention_kernel(model, tokenizer, device, info, exp4_sents)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== Final Summary =====
    print("\n" + "="*60)
    print(f"PHASE 203 SUMMARY — {model_name}")
    print("="*60)

    print("\n1. Cumulative Transport Spectrum (Exp1):")
    mid_layer = info.n_layers // 2
    for mode in CORE_MODES:
        if mode in exp1_results and mid_layer in exp1_results[mode]:
            r = exp1_results[mode][mid_layer]
            print(f"   {mode}: σ_max={r['sigma_max']:.6f}, "
                  f"pr={r['participation_ratio']:.1f}, "
                  f"cond={r['condition_number']:.2f}")

    print("\n2. Output-Sensitivity Decomposition (Exp2):")
    if exp2_results and 'decomp_data' in exp2_results:
        for mode in ALL_MODES:
            if mode in exp2_results['decomp_data']:
                mid_data = exp2_results['decomp_data'][mode].get(mid_layer, {})
                late_data = exp2_results['decomp_data'][mode].get(info.n_layers - 1, {})
                print(f"   {mode}: mid weak={mid_data.get('weak_avg', 0):.4f}, "
                      f"late weak={late_data.get('weak_avg', 0):.4f}")

    print("\n3. Per-layer Growth Rate (Exp3):")
    for mode in CORE_MODES:
        if mode in exp3_results and exp3_results[mode]:
            growth_rates = [la['growth_rate'] for la in exp3_results[mode]
                           if la['growth_rate'] is not None]
            if growth_rates:
                print(f"   {mode}: avg growth={np.mean(growth_rates):.4f}")

    print("\n4. Attention Kernel Geometry (Exp4):")
    if exp4_results:
        for mode in CORE_MODES:
            if mode in exp4_results:
                # 找mid layer
                ml = info.n_layers // 2
                if ml in exp4_results[mode]:
                    heads = exp4_results[mode][ml]
                    all_ent = [np.mean(hd['entropy']) for hd in heads.values()]
                    if all_ent:
                        print(f"   {mode} L{ml}: avg entropy={np.mean(all_ent):.4f}")

    # Save results
    out_path = Path(f"tests/glm5_temp/phase203_{model_name}_results.json")

    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    # Simplified save (exp4 has complex nested structure)
    save_data = {
        'model': model_name,
        'exp1_transport': {
            mode: {str(li): {k: convert(v) for k, v in layer_data.items()
                             if k != 'top_singular_values'}
                   for li, layer_data in mode_data.items()}
            for mode, mode_data in exp1_results.items()
        },
        'exp2_decomposition': {
            k: convert(v) if not isinstance(v, dict) else
               {m: {str(li): {dk: convert(dv) for dk, dv in layer_d.items()}
                    for li, layer_d in m_data.items()}
                for m, m_data in v.items()}
            for k, v in exp2_results.items() if k != 'decomp_data'
        },
        'exp3_growth': {
            mode: [{k: convert(v) for k, v in la.items()} for la in las]
            for mode, las in exp3_results.items()
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Release
    elapsed = time.time() - t0
    print(f"\n[Phase 203] COMPLETE in {elapsed:.1f}s ({model_name})")
    release_model(model)


if __name__ == "__main__":
    main()
