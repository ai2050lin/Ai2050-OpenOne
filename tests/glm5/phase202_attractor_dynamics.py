"""
Phase 202: Attractor Dynamics & Trajectory Manifold Structure
=============================================================

理论框架 (基于Phase 201关键修正):

  Phase 201的核心发现: 路由非线性, B+B>>A+B>>A+A线性度梯度, CoT主导
  但解释有误 — "程序竞争"隐含了离散模块, Transformer根本没有调度器!

  ★核心修正★:
  Transformer是连续动力系统, 不是离散程序系统:
    h_{l+1} = h_l + A_l(h_l) + M_l(h_l)
  
  不同prompt → 不同初始条件 → 不同attractor basin → 不同轨迹
  
  Phase 201的"程序主导"实际是:
  - B+B更线性 = tangent perturbation (切向扰动, 在同一attractor附近)
    F(x+δ₁+δ₂) ≈ F(x+δ₁) + F(x+δ₂) 局部线性成立
  - A+A更非线性 = attractor transition (吸引子跃迁, 跨越basin boundary)
    系统已经跨过盆地边界, 局部线性失效
  - CoT"最强" = 更强的轨迹稳定器, 把系统推入更深/更稳定的attractor basin
  - 深层收敛 = protocol projection (协议投影), 把内部动力学压缩进token概率单纯形

  ★关键转变★:
  不再使用"程序""竞争""吞并"等离散化语言
  而使用: attractor, basin, bifurcation, trajectory, stability, projection

Phase 202实验 (4个子实验):

Exp1: Trajectory Manifold Structure (轨迹流形结构)
  核心: hidden trajectory的真实维度是多少? 不同mode是否在低维流形的不同区域?
  方法: 收集全层轨迹, PCA降维, 计算参与比(effective dimensionality)
  如果d_eff << d_model → 轨迹在低维流形上
  如果不同mode的轨迹在PCA空间中分离 → 支持吸引子分化的假说

Exp2: Basin Boundary Detection via Mode Interpolation (盆地边界检测)
  核心: 从normal到CoT的"相变"发生在哪个插值点?
  方法: 在normal和CoT prompt之间做词级插值, 追踪hidden trajectory
  找到trajectory的"跳变点"= basin boundary
  这比Phase 200的token-level相变更精确

Exp3: Lyapunov-like Stability Analysis (李雅普诺夫稳定性分析)
  核心: 不同层的动力学稳定性如何? 哪些层是"稳定吸引区", 哪些是"分岔区"?
  方法: 在每层注入小噪声, 测量后续层的轨迹发散程度
  发散小 = 强吸引区 (attractor)
  发散大 = 弱吸引区 (basin boundary附近)

Exp4: Protocol Projection Analysis (协议投影分析)
  核心: 深层收敛是"LayerNorm压缩"还是"协议投影"?
  方法: 每层的hidden state在W_U行空间中的投影比
  如果深层的W_U投影比远高于中间层 → 后层在做"计算→语言"的协议投影
  如果中间层W_U投影比低 → 内部计算不在语言空间中, 支持"暗物质"

数据量 (加大!):
  - Qwen3: 40句 × 全层
  - GLM4/DS7B: 20句 × 全层

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
# 基础句子 — 40句 (比Phase 201增加33%)
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
    "The researcher tests the hypothesis",
    "The designer creates the prototype",
    "The manager oversees the project",
    "The consultant advises the client",
    "The technician calibrates the instrument",
    "The inspector checks the equipment",
    "The coordinator organizes the event",
    "The supervisor monitors the process",
    "The developer builds the application",
    "The administrator manages the system",
]

LITE_SENTENCES = BASE_SENTENCES[:20]


# ========================================================================
# Mode Prompt Templates — 动力系统视角
# ========================================================================
# 不再使用"程序"标签, 而用"吸引子类型"标签
MODE_PROMPTS = {
    # baseline attractor (基准吸引子)
    "normal": lambda b: b,

    # attractor transition modes (吸引子跃迁模式) — 跨越basin boundary
    "qa": lambda b: "Does " + b[0].lower() + b[1:] + "?",
    "cot": lambda b: "Problem: " + b[0].lower() + b[1:] + ". Think step by step.",
    "conditional": lambda b: "If " + b[0].lower() + b[1:] + ", then",

    # tangent perturbation modes (切向扰动模式) — 在baseline attractor附近
    "translation": lambda b: "Translate to Chinese: " + b,
    "coding": lambda b: "Write code for: " + b[0].lower() + b[1:],
    "negation": lambda b: "It is false that " + b[0].lower() + b[1:],

    # weak perturbation (微弱扰动)
    "narrative": lambda b: "Once upon a time, " + b[0].lower() + b[1:],
}


# ========================================================================
# 工具函数
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
    print(f"  [{exp_name}] {current}/{total} ({elapsed:.0f}s) GPU={gpu:.2f}GB {extra}")


def extract_full_trajectory(model, tokenizer, device, text, n_layers, max_len=96):
    """
    提取完整的层轨迹: 每层last token的hidden state
    返回: np.array shape (n_layers+1, d_model)
    """
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

    return np.array(trajectory)  # (n_layers+1, d_model)


def compute_effective_dimensionality(data_matrix, threshold=0.95):
    """
    计算有效维度 (参与比 / participation ratio)

    data_matrix: (n_samples, d)
    返回: effective dimensionality (float)
    """
    # 中心化
    centered = data_matrix - data_matrix.mean(axis=0, keepdims=True)

    # SVD
    if centered.shape[0] < centered.shape[1]:
        # 样本数 < 维度, 用X X^T的trick
        cov = centered @ centered.T
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)  # 数值稳定性
        eigenvalues = np.sort(eigenvalues)[::-1]
    else:
        cov = centered.T @ centered
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)
        eigenvalues = np.sort(eigenvalues)[::-1]

    total_energy = np.sum(eigenvalues)
    if total_energy < 1e-10:
        return 0.0, 0

    # 参与比: (∑λ)² / ∑(λ²) — 衡量"有多少维真正参与"
    participation_ratio = (np.sum(eigenvalues))**2 / max(np.sum(eigenvalues**2), 1e-20)

    # 95%方差维度
    cumsum = np.cumsum(eigenvalues) / total_energy
    n_95 = np.searchsorted(cumsum, threshold) + 1

    return float(participation_ratio), int(n_95)


def compute_trajectory_curvature(trajectory):
    """
    计算轨迹曲率: ||dh/dl||的变化率

    trajectory: (n_layers+1, d_model)
    返回: per-layer curvature (n_layers,)
    """
    # 一阶差分: 速度
    velocity = np.diff(trajectory, axis=0)  # (n_layers, d_model)

    # 二阶差分: 加速度 → 曲率的近似
    acceleration = np.diff(velocity, axis=0)  # (n_layers-1, d_model)

    # 曲率 = ||acceleration|| / ||velocity|| (归一化)
    curvature = []
    for i in range(len(acceleration)):
        v_norm = np.linalg.norm(velocity[i])
        a_norm = np.linalg.norm(acceleration[i])
        if v_norm > 1e-10:
            curvature.append(a_norm / v_norm)
        else:
            curvature.append(0.0)

    return np.array(curvature)  # (n_layers-1,)


# ========================================================================
# EXP1: Trajectory Manifold Structure
# ========================================================================
def exp1_trajectory_manifold(model, tokenizer, device, info, sentences):
    """
    核心实验: hidden trajectory的真实维度是多少?

    方法:
    1. 对每种mode, 收集所有句子的全层轨迹
    2. 将所有轨迹堆叠成矩阵
    3. 计算有效维度 (参与比)
    4. 分析不同mode在PCA空间中的分布

    如果d_eff << d_model → 轨迹确实在低维流形上
    如果不同mode在PCA空间中分离 → 支持吸引子分化
    """
    print("\n" + "="*60)
    print("Exp1: Trajectory Manifold Structure")
    print("="*60)

    n_layers = info.n_layers
    d_model = info.d_model
    modes = list(MODE_PROMPTS.keys())

    print(f"  Modes: {modes}")
    print(f"  n_layers: {n_layers}, d_model: {d_model}")
    print(f"  Sentences: {len(sentences)}")

    # 收集: {mode: {layer: [hidden_states]}}  — 每个mode在每个层的表示
    # 和 {mode: [trajectories]}  — 完整轨迹
    layer_activations = {mode: {li: [] for li in range(n_layers+1)} for mode in modes}
    all_trajectories = {mode: [] for mode in modes}

    t_start = time.time()
    total = len(sentences) * len(modes)

    for si, base in enumerate(sentences):
        for mode in modes:
            text = MODE_PROMPTS[mode](base)
            traj = extract_full_trajectory(model, tokenizer, device, text, n_layers)

            all_trajectories[mode].append(traj)

            for li in range(min(traj.shape[0], n_layers+1)):
                layer_activations[mode][li].append(traj[li])

        if (si + 1) % 5 == 0:
            log_progress("Exp1", si+1, len(sentences), t_start,
                         f"({(si+1)*len(modes)}/{total})")

    # === 分析1A: 每个mode的有效维度 ===
    print("\n--- Exp1A: Effective Dimensionality per Mode ---")

    mid_layer = n_layers // 2
    sample_layers = [0, mid_layer, n_layers-1]

    for li in sample_layers:
        print(f"\n  Layer {li}:")
        for mode in modes:
            acts = layer_activations[mode][li]
            if len(acts) < 3:
                print(f"    {mode}: insufficient data")
                continue
            data_matrix = np.array(acts)  # (n_sents, d_model)
            d_eff, n_95 = compute_effective_dimensionality(data_matrix)
            print(f"    {mode:<15} d_eff={d_eff:.1f}  n_95={n_95}  "
                  f"(d_eff/d_model={d_eff/d_model:.4f})")

    # === 分析1B: 全轨迹的有效维度 ===
    print("\n--- Exp1B: Full Trajectory Effective Dimensionality ---")

    # 把轨迹展开: (n_sents * n_layers, d_model)
    for mode in modes:
        trajs = all_trajectories[mode]
        if not trajs:
            continue
        # 取中间层到晚层的轨迹段 (计算最丰富的部分)
        mid_start = n_layers // 4
        mid_end = 3 * n_layers // 4
        segments = []
        for traj in trajs:
            segments.append(traj[mid_start:mid_end+1])
        segment_matrix = np.vstack(segments)  # (n_sents * segment_len, d_model)

        d_eff, n_95 = compute_effective_dimensionality(segment_matrix)
        print(f"  {mode:<15} d_eff={d_eff:.1f}  n_95={n_95}  "
              f"(d_eff/d_model={d_eff/d_model:.4f})")

    # === 分析1C: PCA空间中不同mode的分布 ===
    print("\n--- Exp1C: Mode Distribution in PCA Space ---")

    # 用中间层的所有mode数据做PCA
    mid_acts_all = []
    mode_labels = []
    for mode in modes:
        acts = layer_activations[mode][mid_layer]
        mid_acts_all.extend(acts)
        mode_labels.extend([mode] * len(acts))

    mid_acts_all = np.array(mid_acts_all)
    # PCA
    centered = mid_acts_all - mid_acts_all.mean(axis=0)
    n_components = min(50, centered.shape[0]-1, centered.shape[1]-1)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # 投影到前2个主成分
    pc1 = centered @ Vt[0]
    pc2 = centered @ Vt[1] if n_components > 1 else np.zeros_like(pc1)

    # 计算每个mode在PCA空间中的centroid
    print(f"\n  PCA at Layer {mid_layer} — top-5 singular values: {S[:5].tolist()}")
    print(f"  Variance explained by PC1: {S[0]**2 / np.sum(S**2):.4f}")

    mode_centroids_pca = {}
    for mode in modes:
        mask = np.array([l == mode for l in mode_labels])
        if mask.any():
            mode_centroids_pca[mode] = (
                float(np.mean(pc1[mask])),
                float(np.mean(pc2[mask]))
            )

    print(f"\n  Mode centroids in PCA space:")
    print(f"  {'Mode':<15} {'PC1':>10} {'PC2':>10}")
    for mode, (c1, c2) in sorted(mode_centroids_pca.items(), key=lambda x: x[1][0]):
        print(f"  {mode:<15} {c1:10.4f} {c2:10.4f}")

    # 在PCA空间中计算mode间距离
    print(f"\n  Mode-mode distance in PCA space (Euclidean, 2D):")
    mode_list = list(mode_centroids_pca.keys())
    print(f"  {'':>15}", end="")
    for m in mode_list:
        print(f" {m[:7]:>7}", end="")
    print()
    pca_dist = {}
    for m1 in mode_list:
        print(f"  {m1[:15]:>15}", end="")
        for m2 in mode_list:
            c1 = mode_centroids_pca[m1]
            c2 = mode_centroids_pca[m2]
            d = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            print(f" {d:7.3f}", end="")
            pca_dist[f"{m1}_vs_{m2}_pca"] = float(d)
        print()

    # === 分析1D: 轨迹曲率 ===
    print("\n--- Exp1D: Trajectory Curvature ---")
    print("  (High curvature = strong nonlinear deformation = possible attractor boundary)")

    for mode in modes:
        trajs = all_trajectories[mode]
        if not trajs:
            continue
        curvatures = [compute_trajectory_curvature(traj) for traj in trajs]
        # 对齐长度
        min_len = min(len(c) for c in curvatures)
        curv_matrix = np.array([c[:min_len] for c in curvatures])  # (n_sents, n_layers-1)
        avg_curv = np.mean(curv_matrix, axis=0)

        # 找曲率峰值
        peak_layer = np.argmax(avg_curv)
        peak_val = avg_curv[peak_layer]
        mean_curv = np.mean(avg_curv)

        print(f"  {mode:<15} peak_curv={peak_val:.4f} at L{peak_layer}, "
              f"mean_curv={mean_curv:.4f}")

    # 找所有mode共有的高曲率层 (可能是attractor boundary)
    all_curvatures = {}
    for mode in modes:
        trajs = all_trajectories[mode]
        if not trajs:
            continue
        curvatures = [compute_trajectory_curvature(traj) for traj in trajs]
        min_len = min(len(c) for c in curvatures)
        curv_matrix = np.array([c[:min_len] for c in curvatures])
        all_curvatures[mode] = np.mean(curv_matrix, axis=0)

    if all_curvatures:
        min_len = min(len(v) for v in all_curvatures.values())
        # 全mode平均曲率
        avg_all = np.mean([v[:min_len] for v in all_curvatures.values()], axis=0)
        high_curv_layers = np.where(avg_all > np.percentile(avg_all, 75))[0]
        print(f"\n  High curvature layers (all modes): {high_curv_layers.tolist()}")
        print(f"  → These may be attractor transition zones")

    return {
        'pca_distances': pca_dist,
        'mode_centroids_pca': {k: list(v) for k, v in mode_centroids_pca.items()},
    }


# ========================================================================
# EXP2: Basin Boundary Detection via Mode Interpolation
# ========================================================================
def exp2_basin_boundary(model, tokenizer, device, info, sentences):
    """
    核心实验: 从baseline attractor到其他attractor的盆地边界在哪?

    方法:
    1. 选择normal→CoT, normal→QA, normal→translation等转换
    2. 在两种prompt之间做"插值":
       - 方式1: 逐步添加模式指令 (如normal → normal+"." → normal+". Think" → normal+". Think step by step.")
       - 方式2: 词级替换 (如"The cat chases" → "Does the cat chase" → "Does the cat chase? Think step by step.")
    3. 追踪每步的hidden trajectory
    4. 找到trajectory的"跳变点" = basin boundary

    关键判断:
    - 如果轨迹在某步突然偏转 → basin boundary at that step
    - 如果轨迹平滑变化 → 没有sharp basin boundary
    - 跳变幅度 = basin boundary的"陡度"
    """
    print("\n" + "="*60)
    print("Exp2: Basin Boundary Detection via Mode Interpolation")
    print("="*60)

    n_layers = info.n_layers
    mid_layer = n_layers // 2

    # 定义插值路径: 每一步是一个prompt
    # 格式: (transition_name, [step_prompts])
    # 使用3个基础句子
    test_sentences = sentences[:6]

    transitions = {
        # normal → CoT: 逐步添加CoT指令
        "normal_to_cot": lambda b: [
            b,                            # step 0: normal
            b + ".",                      # step 1: 加句号
            b + ". Think",                # step 2: 加Think
            b + ". Think step",           # step 3: 加step
            b + ". Think step by",        # step 4: 加by
            b + ". Think step by step.",  # step 5: 完整CoT
        ],
        # normal → QA: 句首疑问化
        "normal_to_qa": lambda b: [
            b,                            # step 0: normal
            "Does " + b[0].lower() + b[1:],     # step 1: Does开头
            "Does " + b[0].lower() + b[1:] + "?",  # step 2: 加问号
        ],
        # normal → translation: 逐步添加翻译指令
        "normal_to_translation": lambda b: [
            b,                            # step 0: normal
            "Translate: " + b,            # step 1: 加Translate
            "Translate to Chinese: " + b,  # step 2: 完整翻译指令
        ],
        # normal → conditional: 逐步添加条件
        "normal_to_conditional": lambda b: [
            b,                            # step 0: normal
            "If " + b[0].lower() + b[1:],  # step 1: If开头
            "If " + b[0].lower() + b[1:] + ", then",  # step 2: 完整条件句
        ],
        # normal → coding: 逐步添加代码指令
        "normal_to_coding": lambda b: [
            b,                            # step 0: normal
            "Write code: " + b[0].lower() + b[1:],  # step 1: 加Write code
            "Write code for: " + b[0].lower() + b[1:],  # step 2: 完整代码指令
        ],
    }

    results = {}

    for trans_name, trans_fn in transitions.items():
        print(f"\n--- Transition: {trans_name} ---")

        step_trajectories = {}  # {step: [trajectories]}
        step_labels = {}

        t_start = time.time()

        for si, base in enumerate(test_sentences):
            steps = trans_fn(base)

            for step_i, step_text in enumerate(steps):
                traj = extract_full_trajectory(model, tokenizer, device, step_text, n_layers)

                if step_i not in step_trajectories:
                    step_trajectories[step_i] = []
                step_trajectories[step_i].append(traj)

                if step_i not in step_labels:
                    step_labels[step_i] = step_text[:50] + "..." if len(step_text) > 50 else step_text

            if (si + 1) % 3 == 0:
                log_progress("Exp2", si+1, len(test_sentences), t_start,
                            f"trans={trans_name}")

        # === 分析: 相邻step的轨迹距离 ===
        print(f"\n  Step-by-step trajectory shift (mid layer L{mid_layer}):")

        max_step = max(step_trajectories.keys())
        step_dists = []
        step_sims_to_start = []

        for step_i in range(max_step + 1):
            trajs = step_trajectories[step_i]
            # 与step 0的距离
            if step_i == 0:
                dist_from_start = 0.0
                sim_to_start = 1.0
            else:
                dists = []
                sims = []
                for si_idx in range(min(len(trajs), len(step_trajectories[0]))):
                    if mid_layer < trajs[si_idx].shape[0] and mid_layer < step_trajectories[0][si_idx].shape[0]:
                        d = compute_cosine_dist(trajs[si_idx][mid_layer], step_trajectories[0][si_idx][mid_layer])
                        s = compute_cosine_sim(trajs[si_idx][mid_layer], step_trajectories[0][si_idx][mid_layer])
                        dists.append(d)
                        sims.append(s)
                dist_from_start = float(np.mean(dists)) if dists else 0.0
                sim_to_start = float(np.mean(sims)) if sims else 1.0

            step_dists.append(dist_from_start)
            step_sims_to_start.append(sim_to_start)

            label = step_labels.get(step_i, "")
            print(f"    Step {step_i}: dist_from_start={dist_from_start:.4f} "
                  f"sim_to_start={sim_to_start:.4f}  '{label}'")

        # === 找跳变点: 相邻step之间距离最大的 ===
        print(f"\n  Step-to-step trajectory shift:")
        jump_points = []
        for step_i in range(1, max_step + 1):
            dist_jump = step_dists[step_i] - step_dists[step_i-1]
            jump_points.append((step_i, dist_jump))
            print(f"    Step {step_i-1}→{step_i}: dist change = {dist_jump:+.4f}")

        if jump_points:
            max_jump_step, max_jump_val = max(jump_points, key=lambda x: x[1])
            print(f"\n  >>> MAXIMUM JUMP at Step {max_jump_step-1}→{max_jump_step} "
                  f"(Δdist={max_jump_val:.4f})")
            print(f"  >>> This is likely the BASIN BOUNDARY for {trans_name}")

            # 跳变的层分布
            if max_jump_step in step_trajectories and (max_jump_step-1) in step_trajectories:
                prev_trajs = step_trajectories[max_jump_step-1]
                next_trajs = step_trajectories[max_jump_step]

                layer_jumps = []
                for li in range(0, n_layers, max(1, n_layers//10)):
                    dists = []
                    for si_idx in range(min(len(prev_trajs), len(next_trajs))):
                        if li < prev_trajs[si_idx].shape[0] and li < next_trajs[si_idx].shape[0]:
                            d = compute_cosine_dist(prev_trajs[si_idx][li], next_trajs[si_idx][li])
                            dists.append(d)
                    if dists:
                        layer_jumps.append((li, float(np.mean(dists))))

                print(f"  Layer-wise jump at boundary:")
                for li, d in layer_jumps:
                    print(f"    L{li}: dist={d:.4f}")

        results[trans_name] = {
            'step_dists': step_dists,
            'step_sims': step_sims_to_start,
            'max_jump_step': int(max_jump_step) if jump_points else 0,
            'max_jump_val': float(max_jump_val) if jump_points else 0,
        }

    # === 跨转换比较 ===
    print("\n--- Exp2 Summary: Basin Boundary Comparison ---")
    print(f"  {'Transition':<25} {'Max Jump Step':>15} {'Jump Value':>12} {'Boundary Type':>20}")
    for trans_name, r in results.items():
        # 判断boundary类型
        # 大跳变 = attractor transition (A类)
        # 小跳变 = tangent perturbation (B类)
        jump_val = r['max_jump_val']
        if jump_val > 0.15:
            btype = "Attractor Transition"
        elif jump_val > 0.08:
            btype = "Weak Transition"
        else:
            btype = "Tangent Perturbation"
        print(f"  {trans_name:<25} {r['max_jump_step']:>15} {jump_val:>12.4f} {btype:>20}")

    return results


# ========================================================================
# EXP3: Lyapunov-like Stability Analysis
# ========================================================================
def exp3_lyapunov_stability(model, tokenizer, device, info, sentences):
    """
    核心实验: 不同层的动力学稳定性如何?

    方法:
    1. 对每个prompt, 提取正常轨迹
    2. 在每层注入小噪声 (σ = 0.01 * ||h_l||)
    3. 从该层开始继续前向传播
    4. 测量最终层hidden state与无噪声版本的偏离

    如果偏离小 → 强吸引区 (robust attractor)
    如果偏离大 → 弱吸引区 (near basin boundary)

    由于我们无法从中间层注入噪声后继续前向传播(需要修改模型内部),
    改用以下替代方案:
    - 在输入层注入不同尺度的噪声
    - 测量每层的传播效应
    - 比较不同mode对噪声的鲁棒性
    """
    print("\n" + "="*60)
    print("Exp3: Lyapunov-like Stability Analysis")
    print("="*60)

    n_layers = info.n_layers
    d_model = info.d_model
    mid_layer = n_layers // 2

    modes = ["normal", "qa", "cot", "translation", "coding", "conditional"]
    noise_scales = [0.001, 0.005, 0.01, 0.02]  # 相对于embedding范数

    print(f"  Modes: {modes}")
    print(f"  Noise scales: {noise_scales}")
    print(f"  Sentences: {len(sentences[:8])}")

    # 收集: {mode: {noise_scale: {layer: [deltas]}}}
    stability_data = {mode: {ns: {li: [] for li in range(n_layers+1)} for ns in noise_scales} for mode in modes}

    test_sents = sentences[:8]
    t_start = time.time()
    total = len(test_sents) * len(modes) * (1 + len(noise_scales))

    for si, base in enumerate(test_sents):
        for mode in modes:
            text = MODE_PROMPTS[mode](base)

            # 基准轨迹
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)
            with torch.no_grad():
                out_base = model(input_ids=ids, output_hidden_states=True)
            base_states = [h[0, -1, :].detach().float().cpu().numpy() for h in out_base.hidden_states]

            # 噪声注入
            embed_layer = model.get_input_embeddings()
            with torch.no_grad():
                base_embed = embed_layer(ids)

            for ns in noise_scales:
                # 注入噪声到embedding
                noise = torch.randn_like(base_embed) * ns
                noisy_embed = base_embed + noise

                with torch.no_grad():
                    try:
                        out_noisy = model(inputs_embeds=noisy_embed, output_hidden_states=True)
                    except Exception:
                        continue

                noisy_states = [h[0, -1, :].detach().float().cpu().numpy() for h in out_noisy.hidden_states]

                # 计算每层的偏离
                for li in range(min(len(base_states), len(noisy_states), n_layers+1)):
                    delta = compute_cosine_dist(base_states[li], noisy_states[li])
                    euc_delta = float(np.linalg.norm(base_states[li] - noisy_states[li]))
                    stability_data[mode][ns][li].append({
                        'cos_delta': delta,
                        'euc_delta': euc_delta,
                        'base_norm': float(np.linalg.norm(base_states[li])),
                    })

            # 释放
            del out_base, base_embed
            gc.collect()

        log_progress("Exp3", si+1, len(test_sents), t_start)

    # === 分析 ===
    print("\n--- Exp3A: Noise Amplification by Layer ---")
    print("  (How much does small input noise get amplified at each layer?)")

    # 使用最小噪声尺度来测量线性化敏感性
    ns = noise_scales[0]  # 0.001

    for mode in modes:
        print(f"\n  Mode: {mode} (noise_scale={ns})")
        print(f"  {'Layer':>6} {'CosΔ':>10} {'EucΔ':>10} {'BaseNorm':>10} {'Amplif':>10}")
        amplifications = []
        for li in range(0, n_layers+1, max(1, n_layers//8)):
            deltas = stability_data[mode][ns].get(li, [])
            if not deltas:
                continue
            avg_cos = np.mean([d['cos_delta'] for d in deltas])
            avg_euc = np.mean([d['euc_delta'] for d in deltas])
            avg_norm = np.mean([d['base_norm'] for d in deltas])
            # 放大率: 偏离/原始噪声
            amplification = avg_euc / max(ns * avg_norm, 1e-10)
            amplifications.append((li, amplification, avg_cos, avg_euc, avg_norm))
            print(f"  {li:6d} {avg_cos:10.4f} {avg_euc:10.4f} {avg_norm:10.2f} {amplification:10.2f}")

    # === Exp3B: 不同mode的噪声鲁棒性比较 ===
    print("\n--- Exp3B: Noise Robustness by Mode ---")
    print("  (Which modes are more robust to input noise? = deeper attractors)")

    ns = noise_scales[1]  # 0.005
    for li in [0, mid_layer, n_layers-1]:
        print(f"\n  Layer {li} (noise_scale={ns}):")
        mode_robustness = {}
        for mode in modes:
            deltas = stability_data[mode][ns].get(li, [])
            if deltas:
                avg_cos = np.mean([d['cos_delta'] for d in deltas])
                avg_euc = np.mean([d['euc_delta'] for d in deltas])
                mode_robustness[mode] = (avg_cos, avg_euc)

        # 排序: cos_delta越小越鲁棒
        sorted_modes = sorted(mode_robustness.items(), key=lambda x: x[1][0])
        print(f"  {'Mode':<15} {'CosΔ':>10} {'EucΔ':>10} {'Interpretation':>20}")
        for mode, (cos_d, euc_d) in sorted_modes:
            interp = "Deep attractor" if cos_d < 0.02 else "Shallow attractor" if cos_d > 0.1 else "Medium attractor"
            print(f"  {mode:<15} {cos_d:10.4f} {euc_d:10.4f} {interp:>20}")

    # === Exp3C: 噪声放大随尺度的变化 ===
    print("\n--- Exp3C: Noise Amplification vs Scale ---")
    print("  (Does the system show nonlinear amplification at larger noise?)")

    for mode in ["normal", "cot", "translation"]:
        print(f"\n  Mode: {mode}, Layer {mid_layer}:")
        print(f"  {'Noise':>8} {'CosΔ':>10} {'EucΔ':>10} {'Amplif':>10}")
        for ns in noise_scales:
            deltas = stability_data[mode][ns].get(mid_layer, [])
            if deltas:
                avg_cos = np.mean([d['cos_delta'] for d in deltas])
                avg_euc = np.mean([d['euc_delta'] for d in deltas])
                avg_norm = np.mean([d['base_norm'] for d in deltas])
                amplif = avg_euc / max(ns * avg_norm, 1e-10)
                print(f"  {ns:8.3f} {avg_cos:10.4f} {avg_euc:10.4f} {amplif:10.2f}")

    # === 关键综合判断 ===
    print("\n--- Exp3 Overall: Attractor Depth Ranking ---")

    # 使用中等噪声尺度, 在中间层比较
    ns = noise_scales[1]
    attractor_depth = {}
    for mode in modes:
        deltas = stability_data[mode][ns].get(mid_layer, [])
        if deltas:
            avg_cos = np.mean([d['cos_delta'] for d in deltas])
            attractor_depth[mode] = avg_cos

    if attractor_depth:
        sorted_depth = sorted(attractor_depth.items(), key=lambda x: x[1])
        print(f"  Attractor depth (smaller cosΔ = deeper attractor):")
        for mode, depth in sorted_depth:
            print(f"    {mode:<15} cosΔ = {depth:.4f}")
        deepest = sorted_depth[0][0]
        shallowest = sorted_depth[-1][0]
        print(f"\n  >>> Deepest attractor: {deepest} (most robust to noise)")
        print(f"  >>> Shallowest attractor: {shallowest} (most sensitive to noise)")

    return {
        'attractor_depth': {k: float(v) for k, v in attractor_depth.items()},
    }


# ========================================================================
# EXP4: Protocol Projection Analysis
# ========================================================================
def exp4_protocol_projection(model, tokenizer, device, info, sentences):
    """
    核心实验: 深层收敛是"LayerNorm压缩"还是"协议投影"?

    方法:
    1. 在每层, 计算hidden state在W_U行空间中的投影比
    2. 如果深层的W_U投影比远高于中间层 → 后层在做"计算→语言"的协议投影
    3. 如果中间层W_U投影比低 → 内部计算不在语言空间中 → "暗物质"支持

    W_U投影比 = ||proj_{W_U}(h_l)||² / ||h_l||²
    """
    print("\n" + "="*60)
    print("Exp4: Protocol Projection Analysis")
    print("="*60)

    n_layers = info.n_layers
    d_model = info.d_model
    mid_layer = n_layers // 2
    modes = ["normal", "qa", "cot", "translation", "coding", "conditional"]

    print(f"  Modes: {modes}")
    print(f"  n_layers: {n_layers}, d_model: {d_model}")

    # 加载W_U
    print("  Loading W_U...")
    W_U = get_W_U(model, info.name if hasattr(info, 'name') else None)
    print(f"  W_U shape: {W_U.shape}")

    # SVD of W_U^T for projection
    print("  Computing W_U column space basis...")
    W_U_T = W_U.T.astype(np.float32)
    k = min(200, min(W_U_T.shape) - 2)
    from scipy.sparse.linalg import svds
    U_wut, s_wut, _ = svds(W_U_T, k=k)
    U_wut = np.asarray(U_wut, dtype=np.float64)  # (d_model, k)

    # U_wut的列是W_U行空间的正交基
    # 投影比 = ||U_wut^T @ h||² / ||h||²

    print(f"  W_U row space basis: {U_wut.shape}")
    print(f"  Top-5 singular values: {s_wut[:5].tolist()}")

    # 收集每层每个mode的投影比
    # {mode: {layer: [projection_ratios]}}
    projection_data = {mode: {li: [] for li in range(n_layers+1)} for mode in modes}

    test_sents = sentences[:10]
    t_start = time.time()
    total = len(test_sents) * len(modes)

    for si, base in enumerate(test_sents):
        for mode in modes:
            text = MODE_PROMPTS[mode](base)
            traj = extract_full_trajectory(model, tokenizer, device, text, n_layers)

            for li in range(min(traj.shape[0], n_layers+1)):
                h = traj[li]
                # 投影到W_U行空间
                proj_coeffs = U_wut.T @ h  # (k,)
                proj_energy = np.sum(proj_coeffs ** 2)
                h_energy = np.sum(h ** 2)
                ratio = proj_energy / max(h_energy, 1e-20)
                projection_data[mode][li].append(ratio)

        log_progress("Exp4", si+1, len(test_sents), t_start,
                     f"({(si+1)*len(modes)}/{total})")

    # === 分析 ===
    print("\n--- Exp4A: W_U Projection Ratio by Layer (all modes averaged) ---")
    print("  (How much of hidden state is in the 'language space'?)")

    avg_projection = {}
    for li in range(0, n_layers+1, max(1, n_layers//12)):
        all_ratios = []
        for mode in modes:
            all_ratios.extend(projection_data[mode].get(li, []))
        if all_ratios:
            avg = float(np.mean(all_ratios))
            avg_projection[li] = avg
            print(f"  Layer {li:3d}: W_U projection ratio = {avg:.4f} "
                  f"({'HIGH' if avg > 0.5 else 'MEDIUM' if avg > 0.2 else 'LOW'})")

    # === Exp4B: 每个mode的投影比曲线 ===
    print("\n--- Exp4B: W_U Projection Ratio by Mode ---")
    print(f"  {'Layer':>6}", end="")
    for mode in modes:
        print(f" {mode[:7]:>7}", end="")
    print("  Interpretation")

    for li in [0, n_layers//4, mid_layer, 3*n_layers//4, n_layers-1]:
        if li > n_layers:
            continue
        print(f"  {li:6d}", end="")
        for mode in modes:
            ratios = projection_data[mode].get(li, [])
            avg = float(np.mean(ratios)) if ratios else 0
            print(f" {avg:7.4f}", end="")
        print()

    # === Exp4C: 协议投影的"梯度" ===
    print("\n--- Exp4C: Protocol Projection Gradient ---")
    print("  (Where does the 'language projection' accelerate?)")

    for mode in modes:
        ratios_by_layer = []
        for li in range(n_layers+1):
            rs = projection_data[mode].get(li, [])
            ratios_by_layer.append(float(np.mean(rs)) if rs else 0)

        # 计算梯度
        grads = np.diff(ratios_by_layer)
        if len(grads) > 0:
            peak_grad_layer = np.argmax(grads)
            peak_grad = grads[peak_grad_layer]
            print(f"  {mode:<15} peak gradient at L{peak_grad_layer} "
                  f"(Δratio={peak_grad:.4f}), "
                  f"ratio range: [{min(ratios_by_layer):.4f}, {max(ratios_by_layer):.4f}]")

    # === Exp4D: "暗物质"比例 ===
    print("\n--- Exp4D: 'Dark Matter' Ratio (1 - W_U projection) ---")
    print("  (How much internal computation is NOT in language space?)")

    for li in [0, mid_layer, n_layers-1]:
        print(f"  Layer {li}:")
        for mode in modes:
            rs = projection_data[mode].get(li, [])
            if rs:
                avg_ratio = float(np.mean(rs))
                dark_matter = 1.0 - avg_ratio
                print(f"    {mode:<15} W_U={avg_ratio:.4f}  Dark={dark_matter:.4f}")

    # === 综合判断 ===
    print("\n--- Exp4 Overall: Protocol Projection vs LayerNorm ---")

    # 如果深层的投影比高于中间层, 且中间层的投影比很低
    # → 深层收敛不只是LayerNorm, 还有"主动投影到语言空间"
    early_ratios = []
    mid_ratios = []
    late_ratios = []
    for mode in modes:
        for li in range(0, n_layers//4):
            early_ratios.extend(projection_data[mode].get(li, []))
        for li in range(mid_layer-2, mid_layer+3):
            mid_ratios.extend(projection_data[mode].get(li, []))
        for li in range(max(0, n_layers-3), n_layers+1):
            late_ratios.extend(projection_data[mode].get(li, []))

    avg_early = float(np.mean(early_ratios)) if early_ratios else 0
    avg_mid = float(np.mean(mid_ratios)) if mid_ratios else 0
    avg_late = float(np.mean(late_ratios)) if late_ratios else 0

    print(f"  Early layers (0-L{n_layers//4}): W_U projection = {avg_early:.4f}")
    print(f"  Mid layers (L{mid_layer}):       W_U projection = {avg_mid:.4f}")
    print(f"  Late layers (L{n_layers-3}-L{n_layers-1}): W_U projection = {avg_late:.4f}")

    if avg_late > avg_mid + 0.05:
        print(f"\n  >>> DEEP LAYERS ACTIVELY PROJECT TO LANGUAGE SPACE")
        print(f"  >>> Late - Mid = {avg_late - avg_mid:.4f} (significant increase)")
        print(f"  >>> Supports 'Protocol Projection' hypothesis")
        print(f"  >>> Deep convergence is NOT just LayerNorm compression")
    elif avg_late > avg_mid:
        print(f"\n  >>> Weak evidence for Protocol Projection")
        print(f"  >>> Late - Mid = {avg_late - avg_mid:.4f} (small increase)")
    else:
        print(f"\n  >>> No evidence for Protocol Projection")
        print(f"  >>> Deep convergence may be primarily LayerNorm")

    if avg_mid < 0.2:
        print(f"\n  >>> STRONG 'DARK MATTER': Mid layers have only {avg_mid:.1%} in language space")
        print(f"  >>> {1-avg_mid:.1%} of computation is in non-language dimensions")
        print(f"  >>> Internal dynamics is NOT linguistic — it's computational")

    return {
        'avg_early': avg_early,
        'avg_mid': avg_mid,
        'avg_late': avg_late,
        'dark_matter_mid': 1 - avg_mid,
    }


# ========================================================================
# MAIN
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    is_lite = model_name != "qwen3"

    t0 = time.time()
    print(f"[Phase 202] Attractor Dynamics & Trajectory Manifold — {model_name}")
    print(f"[Phase 202] Time: {datetime.now()}")
    print(f"[Phase 202] Lite mode: {is_lite}")
    print(f"[Phase 202] Theory: LLM = continuous dynamical system with attractor basins")
    print(f"[Phase 202] Key shift: 'program competition' → 'attractor basin dynamics'")
    print(f"[Phase 202] h_{{l+1}} = h_l + A_l(h_l) + M_l(h_l)")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"[load] {model_name}: {info.model_class}, {info.n_layers}L, d={info.d_model}")

    # Config
    sentences = LITE_SENTENCES if is_lite else BASE_SENTENCES
    print(f"  Sentences: {len(sentences)}")

    # Run experiments
    print(f"\n{'='*60}")
    print("Starting Phase 202 experiments...")
    print(f"{'='*60}")

    exp1_results = exp1_trajectory_manifold(model, tokenizer, device, info, sentences)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    exp2_results = exp2_basin_boundary(model, tokenizer, device, info, sentences)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    exp3_results = exp3_lyapunov_stability(model, tokenizer, device, info, sentences)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    exp4_results = exp4_protocol_projection(model, tokenizer, device, info, sentences)

    # ===== Final Summary =====
    print("\n" + "="*60)
    print(f"PHASE 202 SUMMARY — {model_name}")
    print("="*60)

    print("\n1. Trajectory Manifold (Exp1):")
    print("   (see detailed output above)")

    print("\n2. Basin Boundary (Exp2):")
    for trans_name, r in exp2_results.items():
        print(f"   {trans_name}: max jump at step {r.get('max_jump_step', 'N/A')}, "
              f"jump={r.get('max_jump_val', 0):.4f}")

    print("\n3. Lyapunov Stability (Exp3):")
    if exp3_results and 'attractor_depth' in exp3_results:
        for mode, depth in sorted(exp3_results['attractor_depth'].items(), key=lambda x: x[1]):
            print(f"   {mode:<15} cosΔ = {depth:.4f} "
                  f"({'deep attractor' if depth < 0.02 else 'shallow attractor' if depth > 0.1 else 'medium attractor'})")

    print("\n4. Protocol Projection (Exp4):")
    if exp4_results:
        print(f"   Early layers: W_U projection = {exp4_results.get('avg_early', 0):.4f}")
        print(f"   Mid layers:   W_U projection = {exp4_results.get('avg_mid', 0):.4f}")
        print(f"   Late layers:  W_U projection = {exp4_results.get('avg_late', 0):.4f}")
        print(f"   Dark matter (mid): {exp4_results.get('dark_matter_mid', 0):.4f}")

    # Save results
    out_path = Path(f"tests/glm5_temp/phase202_{model_name}_results.json")

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

    save_data = {
        'model': model_name,
        'n_sentences': len(sentences),
        'exp1_manifold': {k: convert(v) for k, v in exp1_results.items()},
        'exp2_basin': {k: convert(v) for k, v in exp2_results.items()},
        'exp3_stability': {k: convert(v) for k, v in exp3_results.items()},
        'exp4_projection': {k: convert(v) for k, v in exp4_results.items()},
        'timestamp': datetime.now().isoformat(),
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Release
    elapsed = time.time() - t0
    print(f"\n[Phase 202] COMPLETE in {elapsed:.1f}s ({model_name})")
    release_model(model)


if __name__ == "__main__":
    main()
