"""
Phase 204: Transport Geometry — Full Spectrum, Eigenvector Transport, Cross-Task Overlap
=========================================================================================

Phase 203 → Phase 204 关键理论修正:
1. σ_max增长≠混沌 → 是"选择性放大"(selective amplification)
2. 真正关键不是σ_max而是完整谱分布 — 三类模态:
   - Stable modes (σ≈1): 长程稳定传播方向 — "概念载体"
   - Expanding modes (σ≫1): 注意力聚焦/决策分叉
   - Collapsing modes (σ≪1): 快速衰减方向
3. ker(W_U)≈{0}不推翻"暗物质" → 修正为"弱可观测子空间"(weakly observable subspace)
4. CoT不是"推理模块" → 而是"长程平滑信息整合机制"
5. 最关键缺失: Jacobian composition law — 为什么某些方向在T=J_{L-1}...J_0下仍稳定?

核心数学框架:
  h_{l+1} = h_l + F_l(h_l, p)    (p=prompt constraint field)
  δh_{l+1} = J_l · δh_l          (线性化传输)
  T_l = J_{l-1} · ... · J_0      (累计传输算子)

Phase 204实验:
  Exp1: Full Jacobian Spectrum — 完整奇异值分布, 三类模态分类
  Exp2: Eigenvector Transport — 子空间对齐度, 概念载体识别
  Exp3: Cross-Task Spectral Overlap — 跨任务谱重叠, 概念复用
  Exp4: Weakly Observable Subspace — W_U低σ方向动力学

术语修正 (Phase 203 → Phase 204):
  "chaos" → "selective amplification"
  "dark matter" → "weakly observable subspace"
  "protocol projection" → "optimization-induced convergence"

数据量 (关键实验加大):
  Qwen3: 25句×3mode×80probe, GLM4/DS7B: 15句×3mode×60probe
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

from model_utils import get_model_info, release_model, get_layers, get_W_U, MODEL_CONFIGS

warnings.filterwarnings('ignore')


# ========================================================================
# 基础句子 (增大到30句)
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
EXP_SENTENCES_Q = BASE_SENTENCES[:25]
EXP_SENTENCES_L = BASE_SENTENCES[:15]


# ========================================================================
# Mode Prompts
# ========================================================================
CORE_MODES = ["normal", "cot", "translation"]

MODE_PROMPTS = {
    "normal": lambda b: b,
    "cot": lambda b: "Problem: " + b[0].lower() + b[1:] + ". Think step by step.",
    "translation": lambda b: "Translate to Chinese: " + b,
}


# ========================================================================
# Model Loading — Flash Attention优先
# ========================================================================
def load_model_flash(model_name: str):
    """
    BF16 + Flash Attention优先加载
    尝试顺序: flash_attention_2 → sdpa → eager
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 依次尝试flash/sdpa/eager
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            model.eval()
            device = next(model.parameters()).device
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"[load] {model_name} loaded with attn={attn_impl}, "
                  f"class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
            return model, tokenizer, device
        except Exception as e:
            print(f"[load] {attn_impl} failed for {model_name}: {str(e)[:80]}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    raise RuntimeError(f"Failed to load {model_name} with any attention implementation")


# ========================================================================
# Utility
# ========================================================================
def log_progress(tag, current, total, t_start, extra=""):
    elapsed = time.time() - t_start
    gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  [{tag}] {current}/{total} ({elapsed:.0f}s) GPU={gpu:.2f}GB {extra}", flush=True)


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
# CORE: Perturbation Data Collection
# ========================================================================
def collect_perturbation_data(model, tokenizer, device, info, sentences, modes, n_probe=80):
    """
    核心数据收集: 扰动响应矩阵 D_l at each layer for each mode

    返回: {mode: {layer_idx: {
        'singular_values': np.array,      # top-k singular values
        'left_vectors': np.array,         # (d_model, k) top left singular vectors
        'sigma_max': float,
        'participation_ratio': float,
    }}}

    这是所有后续分析的基础数据。
    """
    n_layers = info.n_layers
    d_model = info.d_model
    epsilon = 0.01  # 1% of embedding norm

    # 采样层 (密集采样, 特别是首尾层)
    sample_layers = sorted(set(
        [0, 1, 2] +
        list(range(0, n_layers, max(1, n_layers // 15))) +
        [n_layers - 3, n_layers - 2, n_layers - 1]
    ))

    print(f"  Sample layers: {sample_layers}")
    print(f"  d_model={d_model}, n_probe={n_probe}, sentences={len(sentences)}")

    results = {mode: {} for mode in modes}

    for mode in modes:
        print(f"\n--- Collecting perturbation data: {mode} ---")
        t_start = time.time()

        # 累积扰动响应: {layer: [delta_vectors]}
        all_delta_by_layer = defaultdict(list)
        base_norm_avg = 0.0

        for si, base in enumerate(sentences):
            text = MODE_PROMPTS[mode](base)
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)

            # 基准前向
            with torch.no_grad():
                embed_layer = model.get_input_embeddings()
                base_embed = embed_layer(ids)

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
                delta = torch.randn(d_model, device=base_embed.device, dtype=base_embed.dtype)
                delta = delta / delta.norm() * epsilon * base_norm

                pert_embed = base_embed.clone()
                pert_embed[0, -1, :] += delta

                with torch.no_grad():
                    try:
                        out_pert = model(inputs_embeds=pert_embed, output_hidden_states=True)
                    except Exception:
                        continue

                for li in sample_layers:
                    if li < len(out_pert.hidden_states):
                        pert_h = out_pert.hidden_states[li][0, -1, :].detach().float().cpu().numpy()
                        if li in base_states:
                            delta_h = pert_h - base_states[li]
                            all_delta_by_layer[li].append(delta_h)

                del out_pert, pert_embed

            # 每句清理
            if (si + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log_progress(f"Perturb-{mode}", si + 1, len(sentences), t_start)

        base_norm_avg /= max(len(sentences), 1)

        # === SVD分析每层 ===
        print(f"\n  SVD analysis for '{mode}':")
        print(f"  {'Layer':>6} {'σ_max':>10} {'σ_min':>10} {'d_eff':>8} "
              f"{'n_expand':>9} {'n_stable':>9} {'n_collapse':>11} {'Interpretation':>20}")

        for li in sample_layers:
            deltas = all_delta_by_layer[li]
            if len(deltas) < 10:
                continue

            D = np.array(deltas).T  # (d_model, n_probes_total)

            # SVD — 计算足够多的分量用于谱分析
            n_svd = min(D.shape[1], D.shape[0], 300)
            try:
                U, S, Vt = np.linalg.svd(D[:, :n_svd], full_matrices=False)
            except Exception:
                continue

            # 分类: expanding(σ>3), stable(0.3<σ<3), collapsing(σ<0.3)
            # 注意: 这里σ是累计传输的奇异值, 不是每层的
            # 对于cumulative: expanding=σ>10, stable=0.1<σ<10, collapsing=σ<0.1
            # 用log10分类更合理
            log_s = np.log10(np.maximum(S, 1e-20))
            n_expanding = int(np.sum(log_s > 1.0))    # σ > 10
            n_stable = int(np.sum((log_s > -1.0) & (log_s <= 1.0)))  # 0.1 < σ ≤ 10
            n_collapsing = int(np.sum(log_s <= -1.0))  # σ ≤ 0.1

            sigma_max = float(S[0]) if len(S) > 0 else 0
            sigma_min = float(S[-1]) if len(S) > 0 else 0

            # 参与比
            total_energy = np.sum(S ** 2)
            pr = float((total_energy)**2 / np.sum(S ** 4)) if total_energy > 1e-20 else 0

            # 主要模式
            if n_expanding > n_collapsing:
                interp = "AMPLIFYING"
            elif n_collapsing > n_expanding:
                interp = "COMPRESSING"
            else:
                interp = "MIXED"

            print(f"  {li:6d} {sigma_max:10.6f} {sigma_min:10.6f} {pr:8.1f} "
                  f"{n_expanding:9d} {n_stable:9d} {n_collapsing:11d} {interp:>20}")

            # 保存top-k left singular vectors (用于eigenvector transport)
            k_save = min(100, U.shape[1])
            results[mode][li] = {
                'singular_values': S.tolist(),
                'left_vectors': U[:, :k_save].copy(),  # (d_model, k_save)
                'sigma_max': sigma_max,
                'sigma_min': sigma_min,
                'participation_ratio': pr,
                'n_expanding': n_expanding,
                'n_stable': n_stable,
                'n_collapsing': n_collapsing,
                'base_norm': base_norm_avg,
            }

        # 清理
        del all_delta_by_layer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, sample_layers


# ========================================================================
# EXP1: Full Jacobian Spectrum Analysis
# ========================================================================
def exp1_full_spectrum(perturb_data, sample_layers, n_layers, modes):
    """
    完整Jacobian谱分析 — 不只看σ_max, 看完整分布

    三类模态:
    - Expanding (σ > 10): 选择性放大方向 — 注意力聚焦/决策分叉
    - Stable (0.1 < σ ≤ 10): 稳定传输方向 — "概念载体"候选
    - Collapsing (σ ≤ 0.1): 快速衰减方向 — 被抑制的扰动

    关键问题:
    1. 谱形状如何随层数演化? (从"宽谱"→"窄谱"?)
    2. CoT vs Normal vs Translation的谱分布差异?
    3. Stable modes的比例在深层是增加还是减少?
    """
    print("\n" + "="*70)
    print("Exp1: Full Jacobian Spectrum Analysis")
    print("  ★ 核心修正: σ_max增长≠混沌, 而是选择性放大 ★")
    print("  ★ 真正关键: 完整谱分布的三类模态结构 ★")
    print("="*70)

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        print(f"  {'Layer':>6} {'σ_max':>10} {'σ_med':>10} {'σ_10':>10} "
              f"{'Expand%':>9} {'Stable%':>9} {'Collap%':>9} {'Spectral Shape':>20}")

        for li in sample_layers:
            if li not in perturb_data[mode]:
                continue
            d = perturb_data[mode][li]
            S = np.array(d['singular_values'])
            if len(S) < 5:
                continue

            n_total = len(S)
            pct_expand = d['n_expanding'] / n_total * 100
            pct_stable = d['n_stable'] / n_total * 100
            pct_collapse = d['n_collapsing'] / n_total * 100

            # 谱形状指标
            sigma_max = d['sigma_max']
            sigma_median = float(S[n_total // 2]) if n_total > 0 else 0
            sigma_p10 = float(S[int(n_total * 0.1)]) if n_total > 5 else 0

            # 分类谱形状
            if pct_expand > 50:
                shape = "AMPLIFY-DOMINANT"
            elif pct_collapse > 50:
                shape = "COLLAPSE-DOMINANT"
            elif pct_stable > 50:
                shape = "STABLE-DOMINANT"
            else:
                shape = "MIXED"

            print(f"  {li:6d} {sigma_max:10.4f} {sigma_median:10.6f} {sigma_p10:10.8f} "
                  f"{pct_expand:8.1f}% {pct_stable:8.1f}% {pct_collapse:8.1f}% {shape:>20}")

    # === 跨模式谱形状比较 ===
    print("\n--- Exp1 Summary: Spectral Shape Evolution ---")

    for mode in modes:
        expand_profile = []
        stable_profile = []
        collapse_profile = []
        for li in sample_layers:
            if li not in perturb_data[mode]:
                continue
            d = perturb_data[mode][li]
            S = np.array(d['singular_values'])
            n_total = len(S)
            if n_total < 5:
                continue
            expand_profile.append((li, d['n_expanding'] / n_total))
            stable_profile.append((li, d['n_stable'] / n_total))
            collapse_profile.append((li, d['n_collapsing'] / n_total))

        if expand_profile:
            # 找expanding比例峰值层
            peak_expand_li, peak_expand_val = max(expand_profile, key=lambda x: x[1])
            # 找stable比例峰值层
            peak_stable_li, peak_stable_val = max(stable_profile, key=lambda x: x[1])

            print(f"  {mode}: peak expanding at L{peak_expand_li} ({peak_expand_val:.1%}), "
                  f"peak stable at L{peak_stable_li} ({peak_stable_val:.1%})")

    # === CoT vs Normal谱差异 ===
    print("\n--- Exp1 Key: CoT vs Normal Spectral Difference ---")
    if "normal" in perturb_data and "cot" in perturb_data:
        for li in sample_layers:
            if li not in perturb_data["normal"] or li not in perturb_data["cot"]:
                continue
            n_d = perturb_data["normal"][li]
            c_d = perturb_data["cot"][li]
            n_S = np.array(n_d['singular_values'])
            c_S = np.array(c_d['singular_values'])
            n_total = len(n_S)

            # 奇异值比: CoT/Normal 在top-1, top-10, top-50
            n_expand = n_d['n_expanding'] / max(n_total, 1)
            c_expand = c_d['n_expanding'] / max(len(c_S), 1)
            n_stable = n_d['n_stable'] / max(n_total, 1)
            c_stable = c_d['n_stable'] / max(len(c_S), 1)

            if li in [0, 1, n_layers // 2, n_layers - 2, n_layers - 1] or li % 5 == 0:
                print(f"  L{li}: σ_max ratio={c_d['sigma_max']/max(n_d['sigma_max'],1e-20):.2f}, "
                      f"expand% {n_expand:.1%}→{c_expand:.1%}, "
                      f"stable% {n_stable:.1%}→{c_stable:.1%}")

    return


# ========================================================================
# EXP2: Eigenvector Transport (特征向量传输 — 概念载体识别)
# ========================================================================
def exp2_eigenvector_transport(perturb_data, sample_layers, n_layers, modes, k_subspace=50):
    """
    特征向量传输分析 — 概念载体识别

    核心思想:
    T_l的左奇异向量U_l描述"hidden state空间中扰动响应最强的方向"
    如果U_l的top-k子空间和U_{l+1}的top-k子空间高度对齐,
    说明这些方向是"稳定传输方向" — 概念载体候选

    子空间对齐度:
    alignment(U1, U2) = ||U1^T U2||_F^2 / k
    - 1.0 = 完全对齐 (同一子空间)
    - k/d_model = 随机基线
    - 0.0 = 正交

    持续性 (persistence):
    方向在多少连续层中保持对齐?
    高持续性 → 强概念载体
    """
    print("\n" + "="*70)
    print("Exp2: Eigenvector Transport — Concept Carrier Identification")
    print("  ★ 核心问题: 哪些方向跨层保持稳定? → '概念载体' ★")
    print("  ★ Jacobian composition law: 为什么某些方向在T=J_{L-1}...J_0下仍稳定? ★")
    print("="*70)

    d_model = None
    for mode in modes:
        for li in sample_layers:
            if li in perturb_data[mode] and 'left_vectors' in perturb_data[mode][li]:
                d_model = perturb_data[mode][li]['left_vectors'].shape[0]
                break
        if d_model is not None:
            break

    if d_model is None:
        print("  No left_vectors data available!")
        return

    random_baseline = k_subspace / d_model
    print(f"  d_model={d_model}, k_subspace={k_subspace}, random_baseline={random_baseline:.4f}")

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        print(f"  {'Layer':>6} {'→Next':>6} {'Alignment':>12} {'vs_random':>12} {'Interpretation':>25}")

        prev_U = None
        prev_li = None
        alignments = []

        for li in sample_layers:
            if li not in perturb_data[mode]:
                prev_U = None
                prev_li = None
                continue

            U = perturb_data[mode][li]['left_vectors'][:, :k_subspace]  # (d_model, k)

            if prev_U is not None and U.shape[1] >= k_subspace and prev_U.shape[1] >= k_subspace:
                # 子空间对齐: ||U_prev^T U||_F^2 / k
                M = prev_U.T @ U  # (k, k)
                alignment = float(np.sum(M ** 2)) / k_subspace

                # vs random
                vs_random = alignment / random_baseline

                # 解释
                if alignment > 0.8:
                    interp = "STRONG PERSISTENCE"
                elif alignment > 0.5:
                    interp = "MODERATE PERSISTENCE"
                elif alignment > 0.3:
                    interp = "WEAK PERSISTENCE"
                else:
                    interp = "NO PERSISTENCE"

                print(f"  {prev_li:6d} →{li:5d} {alignment:12.4f} {vs_random:12.1f}x {interp:>25}")
                alignments.append((prev_li, li, alignment))

            prev_U = U.copy()
            prev_li = li

        # === 持续性分析 ===
        if alignments:
            avg_align = np.mean([a[2] for a in alignments])
            strong_count = sum(1 for a in alignments if a[2] > 0.8)
            print(f"\n  Persistence Summary for '{mode}':")
            print(f"    Avg alignment: {avg_align:.4f} ({avg_align/random_baseline:.1f}x random)")
            print(f"    Strong persistence segments (>0.8): {strong_count}/{len(alignments)}")

            # 找"最长持续段"
            max_run = 0
            current_run = 0
            for _, _, a in alignments:
                if a > 0.5:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            print(f"    Longest moderate+ persistence run: {max_run} consecutive layers")

    # === 跨模式对齐比较 ===
    print("\n--- Exp2 Key: Cross-Mode Alignment at Same Layer ---")
    print("  (Do different tasks share the same transport directions?)")

    if all(m in perturb_data for m in modes):
        for li in sample_layers:
            if not all(li in perturb_data[m] for m in modes):
                continue
            if not all('left_vectors' in perturb_data[m][li] for m in modes):
                continue

            # 只打印关键层
            if li not in [0, 1, n_layers // 2, n_layers - 2, n_layers - 1]:
                if li % 5 != 0:
                    continue

            Us = {m: perturb_data[m][li]['left_vectors'][:, :k_subspace] for m in modes}

            # 计算模式间的对齐度
            pairs = [("normal", "cot"), ("normal", "translation"), ("cot", "translation")]
            print(f"  L{li}:")
            for m1, m2 in pairs:
                M = Us[m1].T @ Us[m2]
                align = float(np.sum(M ** 2)) / k_subspace
                print(f"    {m1[:4]}-{m2[:4]}: alignment={align:.4f} ({align/random_baseline:.1f}x random)")

    return


# ========================================================================
# EXP3: Cross-Task Spectral Overlap (跨任务谱重叠 — 概念复用)
# ========================================================================
def exp3_cross_task_overlap(perturb_data, sample_layers, n_layers, modes, k_subspace=50):
    """
    跨任务谱重叠 — 概念复用的数学基础

    核心问题:
    不同任务(normal/CoT/translation)是否共享相同的稳定模态?
    如果共享 → "概念复用" (concept reuse) 有数学基础
    如果不共享 → 每种任务有完全独立的传输走廊

    方法:
    1. 提取各模式在各层的top-k左奇异向量
    2. 计算跨模式子空间重叠度
    3. 识别"共享方向"和"专有方向"
    """
    print("\n" + "="*70)
    print("Exp3: Cross-Task Spectral Overlap — Concept Reuse")
    print("  ★ 核心问题: 不同任务是否共享相同的稳定模态? ★")
    print("  ★ 共享→概念复用, 不共享→独立传输走廊 ★")
    print("="*70)

    d_model = None
    for mode in modes:
        for li in sample_layers:
            if li in perturb_data[mode] and 'left_vectors' in perturb_data[mode][li]:
                d_model = perturb_data[mode][li]['left_vectors'].shape[0]
                break
        if d_model is not None:
            break

    if d_model is None:
        print("  No data available!")
        return

    random_baseline = k_subspace / d_model

    # === 逐层分析 ===
    print(f"\n  {'Layer':>6}", end="")
    for i, m1 in enumerate(modes):
        for m2 in modes[i+1:]:
            print(f" {m1[:4]}-{m2[:4]:>4}", end="")
    print(f"  {'Avg':>8} {'vs_random':>10}")

    overlap_by_layer = {}

    for li in sample_layers:
        if not all(li in perturb_data[m] for m in modes):
            continue
        if not all('left_vectors' in perturb_data[m][li] for m in modes):
            continue

        Us = {m: perturb_data[m][li]['left_vectors'][:, :k_subspace] for m in modes}

        overlaps = []
        for i, m1 in enumerate(modes):
            for m2 in modes[i+1:]:
                M = Us[m1].T @ Us[m2]
                align = float(np.sum(M ** 2)) / k_subspace
                overlaps.append(align)

        avg_overlap = np.mean(overlaps)
        print(f"  {li:6d}", end="")
        for ov in overlaps:
            print(f" {ov:9.4f}", end="")
        print(f" {avg_overlap:8.4f} {avg_overlap/random_baseline:10.1f}x")

        overlap_by_layer[li] = avg_overlap

    # === 演化趋势 ===
    print(f"\n--- Exp3: Overlap Evolution (early → mid → late) ---")

    for mode in modes:
        early_layers = [li for li in sample_layers if li < n_layers // 3 and li in perturb_data.get(mode, {})]
        mid_layers = [li for li in sample_layers if n_layers // 3 <= li < 2 * n_layers // 3 and li in perturb_data.get(mode, {})]
        late_layers = [li for li in sample_layers if li >= 2 * n_layers // 3 and li in perturb_data.get(mode, {})]

    early_overlaps = [overlap_by_layer[li] for li in overlap_by_layer if li < n_layers // 3]
    mid_overlaps = [overlap_by_layer[li] for li in overlap_by_layer if n_layers // 3 <= li < 2 * n_layers // 3]
    late_overlaps = [overlap_by_layer[li] for li in overlap_by_layer if li >= 2 * n_layers // 3]

    if early_overlaps:
        print(f"  Early: avg overlap = {np.mean(early_overlaps):.4f} ({np.mean(early_overlaps)/random_baseline:.1f}x random)")
    if mid_overlaps:
        print(f"  Mid:   avg overlap = {np.mean(mid_overlaps):.4f} ({np.mean(mid_overlaps)/random_baseline:.1f}x random)")
    if late_overlaps:
        print(f"  Late:  avg overlap = {np.mean(late_overlaps):.4f} ({np.mean(late_overlaps)/random_baseline:.1f}x random)")

    # === 共享vs专有方向 ===
    print(f"\n--- Exp3: Shared vs Mode-Specific Directions ---")

    for li in [n_layers // 2, n_layers - 1]:
        if not all(li in perturb_data[m] for m in modes):
            continue
        if not all('left_vectors' in perturb_data[m][li] for m in modes):
            continue

        Us = {m: perturb_data[m][li]['left_vectors'][:, :k_subspace] for m in modes}

        # 合并所有模式的top-k向量
        all_U = np.concatenate(list(Us.values()), axis=1)  # (d_model, 3*k)

        # SVD of combined matrix → 找"共享方向"
        U_comb, S_comb, Vt_comb = np.linalg.svd(all_U, full_matrices=False)

        # 重叠度: 前k个奇异值的能量
        total_energy = np.sum(S_comb ** 2)
        top_k_energy = np.sum(S_comb[:k_subspace] ** 2)
        overlap_ratio = top_k_energy / max(total_energy, 1e-20)

        # 理论最大: 3个模式完全独立 → top-k capture 1/3
        # 理论最小: 3个模式完全共享 → top-k capture 1.0
        print(f"  L{li}: combined top-{k_subspace} captures {overlap_ratio:.1%} of total energy")
        print(f"    (1.0 = fully shared, 0.33 = fully independent)")

    return overlap_by_layer


# ========================================================================
# EXP4: Weakly Observable Subspace (弱可观测子空间)
# ========================================================================
def exp4_weakly_observable(model, tokenizer, device, info, perturb_data, sample_layers, modes):
    """
    弱可观测子空间分析

    核心修正 (Phase 203 → Phase 204):
    Phase 203发现ker(W_U)≈{0} → 不存在真正的零空间
    但这不推翻"暗物质" → 修正为"弱可观测子空间"

    定义: W_obs = span{v_i : σ_i(W_U) < threshold}
    其中v_i是W_U的右奇异向量, σ_i是对应的奇异值

    "弱可观测"意味着:
    - 虽然理论上可恢复(因为ker(W_U)≈{0})
    - 但在实际输出中影响极小(因为σ_i≪σ_max)
    - 等价于"有效零空间"

    关键问题:
    1. 弱可观测子空间有多大?
    2. hidden state的多少能量在弱可观测方向上?
    3. 扰动传输的多少发生在弱可观测方向上? → "计算"vs"语言"
    4. CoT和Normal在弱可观测方向上的差异?
    """
    print("\n" + "="*70)
    print("Exp4: Weakly Observable Subspace Analysis")
    print("  ★ 核心修正: ker(W_U)≈{0}不推翻暗物质 → 弱可观测子空间 ★")
    print("  ★ σ_i≪σ_max的方向在实际动力学中等价于'有效零空间' ★")
    print("="*70)

    n_layers = info.n_layers
    d_model = info.d_model
    mid_layer = n_layers // 2

    # === Step 1: W_U的完整SVD ===
    print("\n  Loading W_U and computing full SVD...")
    W_U = get_W_U(model, info.name if hasattr(info, 'name') else None)
    print(f"  W_U shape: {W_U.shape}")

    # 完整SVD (不再用截断SVD!)
    print("  Computing full SVD of W_U...")
    W_U_f32 = W_U.astype(np.float32)
    # 对于大矩阵, 使用截断SVD但取足够多的分量
    from scipy.sparse.linalg import svds
    k_full = min(d_model - 2, min(W_U_f32.shape) - 2, 2000)
    U_wu, sigma_wu, Vt_wu = svds(W_U_f32, k=k_full)

    # 排序
    sort_idx = np.argsort(sigma_wu)[::-1]
    sigma_wu = sigma_wu[sort_idx]
    Vt_wu = Vt_wu[sort_idx]
    U_wu = U_wu[:, sort_idx]

    V_full = Vt_wu.T  # (d_model, k_full) — 右奇异向量

    print(f"  W_U SVD: top-5 σ = {sigma_wu[:5].tolist()}")
    print(f"  W_U SVD: bottom-5 σ = {sigma_wu[-5:].tolist()}")
    print(f"  Condition number: σ_max/σ_min = {sigma_wu[0]/max(sigma_wu[-1], 1e-20):.2f}")

    # === Step 2: 定义弱可观测子空间 ===
    # 使用百分位阈值: σ < σ_max * 1% 的方向为"弱可观测"
    threshold = sigma_wu[0] * 0.01
    n_weak = int(np.sum(sigma_wu < threshold))
    n_strong = k_full - n_weak

    # 也使用能量阈值: 累计90%能量的方向为"强可观测"
    cum_energy = np.cumsum(sigma_wu ** 2) / np.sum(sigma_wu ** 2)
    n_90 = int(np.searchsorted(cum_energy, 0.9) + 1)

    print(f"\n  Weakly observable subspace definition:")
    print(f"    Threshold: σ < {threshold:.4f} (1% of σ_max={sigma_wu[0]:.2f})")
    print(f"    n_weak (σ<threshold): {n_weak}/{k_full} ({n_weak/k_full:.1%})")
    print(f"    n_90 (90% energy): {n_90}/{k_full}")
    print(f"    n_strong = {n_strong}/{k_full}")

    # 定义投影矩阵
    V_strong = V_full[:, :n_90]    # (d_model, n_90) — 强可观测方向
    V_weak = V_full[:, n_90:]      # (d_model, k_full-n_90) — 弱可观测方向

    # === Step 3: 分析hidden state能量分布 ===
    print(f"\n--- Exp4A: Hidden State Energy in Weakly Observable Subspace ---")

    for mode in modes:
        print(f"\n  Mode: {mode}")
        print(f"  {'Layer':>6} {'Total Energy':>13} {'Strong%':>9} {'Weak%':>9} "
              f"{'Weak|Strong ratio':>18}")

        # 用采样层的hidden state分析
        for li in sample_layers:
            if li not in perturb_data.get(mode, {}):
                continue
            d = perturb_data[mode][li]
            if 'left_vectors' not in d:
                continue

            # 用左奇异向量作为"典型方向"的代理
            # 更好的方法: 直接用hidden states, 但我们这里用singular vectors作为近似
            # 左奇异向量的能量分布反映扰动响应的能量分布
            U = d['left_vectors'][:, :min(50, d['left_vectors'].shape[1])]

            # 每个左奇异向量在强/弱可观测方向上的投影
            strong_energies = []
            weak_energies = []
            for vi in range(U.shape[1]):
                v = U[:, vi]
                proj_strong = V_strong @ (V_strong.T @ v)
                proj_weak = v - proj_strong
                strong_energies.append(np.sum(proj_strong ** 2))
                weak_energies.append(np.sum(proj_weak ** 2))

            avg_strong_pct = np.mean(strong_energies) / max(np.mean(strong_energies) + np.mean(weak_energies), 1e-20)
            avg_weak_pct = 1 - avg_strong_pct

            # 只打印关键层
            if li in [0, 1, n_layers // 2, n_layers - 2, n_layers - 1] or li % 5 == 0:
                ratio = avg_weak_pct / max(avg_strong_pct, 1e-20)
                print(f"  {li:6d} {'—':>13} {avg_strong_pct:9.1%} {avg_weak_pct:9.1%} {ratio:18.3f}")

    # === Step 4: 扰动传输在弱可观测方向上的分布 ===
    print(f"\n--- Exp4B: Perturbation Transport in Weakly Observable Subspace ---")
    print("  (How much of the 'transport' happens in weakly observable directions?)")

    for mode in modes:
        print(f"\n  Mode: {mode}")

        for li in [0, 1, n_layers // 2, n_layers - 2, n_layers - 1]:
            if li not in perturb_data.get(mode, {}):
                continue
            d = perturb_data[mode][li]
            if 'left_vectors' not in d:
                continue

            U = d['left_vectors']
            S = np.array(d['singular_values'])

            # 扰动响应在强/弱可观测方向上的能量
            # Δ_l = U Σ V^T
            # Δ_l在V_strong上的投影能量 = ||V_strong^T Δ_l||_F^2
            #                              = ||V_strong^T U Σ||_F^2

            n_s = min(100, U.shape[1])
            U_s = U[:, :n_s]
            S_s = S[:n_s]

            # V_strong^T U_s: (n_90, n_s)
            V_strong_U = V_strong.T @ U_s  # (n_90, n_s)
            # 加权: V_strong_U * S_s
            weighted_strong = V_strong_U * S_s[np.newaxis, :]  # (n_90, n_s)
            energy_strong = float(np.sum(weighted_strong ** 2))

            # 总能量
            weighted_total = U_s * S_s[np.newaxis, :]
            energy_total = float(np.sum(weighted_total ** 2))

            energy_weak = energy_total - energy_strong
            weak_pct = energy_weak / max(energy_total, 1e-20)

            print(f"  L{li}: weak_observable transport = {weak_pct:.1%} "
                  f"(strong={1-weak_pct:.1%})")

    # === Step 5: CoT vs Normal差异 ===
    print(f"\n--- Exp4C: CoT vs Normal in Weakly Observable Subspace ---")

    if "normal" in perturb_data and "cot" in perturb_data:
        for li in [n_layers // 2, n_layers - 1]:
            if li not in perturb_data["normal"] or li not in perturb_data["cot"]:
                continue
            for attr in ['singular_values']:
                n_S = np.array(perturb_data["normal"][li][attr])
                c_S = np.array(perturb_data["cot"][li][attr])

                if len(n_S) > 0 and len(c_S) > 0:
                    print(f"  L{li}: Normal σ_max={n_S[0]:.6f}, CoT σ_max={c_S[0]:.6f}, "
                          f"ratio={c_S[0]/max(n_S[0], 1e-20):.2f}")

    # === Step 6: 弱可观测子空间的"功能性" ===
    print(f"\n--- Exp4 Summary: Weakly Observable Subspace ---")
    print(f"  W_U condition number: {sigma_wu[0]/max(sigma_wu[-1], 1e-20):.2f}")
    print(f"  n_weak (σ < 1% σ_max): {n_weak}/{k_full} ({n_weak/k_full:.1%})")
    print(f"  n_90 (90% energy): {n_90}/{k_full} ({n_90/k_full:.1%})")

    if n_weak > k_full * 0.3:
        print(f"  ★ {n_weak/k_full:.0%} of W_U directions are 'weakly observable'")
        print(f"  ★ This IS the 'dark matter' — not in ker(W_U), but practically invisible to output")
    else:
        print(f"  Only {n_weak/k_full:.0%} of directions are weakly observable")

    return {
        'n_weak': n_weak,
        'n_strong': n_strong,
        'n_90': n_90,
        'k_full': k_full,
        'sigma_wu_top5': sigma_wu[:5].tolist(),
        'sigma_wu_bottom5': sigma_wu[-5:].tolist(),
        'condition_number': float(sigma_wu[0] / max(sigma_wu[-1], 1e-20)),
    }


# ========================================================================
# MAIN
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    is_lite = model_name != "qwen3"

    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"PHASE 204: TRANSPORT GEOMETRY — {model_name}")
    print(f"{'='*70}")
    print(f"  Time: {datetime.now()}")
    print(f"  Lite mode: {is_lite}")
    print(f"  Theory: Selective Transport Geometry (not chaos, not attractors)")
    print(f"  Core objects:")
    print(f"    - Full Jacobian spectrum (stable/expanding/collapsing modes)")
    print(f"    - Eigenvector transport (subspace alignment → concept carriers)")
    print(f"    - Cross-task spectral overlap (concept reuse)")
    print(f"    - Weakly observable subspace (practical nullspace)")

    # Load model
    model, tokenizer, device = load_model_flash(model_name)
    info = get_model_info(model, model_name)
    print(f"[load] {model_name}: {info.model_class}, {info.n_layers}L, d={info.d_model}")

    # Config
    if is_lite:
        sentences = EXP_SENTENCES_L
        n_probe = 60
    else:
        sentences = EXP_SENTENCES_Q
        n_probe = 80

    print(f"  Sentences: {len(sentences)}, Probes: {n_probe}")
    print(f"  Modes: {CORE_MODES}")

    # ==========================================
    # Step 1: Collect perturbation data
    # ==========================================
    print(f"\n{'='*70}")
    print("Step 1: Collecting perturbation data (this is the expensive part)...")
    print(f"{'='*70}")

    perturb_data, sample_layers = collect_perturbation_data(
        model, tokenizer, device, info, sentences, CORE_MODES, n_probe=n_probe)

    # ==========================================
    # Step 2: Exp1 — Full Spectrum
    # ==========================================
    exp1_full_spectrum(perturb_data, sample_layers, info.n_layers, CORE_MODES)

    # ==========================================
    # Step 3: Exp2 — Eigenvector Transport
    # ==========================================
    exp2_eigenvector_transport(perturb_data, sample_layers, info.n_layers, CORE_MODES, k_subspace=50)

    # ==========================================
    # Step 4: Exp3 — Cross-Task Overlap
    # ==========================================
    exp3_cross_task_overlap(perturb_data, sample_layers, info.n_layers, CORE_MODES, k_subspace=50)

    # ==========================================
    # Step 5: Exp4 — Weakly Observable Subspace
    # ==========================================
    exp4_results = exp4_weakly_observable(model, tokenizer, device, info,
                                          perturb_data, sample_layers, CORE_MODES)

    # ==========================================
    # Final Summary
    # ==========================================
    print(f"\n{'='*70}")
    print(f"PHASE 204 SUMMARY — {model_name}")
    print(f"{'='*70}")

    mid = info.n_layers // 2

    print("\n1. Full Spectrum (Exp1):")
    for mode in CORE_MODES:
        if mid in perturb_data.get(mode, {}):
            d = perturb_data[mode][mid]
            S = np.array(d['singular_values'])
            n_total = len(S)
            print(f"   {mode} L{mid}: σ_max={d['sigma_max']:.4f}, "
                  f"expand={d['n_expanding']/max(n_total,1):.0%}, "
                  f"stable={d['n_stable']/max(n_total,1):.0%}, "
                  f"collapse={d['n_collapsing']/max(n_total,1):.0%}")

    print("\n2. Eigenvector Transport (Exp2):")
    for mode in CORE_MODES:
        if mid in perturb_data.get(mode, {}) and mid+1 in perturb_data.get(mode, {}):
            U1 = perturb_data[mode][mid]['left_vectors'][:, :50]
            U2 = perturb_data[mode][mid+1 if mid+1 in perturb_data[mode] else mid]['left_vectors'][:, :50]
            # Simple alignment check
            M = U1.T @ U2
            align = float(np.sum(M ** 2)) / 50
            print(f"   {mode} L{mid}→L{mid+1 if mid+1 in perturb_data.get(mode,{}) else mid}: alignment={align:.4f}")

    print("\n3. Weakly Observable Subspace (Exp4):")
    if exp4_results:
        print(f"   W_U condition number: {exp4_results.get('condition_number', 'N/A')}")
        print(f"   n_weak: {exp4_results.get('n_weak', 'N/A')}/{exp4_results.get('k_full', 'N/A')}")
        print(f"   n_90: {exp4_results.get('n_90', 'N/A')}/{exp4_results.get('k_full', 'N/A')}")

    # Save results
    out_path = Path(f"tests/glm5_temp/phase204_{model_name}_results.json")

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

    # 轻量保存 (不保存left_vectors, 太大)
    save_data = {
        'model': model_name,
        'n_layers': info.n_layers,
        'd_model': info.d_model,
        'timestamp': datetime.now().isoformat(),
        'exp4': {k: convert(v) for k, v in exp4_results.items()} if exp4_results else {},
        'spectrum_summary': {},
    }

    for mode in CORE_MODES:
        save_data['spectrum_summary'][mode] = {}
        for li in sample_layers:
            if li in perturb_data.get(mode, {}):
                d = perturb_data[mode][li]
                S = np.array(d['singular_values'])
                save_data['spectrum_summary'][mode][str(li)] = {
                    'sigma_max': convert(d['sigma_max']),
                    'sigma_min': convert(d['sigma_min']),
                    'sigma_median': convert(float(S[len(S)//2])) if len(S) > 0 else 0,
                    'participation_ratio': convert(d['participation_ratio']),
                    'n_expanding': convert(d['n_expanding']),
                    'n_stable': convert(d['n_stable']),
                    'n_collapsing': convert(d['n_collapsing']),
                    'top10_sv': [convert(s) for s in S[:10].tolist()],
                    'bottom10_sv': [convert(s) for s in S[-10:].tolist()],
                }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Release
    elapsed = time.time() - t0
    print(f"\n[Phase 204] COMPLETE in {elapsed:.1f}s ({model_name})")
    release_model(model)


if __name__ == "__main__":
    main()
