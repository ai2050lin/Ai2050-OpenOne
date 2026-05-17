"""
Phase 205: Covariance Transport Geometry
=========================================

Phase 204 → Phase 205 关键理论跃迁:
  从"方向"→"约束几何"(constraint geometry)

核心修正:
1. concept不是单向量 → 是分布式子空间编码(distributed subspace code)
2. 没有STRONG persistence → 因为系统保持的不是方向而是关系结构
3. 真正稳定传播的是协方差结构 Cov(h_l) 而非单向量
4. Normal-Translation共享约束结构 → CoT构建独立约束几何
5. 弱可观测子空间承载>50%计算 → "隐藏计算储层"(hidden computational reservoir)

核心数学框架:
  方向传输: δh_{l+1} = J_l · δh_l          (Phase 204)
  协方差传输: Σ_{l+1} ≈ J_l Σ_l J_l^T + Q_l  (Phase 205, 新增!)
  
  其中 Σ_l = Cov(H_l) 是隐藏状态的协方差矩阵
  Q_l 是过程噪声(非线性残余)

关键假设:
  如果协方差子空间对齐度 > 方向子空间对齐度
  → "约束几何"比"方向"更稳定
  → 概念以关系结构编码，而非单向量

五组实验:
  Exp1: Natural Covariance Spectrum — Σ_l特征值演化
  Exp2: Covariance Subspace Alignment — 约束几何保持度(核心!)
  Exp3: Covariance vs Direction Alignment — 约束>方向? (核心对比!)
  Exp4: Jacobian Composition Law — 多层级联T稳定性
  Exp5: Cross-task Covariance Overlap — 跨任务约束结构共享

数据量 (加大):
  50 sentences per mode (更充分的协方差估计)
  3 modes: normal, cot, translation
  perturbation: 10 sentences × 60 probes (for Jacobian)
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

LITE = os.environ.get('LITE', '1') == '1'


# ========================================================================
# 句子集 (50句, 更充分的协方差估计)
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
    "The manager directs the project",
    "The professor explains the theory",
    "The technician fixes the machine",
    "The designer creates the product",
    "The researcher studies the phenomenon",
    "The editor revises the document",
    "The consultant advises the client",
    "The inspector checks the facility",
    "The coordinator organizes the event",
    "The administrator manages the system",
    "The operator runs the equipment",
    "The supervisor oversees the operation",
    "The specialist analyzes the sample",
    "The practitioner applies the method",
    "The instructor teaches the course",
    "The assistant supports the team",
    "The representative presents the proposal",
    "The candidate applies for the position",
    "The witness describes the incident",
    "The observer records the behavior",
]

# Lite mode: fewer sentences
N_COV = 15 if LITE else 50  # for covariance estimation
N_PERTURB = 10 if LITE else 15  # for Jacobian estimation


# ========================================================================
# Mode Prompts
# ========================================================================
MODE_PROMPTS = {
    "normal": lambda b: b,
    "cot": lambda b: "Problem: " + b[0].lower() + b[1:] + ". Think step by step.",
    "translation": lambda b: "Translate to Chinese: " + b,
}
CORE_MODES = ["normal", "cot", "translation"]


# ========================================================================
# Model Loading
# ========================================================================
def load_model_flash(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16 + device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=attn_impl)
            model.eval()
            device = next(model.parameters()).device
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"[load] {model_name} loaded with attn={attn_impl}, "
                  f"class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
            return model, tokenizer, device
        except Exception as e:
            print(f"[load] {attn_impl} failed for {model_name}: {str(e)[:80]}")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue
    raise RuntimeError(f"Failed to load {model_name}")


def log_progress(tag, current, total, t_start, extra=""):
    elapsed = time.time() - t_start
    gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  [{tag}] {current}/{total} ({elapsed:.0f}s) GPU={gpu:.2f}GB {extra}", flush=True)


# ========================================================================
# Step 1: Collect Natural Hidden States for Covariance Estimation
# ========================================================================
def collect_hidden_states(model, tokenizer, device, info, sentences, modes):
    """
    Collect natural hidden states (no perturbation) for covariance estimation.
    Returns: {mode: {layer_idx: np.array (n_tokens, d_model)}}
    """
    n_layers = info.n_layers
    d_model = info.d_model
    sample_layers = sorted(set(
        [0, 1, 2] +
        list(range(0, n_layers, max(1, n_layers // 15))) +
        [n_layers - 3, n_layers - 2, n_layers - 1]
    ))

    print(f"  Sample layers: {sample_layers}")
    print(f"  d_model={d_model}, sentences={len(sentences)}, modes={modes}")

    results = {mode: {li: [] for li in sample_layers} for mode in modes}

    for mode in modes:
        print(f"\n--- Collecting hidden states: {mode} ---")
        t_start = time.time()

        for si, base in enumerate(sentences):
            text = MODE_PROMPTS[mode](base)
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)

            with torch.no_grad():
                try:
                    out = model(input_ids=ids, output_hidden_states=True)
                except Exception:
                    attn_mask = torch.ones_like(ids)
                    out = model(input_ids=ids, attention_mask=attn_mask, output_hidden_states=True)

            # Collect ALL token positions (not just last) for richer covariance
            for li in sample_layers:
                if li < len(out.hidden_states):
                    # All tokens except BOS/EOS special tokens
                    h = out.hidden_states[li][0, :, :].detach().float().cpu().numpy()
                    results[mode][li].append(h)

            del out
            if (si + 1) % 10 == 0 or si == len(sentences) - 1:
                log_progress(f"HS-{mode}", si + 1, len(sentences), t_start)

    # Stack into arrays
    for mode in modes:
        for li in sample_layers:
            if results[mode][li]:
                results[mode][li] = np.vstack(results[mode][li])
            else:
                results[mode][li] = np.zeros((1, d_model))

    return results, sample_layers


# ========================================================================
# Step 2: Collect Perturbation Data for Jacobian Estimation
# ========================================================================
def collect_perturbation_data(model, tokenizer, device, info, sentences, modes, n_probe=60):
    """
    Collect perturbation responses for Jacobian estimation.
    Returns: {mode: {layer_idx: {'U': np.array, 'V': np.array, 'sigma': np.array}}}
    """
    n_layers = info.n_layers
    d_model = info.d_model
    epsilon = 0.01

    sample_layers = sorted(set(
        [0, 1, 2] +
        list(range(0, n_layers, max(1, n_layers // 15))) +
        [n_layers - 3, n_layers - 2, n_layers - 1]
    ))

    results = {mode: {} for mode in modes}

    for mode in modes:
        print(f"\n--- Collecting perturbation data: {mode} ---")
        t_start = time.time()

        all_dh_in = defaultdict(list)
        all_dh_out = defaultdict(list)

        for si, base in enumerate(sentences):
            text = MODE_PROMPTS[mode](base)
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)

            # Base forward
            with torch.no_grad():
                embed_layer = model.get_input_embeddings()
                base_embed = embed_layer(ids)
                try:
                    out_base = model(input_ids=ids, output_hidden_states=True)
                except Exception:
                    attn_mask = torch.ones_like(ids)
                    out_base = model(input_ids=ids, attention_mask=attn_mask, output_hidden_states=True)

            base_states = {}
            for li in sample_layers:
                if li < len(out_base.hidden_states):
                    base_states[li] = out_base.hidden_states[li][0, -1, :].detach().float().cpu().numpy()
            del out_base

            # Perturbation forward
            for pi in range(n_probe):
                delta = torch.randn_like(base_embed) * epsilon
                perturbed_embed = base_embed + delta

                with torch.no_grad():
                    try:
                        out_pert = model(inputs_embeds=perturbed_embed, output_hidden_states=True)
                    except Exception:
                        continue

                for li in sample_layers:
                    if li < len(out_pert.hidden_states) and li in base_states:
                        h_pert = out_pert.hidden_states[li][0, -1, :].detach().float().cpu().numpy()
                        dh_out = h_pert - base_states[li]
                        dh_in = delta[0, -1, :].float().cpu().numpy()
                        all_dh_in[li].append(dh_in)
                        all_dh_out[li].append(dh_out)

                del out_pert

            if (si + 1) % 5 == 0 or si == len(sentences) - 1:
                log_progress(f"Perturb-{mode}", si + 1, len(sentences), t_start)

        # SVD for Jacobian approximation
        for li in sample_layers:
            if len(all_dh_in[li]) > 10:
                Dh_in = np.array(all_dh_in[li])   # (n, d)
                Dh_out = np.array(all_dh_out[li])  # (n, d)

                # SVD of perturbation matrices
                k = min(50, min(Dh_in.shape) - 1, min(Dh_out.shape) - 1)
                if k < 1: continue

                U_in, s_in, Vt_in = np.linalg.svd(Dh_in, full_matrices=False)
                U_out, s_out, Vt_out = np.linalg.svd(Dh_out, full_matrices=False)

                results[mode][li] = {
                    'V_in': Vt_in[:k],       # (k, d) - input singular vectors
                    'U_out': U_out[:, :k].T,  # (k, d) - output singular vectors
                    'sigma_in': s_in[:k],
                    'sigma_out': s_out[:k],
                    'Dh_in': Dh_in,
                    'Dh_out': Dh_out,
                }

    return results, sample_layers


# ========================================================================
# Exp1: Natural Covariance Spectrum
# ========================================================================
def exp1_covariance_spectrum(hidden_states, sample_layers, modes, d_model):
    """Study eigenvalue spectrum of Cov(h_l) across layers."""
    print("\n" + "="*70)
    print("Exp1: Natural Covariance Spectrum — Σ_l特征值演化")
    print("  ★ 核心问题: 协方差结构如何随层演化? ★")
    print("  ★ 约束几何 vs 方向: 协方差谱是否比Jacobian谱更稳定? ★")
    print("="*70)

    k_cov = min(50, d_model)  # top-k eigenvalues

    results = {}
    for mode in modes:
        results[mode] = {}
        print(f"\n--- Mode: {mode} ---")
        print(f"  {'Layer':>6}  {'λ_max':>10}  {'λ_med':>10}  {'λ_10':>10}  "
              f"{'d_eff':>8}  {'n_>1%':>6}  {'90%_energy':>10}  {'Shape':>20}")

        for li in sample_layers:
            H = hidden_states[mode].get(li)
            if H is None or H.shape[0] < 10:
                continue

            # Center the data
            H_centered = H - H.mean(axis=0, keepdims=True)

            # Use SVD of data matrix for eigenvalue computation
            # H_centered = U S Vt, so Cov eigenvalues = S^2 / (n-1)
            n_samples = H_centered.shape[0]
            rank = min(n_samples, H_centered.shape[1])
            k_svd = min(k_cov, rank - 1)
            if k_svd < 1:
                continue

            U, s, Vt = np.linalg.svd(H_centered, full_matrices=False)
            eigenvalues = s**2 / max(n_samples - 1, 1)

            # Compute metrics
            total_energy = np.sum(eigenvalues)
            if total_energy < 1e-12:
                continue

            cum_energy = np.cumsum(eigenvalues) / total_energy
            n_90 = np.searchsorted(cum_energy, 0.9) + 1
            n_significant = np.sum(eigenvalues > 0.01 * eigenvalues[0])
            d_eff = (np.sum(eigenvalues))**2 / max(np.sum(eigenvalues**2), 1e-20)

            # Shape classification
            if len(eigenvalues) >= 10:
                ratio_10 = eigenvalues[9] / eigenvalues[0] if eigenvalues[0] > 0 else 0
            else:
                ratio_10 = 0

            if n_significant < 5:
                shape = "ULTRA-SPARSE"
            elif d_eff < 10:
                shape = "SPARSE"
            elif d_eff < 50:
                shape = "MODERATE"
            elif d_eff < 200:
                shape = "DENSE"
            else:
                shape = "ISOTROPIC"

            print(f"  {li:>6}  {eigenvalues[0]:>10.4f}  {eigenvalues[min(25,len(eigenvalues)-1)]:>10.6f}  "
                  f"{eigenvalues[min(9,len(eigenvalues)-1)]:>10.6f}  {d_eff:>8.1f}  "
                  f"{n_significant:>6}  {n_90:>10}  {shape:>20}")

            results[mode][li] = {
                'eigenvalues': eigenvalues[:k_cov].tolist(),
                'eigenvectors': Vt[:k_cov],  # (k, d)
                'd_eff': float(d_eff),
                'n_90': int(n_90),
                'total_energy': float(total_energy),
                'n_significant': int(n_significant),
                'n_samples': int(n_samples),
            }

    return results


# ========================================================================
# Exp2: Covariance Subspace Alignment — 约束几何保持度
# ========================================================================
def exp2_covariance_alignment(cov_results, sample_layers, modes, d_model):
    """
    KEY EXPERIMENT: Test if covariance subspaces are more aligned than
    individual directions. This directly tests "constraint geometry > direction".

    Compare with Phase 204's direction alignment (Jacobian eigenvector alignment).
    """
    print("\n" + "="*70)
    print("Exp2: Covariance Subspace Alignment — 约束几何保持度")
    print("  ★ 核心假设: 约束几何(协方差子空间)比方向更稳定 ★")
    print("  ★ 如果alignment > Phase 204的direction alignment → 约束>方向 ★")
    print("="*70)

    k_subspace = min(50, d_model)
    random_baseline = k_subspace / d_model  # Expected random alignment

    print(f"  d_model={d_model}, k_subspace={k_subspace}, random_baseline={random_baseline:.4f}")

    results = {}
    for mode in modes:
        results[mode] = {}
        print(f"\n--- Mode: {mode} ---")
        print(f"  {'Layer':>6}  →{'Next':>5}  {'Alignment':>10}  {'vs_random':>10}  {'Interpretation':>25}")

        sorted_layers = sorted(cov_results[mode].keys())
        for i in range(len(sorted_layers) - 1):
            li = sorted_layers[i]
            li_next = sorted_layers[i + 1]

            V1 = cov_results[mode][li].get('eigenvectors')
            V2 = cov_results[mode][li_next].get('eigenvectors')
            if V1 is None or V2 is None:
                continue

            k = min(k_subspace, V1.shape[0], V2.shape[0])
            if k < 1:
                continue

            # Subspace alignment: ||V1 V1^T V2^T V2|| / k
            # Measured by average singular value of V1[:k] @ V2[:k]^T
            M = V1[:k] @ V2[:k].T  # (k, k)
            sv = np.linalg.svd(M, compute_uv=False)
            alignment = float(sv.mean())
            vs_random = alignment / random_baseline

            if alignment > 0.8:
                interp = "STRONG CONSTRAINT PRESERVATION"
            elif alignment > 0.5:
                interp = "MODERATE CONSTRAINT PRESERV."
            elif alignment > 0.3:
                interp = "WEAK CONSTRAINT PRESERV."
            else:
                interp = "NO CONSTRAINT PRESERVATION"

            print(f"  {li:>6}  →{li_next:>5}  {alignment:>10.4f}  {vs_random:>10.1f}x  {interp:>25}")

            results[mode][f"{li}->{li_next}"] = {
                'alignment': alignment,
                'vs_random': vs_random,
                'interpretation': interp,
            }

        # Summary
        alignments = [v['alignment'] for v in results[mode].values()]
        if alignments:
            avg_align = np.mean(alignments)
            strong_segments = sum(1 for a in alignments if a > 0.8)
            print(f"\n  Covariance Alignment Summary for '{mode}':")
            print(f"    Avg alignment: {avg_align:.4f} ({avg_align/random_baseline:.1f}x random)")
            print(f"    Strong preservation segments (>0.8): {strong_segments}/{len(alignments)}")

            # Find longest run of moderate+ alignment
            moderate_flags = [1 if a > 0.5 else 0 for a in alignments]
            max_run = 0
            current_run = 0
            for f in moderate_flags:
                if f:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            print(f"    Longest moderate+ preservation run: {max_run} consecutive layer-pairs")

    return results


# ========================================================================
# Exp3: Covariance vs Direction Alignment — 核心对比!
# ========================================================================
def exp3_cov_vs_direction(cov_alignment, perturb_data, sample_layers, modes, d_model):
    """
    KEY COMPARISON: Is covariance subspace alignment higher than
    Jacobian eigenvector direction alignment?

    If YES → constraint geometry is more stable than direction
    If NO  → direction is the right level of analysis

    Phase 204 direction alignment was ~0.4-0.7 (no STRONG persistence)
    We expect covariance alignment to be HIGHER.
    """
    print("\n" + "="*70)
    print("Exp3: Covariance vs Direction Alignment — 约束>方向? (核心对比!)")
    print("  ★ 如果协方差对齐 > 方向对齐 → 概念以关系结构编码 ★")
    print("  ★ 如果协方差对齐 ≤ 方向对齐 → 方向是正确分析层次 ★")
    print("="*70)

    k_subspace = min(50, d_model)
    random_baseline = k_subspace / d_model

    results = {}
    for mode in modes:
        results[mode] = {}
        print(f"\n--- Mode: {mode} ---")
        print(f"  {'Layer':>6}  {'Cov_align':>10}  {'Dir_align':>10}  "
              f"{'Ratio':>8}  {'Winner':>15}  {'Interpretation':>30}")

        # Get integer layers from perturbation data
        perturb_layers = sorted([li for li in perturb_data.get(mode, {}).keys() if isinstance(li, int)])

        for li in perturb_layers:
            # Covariance alignment: find the key matching this layer
            cov_align = None
            for k in cov_alignment.get(mode, {}):
                # k is like "0->1" or "2->4"
                parts = k.split('->')
                if len(parts) == 2:
                    try:
                        if int(parts[0]) == li:
                            cov_align = cov_alignment[mode][k]['alignment']
                            break
                    except ValueError:
                        continue

            # Direction alignment from perturbation data
            dir_align = None
            if li in perturb_data.get(mode, {}) and (li + 1) in perturb_data.get(mode, {}):
                V_in = perturb_data[mode][li].get('V_in')

                # Find next layer's input vectors
                V_in_next = perturb_data[mode].get(li + 1, {}).get('V_in')
                if V_in is not None and V_in_next is not None:
                    k = min(k_subspace, V_in.shape[0], V_in_next.shape[0])
                    M = V_in[:k] @ V_in_next[:k].T
                    sv = np.linalg.svd(M, compute_uv=False)
                    dir_align = float(sv.mean())

            if cov_align is not None and dir_align is not None:
                ratio = cov_align / max(dir_align, 0.001)
                if ratio > 1.2:
                    winner = "COVARIANCE"
                    interp = "Constraint geometry preserved!"
                elif ratio < 0.8:
                    winner = "DIRECTION"
                    interp = "Direction is more stable"
                else:
                    winner = "SIMILAR"
                    interp = "Both at same level"

                print(f"  {li:>6}  {cov_align:>10.4f}  {dir_align:>10.4f}  "
                      f"{ratio:>8.2f}  {winner:>15}  {interp:>30}")

                results[mode][li] = {
                    'cov_alignment': cov_align,
                    'dir_alignment': dir_align,
                    'ratio': ratio,
                    'winner': winner,
                }

    # Cross-mode summary
    print("\n--- Exp3 Cross-Mode Summary ---")
    for mode in modes:
        ratios = [v['ratio'] for v in results.get(mode, {}).values()]
        if ratios:
            avg_ratio = np.mean(ratios)
            cov_wins = sum(1 for r in ratios if r > 1.2)
            dir_wins = sum(1 for r in ratios if r < 0.8)
            print(f"  {mode}: avg ratio={avg_ratio:.2f}, "
                  f"cov wins={cov_wins}, dir wins={dir_wins}")

    return results


# ========================================================================
# Exp4: Jacobian Composition Law — 多层级联稳定性
# ========================================================================
def exp4_jacobian_composition(perturb_data, sample_layers, modes, d_model):
    """
    Study the multi-step Jacobian product T = J_{l+n} ··· J_l

    Key question: Are there directions where the product remains stable?
    This tests the "stable transport directions" hypothesis.

    Using perturbation data, we approximate J_l ≈ ΔH_{l+1} @ pinv(ΔH_l)
    Then T_{0→l} ≈ ΔH_l @ pinv(ΔH_0)
    """
    print("\n" + "="*70)
    print("Exp4: Jacobian Composition Law — 多层级联稳定性")
    print("  ★ 核心问题: 某些方向在T=J_{L-1}...J_0下仍稳定? ★")
    print("  ★ 这是长程推理/概念保持/组合能力的数学基础 ★")
    print("="*70)

    results = {}

    for mode in modes:
        if mode not in perturb_data:
            continue
        print(f"\n--- Mode: {mode} ---")

        mode_data = perturb_data[mode]
        sorted_layers = sorted(mode_data.keys())

        if len(sorted_layers) < 2:
            print("  Not enough layers for composition analysis")
            continue

        # Compute composition spectrum using perturbation data
        # T_{0→l} maps input perturbation at L0 to output perturbation at Ll
        # Approximate: T_{0→l} ≈ ΔH_l @ pinv(ΔH_0)

        Dh_0 = mode_data.get(sorted_layers[0], {}).get('Dh_in')
        if Dh_0 is None:
            # Try to use Dh_out
            Dh_0 = mode_data.get(sorted_layers[0], {}).get('Dh_out')

        if Dh_0 is None:
            print("  Cannot compute composition: missing base layer data")
            continue

        print(f"  {'Layer':>6}  {'σ_max':>10}  {'σ_med':>10}  "
              f"{'σ_10':>10}  {'n_near_1':>8}  {'Stable%':>8}  {'Interpretation':>25}")

        # For each target layer, compute T_{0→l}
        Dh_0_pinv = np.linalg.pinv(Dh_0)  # (d, n) - pseudoinverse

        for li in sorted_layers[1:]:
            Dh_l = mode_data[li].get('Dh_out')
            if Dh_l is None:
                continue

            # T ≈ Dh_l @ Dh_0_pinv  (n_l × d) @ (d × n_0) = (n_l × n_0)
            # For eigenvalue analysis, compute T @ T^T
            T = Dh_l @ Dh_0_pinv  # (n_l × n_0)

            # SVD of T
            try:
                U, s, Vt = np.linalg.svd(T, full_matrices=False)
            except Exception:
                continue

            # The singular values of T are the "composition singular values"
            n_near_1 = np.sum((s > 0.7) & (s < 1.3))
            n_expand = np.sum(s > 1.5)
            n_collapse = np.sum(s < 0.5)
            n_stable = np.sum((s >= 0.5) & (s <= 1.5))
            total = len(s) if len(s) > 0 else 1
            stable_pct = 100.0 * n_stable / total

            if stable_pct > 80:
                interp = "STABLE COMPOSITION"
            elif stable_pct > 50:
                interp = "MODERATE COMPOSITION"
            elif n_expand > n_collapse:
                interp = "EXPANDING COMPOSITION"
            else:
                interp = "COLLAPSING COMPOSITION"

            sigma_med = s[min(len(s)//2, len(s)-1)] if len(s) > 0 else 0
            sigma_10 = s[min(9, len(s)-1)] if len(s) > 0 else 0

            print(f"  {li:>6}  {s[0] if len(s)>0 else 0:>10.4f}  {sigma_med:>10.4f}  "
                  f"{sigma_10:>10.4f}  {n_near_1:>8}  {stable_pct:>7.1f}%  {interp:>25}")

            results[f"{mode}_L{li}"] = {
                'sigma_max': float(s[0]) if len(s) > 0 else 0,
                'sigma_med': float(sigma_med),
                'n_near_1': int(n_near_1),
                'stable_pct': float(stable_pct),
                'expand_pct': float(100.0 * n_expand / total),
                'collapse_pct': float(100.0 * n_collapse / total),
            }

    return results


# ========================================================================
# Exp5: Cross-task Covariance Overlap — 跨任务约束结构共享
# ========================================================================
def exp5_cross_task_covariance(cov_results, sample_layers, modes, d_model):
    """
    Compare covariance structures across tasks.
    Do normal/CoT/translation share the same constraint geometry?
    """
    print("\n" + "="*70)
    print("Exp5: Cross-task Covariance Overlap — 跨任务约束结构共享")
    print("  ★ 核心问题: 不同任务共享约束几何吗? ★")
    print("  ★ 共享→概念复用在约束层面, 不共享→独立约束空间 ★")
    print("="*70)

    k_subspace = min(50, d_model)
    random_baseline = k_subspace / d_model

    results = {}
    print(f"\n  {'Layer':>6}  {'norm-cot':>10}  {'norm-tran':>10}  {'cot-tran':>10}  "
          f"{'Avg':>8}  {'vs_random':>10}")

    common_layers = sorted(set(
        li for mode in modes for li in cov_results.get(mode, {}).keys()
    ))

    for li in common_layers:
        overlaps = {}
        mode_pairs = [("normal", "cot"), ("normal", "translation"), ("cot", "translation")]

        for m1, m2 in mode_pairs:
            V1 = cov_results.get(m1, {}).get(li, {}).get('eigenvectors')
            V2 = cov_results.get(m2, {}).get(li, {}).get('eigenvectors')

            if V1 is not None and V2 is not None:
                k = min(k_subspace, V1.shape[0], V2.shape[0])
                M = V1[:k] @ V2[:k].T
                sv = np.linalg.svd(M, compute_uv=False)
                overlaps[f"{m1}-{m2}"] = float(sv.mean())
            else:
                overlaps[f"{m1}-{m2}"] = 0.0

        avg_overlap = np.mean(list(overlaps.values())) if overlaps else 0
        vs_random = avg_overlap / random_baseline if random_baseline > 0 else 0

        print(f"  {li:>6}  {overlaps.get('normal-cot', 0):>10.4f}  "
              f"{overlaps.get('normal-translation', 0):>10.4f}  "
              f"{overlaps.get('cot-translation', 0):>10.4f}  "
              f"{avg_overlap:>8.4f}  {vs_random:>10.1f}x")

        results[li] = {**overlaps, 'avg': avg_overlap, 'vs_random': vs_random}

    # Evolution analysis
    print("\n--- Exp5: Covariance Overlap Evolution (early → mid → late) ---")
    early_layers = [li for li in common_layers if li < len(common_layers) // 3]
    mid_layers = [li for li in common_layers if len(common_layers) // 3 <= li < 2 * len(common_layers) // 3]
    late_layers = [li for li in common_layers if li >= 2 * len(common_layers) // 3]

    for phase_name, phase_layers in [("Early", early_layers), ("Mid", mid_layers), ("Late", late_layers)]:
        phase_overlaps = [results[li]['avg'] for li in phase_layers if li in results]
        if phase_overlaps:
            print(f"  {phase_name}: avg overlap = {np.mean(phase_overlaps):.4f} "
                  f"({np.mean(phase_overlaps)/random_baseline:.1f}x random)")

    # Compare with Phase 204 direction overlap
    print("\n--- Exp5: Covariance Overlap vs Phase 204 Direction Overlap ---")
    print("  (Phase 204 direction overlap was: norm-tran >> norm-cot ≈ cot-tran)")
    for pair_name in ["normal-cot", "normal-translation", "cot-translation"]:
        values = [results[li].get(pair_name, 0) for li in common_layers if li in results]
        if values:
            print(f"  {pair_name}: avg={np.mean(values):.4f} "
                  f"({np.mean(values)/random_baseline:.1f}x random)")

    return results


# ========================================================================
# Main
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    t_total = time.time()

    print(f"\n{'='*70}")
    print(f"PHASE 205: COVARIANCE TRANSPORT GEOMETRY — {model_name}")
    print(f"{'='*70}")
    print(f"  Time: {datetime.now()}")
    print(f"  Lite mode: {LITE}")
    print(f"  Theory: Constraint geometry > direction (concepts are relations, not vectors)")
    print(f"  Core hypothesis:")
    print(f"    Covariance subspace alignment > Direction alignment")
    print(f"    → Concepts are constraint geometries, not feature directions")
    print(f"  Experiments:")
    print(f"    Exp1: Natural Covariance Spectrum")
    print(f"    Exp2: Covariance Subspace Alignment (constraint preservation)")
    print(f"    Exp3: Covariance vs Direction Alignment (core comparison!)")
    print(f"    Exp4: Jacobian Composition Law (multi-step stability)")
    print(f"    Exp5: Cross-task Covariance Overlap (shared constraints)")

    # Load model
    model, tokenizer, device = load_model_flash(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"[load] {model_name}: {info.model_class}, {n_layers}L, d={d_model}")

    n_cov_sentences = N_COV
    n_perturb_sentences = N_PERTURB
    print(f"  Covariance sentences: {n_cov_sentences}, Modes: {CORE_MODES}")
    print(f"  Perturbation sentences: {n_perturb_sentences}")

    # ====================================================================
    # Step 1: Collect natural hidden states for covariance
    # ====================================================================
    print(f"\n{'='*70}")
    print("Step 1: Collecting natural hidden states for covariance estimation...")
    print(f"{'='*70}")

    cov_sentences = BASE_SENTENCES[:n_cov_sentences]
    hidden_states, sample_layers = collect_hidden_states(
        model, tokenizer, device, info, cov_sentences, CORE_MODES)

    # Print stats
    for mode in CORE_MODES:
        for li in sample_layers[:3]:
            H = hidden_states[mode].get(li)
            if H is not None:
                print(f"  {mode} L{li}: {H.shape[0]} tokens × {H.shape[1]} dims")

    # ====================================================================
    # Step 2: Collect perturbation data for Jacobian
    # ====================================================================
    print(f"\n{'='*70}")
    print("Step 2: Collecting perturbation data for Jacobian estimation...")
    print(f"{'='*70}")

    perturb_sentences = BASE_SENTENCES[:n_perturb_sentences]
    n_probe = 40 if LITE else 60
    perturb_data, perturb_layers = collect_perturbation_data(
        model, tokenizer, device, info, perturb_sentences, CORE_MODES, n_probe=n_probe)

    # ====================================================================
    # Exp1: Natural Covariance Spectrum
    # ====================================================================
    cov_results = exp1_covariance_spectrum(hidden_states, sample_layers, CORE_MODES, d_model)

    # ====================================================================
    # Exp2: Covariance Subspace Alignment
    # ====================================================================
    cov_alignment = exp2_covariance_alignment(cov_results, sample_layers, CORE_MODES, d_model)

    # ====================================================================
    # Exp3: Covariance vs Direction Alignment (CORE!)
    # ====================================================================
    cov_vs_dir = exp3_cov_vs_direction(
        cov_alignment, perturb_data, sample_layers, CORE_MODES, d_model)

    # ====================================================================
    # Exp4: Jacobian Composition Law
    # ====================================================================
    composition_results = exp4_jacobian_composition(
        perturb_data, sample_layers, CORE_MODES, d_model)

    # ====================================================================
    # Exp5: Cross-task Covariance Overlap
    # ====================================================================
    cross_task_results = exp5_cross_task_covariance(
        cov_results, sample_layers, CORE_MODES, d_model)

    # ====================================================================
    # Summary
    # ====================================================================
    total_time = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"PHASE 205 SUMMARY — {model_name}")
    print(f"{'='*70}")

    # Key findings
    print("\n1. Covariance Spectrum (Exp1):")
    for mode in CORE_MODES:
        mid_layer = sample_layers[len(sample_layers)//2]
        if mid_layer in cov_results.get(mode, {}):
            r = cov_results[mode][mid_layer]
            print(f"   {mode} L{mid_layer}: d_eff={r['d_eff']:.1f}, "
                  f"n_90={r['n_90']}, n_significant={r['n_significant']}")

    print("\n2. Covariance Alignment (Exp2):")
    for mode in CORE_MODES:
        alignments = [v['alignment'] for v in cov_alignment.get(mode, {}).values()]
        if alignments:
            print(f"   {mode}: avg={np.mean(alignments):.4f}, "
                  f"max={np.max(alignments):.4f}")

    print("\n3. Cov vs Direction (Exp3):")
    for mode in CORE_MODES:
        ratios = [v['ratio'] for v in cov_vs_dir.get(mode, {}).values()]
        if ratios:
            cov_wins = sum(1 for r in ratios if r > 1.2)
            print(f"   {mode}: avg ratio={np.mean(ratios):.2f}, "
                  f"cov wins={cov_wins}/{len(ratios)}")

    print("\n4. Composition (Exp4):")
    for key in list(composition_results.keys())[:6]:
        r = composition_results[key]
        print(f"   {key}: σ_max={r['sigma_max']:.2f}, stable={r['stable_pct']:.1f}%")

    print("\n5. Cross-task Covariance (Exp5):")
    mid_key = sample_layers[len(sample_layers)//2]
    if mid_key in cross_task_results:
        r = cross_task_results[mid_key]
        print(f"   L{mid_key}: norm-cot={r.get('normal-cot',0):.4f}, "
              f"norm-tran={r.get('normal-translation',0):.4f}, "
              f"cot-tran={r.get('cot-translation',0):.4f}")

    # Save results
    save_dir = Path(__file__).parent.parent / "glm5_temp"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"phase205_{model_name}_results.json"

    def make_serializable(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()
                    if k != 'eigenvectors' and k != 'Dh_in' and k != 'Dh_out'}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, int):
            return obj
        elif isinstance(obj, float):
            return obj
        elif isinstance(obj, str):
            return obj
        else:
            return str(obj)

    json_results = make_serializable({
        "exp1_cov_spectrum": cov_results,
        "exp2_cov_alignment": cov_alignment,
        "exp3_cov_vs_dir": cov_vs_dir,
        "exp4_composition": composition_results,
        "exp5_cross_task": cross_task_results,
    })

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {save_path}")

    # Release
    release_model(model)
    print(f"\n[Phase 205] COMPLETE in {total_time:.1f}s ({model_name})")


if __name__ == "__main__":
    main()
