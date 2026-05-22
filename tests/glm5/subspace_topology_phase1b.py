"""
Subspace Topology Phase 1b: 改进版 — 使用全token位置 + 大样本量
================================================================

改进点:
  1. 使用所有token位置（不仅最后一个），大幅增加样本量
  2. 用协方差矩阵特征值计算参与率（与Gemini方法论一致）
  3. 增加prompt数量
  4. 同时计算两种ID估计: 参与率(PR)和Two-NN

Run:
  python tests/glm5/subspace_topology_phase1b.py --model qwen3
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import json
import time
from pathlib import Path

from model_utils import load_model, get_layers, get_model_info, release_model

OUTPUT_DIR = Path("results/subspace_topology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 大样本量 — 60个多样化的prompts
DIVERSE_PROMPTS = [
    # 简单陈述 (10)
    "The apple is red.",
    "Paris is the capital of France.",
    "Water boils at 100 degrees.",
    "The sky is blue on a clear day.",
    "Dogs are loyal animals.",
    "The earth revolves around the sun.",
    "Ice melts when heated.",
    "Books are made of paper.",
    "The ocean is very deep.",
    "Birds can fly in the sky.",
    # 否定 (5)
    "The apple is NOT red.",
    "Paris is NOT the capital of France.",
    "The sky is NOT blue.",
    "Dogs are NOT loyal animals.",
    "Ice does NOT melt when heated.",
    # 疑问 (5)
    "Is the apple red?",
    "What is the capital of France?",
    "How hot does water boil?",
    "Why is the sky blue?",
    "Can dogs be loyal?",
    # 双重否定 (3)
    "It is not true that the apple is not red.",
    "It is not false that Paris is the capital.",
    "It is not the case that ice does not melt.",
    # 三段论/推理 (5)
    "All cats are animals. Whiskers is a cat. Therefore Whiskers is",
    "If A equals B and B equals C, then A",
    "Step by step: If John is taller than Mary, and Mary is taller than Tom,",
    "All birds can fly. Penguins are birds. Therefore penguins",
    "If it rains, the ground gets wet. It is raining. Therefore",
    # 翻译 (5)
    "Translate to French: The apple is red.",
    "Translate to Chinese: The apple is red.",
    "Translate to German: The sky is blue.",
    "Translate to Spanish: Dogs are loyal.",
    "Translate to Japanese: Water boils at 100 degrees.",
    # 代码 (5)
    "Python code to sort a list:",
    "Write a function to reverse a string:",
    "How to read a file in Python:",
    "Implement binary search in Python:",
    "Create a class for a linked list:",
    # 数学 (5)
    "What is 41 + 2?",
    "What is 49 + 2?",
    "Calculate 15 times 7.",
    "What is the square root of 144?",
    "Solve for x: 2x + 5 = 15",
    # 情感/角色 (4)
    "In the style of Shakespeare, explain quantum physics.",
    "As a pirate, describe the treasure.",
    "Like a scientist, analyze the data.",
    "With great sadness, I must tell you",
    # 抽象概念 (4)
    "Justice is a fundamental concept in philosophy.",
    "Freedom means different things to different people.",
    "Love cannot be measured in numbers.",
    "Time passes differently when you are happy.",
    # 属性查询 (4)
    "The physical color of an apple is",
    "The physical color of justice is",
    "The shape of a ball is",
    "The shape of freedom is",
    # 代词/指代 (5)
    "John gave Mary a book. She read",
    "John gave Mary a book. He read",
    "The cat chased the mouse. It caught",
    "Alice told Bob a secret. He kept",
    "Alice told Bob a secret. She kept",
]


def compute_participation_ratio_eig(eigenvalues):
    """用协方差矩阵特征值计算参与率: PR = (sum λ)^2 / sum(λ^2)"""
    lam = np.array(eigenvalues, dtype=np.float64)
    lam = lam[lam > 1e-12]
    if len(lam) == 0:
        return 0.0
    return float((np.sum(lam))**2 / np.sum(lam**2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    print(f"\n模型: {model_info.name}, {n_layers}层, d_model={d_model}")
    
    # ========================================
    # Step1: 收集所有token位置的残差流激活
    # ========================================
    print(f"\n收集 {len(DIVERSE_PROMPTS)} 个prompt的全token残差流...")
    
    # 每层收集: [n_total_tokens, d_model]
    # 用内存高效的方式: 先收集，再分批SVD
    layer_all_acts = {li: [] for li in range(n_layers)}
    total_tokens = 0
    
    for pi, prompt in enumerate(DIVERSE_PROMPTS):
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        captured = {}
        hooks = []
        for li in range(n_layers):
            layer = layers[li]
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float()
                    else:
                        captured[key] = output.detach().float()
                return hook
            hooks.append(layer.register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            except Exception as e:
                print(f"  Prompt {pi} failed: {e}")
        
        for h in hooks:
            h.remove()
        
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured:
                # 收集所有token位置 (不只是最后一个)
                acts = captured[key][0, :, :].cpu().numpy()  # [seq_len, d_model]
                layer_all_acts[li].append(acts)
                total_tokens += seq_len
        
        del captured
        gc.collect()
        
        if (pi + 1) % 20 == 0:
            print(f"  已处理 {pi+1}/{len(DIVERSE_PROMPTS)} prompts, total_tokens≈{total_tokens}")
    
    print(f"  总共收集了约 {total_tokens} 个token的激活")
    
    # ========================================
    # Step2: 对每层做SVD + 计算ID
    # ========================================
    print(f"\n对每层做SVD分析（使用所有token位置）...")
    
    results = {}
    id_profile = []  # [(layer, ID)]
    
    for li in range(n_layers):
        # 合并所有prompt的所有token
        all_acts = np.concatenate(layer_all_acts[li], axis=0)  # [n_tokens, d_model]
        n_tokens = all_acts.shape[0]
        
        # 中心化
        mean_act = all_acts.mean(axis=0, keepdims=True)
        acts_centered = all_acts - mean_act
        
        # 由于n_tokens >> d_model, 做economy SVD: acts_centered = U @ diag(S) @ Vt
        # 协方差矩阵 = acts_centered^T @ acts_centered / (n-1) = V @ diag(S^2/(n-1)) @ V^T
        # 参与率 = (sum S^2)^2 / sum S^4 = (sum λ)^2 / sum(λ^2) 其中λ=S^2/(n-1)
        
        # 为了节省内存，用协方差矩阵的特征值
        # 如果 n_tokens < d_model: SVD of acts_centered [n, d] → U[n,n] S[n] Vt[n,d]
        # 如果 n_tokens >= d_model: SVD of acts_centered [n, d] → U[n,d] S[d] Vt[d,d]
        
        # 更高效: 计算 Gram 矩阵 acts_centered @ acts_centered^T 的特征值
        # 或直接 SVD
        
        print(f"  L{li}: {n_tokens} tokens × {d_model} dim, computing SVD...", end="")
        t0 = time.time()
        
        try:
            # Economy SVD
            U, S, Vt = np.linalg.svd(acts_centered, full_matrices=False)
        except Exception as e:
            print(f" FAILED: {e}")
            continue
        
        dt = time.time() - t0
        print(f" done in {dt:.1f}s, rank={len(S)}")
        
        # 协方差矩阵特征值 = S^2 / (n-1)
        n_eff = n_tokens - 1
        eigenvalues = (S ** 2) / n_eff
        
        # 参与率 (基于特征值，与Gemini方法论一致)
        pr_eig = compute_participation_ratio_eig(eigenvalues)
        
        # 参与率 (基于奇异值)
        pr_sv = compute_participation_ratio_eig(S)
        
        # 有效秩
        max_eig = np.max(eigenvalues) if len(eigenvalues) > 0 else 0
        eRank = int(np.sum(eigenvalues > 0.01 * max_eig)) if max_eig > 0 else 0
        eRank_001 = int(np.sum(eigenvalues > 0.001 * max_eig)) if max_eig > 0 else 0
        
        # 方差解释比
        total_var = np.sum(eigenvalues)
        cumvar = np.cumsum(eigenvalues) / total_var if total_var > 0 else np.zeros_like(eigenvalues)
        n_90 = int(np.searchsorted(cumvar, 0.90)) + 1 if len(cumvar) > 0 else 0
        n_95 = int(np.searchsorted(cumvar, 0.95)) + 1 if len(cumvar) > 0 else 0
        n_99 = int(np.searchsorted(cumvar, 0.99)) + 1 if len(cumvar) > 0 else 0
        
        # 奇异值衰减特性
        if len(S) > 1 and S[0] > 1e-10:
            sv_ratio_12 = float(S[1] / S[0])
            sv_ratio_110 = float(S[min(9, len(S)-1)] / S[0])
            # 频谱指数衰减拟合: S_k ~ C * exp(-alpha * k)
            # log(S_k) = log(C) - alpha * k
            valid = S > 1e-10
            if np.sum(valid) > 5:
                log_S = np.log(S[valid])
                ks = np.arange(len(log_S))
                # 线性拟合
                alpha_fit = float(-np.polyfit(ks[:20], log_S[:20], 1)[0]) if len(log_S) >= 20 else 0
            else:
                alpha_fit = 0
        else:
            sv_ratio_12 = 0
            sv_ratio_110 = 0
            alpha_fit = 0
        
        results[f"L{li}"] = {
            "layer": li,
            "n_tokens": n_tokens,
            "d_model": d_model,
            "rank": int(len(S)),
            "intrinsic_dim_PR": pr_eig,       # 基于特征值的参与率
            "intrinsic_dim_SV": pr_sv,         # 基于奇异值的参与率
            "effective_rank_1pct": eRank,
            "effective_rank_01pct": eRank_001,
            "dim_for_90pct_var": n_90,
            "dim_for_95pct_var": n_95,
            "dim_for_99pct_var": n_99,
            "sv_ratio_12": sv_ratio_12,
            "sv_ratio_110": sv_ratio_110,
            "spectral_decay_alpha": alpha_fit,
            "top_30_singular_values": [float(x) for x in S[:30]],
            "top_30_eigenvalues": [float(x) for x in eigenvalues[:30]],
            "total_variance": float(total_var),
        }
        
        id_profile.append((li, pr_eig))
        
        print(f"    ID(PR_eig)={pr_eig:.2f}, ID(PR_SV)={pr_sv:.2f}, "
              f"eRank(1%)={eRank}, eRank(0.1%)={eRank_001}, "
              f"90%var={n_90}d, 95%var={n_95}d, 99%var={n_99}d, "
              f"α={alpha_fit:.4f}")
        
        # 释放内存
        del all_acts, acts_centered, U, S, Vt, eigenvalues
        gc.collect()
    
    # ========================================
    # Step3: 计算Ω压缩比
    # ========================================
    print(f"\n{'='*70}")
    print("Ω压缩比分析")
    print(f"{'='*70}")
    
    # 用不同层范围计算
    shallow_layers = [li for li, _ in id_profile[:max(1, n_layers//3)]]
    deep_layers = [li for li, _ in id_profile[n_layers//3:]]
    
    shallow_ids = [id_ for li, id_ in id_profile if li in shallow_layers]
    deep_ids = [id_ for li, id_ in id_profile if li in deep_layers]
    
    if shallow_ids and deep_ids:
        max_id_shallow = max(shallow_ids)
        min_id_deep = min(deep_ids)
        omega = max_id_shallow / max(min_id_deep, 0.01)
        
        # 也在深层取最后5层的最小
        last5_ids = [id_ for li, id_ in id_profile if li >= n_layers - 5]
        min_id_last5 = min(last5_ids) if last5_ids else min_id_deep
        omega_last5 = max_id_shallow / max(min_id_last5, 0.01)
        
        print(f"  max(ID_shallow[0:{n_layers//3}]) = {max_id_shallow:.2f}")
        print(f"  min(ID_deep[{n_layers//3}:{n_layers}]) = {min_id_deep:.2f}")
        print(f"  min(ID_last5) = {min_id_last5:.2f}")
        print(f"  Ω(deep) = {omega:.2f}")
        print(f"  Ω(last5) = {omega_last5:.2f}")
        
        results["omega"] = {
            "max_id_shallow": max_id_shallow,
            "min_id_deep": min_id_deep,
            "min_id_last5": min_id_last5,
            "omega_deep": omega,
            "omega_last5": omega_last5,
        }
    
    # 打印完整ID profile
    print(f"\n{'='*70}")
    print("完整ID profile:")
    print(f"{'='*70}")
    for li, id_ in id_profile:
        bar = "█" * int(id_ / 2)
        print(f"  L{li:2d}: ID={id_:6.2f} {bar}")
    
    # 保存
    out_path = OUTPUT_DIR / f"exp1b_full_svd_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    release_model(model)
    print(f"\nDone!")


if __name__ == "__main__":
    main()
