"""
Subspace Topology Phase 1d: 去偏置后真实语义维度
=================================================

核心发现: 中间层Rank-1方向解码为空格/标点，不携带语义
验证: 去除Rank-1分量后，真实的语义ID是多少？

Run:
  python tests/glm5/subspace_topology_phase1d.py --model qwen3
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

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U

OUTPUT_DIR = Path("results/subspace_topology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIVERSE_PROMPTS = [
    "The apple is red.",
    "Paris is the capital of France.",
    "Water boils at 100 degrees.",
    "The apple is NOT red.",
    "Is the apple red?",
    "Translate to French: The apple is red.",
    "Step by step: If John is taller than Mary, who is shorter?",
    "Python code to sort a list:",
    "What is 49 + 2?",
    "In the style of Shakespeare, explain physics.",
    "Justice is a fundamental concept.",
    "John gave Mary a book. She read",
    "饕餮是一种传说中的神兽。",
    "The sky is blue.",
    "The sky is NOT blue.",
    "Is the sky blue?",
    "Translate to Chinese: Dogs are loyal.",
    "All cats are animals. Whiskers is a cat. Therefore",
    "Write a function to reverse a string:",
    "What is 41 + 2?",
    "The physical color of an apple is",
    "The physical color of justice is",
    "It is not true that the apple is not red.",
    "If A=B and B=C, then A=",
    "In the style of a pirate, describe the treasure.",
    "Freedom means different things to different people.",
    "1, 2, 3, 4,",
    "5, 6, 7,",
    "What is 15 times 7?",
    "Solve for x: 2x + 5 = 15",
    "Create a class for a linked list:",
    "The shape of a ball is",
    "Alice told Bob a secret. He kept",
    "As a scientist, analyze the data.",
    "Love cannot be measured in numbers.",
    "Time passes differently when you are happy.",
    "The earth revolves around the sun.",
    "Ice melts when heated.",
    "Books are made of paper.",
    "Birds can fly in the sky.",
]


def compute_participation_ratio(eigenvalues):
    lam = np.array(eigenvalues, dtype=np.float64)
    lam = lam[lam > 1e-12]
    if len(lam) == 0:
        return 0.0
    return float((np.sum(lam))**2 / np.sum(lam**2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    W_U = get_W_U(model, model_info.name)
    
    print(f"\n模型: {model_info.name}, {n_layers}层, d_model={d_model}")
    
    # 收集激活
    print(f"\n收集 {len(DIVERSE_PROMPTS)} 个prompt的残差流...")
    layer_acts = {li: [] for li in range(n_layers)}
    
    for pi, prompt in enumerate(DIVERSE_PROMPTS):
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = toks.input_ids.shape[1]
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(toks.input_ids).detach().clone().to(model.dtype)
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
            except Exception:
                pass
        
        for h in hooks:
            h.remove()
        
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured:
                acts = captured[key][0, :, :].cpu().numpy()
                layer_acts[li].append(acts)
        
        del captured
        gc.collect()
    
    # ========================================
    # 核心分析: 逐步去除主成分后的ID变化
    # ========================================
    print(f"\n{'='*70}")
    print("逐步去除主成分后的ID变化 — 揭示真实语义维度")
    print(f"{'='*70}")
    
    results = {}
    n_remove_list = [0, 1, 2, 3, 5, 10, 20, 50]  # 去除前N个主成分
    
    for li in range(n_layers):
        all_acts = np.concatenate(layer_acts[li], axis=0)  # [n_tokens, d_model]
        n_tokens = all_acts.shape[0]
        mean_act = all_acts.mean(axis=0)
        acts_centered = all_acts - mean_act
        
        # 清理NaN/Inf（8bit量化可能引入）
        acts_clean = np.nan_to_num(acts_centered, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            U, S, Vt = np.linalg.svd(acts_clean, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback: 用sklearn的randomized SVD
            from sklearn.decomposition import TruncatedSVD
            k = min(100, acts_clean.shape[1] - 1, acts_clean.shape[0] - 1)
            svd_obj = TruncatedSVD(n_components=k, random_state=42)
            svd_obj.fit(acts_clean.astype(np.float32))
            S = svd_obj.singular_values_
            Vt = svd_obj.components_
            U = None  # 不需要U矩阵
        
        layer_result = {
            "layer": li,
            "n_tokens": n_tokens,
            "total_ID": compute_participation_ratio(S**2 / (n_tokens - 1)),
        }
        
        # 记录前10个奇异值
        layer_result["top10_S"] = [float(x) for x in S[:10]]
        
        # 逐步去除主成分
        for n_remove in n_remove_list:
            if n_remove >= len(S):
                continue
            
            # 去除前n_remove个分量后的残差
            if n_remove == 0:
                residual = acts_clean
            else:
                # 去除前n_remove个主成分的贡献
                if U is not None:
                    low_rank = U[:, :n_remove] @ np.diag(S[:n_remove]) @ Vt[:n_remove, :]
                    residual = acts_clean - low_rank
                else:
                    # TruncatedSVD fallback: 只能用Vt投影
                    # residual = acts - acts @ Vt[:n_remove].T @ Vt[:n_remove]
                    proj = acts_clean @ Vt[:n_remove, :].T  # [n, n_remove]
                    low_rank = proj @ Vt[:n_remove, :]  # [n, d]
                    residual = acts_clean - low_rank
            
            # 重新计算ID (使用TruncatedSVD更稳定)
            try:
                U2, S2, Vt2 = np.linalg.svd(residual, full_matrices=False)
            except np.linalg.LinAlgError:
                from sklearn.decomposition import TruncatedSVD as TSVDE
                k2 = min(100, residual.shape[1] - 1, residual.shape[0] - 1)
                svd2 = TSVDE(n_components=k2, random_state=42)
                svd2.fit(residual.astype(np.float32))
                S2 = svd2.singular_values_
            eigenvalues2 = S2**2 / (n_tokens - 1)
            id_after = compute_participation_ratio(eigenvalues2)
            
            # 方差保留比
            total_var = np.sum(S**2)
            remaining_var = np.sum(S2**2)
            var_ratio = remaining_var / total_var if total_var > 0 else 0
            
            layer_result[f"ID_remove_{n_remove}"] = id_after
            layer_result[f"var_ratio_remove_{n_remove}"] = float(var_ratio)
            
            # 解码第(n_remove+1)个主方向
            if n_remove < len(S):
                v_next = Vt[n_remove]  # 去除n_remove后的第1主方向
                logits_v = W_U @ v_next
                top5_ids = np.argsort(logits_v)[-5:][::-1]
                top5_tokens = [tokenizer.decode([int(tid)]) for tid in top5_ids]
                layer_result[f"v{ n_remove}_top5"] = top5_tokens
        
        results[f"L{li}"] = layer_result
        
        # 打印关键层的结果
        if li in [0, 1, 5, 6, 10, 15, 20, 25, 30, 34, 35]:
            print(f"\n  L{li}: total_ID={layer_result['total_ID']:.2f}")
            for n_remove in n_remove_list:
                key_id = f"ID_remove_{n_remove}"
                key_var = f"var_ratio_remove_{n_remove}"
                if key_id in layer_result:
                    print(f"    去除{ n_remove:2d}PC: ID={layer_result[key_id]:8.2f}, "
                          f"var保留={layer_result[key_var]:.4f}", end="")
                    key_tok = f"v{n_remove}_top5"
                    if key_tok in layer_result:
                        print(f", 下一PC→{layer_result[key_tok][:3]}", end="")
                    print()
    
    # ========================================
    # 关键对比: 去除1个主成分后的ID profile
    # ========================================
    print(f"\n{'='*70}")
    print("去除Rank-1偏置后的真实语义ID Profile")
    print(f"{'='*70}")
    
    for li in range(n_layers):
        total_id = results[f"L{li}"]["total_ID"]
        id_r1 = results[f"L{li}"].get("ID_remove_1", total_id)
        var_r1 = results[f"L{li}"].get("var_ratio_remove_1", 1.0)
        id_r3 = results[f"L{li}"].get("ID_remove_3", id_r1)
        
        bar_total = "█" * min(int(total_id), 60)
        bar_r1 = "▓" * min(int(id_r1), 60)
        
        print(f"  L{li:2d}: total_ID={total_id:7.2f} {bar_total}")
        print(f"       ID(-1PC)={id_r1:7.2f} var={var_r1:.4f} {bar_r1}")
        print(f"       ID(-3PC)={id_r3:7.2f}")
    
    # 保存
    out_path = OUTPUT_DIR / f"exp1d_debiased_id_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    release_model(model)
    print("Done!")


if __name__ == "__main__":
    main()
