"""
Subspace Topology Phase 1c: SVCCA层间对齐 + Rank-1方向追踪
==========================================================

核心问题:
  1. 中间层的Rank-1方向在不同层之间是否相同？
  2. 如果不同，在哪里发生了"方向跳转"？
  3. Rank-1方向携带了什么语义信息？

Run:
  python tests/glm5/subspace_topology_phase1c.py --model qwen3
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
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    W_U = get_W_U(model, model_info.name)
    
    print(f"\n模型: {model_info.name}, {n_layers}层, d_model={d_model}")
    
    # ========================================
    # Step1: 收集各层残差流
    # ========================================
    print(f"\n收集 {len(DIVERSE_PROMPTS)} 个prompt的残差流...")
    
    # 每层: 收集所有token的激活 [n_total_tokens, d_model]
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
    # Step2: 提取每层的主方向 (前3个奇异向量)
    # ========================================
    print(f"\n提取每层的主方向...")
    
    layer_directions = {}  # {li: (V, S)} — V: [d_model, k] 右奇异向量
    layer_means = {}
    
    for li in range(n_layers):
        all_acts = np.concatenate(layer_acts[li], axis=0)
        mean_act = all_acts.mean(axis=0)
        layer_means[li] = mean_act
        acts_centered = all_acts - mean_act
        
        U, S, Vt = np.linalg.svd(acts_centered, full_matrices=False)
        # 取前3个右奇异向量 (d_model维方向)
        k = min(3, Vt.shape[0])
        layer_directions[li] = (Vt[:k, :], S[:k])  # directions: [k, d_model], singular_values: [k]
        
        print(f"  L{li:2d}: S1={S[0]:.1f}, S2={S[1]:.1f}, S3={S[2]:.1f}, "
              f"S1/S2={S[0]/max(S[1],1e-10):.1f}")
    
    # ========================================
    # Step3: 相邻层主方向对齐度
    # ========================================
    print(f"\n{'='*70}")
    print("相邻层Rank-1方向对齐度")
    print(f"{'='*70}")
    
    alignment_results = {}
    
    for li in range(n_layers - 1):
        v1 = layer_directions[li][0][0]   # L_i的第1主方向
        v2 = layer_directions[li+1][0][0] # L_{i+1}的第1主方向
        
        # 余弦相似度 (方向对齐度)
        cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
        
        # 角度
        angle_deg = float(np.arccos(np.clip(abs(cos_sim), 0, 1)) * 180 / np.pi)
        
        alignment_results[f"L{li}_L{li+1}"] = {
            "cos_sim": cos_sim,
            "abs_cos_sim": float(abs(cos_sim)),
            "angle_deg": angle_deg,
            "alignment": "SAME" if abs(cos_sim) > 0.95 else "ROTATED" if abs(cos_sim) > 0.5 else "JUMP",
        }
        
        if abs(cos_sim) < 0.95 or li % 5 == 0:
            print(f"  L{li:2d}→L{li+1:2d}: cos={cos_sim:+.4f}, |cos|={abs(cos_sim):.4f}, "
                  f"angle={angle_deg:.1f}° [{alignment_results[f'L{li}_L{li+1}']['alignment']}]")
    
    # 找到"方向跳转"的关键层
    jumps = [(k, v) for k, v in alignment_results.items() if v["alignment"] == "JUMP"]
    rotations = [(k, v) for k, v in alignment_results.items() if v["alignment"] == "ROTATED"]
    
    print(f"\n=== 方向跳转层 (|cos| < 0.5) ===")
    for k, v in jumps:
        print(f"  {k}: cos={v['cos_sim']:+.4f}, angle={v['angle_deg']:.1f}°")
    
    print(f"\n=== 方向旋转层 (0.5 < |cos| < 0.95) ===")
    for k, v in rotations:
        print(f"  {k}: cos={v['cos_sim']:+.4f}, angle={v['angle_deg']:.1f}°")
    
    # ========================================
    # Step4: Rank-1方向解码 — 它指向什么词？
    # ========================================
    print(f"\n{'='*70}")
    print("Rank-1方向解码 (Logit Lens)")
    print(f"{'='*70}")
    
    # W_U: [vocab_size, d_model]
    # 主方向v1: [d_model]
    # logits = W_U @ v1 → [vocab_size]
    
    rank1_decode = {}
    for li in range(n_layers):
        v1 = layer_directions[li][0][0]  # [d_model]
        mean_h = layer_means[li]
        
        # Logit Lens: 用W_U解码主方向
        logits_v1 = W_U @ v1  # [vocab_size]
        top10_ids = np.argsort(logits_v1)[-10:][::-1]
        top10_tokens = [tokenizer.decode([int(tid)]) for tid in top10_ids]
        top10_scores = [float(logits_v1[tid]) for tid in top10_ids]
        
        # 也解码均值方向
        logits_mean = W_U @ mean_h
        top10_mean_ids = np.argsort(logits_mean)[-10:][::-1]
        top10_mean_tokens = [tokenizer.decode([int(tid)]) for tid in top10_mean_ids]
        
        rank1_decode[f"L{li}"] = {
            "top10_v1": list(zip(top10_tokens, top10_scores)),
            "top10_mean": top10_mean_tokens,
            "s1": float(layer_directions[li][1][0]),
            "s2": float(layer_directions[li][1][1]) if len(layer_directions[li][1]) > 1 else 0,
        }
        
        if li % 5 == 0 or li == n_layers - 1:
            print(f"  L{li:2d} [S1={layer_directions[li][1][0]:.1f}]: "
                  f"v1→{top10_tokens[:5]}, mean→{top10_mean_tokens[:5]}")
    
    # ========================================
    # Step5: 第2主方向的变化 (Rank-1之外的信号)
    # ========================================
    print(f"\n{'='*70}")
    print("第2主方向追踪 (Rank-1之外的信号)")
    print(f"{'='*70}")
    
    for li in range(n_layers):
        v2 = layer_directions[li][0][1] if layer_directions[li][0].shape[0] > 1 else None
        if v2 is None:
            continue
        logits_v2 = W_U @ v2
        top5_ids = np.argsort(logits_v2)[-5:][::-1]
        top5_tokens = [tokenizer.decode([int(tid)]) for tid in top5_ids]
        
        if li % 5 == 0 or li == n_layers - 1:
            s2 = layer_directions[li][1][1] if len(layer_directions[li][1]) > 1 else 0
            s1 = layer_directions[li][1][0]
            print(f"  L{li:2d} [S2={s2:.1f}, S1/S2={s1/max(s2,1e-10):.0f}]: "
                  f"v2→{top5_tokens}")
    
    # 保存结果
    results = {
        "model": model_info.name,
        "n_layers": n_layers,
        "d_model": d_model,
        "alignment": alignment_results,
        "rank1_decode": {k: {
            "top10_v1_tokens": [t for t, s in v["top10_v1"][:10]],
            "top10_v1_scores": [s for t, s in v["top10_v1"][:10]],
            "top10_mean_tokens": v["top10_mean"][:10],
            "s1": v["s1"],
            "s2": v["s2"],
        } for k, v in rank1_decode.items()},
    }
    
    out_path = OUTPUT_DIR / f"exp1c_svcca_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    release_model(model)
    print("Done!")


if __name__ == "__main__":
    main()
