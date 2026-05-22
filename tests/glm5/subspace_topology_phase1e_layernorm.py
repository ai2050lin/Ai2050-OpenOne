"""
Subspace Topology Phase 1e: LayerNorm前后对比
=============================================

核心假说: LayerNorm是编码机制的"均衡器"
- 去除偏置方向的影响
- 将0.01%的语义信号放大到可操作范围

验证: 对比LayerNorm前后(或等效)的ID和奇异值谱

Run:
  python tests/glm5/subspace_topology_phase1e_layernorm.py --model qwen3
  python tests/glm5/subspace_topology_phase1e_layernorm.py --model glm4
  python tests/glm5/subspace_topology_phase1e_layernorm.py --model deepseek7b
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


def robust_svd(matrix, k=100):
    """鲁棒SVD，处理数值不稳定"""
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        return U, S, Vt
    except np.linalg.LinAlgError:
        from sklearn.decomposition import TruncatedSVD
        k = min(k, matrix.shape[1] - 1, matrix.shape[0] - 1)
        svd_obj = TruncatedSVD(n_components=k, random_state=42)
        svd_obj.fit(matrix.astype(np.float32))
        return None, svd_obj.singular_values_, svd_obj.components_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    print(f"\n模型: {model_info.name}, {n_layers}层, d_model={d_model}")
    
    # ========================================
    # 收集三种信号: LayerNorm前, LayerNorm后, 手动去均值+归一化
    # ========================================
    print(f"\n收集 {len(DIVERSE_PROMPTS)} 个prompt的残差流...")
    
    # 存储: pre_ln[layer_idx] = [token_activations], post_ln同理
    pre_ln_acts = {li: [] for li in range(n_layers)}   # MLP输出后的残差流(=LayerNorm输入)
    post_ln_acts = {li: [] for li in range(n_layers)}  # LayerNorm输出后的残差流
    manual_ln_acts = {li: [] for li in range(n_layers)} # 手动去均值+归一化后的残差流
    
    for pi, prompt in enumerate(DIVERSE_PROMPTS):
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = toks.input_ids.shape[1]
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(toks.input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Hook: 收集MLP输出后(LayerNorm输入前)和LayerNorm输出后
        captured_pre = {}
        captured_post = {}
        hooks = []
        
        for li in range(n_layers):
            layer = layers[li]
            
            # Hook1: MLP输出 → 这是LayerNorm的输入
            mlp = layer.mlp if hasattr(layer, "mlp") else None
            if mlp is not None:
                def make_pre_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured_pre[key] = output[0].detach().float()
                        else:
                            captured_pre[key] = output.detach().float()
                    return hook
                hooks.append(mlp.register_forward_hook(make_pre_hook(f"L{li}")))
            
            # Hook2: LayerNorm输出 → 需要找到post-attention LayerNorm
            # 不同模型位置不同
            # Qwen2/Qwen3: layer.post_attention_layernorm
            # GLM4: layer.post_self_attn_layernorm 或类似
            ln_post = None
            for ln_name in ["post_attention_layernorm", "post_self_attn_layernorm", "ln_2"]:
                if hasattr(layer, ln_name):
                    ln_post = getattr(layer, ln_name)
                    break
            
            if ln_post is not None:
                def make_post_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured_post[key] = output[0].detach().float()
                        else:
                            captured_post[key] = output.detach().float()
                    return hook
                hooks.append(ln_post.register_forward_hook(make_post_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            except Exception:
                pass
        
        for h in hooks:
            h.remove()
        
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured_pre:
                acts = captured_pre[key][0, :, :].cpu().numpy()
                pre_ln_acts[li].append(acts)
                
                # 手动模拟LayerNorm效果: 去均值 + 归一化
                mean = acts.mean(axis=0, keepdims=True)
                acts_centered = acts - mean
                std = np.sqrt(np.mean(acts_centered**2) + 1e-5)
                acts_normalized = acts_centered / std
                manual_ln_acts[li].append(acts_normalized)
            
            if key in captured_post:
                acts = captured_post[key][0, :, :].cpu().numpy()
                post_ln_acts[li].append(acts)
        
        del captured_pre, captured_post
        gc.collect()
    
    # ========================================
    # 核心分析: 对比三种信号的ID和奇异值谱
    # ========================================
    print(f"\n{'='*80}")
    print("LayerNorm前后对比: ID, 奇异值谱, 偏置方向变化")
    print(f"{'='*80}")
    
    results = {}
    sample_layers = sorted(set([0, 1, 2, 3, 5] + list(range(0, n_layers, max(1, n_layers//10))) + [n_layers-2, n_layers-1]))
    sample_layers = sorted(set([l for l in sample_layers if l < n_layers]))
    
    for li in sample_layers:
        layer_result = {"layer": li}
        
        # 1. Pre-LayerNorm分析
        if len(pre_ln_acts[li]) > 0:
            all_pre = np.concatenate(pre_ln_acts[li], axis=0)
            all_pre = np.nan_to_num(all_pre, nan=0.0, posinf=0.0, neginf=0.0)
            mean_pre = all_pre.mean(axis=0)
            centered_pre = all_pre - mean_pre
            U_pre, S_pre, Vt_pre = robust_svd(centered_pre)
            
            if S_pre is not None and len(S_pre) > 0:
                total_id_pre = compute_participation_ratio(S_pre**2 / (all_pre.shape[0] - 1))
                # 去除Rank-1后的ID
                if U_pre is not None:
                    residual_pre = centered_pre - U_pre[:, :1] @ np.diag(S_pre[:1]) @ Vt_pre[:1, :]
                else:
                    proj = centered_pre @ Vt_pre[:1, :].T
                    residual_pre = centered_pre - proj @ Vt_pre[:1, :]
                _, S_pre_debiased, _ = robust_svd(residual_pre)
                id_pre_debiased = compute_participation_ratio(S_pre_debiased**2 / (all_pre.shape[0] - 1)) if S_pre_debiased is not None else 0
                
                var_ratio_r1 = float(S_pre[0]**2 / np.sum(S_pre**2)) if np.sum(S_pre**2) > 0 else 0
                
                layer_result["pre_ln"] = {
                    "total_ID": float(total_id_pre),
                    "ID_minus_1PC": float(id_pre_debiased),
                    "var_ratio_r1": var_ratio_r1,
                    "top5_S": [float(x) for x in S_pre[:5]],
                    "top5_S_ratio": [float(S_pre[i]/S_pre[0]) for i in range(min(5, len(S_pre)))],
                }
        
        # 2. Post-LayerNorm分析
        if len(post_ln_acts[li]) > 0:
            all_post = np.concatenate(post_ln_acts[li], axis=0)
            all_post = np.nan_to_num(all_post, nan=0.0, posinf=0.0, neginf=0.0)
            mean_post = all_post.mean(axis=0)
            centered_post = all_post - mean_post
            U_post, S_post, Vt_post = robust_svd(centered_post)
            
            if S_post is not None and len(S_post) > 0:
                total_id_post = compute_participation_ratio(S_post**2 / (all_post.shape[0] - 1))
                if U_post is not None:
                    residual_post = centered_post - U_post[:, :1] @ np.diag(S_post[:1]) @ Vt_post[:1, :]
                else:
                    proj = centered_post @ Vt_post[:1, :].T
                    residual_post = centered_post - proj @ Vt_post[:1, :]
                _, S_post_debiased, _ = robust_svd(residual_post)
                id_post_debiased = compute_participation_ratio(S_post_debiased**2 / (all_post.shape[0] - 1)) if S_post_debiased is not None else 0
                
                var_ratio_r1 = float(S_post[0]**2 / np.sum(S_post**2)) if np.sum(S_post**2) > 0 else 0
                
                layer_result["post_ln"] = {
                    "total_ID": float(total_id_post),
                    "ID_minus_1PC": float(id_post_debiased),
                    "var_ratio_r1": var_ratio_r1,
                    "top5_S": [float(x) for x in S_post[:5]],
                    "top5_S_ratio": [float(S_post[i]/S_post[0]) for i in range(min(5, len(S_post)))],
                }
        
        # 3. 手动去均值+归一化分析
        if len(manual_ln_acts[li]) > 0:
            all_manual = np.concatenate(manual_ln_acts[li], axis=0)
            all_manual = np.nan_to_num(all_manual, nan=0.0, posinf=0.0, neginf=0.0)
            mean_manual = all_manual.mean(axis=0)
            centered_manual = all_manual - mean_manual
            _, S_manual, Vt_manual = robust_svd(centered_manual)
            
            if S_manual is not None and len(S_manual) > 0:
                total_id_manual = compute_participation_ratio(S_manual**2 / (all_manual.shape[0] - 1))
                
                layer_result["manual_ln"] = {
                    "total_ID": float(total_id_manual),
                    "top5_S": [float(x) for x in S_manual[:5]],
                    "top5_S_ratio": [float(S_manual[i]/S_manual[0]) for i in range(min(5, len(S_manual)))],
                }
        
        results[f"L{li}"] = layer_result
        
        # 打印
        pre = layer_result.get("pre_ln", {})
        post = layer_result.get("post_ln", {})
        manual = layer_result.get("manual_ln", {})
        
        pre_id = pre.get("total_ID", 0)
        post_id = post.get("total_ID", 0)
        manual_id = manual.get("total_ID", 0)
        pre_id_d = pre.get("ID_minus_1PC", 0)
        post_id_d = post.get("ID_minus_1PC", 0)
        pre_vr1 = pre.get("var_ratio_r1", 0)
        post_vr1 = post.get("var_ratio_r1", 0)
        
        print(f"\n  L{li:2d}: pre_LN  ID={pre_id:7.2f}  ID(-1PC)={pre_id_d:7.2f}  var(R1)={pre_vr1:.4f}")
        print(f"        post_LN ID={post_id:7.2f}  ID(-1PC)={post_id_d:7.2f}  var(R1)={post_vr1:.4f}")
        print(f"        manual   ID={manual_id:7.2f}")
        
        # LayerNorm效果: ID变化
        if pre_id > 0 and post_id > 0:
            id_ratio = post_id / pre_id
            print(f"        LN效果: total_ID ×{id_ratio:.2f}", end="")
            if id_ratio > 1.5:
                print(" ← LayerNorm显著提升了有效维度!")
            elif id_ratio < 0.7:
                print(" ← LayerNorm降低了有效维度")
            else:
                print()
    
    # ========================================
    # 核心对比总结
    # ========================================
    print(f"\n{'='*80}")
    print("核心对比: LayerNorm如何改变有效维度")
    print(f"{'='*80}")
    
    for li in sample_layers:
        r = results.get(f"L{li}", {})
        pre = r.get("pre_ln", {})
        post = r.get("post_ln", {})
        
        pre_id = pre.get("total_ID", 0)
        post_id = post.get("total_ID", 0)
        pre_id_d = pre.get("ID_minus_1PC", 0)
        post_id_d = post.get("ID_minus_1PC", 0)
        pre_vr1 = pre.get("var_ratio_r1", 0)
        post_vr1 = post.get("var_ratio_r1", 0)
        
        if pre_id > 0 and post_id > 0:
            print(f"  L{li:2d}: total_ID {pre_id:7.2f}→{post_id:7.2f} (×{post_id/pre_id:.2f})  "
                  f"ID(-1PC) {pre_id_d:7.2f}→{post_id_d:7.2f}  "
                  f"var(R1) {pre_vr1:.3f}→{post_vr1:.3f}")
    
    # 保存
    out_path = OUTPUT_DIR / f"exp1e_layernorm_effect_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    release_model(model)
    print("Done!")


if __name__ == "__main__":
    main()
