"""
Subspace Topology Phase 1i: 跨位置共享语义核心提取与解码
========================================================

核心问题: 不同token位置的语义子空间虽然有5-23%重叠，这个"共享核心"编码了什么？

方法:
1. 提取所有位置的公共子空间(GRV: Generalized Right Vectors)
2. 解码共享核心的语义内容(投影到W_U词空间)
3. 与位置特异成分对比

Run:
  python tests/glm5/subspace_topology_phase1i_shared_core.py --model qwen3
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
import gc
import json
from pathlib import Path

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U

OUTPUT_DIR = Path("results/subspace_topology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 扩展句子集 — 增加语义多样性
SENTENCES = [
    # 颜色/属性
    "The apple is red and sweet.",
    "The sky is blue and clear.",
    "The grass is green and soft.",
    "The snow is white and cold.",
    "The night is dark and quiet.",
    # 否定
    "The apple is NOT red at all.",
    "The sky is NOT blue today.",
    "It is not true that red.",
    # 疑问
    "Is the apple really red now?",
    "Is the sky blue and clear?",
    "What is 49 plus 2 equal?",
    "What is 15 times 7 equal?",
    # 知识
    "Paris is the capital of France.",
    "Water boils at 100 degrees.",
    "The earth revolves around us.",
    "Ice melts when heated slowly.",
    # 逻辑
    "If A=B and B=C, then A=",
    "All cats are animals, clearly.",
    "5, 6, 7, and 8 follow on.",
    # 动作
    "John gave Mary a book today.",
    "Alice told Bob a big secret.",
    "Write code to sort a list.",
    "Create a class for linked list.",
    # 抽象
    "Justice is a fundamental concept.",
    "Freedom means different things.",
    "Time passes differently always.",
    "Books are made of paper too.",
    # 形状/物理
    "The shape of a ball is round.",
    "Birds can fly in the sky.",
    # 语言
    "Translate this to French now.",
    "饕餮是一种传说中的神兽。",
    "Dogs are loyal and friendly.",
]


def extract_shared_subspace(subspace_list, n_dims=15):
    """
    提取多个子空间的"共享核心"
    
    方法: 将所有子空间的基向量拼接，做SVD，取前n_dims个右奇异向量
    这些是"在最多子空间中有投影"的方向
    
    Args:
        subspace_list: list of [d, k] 矩阵 (每个位置的去偏置语义子空间基)
        n_dims: 返回的共享维度数
    
    Returns:
        shared_V: [d, n_shared] 共享子空间基
        eigenvalues: 各共享维度的"共享强度"
    """
    # 将所有子空间基拼接
    all_bases = np.hstack(subspace_list)  # [d, k*n_pos]
    
    # SVD: 找到跨越最多子空间的方向
    U, S, Vt = np.linalg.svd(all_bases, full_matrices=False)
    
    # 前 n_dims 个右奇异向量 = 共享核心
    n_shared = min(n_dims, len(S))
    shared_V = U[:, :n_shared]  # [d, n_shared]
    
    return shared_V, S[:n_shared]


def compute_subspace_energy(acts_matrix, subspace_V):
    """
    计算激活矩阵在给定子空间中的投影能量比
    
    Args:
        acts_matrix: [n, d] 中心化的激活矩阵
        subspace_V: [d, k] 正交基
    
    Returns:
        ratio: 投影能量 / 总能量
    """
    if acts_matrix.shape[0] < 2:
        return 0.0
    total_energy = np.sum(acts_matrix ** 2)
    if total_energy < 1e-20:
        return 0.0
    
    # 投影
    Q, _ = np.linalg.qr(subspace_V)
    proj = acts_matrix @ Q @ Q.T  # [n, d]
    proj_energy = np.sum(proj ** 2)
    
    return float(proj_energy / total_energy)


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
    
    # 获取W_U用于解码
    print("加载W_U...")
    W_U = get_W_U(model, args.model)  # [vocab, d_model]
    
    # ========================================
    # 收集激活
    # ========================================
    print(f"\n收集 {len(SENTENCES)} 个句子的残差流...")
    
    max_len = 0
    tokenized = []
    for sent in SENTENCES:
        toks = tokenizer(sent, return_tensors="pt")
        seq_len = toks.input_ids.shape[1]
        tokenized.append((sent, toks, seq_len))
        max_len = max(max_len, seq_len)
    
    pos_layer_acts = {pos: {li: [] for li in range(n_layers)} for pos in range(max_len)}
    
    for si, (sent, toks, seq_len) in enumerate(tokenized):
        toks = toks.to(device)
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
                for pos in range(seq_len):
                    pos_layer_acts[pos][li].append(acts[pos, :])
        
        del captured
        gc.collect()
    
    # ========================================
    # 分析
    # ========================================
    target_layers = sorted(set(
        [0, 1, 5, 6] + 
        list(range(0, n_layers, max(1, n_layers//6))) + 
        [n_layers-2, n_layers-1]
    ))
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    valid_positions = [p for p in range(max_len) 
                       if len(pos_layer_acts[p][0]) >= 10]
    n_dims = 15
    
    print(f"\n有效位置: {valid_positions}")
    print(f"目标层: {target_layers}")
    
    results = {}
    
    for li in target_layers:
        print(f"\n{'='*70}")
        print(f"  Layer {li}")
        print(f"{'='*70}")
        
        layer_result = {"layer": li}
        
        # 1. 获取每个位置的去偏置语义子空间
        pos_subspaces = {}
        pos_centered = {}
        for pos in valid_positions:
            acts = np.array(pos_layer_acts[pos][li])
            if len(acts) < 5:
                continue
            acts = np.nan_to_num(acts, nan=0.0, posinf=0.0, neginf=0.0)
            mean = acts.mean(axis=0)
            centered = acts - mean
            pos_centered[pos] = centered
            
            try:
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            except:
                continue
            
            # 去偏置: 跳过PC1
            n_ret = min(n_dims, len(S) - 1)
            pos_subspaces[pos] = Vt[1:1+n_ret, :].T  # [d, n_dims]
        
        if len(pos_subspaces) < 2:
            print("  位置数不足，跳过")
            continue
        
        # 2. 提取共享核心
        subspace_list = [pos_subspaces[p] for p in sorted(pos_subspaces.keys())]
        shared_V, shared_S = extract_shared_subspace(subspace_list, n_dims=n_dims)
        
        print(f"\n  共享核心奇异值: {shared_S[:10].round(3)}")
        print(f"  共享核心有效维度(PR): ", end="")
        lam = shared_S ** 2
        lam = lam[lam > 1e-12]
        pr = float(np.sum(lam)**2 / np.sum(lam**2)) if len(lam) > 0 else 0
        print(f"{pr:.1f}")
        
        # 3. 共享核心 vs 位置特异成分的能量比
        print(f"\n  各位置在共享核心中的能量比:")
        pos_energy_in_shared = {}
        pos_energy_in_specific = {}
        for pos in sorted(pos_centered.keys()):
            centered = pos_centered[pos]
            e_shared = compute_subspace_energy(centered, shared_V)
            
            # 位置特异 = 总能量 - 共享核心能量
            e_specific = max(0, 1.0 - e_shared)
            pos_energy_in_shared[pos] = e_shared
            pos_energy_in_specific[pos] = e_specific
            
            marker = " ← LAST" if pos == max(valid_positions) else ""
            print(f"    pos{pos}: shared={e_shared:.3f}  specific={e_specific:.3f}{marker}")
        
        # 4. 解码共享核心维度 → W_U投影
        print(f"\n  共享核心维度解码 (W_U投影top-5词):")
        shared_dim_decode = {}
        for di in range(min(10, shared_V.shape[1])):
            direction = shared_V[:, di]  # [d]
            # 归一化
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction = direction / norm
            
            # 投影到W_U
            cos_sims = W_U @ direction  # [vocab]
            top_idx = np.argsort(cos_sims)[-10:][::-1]
            top_words = [(tokenizer.decode([idx]).strip(), float(cos_sims[idx])) 
                        for idx in top_idx if cos_sims[idx] > 0.1]
            
            shared_dim_decode[f"dim{di}"] = top_words[:5]
            if top_words:
                words_str = ", ".join([f"{w}({c:.2f})" for w, c in top_words[:5]])
                print(f"    dim{di}: {words_str}")
        
        # 5. 对比: 位置特异维度的解码
        print(f"\n  位置特异维度解码 (pos3为例, W_U投影top-5词):")
        sample_pos = min(3, max(valid_positions))
        if sample_pos in pos_subspaces:
            V_pos = pos_subspaces[sample_pos]
            pos_dim_decode = {}
            for di in range(min(5, V_pos.shape[1])):
                direction = V_pos[:, di]
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    continue
                direction = direction / norm
                
                cos_sims = W_U @ direction
                top_idx = np.argsort(cos_sims)[-10:][::-1]
                top_words = [(tokenizer.decode([idx]).strip(), float(cos_sims[idx])) 
                            for idx in top_idx if cos_sims[idx] > 0.1]
                
                pos_dim_decode[f"dim{di}"] = top_words[:5]
                if top_words:
                    words_str = ", ".join([f"{w}({c:.2f})" for w, c in top_words[:5]])
                    print(f"    dim{di}: {words_str}")
        
        # 6. 偏置方向(PC1)的解码
        print(f"\n  偏置方向(PC1)解码 (pos3为例):")
        if sample_pos in pos_centered:
            centered = pos_centered[sample_pos]
            try:
                _, S_p1, Vt_p1 = np.linalg.svd(centered, full_matrices=False)
                bias_dir = Vt_p1[0, :]
                bias_dir = bias_dir / np.linalg.norm(bias_dir)
                cos_sims = W_U @ bias_dir
                top_idx = np.argsort(cos_sims)[-10:][::-1]
                top_words = [(tokenizer.decode([idx]).strip(), float(cos_sims[idx])) 
                            for idx in top_idx if cos_sims[idx] > 0.1]
                if top_words:
                    words_str = ", ".join([f"{w}({c:.2f})" for w, c in top_words[:5]])
                    print(f"    PC1: {words_str}")
                
                # PC1的方差占比
                var_ratio = S_p1[0]**2 / np.sum(S_p1**2)
                print(f"    PC1方差占比: {var_ratio:.4f}")
            except:
                pass
        
        # 保存
        layer_result["shared_PR"] = pr
        layer_result["shared_S"] = shared_S[:20].tolist()
        layer_result["pos_energy_in_shared"] = pos_energy_in_shared
        layer_result["shared_dim_decode"] = shared_dim_decode
        results[f"L{li}"] = layer_result
    
    # ========================================
    # 核心总结
    # ========================================
    print(f"\n{'='*80}")
    print("核心总结: 跨位置共享语义核心")
    print(f"{'='*80}")
    
    for li in target_layers:
        r = results.get(f"L{li}", {})
        pr = r.get("shared_PR", 0)
        energies = r.get("pos_energy_in_shared", {})
        avg_shared = np.mean(list(energies.values())) if energies else 0
        print(f"  L{li:2d}: shared_core_PR={pr:.1f}  avg_energy_in_shared={avg_shared:.3f}")
    
    # 保存
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    
    out_path = OUTPUT_DIR / f"exp1i_shared_core_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    release_model(model)
    print("Done!")


if __name__ == "__main__":
    main()
