"""
Subspace Topology Phase 1h: 语义子空间对齐分析
================================================

核心问题: 不同token位置的~15维语义子空间是否共享同一组基？
- 如果子空间高度对齐 → 网络使用"通用语义基底"，位置信息只修改系数
- 如果子空间低对齐 → 网络使用"位置特异表示"，每个位置有不同的语义编码

方法: CCA (Canonical Correlation Analysis) + 子空间投影重叠度

Run:
  python tests/glm5/subspace_topology_phase1h_subspace_align.py --model qwen3
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

from model_utils import load_model, get_layers, get_model_info, release_model

OUTPUT_DIR = Path("results/subspace_topology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 句子集
SENTENCES = [
    "The apple is red and sweet.",
    "Paris is the capital of France.",
    "Water boils at 100 degrees.",
    "The apple is NOT red at all.",
    "Is the apple really red now?",
    "Justice is a fundamental concept.",
    "John gave Mary a book today.",
    "The sky is blue and clear.",
    "The sky is NOT blue today.",
    "Is the sky blue and clear?",
    "All cats are animals, clearly.",
    "Write code to sort a list.",
    "What is 49 plus 2 equal?",
    "If A=B and B=C, then A=",
    "The earth revolves around us.",
    "Ice melts when heated slowly.",
    "Books are made of paper too.",
    "Birds can fly in the sky.",
    "Freedom means different things.",
    "Time passes differently always.",
    "The physical color is green.",
    "It is not true that red.",
    "5, 6, 7, and 8 follow on.",
    "What is 15 times 7 equal?",
    "Translate this to French now.",
    "Create a class for linked list.",
    "The shape of a ball is round.",
    "Alice told Bob a big secret.",
    "Dogs are loyal and friendly.",
    "饕餮是一种传说中的神兽。",
]


def compute_principal_angles(subspace1, subspace2):
    """
    计算两个子空间之间的主角度(principal angles)
    
    Args:
        subspace1: [d, k1] 正交基矩阵 (列向量是基)
        subspace2: [d, k2] 正交基矩阵
    
    Returns:
        angles: list of principal angles (in degrees)
    """
    # 投影矩阵
    Q1 = np.linalg.qr(subspace1)[0]  # [d, k1]
    Q2 = np.linalg.qr(subspace2)[0]  # [d, k2]
    
    # SVD of Q1^T Q2 → 主角度的余弦 = 奇异值
    M = Q1.T @ Q2
    svals = np.linalg.svd(M, compute_uv=False)
    svals = np.clip(svals, 0, 1)
    angles = np.arccos(svals) * 180 / np.pi
    
    return sorted(angles)


def subspace_overlap_ratio(subspace1, subspace2, k=None):
    """
    计算子空间重叠度: subspace1在subspace2上的投影能量比
    
    ratio = ||P_2 v||^2 / ||v||^2, 对v in subspace1的所有基向量平均
    """
    Q1 = np.linalg.qr(subspace1)[0][:, :k] if k else np.linalg.qr(subspace1)[0]
    Q2 = np.linalg.qr(subspace2)[0]
    
    # Q2的投影矩阵
    P2 = Q2 @ Q2.T  # [d, d]
    
    # 每个Q1基向量在Q2上的投影能量
    proj = Q1.T @ P2 @ Q1  # [k1, k1]
    overlap = np.mean(np.diag(proj))
    
    return float(overlap)


def get_debiased_subspace(acts, n_dims=20):
    """
    获取去偏置后的语义子空间基
    
    Args:
        acts: [n, d] 激活矩阵
        n_dims: 返回的维度数
    
    Returns:
        V_semantic: [d, n_dims] 语义子空间的正交基
    """
    acts = np.nan_to_num(acts, nan=0.0, posinf=0.0, neginf=0.0)
    mean = acts.mean(axis=0)
    centered = acts - mean
    
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        from sklearn.decomposition import TruncatedSVD
        k = min(n_dims + 10, centered.shape[1] - 1, centered.shape[0] - 1)
        svd_obj = TruncatedSVD(n_components=k, random_state=42)
        svd_obj.fit(centered.astype(np.float32))
        S = svd_obj.singular_values_
        Vt = svd_obj.components_
        U = None
    
    # 去偏置: 跳过第一个主成分
    n_return = min(n_dims, len(S) - 1)
    V_semantic = Vt[1:1+n_return, :].T  # [d, n_dims]
    
    return V_semantic, S


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
    
    # 按位置和层存储
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
    # 子空间对齐分析
    # ========================================
    target_layers = sorted(set(
        [0, 1, 5, 6] + 
        list(range(0, n_layers, max(1, n_layers//6))) + 
        [n_layers-2, n_layers-1]
    ))
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    valid_positions = [p for p in range(max_len) 
                       if len(pos_layer_acts[p][0]) >= 10]
    n_dims = 15  # 语义子空间维度
    
    print(f"\n有效位置: {valid_positions}")
    print(f"目标层: {target_layers}")
    print(f"分析维度: {n_dims}")
    
    results = {}
    
    for li in target_layers:
        print(f"\n{'='*60}")
        print(f"  Layer {li}")
        print(f"{'='*60}")
        
        layer_result = {"layer": li}
        
        # 1. 获取每个位置的语义子空间
        pos_subspaces = {}
        pos_spectrums = {}
        for pos in valid_positions:
            acts = np.array(pos_layer_acts[pos][li])
            if len(acts) < 5:
                continue
            V_sem, S = get_debiased_subspace(acts, n_dims=n_dims)
            pos_subspaces[pos] = V_sem
            pos_spectrums[pos] = S[:10].tolist()  # 前10个奇异值
        
        # 2. 逐位置对的主角度
        angle_matrix = {}
        positions_with_data = sorted(pos_subspaces.keys())
        
        for i, p1 in enumerate(positions_with_data):
            for j, p2 in enumerate(positions_with_data):
                if j <= i:
                    continue
                angles = compute_principal_angles(pos_subspaces[p1], pos_subspaces[p2])
                # 只保留最小的n_dims个角度
                k = min(n_dims, len(angles))
                angles_k = angles[:k]
                angle_matrix[f"{p1}-{p2}"] = {
                    "angles": angles_k,
                    "mean_angle": float(np.mean(angles_k)),
                    "median_angle": float(np.median(angles_k)),
                    "min_angle": float(angles_k[0]),
                }
        
        # 3. 子空间重叠度矩阵
        overlap_matrix = {}
        for i, p1 in enumerate(positions_with_data):
            for j, p2 in enumerate(positions_with_data):
                if j <= i:
                    continue
                overlap = subspace_overlap_ratio(pos_subspaces[p1], pos_subspaces[p2], k=n_dims)
                overlap_matrix[f"{p1}-{p2}"] = float(overlap)
        
        # 4. 与"随机子空间"的对比 (null model)
        # 随机两个d_model维空间中的n_dims维子空间，期望重叠度 = n_dims/d_model
        expected_random_overlap = n_dims / d_model
        
        # 5. 相邻位置 vs 远距离位置的对齐度对比
        adjacent_overlaps = []
        distant_overlaps = []
        for key, overlap in overlap_matrix.items():
            p1, p2 = map(int, key.split('-'))
            if abs(p2 - p1) <= 1:
                adjacent_overlaps.append(overlap)
            elif abs(p2 - p1) >= 3:
                distant_overlaps.append(overlap)
        
        # 6. 打印结果
        print(f"\n  位置对齐度 (overlap ratio, 期望随机值={expected_random_overlap:.4f}):")
        for key in sorted(overlap_matrix.keys(), key=lambda x: int(x.split('-')[0])*100+int(x.split('-')[1])):
            p1, p2 = map(int, key.split('-'))
            overlap = overlap_matrix[key]
            mean_angle = angle_matrix[key]["mean_angle"]
            marker = " **" if overlap > 0.5 else ""
            print(f"    pos{p1}-pos{p2}: overlap={overlap:.4f}  mean_angle={mean_angle:.1f}°{marker}")
        
        print(f"\n  统计:")
        print(f"    相邻位置平均重叠度: {np.mean(adjacent_overlaps):.4f}" if adjacent_overlaps else "    无相邻位置")
        print(f"    远距离位置平均重叠度: {np.mean(distant_overlaps):.4f}" if distant_overlaps else "    无远距离位置")
        print(f"    期望随机重叠度: {expected_random_overlap:.4f}")
        print(f"    实际/随机比值: {np.mean(list(overlap_matrix.values()))/expected_random_overlap:.1f}x" if overlap_matrix else "")
        
        # 7. 偏置方向的跨位置一致性
        # 检查各位置的PC1（偏置方向）是否对齐
        bias_alignment = {}
        bias_directions = {}
        for pos in valid_positions:
            acts = np.array(pos_layer_acts[pos][li])
            if len(acts) < 5:
                continue
            acts = np.nan_to_num(acts, nan=0.0, posinf=0.0, neginf=0.0)
            mean = acts.mean(axis=0)
            centered = acts - mean
            try:
                _, S, Vt = np.linalg.svd(centered, full_matrices=False)
            except:
                continue
            bias_directions[pos] = Vt[0, :]  # PC1 = 偏置方向
        
        for i, p1 in enumerate(bias_directions):
            for j, p2 in enumerate(bias_directions):
                if j <= i:
                    continue
                cos = abs(np.dot(bias_directions[p1], bias_directions[p2]))
                bias_alignment[f"{p1}-{p2}"] = float(cos)
        
        if bias_alignment:
            print(f"\n  偏置方向(PC1)跨位置对齐度 (|cos|):")
            for key in sorted(bias_alignment.keys(), key=lambda x: int(x.split('-')[0])*100+int(x.split('-')[1])):
                print(f"    pos{key}: |cos|={bias_alignment[key]:.4f}")
        
        layer_result["overlap_matrix"] = overlap_matrix
        layer_result["angle_matrix"] = angle_matrix
        layer_result["bias_alignment"] = bias_alignment
        layer_result["pos_spectrums"] = pos_spectrums
        layer_result["stats"] = {
            "adjacent_overlap_mean": float(np.mean(adjacent_overlaps)) if adjacent_overlaps else 0,
            "distant_overlap_mean": float(np.mean(distant_overlaps)) if distant_overlaps else 0,
            "random_overlap_expected": expected_random_overlap,
            "bias_alignment_mean": float(np.mean(list(bias_alignment.values()))) if bias_alignment else 0,
        }
        results[f"L{li}"] = layer_result
    
    # ========================================
    # 核心总结
    # ========================================
    print(f"\n{'='*80}")
    print("核心总结: 跨位置语义子空间对齐度")
    print(f"{'='*80}")
    
    for li in target_layers:
        r = results[f"L{li}"]
        stats = r["stats"]
        bias_mean = stats.get("bias_alignment_mean", 0)
        print(f"  L{li:2d}: 相邻重叠={stats['adjacent_overlap_mean']:.3f}  "
              f"远距重叠={stats['distant_overlap_mean']:.3f}  "
              f"随机期望={stats['random_overlap_expected']:.4f}  "
              f"偏置对齐={bias_mean:.3f}")
    
    # 保存 (确保所有numpy类型转换为Python原生类型)
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
    
    out_path = OUTPUT_DIR / f"exp1h_subspace_align_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    release_model(model)
    print("Done!")


if __name__ == "__main__":
    main()
