"""
Subspace Topology Phase 1j: 真正的跨位置总语义维度
====================================================

核心问题: 如果每个位置有~15维语义空间且跨位置80%正交，
那么合并所有位置后的总语义维度是多少？

方法: 对每个位置先各自中心化(去除位置偏置)，然后合并计算ID
对比: 直接合并(不中心化)的ID → 之前显示只有1-2维(假性)

Run:
  python tests/glm5/subspace_topology_phase1j_total_semantic_dim.py --model qwen3
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
    "The cat sat on the warm mat.",
    "Seven times eight equals fifty.",
]


def compute_participation_ratio(eigenvalues):
    lam = np.array(eigenvalues, dtype=np.float64)
    lam = lam[lam > 1e-12]
    if len(lam) == 0:
        return 0.0
    return float((np.sum(lam))**2 / np.sum(lam**2))


def robust_svd(matrix, k=None):
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        return U, S, Vt
    except np.linalg.LinAlgError:
        from sklearn.decomposition import TruncatedSVD
        k = k or min(200, matrix.shape[1] - 1, matrix.shape[0] - 1)
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
    # 多种方式计算总语义维度
    # ========================================
    target_layers = sorted(set(
        [0, 1, 5, 6] + 
        list(range(0, n_layers, max(1, n_layers//6))) + 
        [n_layers-2, n_layers-1]
    ))
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    valid_positions = [p for p in range(max_len) 
                       if len(pos_layer_acts[p][0]) >= 10]
    
    print(f"\n有效位置: {valid_positions}")
    print(f"目标层: {target_layers}")
    
    results = {}
    
    for li in target_layers:
        print(f"\n{'='*70}")
        print(f"  Layer {li}")
        print(f"{'='*70}")
        
        layer_result = {"layer": li}
        
        # ===== 方法1: 直接合并(不中心化) — baseline =====
        all_acts_raw = []
        for pos in valid_positions:
            all_acts_raw.extend(pos_layer_acts[pos][li])
        all_acts_raw = np.array(all_acts_raw)
        
        mean_all = all_acts_raw.mean(axis=0)
        centered_all = all_acts_raw - mean_all
        _, S_raw, _ = robust_svd(centered_all)
        id_raw = compute_participation_ratio(S_raw**2 / (len(all_acts_raw) - 1)) if S_raw is not None else 0
        
        # 去偏置ID(去1个PC)
        if S_raw is not None and len(S_raw) > 1:
            var_ratio_1 = S_raw[0]**2 / np.sum(S_raw**2)
            residual_1 = centered_all - (centered_all @ robust_svd(centered_all)[2][:1, :].T) @ robust_svd(centered_all)[2][:1, :]
            _, S_r1, _ = robust_svd(residual_1)
            id_debiased_1 = compute_participation_ratio(S_r1**2 / (len(all_acts_raw) - 1)) if S_r1 is not None else 0
        else:
            var_ratio_1 = 1.0
            id_debiased_1 = 0
        
        print(f"  方法1(直接合并): total_ID={id_raw:.1f}, debiased_ID(-1PC)={id_debiased_1:.1f}, PC1方差={var_ratio_1:.4f}")
        
        # ===== 方法2: 逐位置中心化后合并 — 真正的总语义维度 =====
        per_position_centered = []
        for pos in valid_positions:
            acts = np.array(pos_layer_acts[pos][li])
            mean_p = acts.mean(axis=0)
            centered_p = acts - mean_p
            per_position_centered.append(centered_p)
        
        all_centered = np.vstack(per_position_centered)  # [N_total, d_model]
        _, S_centered, _ = robust_svd(all_centered)
        id_centered = compute_participation_ratio(S_centered**2 / (len(all_centered) - 1)) if S_centered is not None else 0
        
        # 去偏置(去1个PC of the centered data)
        if S_centered is not None and len(S_centered) > 1:
            U_c, S_c, Vt_c = robust_svd(all_centered)
            if U_c is not None:
                residual_c = all_centered - U_c[:, :1] @ np.diag(S_c[:1]) @ Vt_c[:1, :]
            else:
                proj = all_centered @ Vt_c[:1, :].T
                residual_c = all_centered - proj @ Vt_c[:1, :]
            _, S_rc, _ = robust_svd(residual_c)
            id_centered_debiased = compute_participation_ratio(S_rc**2 / (len(all_centered) - 1)) if S_rc is not None else 0
            
            # 前N个PC的方差解释
            var_explained = np.cumsum(S_c[:50]**2) / np.sum(S_c**2)
        else:
            id_centered_debiased = 0
            var_explained = np.array([1.0])
        
        print(f"  方法2(逐位置中心化): total_ID={id_centered:.1f}, debiased_ID={id_centered_debiased:.1f}")
        
        # ===== 方法3: 逐位置去偏置后合并 — 更彻底的偏置去除 =====
        per_position_debiased = []
        for pos in valid_positions:
            acts = np.array(pos_layer_acts[pos][li])
            mean_p = acts.mean(axis=0)
            centered_p = acts - mean_p
            _, S_p, Vt_p = robust_svd(centered_p)
            if S_p is not None and len(S_p) > 1:
                if U_c is not None:
                    # 去除每个位置的PC1
                    # 先做完整SVD
                    try:
                        U_p, S_p2, Vt_p2 = np.linalg.svd(centered_p, full_matrices=False)
                        debiased_p = centered_p - U_p[:, :1] @ np.diag(S_p2[:1]) @ Vt_p2[:1, :]
                    except:
                        proj = centered_p @ Vt_p[:1, :].T
                        debiased_p = centered_p - proj @ Vt_p[:1, :]
                else:
                    proj = centered_p @ Vt_p[:1, :].T
                    debiased_p = centered_p - proj @ Vt_p[:1, :]
                per_position_debiased.append(debiased_p)
            else:
                per_position_debiased.append(centered_p)
        
        all_debiased = np.vstack(per_position_debiased)
        _, S_all_deb, _ = robust_svd(all_debiased)
        id_all_debiased = compute_participation_ratio(S_all_deb**2 / (len(all_debiased) - 1)) if S_all_deb is not None else 0
        
        print(f"  方法3(逐位置去偏置后合并): total_ID={id_all_debiased:.1f}")
        
        # ===== 方法4: 奇异值衰减分析 — 多少PC解释95%方差? =====
        if S_centered is not None:
            n_90 = int(np.searchsorted(var_explained, 0.90)) + 1
            n_95 = int(np.searchsorted(var_explained, 0.95)) + 1
            n_99 = int(np.searchsorted(var_explained, 0.99)) + 1
            print(f"\n  奇异值衰减(逐位置中心化):")
            print(f"    90%方差需要 {n_90} 个PC")
            print(f"    95%方差需要 {n_95} 个PC")
            print(f"    99%方差需要 {n_99} 个PC")
            print(f"    前20个S值: {S_centered[:20].round(2)}")
        else:
            n_90 = n_95 = n_99 = 0
        
        # ===== 方法5: 理论预测对比 =====
        # 如果每个位置有d_p=15维独立子空间，N=7个位置，overlap=0.15
        # 理论总维度 ≈ N * d_p * (1-overlap) + d_p * overlap * N
        #            ≈ d_p * N * (1 - (1-overlap)*(1-1/N))
        n_pos = len(valid_positions)
        avg_per_pos_id = np.mean([compute_participation_ratio(
            robust_svd(np.array(pos_layer_acts[p][li]) - np.array(pos_layer_acts[p][li]).mean(axis=0))[1]**2 / (len(pos_layer_acts[p][li]) - 1)
        ) for p in valid_positions if len(pos_layer_acts[p][li]) >= 5])
        
        print(f"\n  理论分析:")
        print(f"    位置数: {n_pos}")
        print(f"    平均单位置ID: {avg_per_pos_id:.1f}")
        print(f"    方法2总ID: {id_centered:.1f}")
        print(f"    方法3总ID: {id_all_debiased:.1f}")
        print(f"    如果完全独立: {avg_per_pos_id * n_pos:.1f}")
        print(f"    实际/完全独立比: {id_all_debiased / (avg_per_pos_id * n_pos) if avg_per_pos_id * n_pos > 0 else 0:.2f}")
        
        # 保存
        layer_result["method1_raw"] = {"total_ID": id_raw, "debiased_ID": id_debiased_1, "PC1_var": var_ratio_1}
        layer_result["method2_centered"] = {"total_ID": id_centered, "debiased_ID": id_centered_debiased}
        layer_result["method3_debiased"] = {"total_ID": id_all_debiased}
        layer_result["method4_pc_count"] = {"n_90": n_90, "n_95": n_95, "n_99": n_99}
        layer_result["method5_theory"] = {
            "n_positions": n_pos,
            "avg_per_pos_ID": avg_per_pos_id,
            "predicted_independent": avg_per_pos_id * n_pos,
            "actual_total": id_all_debiased,
        }
        results[f"L{li}"] = layer_result
    
    # ========================================
    # 核心总结
    # ========================================
    print(f"\n{'='*80}")
    print("核心总结: 跨位置总语义维度")
    print(f"{'='*80}")
    
    for li in target_layers:
        r = results[f"L{li}"]
        m1 = r["method1_raw"]
        m2 = r["method2_centered"]
        m3 = r["method3_debiased"]
        m5 = r["method5_theory"]
        print(f"  L{li:2d}: raw_debiased={m1['debiased_ID']:5.1f}  "
              f"centered={m2['total_ID']:5.1f}  "
              f"per_pos_debiased_merged={m3['total_ID']:5.1f}  "
              f"theory(indep)={m5['predicted_independent']:5.1f}  "
              f"ratio={m5['actual_total']/m5['predicted_independent']:.2f}" if m5['predicted_independent'] > 0 else "")
    
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
    
    out_path = OUTPUT_DIR / f"exp1j_total_semantic_dim_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    release_model(model)
    print("Done!")


if __name__ == "__main__":
    main()
