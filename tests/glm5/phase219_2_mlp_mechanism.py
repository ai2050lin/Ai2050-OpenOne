"""
Phase 219-2: MLP约束传播机制验证

Phase 219-1的核心发现:
  - MLP主导约束传播(70-95%), 不是Attention
  - 约束信号在W_U低奇异值方向更多(31.7% > 27.2%)
  - L14是MLP贡献峰值(85.9%)

本实验目标:
  1. 单层MLP ablation对后续约束传播的影响
  2. GLU门控模式: sg vs pl的门控差异
  3. MLP权重矩阵的奇异值结构
  4. MLP残差增量的方向分析

执行时间: 2026-05-17 20:45
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
import json
import time
from pathlib import Path

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SVA_PAIRS = [
    ("The cat chases", "The cats chase"),
    ("The dog runs", "The dogs run"),
    ("The bird sings", "The birds sing"),
    ("The girl reads", "The girls read"),
    ("The boy walks", "The boys walk"),
    ("The tree falls", "The trees fall"),
    ("The car moves", "The cars move"),
    ("The child plays", "The children play"),
    ("The woman writes", "The women write"),
    ("The man speaks", "The men speak"),
]

def compute_kl(p, q, eps=1e-10):
    p = p.float() + eps
    q = q.float() + eps
    p = p / p.sum()
    q = q / q.sum()
    return (0.5 * (p * (p/q).log()).sum() + 0.5 * (q * (q/p).log()).sum()).item()

def experiment_mlp_single_ablation(model):
    """
    实验1: 单层MLP ablation
    对每层, 将sg和pl的mlp_out替换为0(等价于ablate MLP)
    测量后续层的KL变化
    """
    print("\n=== 实验1: 单层MLP Ablation ===")
    
    n_layers = model.cfg.n_layers
    
    # 基线KL
    baseline_kl = []
    for sg, pl in SVA_PAIRS:
        with torch.no_grad():
            _, cache_sg = model.run_with_cache(sg)
            _, cache_pl = model.run_with_cache(pl)
        
        pair_kl = []
        for l in range(n_layers):
            h_sg = cache_sg["resid_post", l][0, -1]
            h_pl = cache_pl["resid_post", l][0, -1]
            logits_sg = h_sg.float() @ model.W_U.float()
            logits_pl = h_pl.float() @ model.W_U.float()
            probs_sg = torch.softmax(logits_sg, dim=-1)
            probs_pl = torch.softmax(logits_pl, dim=-1)
            kl = compute_kl(probs_sg, probs_pl)
            pair_kl.append(kl)
        baseline_kl.append(pair_kl)
    
    baseline_kl = np.array(baseline_kl)
    
    # 逐层ablate MLP: 在ablate_layer层, 把MLP输出置零
    # 简化方法: 直接计算 resid_post_ablated = resid_mid (不含mlp_out)
    ablation_results = {}
    
    for ablate_layer in range(n_layers):
        kl_after = []
        for sg, pl in SVA_PAIRS:
            with torch.no_grad():
                _, cache_sg = model.run_with_cache(sg)
                _, cache_pl = model.run_with_cache(pl)
            
            # 在ablate_layer层, 用resid_mid代替resid_post
            # resid_post = resid_mid + mlp_out
            # ablated: resid_post_ablated = resid_mid = resid_post - mlp_out
            h_sg_ablated = cache_sg["resid_post", ablate_layer][0, -1] - cache_sg["mlp_out", ablate_layer][0, -1]
            h_pl_ablated = cache_pl["resid_post", ablate_layer][0, -1] - cache_pl["mlp_out", ablate_layer][0, -1]
            
            # 计算ablated后的KL (用W_U投影)
            logits_sg = h_sg_ablated.float() @ model.W_U.float()
            logits_pl = h_pl_ablated.float() @ model.W_U.float()
            probs_sg = torch.softmax(logits_sg, dim=-1)
            probs_pl = torch.softmax(logits_pl, dim=-1)
            kl = compute_kl(probs_sg, probs_pl)
            kl_after.append(kl)
        
        mean_kl = np.mean(kl_after)
        baseline_at_layer = baseline_kl.mean(axis=0)[ablate_layer]
        kl_reduction = 1.0 - mean_kl / max(baseline_at_layer, 1e-10)
        ablation_results[ablate_layer] = {
            "kl_ablated": float(mean_kl),
            "kl_baseline": float(baseline_at_layer),
            "kl_reduction_pct": float(kl_reduction * 100)
        }
        
        if ablate_layer % 2 == 0 or ablate_layer == n_layers - 1:
            print(f"  L{ablate_layer:2d}: KL_baseline={baseline_at_layer:.4f}, KL_ablated={mean_kl:.4f}, reduction={kl_reduction*100:.1f}%")
    
    return ablation_results

def experiment_glu_gating_pattern(model):
    """
    实验2: GLU门控模式分析
    分析sg vs pl在MLP中的门控差异
    
    Qwen2.5的MLP结构:
      mlp_out = W_out @ (GeLU(W_gate @ h) ⊙ (W_up @ h))
    
    门控信号 = GeLU(W_gate @ h)
    上投影 = W_up @ h
    输出 = 门控 ⊙ 上投影
    """
    print("\n=== 实验2: GLU门控模式 ===")
    
    n_layers = model.cfg.n_layers
    
    gate_diff_norms = []  # [n_pairs, n_layers]
    up_diff_norms = []    # [n_pairs, n_layers]
    gate_dot_products = []  # [n_pairs, n_layers]
    
    for sg, pl in SVA_PAIRS:
        with torch.no_grad():
            _, cache_sg = model.run_with_cache(sg)
            _, cache_pl = model.run_with_cache(pl)
        
        pair_gate_diff = []
        pair_up_diff = []
        pair_gate_dot = []
        
        for l in range(n_layers):
            # 获取pre-activation (如果cache中有)
            # 否则用MLP权重手动计算
            h_sg = cache_sg["resid_mid", l][0, -1]  # MLP输入 = resid_mid
            h_pl = cache_pl["resid_mid", l][0, -1]
            
            # MLP权重
            W_gate = model.blocks[l].mlp.W_gate  # [d_model, d_mlp]
            W_in = model.blocks[l].mlp.W_in      # [d_model, d_mlp]
            
            # 门控信号: x @ W_gate -> GeLU
            gate_sg = torch.nn.functional.gelu(h_sg.float() @ W_gate.float())
            gate_pl = torch.nn.functional.gelu(h_pl.float() @ W_gate.float())
            
            # 线性投影: x @ W_in
            up_sg = h_sg.float() @ W_in.float()
            up_pl = h_pl.float() @ W_in.float()
            
            # 门控差异
            gate_diff = (gate_sg - gate_pl).norm().item()
            gate_norm = gate_sg.norm().item()
            
            # 上投影差异
            up_diff = (up_sg - up_pl).norm().item()
            up_norm = up_sg.norm().item()
            
            # 归一化差异
            rel_gate_diff = gate_diff / max(gate_norm, 1e-10)
            rel_up_diff = up_diff / max(up_norm, 1e-10)
            
            # 门控方向一致性 (cosine similarity)
            cos_sim = torch.nn.functional.cosine_similarity(
                gate_sg.unsqueeze(0), gate_pl.unsqueeze(0)
            ).item()
            
            pair_gate_diff.append(rel_gate_diff)
            pair_up_diff.append(rel_up_diff)
            pair_gate_dot.append(cos_sim)
        
        gate_diff_norms.append(pair_gate_diff)
        up_diff_norms.append(pair_up_diff)
        gate_dot_products.append(pair_gate_dot)
    
    gate_diff_norms = np.array(gate_diff_norms)
    up_diff_norms = np.array(up_diff_norms)
    gate_dot_products = np.array(gate_dot_products)
    
    # 打印
    print(f"\n{'Layer':>6} | {'Gate_rel_diff':>14} | {'Up_rel_diff':>12} | {'Gate_cos_sim':>13}")
    print("-" * 55)
    for l in range(n_layers):
        if l % 2 == 0 or l == n_layers - 1:
            gd = np.mean(gate_diff_norms[:, l])
            ud = np.mean(up_diff_norms[:, l])
            gc = np.mean(gate_dot_products[:, l])
            print(f"L{l:4d}  | {gd:14.6f} | {ud:12.6f} | {gc:13.6f}")
    
    return {
        "gate_rel_diff_mean": gate_diff_norms.mean(axis=0).tolist(),
        "up_rel_diff_mean": up_diff_norms.mean(axis=0).tolist(),
        "gate_cos_sim_mean": gate_dot_products.mean(axis=0).tolist(),
    }

def experiment_mlp_weight_svd(model):
    """
    实验3: MLP权重矩阵的奇异值结构
    分析W_gate和W_up的奇异值分布
    """
    print("\n=== 实验3: MLP权重奇异值结构 ===")
    
    n_layers = model.cfg.n_layers
    
    gate_sv_stats = []
    up_sv_stats = []
    
    for l in range(n_layers):
        W_gate = model.blocks[l].mlp.W_gate.float()
        W_in = model.blocks[l].mlp.W_in.float()
        
        # SVD (只取前100个奇异值,避免OOM)
        _, S_gate, _ = torch.linalg.svd(W_gate, full_matrices=False)
        _, S_in, _ = torch.linalg.svd(W_in, full_matrices=False)
        
        gate_sv_stats.append({
            "layer": l,
            "max": float(S_gate[0]),
            "min": float(S_gate[-1]),
            "mean": float(S_gate.mean()),
            "condition_number": float(S_gate[0] / max(S_gate[-1], 1e-10)),
            "top10": S_gate[:10].tolist(),
        })
        up_sv_stats.append({
            "layer": l,
            "max": float(S_in[0]),
            "min": float(S_in[-1]),
            "mean": float(S_in.mean()),
            "condition_number": float(S_in[0] / max(S_in[-1], 1e-10)),
            "top10": S_in[:10].tolist(),
        })
    
    # 打印摘要
    print(f"\n{'Layer':>6} | {'Gate_cond':>10} | {'W_in_cond':>9} | {'Gate_top1':>10} | {'W_in_top1':>9}")
    print("-" * 55)
    for l in range(n_layers):
        if l % 4 == 0 or l == n_layers - 1:
            gs = gate_sv_stats[l]
            us = up_sv_stats[l]
            print(f"L{l:4d}  | {gs['condition_number']:10.1f} | {us['condition_number']:9.1f} | {gs['max']:10.3f} | {us['max']:9.3f}")
    
    return {"gate_sv": gate_sv_stats, "up_sv": up_sv_stats}

def experiment_mlp_residual_increment(model):
    """
    实验4: MLP残差增量方向分析
    Δh_mlp(l) = mlp_out_sg(l) - mlp_out_pl(l) 的方向特性
    - 在W_U高/低奇异值方向上的投影
    - 与前一层的Δh(l-1)的角度关系
    """
    print("\n=== 实验4: MLP残差增量方向 ===")
    
    n_layers = model.cfg.n_layers
    
    # W_U SVD
    W_U = model.W_U.float()
    U_wu, S_wu, _ = torch.linalg.svd(W_U, full_matrices=False)
    U_high = U_wu[:, :100]
    U_low = U_wu[:, -100:]
    
    high_ratios = []
    low_ratios = []
    prev_angles = []  # 与前一层Δh的cosine similarity
    
    for sg, pl in SVA_PAIRS[:10]:
        with torch.no_grad():
            _, cache_sg = model.run_with_cache(sg)
            _, cache_pl = model.run_with_cache(pl)
        
        pair_high = []
        pair_low = []
        pair_angle = []
        
        prev_delta = None
        
        for l in range(n_layers):
            delta_mlp = (cache_sg["mlp_out", l][0, -1] - cache_pl["mlp_out", l][0, -1]).float()
            delta_norm = delta_mlp.norm().item()
            
            if delta_norm > 1e-8:
                # 在W_U方向上的投影
                proj_high = (delta_mlp.unsqueeze(0) @ U_high).norm().item()
                proj_low = (delta_mlp.unsqueeze(0) @ U_low).norm().item()
                pair_high.append(proj_high / delta_norm)
                pair_low.append(proj_low / delta_norm)
                
                # 与前一层Δh的角度
                if prev_delta is not None and prev_delta.norm() > 1e-8:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        delta_mlp.unsqueeze(0), prev_delta.unsqueeze(0)
                    ).item()
                    pair_angle.append(cos_sim)
                else:
                    pair_angle.append(0.0)
            else:
                pair_high.append(0.0)
                pair_low.append(0.0)
                pair_angle.append(0.0)
            
            # 当前层的总Δh
            delta_total = (cache_sg["resid_post", l][0, -1] - cache_pl["resid_post", l][0, -1]).float()
            prev_delta = delta_total
        
        high_ratios.append(pair_high)
        low_ratios.append(pair_low)
        prev_angles.append(pair_angle)
    
    high_ratios = np.array(high_ratios)
    low_ratios = np.array(low_ratios)
    prev_angles = np.array(prev_angles)
    
    print(f"\n{'Layer':>6} | {'High(%)':>8} | {'Low(%)':>8} | {'Angle_to_prev':>14}")
    print("-" * 45)
    for l in range(n_layers):
        if l % 2 == 0 or l == n_layers - 1:
            hr = np.mean(high_ratios[:, l]) * 100
            lr = np.mean(low_ratios[:, l]) * 100
            ang = np.mean(prev_angles[:, l])
            print(f"L{l:4d}  | {hr:7.2f}% | {lr:7.2f}% | {ang:14.6f}")
    
    return {
        "high_ratios_mean": high_ratios.mean(axis=0).tolist(),
        "low_ratios_mean": low_ratios.mean(axis=0).tolist(),
        "prev_angles_mean": prev_angles.mean(axis=0).tolist(),
    }

def main():
    print("=" * 70)
    print("Phase 219-2: MLP约束传播机制验证")
    print("=" * 70)
    print(f"执行时间: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"设备: {DEVICE}")
    
    print("\n加载Qwen2.5-1.5B模型...")
    model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", device=DEVICE)
    n_layers = model.cfg.n_layers
    print(f"模型: {n_layers}层, d_model={model.cfg.d_model}")
    
    # ===== 执行4个实验 =====
    ablation_results = experiment_mlp_single_ablation(model)
    glu_results = experiment_glu_gating_pattern(model)
    svd_results = experiment_mlp_weight_svd(model)
    increment_results = experiment_mlp_residual_increment(model)
    
    # ===== 综合分析 =====
    print("\n" + "=" * 70)
    print("综合分析")
    print("=" * 70)
    
    # 关键层
    print("\n关键层(MLP ablation减少KL最多的层):")
    sorted_layers = sorted(ablation_results.items(), key=lambda x: x[1]["kl_reduction_pct"], reverse=True)
    for l, data in sorted_layers[:5]:
        print(f"  L{l}: reduction={data['kl_reduction_pct']:.1f}%")
    
    # 门控 vs 上投影
    print("\n门控vs上投影的差异(最显著的层):")
    gate_diff = np.array(glu_results["gate_rel_diff_mean"])
    up_diff = np.array(glu_results["up_rel_diff_mean"])
    ratio = gate_diff / np.maximum(up_diff, 1e-10)
    top_layers = np.argsort(ratio)[-3:][::-1]
    for l in top_layers:
        print(f"  L{l}: gate/up ratio={ratio[l]:.3f}")
    
    # ===== 保存结果 =====
    output = {
        "experiment": "Phase219-2_MLP_Constraint_Mechanism",
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "model": "Qwen2.5-1.5B",
        "n_layers": n_layers,
        "ablation_results": {str(k): v for k, v in ablation_results.items()},
        "glu_results": glu_results,
        "svd_results": svd_results,
        "increment_results": increment_results,
        "summary": {
            "most_critical_mlp_layers": [l for l, _ in sorted_layers[:5]],
            "gate_up_ratio_top_layers": top_layers.tolist(),
        }
    }
    
    result_file = OUTPUT_DIR / "phase219_2_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到: {result_file}")

if __name__ == "__main__":
    main()
