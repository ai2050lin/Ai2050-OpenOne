"""
Phase 219-1: 中间层约束传播路径追踪

P0实验的核心发现:
  - W_U各向同性化后, 约束传播KL保留80% → 约束传播不依赖W_U
  - 但Top-1预测90%改变 → W_U只负责"token选择锐化"
  - 真正的约束传播在中间层完成

本实验目标:
  1. 追踪约束"数"(sg/pl)从L0到L27的传播路径
  2. 识别哪些attention head参与了约束传播
  3. 识别MLP层的贡献
  4. 验证约束传播是否在中间层完成

方法:
  - 对比sg/pl句的residual stream差异 Δh(l) = h_sg(l) - h_pl(l)
  - 将Δh投影到attention和MLP的贡献上
  - 测量每层、每个head对约束信号的贡献

执行时间: 2026-05-17 20:40
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
    ("The king rules", "The kings rule"),
    ("The horse gallops", "The horses gallop"),
    ("The student studies", "The students study"),
    ("The lamp shines", "The lamps shine"),
    ("The clock ticks", "The clocks tick"),
    ("The river flows", "The rivers flow"),
    ("The wind blows", "The winds blow"),
    ("The star glows", "The stars glow"),
    ("The bell rings", "The bells ring"),
    ("The fire burns", "The fires burn"),
]

def compute_kl(p, q, eps=1e-10):
    """对称KL散度"""
    p = p.float() + eps
    q = q.float() + eps
    p = p / p.sum()
    q = q / q.sum()
    return (0.5 * (p * (p/q).log()).sum() + 0.5 * (q * (q/p).log()).sum()).item()

def analyze_constraint_propagation(model):
    """
    分析约束在每层的传播:
    - Δh(l) = h_sg(l) - h_pl(l) 的范数
    - Δh(l)在W_U上的KL贡献
    - attention vs MLP的贡献分解
    """
    print("\n=== 约束传播路径分析 ===")
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head
    
    # 收集所有pair的层间差异
    delta_h_norms = []  # [n_pairs, n_layers]
    kl_by_layer = []    # [n_pairs, n_layers]
    
    # Attention head贡献
    attn_contributions = []  # [n_pairs, n_layers, n_heads]
    
    # MLP贡献
    mlp_contributions = []  # [n_pairs, n_layers]
    
    for sg, pl in SVA_PAIRS:
        with torch.no_grad():
            _, cache_sg = model.run_with_cache(sg)
            _, cache_pl = model.run_with_cache(pl)
        
        pair_delta_norms = []
        pair_kl = []
        pair_attn = []
        pair_mlp = []
        
        for l in range(n_layers):
            # Residual stream差异
            h_sg = cache_sg["resid_post", l][0, -1]  # [d_model]
            h_pl = cache_pl["resid_post", l][0, -1]  # [d_model]
            delta_h = h_sg - h_pl
            pair_delta_norms.append(delta_h.norm().item())
            
            # KL: 用W_U投影
            logits_sg = h_sg.float() @ model.W_U.float()
            logits_pl = h_pl.float() @ model.W_U.float()
            probs_sg = torch.softmax(logits_sg, dim=-1)
            probs_pl = torch.softmax(logits_pl, dim=-1)
            kl = compute_kl(probs_sg, probs_pl)
            pair_kl.append(kl)
            
            # Attention贡献: 每个head
            # attn_out = sum over heads of (head_output)
            # 每个head的贡献 = W_O @ head_pattern @ V
            # 我们用hook的结果
            attn_out_sg = cache_sg["attn_out", l][0, -1]  # [d_model]
            attn_out_pl = cache_pl["attn_out", l][0, -1]  # [d_model]
            delta_attn = attn_out_sg - attn_out_pl
            pair_attn.append(delta_attn.norm().item())
            
            # MLP贡献
            mlp_out_sg = cache_sg["mlp_out", l][0, -1]  # [d_model]
            mlp_out_pl = cache_pl["mlp_out", l][0, -1]  # [d_model]
            delta_mlp = mlp_out_sg - mlp_out_pl
            pair_mlp.append(delta_mlp.norm().item())
        
        delta_h_norms.append(pair_delta_norms)
        kl_by_layer.append(pair_kl)
        attn_contributions.append(pair_attn)
        mlp_contributions.append(pair_mlp)
    
    # 转为numpy便于统计
    delta_h_norms = np.array(delta_h_norms)  # [n_pairs, n_layers]
    kl_by_layer = np.array(kl_by_layer)
    attn_contributions = np.array(attn_contributions)
    mlp_contributions = np.array(mlp_contributions)
    
    return delta_h_norms, kl_by_layer, attn_contributions, mlp_contributions

def analyze_head_level_contributions(model):
    """
    分析每个attention head对约束传播的贡献
    """
    print("\n=== Head级别贡献分析 ===")
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    # 每个head的贡献: 通过W_O投影后的输出差异
    head_contributions = []  # [n_pairs, n_layers, n_heads]
    
    for sg, pl in SVA_PAIRS[:10]:  # 取前10对减少计算
        with torch.no_grad():
            _, cache_sg = model.run_with_cache(sg)
            _, cache_pl = model.run_with_cache(pl)
        
        pair_head = []
        
        for l in range(n_layers):
            # 获取每个head的输出
            # cache["result", l] 形状: [batch, seq, n_heads, d_head]
            # 需要通过W_O投影
            W_O = model.blocks[l].attn.W_O  # [n_heads, d_head, d_model]
            
            head_diffs = []
            for h in range(n_heads):
                # result: [batch, seq_pos, n_heads, d_head]
                z_sg = cache_sg["z", l][0, -1, h]  # [d_head]
                z_pl = cache_pl["z", l][0, -1, h]  # [d_head]
                
                # 通过W_O投影
                out_sg = z_sg.float() @ W_O[h].float()  # [d_model]
                out_pl = z_pl.float() @ W_O[h].float()  # [d_model]
                
                delta = (out_sg - out_pl).norm().item()
                head_diffs.append(delta)
            
            pair_head.append(head_diffs)
        
        head_contributions.append(pair_head)
    
    head_contributions = np.array(head_contributions)  # [n_pairs, n_layers, n_heads]
    return head_contributions

def analyze_constraint_direction(model):
    """
    分析约束信号Δh(l)的方向: 
    - 在W_U的高/低奇异值方向上的投影
    - 在各层的子空间中的分布
    """
    print("\n=== 约束方向分析 ===")
    
    n_layers = model.cfg.n_layers
    
    # W_U的SVD
    W_U = model.W_U.float()
    U, S, Vh = torch.linalg.svd(W_U, full_matrices=False)
    
    # 定义高/低奇异值方向
    n_high = 100  # 前100个方向
    n_low = 100   # 后100个方向
    
    U_high = U[:, :n_high]  # [d_model, n_high]
    U_low = U[:, -n_low:]   # [d_model, n_low]
    
    high_ratios = []
    low_ratios = []
    total_norms = []
    
    for sg, pl in SVA_PAIRS[:10]:
        with torch.no_grad():
            _, cache_sg = model.run_with_cache(sg)
            _, cache_pl = model.run_with_cache(pl)
        
        pair_high = []
        pair_low = []
        pair_total = []
        
        for l in range(n_layers):
            h_sg = cache_sg["resid_post", l][0, -1]
            h_pl = cache_pl["resid_post", l][0, -1]
            delta = h_sg.float() - h_pl.float()
            
            total_norm = delta.norm().item()
            
            # 在高奇异值方向上的投影
            proj_high = (delta.unsqueeze(0) @ U_high).norm().item()
            # 在低奇异值方向上的投影
            proj_low = (delta.unsqueeze(0) @ U_low).norm().item()
            
            pair_high.append(proj_high / max(total_norm, 1e-10))
            pair_low.append(proj_low / max(total_norm, 1e-10))
            pair_total.append(total_norm)
        
        high_ratios.append(pair_high)
        low_ratios.append(pair_low)
        total_norms.append(pair_total)
    
    high_ratios = np.array(high_ratios)
    low_ratios = np.array(low_ratios)
    total_norms = np.array(total_norms)
    
    return high_ratios, low_ratios, total_norms

def analyze_layer_ablation_effect(model):
    """
    关键实验: 逐层ablation对约束传播的影响
    - 在第l层, 把sg和pl的residual stream替换为它们的均值
    - 测量后续层的KL变化
    - 这能揭示约束传播的"因果路径"
    """
    print("\n=== 逐层Ablation因果实验 ===")
    
    n_layers = model.cfg.n_layers
    
    # 基线: 无ablation的KL
    baseline_kl = []
    for sg, pl in SVA_PAIRS[:10]:
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
    
    baseline_kl = np.array(baseline_kl)  # [n_pairs, n_layers]
    
    # Ablation: 在第ablate_layer层混合sg和pl
    ablation_effects = {}  # ablate_layer -> [n_pairs, n_layers]
    
    for ablate_layer in range(0, n_layers, 4):  # 每4层做一次
        print(f"  Ablation层 {ablate_layer}...")
        effects = []
        
        for sg, pl in SVA_PAIRS[:10]:
            # 在ablate_layer层混合: h_mixed = (h_sg + h_pl) / 2
            def ablation_hook(value, hook):
                # value: [batch, seq, d_model]
                # 混合最后一个位置的residual
                value[:, -1, :] = (cache_sg_current["resid_post", ablate_layer][0, -1] + 
                                   cache_pl_current["resid_post", ablate_layer][0, -1]) / 2
                return value
            
            # 这需要更复杂的方法, 用hook实现
            # 简化: 直接计算
            with torch.no_grad():
                _, cache_sg_current = model.run_with_cache(sg)
                _, cache_pl_current = model.run_with_cache(pl)
            
            # 在ablate_layer层之后, 用混合值替代
            # 简化方法: 直接从混合h开始计算后续层
            h_mixed = (cache_sg_current["resid_post", ablate_layer][0, -1] + 
                       cache_pl_current["resid_post", ablate_layer][0, -1]) / 2
            
            # 计算混合h的logits
            logits_mixed = h_mixed.float() @ model.W_U.float()
            probs_mixed = torch.softmax(logits_mixed, dim=-1)
            
            # 计算从混合h开始的KL (相对于原始sg和pl)
            logits_sg = cache_sg_current["resid_post", ablate_layer][0, -1].float() @ model.W_U.float()
            probs_sg = torch.softmax(logits_sg, dim=-1)
            
            kl_after_ablation = compute_kl(probs_sg, probs_mixed)
            effects.append(kl_after_ablation)
        
        ablation_effects[ablate_layer] = np.mean(effects)
    
    return baseline_kl, ablation_effects

def main():
    print("=" * 70)
    print("Phase 219-1: 中间层约束传播路径追踪")
    print("=" * 70)
    print(f"执行时间: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"设备: {DEVICE}")
    
    # 加载模型
    print("\n加载Qwen2.5-1.5B模型...")
    model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", device=DEVICE)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"模型: {n_layers}层, {n_heads}heads, d_model={model.cfg.d_model}")
    
    # ===== 实验1: 约束传播路径分析 =====
    delta_h_norms, kl_by_layer, attn_contrib, mlp_contrib = analyze_constraint_propagation(model)
    
    print("\n--- Δh范数和KL按层分布 ---")
    print(f"{'Layer':>6} | {'Δh_norm':>10} | {'KL':>10} | {'Attn_Δ':>10} | {'MLP_Δ':>10} | {'Attn%':>6} | {'MLP%':>6}")
    print("-" * 75)
    for l in range(n_layers):
        if l % 2 == 0 or l == n_layers - 1:
            dh = np.mean(delta_h_norms[:, l])
            kl = np.mean(kl_by_layer[:, l])
            attn = np.mean(attn_contrib[:, l])
            mlp = np.mean(mlp_contrib[:, l])
            total = attn + mlp
            attn_pct = attn / max(total, 1e-10) * 100
            mlp_pct = mlp / max(total, 1e-10) * 100
            print(f"L{l:4d}  | {dh:10.4f} | {kl:10.6f} | {attn:10.4f} | {mlp:10.4f} | {attn_pct:5.1f}% | {mlp_pct:5.1f}%")
    
    # ===== 实验2: Head级别贡献 =====
    head_contrib = analyze_head_level_contributions(model)
    
    print("\n--- 每层最活跃的head (对约束传播贡献最大) ---")
    head_contrib_mean = head_contrib.mean(axis=0)  # [n_layers, n_heads]
    for l in range(n_layers):
        if l % 4 == 0 or l == n_layers - 1:
            top_heads = np.argsort(head_contrib_mean[l])[-3:][::-1]
            top_vals = head_contrib_mean[l][top_heads]
            print(f"  L{l:2d}: top heads = {list(zip(top_heads.tolist(), [f'{v:.4f}' for v in top_vals]))}")
    
    # ===== 实验3: 约束方向分析 =====
    high_ratios, low_ratios, total_norms = analyze_constraint_direction(model)
    
    print("\n--- Δh在W_U高/低奇异值方向上的投影比例 ---")
    print(f"{'Layer':>6} | {'High(%)':>8} | {'Low(%)':>8} | {'Δh_norm':>10}")
    print("-" * 45)
    for l in range(n_layers):
        if l % 2 == 0 or l == n_layers - 1:
            hr = np.mean(high_ratios[:, l]) * 100
            lr = np.mean(low_ratios[:, l]) * 100
            tn = np.mean(total_norms[:, l])
            print(f"L{l:4d}  | {hr:7.2f}% | {lr:7.2f}% | {tn:10.4f}")
    
    # ===== 实验4: 逐层Ablation =====
    baseline_kl, ablation_effects = analyze_layer_ablation_effect(model)
    
    print("\n--- 逐层Ablation效果 (混合sg/pl后的KL) ---")
    print(f"{'Ablate_Layer':>12} | {'KL_after_ablation':>18}")
    print("-" * 35)
    for layer, kl in sorted(ablation_effects.items()):
        print(f"L{layer:10d}  | {kl:18.6f}")
    
    # ===== 综合分析 =====
    print("\n" + "=" * 70)
    print("综合分析")
    print("=" * 70)
    
    # 1. 约束传播的关键层
    kl_mean = kl_by_layer.mean(axis=0)
    kl_growth = np.diff(kl_mean)
    peak_growth_layer = np.argmax(kl_growth[:21])
    print(f"\n1. KL增长最快的层: L{peak_growth_layer} (ΔKL={kl_growth[peak_growth_layer]:.4f})")
    
    # 2. Attention vs MLP的相对贡献
    attn_total = attn_contrib.mean(axis=0)
    mlp_total = mlp_contrib.mean(axis=0)
    attn_dominant_layers = np.where(attn_total > mlp_total)[0]
    mlp_dominant_layers = np.where(mlp_total >= attn_total)[0]
    print(f"2. Attention主导的层: {attn_dominant_layers.tolist()}")
    print(f"   MLP主导的层: {mlp_dominant_layers.tolist()}")
    
    # 3. 约束信号方向
    high_ratio_mean = high_ratios.mean(axis=0)
    low_ratio_mean = low_ratios.mean(axis=0)
    print(f"3. 约束信号在W_U高奇异值方向的比例 (L0-L20):")
    print(f"   均值: {high_ratio_mean[:21].mean()*100:.1f}%")
    print(f"   低奇异值方向: {low_ratio_mean[:21].mean()*100:.1f}%")
    
    # ===== 保存结果 =====
    output = {
        "experiment": "Phase219-1_Constraint_Propagation_Path",
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "model": "Qwen2.5-1.5B",
        "n_layers": n_layers,
        "n_heads": n_heads,
        "summary": {
            "peak_kl_growth_layer": int(peak_growth_layer),
            "peak_kl_growth": float(kl_growth[peak_growth_layer]),
            "attn_dominant_layers": attn_dominant_layers.tolist(),
            "mlp_dominant_layers": mlp_dominant_layers.tolist(),
            "high_sv_ratio_mean": float(high_ratio_mean[:21].mean()),
            "low_sv_ratio_mean": float(low_ratio_mean[:21].mean()),
        },
        "delta_h_norms_mean": delta_h_norms.mean(axis=0).tolist(),
        "kl_by_layer_mean": kl_mean.tolist(),
        "attn_contrib_mean": attn_total.tolist(),
        "mlp_contrib_mean": mlp_total.tolist(),
        "high_ratios_mean": high_ratio_mean.tolist(),
        "low_ratios_mean": low_ratio_mean.tolist(),
        "head_contributions_mean": head_contrib_mean.tolist(),
        "ablation_effects": {str(k): float(v) for k, v in ablation_effects.items()},
    }
    
    result_file = OUTPUT_DIR / "phase219_1_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到: {result_file}")
    
    return output

if __name__ == "__main__":
    results = main()
