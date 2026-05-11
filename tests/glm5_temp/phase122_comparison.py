"""
临时脚本: 对比 True Jacobian 谱 vs 协方差输运谱 (Phase 121)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import json
import numpy as np

# Load Phase 122 True Jacobian results
with open('tests/glm5_temp/phase122_exp1_qwen3_true_jacobian.json', 'r') as f:
    jacobian_data = json.load(f)

# Load Phase 121 covariance transport results
with open('tests/glm5_temp/phase121_exp3_qwen3_jacobian.json', 'r') as f:
    cov_data = json.load(f)

print("=" * 70)
print("Phase 122 vs Phase 121: True Jacobian vs Covariance Transport")
print("=" * 70)

# Compare key layers
summary = jacobian_data['summary']

print("\n--- True Local Jacobian (Phase 122) ---")
print(f"{'Layer':<8} {'σ_max':<10} {'σ_min':<10} {'κ':<10} {'n_expand':<10} {'n_contract':<10}")
print("-" * 58)
for l_str, data in sorted(summary.items(), key=lambda x: int(x[0])):
    sv = data['mean_singular_values_top50']
    print(f"L{l_str:<7} {sv[0]:<10.4f} {sv[-1]:<10.6f} {data['condition_number']:<10.1f} "
          f"{data['mean_n_expanding']:<10.0f} {data['mean_n_contracting']:<10.0f}")

print("\n--- Covariance Transport (Phase 121) ---")
print(f"{'Layer':<8} {'max_eig':<12} {'min_eig':<12} {'median_eig':<10} {'n_expand':<10} {'n_contract':<10}")
print("-" * 62)
for l_str, data in sorted(cov_data.items(), key=lambda x: int(x[0])):
    print(f"L{l_str:<7} {data['max_eigenvalue']:<12.2f} {data['min_eigenvalue']:<12.2e} "
          f"{data['median_eigenvalue']:<10.4f} {data['n_expanding']:<10} {data['n_contracting']:<10}")

print("\n--- KEY COMPARISON ---")
print("Phase 121 claimed: 'bimodal spectrum with extreme expansion/contraction'")
print("Phase 122 shows:  'mild spectrum with σ ∈ [0.8, 4.5], κ ≈ 3'")
print()
print("Ratio of Phase 121 max_eig / Phase 122 σ_max:")
for l_str in sorted(summary.keys(), key=int):
    l_int = int(l_str)
    if str(l_int) in cov_data:
        p121_max = cov_data[str(l_int)]['max_eigenvalue']
        p122_max = summary[l_str]['mean_singular_values_top50'][0]
        ratio = p121_max / p122_max
        print(f"  L{l_str}: Phase121={p121_max:.1f}, Phase122={p122_max:.4f}, ratio={ratio:.0f}x")

print("\n--- INTERPRETATION ---")
print("Phase 121的'极端特征值'是协方差输运比的伪影：")
print("  广义特征值 λ = (v^T Σ_{l+1} v) / (v^T Σ_l v)")
print("  当v是高方差方向时，分子和分母都很大，但比值可以极端")
print("  真正Jacobian的σ是逐点微分的局部性质，温和得多")
print()
print("关键结论：")
print("  1. 真正Jacobian的谱是温和的(σ∈[0.8,4.5])，不是极端的")
print("  2. Phase 121的'bimodal谱'来自协方差输运，不是局部动力学")
print("  3. L12是最'收缩'的层(n_contract=61)，但σ_min仍≈0.8")
print("  4. L24-L30主要是扩张的(n_expand=85-93)")

# 轨迹恢复分析
print("\n" + "=" * 70)
print("Trajectory Recovery Analysis")
print("=" * 70)

with open('tests/glm5_temp/phase122_exp2_qwen3_trajectory_recovery.json', 'r') as f:
    traj_data = json.load(f)

summary_traj = traj_data['summary']

print("\n--- Final Layer Distance Ratio (perturbation amplification) ---")
print(f"{'Inject':<8} {'spike':<15} {'comp':<15} {'random':<15}")
print("-" * 53)
for inject_l in [0, 6, 12, 18, 24, 30]:
    s_key = f"L{inject_l}_spike_5"
    c_key = f"L{inject_l}_comp_5"
    r_key = f"L{inject_l}_random_5"
    
    s_ratio = summary_traj.get(s_key, {}).get('final_layer_mean_ratio', float('nan'))
    c_ratio = summary_traj.get(c_key, {}).get('final_layer_mean_ratio', float('nan'))
    r_ratio = summary_traj.get(r_key, {}).get('final_layer_mean_ratio', float('nan'))
    
    print(f"L{inject_l:<7} {s_ratio:<15.3f} {c_ratio:<15.3f} {r_ratio:<15.3f}")

print("\n关键发现：")
print("  1. 恢复率 = 0% (所有方向，所有层)")
print("  2. 扰动被放大，不是恢复")
print("  3. spike/complement/random方向无显著差异")
print("  4. 这是否定'吸引子流形'假说的强证据")

# 敏感性场分析
print("\n" + "=" * 70)
print("Sensitivity Field Analysis")
print("=" * 70)

with open('tests/glm5_temp/phase122_exp3_qwen3_sensitivity_field.json', 'r') as f:
    sens_data = json.load(f)

summary_sens = sens_data['summary']

print("\n--- Mean KL per direction type (across layers) ---")
for layer in [6, 12, 18, 24]:
    spike_key = f"L{layer}_spike_5"
    comp_key = f"L{layer}_comp_5"
    rand_key = f"L{layer}_random_5"
    
    s_kl = summary_sens.get(spike_key, {}).get('mean_kl', float('nan'))
    c_kl = summary_sens.get(comp_key, {}).get('mean_kl', float('nan'))
    r_kl = summary_sens.get(rand_key, {}).get('mean_kl', float('nan'))
    
    print(f"  L{layer}: spike_KL={s_kl:.6f}, comp_KL={c_kl:.6f}, rand_KL={r_kl:.6f}, "
          f"ratio spike/comp={s_kl/c_kl:.3f}")

print("\n关键发现：")
print("  1. 所有方向的KL值几乎相同(~0.001-0.002)")
print("  2. 敏感性场近似各向同性")
print("  3. Causal Effect ≈ Energy × (≈常数)")
print("  4. 这解释了Phase 121的'能量决定因果效应'")

print("\n" + "=" * 70)
print("PHASE 122 综合结论")
print("=" * 70)
print("""
1. 真正Jacobian的谱是温和的 (σ∈[0.8, 4.5], κ≈3)
   → Phase 121的'极端bimodal谱'是协方差输运伪影

2. 轨迹恢复率 = 0%
   → 否定'吸引子流形'假说
   → 扰动被放大，不是恢复

3. 敏感性场近似各向同性
   → Causal Effect = Energy × Sensitivity, Sensitivity≈常数
   → 因果效应的各向异性完全来自能量，不是动力学敏感性

4. 修正后的理论框架：
   Transformer不存在显式"语义码"，也不是"约束系统"或"吸引子流形"。
   它是一个温和的放大器(mild amplifier)：
   - 权重空间各向同性
   - 输入诱导各向异性能量结构
   - 层间温和放大/收缩(σ≈1-4)
   - 残差连接保持累积效应
   - 敏感性均匀分布

   语言能力来自：各向异性的输入能量 × 温和的层间放大 × 残差累积
   而不是：显式编码、吸引子动力学、或约束系统
""")
