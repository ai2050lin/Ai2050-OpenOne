"""
Phase 144 综合分析: 约束传播系统的数学结构
==========================================
汇总Qwen3和GLM4的Phase 144+144b结果，进行跨模型交叉验证和理论分析
"""

import json
import numpy as np
from pathlib import Path

TEMP_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")

# 加载数据
with open(TEMP_DIR / "phase144b_qwen3_expa_enhanced.json") as f:
    qwen3_expa = json.load(f)
with open(TEMP_DIR / "phase144b_glm4_expa_enhanced.json") as f:
    glm4_expa = json.load(f)
with open(TEMP_DIR / "phase144_glm4_constraint_20260512_34.json") as f:
    glm4_full = json.load(f)

print("=" * 70)
print("Phase 144 综合分析: 约束传播系统")
print("=" * 70)

# ===== 1. Exp A: 跨领域Jacobian一致性 =====
print("\n" + "=" * 50)
print("1. Exp A: 跨领域Jacobian一致性 (大数据量, n=10/category)")
print("=" * 50)

categories = ["NOT", "TENSE", "SYN", "SCOPE", "CROSS", "NONSENSE"]
print(f"\n{'类别':12s} {'Qwen3 cos':>12s} {'GLM4 cos':>12s} {'平均':>8s} {'趋势':>6s}")
print("-" * 55)
for cat in categories:
    q3 = qwen3_expa["by_category"].get(cat, {})
    g4 = glm4_expa["by_category"].get(cat, {})
    q3m = q3.get("mean_cos", 0)
    g4m = g4.get("mean_cos", 0)
    avg = (q3m + g4m) / 2
    print(f"  {cat:10s} {q3m:10.4f}±{q3.get('std_cos',0):.3f} {g4m:8.4f}±{g4.get('std_cos',0):.3f} {avg:8.4f}")

# 关键分析: cos随语义距离单调递减
print("\n关键发现:")
print("  1. cos随语义距离单调递减: NOT/TENSE(0.77-0.87) → SYN/SCOPE(0.73-0.79) → CROSS/NONSENSE(0.29-0.34)")
print("  2. 既不是全局光滑(cos≈1), 也不是完全分段(cos≈0)")
print("  3. 支持的是: 局部光滑的分层结构 — 语义近邻处Jacobian一致,跨领域时急剧下降")

# ===== 2. Exp B: 约束违背动力学 =====
print("\n" + "=" * 50)
print("2. Exp B: 约束违背动力学")
print("=" * 50)

if "exp_b" in glm4_full:
    print("\nGLM4 按类别约束违背信号:")
    for cat, stats in glm4_full["exp_b"]["by_category"].items():
        print(f"  {cat:12s}: amp={stats['mean_amplification']:.1f}x, "
              f"first={stats['mean_first_delta']:.4f}, last={stats['mean_last_delta']:.2f}")

print("\n关键发现:")
print("  1. 所有约束违背信号被放大30-770倍")
print("  2. retention=1.0: 信号从未衰减(与Phase 143的各向同性传播一致)")
print("  3. SEMANTIC类的末层delta最大(347 vs 173-282 for others)")

# ===== 3. Exp C: MLP约束修正 =====
print("\n" + "=" * 50)
print("3. Exp C: MLP约束修正效应")
print("=" * 50)

if "exp_c" in glm4_full:
    print("\nGLM4 按类别MLP修正:")
    for cat, stats in glm4_full["exp_c"]["by_category"].items():
        print(f"  {cat:12s}: alignment={stats['mean_alignment']:.4f}, strength={stats['mean_correction_strength']:.4f}")

print("\n关键发现:")
print("  1. MLP修正与约束信号的对齐度alignment≈0.42-0.53 (中等, 不是强对齐)")
print("  2. MLP修正强度strength≈0.48-0.55 (约为约束信号的50%)")
print("  3. MLP不是'强约束修正器' — 修正效应中等,不是主导因素")

# ===== 4. Exp D: Attention聚类 =====
print("\n" + "=" * 50)
print("4. Exp D: Attention Head功能聚类")
print("=" * 50)

if "exp_d" in glm4_full and "cluster_analysis" in glm4_full["exp_d"]:
    print("\nGLM4 Attention聚类:")
    for cid, info in glm4_full["exp_d"]["cluster_analysis"].items():
        print(f"  Cluster {cid}: {info['n_heads']} heads, "
              f"layer {info['min_layer']}-{info['max_layer']} (mean={info['mean_layer']:.1f})")
    
    # 检查是否有层特异性cluster
    for cid, info in glm4_full["exp_d"]["cluster_analysis"].items():
        if info["max_layer"] - info["min_layer"] < 10:
            print(f"  >>> Cluster {cid} 是层特异性的! (layer {info['min_layer']}-{info['max_layer']})")

print("\n关键发现:")
print("  1. 大多数cluster跨所有层(没有强功能分化)")
print("  2. GLM4有一个高层特异cluster: Cluster 2只有8个heads在L35-39")
print("  3. 注意力头的功能分化目前证据不足")

# ===== 5. 综合理论分析 =====
print("\n" + "=" * 50)
print("5. 综合理论分析")
print("=" * 50)

print("""
=== 对用户理论框架的逐一分析 ===

用户论点                           | Phase 144结论               | 评价
----------------------------------|----------------------------|------
1. "PR低!=流形"                    | 仍然正确,但局部线性支持流形 | [Y] 仍正确
2. "分段低秩动力系统"             | cos随语义距离递减,不是二值  | [!] 需要修正
3. "Jacobian不是几何主对象"       | Jacobian确实包含所有信息    | [N] 不完全正确
4. "Attention是约束路由"          | 功能分化证据不足            | [!] 待验证
5. "MLP是约束修正器"              | 修正效应中等(alignment~0.5) | [!] 部分正确
6. "约束传播系统"                 | 整体框架正确但需精确化      | [Y] 方向正确
7. "不存在全局光滑流形"           | 确认:跨领域cos仅0.29-0.34  | [Y] 确认
8. "局部可能有流形结构"           | 确认:语义近邻cos~0.77-0.87 | [Y] 确认

=== 最重要发现: 局部光滑的分层结构 ===

Phase 144核心发现:

  cos(h1, h2) = f(semantic_distance(h1, h2))

  其中 f 是单调递减函数:
  - semantic_distance ≈ 0 (同一句): cos ≈ 1 (Phase 143b)
  - semantic_distance 小 (NOT/TENSE): cos ≈ 0.77-0.87
  - semantic_distance 中 (SYN/SCOPE): cos ≈ 0.73-0.79
  - semantic_distance 大 (CROSS): cos ≈ 0.29-0.34
  - semantic_distance 极大 (NONSENSE): cos ≈ 0.29

这意味着:
  1. 不存在全局光滑流形(跨领域cos太低)
  2. 但存在局部光滑的"patch"(语义近邻内cos高)
  3. patch之间有"边界"(cos急剧下降)
  4. 边界位置与语义距离对齐

更精确的数学对象:

  Transformer hidden state space = ∪_{α} M_α

  其中 M_α 是局部光滑patch, 定义为:
  M_α = {h : cos(J(h)·v, J(h_α)·v) > θ, ∀‖v‖=1}

  θ ≈ 0.7-0.8 时, M_α 大约覆盖一个"语义领域"
  θ ≈ 0.3 时, M_α 覆盖多个"语义领域"

=== 对"约束传播系统"的修正 ===

用户的框架: "约束传播系统"
Phase 144数据支持的修正框架:

  "局部光滑约束传播系统" (Locally Smooth Constraint Propagation System)

  特征:
  1. 局部光滑: 在语义近邻内, Jacobian近似常数(cos≈0.8)
  2. 全局分段: 跨语义领域时, Jacobian急剧变化(cos≈0.3)
  3. 约束传播: MLP提供中等修正(alignment≈0.5), 不是强修正
  4. 分层结构: patch的大小和位置由语义距离决定

=== 硬伤和问题 ===

1. Qwen3和GLM4的cos绝对值有差异:
   - GLM4整体cos更高(NOT: 0.83 vs 0.72, CROSS: 0.44 vs 0.24)
   - 这可能反映了模型规模(d_model 4096 vs 2560)的差异
   - 更大的模型可能有更光滑的内部表示?

2. ε=2.0的选择影响:
   - 我们只在ε=2.0下测了跨领域一致性
   - 更小的ε可能给出不同的结果
   - 但Phase 143b已证明ε<0.05时是噪声主导

3. "语义距离"需要精确定义:
   - 目前用的分类(NOT/TENSE/SYN/...)是直觉性的
   - 需要一个定量的语义距离度量
   - 可能的度量: token embedding距离, 困惑度差异等

4. MLP修正alignment≈0.5的含义:
   - 0.5是"随机对齐"还是"中等对齐"?
   - 如果随机: MLP不是约束修正器
   - 如果中等: MLP有部分修正功能
   - 需要baseline: 随机方向的alignment是多少?

5. Attention聚类方法太粗糙:
   - 只用了4维特征(mean, max, entropy, self-attn)
   - K-means假设球形cluster
   - 需要更精细的head功能分析
""")

print("\n分析完成!")
