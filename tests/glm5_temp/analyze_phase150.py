"""分析Phase 150 Exp1的深层含义 — 关键问题: Asymmetry的真正来源"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

with open('tests/glm5_temp/phase150_qwen3_20260513_1043.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# W_U分析
wu = data['W_U_analysis']
print("="*60)
print("W_U Rank Analysis (Critical for Exp1 Interpretation)")
print("="*60)
print(f"W_U shape: {wu['shape']}")
print(f"Effective rank (90%): {wu['rank_90']}")
print(f"Effective rank (95%): {wu['rank_95']}")
print(f"Effective rank (99%): {wu['rank_99']}")
print(f"Top-10 SV: {[f'{s:.2f}' for s in wu['top_10_sv']]}")
print(f"Tail-10 SV: {[f'{s:.2f}' for s in wu['sv_tail_10']]}")

# SV分布
sv_top = wu['top_10_sv']
sv_tail = wu['sv_tail_10']
print(f"\nSV ratio (max/min): {sv_top[0]/sv_tail[-1]:.1f}")
print(f"SV ratio (top1/top10): {sv_top[0]/sv_top[9]:.1f}")
print(f"Condition number (approx): {sv_top[0]/sv_tail[-1]:.1f}")

# Exp1: 条件转移矩阵
print("\n" + "="*60)
print("Exp 1: Conditional Transfer Matrix — Deep Analysis")
print("="*60)

transfer = data['exp1_conditional_transfer']

# 关键问题: Asymmetry的来源
# P(r→n) >> P(n→r) 可能有两个来源:
# 1. 主动路由: 系统主动将row-space分量推到null-space
# 2. 几何必然: row-space是低维(458维), null-space是高维(2102维)
#    随机旋转自然会将更多能量从低维→高维

# 检查: 如果只是随机旋转, P(r→n) 和 P(n→r) 应该满足什么关系?
# 在纯随机旋转下:
# P(r→n) = 1 - P(r→r), P(r→r) ≈ rank/d_model = 458/2560 ≈ 0.179
# P(n→r) ≈ rank/d_model ≈ 0.179 (因为null-space → row-space由维度比决定)

d_model = 2560
rank_95 = wu['rank_95']
null_dim = d_model - rank_95

print(f"\nGeometric prediction (random rotation):")
print(f"  rank/d_model = {rank_95}/{d_model} = {rank_95/d_model:.4f}")
print(f"  null_dim/d_model = {null_dim}/{d_model} = {null_dim/d_model:.4f}")
print(f"  Expected P(r→r) under random rotation: {rank_95/d_model:.4f}")
print(f"  Expected P(r→n) under random rotation: {null_dim/d_model:.4f}")
print(f"  Expected P(n→r) under random rotation: {rank_95/d_model:.4f}")
print(f"  Expected P(n→n) under random rotation: {null_dim/d_model:.4f}")
print(f"  Expected Asymmetry: {null_dim/d_model - rank_95/d_model:.4f}")

print(f"\nActual measurements (mid-layers L9-L27):")
for li_str, tm in transfer.items():
    li = int(li_str.replace('L', ''))
    if li < 6 or li > 30:
        continue
    print(f"  {li_str}: P(r→r)={tm['P_rr']:.4f}, P(r→n)={tm['P_rn']:.4f}, "
          f"P(n→r)={tm['P_nr']:.4f}, P(n→n)={tm['P_nn']:.4f}, "
          f"Asym={tm['asymmetry']:+.4f}")

# 关键判据:
# 如果Asymmetry ≈ (null_dim - rank)/d_model = (2102 - 458)/2560 ≈ 0.642
# 那就是几何必然性, 不是主动路由!
# 如果Asymmetry 显著 > 0.642, 那才是主动路由

geo_asymmetry = (null_dim - rank_95) / d_model
print(f"\n  Geometric Asymmetry = (null_dim - rank)/d = ({null_dim} - {rank_95})/{d_model} = {geo_asymmetry:.4f}")

# 等等, 这不对. Asymmetry = P(r→n) - P(n→r)
# 在随机旋转下:
# P(r→n) ≈ null_dim/d = 0.821 (row-space的能量有82.1%泄漏到null-space)
# P(n→r) ≈ rank/d = 0.179 (null-space的能量有17.9%泄漏到row-space)
# 所以几何Asymmetry = 0.821 - 0.179 = 0.642

# 但实际测量的mid-layer Asymmetry ≈ 0.59
# 这比几何预测 0.642 稍低!

print(f"\n  实际mid-layer平均Asymmetry ≈ 0.59")
print(f"  几何预测Asymmetry = {geo_asymmetry:.4f}")
print(f"  差异 = {0.59 - geo_asymmetry:.4f}")

if 0.59 < geo_asymmetry:
    print(f"\n  → 实际Asymmetry LESS than geometric prediction!")
    print(f"  → System is actually BETTER at preserving row-space than random rotation!")
    print(f"  → This is AGAINST active routing to null-space!")
    print(f"  → But also AGAINST symmetric mixing!")
    print(f"  → The system seems to actively RETAIN some row-space energy!")
else:
    print(f"\n  → Actual Asymmetry exceeds geometric prediction")
    print(f"  → Possible evidence for active routing to null-space")

# 更精确分析: 早期层 (L3) 的P(r→r)=0.815 — 这远高于随机旋转的0.179!
print(f"\n  Early layer L3: P(r→r)=0.815 >> geometric 0.179")
print(f"  → Strong row-space preservation in early layers!")
print(f"  → This IS evidence for active structure!")

# 检查有效rank对结果的影响
print(f"\n" + "="*60)
print(f"Impact of Rank Definition")
print(f"="*60)
for rank_label, rank_val in [("90%", wu['rank_90']), ("95%", wu['rank_95']), ("99%", wu['rank_99'])]:
    geo_rn = 1 - rank_val/d_model
    geo_nr = rank_val/d_model
    geo_asym = geo_rn - geo_nr
    print(f"  rank={rank_label}({rank_val}): geo P(r→n)={geo_rn:.4f}, geo P(n→r)={geo_nr:.4f}, "
          f"geo Asym={geo_asym:+.4f}")

print(f"\n  关键: 无论rank定义如何, 几何Asymmetry都是正的!")
print(f"  这意味着 P(r→n) > P(n→r) 在纯随机旋转下就是必然的!")
print(f"  所以Asymmetry>0本身不是主动路由证据!")
print(f"  关键是Asymmetry是否超过几何预测值!")

# 更深层分析: 使用原始Phase 148的top-200定义
print(f"\n  Phase 148 used top-200 components:")
rank_200 = 200
geo_rn_200 = 1 - 200/d_model
geo_nr_200 = 200/d_model
geo_asym_200 = geo_rn_200 - geo_nr_200
print(f"  rank=200: geo P(r→n)={geo_rn_200:.4f}, geo P(n→r)={geo_nr_200:.4f}, "
      f"geo Asym={geo_asym_200:+.4f}")
print(f"  实际Phase 148 null_ratio ≈ 0.92")
print(f"  几何预测 P(n→n) = {1-200/d_model:.4f}")
print(f"  差异 = {0.92 - (1-200/d_model):.4f}")
