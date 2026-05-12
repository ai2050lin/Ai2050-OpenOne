"""Phase 140 Qwen3结果分析"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

fn = 'tests/glm5_temp/phase140_qwen3_operator_mechanics_20260512_1510.json'
with open(fn, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Model: {data['model_name']}")
print(f"Info: {data['model_info']}")

# === Exp A Part 1: 算子敏感性 ===
print("\n" + "="*60)
print("Exp A Part 1: 算子敏感性分析")
print("="*60)

op_sens = data['expA']['operator_sensitivity']
for op_name in ['NOT', 'PAST', 'PLURAL', 'FUTURE', 'MODAL']:
    if op_name not in op_sens:
        continue
    print(f"\n  {op_name}:")
    # 找到峰值层
    peak_layer = None
    peak_val = 0
    for layer_key in sorted(op_sens[op_name].keys(), key=lambda x: int(x[1:])):
        val = op_sens[op_name][layer_key]['mean_rel_change']
        li = int(layer_key[1:])
        if li % 6 == 0 or li <= 2 or li >= 34:
            print(f"    {layer_key}: rel_change={val:.6f} (std={op_sens[op_name][layer_key]['std_rel_change']:.6f})")
        if val > peak_val:
            peak_val = val
            peak_layer = layer_key
    print(f"    ** 峰值层: {peak_layer} ({peak_val:.6f})")

# === Exp A Part 2: 语义vs随机传播效率 ===
print("\n" + "="*60)
print("Exp A Part 2: 语义vs随机传播效率比")
print("="*60)

svr = data['expA']['semantic_vs_random']
for eps_label in ['0.01%', '0.05%', '0.1%', '0.5%', '1%', '2%', '5%']:
    if eps_label not in svr:
        continue
    print(f"\n  扰动强度 {eps_label}:")
    
    # NOT vs random
    not_ratio_key = 'NOT_vs_random_ratio'
    if not_ratio_key in svr[eps_label]:
        ratio_data = svr[eps_label][not_ratio_key]
        # 找到关键层的比值
        key_layers = []
        for lk in sorted(ratio_data.keys(), key=lambda x: int(x[1:])):
            li = int(lk[1:])
            if li in [0, 1, 6, 12, 18, 24, 30, 35, 36]:
                key_layers.append((lk, ratio_data[lk]))
        for lk, rv in key_layers:
            print(f"    NOT vs Random @ {lk}: ratio={rv:.4f} ({'语义更稳定' if rv < 1 else '语义更敏感'})")
    
    # PAST vs random
    past_ratio_key = 'PAST_vs_random_ratio'
    if past_ratio_key in svr[eps_label]:
        ratio_data = svr[eps_label][past_ratio_key]
        key_layers = []
        for lk in sorted(ratio_data.keys(), key=lambda x: int(x[1:])):
            li = int(lk[1:])
            if li in [0, 1, 6, 12, 18, 24, 30, 35, 36]:
                key_layers.append((lk, ratio_data[lk]))
        for lk, rv in key_layers:
            print(f"    PAST vs Random @ {lk}: ratio={rv:.4f} ({'语义更稳定' if rv < 1 else '语义更敏感'})")

# === Exp B: LM Head SVD ===
print("\n" + "="*60)
print("Exp B: LM Head SVD分析")
print("="*60)

expB = data['expB']
print(f"  W_U shape: {expB['W_U_shape']}")
print(f"  条件数: {expB['condition_number']:.2f}")
print(f"  90%能量维度: {expB['dim_90pct']}/{expB['W_U_shape'][1]} ({expB['dim_90pct']/expB['W_U_shape'][1]*100:.1f}%)")
print(f"  95%能量维度: {expB['dim_95pct']}/{expB['W_U_shape'][1]} ({expB['dim_95pct']/expB['W_U_shape'][1]*100:.1f}%)")
print(f"  99%能量维度: {expB['dim_99pct']}/{expB['W_U_shape'][1]} ({expB['dim_99pct']/expB['W_U_shape'][1]*100:.1f}%)")
print(f"  有效秩(1%): {expB['effective_rank_1pct']}")
print(f"  参与率(0.1%): {expB['participation_ratio']}")
print(f"  Top1: {expB['top1_singular']:.2f}")
print(f"  Top10/Top1: {expB['top10_ratio']:.4f}")
print(f"  Top100/Top1: {expB['top100_ratio']:.4f}")
print(f"  Top500/Top1: {expB['top500_ratio']:.4f}")
print(f"  谱衰减类型: {expB['spectral_decay_type']}")

# Top50 S
S = expB['singular_values_top50']
print(f"\n  Top50 奇异值:")
for i in range(0, 50, 10):
    vals = [f"{S[j]:.2f}" for j in range(i, min(i+10, len(S)))]
    print(f"    {i+1}-{i+10}: {', '.join(vals)}")

# === Exp C: 语言算子代数 ===
print("\n" + "="*60)
print("Exp C: 语言算子代数")
print("="*60)

expC = data['expC']

# Part 1: 算子响应
print("\n  Part 1: 算子响应 (每层相对变化)")
for op_name in ['NOT', 'PAST', 'PLURAL', 'FUTURE', 'MODAL']:
    if op_name not in expC['operator_responses']:
        continue
    print(f"\n    {op_name}:")
    op_resp = expC['operator_responses'][op_name]
    for lk in sorted(op_resp.keys(), key=lambda x: int(x[1:])):
        li = int(lk[1:])
        if li % 6 == 0 or li <= 2 or li >= 34:
            d = op_resp[lk]
            print(f"      {lk}: rel_change={d['mean_rel_change']:.6f}, "
                  f"direction_consistency={d['direction_consistency']:.4f}")

# Part 2: 算子间重叠
print("\n  Part 2: 算子间方向重叠 (cosine)")
for overlap_key in sorted(expC['operator_overlap'].keys()):
    overlap_data = expC['operator_overlap'][overlap_key]
    print(f"\n    {overlap_key}:")
    for lk in sorted(overlap_data.keys(), key=lambda x: int(x[1:])):
        li = int(lk[1:])
        if li % 9 == 0 or li <= 2 or li >= 34:
            print(f"      {lk}: cosine={overlap_data[lk]:.4f}")

# Part 3: 干涉项
print("\n  Part 3: 干涉项分析")
for comp_name in ['NOT+PAST', 'NOT+FUTURE', 'PLURAL+PAST']:
    if comp_name not in expC['interference_results']:
        continue
    comp = expC['interference_results'][comp_name]
    print(f"\n    {comp_name}:")
    summary = comp.get('summary', {})
    for lk in sorted(summary.keys(), key=lambda x: int(x[1:])):
        li = int(lk[1:])
        if li % 9 == 0 or li <= 2 or li >= 34:
            d = summary[lk]
            ri = d.get('mean_relative_interference')
            lr = d.get('mean_linearity_ratio')
            ri_str = f"{ri:.4f}" if ri else "N/A"
            lr_str = f"{lr:.4f}" if lr else "N/A"
            print(f"      {lk}: relative_interference={ri_str}, "
                  f"linearity_ratio={lr_str}")
