"""提取Qwen3 Phase 140关键数据"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

fn = 'tests/glm5_temp/phase140_qwen3_operator_mechanics_20260512_1510.json'
with open(fn, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === Exp A: 传播效率比 vs 扰动强度 ===
print("="*60)
print("Exp A: 语义vs随机传播效率比 vs 扰动强度")
print("="*60)

svr = data['expA']['semantic_vs_random']
print("\nNOT算子 vs 随机扰动:")
print(f"{'强度':<8} {'L0':<10} {'L6':<10} {'L12':<10} {'L18':<10} {'L24':<10} {'L30':<10} {'L36':<10}")
for eps_label in ['0.01%', '0.05%', '0.1%', '0.5%', '1%', '2%', '5%']:
    if eps_label not in svr:
        continue
    not_key = 'NOT_vs_random_ratio'
    if not_key in svr[eps_label]:
        rd = svr[eps_label][not_key]
        vals = []
        for lk in ['L0', 'L6', 'L12', 'L18', 'L24', 'L30', 'L36']:
            if lk in rd:
                vals.append(f"{rd[lk]:.4f}")
            else:
                vals.append("N/A")
        print(f"{eps_label:<8} {' '.join(f'{v:<10}' for v in vals)}")

print("\nPAST算子 vs 随机扰动:")
print(f"{'强度':<8} {'L0':<10} {'L6':<10} {'L12':<10} {'L18':<10} {'L24':<10} {'L30':<10} {'L36':<10}")
for eps_label in ['0.01%', '0.05%', '0.1%', '0.5%', '1%', '2%', '5%']:
    if eps_label not in svr:
        continue
    past_key = 'PAST_vs_random_ratio'
    if past_key in svr[eps_label]:
        rd = svr[eps_label][past_key]
        vals = []
        for lk in ['L0', 'L6', 'L12', 'L18', 'L24', 'L30', 'L36']:
            if lk in rd:
                vals.append(f"{rd[lk]:.4f}")
            else:
                vals.append("N/A")
        print(f"{eps_label:<8} {' '.join(f'{v:<10}' for v in vals)}")

# === Exp B: LM Head SVD 关键结论 ===
print("\n" + "="*60)
print("Exp B: LM Head SVD 关键结论")
print("="*60)
expB = data['expB']
print(f"  W_U shape: {expB['W_U_shape']}")
print(f"  条件数: {expB['condition_number']:.2f}")
print(f"  90%能量维度: {expB['dim_90pct']}/{expB['W_U_shape'][1]} = {expB['dim_90pct']/expB['W_U_shape'][1]*100:.1f}%")
print(f"  99%能量维度: {expB['dim_99pct']}/{expB['W_U_shape'][1]} = {expB['dim_99pct']/expB['W_U_shape'][1]*100:.1f}%")
print(f"  Top1 S = {expB['top1_singular']:.2f}")
print(f"  Top10/Top1 = {expB['top10_ratio']:.4f}")
print(f"  Top100/Top1 = {expB['top100_ratio']:.4f}")
print(f"  ** LM head是极低秩映射! ~{expB['dim_90pct']}维捕获90%能量")

# === Exp C: 干涉项 ===
print("\n" + "="*60)
print("Exp C: 干涉项关键数据")
print("="*60)
expC = data['expC']
for comp_name in ['NOT+PAST', 'NOT+FUTURE', 'PLURAL+PAST']:
    if comp_name not in expC['interference_results']:
        continue
    comp = expC['interference_results'][comp_name]
    summary = comp.get('summary', {})
    print(f"\n  {comp_name}:")
    for lk in sorted(summary.keys(), key=lambda x: int(x[1:])):
        li = int(lk[1:])
        if li % 6 == 0 or li <= 2 or li >= 34:
            d = summary[lk]
            ri = d.get('mean_relative_interference')
            lr = d.get('mean_linearity_ratio')
            ri_str = f"{ri:.4f}" if ri else "N/A"
            lr_str = f"{lr:.4f}" if lr else "N/A"
            interp = "近线性" if lr and lr > 0.9 else "强干涉" if lr and lr < 0.7 else "弱干涉"
            print(f"    {lk}: rel_int={ri_str}, linearity={lr_str} ({interp})")

# 算子方向一致性
print("\n" + "="*60)
print("Exp C: 算子方向一致性 (direction_consistency)")
print("="*60)
for op_name in ['NOT', 'PAST', 'PLURAL', 'FUTURE', 'MODAL']:
    if op_name not in expC['operator_responses']:
        continue
    op_resp = expC['operator_responses'][op_name]
    # 找到最一致的层
    best_layer = None
    best_cons = 0
    for lk, d in op_resp.items():
        if d['direction_consistency'] > best_cons:
            best_cons = d['direction_consistency']
            best_layer = lk
    print(f"  {op_name}: 最佳一致性层={best_layer} ({best_cons:.4f})")
    # 浅层vs深层
    shallow = [d['direction_consistency'] for lk, d in op_resp.items() if int(lk[1:]) <= 12]
    deep = [d['direction_consistency'] for lk, d in op_resp.items() if int(lk[1:]) >= 24]
    if shallow and deep:
        print(f"    浅层(≤L12)一致性: {np.mean(shallow):.4f}")
        print(f"    深层(≥L24)一致性: {np.mean(deep):.4f}")
