import json, numpy as np

models = ["qwen3", "glm4", "deepseek7b"]
all_data = {}

for m in models:
    with open(f'tests/glm5_temp/phase230_{m}_results.json','r',encoding='utf-8') as f:
        all_data[m] = json.load(f)

print("=" * 80)
print("PHASE 230 三模型对比分析")
print("=" * 80)

# ===== Exp1: 形容词组合性 =====
print("\n### Exp1: 形容词组合性 — 跨名词稳定性 ###")
print(f"{'Layer':<8}", end="")
for m in models:
    print(f"  {m:>8}_same  {m:>8}_cross  {m:>6}_sep", end="")
print()

for m in models:
    layers = sorted([k for k in all_data[m]["exp1"].keys() if all_data[m]["exp1"][k]["same_adj_mean_cos"] > 0], key=lambda x: int(x[1:]))
    if m == models[0]:
        for lk in layers:
            vals = []
            for m2 in models:
                if lk in all_data[m2]["exp1"] and all_data[m2]["exp1"][lk]["same_adj_mean_cos"] > 0:
                    d = all_data[m2]["exp1"][lk]
                    vals.append(f"{d['same_adj_mean_cos']:8.4f}  {d['cross_adj_mean_cos']:8.4f}  {d['separation_ratio']:6.2f}x")
                else:
                    vals.append(f"{'N/A':>8}  {'N/A':>8}  {'N/A':>6}")
            print(f"{lk:<8}  " + "  ".join(vals))
    break

# ===== Exp1: 最佳层对比 =====
print("\n### Exp1: 各模型最佳层(最高same_cos)的形容词排名 ###")
for m in models:
    best_lk = None
    best_val = 0
    for lk, lv in all_data[m]["exp1"].items():
        if lv["same_adj_mean_cos"] > best_val:
            best_val = lv["same_adj_mean_cos"]
            best_lk = lk
    
    pa = all_data[m]["exp1"][best_lk]["per_adj_stability"]
    sorted_adj = sorted(pa.items(), key=lambda x: x[1]["mean_cos"], reverse=True)
    print(f"\n  {m} (best: {best_lk}, same_cos={best_val:.4f}):")
    top5 = [(a, d["mean_cos"]) for a, d in sorted_adj[:5]]
    bot5 = [(a, d["mean_cos"]) for a, d in sorted_adj[-5:]]
    print(f"    Top5: {top5}")
    print(f"    Bot5: {bot5}")

# ===== Exp2: 操作编码 =====
print("\n### Exp2: 操作编码 — 跨句子稳定性 ###")
for m in models:
    print(f"\n  {m}:")
    layers = sorted([k for k in all_data[m]["exp2"].keys() if all_data[m]["exp2"][k]["same_op_mean_cos"] > 0], key=lambda x: int(x[1:]))
    for lk in layers:
        d = all_data[m]["exp2"][lk]
        print(f"    {lk}: same={d['same_op_mean_cos']:.4f} cross={d['cross_op_mean_cos']:.4f} sep={d['separation_ratio']:.2f}x")

# ===== Exp2: 逐操作分析(最高分离度层) =====
print("\n### Exp2: 各操作在最高分离度层的稳定性 ###")
for m in models:
    best_lk = None
    best_sep = 0
    for lk, lv in all_data[m]["exp2"].items():
        if lv["separation_ratio"] > best_sep:
            best_sep = lv["separation_ratio"]
            best_lk = lk
    
    if best_lk:
        per_op = all_data[m]["exp2"][best_lk]["per_op"]
        sorted_ops = sorted(per_op.items(), key=lambda x: x[1]["mean_cos"], reverse=True)
        print(f"\n  {m} (best: {best_lk}, sep={best_sep:.2f}x):")
        for op, d in sorted_ops:
            print(f"    {op:12s}: cos={d['mean_cos']:.4f}±{d['std_cos']:.4f} norm={d['delta_norm']:.4f}")

# ===== 关键对比: 形容词 vs 操作 =====
print("\n### 关键对比: 形容词方向稳定性 vs 操作方向稳定性 ###")
for m in models:
    # 形容词: 中层的平均same_cos
    adj_sames = [lv["same_adj_mean_cos"] for lk, lv in all_data[m]["exp1"].items() if lv["same_adj_mean_cos"] > 0]
    adj_crosses = [lv["cross_adj_mean_cos"] for lk, lv in all_data[m]["exp1"].items() if lv["cross_adj_mean_cos"] > 0]
    
    # 操作: 中层的平均same_cos
    op_sames = [lv["same_op_mean_cos"] for lk, lv in all_data[m]["exp2"].items() if lv["same_op_mean_cos"] > 0]
    op_crosses = [lv["cross_op_mean_cos"] for lk, lv in all_data[m]["exp2"].items() if lv["cross_op_mean_cos"] > 0]
    
    print(f"  {m}:")
    print(f"    Adj: same={np.mean(adj_sames):.4f} cross={np.mean(adj_crosses):.4f}")
    print(f"    Op:  same={np.mean(op_sames):.4f} cross={np.mean(op_crosses):.4f}")

# ===== DS7B异常分析 =====
print("\n### DS7B Exp1 异常: 中层same_cos和cross_cos都≈1.0 ###")
m = "deepseek7b"
for lk in sorted(all_data[m]["exp1"].keys(), key=lambda x: int(x[1:])):
    lv = all_data[m]["exp1"][lk]
    if lv["same_adj_mean_cos"] > 0:
        print(f"  {lk}: same={lv['same_adj_mean_cos']:.6f} cross={lv['cross_adj_mean_cos']:.6f} "
              f"sep={lv['separation_ratio']:.4f}x")

print("\n### DS7B Exp2: 操作编码(有意义的分离) ###")
for lk in sorted(all_data[m]["exp2"].keys(), key=lambda x: int(x[1:])):
    lv = all_data[m]["exp2"][lk]
    if lv["same_op_mean_cos"] > 0:
        print(f"  {lk}: same={lv['same_op_mean_cos']:.6f} cross={lv['cross_op_mean_cos']:.6f} "
              f"sep={lv['separation_ratio']:.4f}x")
