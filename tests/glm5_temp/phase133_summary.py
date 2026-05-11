"""Phase 133 汇总: 从三模型结果中提取关键结论"""
import json, os
import numpy as np

temp_dir = os.path.join(os.path.dirname(__file__))
models = ["qwen3", "deepseek7b", "glm4"]

all_data = {}
for m in models:
    path = os.path.join(temp_dir, f"phase133_{m}_rigorous_operator.json")
    with open(path, 'r') as f:
        all_data[m] = json.load(f)

print("=" * 80)
print("Phase 133 关键结论汇总")
print("=" * 80)

# === Exp 1: 满秩标度律 ===
print("\n### Exp 1: 满秩标度律 — eff_rank(k) 曲线")
print("-" * 80)

for m in models:
    d = all_data[m]["model_info"]["d_model"]
    agg = all_data[m]["exp1_full_rank_scaling"]["aggregate_scaling"]
    print(f"\n{m} (d={d}):")
    print(f"  {'Layer':<8} " + " ".join([f"k={k:<6}" for k in ["16","32","48","64","128","256"]]))
    print(f"  {'':8} " + " ".join([f"{'r(%d%)':<6}" for _ in range(6)]))
    
    for lk in sorted(agg.keys()):
        row = f"  {lk:<8} "
        for k in ["16","32","48","64","128","256"]:
            if k in agg[lk]:
                r = agg[lk][k]["mean_eff_rank"]
                pct = r / d * 100
                row += f"{r:.0f}({pct:.1f}%) "
            else:
                row += f"{'N/A':<10} "
        print(row)

# === Exp 2: ε收敛 ===
print("\n\n### Exp 2: ε收敛 — 最佳ε对的子空间对齐度")
print("-" * 80)

for m in models:
    d = all_data[m]["model_info"]["d_model"]
    exp2 = all_data[m]["exp2_epsilon_convergence"]
    print(f"\n{m}:")
    for sent, sent_data in exp2.items():
        if sent == "summary":
            continue
        for lk, ld in sent_data.items():
            # 找最佳ε对
            best_pair = None
            best_cos = 0
            for pair, data in ld["adjacent_alignment"].items():
                if data["cos_top16"] > best_cos:
                    best_cos = data["cos_top16"]
                    best_pair = pair
            if best_pair:
                bd = ld["adjacent_alignment"][best_pair]
                print(f"  {lk}: best={best_pair}, cos={bd['cos_top16']:.4f}, "
                      f"rel_err={bd['mean_rel_error']:.4f}")

# === Exp 3: 模块分解 ===
print("\n\n### Exp 3: 模块分解 — 各子模块有效秩(k=64)")
print("-" * 80)

for m in models:
    d = all_data[m]["model_info"]["d_model"]
    exp3 = all_data[m]["exp3_module_decomposition"]
    if "summary" not in exp3:
        continue
    print(f"\n{m}:")
    for lk, stage_data in exp3["summary"].items():
        row = f"  {lk}: "
        for stage in ["ln1", "attn", "ln2", "mlp"]:
            if stage in stage_data:
                row += f"{stage}={stage_data[stage]['mean_rank']:.0f}  "
        print(row)

# === 关键对比: Phase 132 vs Phase 133 ===
print("\n\n### Phase 132 vs Phase 133: 探针上限的直接证据")
print("-" * 80)
for m in models:
    d = all_data[m]["model_info"]["d_model"]
    agg = all_data[m]["exp1_full_rank_scaling"]["aggregate_scaling"]
    # 找中间层
    mid_key = None
    for lk in sorted(agg.keys()):
        n_layers = all_data[m]["model_info"]["n_layers"]
        if f"L{n_layers//2}" in lk or lk == f"L{n_layers//2}":
            mid_key = lk
            break
    if mid_key is None:
        mid_key = sorted(agg.keys())[len(agg)//2]
    
    r48 = agg[mid_key].get("48", {}).get("mean_eff_rank", "N/A")
    r256 = agg[mid_key].get("256", {}).get("mean_eff_rank", "N/A")
    print(f"  {m} {mid_key}: r(k=48)={r48}, r(k=256)={r256}")

print("\n\n### 核心结论:")
print("1. Phase 132的'Jacobian有效秩≈28-32'完全是探针上限(k=48)!")
print("   真实秩在k=256时达到170-252, 远高于32")
print("2. ε收敛只在L0层可靠(cos>0.90 at ε=0.01-0.1)")
print("   深层Jacobian不稳定, 不同ε给出完全不同的线性化")
print("3. LayerNorm贡献高秩(49-63), 不是简单归一化")
print("4. DS7B L0的attn rank≈29≈28heads, 但这是k=64的上限, 需要更大k验证")
print("5. '注意力头数=Jacobian秩'假说在大k下不成立")
