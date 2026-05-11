"""Phase 131 汇总分析脚本"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
from collections import defaultdict

models = ["qwen3", "deepseek7b", "glm4"]
base_dir = "tests/glm5_temp"

for model_name in models:
    path = f"{base_dir}/phase131_{model_name}_constraint_algebra.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_layers = data["model_info"]["n_layers"]
    d_model = data["model_info"]["d_model"]
    print(f"\n{'='*70}")
    print(f"模型: {model_name} (L={n_layers}, d={d_model})")
    print(f"{'='*70}")

    # ---- Exp 1: 约束加法性 nl_ratio ----
    print("\n[Exp 1] 约束加法性 nl_ratio (δ(A+B) vs δ(A)+δ(B)):")
    exp1 = data.get("exp1_commutativity", {})
    # 汇总不同约束对的nl_ratio随层变化
    pair_nl_by_layer = {}
    for tk, pairs in exp1.items():
        if not isinstance(pairs, dict):
            continue
        for pair_name, pd in pairs.items():
            if not isinstance(pd, dict) or "layer_results" not in pd:
                continue
            if pd.get("type") != "additivity":
                continue
            if pair_name not in pair_nl_by_layer:
                pair_nl_by_layer[pair_name] = defaultdict(list)
            for l, lr in enumerate(pd["layer_results"]):
                pair_nl_by_layer[pair_name][l].append(lr["nl_ratio"])

    # 打印关键层的平均
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    print(f"  {'约束对':<25} " + " ".join([f"L{l:<4}" for l in sample_layers]))
    for pair_name in sorted(pair_nl_by_layer.keys()):
        vals = []
        for l in sample_layers:
            if l in pair_nl_by_layer[pair_name]:
                vals.append(f"{np.mean(pair_nl_by_layer[pair_name][l]):.3f}")
            else:
                vals.append("  -  ")
        print(f"  {pair_name:<25} " + " ".join(vals))

    # ---- Exp 2: 约束逆元 ----
    print("\n[Exp 2] 约束逆元 (Neg∘Neg vs Identity):")
    exp2 = data.get("exp2_inverse", {})
    inv_metrics = {"rel_diff": defaultdict(list), "neg_neg_ratio": defaultdict(list),
                   "cos_neg_neg": defaultdict(list)}
    for tk, d in exp2.items():
        if not isinstance(d, dict) or "layer_results" not in d:
            continue
        for l, lr in enumerate(d["layer_results"]):
            for key in inv_metrics:
                if key in lr:
                    inv_metrics[key][l].append(lr[key])

    print(f"  {'指标':<20} " + " ".join([f"L{l:<4}" for l in sample_layers]))
    for key in ["rel_diff", "neg_neg_ratio", "cos_neg_neg"]:
        vals = []
        for l in sample_layers:
            if l in inv_metrics[key] and inv_metrics[key][l]:
                vals.append(f"{np.mean(inv_metrics[key][l]):.3f}")
            else:
                vals.append("  -  ")
        print(f"  {key:<20} " + " ".join(vals))

    # ---- Exp 3: 约束传播核有效秩 ----
    print("\n[Exp 3] 约束传播核有效秩 (4约束×6模板):")
    exp3_dim = data.get("exp3_constraint_kernel", {}).get("dimensionality_analysis", {})
    for tk in list(exp3_dim.keys())[:2]:
        eff_ranks = []
        for l in range(n_layers):
            lk = f"L{l}"
            if lk in exp3_dim[tk]:
                eff_ranks.append(exp3_dim[tk][lk]["eff_rank"])
            else:
                eff_ranks.append(0)
        sampled = [eff_ranks[l] for l in sample_layers if l < len(eff_ranks)]
        print(f"  {tk}: " + " ".join([f"{v:.2f}" for v in sampled]))

    # ---- Exp 4: 低秩子空间 ----
    print("\n[Exp 4] 约束效应低秩结构 (全部约束效应):")
    exp4_layer = data.get("exp4_low_rank", {}).get("layer_analysis", {})
    print(f"  {'层':<6} {'eff_rank':<10} {'dim_50':<8} {'dim_90':<8} {'dim_95':<8} {'dim_99':<8} {'N_effects':<10}")
    for l in sample_layers:
        lk = f"L{l}"
        if lk in exp4_layer:
            d = exp4_layer[lk]
            de = d.get("dims_for_energy", {})
            print(f"  {lk:<6} {d['eff_rank']:<10.2f} {de.get('dim_50pct','?'):<8} "
                  f"{de.get('dim_90pct','?'):<8} {de.get('dim_95pct','?'):<8} "
                  f"{de.get('dim_99pct','?'):<8} {d['N_effects']:<10}")

    # 跨模板子空间相似度
    print("\n[Exp 4] 跨模板约束子空间相似度 (mean cosine):")
    exp4_cross = data.get("exp4_low_rank", {}).get("cross_template_analysis", {})
    cross_cos_by_layer = []
    for l in range(n_layers):
        lk = f"L{l}"
        if lk in exp4_cross and "subspace_cosines" in exp4_cross[lk]:
            cos_vals = list(exp4_cross[lk]["subspace_cosines"].values())
            cross_cos_by_layer.append(np.mean(cos_vals))
        else:
            cross_cos_by_layer.append(0)
    sampled = [cross_cos_by_layer[l] for l in sample_layers if l < len(cross_cos_by_layer)]
    print(f"  Mean subspace cosine: " + " ".join([f"{v:.3f}" for v in sampled]))

    # ---- Exp 5: 层间组合 ----
    print("\n[Exp 5] 约束组合非线性比 (所有约束对平均):")
    exp5_summary = data.get("exp5_layerwise_composition", {}).get("layer_summary", {})
    for l in sample_layers:
        lk = f"L{l}"
        if lk in exp5_summary:
            d = exp5_summary[lk]
            print(f"  {lk}: nl_ratio={d['mean_nl_ratio']:.4f} ± {d['std_nl_ratio']:.4f}, "
                  f"lin_acc={d['mean_lin_acc']:.4f}, n={d['n_pairs']}")

# ---- 跨模型对比 ----
print(f"\n{'='*70}")
print("跨模型对比: 约束传播核有效秩")
print(f"{'='*70}")
for model_name in models:
    path = f"{base_dir}/phase131_{model_name}_constraint_algebra.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_layers = data["model_info"]["n_layers"]
    d_model = data["model_info"]["d_model"]
    exp3_dim = data.get("exp3_constraint_kernel", {}).get("dimensionality_analysis", {})
    # 取第一个模板的eff_rank
    tk = list(exp3_dim.keys())[0] if exp3_dim else None
    if tk:
        eff_ranks = []
        for l in range(n_layers):
            lk = f"L{l}"
            if lk in exp3_dim[tk]:
                eff_ranks.append(exp3_dim[tk][lk]["eff_rank"])
        mid = n_layers // 2
        print(f"  {model_name} (d={d_model}): L0={eff_ranks[0]:.2f}, "
              f"L{mid}={eff_ranks[mid]:.2f}, "
              f"L{n_layers-1}={eff_ranks[-1]:.2f}")

print(f"\n跨模型对比: 低秩子空间 dim_90pct")
for model_name in models:
    path = f"{base_dir}/phase131_{model_name}_constraint_algebra.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_layers = data["model_info"]["n_layers"]
    d_model = data["model_info"]["d_model"]
    exp4_layer = data.get("exp4_low_rank", {}).get("layer_analysis", {})
    dims = []
    for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        lk = f"L{l}"
        if lk in exp4_layer:
            dims.append(exp4_layer[lk].get("dims_for_energy", {}).get("dim_90pct", "?"))
        else:
            dims.append("?")
    print(f"  {model_name} (d={d_model}): " + " -> ".join([str(d) for d in dims]))
