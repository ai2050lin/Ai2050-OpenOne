"""Phase 132 汇总分析脚本"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
from collections import defaultdict

models = ["qwen3", "deepseek7b", "glm4"]
base_dir = "tests/glm5_temp"

print("=" * 80)
print("Phase 132 汇总: 算子流形分析")
print("=" * 80)

# ===== 1. Jacobian有效秩对比 =====
print("\n" + "=" * 80)
print("1. 累积Jacobian有效秩 (J_{l:0}, k_perturb=48)")
print("=" * 80)

for model_name in models:
    path = f"{base_dir}/phase132_{model_name}_operator_manifold.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mi = data["model_info"]
    n_layers = mi["n_layers"]
    d_model = mi["d_model"]
    agg_ranks = data.get("exp1_jacobian_spectrum", {}).get("aggregate", {}).get("eff_ranks", {})

    print(f"\n  {model_name} (L={n_layers}, d={d_model}):")
    sample = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
    for l in sample:
        lk = f"L{l}"
        if lk in agg_ranks:
            d = agg_ranks[lk]
            print(f"    {lk}: eff_rank = {d['mean']:.1f} ± {d['std']:.1f} "
                  f"({d['mean']/d_model*100:.1f}% of d_model)")

# ===== 2. 跨句子子空间对比 =====
print("\n" + "=" * 80)
print("2. 跨句子Jacobian子空间cosine (8维主子空间)")
print("=" * 80)

for model_name in models:
    path = f"{base_dir}/phase132_{model_name}_operator_manifold.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mi = data["model_info"]
    n_layers = mi["n_layers"]
    la = data.get("exp2_operator_subspace", {}).get("layer_analysis", {})

    print(f"\n  {model_name}:")
    print(f"    {'层':<6} {'同语义不同语法':<18} {'不同语义同语法':<18} {'不同语义不同语法':<18} {'比值(同/异)'}")
    for lk in sorted(la.keys()):
        d = la[lk]
        same = d['same_sem_diff_syn']['mean']
        diff_syn = d['diff_sem_same_syn']['mean']
        diff_all = d['diff_sem_diff_syn']['mean']
        ratio = same / max(diff_syn, 1e-6)
        print(f"    {lk:<6} {same:<18.3f} {diff_syn:<18.3f} {diff_all:<18.3f} {ratio:.1f}x")

# ===== 3. 对易子对比 =====
print("\n" + "=" * 80)
print("3. 约束对易子 nl_ratio (修正版, 关键层)")
print("=" * 80)

for model_name in models:
    path = f"{base_dir}/phase132_{model_name}_operator_manifold.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mi = data["model_info"]
    n_layers = mi["n_layers"]
    exp3 = data.get("exp3_commutator", {})

    print(f"\n  {model_name}:")
    mid = n_layers // 2
    for sk in ["dog"]:
        if sk not in exp3:
            continue
        for pk, pd in exp3[sk].items():
            lr = pd.get("layer_results", [])
            if mid < len(lr):
                d = lr[mid]
                # cos(c1, c2): 两个约束效应之间的对齐度
                print(f"    {pk} (L{mid}): nl={d['nl_ratio']:.3f}, "
                      f"cos_c1_c2={d['cos_c1_c2']:.3f}, "
                      f"cos_comm_c1={d['cos_commutator_c1']:.3f}, "
                      f"cos_comm_c2={d['cos_commutator_c2']:.3f}")

# ===== 4. 双重否定(修正版) =====
print("\n" + "=" * 80)
print("4. 双重否定(修正版, 语法正确)")
print("=" * 80)

for model_name in models:
    path = f"{base_dir}/phase132_{model_name}_operator_manifold.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mi = data["model_info"]
    n_layers = mi["n_layers"]
    dn = data.get("exp3_commutator", {}).get("double_negation_corrected", {})

    print(f"\n  {model_name}:")
    mid = n_layers // 2
    for sk in list(dn.keys())[:4]:
        lr = dn[sk].get("layer_results", [])
        if mid < len(lr):
            d = lr[mid]
            print(f"    {sk} (L{mid}): cos(neg,dbl_neg)={d['cos_neg_dblneg']:.3f}, "
                  f"recovery={d['recovery_ratio']:.3f}")

# ===== 5. Tokenization控制 =====
print("\n" + "=" * 80)
print("5. Tokenization控制 (does not vs doesn't)")
print("=" * 80)

for model_name in models:
    path = f"{base_dir}/phase132_{model_name}_operator_manifold.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mi = data["model_info"]
    n_layers = mi["n_layers"]
    exp5 = data.get("exp5_tokenization", {})

    print(f"\n  {model_name}:")
    mid = n_layers // 2
    cos_vals = []
    diff_vals = []
    for sk in list(exp5.keys()):
        if sk == "jacobian_comparison":
            continue
        lr = exp5[sk]
        if isinstance(lr, list) and mid < len(lr):
            d = lr[mid]
            cos_vals.append(d['cos_full_contract'])
            diff_vals.append(d['rel_diff'])
            print(f"    {sk} (L{mid}): cos={d['cos_full_contract']:.3f}, "
                  f"rel_diff={d['rel_diff']:.3f}")
    if cos_vals:
        print(f"    平均: cos={np.mean(cos_vals):.3f}, rel_diff={np.mean(diff_vals):.3f}")

# ===== 6. Exp 4: 输运角和per-layer Jacobian =====
print("\n" + "=" * 80)
print("6. 低秩输运: per-layer Jacobian有效秩")
print("=" * 80)

for model_name in models:
    path = f"{base_dir}/phase132_{model_name}_operator_manifold.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mi = data["model_info"]
    n_layers = mi["n_layers"]
    d_model = mi["d_model"]
    plj = data.get("exp4_transport", {}).get("per_layer_jacobian", {})

    print(f"\n  {model_name}:")
    # 汇总所有句子的per-layer结果
    layer_ranks = defaultdict(list)
    for sent, results in plj.items():
        for r in results:
            layer_ranks[r["layer"]].append(r["eff_rank"])

    sample = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2]
    for l in sample:
        if str(l) in layer_ranks or l in layer_ranks:
            ranks = layer_ranks.get(l, layer_ranks.get(str(l), []))
            if ranks:
                print(f"    L{l}: per-layer J_l eff_rank = {np.mean(ranks):.1f} ± {np.std(ranks):.1f} "
                      f"({np.mean(ranks)/d_model*100:.1f}% of d_model)")

# ===== 7. 关键对比: 约束效应秩 vs Jacobian秩 =====
print("\n" + "=" * 80)
print("7. 关键对比: 约束效应有效秩 vs Jacobian有效秩")
print("=" * 80)
print("""
Phase 131发现: 4种约束效应的有效秩 = 1-3 (跨模板平均)
Phase 132发现: 累积Jacobian的有效秩 = 45-48 (k_perturb=48)

解释:
- Jacobian秩 >> 约束效应秩: Jacobian的大部分"自由度"不是用于传播约束的
- 约束效应只占Jacobian秩的 1-3/45-48 ≈ 2-7%
- 这意味着: 约束传播只用了Jacobian的一小部分方向
- Jacobian的其余方向可能用于: 语义传播, token路由, 位置编码传播等
""")

# ===== 8. 跨模型核心发现 =====
print("\n" + "=" * 80)
print("8. 跨模型核心发现总结")
print("=" * 80)
print("""
1. Jacobian有效秩 ≈ 45-48 (k=48截断):
   - 真实秩可能更大, 但前48个方向已覆盖主要传播能量
   - Qwen3/GLM4: 秩随层缓慢递减 (46→45)
   - DS7B: 秩随层显著递减 (48→5), 最后层退化

2. L0层Jacobian子空间主要由语义决定:
   - 同语义不同语法: cos ≈ 0.50-0.89
   - 不同语义同语法: cos ≈ 0.00-0.79
   - 中间层: 所有cos都很低 (0.01-0.05), 无通用算子子空间

3. 对易子: Neg∘Past仍然最强(nl≈0.8-1.1)

4. 双重否定: cos(neg, dbl_neg) ≈ 0.62-0.82, 双否定不恢复base

5. Tokenization: cos("does not", "doesn't") ≈ 0.88-0.97, 但差异仍然显著
""")
