"""
Phase 470 结果分析: 跨模型DCF对比
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
from pathlib import Path

results_dir = Path("results/glm5")
models = ["qwen3", "glm4", "deepseek7b"]

all_data = {}
for model in models:
    path = results_dir / f"phase470_{model}_r1.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            all_data[model] = json.load(f)

print("=" * 80)
print("Phase 470 跨模型DCF对比分析")
print("=" * 80)

# ==================== Exp1对比 ====================
print("\n### Exp1: DCF vs Resid 聚类质量对比\n")

print(f"{'Model':<12} {'Layer':<8} {'DCF_sil':<10} {'DCF_c_sil':<12} {'Resid_sil':<10} {'DCF_adv':<10} {'DCF_disc':<10} {'Resid_disc':<12}")
print("-" * 90)

for model in models:
    if model not in all_data:
        continue
    exp1 = all_data[model].get("exp1_dcf_construction", {})
    for layer_key in sorted([k for k in exp1 if k.startswith("L")], key=lambda x: int(x[1:])):
        layer_data = exp1[layer_key]
        cl = layer_data.get("clustering", {})
        pw = layer_data.get("pairwise_comparison", {})
        print(f"{model:<12} {layer_key:<8} {cl.get('dcf_silhouette',0):<10.4f} "
              f"{cl.get('dcf_centered_silhouette',0):<12.4f} {cl.get('resid_silhouette',0):<10.4f} "
              f"{cl.get('dcf_advantage',0):<10.4f} {pw.get('dcf_discriminability',0):<10.4f} "
              f"{pw.get('resid_discriminability',0):<12.4f}")

# Exp1汇总
print("\n### Exp1 汇总\n")
for model in models:
    if model not in all_data:
        continue
    exp1 = all_data[model].get("exp1_dcf_construction", {})
    summary = exp1.get("summary", {})
    print(f"  {model}: DCF wins in {summary.get('dcf_wins_over_resid','?')}/{summary.get('total_layers_tested','?')} layers")

# ==================== Exp2对比 ====================
print("\n### Exp2: 关系槽位分离\n")

for model in models:
    if model not in all_data:
        continue
    exp2 = all_data[model].get("exp2_relation_slot", {})
    summary = exp2.get("summary", {})
    print(f"  {model}:")
    print(f"    mean_inter_relation_cos: {summary.get('mean_inter_relation_cos',0):.4f}")
    print(f"    kind_of_correct: {summary.get('kind_of_correct_category','?')}")
    print(f"    constraint_theory: {'CONFIRMED' if summary.get('theory_prediction_confirmed') else 'NOT confirmed'}")

    # 每个对象的关系间cos
    per_obj = exp2.get("per_object", {})
    for obj_name, obj_data in per_obj.items():
        print(f"    {obj_name}: inter_rel_cos={obj_data.get('mean_inter_relation_cos',0):.4f}, "
              f"top_constraint={obj_data.get('top_constraint_by_relation',{})}")

# ==================== Exp3对比 ====================
print("\n### Exp3: DCF维度重要性排序\n")

for model in models:
    if model not in all_data:
        continue
    exp3 = all_data[model].get("exp3_dcf_alignment", {})
    dim_ranking = exp3.get("dim_importance_ranking", [])
    top3 = [d["dimension"] for d in dim_ranking[:3]]
    top3_vars = [d["variance"] for d in dim_ranking[:3]]
    entropy = exp3.get("entropy_stats", {})

    print(f"  {model}:")
    print(f"    Top discriminating dims: {top3} (var={top3_vars})")
    print(f"    Entropy: mean={entropy.get('mean',0):.4f}, std={entropy.get('std',0):.4f}")

    # 类别DCF profile
    cat_dcf = exp3.get("category_dcf_profiles", {})
    dim_names = list(json.loads(open(results_dir / f"phase470_{model}_r1.json", 'r', encoding='utf-8').read())
                     .get("exp3_dcf_alignment", {}).get("category_dcf_profiles", {}).get("fruit", {}).get("mean_dcf", []))
    
    # 每个类别的mean DCF(只展示前3个维度)
    for cat in ["fruit", "animal", "vehicle", "tool"]:
        if cat in cat_dcf:
            mean_dcf = cat_dcf[cat].get("mean_dcf", [])
            # 找到最高维度
            if mean_dcf:
                max_idx = np.argmax(mean_dcf)
                dim_names_list = ["fruit", "animal", "tool", "vehicle", "clothing", "furniture", "food", "plant"]
                if max_idx < len(dim_names_list):
                    print(f"    {cat}: max_dim={dim_names_list[max_idx]}(logit={mean_dcf[max_idx]:.2f}), "
                          f"spread={cat_dcf[cat].get('std_dcf',[0]*8)[max_idx]:.2f}")

# ==================== 跨模型DCF对齐分析 ====================
print("\n### 跨模型DCF对齐: 类别约束profile\n")

# 比较每个类别在不同模型中的DCF向量
for cat in ["fruit", "animal", "vehicle", "tool"]:
    print(f"\n  Category: {cat}")
    for model in models:
        if model not in all_data:
            continue
        exp3 = all_data[model].get("exp3_dcf_alignment", {})
        cat_dcf = exp3.get("category_dcf_profiles", {})
        if cat in cat_dcf:
            mean_dcf = cat_dcf[cat].get("mean_dcf", [])
            if mean_dcf:
                # 找top-2维度
                top2_idx = np.argsort(mean_dcf)[-2:][::-1]
                dim_names_list = ["fruit", "animal", "tool", "vehicle", "clothing", "furniture", "food", "plant"]
                top2 = [(dim_names_list[i] if i < len(dim_names_list) else f"dim{i}", round(mean_dcf[i], 2)) for i in top2_idx]
                print(f"    {model}: top2={top2}")

# ==================== 核心发现 ====================
print("\n" + "=" * 80)
print("核心发现总结")
print("=" * 80)

print("""
1. DCF聚类优势:
   - Qwen3: DCF silhouette=0.74 >> Resid silhouette=0.45 (5/5层胜出)
   - GLM4: DCF silhouette=0.57 >> Resid silhouette=0.46 (5/5层胜出)
   - DS7B: DCF silhouette=-0.14 << Resid silhouette=0.27 (0/5层胜出)
   
   解释: DCF直接度量输出分布约束, 对正常模型更有效。
   DS7B的DCF为负说明其logit分布被数学推理模式严重污染,
   族词logit不能反映语义约束。

2. 关系槽位分离(所有3模型确认):
   - Qwen3: inter-relation cos=-0.21 (低=约束条件化)
   - GLM4: inter-relation cos=-0.20
   - DS7B: inter-relation cos=-0.17
   结论: 同一对象在不同关系下产生不同的分布约束,
   支持"意义是条件化约束"而非"固定概念向量"

3. kind_of正确指向类别:
   - Qwen3: 6/8, GLM4: 7/8, DS7B: 6/8
   kind_of模板下对象的DCF最高维度确实指向正确类别

4. DS7B异常:
   DCF silhouette为负, 族词维度重要性排序也不同(clothing/furniture/food),
   与Qwen3(clothing/plant/fruit)和GLM4(fruit/animal/clothing)不一致。
   这与Phase 469的发现一致: DS7B的输出分布被行为模式严重污染。
""")
