"""
Phase 471 结果分析脚本
========================
汇总三个模型的Exp1/Exp2/Exp3结果
"""
import json, os, numpy as np

models = ["qwen3", "glm4", "deepseek7b"]
base = "results/glm5"

all_data = {}
for m in models:
    path = os.path.join(base, f"phase471_{m}_r1.json")
    with open(path, 'r', encoding='utf-8') as f:
        all_data[m] = json.load(f)

# ========== Exp1: Logit-Lens DCF 层间演变 ==========
print("=" * 80)
print("Exp1: Logit-Lens DCF 层间约束可读性追踪")
print("=" * 80)

for m in models:
    d = all_data[m]["exp1_logit_lens_dcf"]
    n_layers = all_data[m]["model_info"]["n_layers"]
    print(f"\n--- {m} (n_layers={n_layers}) ---")
    print(f"  Emergence layer: {d['summary']['emergence_layer']}")
    print(f"  Peak layer: {d['summary']['peak_layer']}")
    print(f"  Peak silhouette: {d['summary']['peak_silhouette']}")
    
    # 打印演变轨迹
    traj = d['summary']['silhouette_trajectory']
    print(f"  Silhouette trajectory:")
    for k, v in sorted(traj.items(), key=lambda x: int(x[0][1:])):
        layer_pct = int(k[1:]) / n_layers * 100
        print(f"    {k} ({layer_pct:.0f}%): sil={v:.4f}")

# ========== 跨模型DCF涌现对比 ==========
print("\n" + "=" * 80)
print("跨模型DCF涌现对比 (归一化到百分比)")
print("=" * 80)

for m in models:
    d = all_data[m]["exp1_logit_lens_dcf"]
    n_layers = all_data[m]["model_info"]["n_layers"]
    traj = d['summary']['silhouette_trajectory']
    
    # 计算各百分位的silhouette
    print(f"\n--- {m} ---")
    # 找到25%, 50%, 75%, 100%位置的值
    target_pcts = [0, 25, 50, 75, 100]
    for pct in target_pcts:
        target_layer = int(pct / 100 * n_layers)
        # 找最接近的已测试层
        best_layer = None
        best_val = None
        for k, v in traj.items():
            li = int(k[1:])
            if best_layer is None or abs(li - target_layer) < abs(best_layer - target_layer):
                best_layer = li
                best_val = v
        print(f"  {pct}% depth (~L{target_layer}): sil={best_val:.4f} (tested at L{best_layer})")

# ========== Exp2: 因果干预汇总 ==========
print("\n" + "=" * 80)
print("Exp2: 因果DCF干预汇总")
print("=" * 80)

for m in models:
    d = all_data[m]["exp2_causal_intervention"]
    s = d["summary"]
    print(f"\n--- {m} ---")
    print(f"  Causal control possible: {s['causal_control_possible']}")
    print(f"  n_interventions: {s['n_interventions']}")
    
    # 汇总成功率
    for layer_key, layer_data in sorted(s["success_by_layer"].items(), key=lambda x: int(x[0][1:])):
        total = layer_data["total"]
        boosted = layer_data["target_boosted"]
        switched = layer_data["dim_switched"]
        print(f"    {layer_key}: target_boosted={boosted}/{total} ({boosted/total*100:.0f}%), dim_switched={switched}/{total} ({switched/total*100:.0f}%)")

# ========== Exp3: 扩展DCF维度对比 ==========
print("\n" + "=" * 80)
print("Exp3: 扩展DCF维度对比")
print("=" * 80)

for m in models:
    d = all_data[m]["exp3_extended_dcf"]
    comp = d["comparison"]
    cva = d["category_vs_attribute_variance"]
    print(f"\n--- {m} ---")
    print(f"  8D DCF: sil={comp['8d_silhouette']:.4f}, disc={comp['8d_discriminability']:.4f}")
    print(f"  20D DCF: sil={comp['20d_silhouette']:.4f}, disc={comp['20d_discriminability']:.4f}")
    print(f"  Improvement (sil): {comp['improvement_sil']:.4f}")
    print(f"  Improvement (disc): {comp['improvement_disc']:.4f}")
    print(f"  Category dim mean var: {cva['category_dim_mean_var']:.4f}")
    print(f"  Attribute dim mean var: {cva['attribute_dim_mean_var']:.4f}")
    print(f"  Ratio (cat/attr): {cva['ratio_cat_to_attr']:.4f}")
    
    # Top-5 dim importance
    for item in d["dim_importance"][:5]:
        print(f"    {item['dimension']}: var={item['variance']:.4f} ({item['type']})")

# ========== 核心发现汇总 ==========
print("\n" + "=" * 80)
print("核心发现汇总")
print("=" * 80)

print("""
Exp1 核心发现: 语义约束的层间涌现具有三阶段模式
  Phase 1 (0-25% depth): 无结构 — DCF silhouette ≈ 0
  Phase 2 (25-75% depth): 弱结构 — DCF silhouette 0.1-0.4
  Phase 3 (75-100% depth): 强结构 — DCF silhouette 0.6-0.85

  但各模型涌现时机不同:
  - Qwen3: Phase 2在L9(25%)开始, Phase 3在L24(67%)跳升
  - GLM4:  Phase 2在L24(60%)才开始, Phase 3在L27(68%)
  - DS7B:   Phase 1持续到L22, Phase 2几乎不存在, Phase 3仅在L27(96%)

Exp2 核心发现: 因果控制可能, 但精确度有限
  - 所有模型注入DCF方向后, 目标维度都有提升(target_boosted > 0)
  - 但维度完全切换(dim_switched)率很低: Qwen3=0%, GLM4=37.5%, DS7B=12.5%
  - 说明DCF方向可以偏移分布, 但不能完全替换语义约束

Exp3 核心发现: 扩展维度反而降低聚类质量
  - 20D DCF的silhouette在所有模型都低于8D DCF
  - 原因: 属性维度的方差(1.5)远低于类别维度(4.6), 增加噪声维度
  - 最有区分力的属性维度: taste (4-5), 排在6个类别维度之后
""")