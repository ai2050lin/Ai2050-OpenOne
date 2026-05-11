"""Phase 128 汇总脚本 — 参数响应拓扑分析"""
import json, os
import numpy as np

temp_dir = os.path.dirname(__file__)
models = ["qwen3", "deepseek7b", "glm4"]

all_data = {}
for m in models:
    path = os.path.join(temp_dir, f"phase128_{m}_param_topology.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            all_data[m] = json.load(f)

print("=" * 70)
print("Phase 128 汇总: 参数响应拓扑分析")
print("=" * 70)

# ============ Exp 1: 维度控制比较 ============
print("\n### Exp 1: 维度控制轨迹比较 (修复维度作弊!) ###")
for m in models:
    if m not in all_data or "exp1_dim_controlled" not in all_data[m]:
        continue
    d = all_data[m]["exp1_dim_controlled"]
    if "error" in d:
        print(f"  {m}: ERROR - {d['error']}")
        continue
    print(f"\n  {m}:")
    print(f"    Best single layer sil: {d.get('best_single_sil', 'N/A')}")
    print(f"    Trajectory PCA-100 sil: {d.get('traj_sil_same_dim', 'N/A')}")
    print(f"    Fourier spectrum sil: {d.get('fourier_sil', 'N/A')}")
    print(f"    Layer-delta sil: {d.get('delta_sil', 'N/A')}")
    print(f"    Per-layer-PCA10 concat sil: {d.get('concat_pca10_sil', 'N/A')}")
    
    dc = d.get("dim_comparison", {})
    print(f"    --- 维度比较 ---")
    for dim in ["10", "20", "50", "100", "200"]:
        if dim in dc:
            print(f"      dim={dim}: single={dc[dim]['single_layer_sil']:.4f}, "
                  f"traj={dc[dim]['trajectory_sil']:.4f}, "
                  f"adv={dc[dim]['trajectory_advantage']:.2f}x")

# ============ Exp 3: 拓扑邻接 ============
print("\n### Exp 3: 概念拓扑邻接 — Jaccard vs Cosine ###")
for m in models:
    if m not in all_data or "exp3_topology_adjacency" not in all_data[m]:
        continue
    d = all_data[m]["exp3_topology_adjacency"]
    if "error" in d:
        print(f"  {m}: ERROR")
        continue
    print(f"\n  {m}:")
    for metric in ["jaccard_attn", "jaccard_mlp", "jaccard_combined", "cosine_mid_layer"]:
        if metric in d:
            print(f"    {metric}: same={d[metric]['same_cat_mean']:.4f}, "
                  f"diff={d[metric]['diff_cat_mean']:.4f}, "
                  f"discrimination={d[metric]['discrimination']:.4f}")

# ============ Exp 4: 组合语义 ============
print("\n### Exp 4: 组合语义 (属性绑定) ###")
for m in models:
    if m not in all_data or "exp4_compositional" not in all_data[m]:
        continue
    d = all_data[m]["exp4_compositional"]
    print(f"\n  {m}:")
    for noun in ["apple", "dog", "city"]:
        if noun in d:
            print(f"    {noun}: MLP_overlap={d[noun]['mlp_overlap_mean']:.4f}, "
                  f"Attn_overlap={d[noun]['attn_overlap_mean']:.4f}, "
                  f"COS={d[noun]['cosine_sim_mean']:.4f}")
    cn = d.get("cross_noun", {})
    for attr in ["red", "big", "small"]:
        if attr in cn:
            print(f"    cross-{attr}: MLP_overlap={cn[attr]['mlp_overlap_mean']:.4f}, "
                  f"COS={cn[attr]['cosine_sim_mean']:.4f}")

# ============ Exp 5: 语法绑定 ============
print("\n### Exp 5: 语法绑定 (同token不同回路) ###")
for m in models:
    if m not in all_data or "exp5_syntactic" not in all_data[m]:
        continue
    d = all_data[m]["exp5_syntactic"]
    if "error" in d:
        print(f"  {m}: ERROR")
        continue
    print(f"\n  {m}: Avg MLP overlap={d['avg_mlp_overlap']:.4f}, "
          f"Avg Attn change={d['avg_attn_change_ratio']:.4f}, "
          f"Avg mid-COS={d['avg_mid_layer_cosine']:.4f}")
    
    # 关键对比: 主宾互换 vs 主动被动
    for pr in d.get("pair_details", []):
        p = pr["pair"]
        mid_cos = list(pr.get("layer_cosine", {}).values())
        mid_cos_val = mid_cos[len(mid_cos)//2] if mid_cos else 0
        mlp_vals = list(pr.get("mlp_overlap", {}).values())
        mlp_early = mlp_vals[0] if mlp_vals else 0
        mlp_late = mlp_vals[-1] if mlp_vals else 0
        print(f"    '{p[0]}' vs '{p[1]}': mid_cos={mid_cos_val:.4f}, "
              f"MLP_early={mlp_early:.4f}, MLP_late={mlp_late:.4f}")

# ============ 关键结论 ============
print("\n" + "=" * 70)
print("关键结论:")
print("=" * 70)
print("""
1. [维度作弊确认!] Phase 127的轨迹3倍优势完全是维度效应!
   - dim=10时: 轨迹仅1.05-1.12x优势 (几乎无差异)
   - dim=100时: 轨迹反而0.82-0.91x (比单层更差!)
   - 但"逐层PCA10拼接→PCA100"给出3-4倍真实优势

2. [Cosine > Jaccard] 欧氏距离仍优于激活图重叠!
   - Cosine discrimination: 0.095-0.120
   - Jaccard-MLP discrimination: 0.034-0.045
   - Jaccard-Attn discrimination: ~0
   - 这反驳了"语义是拓扑对象"的假设 — 至少Jaccard不是正确的拓扑度量

3. [语法绑定确认!] 同token不同语序激活不同MLP neurons!
   - 主宾互换: MLP overlap = 0.10-0.23 (早期层), 0.39-0.56 (末层)
   - 主动→被动: MLP overlap = 0.05-0.12 (早期层) — 更大差异
   - 这直接证实: 语义=条件计算, 不同语法结构激活不同参数回路

4. [层间分化规律] MLP neuron overlap从低到高:
   - 早期层: 低overlap → 正在区分不同语义
   - 末层: 高overlap → 不同语义最终汇聚到共享输出空间
   - 但"分化"发生在中间层, 不是embedding层
""")
