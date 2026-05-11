"""Phase 127 汇总脚本"""
import json
import numpy as np
import os

temp_dir = os.path.join(os.path.dirname(__file__))
models = ['qwen3', 'deepseek7b', 'glm4']

all_data = {}
for m in models:
    path = os.path.join(temp_dir, f"phase127_{m}_semantic_dynamics.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            all_data[m] = json.load(f)

print("=" * 70)
print("Phase 127 汇总: 语义动力学轨迹分析")
print("=" * 70)

# === Exp 1: 容量匹配消融 ===
print("\n" + "=" * 70)
print("Exp 1: 容量匹配消融 — 公平比较MLP vs Attention")
print("=" * 70)

for m in models:
    if m not in all_data or "exp1_capacity_matched_ablation" not in all_data[m]:
        continue
    e1 = all_data[m]["exp1_capacity_matched_ablation"]
    
    print(f"\n--- {m} ---")
    print(f"  MLP params/layer: {e1.get('mlp_params_per_layer', 0):,}")
    print(f"  Attn params/layer: {e1.get('attn_params_per_layer', 0):,}")
    print(f"  MLP/Attn参数比: {e1.get('param_ratio', 0):.2f}x")
    
    mlp_abl = e1.get("mlp_full_layer_ablation", {})
    attn_abl = e1.get("attn_full_layer_ablation", {})
    
    print(f"  全层消融 KL (MLP vs Attn):")
    for li in sorted(set(list(mlp_abl.keys()) + list(attn_abl.keys()))):
        mlp_kl = mlp_abl.get(li, -1)
        attn_kl = attn_abl.get(li, -1)
        ratio = mlp_kl / attn_kl if attn_kl > 0 and mlp_kl > 0 else 0
        print(f"    L{li}: MLP={mlp_kl:.4f}, Attn={attn_kl:.4f}, MLP/Attn={ratio:.2f}x")
    
    # 参数匹配消融
    mlp_neuron = e1.get("mlp_matched_neuron_ablation", {})
    head_abl = e1.get("single_head_ablation", {})
    
    print(f"  参数量匹配消融:")
    for li in sorted(mlp_neuron.keys()):
        n = mlp_neuron[li].get("n_neurons", 0)
        kl_mlp = mlp_neuron[li].get("mean_kl", 0)
        kl_per_neuron = mlp_neuron[li].get("kl_per_neuron", 0)
        
        if li in head_abl:
            kl_head_mean = head_abl[li].get("mean", 0)
            kl_per_head_param = kl_head_mean / (4 * 128 * all_data[m]["model_info"]["d_model"]) if kl_head_mean > 0 else 0
        else:
            kl_head_mean = 0
            kl_per_head_param = 0
        
        # 比较同等参数量下的KL
        print(f"    L{li}: {n} MLP neurons KL={kl_mlp:.6f} vs 1 head KL={kl_head_mean:.6f}")
        print(f"           per-neuron KL={kl_per_neuron:.8f}, per-head KL/param={kl_per_head_param:.10f}")

# === Exp 2: 全层轨迹聚类 ===
print("\n" + "=" * 70)
print("Exp 2: 全层轨迹聚类")
print("=" * 70)

for m in models:
    if m not in all_data or "exp2_trajectory_clustering" not in all_data[m]:
        continue
    e2 = all_data[m]["exp2_trajectory_clustering"]
    
    sil_traj = e2.get("silhouette_score_full_trajectory", 0)
    best_layer = e2.get("best_single_layer", 0)
    best_sil = e2.get("best_single_layer_sil", 0)
    n_words = e2.get("n_words", 0)
    
    print(f"\n--- {m} ({n_words} words) ---")
    print(f"  轨迹聚类 sil={sil_traj:.4f} vs 最佳单层 sil={best_sil:.4f} (L{best_layer})")
    print(f"  轨迹聚类优势: {sil_traj/best_sil:.2f}x" if best_sil > 0 else "  N/A")
    
    cat_dists = e2.get("category_distances", {})
    for cat, dists in cat_dists.items():
        intra = dists.get("intra_mean", 0)
        inter = dists.get("inter_mean", 0)
        ratio = inter / intra if intra > 0 else 0
        print(f"    {cat}: intra={intra:.4f}, inter={inter:.4f}, ratio={ratio:.2f}")

# === Exp 3: MLP神经元选择性 ===
print("\n" + "=" * 70)
print("Exp 3: MLP神经元选择性")
print("=" * 70)

for m in models:
    if m not in all_data or "exp3_mlp_neuron_selectivity" not in all_data[m]:
        continue
    e3 = all_data[m]["exp3_mlp_neuron_selectivity"]
    
    print(f"\n--- {m} ---")
    for li in sorted(e3.keys()):
        layer_data = e3[li]
        if not isinstance(layer_data, dict) or "top_selective_neurons" not in layer_data:
            continue
        
        gate_mean = layer_data.get("gate_mean", 0)
        gate_std = layer_data.get("gate_std", 0)
        print(f"  L{li}: gate_mean={gate_mean:.4f}, gate_std={gate_std:.4f}")
        
        top_sel = layer_data.get("top_selective_neurons", {})
        overlap = layer_data.get("overlap_matrix", {})
        
        # 统计选择性neurons数量
        for cat, sel_data in top_sel.items():
            sel_vals = sel_data.get("top_selectivity_values", [])
            mean_sel = np.mean(sel_vals[:5]) if sel_vals else 0
            print(f"    {cat}: top5 mean selectivity={mean_sel:.4f}")
        
        # 重叠统计
        if overlap:
            mean_overlap = np.mean(list(overlap.values()))
            max_overlap = max(overlap.values())
            print(f"    类间top-10 neurons重叠: mean={mean_overlap:.1f}, max={max_overlap}")

# === Exp 4: 轨迹分叉 ===
print("\n" + "=" * 70)
print("Exp 4: 轨迹分叉深度分析")
print("=" * 70)

for m in models:
    if m not in all_data or "exp4_trajectory_divergence" not in all_data[m]:
        continue
    e4 = all_data[m]["exp4_trajectory_divergence"]
    
    summary = e4.get("_summary", {})
    if "error" in summary:
        print(f"\n--- {m}: {summary['error']} ---")
        continue
    
    print(f"\n--- {m} ---")
    print(f"  min_cos 层分布: mean={summary.get('min_cos_layer_stats', {}).get('mean', 0):.1f}")
    print(f"  max_change 层分布: mean={summary.get('max_change_layer_stats', {}).get('mean', 0):.1f}")
    
    # 每个词的分叉层
    for word in sorted(e4.keys()):
        if word.startswith("_"):
            continue
        word_data = e4[word]
        if "error" in word_data:
            continue
        min_layer = word_data.get("min_cos_layer", -1)
        min_cos = word_data.get("min_cos_value", 0)
        print(f"    '{word}': 分叉层L{min_layer}, min_cos={min_cos:.4f}")

# === Exp 5: 动力学邻接性 ===
print("\n" + "=" * 70)
print("Exp 5: 动力学邻接性")
print("=" * 70)

for m in models:
    if m not in all_data or "exp5_dynamical_adjacency" not in all_data[m]:
        continue
    e5 = all_data[m]["exp5_dynamical_adjacency"]
    
    rel = e5.get("related_pairs_summary", {})
    unrel = e5.get("unrelated_pairs_summary", {})
    
    print(f"\n--- {m} ---")
    print(f"  相关词对: L0 cos={rel.get('L0_cos_mean', 0):.4f}, "
          f"Lmid cos={rel.get('Lmid_cos_mean', 0):.4f}, "
          f"Llast cos={rel.get('Llast_cos_mean', 0):.4f}")
    print(f"  无关词对: L0 cos={unrel.get('L0_cos_mean', 0):.4f}, "
          f"Lmid cos={unrel.get('Lmid_cos_mean', 0):.4f}, "
          f"Llast cos={unrel.get('Llast_cos_mean', 0):.4f}")
    
    # 语义区分度
    l0_diff = rel.get('L0_cos_mean', 0) - unrel.get('L0_cos_mean', 0)
    lmid_diff = rel.get('Lmid_cos_mean', 0) - unrel.get('Lmid_cos_mean', 0)
    llast_diff = rel.get('Llast_cos_mean', 0) - unrel.get('Llast_cos_mean', 0)
    max_disc_layer = e5.get("max_discrimination_layer", -1)
    
    print(f"  语义区分度: L0={l0_diff:.4f}, Lmid={lmid_diff:.4f}, Llast={llast_diff:.4f}")
    print(f"  最大区分层: L{max_disc_layer}")

print("\n" + "=" * 70)
print("汇总完成!")
