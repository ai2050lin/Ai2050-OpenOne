"""Phase 187 Cross-Model Comparison"""
import json, glob
import numpy as np

files = sorted(glob.glob('tests/glm5_temp/phase187_*.json'))
models = {}
for f in files:
    d = json.load(open(f, 'r', encoding='utf-8'))
    mn = d['model']
    models[mn] = d

print("=" * 80)
print("PHASE 187: CROSS-MODEL COMPARISON")
print("=" * 80)

# ===== Exp1: Difference Amplification Spectrum =====
print("\n★★★ Exp1: Difference Amplification Spectrum ★★★")
diff_types = ["category", "subordinate", "syntactic", "paraphrase", "random_control"]
metrics = ["cumul_amp_mean", "cos_dir_mean", "energy_in_pc_mean", "energy_top5_mean"]

for mn in ["qwen3", "glm4", "deepseek7b"]:
    if mn not in models:
        continue
    d = models[mn]
    exp1 = d.get("exp1_diff_amplification_spectrum", {})
    by_type = exp1.get("by_type", {})
    sample_layers = exp1.get("sample_layers", [])
    last_li = str(sample_layers[-1]) if sample_layers else "0"

    print(f"\n--- {mn} (L0→L{last_li}) ---")
    print(f"  {'Type':20s} {'cumul_amp':>12s} {'cos_dir':>10s} {'E_pc':>8s} {'E_top5':>8s}")

    for dt in diff_types:
        if dt in by_type and last_li in by_type[dt]:
            agg = by_type[dt][last_li]
            ca = agg.get("cumul_amp_mean", 0)
            cd = agg.get("cos_dir_mean", 0)
            epc = agg.get("energy_in_pc_mean", 0)
            e5 = agg.get("energy_top5_mean", 0)
            print(f"  {dt:20s} {ca:12.1f} {cd:10.4f} {epc:8.3f} {e5:8.3f}")

    stat = exp1.get("stat_test", {})
    if stat:
        print(f"  ★ p-value: {stat.get('p_value', 1):.4f} → {stat.get('verdict', 'N/A')}")

# ===== Exp1: Norm analysis =====
print("\n\n★★★ Norm Scaling Analysis (||Δ_final|| vs ||Δ_initial||) ★★★")
for mn in ["qwen3", "glm4", "deepseek7b"]:
    if mn not in models:
        continue
    d = models[mn]
    exp1 = d.get("exp1_diff_amplification_spectrum", {})
    by_type = exp1.get("by_type", {})
    sample_layers = exp1.get("sample_layers", [])
    first_li = str(sample_layers[0]) if sample_layers else "0"
    last_li = str(sample_layers[-1]) if sample_layers else "0"

    print(f"\n--- {mn} ---")
    print(f"  {'Type':20s} {'||Δ_0||':>10s} {'||Δ_L||':>10s} {'amp':>8s} {'Δ_L/√Δ_0':>10s}")

    for dt in diff_types:
        if dt in by_type and first_li in by_type[dt] and last_li in by_type[dt]:
            n0 = by_type[dt][first_li].get("norm_mean", 0)
            nl = by_type[dt][last_li].get("norm_mean", 0)
            amp = nl / max(n0, 1e-10) if n0 > 0.01 else 0
            normed = nl / max(np.sqrt(n0), 1e-10) if n0 > 0.01 else 0
            print(f"  {dt:20s} {n0:10.4f} {nl:10.4f} {amp:8.1f} {normed:10.1f}")

# ===== Exp2: Direction-Selective Jacobian =====
print("\n\n★★★ Exp2: Direction-Selective Jacobian ★★★")
for mn in ["qwen3", "glm4", "deepseek7b"]:
    if mn not in models:
        continue
    d = models[mn]
    exp2 = d.get("exp2_direction_selective_jacobian", {})
    comp = exp2.get("_comparison", {})
    if comp:
        print(f"  {mn}: semantic_g={comp.get('semantic_g_mean', 0):.3f}, random_g={comp.get('random_g_mean', 0):.3f}, "
              f"p={comp.get('p_value', 1):.4f} → {comp.get('verdict', 'N/A')}")

# ===== Exp3: Cross-Lingual Direction Alignment =====
print("\n\n★★★ Exp3: Cross-Lingual Direction Alignment ★★★")
for mn in ["qwen3", "glm4", "deepseek7b"]:
    if mn not in models:
        continue
    d = models[mn]
    exp3 = d.get("exp3_cross_lingual_direction_alignment", {})
    meta = exp3.get("_meta", {})
    first_cos = meta.get("cos_en_zh_first", 0)
    last_cos = meta.get("cos_en_zh_last", 0)
    slope = meta.get("cos_slope", 0)
    print(f"  {mn}: cos(Δ_en,Δ_zh) L0={first_cos:.4f} → L_last={last_cos:.4f} (slope={slope:.5f})")

    # Per-contrast
    per_c = exp3.get("_per_contrast", {})
    if per_c:
        # Sort by cos value
        sorted_c = sorted(per_c.items(), key=lambda x: x[1].get("cos", 0), reverse=True)
        print(f"    Top-3 aligned:")
        for cn, cd in sorted_c[:3]:
            print(f"      {cn}: cos={cd.get('cos', 0):.4f}")
        print(f"    Bottom-3 aligned:")
        for cn, cd in sorted_c[-3:]:
            print(f"      {cn}: cos={cd.get('cos', 0):.4f}")

# ===== Summary Table =====
print("\n\n" + "=" * 80)
print("SUMMARY TABLE: Key Metrics Across Models")
print("=" * 80)

# Energy in PC comparison — this is the KEY finding
print("\n★★★ Energy in Principal Subspace (E_pc) at Last Layer ★★★")
print(f"  {'Type':20s} {'Qwen3':>10s} {'GLM4':>10s} {'DS7B':>10s}")
for dt in diff_types:
    vals = []
    for mn in ["qwen3", "glm4", "deepseek7b"]:
        if mn in models:
            exp1 = models[mn].get("exp1_diff_amplification_spectrum", {})
            by_type = exp1.get("by_type", {})
            sample_layers = exp1.get("sample_layers", [])
            last_li = str(sample_layers[-1]) if sample_layers else "0"
            if dt in by_type and last_li in by_type[dt]:
                vals.append(f"{by_type[dt][last_li].get('energy_in_pc_mean', 0):.3f}")
            else:
                vals.append("N/A")
        else:
            vals.append("N/A")
    print(f"  {dt:20s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")

# Cumul amp comparison
print("\n★★★ Cumulative Amplification (||Δ_L||/||Δ_0||) at Last Layer ★★★")
print(f"  {'Type':20s} {'Qwen3':>10s} {'GLM4':>10s} {'DS7B':>10s}")
for dt in diff_types:
    vals = []
    for mn in ["qwen3", "glm4", "deepseek7b"]:
        if mn in models:
            exp1 = models[mn].get("exp1_diff_amplification_spectrum", {})
            by_type = exp1.get("by_type", {})
            sample_layers = exp1.get("sample_layers", [])
            last_li = str(sample_layers[-1]) if sample_layers else "0"
            if dt in by_type and last_li in by_type[dt]:
                vals.append(f"{by_type[dt][last_li].get('cumul_amp_mean', 0):.1f}")
            else:
                vals.append("N/A")
        else:
            vals.append("N/A")
    print(f"  {dt:20s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")

print("\nDone!")
