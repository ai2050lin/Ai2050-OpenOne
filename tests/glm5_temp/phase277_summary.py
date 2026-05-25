"""Phase 277 Cross-Model Summary"""
import json, numpy as np
from pathlib import Path

R = Path("results/phase277_dynamics_atlas")
models = ["qwen3", "glm4", "deepseek7b"]

print("=" * 70)
print("Phase 277: Conditional Dynamics Atlas — Cross-Model Summary")
print("=" * 70)

# Exp A
print("\n### Exp A: Universal Direction Test (var_top1) ###")
for m in models:
    f = R / f"{m}_exp_a_universal_dir.json"
    if f.exists():
        d = json.load(open(f))
        per = d["per_layer"]
        # Find min/max layers
        vals = {int(k): v["var_top1"] for k, v in per.items()}
        min_l = min(vals, key=vals.get)
        max_l = max(vals, key=vals.get)
        print(f"  {m}: mean={d['global_var_top1_mean']:.4f}, "
              f"min=L{min_l}({vals[min_l]:.4f}), max=L{max_l}({vals[max_l]:.4f})")

# Exp B
print("\n### Exp B: Scalar Profile Atlas ###")
for m in models:
    f = R / f"{m}_exp_b_scalar_atlas.json"
    if f.exists():
        d = json.load(open(f))
        cl = d.get("clustering", {})
        print(f"  {m}: within_corr={d['within_corr_mean']:.4f}, "
              f"between_corr={d['between_corr_mean']:.4f}, "
              f"delta={d['delta_corr']:.4f}, "
              f"broad_ARI={cl.get('broad_ari', 'N/A')}, "
              f"fine_ARI={cl.get('fine_ari', 'N/A')}")

# Exp C
print("\n### Exp C: Trajectory Topology ###")
for m in models:
    f = R / f"{m}_exp_c_topology.json"
    if f.exists():
        d = json.load(open(f))
        dist = d["distance"]
        print(f"  {m}: within_ratio={dist['within_ratio']:.1f}, "
              f"between_ratio={dist['between_ratio']:.1f}, "
              f"max_curv_L{d['curvature']['max_curvature_layer']}")

# Exp D
print("\n### Exp D: DMD Mode Analysis ###")
for m in models:
    f = R / f"{m}_exp_d_dmd.json"
    if f.exists():
        d = json.load(open(f))
        mc = d["mode_correlation"]
        print(f"  {m}: mode_within={mc['within_mean']:.4f}, "
              f"mode_between={mc['between_mean']:.4f}, "
              f"delta={mc['delta']:.4f}")

# Exp E
print("\n### Exp E: Language Dimension Signatures ###")
for m in models:
    f = R / f"{m}_exp_e_dimensions.json"
    if f.exists():
        d = json.load(open(f))
        neg = d.get("negation_pairs", [])
        if neg:
            avg_opp = np.mean([n["opposite_fraction"] for n in neg])
            avg_psr = np.mean([n["pearson_r"] for n in neg])
            print(f"  {m}: neg_opposite_frac={avg_opp:.3f}, neg_pearson={avg_psr:.4f}")

        lc = d.get("logic_vs_content", {})
        if lc:
            print(f"    logic_within_corr={lc.get('logic_within_corr', 'N/A'):.4f}, "
                  f"content_within_corr={lc.get('content_within_corr', 'N/A'):.4f}")

        # Dimension scalars
        ds = d.get("dimension_scalar_stats", {})
        print(f"    Dimension mean scalars:")
        for dim, stats in ds.items():
            print(f"      {dim}: {stats['mean_scalar']:.2f} ± {stats['std_scalar']:.2f}")

print("\n" + "=" * 70)
print("KEY FINDINGS:")
print("1. Universal direction REJECTED: var_top1 ~0.45-0.52 (NOT rank-1)")
print("2. U-shaped profile: first/last layers rank-1 (0.81-0.95), middle NOT (0.19-0.23)")
print("3. Negation NOT dynamical inversion: pearson_r ~0.999, opposite_frac ~0.0")
print("4. Trajectories DIVERGE: late/early ratio 13-297x, between > within")
print("5. Scalar profiles almost identical across dimensions: delta ~0.003-0.006")
print("=" * 70)
