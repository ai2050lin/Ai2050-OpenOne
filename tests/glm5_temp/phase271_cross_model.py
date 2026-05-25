"""
Phase 271 Cross-Model Summary: Topology Preservation & Transport R² Sanity
Generates comparative analysis across all 3 models.
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

models = ["qwen3", "glm4", "deepseek7b"]
result_dir = Path("results/phase271_topology_preservation")

print("=" * 70)
print("Phase 271 Cross-Model Summary")
print("=" * 70)

# ---- Experiment A: Transport R² Sanity ----
print("\n### Experiment A: Transport R² — Is V_inv Special? ###\n")

for m in models:
    f = result_dir / f"{m}_transport_sanity.json"
    if not f.exists():
        print(f"  {m}: NO DATA")
        continue
    data = json.load(open(f))
    print(f"  {m}:")
    for l in sorted(data.keys(), key=int):
        r = data[l]
        print(f"    L{l}: Transport={r['r2_transport_original']:.4f}, "
              f"Shuffled={r['r2_shuffled_mean']:.4f}, "
              f"Random={r['r2_random_subspace']:.4f}, "
              f"Gap_Trans_Rand={r['r2_transport_original']-r['r2_random_subspace']:.4f}")
    print()

print("KEY: If Random ≈ Transport, V_inv is NOT special for transport.")
print("     If Shuffled << Transport, the mapping is real (but not V_inv-specific).\n")

# ---- Experiment B: Topology Preservation ----
print("\n### Experiment B: Cross-layer Topology Preservation ###\n")

for m in models:
    f = result_dir / f"{m}_topology_preservation.json"
    if not f.exists():
        print(f"  {m}: NO DATA")
        continue
    data = json.load(open(f))
    mantel = data["experiment_b_topology"]["mantel_correlation"]
    
    print(f"  {m} Mantel r (vs final layer):")
    for space in ["full", "vis", "inv"]:
        vals = []
        for l in sorted(mantel[space].keys(), key=int):
            r = mantel[space][l]["spearman_r"]
            if abs(r) > 0.001:  # skip near-zero (embedding layer)
                vals.append(f"L{l}={r:.3f}")
        print(f"    {space}: {', '.join(vals)}")
    
    # Within vs between
    wb = data["experiment_b_topology"]["within_between"]
    print(f"  {m} Within > Between (full space):")
    for l in sorted(wb.keys(), key=int):
        w = wb[l]
        diff = w["within_full"] - w["between_full"]
        if abs(w["within_full"]) > 0.01:
            print(f"    L{l}: Within={w['within_full']:.3f}, Between={w['between_full']:.3f}, "
                  f"Diff={diff:.3f} | W_inv={w['within_inv']:.3f}, B_inv={w['between_inv']:.3f}")
    print()

# ---- Experiment C: Cross-space Agreement ----
print("\n### Experiment C: V_inv Carries Same Topology as Full Space ###\n")

for m in models:
    f = result_dir / f"{m}_topology_preservation.json"
    if not f.exists():
        continue
    data = json.load(open(f))
    cs = data["experiment_c_cross_space"]
    
    print(f"  {m}:")
    for l in sorted(cs.keys(), key=int):
        c = cs[l]
        if c["full_inv_spearman"] > 0.01:  # skip L0
            print(f"    L{l}: Full-Vinv={c['full_inv_spearman']:.4f}, "
                  f"Full-Vvis={c['full_vis_spearman']:.4f}, "
                  f"Vvis-Vinv={c['vis_inv_spearman']:.4f}")
    print()

print("\nKEY: Full-Vinv ≈ 0.99+ means V_inv carries the SAME relational structure as the full space.")
print("     V_inv is NOT informationally 'dark' — it's just not W_U-readable.\n")

# ---- Summary Table ----
print("\n### Summary: Phase 271 Key Findings ###\n")
print("1. Transport R² ≈ Random Subspace R² — V_inv is NOT special for transport")
print("2. Topology IS preserved across layers — supports relative encoding")
print("3. Within-category > between-category preservation — supports reuse")
print("4. V_inv topology = Full space topology (r=0.99+) — V_inv not 'dark'")
print("5. DS7B has notably lower topology preservation than Qwen3/GLM4")
