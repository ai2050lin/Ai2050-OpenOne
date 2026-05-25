"""Phase 278 Summary — Cross-model comparison"""
import sys, json, numpy as np
from pathlib import Path

RESULT_DIR = Path("results/phase278_global_dynamics")

MODELS = ["qwen3", "glm4", "deepseek7b"]

print("=" * 70)
print("Phase 278: Global Language Dynamics Mapping — Cross-model Summary")
print("=" * 70)

# Block A: Bifurcation
print("\n### Block A: Trajectory Bifurcation ###")
for m in MODELS:
    p = RESULT_DIR / f"{m}_block_a_bifurcation.json"
    if p.exists():
        d = json.loads(p.read_text())
        print(f"\n  {m}:")
        print(f"    bifurcation_broad_layer: {d.get('bifurcation_broad_layer')}")
        print(f"    bifurcation_fine_layer: {d.get('bifurcation_fine_layer')}")
        print(f"    peak_broad_ari: L{d['peak_broad_ari_layer']} = {d['peak_broad_ari_value']:.4f}")
        print(f"    peak_fine_ari: L{d['peak_fine_ari_layer']} = {d['peak_fine_ari_value']:.4f}")
        # Show ARI at key layers
        pl = d.get("per_layer", {})
        for lk in sorted(pl.keys(), key=lambda x: int(x)):
            l = int(lk)
            if l in [0, 5, 9, 13, 18, 27, 36, 40] or lk in pl:
                if l in [0, 9, 13, 18, 27, 36, 40]:
                    v = pl[lk]
                    print(f"    L{l}: ari_broad={v['ari_broad']:.3f}, ari_fine={v['ari_fine']:.3f}, "
                          f"sim_delta={v.get('sim_delta', 'N/A')}")

# Block B: Context
print("\n### Block B: Context-Dependent Dynamics ###")
for m in MODELS:
    p = RESULT_DIR / f"{m}_block_b_context.json"
    if p.exists():
        d = json.loads(p.read_text())
        ls = d.get("layer_sensitivity", {})
        print(f"\n  {m}:")
        # Show sensitivity at key layers
        n_layers = max(int(k) for k in ls.keys()) if ls else 0
        for l in [0, 1, 5, 9, n_layers//2, n_layers-5, n_layers]:
            if str(l) in ls:
                v = ls[str(l)]
                print(f"    L{l}: sensitivity={v['mean_sensitivity']:.4f}, "
                      f"from_baseline={v.get('mean_dist_from_baseline', 'N/A')}")
        # Dimension sensitivity
        ds = d.get("dim_sensitivity", {})
        print(f"    Dimension sensitivities:")
        for dim in sorted(ds.keys()):
            print(f"      {dim}: {ds[dim]['mean_sensitivity']:.4f}")

# Block C: Multi-Direction
print("\n### Block C: Multi-Direction Spectrum ###")
for m in MODELS:
    p = RESULT_DIR / f"{m}_block_c_multidirection.json"
    if p.exists():
        d = json.loads(p.read_text())
        print(f"\n  {m}:")
        print(f"    var_top1={d['mean_var_top1']:.4f}")
        print(f"    var_top3={d['mean_var_top3']:.4f}")
        print(f"    var_top5={d['mean_var_top5']:.4f}")
        print(f"    n_significant_dirs={d['mean_n_significant_dirs']:.1f}")
        print(f"    max_multi_dir: L{d['max_significant_dirs_layer']} "
              f"({d['max_significant_dirs_value']} dirs)")
        # Show key layers
        pl = d.get("per_layer", {})
        n_layers = max(int(k) for k in pl.keys()) if pl else 0
        for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
            if str(l) in pl:
                v = pl[str(l)]
                vtk = v.get("var_topK", {})
                print(f"    L{l}: var_top1={vtk.get('1','N/A'):.3f}, "
                      f"var_top3={vtk.get('3','N/A'):.3f}, "
                      f"var_top5={vtk.get('5','N/A'):.3f}, "
                      f"n_sig={v['n_dirs_significant']}")

# Block D: Basin Radius
print("\n### Block D: Attractor Basin Radius ###")
for m in MODELS:
    p = RESULT_DIR / f"{m}_block_d_basin.json"
    if p.exists():
        d = json.loads(p.read_text())
        br = d.get("basin_radius_per_layer", {})
        print(f"\n  {m}:")
        for lk in sorted(br.keys(), key=lambda x: int(x)):
            v = br[lk]
            print(f"    L{lk}: mean={v['mean']:.4f}, range=[{v['min']:.4f}, {v['max']:.4f}]")
        # Show specific deviation data
        bd = d.get("basin_data", {})
        if "dog" in bd:
            print(f"    'dog' deviation profile:")
            for lk in sorted(bd["dog"].keys(), key=lambda x: int(x)):
                ld = bd["dog"][lk]
                dev_001 = ld.get("0.01", {}).get("relative_deviation", "N/A")
                dev_1 = ld.get("1.0", {}).get("relative_deviation", "N/A")
                if dev_001 != "N/A":
                    print(f"      L{lk}: dev(0.01)={dev_001:.4f}, dev(1.0)={dev_1:.4f}")

print("\n" + "=" * 70)
print("Phase 278 Summary Complete")
