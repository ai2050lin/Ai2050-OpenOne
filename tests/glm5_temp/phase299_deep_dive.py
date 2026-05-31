"""
Phase 299 Deep Dive: DS7B Anomaly Analysis
============================================
Focus on:
1. DS7B adj_noun: role_only=-0.8253, rpf=-0.7459 (huge negative!)
2. DS7B role+frame negative while role_only positive
3. Per-token breakdown of DS7B bundle effect
"""
import json, os
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results/phase299_dir_norm_bundle")

def load_results(model_key):
    fp = RESULT_DIR / f"{model_key}_dir_norm_bundle.json"
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_ds7b_per_token():
    data = load_results("deepseek7b")
    
    print("=" * 70)
    print("DS7B Per-Token Part A Analysis (L14)")
    print("=" * 70)
    
    pa = data["part_a_dir_norm"]["14"]
    print(f"\n{'Token':>20s} {'Full':>8s} {'DirOnly':>8s} {'NormGate':>8s} {'DirUnit':>8s} {'RPFul':>8s}")
    print("-" * 70)
    
    for key in sorted(pa.keys()):
        v = pa[key]
        print(f"{key:>20s} {v['full_correct_cos_shift']:+.3f}  {v['dir_only_avg_norm_cos_shift']:+.3f}  "
              f"{v['norm_gate_rand_dir_cos_shift']:+.3f}  {v['dir_unit_norm_cos_shift']:+.3f}  "
              f"{v['rp_full_cos_shift']:+.3f}")
    
    print("\n" + "=" * 70)
    print("DS7B Per-Token Part B Analysis (L14)")
    print("=" * 70)
    
    pb = data["part_b_role_frame_bundle"]["14"]
    print(f"\n{'Token':>20s} {'RoleOnly':>8s} {'FrameOnly':>8s} {'RPF':>8s} {'LooRole':>8s} {'LooRPF':>8s} {'Rnorm':>7s} {'Fnorm':>7s}")
    print("-" * 90)
    
    for key in sorted(pb.keys()):
        v = pb[key]
        print(f"{key:>20s} {v['role_only_cos_shift']:+.3f}  {v['frame_only_cos_shift']:+.3f}  "
              f"{v['role_plus_frame_cos_shift']:+.3f}  {v['loo_role_only_cos_shift']:+.3f}  "
              f"{v['loo_role_plus_frame_cos_shift']:+.3f}  {v['R_norm']:.1f}  {v['F_norm']:.1f}")
    
    # Per role-pair summary
    print("\n" + "=" * 70)
    print("DS7B Per Role-Pair Bundle Analysis (L14)")
    print("=" * 70)
    
    rp_data = defaultdict(lambda: {"role_only": [], "frame_only": [], "rpf": [], "loo_role": [], "loo_rpf": []})
    for key, v in pb.items():
        if "adj->verb" in key: rp = "adj_verb"
        elif "adj->noun" in key: rp = "adj_noun"
        elif "noun->verb" in key: rp = "noun_verb"
        else: continue
        rp_data[rp]["role_only"].append(v["role_only_cos_shift"])
        rp_data[rp]["frame_only"].append(v["frame_only_cos_shift"])
        rp_data[rp]["rpf"].append(v["role_plus_frame_cos_shift"])
        rp_data[rp]["loo_role"].append(v["loo_role_only_cos_shift"])
        rp_data[rp]["loo_rpf"].append(v["loo_role_plus_frame_cos_shift"])
    
    for rp, vals in rp_data.items():
        print(f"\n--- {rp} ---")
        print(f"  role_only:  mean={np.mean(vals['role_only']):+.4f}  individual={[f'{x:+.3f}' for x in vals['role_only']]}")
        print(f"  frame_only: mean={np.mean(vals['frame_only']):+.4f}  individual={[f'{x:+.3f}' for x in vals['frame_only']]}")
        print(f"  rpf:        mean={np.mean(vals['rpf']):+.4f}  individual={[f'{x:+.3f}' for x in vals['rpf']]}")
        print(f"  loo_role:   mean={np.mean(vals['loo_role']):+.4f}  individual={[f'{x:+.3f}' for x in vals['loo_role']]}")
        print(f"  loo_rpf:    mean={np.mean(vals['loo_rpf']):+.4f}  individual={[f'{x:+.3f}' for x in vals['loo_rpf']]}")

def compare_bundle_all_layers():
    """Compare bundle effect across layers for all models"""
    for mk in ["qwen3", "glm4", "deepseek7b"]:
        data = load_results(mk)
        pb = data["part_b_role_frame_bundle"]
        
        print(f"\n{'='*70}")
        print(f"{mk.upper()} Bundle Effect Across Layers")
        print(f"{'='*70}")
        print(f"{'Layer':>6s} {'RoleOnly':>9s} {'FrameOnly':>9s} {'RPF':>9s} {'LooRole':>9s} {'LooRPF':>9s} {'Ratio':>7s}")
        print("-" * 60)
        
        for layer in sorted([int(x) for x in pb.keys()]):
            ld = pb[str(layer)]
            ro = np.mean([v["role_only_cos_shift"] for v in ld.values()])
            fo = np.mean([v["frame_only_cos_shift"] for v in ld.values()])
            rpf = np.mean([v["role_plus_frame_cos_shift"] for v in ld.values()])
            lro = np.mean([v["loo_role_only_cos_shift"] for v in ld.values()])
            lrpf = np.mean([v["loo_role_plus_frame_cos_shift"] for v in ld.values()])
            ratio = abs(rpf) / abs(ro) if abs(ro) > 0.001 else 0
            print(f"L{layer:>4d} {ro:+9.4f} {fo:+9.4f} {rpf:+9.4f} {lro:+9.4f} {lrpf:+9.4f} {ratio:7.2f}x")

def analyze_dir_norm_coupling():
    """Analyze direction-norm coupling for DS7B"""
    print("\n" + "=" * 70)
    print("Direction-Norm Coupling Analysis (All Models, Mid Layer)")
    print("=" * 70)
    
    models = {"qwen3": 18, "glm4": 20, "deepseek7b": 14}
    
    for mk, mid in models.items():
        data = load_results(mk)
        pa = data["part_a_dir_norm"][str(mid)]
        
        # Calculate coupling metric
        # If direction-norm coupled, then:
        # - dir_only (correct dir + avg norm) should work
        # - dir_unit_norm (unit dir + unit norm) should work less
        # - norm_gate (random dir + correct norm) should work less than dir_only
        
        full = [v["full_correct_cos_shift"] for v in pa.values()]
        dir_only = [v["dir_only_avg_norm_cos_shift"] for v in pa.values()]
        norm_gate = [v["norm_gate_rand_dir_cos_shift"] for v in pa.values()]
        dir_unit = [v["dir_unit_norm_cos_shift"] for v in pa.values()]
        
        # Coupling index = (dir_only - norm_gate) / (dir_only + norm_gate)
        # Positive = direction more important, Negative = norm more important
        dir_arr = np.array(dir_only)
        norm_arr = np.array(norm_gate)
        
        # Exclude cases where both are near zero
        mask = (np.abs(dir_arr) + np.abs(norm_arr)) > 0.01
        if mask.sum() > 0:
            coupling = (np.abs(dir_arr[mask]) - np.abs(norm_arr[mask])) / (np.abs(dir_arr[mask]) + np.abs(norm_arr[mask]))
            avg_coupling = np.mean(coupling)
        else:
            avg_coupling = 0
        
        print(f"\n--- {mk.upper()} L{mid} ---")
        print(f"  avg |dir_only| = {np.mean(np.abs(dir_arr)):.4f}")
        print(f"  avg |norm_gate| = {np.mean(np.abs(norm_arr)):.4f}")
        print(f"  avg |dir_unit| = {np.mean(np.abs(np.array(dir_unit))):.4f}")
        print(f"  Coupling index: {avg_coupling:+.3f} (+1=dir-only, -1=norm-only, 0=balanced)")

if __name__ == "__main__":
    analyze_ds7b_per_token()
    compare_bundle_all_layers()
    analyze_dir_norm_coupling()
