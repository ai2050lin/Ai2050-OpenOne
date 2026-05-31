"""
Phase 299 Cross-Model Analysis: Direction-Norm & Role-Frame Bundle
===================================================================
Summarize key findings across Qwen3, GLM4, DS7B
"""
import json, os
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results/phase299_dir_norm_bundle")
models = {
    "qwen3": {"mid": 18, "name": "Qwen3-8B"},
    "glm4":  {"mid": 20, "name": "GLM4-9B"},
    "deepseek7b": {"mid": 14, "name": "DS7B"},
}

def load_results(model_key):
    fp = RESULT_DIR / f"{model_key}_dir_norm_bundle.json"
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_part_a(data, mid_layer):
    """Direction-Norm Dual-Channel analysis"""
    pa = data["part_a_dir_norm"]
    layer_data = pa.get(str(mid_layer), {})
    
    results = {
        "full_correct": [], "dir_only_avg_norm": [], "norm_gate_rand_dir": [],
        "dir_unit_norm": [], "rp_full": [], "rp_dir_avg_norm": [], "avg_random": []
    }
    
    for key, vals in layer_data.items():
        results["full_correct"].append(vals["full_correct_cos_shift"])
        results["dir_only_avg_norm"].append(vals["dir_only_avg_norm_cos_shift"])
        results["norm_gate_rand_dir"].append(vals["norm_gate_rand_dir_cos_shift"])
        results["dir_unit_norm"].append(vals["dir_unit_norm_cos_shift"])
        results["rp_full"].append(vals["rp_full_cos_shift"])
        results["rp_dir_avg_norm"].append(vals["rp_dir_avg_norm_cos_shift"])
        results["avg_random"].append(vals["avg_random_shift"])
    
    summary = {}
    for k, v in results.items():
        arr = np.array(v)
        summary[k] = {
            "mean": float(np.mean(arr)),
            "pos_rate": float(np.mean(arr > 0)),
        }
    
    # Key metrics
    dir_effect = summary["dir_only_avg_norm"]["mean"]
    norm_gate_effect = summary["norm_gate_rand_dir"]["mean"]
    full_effect = summary["full_correct"]["mean"]
    
    # Direction dominance ratio
    if abs(norm_gate_effect) > 0.001:
        dir_dominance = abs(dir_effect) / abs(norm_gate_effect)
    else:
        dir_dominance = float('inf')
    
    # rp (role-pair specific) advantage
    rp_advantage = summary["rp_full"]["mean"] / abs(full_effect) if abs(full_effect) > 0.001 else 0
    
    return {
        "full_correct": summary["full_correct"],
        "dir_only": summary["dir_only_avg_norm"],
        "norm_gate": summary["norm_gate_rand_dir"],
        "dir_unit_norm": summary["dir_unit_norm"],
        "rp_full": summary["rp_full"],
        "dir_dominance": dir_dominance,
        "rp_advantage": rp_advantage,
        "per_token": layer_data,
    }

def analyze_part_b(data, mid_layer):
    """Role-Frame Bundle analysis"""
    pb = data["part_b_role_frame_bundle"]
    layer_data = pb.get(str(mid_layer), {})
    
    results = {
        "role_only": [], "frame_only": [], "role_plus_frame": [],
        "loo_role_only": [], "loo_role_plus_frame": [], "random": []
    }
    
    for key, vals in layer_data.items():
        results["role_only"].append(vals["role_only_cos_shift"])
        results["frame_only"].append(vals["frame_only_cos_shift"])
        results["role_plus_frame"].append(vals["role_plus_frame_cos_shift"])
        results["loo_role_only"].append(vals["loo_role_only_cos_shift"])
        results["loo_role_plus_frame"].append(vals["loo_role_plus_frame_cos_shift"])
        results["random"].append(vals["avg_random_shift"])
    
    summary = {}
    for k, v in results.items():
        arr = np.array(v)
        summary[k] = {
            "mean": float(np.mean(arr)),
            "pos_rate": float(np.mean(arr > 0)),
        }
    
    # Bundle effect
    role_only_mean = abs(summary["role_only"]["mean"])
    rpf_mean = abs(summary["role_plus_frame"]["mean"])
    bundle_ratio = rpf_mean / role_only_mean if role_only_mean > 0.001 else 0
    
    # LOO comparison
    loo_role = summary["loo_role_only"]["mean"]
    loo_rpf = summary["loo_role_plus_frame"]["mean"]
    
    return {
        "role_only": summary["role_only"],
        "frame_only": summary["frame_only"],
        "role_plus_frame": summary["role_plus_frame"],
        "loo_role_only": summary["loo_role_only"],
        "loo_role_plus_frame": summary["loo_role_plus_frame"],
        "bundle_ratio": bundle_ratio,
        "loo_role": loo_role,
        "loo_rpf": loo_rpf,
        "per_token": layer_data,
    }

def per_role_pair_analysis(data, mid_layer):
    """Analyze by role pair type"""
    pa = data["part_a_dir_norm"]
    pb = data["part_b_role_frame_bundle"]
    
    rp_results = defaultdict(lambda: {"dir_only": [], "norm_gate": [], "rp_full": [],
                                       "role_only": [], "frame_only": [], "rpf": []})
    
    for part, prefix in [(pa, "a"), (pb, "b")]:
        layer_data = part.get(str(mid_layer), {})
        for key, vals in layer_data.items():
            rp = vals.get("role_pair", "")
            if not rp:
                # Extract from key
                if "adj->verb" in key: rp = "adj_verb"
                elif "adj->noun" in key: rp = "adj_noun"
                elif "noun->verb" in key: rp = "noun_verb"
            
            if prefix == "a":
                rp_results[rp]["dir_only"].append(vals.get("dir_only_avg_norm_cos_shift", 0))
                rp_results[rp]["norm_gate"].append(vals.get("norm_gate_rand_dir_cos_shift", 0))
                rp_results[rp]["rp_full"].append(vals.get("rp_full_cos_shift", 0))
            else:
                rp_results[rp]["role_only"].append(vals.get("role_only_cos_shift", 0))
                rp_results[rp]["frame_only"].append(vals.get("frame_only_cos_shift", 0))
                rp_results[rp]["rpf"].append(vals.get("role_plus_frame_cos_shift", 0))
    
    summary = {}
    for rp, vals in rp_results.items():
        summary[rp] = {
            "dir_only_mean": float(np.mean(vals["dir_only"])) if vals["dir_only"] else 0,
            "norm_gate_mean": float(np.mean(vals["norm_gate"])) if vals["norm_gate"] else 0,
            "rp_full_mean": float(np.mean(vals["rp_full"])) if vals["rp_full"] else 0,
            "role_only_mean": float(np.mean(vals["role_only"])) if vals["role_only"] else 0,
            "frame_only_mean": float(np.mean(vals["frame_only"])) if vals["frame_only"] else 0,
            "rpf_mean": float(np.mean(vals["rpf"])) if vals["rpf"] else 0,
        }
    return summary

print("=" * 70)
print("Phase 299 Cross-Model Analysis: Direction-Norm & Role-Frame Bundle")
print("=" * 70)

all_results = {}
for mk, info in models.items():
    data = load_results(mk)
    mid = info["mid"]
    
    pa = analyze_part_a(data, mid)
    pb = analyze_part_b(data, mid)
    rp = per_role_pair_analysis(data, mid)
    
    all_results[mk] = {"part_a": pa, "part_b": pb, "per_rp": rp}

# ========== PART A SUMMARY ==========
print("\n" + "=" * 70)
print("PART A: Direction-Norm Dual-Channel (Mid Layer)")
print("=" * 70)

for mk, info in models.items():
    r = all_results[mk]["part_a"]
    print(f"\n--- {info['name']} (L{models[mk]['mid']}) ---")
    print(f"  full_correct:    avg={r['full_correct']['mean']:+.4f}  pos_rate={r['full_correct']['pos_rate']:.0%}")
    print(f"  dir_only:        avg={r['dir_only']['mean']:+.4f}  pos_rate={r['dir_only']['pos_rate']:.0%}")
    print(f"  norm_gate_rand:  avg={r['norm_gate']['mean']:+.4f}  pos_rate={r['norm_gate']['pos_rate']:.0%}")
    print(f"  dir_unit_norm:   avg={r['dir_unit_norm']['mean']:+.4f}  pos_rate={r['dir_unit_norm']['pos_rate']:.0%}")
    print(f"  rp_specific:     avg={r['rp_full']['mean']:+.4f}  pos_rate={r['rp_full']['pos_rate']:.0%}")
    print(f"  --> Direction dominance: {r['dir_dominance']:.1f}x (dir_only / |norm_gate|)")
    print(f"  --> RP-specific advantage: {r['rp_advantage']:.2f}x (rp_full / full_correct)")

# ========== PART B SUMMARY ==========
print("\n" + "=" * 70)
print("PART B: Role-Frame Bundle Causal Test (Mid Layer)")
print("=" * 70)

for mk, info in models.items():
    r = all_results[mk]["part_b"]
    print(f"\n--- {info['name']} (L{models[mk]['mid']}) ---")
    print(f"  role_only:       avg={r['role_only']['mean']:+.4f}  pos_rate={r['role_only']['pos_rate']:.0%}")
    print(f"  frame_only:      avg={r['frame_only']['mean']:+.4f}  pos_rate={r['frame_only']['pos_rate']:.0%}")
    print(f"  role_plus_frame: avg={r['role_plus_frame']['mean']:+.4f}  pos_rate={r['role_plus_frame']['pos_rate']:.0%}")
    print(f"  loo_role_only:   avg={r['loo_role_only']['mean']:+.4f}  pos_rate={r['loo_role_only']['pos_rate']:.0%}")
    print(f"  loo_rpf:         avg={r['loo_role_plus_frame']['mean']:+.4f}  pos_rate={r['loo_role_plus_frame']['pos_rate']:.0%}")
    print(f"  --> Bundle ratio: {r['bundle_ratio']:.2f}x (|rpf| / |role_only|)")
    print(f"  --> LOO comparison: role_only={r['loo_role']:+.4f} vs rpf={r['loo_rpf']:+.4f}")

# ========== PER ROLE-PAIR ANALYSIS ==========
print("\n" + "=" * 70)
print("Per Role-Pair Analysis (Mid Layer)")
print("=" * 70)

for rp in ["adj_verb", "adj_noun", "noun_verb"]:
    print(f"\n--- {rp} ---")
    for mk, info in models.items():
        rp_data = all_results[mk]["per_rp"].get(rp, {})
        dir_only = rp_data.get("dir_only_mean", 0)
        norm_gate = rp_data.get("norm_gate_mean", 0)
        role_only = rp_data.get("role_only_mean", 0)
        frame_only = rp_data.get("frame_only_mean", 0)
        rpf = rp_data.get("rpf_mean", 0)
        print(f"  {info['name']:8s}: dir_only={dir_only:+.4f} norm_gate={norm_gate:+.4f} | "
              f"role={role_only:+.4f} frame={frame_only:+.4f} rpf={rpf:+.4f}")

# ========== KEY CONCLUSIONS ==========
print("\n" + "=" * 70)
print("KEY CONCLUSIONS")
print("=" * 70)

# Part A: Direction vs Norm
print("\n1. Direction-Norm Dual Channel:")
for mk, info in models.items():
    r = all_results[mk]["part_a"]
    dir_eff = r["dir_only"]["mean"]
    norm_eff = r["norm_gate"]["mean"]
    if abs(dir_eff) > abs(norm_eff):
        channel = "DIRECTION-DOMINANT"
    elif abs(norm_eff) > abs(dir_eff):
        channel = "NORM-DOMINANT"
    else:
        channel = "COUPLED"
    print(f"  {info['name']:8s}: {channel} (dir={dir_eff:+.4f}, norm_gate={norm_eff:+.4f}, dom={r['dir_dominance']:.1f}x)")

# Part B: Bundle effect
print("\n2. Role-Frame Bundle Effect:")
for mk, info in models.items():
    r = all_results[mk]["part_b"]
    role = r["role_only"]["mean"]
    rpf = r["role_plus_frame"]["mean"]
    ratio = r["bundle_ratio"]
    if ratio > 1.1:
        verdict = "BUNDLE > ROLE (frame helps)"
    elif ratio < 0.9 and rpf < 0:
        verdict = "BUNDLE FAILS (negative!)"
    elif ratio < 0.9:
        verdict = "ROLE > BUNDLE (frame hurts)"
    else:
        verdict = "SIMILAR"
    print(f"  {info['name']:8s}: {verdict} (role={role:+.4f}, rpf={rpf:+.4f}, ratio={ratio:.2f}x)")

print("\n3. DS7B Specific Pattern:")
ds7b_a = all_results["deepseek7b"]["part_a"]
ds7b_b = all_results["deepseek7b"]["part_b"]
print(f"  dir_only:        {ds7b_a['dir_only']['mean']:+.4f} ({ds7b_a['dir_only']['pos_rate']:.0%} positive)")
print(f"  norm_gate_rand:  {ds7b_a['norm_gate']['mean']:+.4f} ({ds7b_a['norm_gate']['pos_rate']:.0%} positive)")
print(f"  role_only:       {ds7b_b['role_only']['mean']:+.4f} ({ds7b_b['role_only']['pos_rate']:.0%} positive)")
print(f"  role+frame:      {ds7b_b['role_plus_frame']['mean']:+.4f} ({ds7b_b['role_plus_frame']['pos_rate']:.0%} positive)")
print(f"  loo_role_only:   {ds7b_b['loo_role_only']['mean']:+.4f} ({ds7b_b['loo_role_only']['pos_rate']:.0%} positive)")
