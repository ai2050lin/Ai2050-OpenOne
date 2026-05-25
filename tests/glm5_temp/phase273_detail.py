"""Phase 273 detailed analysis: Extract impact profiles and cross-pair correlations."""
import json
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

WITHIN_PAIRS = [
    ("apple", "banana"), ("apple", "orange"), ("dog", "cat"), ("lion", "tiger"), ("car", "bus"),
    ("banana", "grape"), ("wolf", "fox"), ("bike", "truck"), ("peach", "cherry"), ("train", "plane"),
]
BETWEEN_PAIRS = [
    ("apple", "dog"), ("banana", "car"), ("orange", "lion"), ("grape", "train"), ("mango", "bike"),
    ("peach", "wolf"), ("cherry", "fox"), ("lemon", "bus"), ("pear", "tiger"), ("lime", "ship"),
]

for model in ["qwen3", "glm4", "deepseek7b"]:
    path = Path(f"results/phase273_causal_path/{model}_exp_a.json")
    if not path.exists():
        continue
    with open(path) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"  {model} — Causal Importance Map (Residual Noise)")
    print(f"{'='*70}")
    layers = sorted([int(l) for l in data["apple"].keys()])

    # Print impact profiles for key words
    sample_layers = layers[:15]
    header = f"{'Word':<12}" + "".join([f"L{l:>2}  " for l in sample_layers])
    print(header)
    for w in ["apple", "banana", "orange", "dog", "cat", "lion", "car", "bus", "train"]:
        if w in data:
            vals = [f"{data[w].get(str(l), {}).get('abs_impact', 0):>5.1f}" for l in sample_layers]
            print(f"{w:<12}" + "  ".join(vals))

    # Compute pairwise impact profile correlations
    within_corr = []
    between_corr = []
    
    for a, b in WITHIN_PAIRS:
        if a in data and b in data:
            a_vals = [data[a].get(str(l), {}).get("abs_impact", 0) for l in layers]
            b_vals = [data[b].get(str(l), {}).get("abs_impact", 0) for l in layers]
            r, _ = spearmanr(a_vals, b_vals)
            within_corr.append(r)

    for a, b in BETWEEN_PAIRS:
        if a in data and b in data:
            a_vals = [data[a].get(str(l), {}).get("abs_impact", 0) for l in layers]
            b_vals = [data[b].get(str(l), {}).get("abs_impact", 0) for l in layers]
            r, _ = spearmanr(a_vals, b_vals)
            between_corr.append(r)

    print(f"\n  Within-category impact profile corr:  {np.mean(within_corr):.4f} ± {np.std(within_corr):.4f} (n={len(within_corr)})")
    print(f"  Between-category impact profile corr: {np.mean(between_corr):.4f} ± {np.std(between_corr):.4f} (n={len(between_corr)})")
    print(f"  Delta (Within - Between): {np.mean(within_corr) - np.mean(between_corr):+.4f}")

    # Layer-level impact curve
    words = [w for w in data if not w.startswith("_")]
    print(f"\n  Mean |logit change| across {len(words)} words:")
    for l in layers:
        imps = [data[w].get(str(l), {}).get("abs_impact", 0) for w in words]
        bar = "#" * int(np.mean(imps))
        print(f"    L{l:>2}: {np.mean(imps):>6.2f} {bar}")

# ===== MLP vs Residual comparison =====
print(f"\n{'='*70}")
print(f"  MLP vs Residual Causal Impact")
print(f"{'='*70}")

for model in ["qwen3", "glm4", "deepseek7b"]:
    path_res = Path(f"results/phase273_causal_path/{model}_exp_a.json")
    path_mlp = Path(f"results/phase273_causal_path/{model}_exp_b.json")
    if not path_res.exists() or not path_mlp.exists():
        continue
    with open(path_res) as f:
        res_data = json.load(f)
    with open(path_mlp) as f:
        mlp_data = json.load(f)
    
    print(f"\n  {model}:")
    test_words = ["apple", "banana", "dog", "cat", "car", "bus", "orange", "lion", "train", "grape"]
    for w in test_words:
        if w in res_data and w in mlp_data:
            res_layers = sorted([int(l) for l in res_data[w].keys()])
            mlp_layers = sorted([int(l) for l in mlp_data[w].keys()])
            
            res_impacts = [res_data[w].get(str(l), {}).get("abs_impact", 0) for l in res_layers]
            mlp_impacts = [mlp_data[w].get(str(l), {}).get("abs_impact", 0) for l in mlp_layers]
            
            res_peak_l = res_layers[int(np.argmax(res_impacts))]
            mlp_peak_l = mlp_layers[int(np.argmax(mlp_impacts))]
            
            print(f"    {w:<10}: Residual peak=L{res_peak_l}({max(res_impacts):.1f}), "
                  f"MLP peak=L{mlp_peak_l}({max(mlp_impacts):.1f}), "
                  f"MLP/Residual ratio={np.mean(mlp_impacts)/max(np.mean(res_impacts),0.01):.2f}")

# ===== Exp D detailed analysis =====
print(f"\n{'='*70}")
print(f"  Causal Divergence Points")
print(f"{'='*70}")

for model in ["qwen3", "glm4", "deepseek7b"]:
    path = Path(f"results/phase273_causal_path/{model}_exp_d.json")
    if not path.exists():
        continue
    with open(path) as f:
        data = json.load(f)
    
    pairs = data.get("pairs", [])
    print(f"\n  {model}:")
    
    within_diff_layers = []
    between_diff_layers = []
    
    for p in pairs:
        is_within = p.get("is_within", False)
        pair_name = p.get("pair", "")
        a_layer = p.get("max_diff_layer_a_context", "N/A")
        b_layer = p.get("max_diff_layer_b_context", "N/A")
        label = "W" if is_within else "B"
        
        if isinstance(a_layer, int):
            if is_within:
                within_diff_layers.append(a_layer)
            else:
                between_diff_layers.append(a_layer)
        
        print(f"    [{label}] {pair_name}: L{a_layer} / L{b_layer}")
    
    if within_diff_layers:
        print(f"    Within mean divergence layer: {np.mean(within_diff_layers):.1f} ± {np.std(within_diff_layers):.1f}")
    if between_diff_layers:
        print(f"    Between mean divergence layer: {np.mean(between_diff_layers):.1f} ± {np.std(between_diff_layers):.1f}")
