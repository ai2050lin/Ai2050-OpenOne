"""Phase 390 cross-model summary"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

print("=" * 80)
print("PHASE 390: MULTI-LAYER PER-CATEGORY CENTROID TRACKING - CROSS-MODEL SUMMARY")
print("=" * 80)

models_data = {}
for model_name in ['qwen3', 'deepseek7b', 'glm4']:
    fpath = ROOT / f"results/phase390_conditional_centroid/{model_name}_phase390_v3.json"
    if not fpath.exists():
        print(f"{model_name}: NOT FOUND")
        continue
    with open(fpath) as f:
        models_data[model_name] = json.load(f)

# Cross-model category trajectory comparison
categories = ['color', 'temperature', 'moisture', 'size', 'weight', 'speed', 'brightness']

print("\n=== CATEGORY CENTROID EFFECT BY LAYER (ALL MODELS) ===\n")

for cat in categories:
    print(f"--- {cat} ---")
    for model_name, data in models_data.items():
        traj = data.get('category_trajectory', {}).get(cat, {})
        if traj:
            vals = []
            for layer_key in sorted(traj.keys(), key=lambda x: int(x[1:])):
                v = traj[layer_key]['add_mean']
                p = traj[layer_key]['add_pos_pct']
                vals.append(f"{layer_key}={v:+.4f}({p:.0f}%)")
            print(f"  {model_name:12s}: {' -> '.join(vals)}")
    print()

# Cross-model direction reversal comparison
print("\n=== DIRECTION REVERSALS (CROSS-MODEL) ===\n")
for cat in categories:
    reversals = {}
    for model_name, data in models_data.items():
        traj = data.get('category_trajectory', {}).get(cat, {})
        if len(traj) < 2:
            continue
        layers_sorted = sorted([int(k[1:]) for k in traj.keys()])
        signs = [np.sign(traj[f"L{li}"]['add_mean']) for li in layers_sorted]
        n_rev = sum(1 for i in range(len(signs)-1) if signs[i] != signs[i+1] and signs[i] != 0 and signs[i+1] != 0)
        has_rev = n_rev > 0
        reversals[model_name] = has_rev

    any_rev = any(reversals.values())
    all_rev = all(reversals.values())
    if all_rev:
        status = "ALL REVERSE"
    elif any_rev:
        status = "SOME REVERSE"
    else:
        status = "ALL STABLE"
    
    details = []
    for model_name, has_rev in reversals.items():
        traj = models_data[model_name].get('category_trajectory', {}).get(cat, {})
        if traj:
            layers_sorted = sorted([int(k[1:]) for k in traj.keys()])
            signs = [np.sign(traj[f"L{li}"]['add_mean']) for li in layers_sorted]
            direction = "+" if signs[0] > 0 else ("-" if signs[0] < 0 else "0")
            details.append(f"{model_name}={direction}{'(rev)' if has_rev else ''}")
    
    print(f"  {cat:12s}: {status:12s} | {' '.join(details)}")

# Key finding: brightness direction across models
print("\n\n=== KEY FINDING: BRIGHTNESS CENTROID DIRECTION ===\n")
for model_name, data in models_data.items():
    traj = data.get('category_trajectory', {}).get('brightness', {})
    if traj:
        print(f"  {model_name}:")
        for layer_key in sorted(traj.keys(), key=lambda x: int(x[1:])):
            v = traj[layer_key]['add_mean']
            p = traj[layer_key]['add_pos_pct']
            print(f"    {layer_key}: add={v:+.4f}, pos={p:.0f}%")

# Speed across models
print("\n=== KEY FINDING: SPEED CENTROID DIRECTION ===\n")
for model_name, data in models_data.items():
    traj = data.get('category_trajectory', {}).get('speed', {})
    if traj:
        print(f"  {model_name}:")
        for layer_key in sorted(traj.keys(), key=lambda x: int(x[1:])):
            v = traj[layer_key]['add_mean']
            p = traj[layer_key]['add_pos_pct']
            print(f"    {layer_key}: add={v:+.4f}, pos={p:.0f}%")

# DS7B size anomaly
print("\n=== ANOMALY: DS7B SIZE CENTROID ===\n")
for model_name, data in models_data.items():
    traj = data.get('category_trajectory', {}).get('size', {})
    if traj:
        for layer_key in sorted(traj.keys(), key=lambda x: int(x[1:])):
            v = traj[layer_key]['add_mean']
            p = traj[layer_key]['add_pos_pct']
            print(f"  {model_name} {layer_key}: add={v:+.4f}, pos={p:.0f}%")
