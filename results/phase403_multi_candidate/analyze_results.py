"""Phase 403 Result Analysis - Cross-model comparison"""
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(r"d:\Ai2050\TransformerLens-Project\results\phase403_multi_candidate")

models = ['qwen3', 'deepseek7b', 'glm4']
all_data = {}

for model_name in models:
    path = RESULTS_DIR / f"{model_name}_phase403.json"
    if path.exists():
        with open(path) as f:
            all_data[model_name] = json.load(f)

print("=" * 100)
print("Phase 403 Cross-Model Analysis: Multi-Candidate Speed Distribution Dynamics")
print("=" * 100)

# 1. Cross-layer summary for each model
print("\n## 1. Cross-Layer Monotonicity Change (speed_only patch)")
print(f"{'Model':>15s} {'Layer':>6s} {'Mono_Δ_within':>14s} {'Mono_Δ_across':>14s} {'Ent_Δ_within':>14s} {'Ent_Δ_across':>14s}")
for model_name in models:
    if model_name not in all_data:
        continue
    for li_str, lr in all_data[model_name]['per_layer'].items():
        agg = lr.get('aggregate', {})
        sp = agg.get('speed_only', {})
        print(f"{model_name:>15s} {li_str:>6s} {sp.get('mono_change_within',0):>+14.4f} {sp.get('mono_change_across',0):>+14.4f} "
              f"{sp.get('ent_change_within',0):>+14.4f} {sp.get('ent_change_across',0):>+14.4f}")

# 2. Key comparison: SPEED-only vs norm_control
print("\n## 2. SPEED-only vs Norm Control (deep layer)")
deep_layers = {'qwen3': '28', 'deepseek7b': '20', 'glm4': '35'}
print(f"{'Model':>15s} {'SPEED_mono_Δ':>14s} {'NORM_mono_Δ':>14s} {'SPEED-Fast_odd':>14s} {'SPEED-Slow_odd':>14s} {'NORM-Fast_odd':>14s} {'NORM-Slow_odd':>14s}")
for model_name in models:
    if model_name not in all_data:
        continue
    li = deep_layers.get(model_name)
    if not li:
        continue
    lr = all_data[model_name]['per_layer'].get(li, {})
    agg = lr.get('aggregate', {})
    sp = agg.get('speed_only', {})
    nc = agg.get('norm_control', {})
    # Use within-type values
    print(f"{model_name:>15s} {sp.get('mono_change_within',0):>+14.4f} {nc.get('mono_change_within',0):>+14.4f} "
          f"{sp.get('fast_cand_odd_within',0):>+14.4f} {sp.get('slow_cand_odd_within',0):>+14.4f} "
          f"{nc.get('fast_cand_odd_within',0):>+14.4f} {nc.get('slow_cand_odd_within',0):>+14.4f}")

# 3. Fast vs Slow candidate odd effects
print("\n## 3. Full Direction Patch: Fast vs Slow Candidate Odd (deep layer)")
print(f"{'Model':>15s} {'Fast_within':>13s} {'Fast_across':>13s} {'Fast_diff':>10s} {'Slow_within':>13s} {'Slow_across':>13s} {'Slow_diff':>10s}")
for model_name in models:
    if model_name not in all_data:
        continue
    li = deep_layers.get(model_name)
    lr = all_data[model_name]['per_layer'].get(li, {})
    agg = lr.get('aggregate', {})
    full = agg.get('full', {})
    fw = full.get('fast_cand_odd_within', 0)
    fa = full.get('fast_cand_odd_across', 0)
    sw = full.get('slow_cand_odd_within', 0)
    sa = full.get('slow_cand_odd_across', 0)
    print(f"{model_name:>15s} {fw:>+13.4f} {fa:>+13.4f} {fw-fa:>+10.4f} {sw:>+13.4f} {sa:>+13.4f} {sw-sa:>+10.4f}")

# 4. Type-specific analysis: speed_only patch
print("\n## 4. SPEED-only: Fast/Slow Candidate by Type Relation (deep layer)")
print(f"{'Model':>15s} {'Fast_within':>13s} {'Fast_across':>13s} {'Fast_diff':>10s} {'Slow_within':>13s} {'Slow_across':>13s} {'Slow_diff':>10s}")
for model_name in models:
    if model_name not in all_data:
        continue
    li = deep_layers.get(model_name)
    lr = all_data[model_name]['per_layer'].get(li, {})
    agg = lr.get('aggregate', {})
    sp = agg.get('speed_only', {})
    fw = sp.get('fast_cand_odd_within', 0)
    fa = sp.get('fast_cand_odd_across', 0)
    sw = sp.get('slow_cand_odd_within', 0)
    sa = sp.get('slow_cand_odd_across', 0)
    print(f"{model_name:>15s} {fw:>+13.4f} {fa:>+13.4f} {fw-fa:>+10.4f} {sw:>+13.4f} {sa:>+13.4f} {sw-sa:>+10.4f}")

# 5. Entropy changes (distribution compression)
print("\n## 5. Distribution Entropy Change (deep layer)")
print(f"{'Model':>15s} {'Patch':>15s} {'Ent_Δ_within':>14s} {'Ent_Δ_across':>14s}")
for model_name in models:
    if model_name not in all_data:
        continue
    li = deep_layers.get(model_name)
    lr = all_data[model_name]['per_layer'].get(li, {})
    agg = lr.get('aggregate', {})
    for pt in ['full', 'type_only', 'speed_only', 'norm_control']:
        d = agg.get(pt, {})
        print(f"{model_name:>15s} {pt:>15s} {d.get('ent_change_within',0):>+14.4f} {d.get('ent_change_across',0):>+14.4f}")

# 6. Rank correlation stability
print("\n## 6. Rank Correlation (baseline vs patched distribution)")
print(f"{'Model':>15s} {'Layer':>6s} {'Full':>10s} {'TYPE':>10s} {'SPEED':>10s} {'NORM':>10s}")
for model_name in models:
    if model_name not in all_data:
        continue
    for li_str, lr in all_data[model_name]['per_layer'].items():
        agg = lr.get('aggregate', {})
        full_rc = agg.get('full', {}).get('rank_corr_mean', 0)
        type_rc = agg.get('type_only', {}).get('rank_corr_mean', 0)
        speed_rc = agg.get('speed_only', {}).get('rank_corr_mean', 0)
        norm_rc = agg.get('norm_control', {}).get('rank_corr_mean', 0)
        print(f"{model_name:>15s} {li_str:>6s} {full_rc:>+10.4f} {type_rc:>+10.4f} {speed_rc:>+10.4f} {norm_rc:>+10.4f}")

print("\n" + "=" * 100)
print("Key Observations")
print("=" * 100)
