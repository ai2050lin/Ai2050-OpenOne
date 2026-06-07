"""
Validate W_U projection approximation vs actual forward pass effects.
This is critical for Phase 390 which will use W_U projection for fast multi-layer analysis.
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Load Phase 389 results (actual forward pass effects)
for model_name in ['qwen3', 'deepseek7b']:
    result_path = ROOT / "results" / "phase389_per_pair_analysis" / f"{model_name}_phase389.json"
    if not result_path.exists():
        print(f"{model_name}: Phase 389 result not found, skipping")
        continue

    with open(result_path) as f:
        data = json.load(f)

    print(f"\n=== {model_name} Phase 389 Data ===")
    print(f"Layers tested: {data['layers']}")
    
    for li_str in data['condition_comparison']:
        cc = data['condition_comparison'][li_str]
        print(f"\nL{li_str}:")
        print(f"  Correct:   mean={cc['correct_mean']:+.4f}, t={cc['correct_t']:+.2f}")
        print(f"  Incorrect: mean={cc['incorrect_mean']:+.4f}, t={cc['incorrect_t']:+.2f}")
        
        # Per-pair data
        per_pair = data['per_pair'][li_str]
        correct_pairs = [p for p in per_pair if p['condition'] == 'correct']
        incorrect_pairs = [p for p in per_pair if p['condition'] == 'incorrect']
        
        # Analyze heterogeneity
        add_effects_correct = [p['add_effect'] for p in correct_pairs]
        add_effects_incorrect = [p['add_effect'] for p in incorrect_pairs]
        
        print(f"  Correct add_effect range: [{min(add_effects_correct):.4f}, {max(add_effects_correct):.4f}]")
        print(f"  Incorrect add_effect range: [{min(add_effects_incorrect):.4f}, {max(add_effects_incorrect):.4f}]")
        print(f"  Correct std: {np.std(add_effects_correct):.4f}")
        print(f"  Incorrect std: {np.std(add_effects_incorrect):.4f}")
        
        # Per-category analysis
        cats = {}
        for p in correct_pairs:
            cat = p['category']
            if cat not in cats:
                cats[cat] = []
            cats[cat].append(p['add_effect'])
        
        print(f"\n  Category breakdown (correct condition):")
        for cat in sorted(cats.keys()):
            vals = cats[cat]
            pos_pct = sum(1 for v in vals if v > 0) / len(vals) * 100
            print(f"    {cat:12s}: mean={np.mean(vals):+.4f}, pos={pos_pct:.0f}%, n={len(vals)}")

# Cross-model comparison of category patterns
print("\n\n=== Cross-Model Category Pattern Comparison ===")

# Qwen3 L4 and L20
for model_name, layers in [('qwen3', ['4', '20']), ('deepseek7b', ['4'])]:
    result_path = ROOT / "results" / "phase389_per_pair_analysis" / f"{model_name}_phase389.json"
    if not result_path.exists():
        continue
    with open(result_path) as f:
        data = json.load(f)
    
    for li_str in layers:
        if li_str not in data['category_summary']:
            continue
        cs = data['category_summary'][li_str]
        print(f"\n{model_name} L{li_str}:")
        for cat in sorted(cs.keys()):
            c = cs[cat]
            sym = ""
            if c['correct_mean'] > 0 and c['incorrect_mean'] < 0:
                sym = "SYMMETRIC"
            elif c['correct_mean'] < 0 and c['incorrect_mean'] > 0:
                sym = "SYMMETRIC(reversed)"
            elif abs(c['correct_mean']) < 0.001 and abs(c['incorrect_mean']) < 0.001:
                sym = "NULL"
            else:
                sym = "ASYMMETRIC"
            print(f"  {cat:12s}: C={c['correct_mean']:+.4f} I={c['incorrect_mean']:+.4f} -> {sym}")
