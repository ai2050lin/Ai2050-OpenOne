"""Compare Phase 389 vs Phase 390 results"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Load Phase 389
with open(ROOT / "results/phase389_per_pair_analysis/qwen3_phase389.json") as f:
    p389 = json.load(f)

# Load Phase 390 v3
with open(ROOT / "results/phase390_conditional_centroid/qwen3_phase390_v3.json") as f:
    p390 = json.load(f)

print("=== Phase 389 vs Phase 390 Comparison (Qwen3) ===")

for li in [4, 20]:
    li_str = str(li)
    
    # Phase 389
    if li_str in p389['condition_comparison']:
        cc = p389['condition_comparison'][li_str]
        print(f"\nL{li} Phase 389:")
        print(f"  Correct: mean={cc['correct_mean']:+.4f}, t={cc['correct_t']:+.2f}")
        
        # Per-pair correct effects
        p389_correct = [p for p in p389['per_pair'][li_str] if p['condition'] == 'correct']
        p389_adds = [p['add_effect'] for p in p389_correct]
        print(f"  Per-pair range: [{min(p389_adds):.4f}, {max(p389_adds):.4f}]")
        print(f"  Per-pair std: {np.std(p389_adds):.4f}")
    
    # Phase 390 v3
    if li_str in p390['per_layer']:
        lr = p390['per_layer'][li_str]
        print(f"\nL{li} Phase 390v3:")
        print(f"  Overall: mean={lr['add_mean']:+.4f}, t={lr['add_t']:+.1f}")
        adds = lr['per_pair_add']
        print(f"  Per-pair range: [{min(adds):.4f}, {max(adds):.4f}]")
        print(f"  Per-pair std: {np.std(adds):.4f}")
        
        for cat in sorted(lr['category_effects'].keys()):
            ce = lr['category_effects'][cat]
            print(f"    {cat:12s}: {ce['add_mean']:+.4f}({ce['add_pos_pct']:.0f}%)")

# Detailed per-pair comparison for L4
print("\n\n=== Detailed L4 Comparison ===")
li_str = '4'
p389_correct = {p['idx']: p for p in p389['per_pair'][li_str] if p['condition'] == 'correct'}
p390_adds = p390['per_layer'][li_str]['per_pair_add']

# Compare a few pairs
diffs = []
for idx in range(min(20, len(p389_correct))):
    p389_val = p389_correct.get(idx, {}).get('add_effect', None)
    p390_val = p390_adds[idx] if idx < len(p390_adds) else None
    if p389_val is not None and p390_val is not None:
        print(f"  idx={idx}: p389={p389_val:+.4f}, p390={p390_val:+.4f}, diff={p390_val-p389_val:+.4f}")
        diffs.append(p390_val - p389_val)

if diffs:
    print(f"\n  Mean diff: {np.mean(diffs):+.4f}, Std: {np.std(diffs):.4f}")
    print(f"  Ratio p390/p389: {np.mean([abs(p390_adds[i]) / (abs(p389_correct[i]['add_effect']) + 1e-6) for i in range(len(diffs)) if i in p389_correct and abs(p389_correct[i]['add_effect']) > 0.001]):.2f}")
