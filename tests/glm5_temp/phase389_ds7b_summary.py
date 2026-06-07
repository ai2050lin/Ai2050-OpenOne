"""Phase 389 DS7B result summary"""
import json

with open('results/phase389_per_pair_analysis/deepseek7b_phase389.json') as f:
    data = json.load(f)

print("=== DS7B Phase 389 ===")
print(f"Layers: {data['layers']}")
print(f"n_correct: {data['n_correct']}, n_incorrect: {data['n_incorrect']}")

for li in [str(l) for l in data['layers']]:
    cc = data['condition_comparison'][li]
    print(f"\nL{li}:")
    print(f"  Correct:   mean={cc['correct_mean']:+.4f}, t={cc['correct_t']:+.2f}, pos={cc['correct_pos_pct']:.0f}%")
    print(f"  Incorrect: mean={cc['incorrect_mean']:+.4f}, t={cc['incorrect_t']:+.2f}, pos={cc['incorrect_pos_pct']:.0f}%")
    print(f"  Corr(baseline,add)={cc['baseline_add_corr']:.3f}")
    
    cs = data['category_summary'][li]
    print(f"\n  Category breakdown:")
    cats_sorted = sorted(cs.items(), key=lambda x: abs(x[1]['correct_mean']), reverse=True)
    for cat, c in cats_sorted:
        print(f"    {cat:12s}: C={c['correct_mean']:+.4f}({c['correct_pos_pct']:.0f}%pos,n={c['n_correct']}) "
              f"I={c['incorrect_mean']:+.4f}({c['incorrect_pos_pct']:.0f}%pos,n={c['n_incorrect']})")

# Also check Qwen3 for comparison
print("\n\n=== Qwen3 Phase 389 (for comparison) ===")
with open('results/phase389_per_pair_analysis/qwen3_phase389.json') as f:
    qdata = json.load(f)

for li in ['4', '20']:
    if li in qdata['condition_comparison']:
        cc = qdata['condition_comparison'][li]
        print(f"\nL{li}:")
        print(f"  Correct:   mean={cc['correct_mean']:+.4f}, t={cc['correct_t']:+.2f}, pos={cc['correct_pos_pct']:.0f}%")
        print(f"  Incorrect: mean={cc['incorrect_mean']:+.4f}, t={cc['incorrect_t']:+.2f}, pos={cc['incorrect_pos_pct']:.0f}%")
        
        cs = qdata['category_summary'][li]
        print(f"  Category breakdown:")
        cats_sorted = sorted(cs.items(), key=lambda x: abs(x[1]['correct_mean']), reverse=True)
        for cat, c in cats_sorted:
            print(f"    {cat:12s}: C={c['correct_mean']:+.4f}({c['correct_pos_pct']:.0f}%pos,n={c['n_correct']}) "
                  f"I={c['incorrect_mean']:+.4f}({c['incorrect_pos_pct']:.0f}%pos,n={c['n_incorrect']})")
