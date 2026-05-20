"""Deep analysis: Why is Qwen3-DS7B logit d_not cosine=0.91 but DS7B behavior is bad?"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

results = {}
for m in ['qwen3', 'glm4', 'deepseek7b']:
    results[m] = json.load(open(f'tests/glm5_temp/phase237_{m}_results.json', encoding='utf-8'))

# Get logit d_not
d_qwen3 = np.array(results['qwen3']['expA']['d_not_logit'])
d_ds7b = np.array(results['deepseek7b']['expA']['d_not_logit'])
d_glm4 = np.array(results['glm4']['expA']['d_not_logit'])

# Qwen3 and DS7B share Qwen tokenizer
min_v = min(len(d_qwen3), len(d_ds7b))

# Full cosine
d_q = d_qwen3[:min_v]
d_d = d_ds7b[:min_v]
cos_full = np.dot(d_q, d_d) / (np.linalg.norm(d_q) * np.linalg.norm(d_d) + 1e-10)
print(f"Full cosine (Qwen3 vs DS7B, first {min_v} tokens): {cos_full:.4f}")

# Check: is the cosine dominated by a few tokens?
# Remove top-100 tokens by absolute value from both, recompute
abs_q = np.abs(d_q)
abs_d = np.abs(d_d)
combined_abs = abs_q + abs_d  # Combined importance

# Progressive masking: remove top-k tokens
for k in [0, 10, 50, 100, 500, 1000, 5000]:
    if k == 0:
        mask = np.ones(min_v, dtype=bool)
    else:
        top_k_idx = np.argsort(combined_abs)[-k:]
        mask = np.ones(min_v, dtype=bool)
        mask[top_k_idx] = False
    
    d_q_masked = d_q[mask]
    d_d_masked = d_d[mask]
    
    if len(d_q_masked) < 100:
        continue
    
    cos_masked = np.dot(d_q_masked, d_d_masked) / (np.linalg.norm(d_q_masked) * np.linalg.norm(d_d_masked) + 1e-10)
    print(f"  After removing top-{k} tokens: cosine={cos_masked:.4f} (remaining: {mask.sum()})")

# Compare alpha distributions
print("\n=== Alpha Distribution (Δ projection onto d_not) ===")
for m in ['qwen3', 'glm4', 'deepseek7b']:
    ast = results[m]['expA'].get('alpha_stats', {})
    if ast:
        print(f"  {m}: mean={ast.get('mean',0):.4f}, std={ast.get('std',0):.4f}, "
              f"min={ast.get('min',0):.4f}, max={ast.get('max',0):.4f}")

# Check: Qwen3 and DS7B top tokens overlap
print("\n=== Top Token Overlap ===")
for direction in ['boosted', 'suppressed']:
    key = f'top_{direction}_logit'
    for m in ['qwen3', 'glm4', 'deepseek7b']:
        toks = results[m]['expA'].get(key, [])[:20]
        tok_names = set(t for t, v in toks)
        if direction == 'boosted':
            print(f"\n  {m} top-20 boosted:")
            for t, v in toks[:10]:
                print(f"    {t}: {v:.5f}")

# Critical: Check Qwen3's d_not meaning more carefully
print("\n=== Qwen3 d_not Token Analysis ===")
d_qwen3_full = d_qwen3
# Top tokens by absolute value
top_abs_idx = np.argsort(np.abs(d_qwen3_full))[-30:][::-1]
print("Top-30 tokens by absolute value in Qwen3 logit d_not:")
for idx in top_abs_idx[:15]:
    from transformers import AutoTokenizer
    # Can't load tokenizer here, just show index and value
    print(f"  token_id={idx}: value={d_qwen3_full[idx]:.5f}")

# Key question: Is Qwen3's d_not semantically meaningful?
print("\n=== Semantic Interpretation ===")
qwen3_boosted = results['qwen3']['expA'].get('top_boosted_logit', [])[:20]
qwen3_suppressed = results['qwen3']['expA'].get('top_suppressed_logit', [])[:20]
print("Qwen3 top-20 BOOSTED by d_not:")
for t, v in qwen3_boosted:
    print(f"  {t}: {v:.5f}")
print("\nQwen3 top-20 SUPPRESSED by d_not:")
for t, v in qwen3_suppressed:
    print(f"  {t}: {v:.5f}")

# DS7B d_not
ds7b_boosted = results['deepseek7b']['expA'].get('top_boosted_logit', [])[:20]
ds7b_suppressed = results['deepseek7b']['expA'].get('top_suppressed_logit', [])[:20]
print("\nDS7B top-20 BOOSTED by d_not:")
for t, v in ds7b_boosted:
    print(f"  {t}: {v:.5f}")
print("\nDS7B top-20 SUPPRESSED by d_not:")
for t, v in ds7b_suppressed:
    print(f"  {t}: {v:.5f}")

print("\n=== KEY INSIGHT ===")
print("Qwen3 and DS7B share cosine=0.91 in logit space, BUT:")
print("- Qwen3's d_not boosts semantically meaningful tokens (correct, no, matching)")
print("- DS7B's d_not boosts noise tokens (Utf, intl, code fragments)")
print("The high cosine is likely due to shared tokenizer structure,")
print("NOT shared negation semantics. The bulk of the vector (low-magnitude")
print("dimensions that dominate the cosine) may be alignment noise.")
print("The meaningful negation signal is in the TOP dimensions,")
print("which differ between the models.")
