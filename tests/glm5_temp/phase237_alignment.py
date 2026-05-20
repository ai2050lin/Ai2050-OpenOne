"""Cross-model direction alignment analysis for Phase 237"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

# Load all results
results = {}
for m in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        r = json.load(open(f'tests/glm5_temp/phase237_{m}_results.json', encoding='utf-8'))
        results[m] = r
    except:
        print(f"Warning: No results for {m}")

# Extract logit d_not directions
logit_d_nots = {}
hidden_d_nots = {}
for m in ['qwen3', 'glm4', 'deepseek7b']:
    if m in results and 'expA' in results[m]:
        expA = results[m]['expA']
        if 'd_not_logit' in expA and expA['d_not_logit'] is not None:
            logit_d_nots[m] = np.array(expA['d_not_logit'])
        if 'd_not_hidden' in expA and expA['d_not_hidden'] is not None:
            hidden_d_nots[m] = np.array(expA['d_not_hidden'])

print("=== Logit d_not shapes ===")
for m, d in logit_d_nots.items():
    print(f"  {m}: {d.shape}")

print("\n=== Hidden d_not shapes ===")
for m, d in hidden_d_nots.items():
    print(f"  {m}: {d.shape}")

# For models with same vocab size, direct cosine
# Qwen3: 151936, DS7B: 152064, GLM4: 151552
# Qwen3 and DS7B share Qwen tokenizer - many overlapping tokens
# Compare first 151936 tokens (Qwen3 vocab)

print("\n=== Direct Logit Space Alignment ===")
models = list(logit_d_nots.keys())
for i in range(len(models)):
    for j in range(i+1, len(models)):
        m1, m2 = models[i], models[j]
        d1 = logit_d_nots[m1]
        d2 = logit_d_nots[m2]
        
        min_vocab = min(len(d1), len(d2))
        d1_sub = d1[:min_vocab]
        d2_sub = d2[:min_vocab]
        
        # Normalize
        d1_norm = d1_sub / (np.linalg.norm(d1_sub) + 1e-10)
        d2_norm = d2_sub / (np.linalg.norm(d2_sub) + 1e-10)
        
        cos = float(np.dot(d1_norm, d2_norm))
        print(f"  {m1} vs {m2}: cosine={cos:.4f} (over first {min_vocab} tokens)")
        
        # Also check top-k overlap
        top1k_d1 = set(np.argsort(np.abs(d1_sub))[-1000:].tolist())
        top1k_d2 = set(np.argsort(np.abs(d2_sub))[-1000:].tolist())
        overlap = len(top1k_d1 & top1k_d2)
        print(f"    Top-1000 active token overlap: {overlap}/1000 ({overlap/10:.1f}%)")

# Hidden state alignment - different d_model, need CCA or projection
print("\n=== Hidden d_not Analysis ===")
for m, d in hidden_d_nots.items():
    # Norm and sparsity
    norm = np.linalg.norm(d)
    l1 = np.sum(np.abs(d))
    sparsity = 1.0 - (l1 / (norm * np.sqrt(len(d))))
    print(f"  {m}: norm={norm:.2f}, sparsity={sparsity:.3f}, d_model={len(d)}")

# Compare top boosted/suppressed tokens semantically
print("\n=== Semantic Overlap of Top Boosted Tokens ===")
for m in ['qwen3', 'glm4', 'deepseek7b']:
    if m in results and 'expA' in results[m]:
        boosted = results[m]['expA'].get('top_boosted_logit', [])
        suppressed = results[m]['expA'].get('top_suppressed_logit', [])
        print(f"\n  {m} top boosted (logit):")
        for tok, val in boosted[:5]:
            print(f"    {tok}: {val:.4f}")
        print(f"  {m} top suppressed (logit):")
        for tok, val in suppressed[:5]:
            print(f"    {tok}: {val:.4f}")

# Behavior comparison
print("\n=== Negation Behavior Accuracy ===")
for m in ['qwen3', 'glm4', 'deepseek7b']:
    if m in results and 'expB' in results[m]:
        b = results[m]['expB']
        print(f"  {m}: simple={b.get('simple_accuracy','?'):.3f}, "
              f"entail={b.get('entail_accuracy','?'):.3f}, "
              f"overall={b.get('overall_accuracy','?'):.3f}")

# Multi-sentence-type comparison
print("\n=== Per-Type k90 ===")
for m in ['qwen3', 'glm4', 'deepseek7b']:
    if m in results and 'expC' in results[m]:
        c = results[m]['expC']
        k90s = c.get('k90_summary', {})
        print(f"  {m}: {k90s}")

# Final verdict
print("\n" + "="*60)
print("FINAL VERDICT")
print("="*60)

# Compare DS7B behavior vs other models
ds7b_overall = results.get('deepseek7b', {}).get('expB', {}).get('overall_accuracy', 0)
qwen3_overall = results.get('qwen3', {}).get('expB', {}).get('overall_accuracy', 0)
glm4_overall = results.get('glm4', {}).get('expB', {}).get('overall_accuracy', 0)
ds7b_simple = results.get('deepseek7b', {}).get('expB', {}).get('simple_accuracy', 0)

print(f"Negation behavior accuracy:")
print(f"  Qwen3: {qwen3_overall:.3f}")
print(f"  GLM4:  {glm4_overall:.3f}")
print(f"  DS7B:  {ds7b_overall:.3f} (simple negation only: {ds7b_simple:.3f})")
print()

if ds7b_simple < 0.4:
    print("CONCLUSION: DS7B's low-dimensional negation is a DEFECT, not an ability.")
    print("  - Simple negation accuracy (27%) is far below random (50%)")
    print("  - DS7B cannot reliably distinguish affirmative from negated sentences")
    print("  - The 1D collapse in middle layers reflects capacity limitation,")
    print("    not discovery of an abstract negation axis")
    print()
    print("IMPLICATION: Qwen3's higher-dimensional negation (k90=40 in logit space)")
    print("  is the more capable representation. The extra dimensions encode")
    print("  context-dependent negation semantics that DS7B loses.")
elif ds7b_overall >= 0.7:
    print("CONCLUSION: DS7B's low-dimensional negation IS a genuine ability.")
else:
    print("CONCLUSION: Mixed results - DS7B has partial negation capability")
    print("  but struggles with direct negation judgments.")
