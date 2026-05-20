"""Final cross-model alignment analysis for Phase 237"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

results = {}
for m in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        r = json.load(open(f'tests/glm5_temp/phase237_{m}_results.json', encoding='utf-8'))
        results[m] = r
    except:
        print(f"Warning: No results for {m}")

print("="*70)
print("Phase 237 Cross-Model Summary")
print("="*70)

# ExpA: SVD & d_not decoding
print("\n=== ExpA: SVD & d_not Decoding ===")
print(f"{'Model':<12} {'HS k90':<8} {'LS k90':<8} {'HS top1%':<10} {'LS top1%':<10}")
print("-"*50)
for m in ['qwen3', 'glm4', 'deepseek7b']:
    a = results[m].get('expA', {})
    hs = a.get('hidden_svd', {})
    ls = a.get('logit_svd', {})
    print(f"{m:<12} {hs.get('k90','?'):<8} {ls.get('k90','?'):<8} {hs.get('top1_var',0)*100:<10.1f} {ls.get('top1_var',0)*100:<10.1f}")

# Top-5 boosted tokens per model
print("\n=== Top-5 Boosted Tokens (logit d_not) ===")
for m in ['qwen3', 'glm4', 'deepseek7b']:
    a = results[m].get('expA', {})
    boosted = a.get('top_boosted_logit', [])[:5]
    toks = [t for t, v in boosted]
    print(f"  {m}: {', '.join(toks)}")

print("\n=== Top-5 Suppressed Tokens (logit d_not) ===")
for m in ['qwen3', 'glm4', 'deepseek7b']:
    a = results[m].get('expA', {})
    suppressed = a.get('top_suppressed_logit', [])[:5]
    toks = [t for t, v in suppressed]
    print(f"  {m}: {', '.join(toks)}")

# ExpB: Behavior
print("\n=== ExpB: Negation Behavior ===")
print(f"{'Model':<12} {'Simple':<10} {'Entail':<10} {'Overall':<10}")
print("-"*42)
for m in ['qwen3', 'glm4', 'deepseek7b']:
    b = results[m].get('expB', {})
    print(f"{m:<12} {b.get('simple_accuracy',0):.3f}     {b.get('entail_accuracy',0):.3f}     {b.get('overall_accuracy',0):.3f}")

# ExpC: Multi-type
print("\n=== ExpC: Per-Type k90 ===")
for m in ['qwen3', 'glm4', 'deepseek7b']:
    c = results[m].get('expC', {})
    k90s = c.get('k90_summary', {})
    print(f"  {m}: {k90s}")

# Cross-model logit direction alignment
print("\n=== Cross-Model Logit Direction Alignment ===")
logit_d_nots = {}
for m in ['qwen3', 'glm4', 'deepseek7b']:
    d = results[m].get('expA', {}).get('d_not_logit')
    if d is not None:
        logit_d_nots[m] = np.array(d)

for i, m1 in enumerate(['qwen3', 'glm4', 'deepseek7b']):
    for j, m2 in enumerate(['qwen3', 'glm4', 'deepseek7b']):
        if j > i and m1 in logit_d_nots and m2 in logit_d_nots:
            d1 = logit_d_nots[m1]
            d2 = logit_d_nots[m2]
            min_v = min(len(d1), len(d2))
            d1s = d1[:min_v] / (np.linalg.norm(d1[:min_v]) + 1e-10)
            d2s = d2[:min_v] / (np.linalg.norm(d2[:min_v]) + 1e-10)
            cos = float(np.dot(d1s, d2s))
            print(f"  {m1} vs {m2}: cosine={cos:.4f}")

# Final verdict
print("\n" + "="*70)
print("CRITICAL FINDINGS")
print("="*70)
ds7b_simple = results.get('deepseek7b', {}).get('expB', {}).get('simple_accuracy', 0)
qwen3_simple = results.get('qwen3', {}).get('expB', {}).get('simple_accuracy', 0)
glm4_simple = results.get('glm4', {}).get('expB', {}).get('simple_accuracy', 0)

print(f"""
1. DS7B's 1D negation is a DEFECT, not an ability:
   - DS7B simple negation accuracy: {ds7b_simple:.1%} (below random 50%!)
   - Qwen3: {qwen3_simple:.1%}, GLM4: {glm4_simple:.1%}
   - DS7B CANNOT reliably distinguish affirmative from negated sentences

2. DS7B's d_not is semantically empty:
   - Top boosted tokens: function words (the, comma, in) and code tokens (Utf, intl)
   - No negation-related semantic tokens (no "not", "no", "never", etc.)
   - Qwen3's d_not: boosted "正确" (correct), "否" (no), "相符" (matching)
   - GLM4's d_not: boosted proper nouns, suppressed Chinese "这个" (this)

3. Low-dim structure is robust across sentence types BUT is a capacity limitation:
   - DS7B: k90=3-5 for all sentence types
   - Qwen3/GLM4: k90=26-34 for all sentence types
   - DS7B compresses ALL negation into ~1-5 dimensions, losing critical semantics

4. Qwen3 and GLM4 have capable negation representations:
   - Both achieve ~82% overall accuracy
   - Their higher-dimensional representations encode richer negation semantics

5. The "dimension spectrum" conclusion from Phase 236 needs revision:
   - DS7B's low-dim = capacity limitation, not elegant compression
   - The real question: does Qwen3/GLM4's high-dim representation contain
     a low-dim CORE that DS7B lost, or is negation inherently high-dim?
""")
