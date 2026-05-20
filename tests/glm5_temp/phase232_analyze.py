import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('tests/glm5_temp/phase232_qwen3_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

expA = d.get('expA', {})
lr = expA.get('layer_results', {})
print('ExpA Layer Results (Qwen3):')
for k in sorted(lr.keys(), key=int):
    v = lr[k]
    print(f'  L{k}: KL={v["mean_kl"]:.4f}, flip={v["mean_flip_ratio"]:.4f}, flip_pct={v["flip_below_0.5_pct"]:.1f}')

print()
expB = d.get('expB', {})
print(f'ExpB: cross_cos={expB["mean_cross_cosine"]:.4f}, verdict={expB["verdict"]}')
print(f'  sparsity={expB["sparsity"]:.4f}, suppress_ratio={expB["suppress_ratio"]:.3f}')
print(f'  PCA: {[f"{v:.3f}" for v in expB["pca_var_explained"]]}')
print(f'  Top suppressed: {expB["top_suppressed"][:10]}')
print(f'  Top enhanced: {expB["top_enhanced"][:10]}')

print()
expC = d.get('expC', {})
lr_c = expC.get('layer_results', {})
print('ExpC Patching Results:')
for k in sorted(lr_c.keys(), key=int):
    vals = lr_c[k]
    if vals and isinstance(vals[0], dict):
        mean_kl = sum(v['kl_vs_affirm'] for v in vals) / len(vals)
        mean_cos = sum(v['cosine_vs_negated'] for v in vals) / len(vals)
        print(f'  L{k}: KL={mean_kl:.4f}, cos_vs_neg={mean_cos:.4f}')

print()
expD = d.get('expD', {})
print(f'ExpD: baseline_kl={expD["baseline_kl"]:.4f}')
for k, v in sorted(expD['component_importance'].items(), key=lambda x: -x[1]['mean_reduction']):
    print(f'  {k}: reduction={v["mean_reduction"]:.4f}')

print()
expE = d.get('expE', {})
print(f'ExpE: cross_cos={expE["mean_cross_cosine"]:.4f}, verdict={expE["verdict"]}')
