import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

r = json.load(open('tests/glm5_temp/phase237_qwen3_results.json', encoding='utf-8'))
a = r['expA']
print('=== Qwen3 ExpA ===')
hs = a['hidden_svd']
ls = a['logit_svd']
print(f"HS k90={hs['k90']}, top1={hs['top1_var']*100:.1f}%")
print(f"LS k90={ls['k90']}, top1={ls['top1_var']*100:.1f}%")
print()
print('Top-10 boosted (W_U @ d_not_h):')
for tok, val in a['top_boosted_hidden'][:10]:
    print(f'  {tok}: {val:.4f}')
print()
print('Top-10 suppressed (W_U @ d_not_h):')
for tok, val in a['top_suppressed_hidden'][:10]:
    print(f'  {tok}: {val:.4f}')
print()
print('Top-10 boosted (logit d_not_l):')
for tok, val in a['top_boosted_logit'][:10]:
    print(f'  {tok}: {val:.4f}')
print()
print('Top-10 suppressed (logit d_not_l):')
for tok, val in a['top_suppressed_logit'][:10]:
    print(f'  {tok}: {val:.4f}')
print()
ast = a['alpha_stats']
print(f"Alpha stats: mean={ast['mean']:.4f}, std={ast['std']:.4f}, min={ast['min']:.4f}, max={ast['max']:.4f}")
