import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

r = json.load(open('tests/glm5_temp/phase237_deepseek7b_results.json', encoding='utf-8'))
a = r['expA']
print('=== DS7B ExpA ===')
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
b = r['expB']
print(f"=== DS7B ExpB ===")
print(f"Simple negation: {b['simple_accuracy']:.3f} ({b['simple_correct']}/{b['simple_total']})")
print(f"Entailment: {b['entail_accuracy']:.3f} ({b['entail_correct']}/{b['entail_total']})")
print(f"Overall: {b['overall_accuracy']:.3f} ({b['overall_correct']}/{b['overall_total']})")
print(f"Verdict: {b['verdict']}")
print()
c = r['expC']
print(f"=== DS7B ExpC ===")
for st, res in c['type_results'].items():
    print(f"  {st}: HS k90={res.get('hidden_k90','?')}, LS k90={res.get('logit_k90','?')}, n={res['n_valid']}")
print(f"Verdict: {c['verdict']}")
