"""Phase 236 结果汇总"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

results = {}
for m in ['qwen3', 'glm4', 'deepseek7b']:
    results[m] = json.load(open(f'tests/glm5_temp/phase236_{m}_results.json', encoding='utf-8'))

print('='*70)
print('Phase 236 跨模型汇总')
print('='*70)

print('\n=== ExpA: Δ_not SVD自由度 ===')
for m in ['qwen3', 'glm4', 'deepseek7b']:
    a = results[m]['expA']
    hs = a['hidden_state_svd']
    ls = a['logit_svd']
    print(f'  {m}: HS k90={hs["k90"]} LS k90={ls["k90"]} | HS top1={hs["top1_var"]*100:.1f}% LS top1={ls["top1_var"]*100:.1f}% | n={a["n_valid"]}')

print('\n=== ExpB: 跨算子k90 ===')
for m in ['qwen3', 'glm4', 'deepseek7b']:
    b = results[m].get('expB', {})
    op_svd = b.get('operator_svd', {})
    parts = []
    for op in ['not', 'never', 'always', 'rarely', 'often']:
        if op in op_svd:
            parts.append(f'{op}={op_svd[op]["k90"]}')
    print(f'  {m}: {", ".join(parts)}')

print('\n=== ExpC: 长度控制双重否定 ===')
for m in ['qwen3', 'glm4', 'deepseek7b']:
    if 'expC' not in results[m]:
        print(f'  {m}: N/A (only ExpA was run in large mode)')
        continue
    c = results[m]['expC']
    print(f'  {m}: KL_s={c["mean_kl_single_neg"]:.4f} KL_d={c["mean_kl_double_neg"]:.4f} KL_l={c["mean_kl_length_ctrl"]:.4f} ratio={c["ratio_double_vs_length"]:.2f} | {c["verdict"][:50]}')

print('\n=== 关键发现 ===')
print('1. DS7B: 否定是极低维的 (LS k90=2, top1=86.1%)')
print('2. Qwen3: 中等维度 (LS k90=38, top1=45.0%)')
print('3. GLM4: 高维度 (LS k90=62, top1~18%)')
print('4. DS7B中间层: k90=1, top1=97.5% — 几乎1维否定!')
print('5. 双重否定: Qwen3=语义漂移(3.3x), GLM4/DS7B=长度主导')
