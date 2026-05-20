import json

with open('tests/glm5_temp/phase232_deepseek7b_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

# ExpC
c = d['expC']
lr = c.get('layer_results', {})
print('DS7B ExpC layer results:')
for k in sorted(lr.keys(), key=lambda x: int(x) if x.isdigit() else 0):
    v = lr[k]
    if isinstance(v, dict):
        kl = v.get('mean_kl', 0)
        if kl > 0.01:
            print(f'  L{k}: KL={kl:.4f}')

# ExpB details
b = d['expB']
print(f'\nDS7B ExpB:')
print(f'  cross_cos={b.get("mean_cross_cosine", 0):.4f}')
pca = b.get('pca_var_explained', [])
print(f'  PCA top5: {[round(x, 3) for x in pca[:5]]}')
print(f'  sparsity={b.get("sparsity", 0):.4f}')
print(f'  suppress_ratio={b.get("suppress_ratio", 0):.3f}')

# ExpE pairwise
e = d['expE']
pw = e.get('pairwise_cosines', {})
print(f'\nDS7B ExpE pairwise cosines:')
for k, v in sorted(pw.items()):
    print(f'  {k}: {v:.4f}')

# Also check Qwen3 and GLM4 ExpC
for model in ['qwen3', 'glm4']:
    try:
        with open(f'tests/glm5_temp/phase232_{model}_results.json', 'r', encoding='utf-8') as f:
            dd = json.load(f)
        cc = dd['expC']
        llr = cc.get('layer_results', {})
        print(f'\n{model} ExpC layer results:')
        for k in sorted(llr.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            v = llr[k]
            if isinstance(v, dict):
                kl = v.get('mean_kl', 0)
                if kl > 0.01:
                    print(f'  L{k}: KL={kl:.4f}')
    except:
        print(f'{model}: no data')
