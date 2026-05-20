import json
import numpy as np

for model in ['qwen3', 'glm4']:
    try:
        with open(f'tests/glm5_temp/phase232_{model}_results.json', 'r', encoding='utf-8') as f:
            d = json.load(f)
    except:
        print(f"{model}: NO FILE")
        continue
    
    print(f"\n=== {model.upper()} ===")
    
    # ExpC details
    c = d.get('expC', {})
    lr = c.get('layer_results', {})
    print(f"ExpC: best_layer={c.get('best_patch_layer')}, best_kl={c.get('best_patch_kl')}")
    if lr:
        for k in sorted(lr.keys(), key=int):
            v = lr[k]
            if isinstance(v, list):
                if len(v) > 0 and isinstance(v[0], (int, float)):
                    kl = np.mean(v)
                else:
                    kl = 0
            elif isinstance(v, dict):
                kl = v.get('mean_kl', 0)
            else:
                kl = 0
            if kl > 0.01:
                print(f"  L{k}: KL={kl:.4f}")
    else:
        print("  No layer_results")
    
    # ExpA details (check if broken)
    a = d.get('expA', {})
    alr = a.get('layer_results', {})
    if alr:
        max_kl = max(v.get('mean_kl', 0) for v in alr.values())
        print(f"ExpA: max KL across layers = {max_kl:.6f}")
        if max_kl < 0.001:
            print("  >>> ExpA BROKEN (all KL≈0, missing LayerNorm)")
    
    # ExpB detailed
    b = d.get('expB', {})
    print(f"ExpB: cross_cos={b.get('mean_cross_cosine', 'N/A')}, "
          f"sparsity={b.get('sparsity', 'N/A')}, "
          f"suppress_ratio={b.get('suppress_ratio', 'N/A')}")
    pca = b.get('pca_var_explained', [])
    print(f"  PCA top5: {[f'{x:.3f}' for x in pca[:5]]}")
    
    # ExpD detailed
    dd = d.get('expD', {})
    ci = dd.get('component_importance', dd.get('head_importance', {}))
    baseline = dd.get('baseline_kl', 'N/A')
    print(f"ExpD: baseline_kl={baseline}")
    
    attn_reds = {k: v['mean_reduction'] for k, v in ci.items() if 'self_attn' in k}
    mlp_reds = {k: v['mean_reduction'] for k, v in ci.items() if 'mlp' in k}
    
    print(f"  Self-attn ablations:")
    for k, v in sorted(attn_reds.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v:.4f}")
    print(f"  MLP ablations:")
    for k, v in sorted(mlp_reds.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v:.4f}")
    
    # ExpE detailed
    e = d.get('expE', {})
    print(f"ExpE: cross_cos={e.get('mean_cross_cosine', 'N/A')}, verdict={e.get('verdict', 'N/A')}")
    pairs = e.get('pairwise_cosines', {})
    if pairs:
        print(f"  Pairwise cosines:")
        for k, v in sorted(pairs.items())[:5]:
            print(f"    {k}: {v:.4f}")
