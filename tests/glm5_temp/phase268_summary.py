"""Phase 268: 结果汇总分析"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')

for m in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/phase268_intrinsic_dim/{m}_summary.json') as f:
        s = json.load(f)
    print(f'=== {m} ===')
    print(f'  PR: {s["pr_first_layer"]:.1f} -> {s["pr_last_layer"]:.1f} (min={s["pr_min"]:.1f} at L{s["pr_min_layer"]})')
    print(f'  EffSupport: {s["eff_support_first"]:.0f} -> {s["eff_support_last"]:.0f}')
    print(f'  WU_Align: {s["wu_alignment_first"]:.3f} -> {s["wu_alignment_last"]:.3f}')
    print(f'  Goldilocks: {s["goldilocks_layers"]}')
    print(f'  Conclusion: {s["conclusion"]}')
    print()

# Print detailed layer-by-layer for key layers
for m in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/phase268_intrinsic_dim/{m}_full_results.json') as f:
        data = json.load(f)
    print(f'=== {m} - Layer Details ===')
    n_layers = max(int(k[1:]) for k in data.keys())
    key_layers = [0, 1, 2] + list(range(5, n_layers, 5)) + [n_layers-2, n_layers-1, n_layers]
    key_layers = sorted(set(l for l in key_layers if l <= n_layers))
    header = f'  {"Layer":>6} {"PR":>6} {"n95":>5} {"n>1%":>5} {"Entropy":>8} {"EffSup":>8} {"WU_A":>7} {"PR/ES":>7}'
    print(header)
    for l in key_layers:
        d = data.get(f'L{l}')
        if d:
            es = d['mean_effective_support']
            pr = d['pr']
            ratio = pr / max(es, 1)
            print(f'  L{l:>4} {pr:>6.1f} {d["n_95var"]:>5} {d["n_above_1pct"]:>5} {d["mean_entropy"]:>8.2f} {es:>8.0f} {d["wu_alignment"]:>7.3f} {ratio:>7.1f}')
    print()
