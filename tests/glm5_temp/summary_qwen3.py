import json

r = json.load(open('results/subspace_topology/exp3_reuse_diff_qwen3.json', encoding='utf-8'))

print('=== Qwen3 Concept Pairs ===')
for pk, pd in r['concept_pairs'].items():
    name = pd['pair_name']
    print(f'\n{name} ({pk}):')
    for lk, ld in sorted(pd['layers'].items(), key=lambda x: int(x[0])):
        print(f'  L{lk}: cos={ld["cos_mean"]:.3f} overlap={ld["subspace_overlap"]:.3f} '
              f'shared_A={ld["shared_ratio_A"]:.3f} shared_B={ld["shared_ratio_B"]:.3f} '
              f'delta_unique={ld["unique_delta_ratio"]:.3f}')

print('\n=== Task Pair ===')
for pk, pd in r['task_pair'].items():
    name = pd['pair_name']
    print(f'\n{name}:')
    for lk, ld in sorted(pd['layers'].items(), key=lambda x: int(x[0])):
        print(f'  L{lk}: cos={ld["cos_mean"]:.3f} overlap={ld["subspace_overlap"]:.3f} '
              f'shared={ld["avg_shared_ratio"]:.3f} delta_unique={ld["unique_delta_ratio"]:.3f}')

print('\n=== Logic Pairs ===')
for pk, pd in r['logic_pairs'].items():
    name = pd['pair_name']
    print(f'\n{name}:')
    for lk, ld in sorted(pd['layers'].items(), key=lambda x: int(x[0])):
        print(f'  L{lk}: cos={ld["cos_mean"]:.3f} overlap={ld["subspace_overlap"]:.3f} '
              f'shared={ld["avg_shared_ratio"]:.3f} delta_unique={ld["unique_delta_ratio"]:.3f}')

print('\n=== Cross-Concept Backbone ===')
bb = r.get('cross_concept_backbone', {})
for k, v in bb.get('backbone_overlaps', {}).items():
    print(f'  {k}: cos={v:.4f}')
for k, v in bb.get('pair_strengths', {}).items():
    print(f'  strength_{k}: {v:.4f}')
