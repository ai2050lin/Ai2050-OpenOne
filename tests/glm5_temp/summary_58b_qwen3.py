import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

r = json.load(open('results/subspace_topology/exp4b_backbone_decode_qwen3.json', encoding='utf-8'))

print('=== Qwen3 Phase 58b: Shared Ratio by Relation ===')
sim = r['similarity_function']
mid = sim['mid_layer']

print(f'\nMid layer: L{mid}')
for rel in ['hyponym', 'synonym', 'antonym', 'associated', 'unrelated']:
    if rel in sim['relation_summary']:
        rs = sim['relation_summary'][rel]
        print("  {:12s}: shared={:.3f}+-{:.3f} cos={:.3f}+-{:.3f} delta={:.3f} overlap={:.3f} n={}".format(
            rel, rs['mean_shared'], rs['std_shared'],
            rs['mean_cos'], rs['std_cos'],
            rs['mean_delta'], rs['mean_overlap'], rs['n']))

print('\n=== Per-Pair Detail (sorted by shared_ratio) ===')
for sd in sorted(sim['per_pair'], key=lambda x: -x['shared_ratio']):
    print("  {:20s} {} shared={:.3f} cos={:.3f} delta={:.3f} overlap={:.3f}".format(
        sd['pair_key'], sd['relation'],
        sd['shared_ratio'], sd['cos_mean'], sd['delta_unique'], sd['overlap']))

print('\n=== Backbone Variance Ratio ===')
bd = r['backbone_decode']
for lk in sorted(bd.keys(), key=int):
    print("  L{}: var={:.3f} mean_score={:.3f}".format(
        lk, bd[lk]['backbone_var_ratio'], 
        bd[lk]['neuron_attribution']['mean_backbone_score']))

print('\n=== Backbone vs Specific Decode (L9, L15, L27) ===')
for lk in ['9', '15', '27']:
    if lk in bd:
        layer_bd = bd[lk]
        print("\n  L{} backbone_var={:.3f}:".format(lk, layer_bd['backbone_var_ratio']))
        for d_info in layer_bd['backbone_decoded'][:3]:
            tw = [t['token'].strip()[:12] for t in d_info['top_words'][:5]]
            print("    PC{} var={:.4f}: {}".format(d_info['direction'], d_info['var_explained'], tw))
        for d_info in layer_bd.get('specific_decoded', [])[:2]:
            tw = [t['token'].strip()[:12] for t in d_info['top_words'][:5]]
            print("    Spec{} var={:.4f}: {}".format(d_info['direction'], d_info['var_explained'], tw))

print('\n=== Layer Evolution: shared_ratio by relation ===')
# 需要逐层逐对分析
layers_data = {}
for pk, pr in r['pair_results'].items():
    rel = pr['pair_info']['relation']
    for lk, ld in pr['layers'].items():
        if 'error' not in ld:
            if lk not in layers_data:
                layers_data[lk] = {}
            if rel not in layers_data[lk]:
                layers_data[lk][rel] = []
            layers_data[lk][rel].append(ld.get('avg_shared_ratio', 0))

for lk in sorted(layers_data.keys(), key=int):
    print("  L{}:".format(lk), end="")
    for rel in ['hyponym', 'synonym', 'antonym', 'associated', 'unrelated']:
        if rel in layers_data[lk]:
            mean_s = sum(layers_data[lk][rel]) / len(layers_data[lk][rel])
            print(" {}={:.3f}".format(rel[:4], mean_s), end="")
    print()
