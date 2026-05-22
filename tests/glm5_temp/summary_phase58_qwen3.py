import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open('results/subspace_topology/exp4_backbone_decode_qwen3.json', encoding='utf-8'))

print('=== Part 1: Shared Ratio by Relation (mid layer L15) ===')
sim = r['similarity_function']
for sd in sorted(sim['per_pair_data'], key=lambda x: x['semantic_distance']):
    print("  {:20s} dist={} shared={:.3f} cos={:.3f} delta_u={:.3f}".format(
        sd['pair_key'], sd['semantic_distance'], sd['shared_ratio'], 
        sd['cos_mean'], sd['delta_unique_ratio']))

print()
print('=== Relation Summary ===')
for rel in ['hyponym', 'synonym', 'antonym', 'associated', 'unrelated']:
    if rel in sim['relation_summary']:
        rs = sim['relation_summary'][rel]
        print("  {:12s}: shared={:.3f}+-{:.3f} cos={:.3f} delta_u={:.3f} n={}".format(
            rel, rs['mean_shared_ratio'], rs['std_shared_ratio'], 
            rs['mean_cos'], rs['mean_delta_unique'], rs['n_pairs']))

print()
print('=== Backbone Variance Ratio ===')
bd = r['backbone_decode']
for lk in sorted(bd.keys(), key=int):
    print("  L{}: backbone_var={:.3f} samples={}".format(
        lk, bd[lk]['backbone_var_ratio'], bd[lk]['n_total_samples']))

print()
print('=== Neuron Attribution (L15) ===')
na = bd.get('15', {}).get('neuron_attribution', {})
if na:
    print("  Mean backbone score: {:.4f}".format(na['mean_backbone_score']))
    print("  Median backbone score: {:.4f}".format(na['median_backbone_score']))
    top10 = na['top_backbone_neurons'][:10]
    for n in top10:
        s = na['backbone_score'][n]
        print("    Neuron {}: backbone_score={:.4f}".format(n, s))
    
print()
print('=== Backbone Decode: Top Words by PC ===')
for lk in ['9', '15', '27', '33']:
    if lk in bd:
        layer_bd = bd[lk]
        print("  L{} (backbone_var={:.3f}):".format(lk, layer_bd['backbone_var_ratio']))
        for d_info in layer_bd['backbone_decoded'][:3]:
            tw = [t['token'].strip().replace('\n',' ') for t in d_info['top_words'][:8]]
            print("    PC{} var={:.4f}: {}".format(
                d_info['direction_idx'], d_info['var_explained'], ' | '.join(tw)))
        for d_info in layer_bd.get('specific_decoded', [])[:2]:
            tw = [t['token'].strip().replace('\n',' ') for t in d_info['top_words'][:8]]
            print("    Spec{} var={:.4f}: {}".format(
                d_info['direction_idx'], d_info['var_explained'], ' | '.join(tw)))
