import json, sys, numpy as np
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

models = ['qwen3', 'deepseek7b']
results = {}
for m in models:
    try:
        r = json.load(open(f'results/subspace_topology/exp4c_correct_subspace_{m}.json', encoding='utf-8'))
        results[m] = r
    except:
        print(f"Warning: no results for {m}")

print("=" * 80)
print("CROSS-MODEL COMPARISON: Subspace Overlap by Semantic Relation")
print("=" * 80)

for m in models:
    r = results[m]
    sim = r['similarity_function']
    mid = sim['mid_layer']
    
    print(f"\n--- {m} (mid_layer=L{mid}) ---")
    for rel in ['hyponym', 'synonym', 'antonym', 'associated', 'unrelated']:
        if rel in sim['relation_summary']:
            rs = sim['relation_summary'][rel]
            print("  {:12s}: overlap={:.3f}+-{:.3f} proj={:.3f} cca={:.3f} cos={:.3f}".format(
                rel, rs['mean_overlap'], rs['std_overlap'],
                rs['mean_proj'], rs['mean_cca'], rs['mean_cos']))

print("\n" + "=" * 80)
print("LAYER EVOLUTION COMPARISON")
print("=" * 80)

# 收集每个模型每层每关系的overlap
for m in models:
    r = results[m]
    print(f"\n--- {m} ---")
    layers_data = {}
    for pk, pr in r['pair_results'].items():
        rel = pr['pair_info']['relation']
        for lk, ld in pr['layers'].items():
            if 'error' not in ld:
                if lk not in layers_data:
                    layers_data[lk] = {}
                if rel not in layers_data[lk]:
                    layers_data[lk][rel] = []
                layers_data[lk][rel].append(ld.get('overlap', 0))
    
    for lk in sorted(layers_data.keys(), key=int):
        parts = []
        for rel in ["hyponym", "synonym", "antonym", "associated", "unrelated"]:
            if rel in layers_data[lk]:
                m_val = sum(layers_data[lk][rel]) / len(layers_data[lk][rel])
                parts.append("{}={:.3f}".format(rel[:4], m_val))
        print("  L{}: {}".format(lk, " | ".join(parts)))

print("\n" + "=" * 80)
print("KEY FINDING: Overlap Ranking Consistency")
print("=" * 80)

for m in models:
    r = results[m]
    sim = r['similarity_function']
    print(f"\n--- {m} ---")
    rs = sim['relation_summary']
    ranking = sorted(rs.keys(), key=lambda x: -rs[x]['mean_overlap'])
    print("  Overlap ranking: " + " > ".join(ranking))
    print("  Values: " + " > ".join([f"{rs[r]['mean_overlap']:.3f}" for r in ranking]))

print("\n" + "=" * 80)
print("Per-Pair Top/Bottom 5")
print("=" * 80)

for m in models:
    r = results[m]
    sim = r['similarity_function']
    pairs = sorted(sim['per_pair'], key=lambda x: -x['overlap'])
    print(f"\n--- {m} Top 5 ---")
    for p in pairs[:5]:
        print("  {:20s} {} overlap={:.3f} proj={:.3f}".format(
            p['pair_key'], p['relation'], p['overlap'], p['avg_proj']))
    print(f"--- {m} Bottom 5 ---")
    for p in pairs[-5:]:
        print("  {:20s} {} overlap={:.3f} proj={:.3f}".format(
            p['pair_key'], p['relation'], p['overlap'], p['avg_proj']))

print("\n" + "=" * 80)
print("Overlap vs Semantic Distance Function")
print("=" * 80)

# 合并两个模型的数据
all_data = defaultdict(lambda: defaultdict(list))
for m in models:
    r = results[m]
    sim = r['similarity_function']
    for sd in sim['per_pair']:
        all_data[sd['relation']]['overlap'].append(sd['overlap'])
        all_data[sd['relation']]['distance'].append(sd['distance'])

print("\nRelation        | Avg Overlap | Semantic Distance | Ratio")
print("-" * 70)
for rel in ['hyponym', 'synonym', 'antonym', 'associated', 'unrelated']:
    if rel in all_data:
        avg_overlap = np.mean(all_data[rel]['overlap'])
        avg_dist = np.mean(all_data[rel]['distance'])
        ratio = avg_overlap / avg_dist if avg_dist > 0 else 0
        print("  {:12s}  | {:.3f}      | {:.1f}               | {:.3f}".format(
            rel, avg_overlap, avg_dist, ratio))
