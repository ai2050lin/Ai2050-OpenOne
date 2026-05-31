import json
import numpy as np

print("=" * 80)
print("Phase 314: GRCM CROSS-MODEL COMPARISON")
print("=" * 80)

# Key comparison: Mantel correlation by layer
print("\n--- Mantel Correlation by Layer ---")
print(f"{'Model':<12} {'Layer':>6} {'Mantel_r':>10} {'NN_overlap':>12} {'Conflicts':>10}")
print("-" * 55)

for mn in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        with open(f'results/phase314_GRCM/{mn}_GRCM.json') as f:
            d = json.load(f)
    except:
        continue
    for li in d['test_layers']:
        lr = d['layers'].get(str(li), {})
        if not lr:
            continue
        r = lr.get('mantel_correlation', 0)
        nn = lr.get('neighborhood_overlap_k5', 0)
        conflicts = lr.get('conflict_count', 0)
        print(f"{mn:<12} {li:>6} {r:>10.3f} {nn:>12.3f} {conflicts:>10}")

# Relation type preservation comparison
print("\n\n--- Relation Type Preservation (ratio = random_dist / related_dist) ---")
print("Higher ratio = stronger preservation of this relation type")
print()

for rt in ["same_class", "hypernym", "negation", "antonym", "operator_similar", "cross_category", "attribute", "function"]:
    print(f"\n  {rt}:")
    print(f"  {'Model':<12} {'Layer':>6} {'Related_dist':>14} {'Random_dist':>14} {'Ratio':>8}")
    print(f"  " + "-" * 60)
    for mn in ['qwen3', 'glm4', 'deepseek7b']:
        try:
            with open(f'results/phase314_GRCM/{mn}_GRCM.json') as f:
                d = json.load(f)
        except:
            continue
        for li in d['test_layers']:
            lr = d['layers'].get(str(li), {})
            if not lr:
                continue
            rt_data = lr.get('relation_type_preservation', {}).get(rt, {})
            if rt_data:
                print(f"  {mn:<12} {li:>6} {rt_data['mean_related_dist']:>14.3f} "
                      f"{rt_data['mean_random_dist']:>14.3f} {rt_data['ratio']:>8.2f}")

# Cluster shared_pc1_var across layers
print("\n\n--- Cluster Shared PC1 Variance by Layer ---")
for cluster_name in ["fruit", "animal", "tool", "emotion_pos", "emotion_neg"]:
    print(f"\n  {cluster_name}:")
    print(f"  {'Model':<12} {'Layer':>6} {'shared_pc1_var':>16} {'n_members':>10}")
    print(f"  " + "-" * 50)
    for mn in ['qwen3', 'glm4', 'deepseek7b']:
        try:
            with open(f'results/phase314_GRCM/{mn}_GRCM.json') as f:
                d = json.load(f)
        except:
            continue
        for li in d['test_layers']:
            lr = d['layers'].get(str(li), {})
            if not lr:
                continue
            cd = lr.get('cluster_decomposition', {}).get(cluster_name, {})
            if cd:
                print(f"  {mn:<12} {li:>6} {cd['shared_pc1_var']:>16.3f} {cd['n_members']:>10}")

# Key finding: relation type preservation hierarchy
print("\n\n--- RELATION TYPE PRESERVATION HIERARCHY (Mid-Layer) ---")
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        with open(f'results/phase314_GRCM/{mn}_GRCM.json') as f:
            d = json.load(f)
    except:
        continue
    
    # Use mid-layer
    mid_li = d['test_layers'][len(d['test_layers'])//2]
    lr = d['layers'].get(str(mid_li), {})
    if not lr:
        continue
    
    rt_data = lr.get('relation_type_preservation', {})
    sorted_rts = sorted(rt_data.items(), key=lambda x: x[1].get('ratio', 0), reverse=True)
    
    print(f"\n  {mn.upper()} (Layer {mid_li}):")
    for rt, data in sorted_rts:
        bar = "█" * int(data['ratio'] * 2) if data['ratio'] > 0 else ""
        print(f"    {rt:<20} ratio={data['ratio']:>6.2f}  {bar}")

# Cross-model comparison of negation preservation
print("\n\n--- NEGATION RELATION PRESERVATION ACROSS MODELS ---")
print("Key question: Is negation (happy vs not_happy) preserved as strongly as antonym (happy vs sad)?")
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    try:
        with open(f'results/phase314_GRCM/{mn}_GRCM.json') as f:
            d = json.load(f)
    except:
        continue
    
    for li in d['test_layers']:
        lr = d['layers'].get(str(li), {})
        if not lr:
            continue
        neg_ratio = lr.get('relation_type_preservation', {}).get('negation', {}).get('ratio', 0)
        ant_ratio = lr.get('relation_type_preservation', {}).get('antonym', {}).get('ratio', 0)
        op_ratio = lr.get('relation_type_preservation', {}).get('operator_similar', {}).get('ratio', 0)
        if neg_ratio > 0:
            print(f"  {mn:<12} L{li:>2}: negation={neg_ratio:.2f}, antonym={ant_ratio:.2f}, "
                  f"operator_similar={op_ratio:.2f}, neg/ant={neg_ratio/ant_ratio:.2f}" if ant_ratio > 0 else "")
