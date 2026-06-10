"""Verify Phase 439-442 results against user analysis"""
import json, os, glob

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

# Phase 439
print("=" * 60)
print("PHASE 439: Multi-head ablation")
print("=" * 60)
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    d = load_json(f'results/phase439_multihead_ablation/{mn}_phase439_r1.json')
    if not d:
        print(f"  {mn}: NO DATA")
        continue
    print(f"\n  {mn}:")
    r = d.get('results', {})
    for obj_name, obj_data in r.items():
        print(f"    {obj_name}:")
        # Check structure
        if isinstance(obj_data, dict):
            for k, v in obj_data.items():
                if k in ['summary', 'config']:
                    continue
                if isinstance(v, dict):
                    ns = v.get('norm_score', '?')
                    ds = v.get('direction_cos', '?')
                    rs = v.get('readout_score', '?')
                    if ns != '?' or ds != '?':
                        print(f"      {k}: norm={ns}, dir_cos={ds}, readout={rs}")

# Phase 440
print("\n" + "=" * 60)
print("PHASE 440: Alpha sweep mediation")
print("=" * 60)
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    d = load_json(f'results/phase440_alpha_sweep/{mn}_phase440_r1.json')
    if not d:
        print(f"  {mn}: NO DATA")
        continue
    print(f"\n  {mn}:")
    r = d.get('results', {})
    for obj_name, obj_data in r.items():
        print(f"    {obj_name}:")
        if isinstance(obj_data, dict):
            for alpha_key, alpha_data in obj_data.items():
                if alpha_key.startswith('alpha_'):
                    med = alpha_data.get('mediation', '?')
                    cs = alpha_data.get('cat_shift', '?')
                    print(f"      {alpha_key}: mediation={med}, cat_shift={cs}")

# Phase 441
print("\n" + "=" * 60)
print("PHASE 441: Object-attribute binding")
print("=" * 60)
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    d = load_json(f'results/phase441_object_attribute_binding/{mn}_phase441_r1.json')
    if not d:
        print(f"  {mn}: NO DATA")
        continue
    print(f"\n  {mn}:")
    r = d.get('results', {})
    for test_name, test_data in r.items():
        print(f"    {test_name}:")
        if isinstance(test_data, dict):
            for k, v in test_data.items():
                if isinstance(v, (int, float)):
                    print(f"      {k} = {v}")
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            print(f"      {k}/{kk} = {vv}")

# Phase 442
print("\n" + "=" * 60)
print("PHASE 442: Cross-category transfer")
print("=" * 60)
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    d = load_json(f'results/phase442_cross_category_transfer/{mn}_phase442_r1.json')
    if not d:
        print(f"  {mn}: NO DATA")
        continue
    print(f"\n  {mn}:")
    r = d.get('results', {})
    for test_name, test_data in r.items():
        print(f"    {test_name}:")
        if isinstance(test_data, dict):
            for k, v in test_data.items():
                if isinstance(v, (int, float)):
                    print(f"      {k} = {v}")
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            print(f"      {k}/{kk} = {vv}")
