"""深入检查结果数据结构"""
import json, os

base = "d:/Ai2050/TransformerLens-Project/results"

# Phase 434 - 检查per_object结构
fp = f"{base}/phase434_head_causal_ablation/qwen3_phase434_r1.json"
d = json.load(open(fp, 'r', encoding='utf-8'))
per_obj = d.get('per_object', {})
if per_obj:
    first_obj = list(per_obj.keys())[0]
    r = per_obj[first_obj]
    print(f"Phase 434 qwen3: first_obj={first_obj}")
    print(f"  keys: {list(r.keys())[:15]}")
    cs = r.get('causal_scores', [])
    if cs:
        print(f"  causal_scores[0] keys: {list(cs[0].keys())}")
        print(f"  causal_scores[0]: {cs[0]}")
    else:
        # Try other keys
        for k, v in r.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                print(f"  {k}[0] keys: {list(v[0].keys())}")
                print(f"  {k}[0]: {v[0]}")
                break

# Phase 437 - 检查per_test结构
fp = f"{base}/phase437_category_property_mediation/qwen3_phase437_r1.json"
d = json.load(open(fp, 'r', encoding='utf-8'))
per_test = d.get('per_test', {})
if per_test:
    first_test = list(per_test.keys())[0]
    r = per_test[first_test]
    print(f"\nPhase 437 qwen3: first_test={first_test}")
    print(f"  keys: {list(r.keys())[:15]}")
    # Find mediation-related keys
    for k, v in r.items():
        if 'med' in k.lower() or 'delta' in k.lower():
            print(f"  {k}: {v}")

# Phase 437b
fp = f"{base}/phase437_category_property_mediation/qwen3_phase437b_r2.json"
d = json.load(open(fp, 'r', encoding='utf-8'))
tests = d.get('tests', {})
if tests:
    first_test = list(tests.keys())[0]
    r = tests[first_test]
    print(f"\nPhase 437b qwen3: first_test={first_test}")
    print(f"  keys: {list(r.keys())}")
    avg_med = r.get('avg_mediation_a2', 'N/A')
    print(f"  avg_mediation_a2: {avg_med}")
    per_obj = r.get('per_object', {})
    for obj, or_ in per_obj.items():
        print(f"  {obj}: mediation={or_.get('mediation_score', or_.get('mediation_a2', 'N/A'))}")

# Phase 438
fp = f"{base}/phase438_cross_object_transport/qwen3_phase438_r1.json"
d = json.load(open(fp, 'r', encoding='utf-8'))
same = d.get('same_category', {})
if same:
    first_k = list(same.keys())[0]
    r = same[first_k]
    print(f"\nPhase 438 qwen3: first_same={first_k}")
    print(f"  keys: {list(r.keys())}")
    for k, v in r.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
        elif isinstance(v, dict):
            print(f"  {k}: {list(v.keys())[:5]}")
        elif isinstance(v, str) and len(v) < 80:
            print(f"  {k}: {v}")

cross = d.get('cross_category', {})
if cross:
    first_k = list(cross.keys())[0]
    r = cross[first_k]
    print(f"\nPhase 438 qwen3: first_cross={first_k}")
    print(f"  keys: {list(r.keys())}")
    for k, v in r.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
