"""检查Phase 434-438结果"""
import json, os

base = "d:/Ai2050/TransformerLens-Project/results"

# Phase 434
print("=" * 60)
print("Phase 434: Head Causal Ablation")
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    fp = f"{base}/phase434_head_causal_ablation/{mn}_phase434_r1.json"
    if os.path.exists(fp):
        d = json.load(open(fp, 'r', encoding='utf-8'))
        print(f"\n--- {mn} ---")
        if 'results' in d:
            for obj, r in d['results'].items():
                if 'causal_scores' in r:
                    scores = r['causal_scores']
                    cand = [s for s in scores if s.get('is_candidate', False)]
                    ctrl = [s for s in scores if not s.get('is_candidate', False)]
                    if cand:
                        cs = [s['causal_score'] for s in cand]
                        print(f"  {obj}: cand_cs={sum(cs)/len(cs):.3f} (n={len(cand)})")
        elif 'summary' in d:
            print(f"  summary keys: {list(d['summary'].keys())[:10]}")
        else:
            print(f"  top keys: {list(d.keys())[:10]}")
    else:
        print(f"  {mn}: file not found")

# Phase 437
print("\n" + "=" * 60)
print("Phase 437: Category-Property Mediation")
for fn in ['qwen3_phase437_r1.json', 'glm4_phase437_r1.json', 'glm4_phase437_r2.json',
           'qwen3_phase437b_r2.json', 'glm4_phase437b_r2.json']:
    fp = f"{base}/phase437_category_property_mediation/{fn}"
    if os.path.exists(fp):
        d = json.load(open(fp, 'r', encoding='utf-8'))
        print(f"\n--- {fn} ---")
        if 'results' in d:
            for obj, r in d['results'].items():
                med = r.get('mediation_score', r.get('avg_mediation', 'N/A'))
                print(f"  {obj}: mediation={med}")
        elif 'summary' in d:
            s = d['summary']
            print(f"  avg_mediation: {s.get('avg_mediation', 'N/A')}")
        else:
            print(f"  top keys: {list(d.keys())[:10]}")
    else:
        print(f"  {fn}: not found")

# Phase 438
print("\n" + "=" * 60)
print("Phase 438: Cross-Object Transport")
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    fp = f"{base}/phase438_cross_object_transport/{mn}_phase438_r1.json"
    if os.path.exists(fp):
        d = json.load(open(fp, 'r', encoding='utf-8'))
        print(f"\n--- {mn} ---")
        if 'results' in d:
            for k, r in d['results'].items():
                ts = r.get('transfer_score', r.get('avg_transfer', 'N/A'))
                print(f"  {k}: transfer={ts}")
        elif 'summary' in d:
            s = d['summary']
            print(f"  summary: {s}")
        else:
            print(f"  top keys: {list(d.keys())[:10]}")
    else:
        print(f"  {mn}: not found")

# Phase 436
print("\n" + "=" * 60)
print("Phase 436: Contextualized Attribute")
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    fp = f"{base}/phase436_contextualized_attribute/{mn}_phase436_r1.json"
    if os.path.exists(fp):
        d = json.load(open(fp, 'r', encoding='utf-8'))
        print(f"\n--- {mn} ---")
        if 'results' in d:
            for attr, r in d['results'].items():
                print(f"  {attr}: {list(r.keys())[:8]}")
        else:
            print(f"  top keys: {list(d.keys())[:10]}")
    else:
        print(f"  {mn}: not found")
