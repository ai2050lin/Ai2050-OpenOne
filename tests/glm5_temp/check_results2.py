"""详细检查Phase 434-438结果"""
import json, os

base = "d:/Ai2050/TransformerLens-Project/results"

# Phase 434
print("=" * 60)
print("Phase 434: Head Causal Ablation")
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    fp = f"{base}/phase434_head_causal_ablation/{mn}_phase434_r1.json"
    if os.path.exists(fp):
        d = json.load(open(fp, 'r', encoding='utf-8'))
        print(f"\n--- {mn} (n_layers={d.get('n_layers')}, n_heads={d.get('n_heads')}) ---")
        per_obj = d.get('per_object', {})
        for obj, r in per_obj.items():
            cs = r.get('causal_scores', [])
            if cs:
                cand = [s for s in cs if s.get('is_candidate', False)]
                ctrl = [s for s in cs if not s.get('is_candidate', False)]
                if cand:
                    cand_avg = sum(s['causal_score'] for s in cand) / len(cand)
                    ctrl_avg = sum(s['causal_score'] for s in ctrl) / len(ctrl) if ctrl else 0
                    print(f"  {obj}: cand_avg={cand_avg:.4f}, ctrl_avg={ctrl_avg:.4f}, top3_cand={sorted([s['causal_score'] for s in cand], reverse=True)[:3]}")

# Phase 437
print("\n" + "=" * 60)
print("Phase 437: Category-Property Mediation")
for fn in ['qwen3_phase437_r1.json', 'glm4_phase437_r1.json', 'glm4_phase437_r2.json',
           'qwen3_phase437b_r2.json', 'glm4_phase437b_r2.json']:
    fp = f"{base}/phase437_category_property_mediation/{fn}"
    if os.path.exists(fp):
        d = json.load(open(fp, 'r', encoding='utf-8'))
        print(f"\n--- {fn} ---")
        per_test = d.get('per_test', d.get('tests', {}))
        for k, r in per_test.items():
            med = r.get('mediation_score', r.get('avg_mediation', 'N/A'))
            if isinstance(med, (int, float)):
                print(f"  {k}: mediation={med:.3f}")
            else:
                print(f"  {k}: mediation={med}, keys={list(r.keys())[:5]}")

# Phase 438
print("\n" + "=" * 60)
print("Phase 438: Cross-Object Transport")
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    fp = f"{base}/phase438_cross_object_transport/{mn}_phase438_r1.json"
    if os.path.exists(fp):
        d = json.load(open(fp, 'r', encoding='utf-8'))
        print(f"\n--- {mn} ---")
        same = d.get('same_category', {})
        cross = d.get('cross_category', {})
        print(f"  Same-category transfers: {len(same)}")
        for k, r in same.items():
            ts = r.get('transfer_score', r.get('avg_transfer', 'N/A'))
            print(f"    {k}: transfer={ts}")
        print(f"  Cross-category transfers: {len(cross)}")
        for k, r in cross.items():
            ts = r.get('transfer_score', r.get('avg_transfer', 'N/A'))
            print(f"    {k}: transfer={ts}")

# Phase 436
print("\n" + "=" * 60)
print("Phase 436: Contextualized Attribute")
for mn in ['qwen3', 'glm4', 'deepseek7b']:
    fp = f"{base}/phase436_contextualized_attribute/{mn}_phase436_r1.json"
    if os.path.exists(fp):
        d = json.load(open(fp, 'r', encoding='utf-8'))
        print(f"\n--- {mn} ---")
        per_attr = d.get('per_attribute', {})
        for attr, r in per_attr.items():
            cos_we = r.get('cos_with_WE', r.get('cos_contextual_vs_WE', 'N/A'))
            cos_wu = r.get('cos_with_WU', r.get('cos_contextual_vs_WU', 'N/A'))
            inject = r.get('injection_results', {})
            best_layer = r.get('best_inject_layer', 'N/A')
            best_switch = r.get('best_switch_score', 'N/A')
            print(f"  {attr}: cos_WE={cos_we}, cos_WU={cos_wu}, best_L={best_layer}, best_sw={best_switch}")
