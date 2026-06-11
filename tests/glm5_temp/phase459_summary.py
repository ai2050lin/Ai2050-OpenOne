"""Phase 459 R2 Summary Analysis"""
import json, sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

for model in ['qwen3', 'glm4', 'deepseek7b']:
    f = f'results/glm5/phase459_{model}_r2.json'
    with open(f, encoding='utf-8') as fh:
        d = json.load(fh)
    
    print(f'\n===== {model.upper()} R2 =====')
    
    # Exp1: is_a sub-slot
    e1 = d.get('exp1_isa_subslot', {})
    print('--- Exp1: is_a Sub-Slot ---')
    for cat, cd in e1.items():
        if isinstance(cd, dict) and '_summary' in cd:
            s = cd['_summary']
            nc = s['n_clusters']
            bt = s['btcat_alone']
            print(f'  {cat}: {nc} clusters, btcat_alone={bt}')
            for cl_name, cl_info in s.get('clusters', {}).items():
                short = [t.replace('The {obj} ', '').replace('{obj} ', '').replace('the {obj} ', '')[:25] for t in cl_info['templates']]
                am = cl_info['avg_margin']
                print(f'    {cl_name}: {short} avg_m={am}')
    
    # Exp2: Negation closure
    e2 = d.get('exp2_negation_closure', {})
    print('--- Exp2: Negation Closure ---')
    for cat, cd in e2.items():
        if isinstance(cd, dict) and '_summary' in cd:
            s = cd['_summary']
            nfd = s['NegatedFamilyDrop']
            sr = s['ScopeRecovery']
            dnr = s['DoubleNegRecovery']
            alt = s.get('AlternativeRelease', {})
            print(f'  {cat}: NFD={nfd}, SR={sr}, DNR={dnr}, AltRel={alt}')
    
    # Exp3: Multi-hop
    e3 = d.get('exp3_multihop_causal', {})
    print('--- Exp3: Multi-Hop Causal ---')
    for pn, pd in e3.items():
        if isinstance(pd, dict) and '_analysis' in pd:
            a = pd['_analysis']
            m2 = a['2hop_margin']
            m1 = a['1hop_margin']
            m0 = a['0hop_margin']
            d20 = a['2hop_vs_0hop']
            print(f'  {pn}: 2hop={m2}, 1hop={m1}, 0hop={m0}, 2vs0={d20}')
    
    # Exp4: has_part repair
    e4 = d.get('exp4_has_part_repair', {})
    print('--- Exp4: has_part Repair ---')
    for cat, cd in e4.items():
        if isinstance(cd, dict):
            for k in sorted(cd.keys()):
                if k.startswith('_summary_'):
                    s = cd[k]
                    pt = k.replace('_summary_', '')
                    am = s['avg_margin']
                    n = s['n_measurements']
                    print(f'  {cat}/{pt}: avg={am}, n={n}')
    
    # Exp5: Path split
    e5 = d.get('exp5_path_split', {})
    if '_flip_summary' in e5:
        fs = e5['_flip_summary']
        fc = fs['flip_count']
        tc = fs['total_count']
        fr = fs['flip_ratio']
        print(f'--- Exp5: Path Split --- flip={fc}/{tc} ({fr})')
        for lk in sorted(fs.get('flip_details', {}).keys()):
            lv = fs['flip_details'][lk]
            if lv.get('any_flip'):
                fa = lv['fruit_attn']
                fm = lv['fruit_mlp']
                aa = lv['animal_attn']
                am = lv['animal_mlp']
                print(f'  {lk}: fruit_a={fa}/m={fm}, animal_a={aa}/m={am}')
    
    # Exp6: Syntax binding
    e6 = d.get('exp6_syntax_binding', {})
    print('--- Exp6: Syntax Binding ---')
    ap = e6.get('_agent_patient', {})
    for k, v in ap.items():
        fam = v.get('all_fam_logits', {})
        an = fam.get('class_animal')
        fr = fam.get('class_fruit')
        tl = fam.get('class_tool')
        vh = fam.get('class_vehicle')
        print(f'  {k}: animal={an}, fruit={fr}, tool={tl}, vehicle={vh}')
    
    # Syntax: main sentence analysis
    print('  Main sentences:')
    for sk, sv in e6.items():
        if sk.startswith('_'): continue
        if isinstance(sv, dict) and 'active' in sv and 'reversed' in sv:
            af = sv['active'].get('all_fam_logits', {})
            rf = sv['reversed'].get('all_fam_logits', {})
            diff = sv.get('difference', {})
            # Show biggest diffs
            max_diff_key = max(diff, key=lambda x: abs(diff.get(x, 0))) if diff else 'N/A'
            max_diff_val = diff.get(max_diff_key, 0)
            print(f'  {sk[:30]}: max_diff={max_diff_key}({max_diff_val})')
