"""Phase 458 R2 Results Summary"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

for model in ['qwen3', 'glm4', 'deepseek7b']:
    key = model
    f = f'results/glm5/phase458_{key}_r2.json'
    with open(f, encoding='utf-8') as fh:
        d = json.load(fh)
    
    print(f'\n===== {model.upper()} R2 =====')
    
    # Exp1
    e1 = d.get('exp1_relation_slot_purification', {})
    print('--- Exp1: Relation Slot Purification ---')
    for rel, rd in e1.items():
        if not isinstance(rd, dict): continue
        consist = {}
        for cat, cd in rd.items():
            if isinstance(cd, dict) and '_summary' in cd:
                consist[cat] = cd['_summary'].get('avg_direction_consistency')
        if consist:
            print(f'  {rel}: {consist}')
    
    # Exp2
    e2 = d.get('exp2_negation_scope_decomposition', {})
    print('--- Exp2: Negation Scope ---')
    for cat, cd in e2.items():
        if not isinstance(cd, dict) or '_summary' not in cd: continue
        s = cd['_summary']
        aff = s.get('affirmative', {}).get('avg_margin', '?')
        sn = s.get('simple_neg', {}).get('avg_margin', '?')
        ea = s.get('explicit_alt', {}).get('avg_margin', '?')
        cn = s.get('contrast_neg', {}).get('avg_margin', '?')
        sc = s.get('scope_control', {}).get('avg_margin', '?')
        dn = s.get('double_neg', {}).get('avg_margin', '?')
        print(f'  {cat}: aff={aff}, simple_neg={sn}, explicit_alt={ea}, contrast_neg={cn}, scope_ctrl={sc}, double_neg={dn}')
    
    # Exp3
    e3 = d.get('exp3_multihop_knowledge_path', {})
    print('--- Exp3: Multi-Hop ---')
    for pn, pd in e3.items():
        if pn.startswith('_'): continue
        if isinstance(pd, dict) and '_analysis' in pd:
            a = pd['_analysis']
            print(f'  {pn}: 2hop={a.get("2hop_margin")}, 1hop={a.get("1hop_margin")}, 0hop={a.get("0hop_margin")}, 2vs0={a.get("2hop_vs_0hop")}')
    
    # Exp4
    e4 = d.get('exp4_path_split_large_sample', {})
    if '_summary' in e4:
        print(f'--- Exp4: Path Split --- {e4["_summary"]}')
        for lk, lv in e4.items():
            if lk.startswith('L') and isinstance(lv, dict):
                flip = lv.get('_flip', {})
                if flip.get('any_flipped'):
                    fa = lv.get('fruit', {})
                    aa = lv.get('animal', {})
                    print(f'  {lk}: fruit=[a={fa.get("attn_type")},m={fa.get("mlp_type")}] animal=[a={aa.get("attn_type")},m={aa.get("mlp_type")}]')
    
    # Exp6
    e6 = d.get('exp6_candidate_vocab_control', {})
    print('--- Exp6: Vocab Control ---')
    for cat, cd in e6.items():
        if isinstance(cd, dict) and '_summary' in cd:
            s = cd['_summary']
            print(f'  {cat}: full={s.get("avg_full")}, single={s.get("avg_single")}, boot={s.get("avg_bootstrap")}, consistent={s.get("direction_consistent")}')
