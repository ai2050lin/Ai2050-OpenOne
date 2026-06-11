"""Phase 460 R2 Result Summary"""
import sys, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

for model in ['qwen3', 'glm4', 'deepseek7b']:
    f = f'results/glm5/phase460_{model}_r2.json'
    try:
        with open(f, encoding='utf-8') as fh:
            d = json.load(fh)
    except:
        print(f"\n!!! {model} R2 not found")
        continue
    
    mi = d.get('model_info', {})
    print(f"\n{'='*70}")
    print(f"  {model.upper()} R2  ({mi.get('class','?')}, {mi.get('n_layers','?')}L, d={mi.get('d_model','?')})")
    print(f"{'='*70}")
    
    # ===== Exp1: Code Decomposition =====
    e1 = d.get('exp1_code_decomposition', {})
    
    # Part1: Relation Access Code
    rac = e1.get('relation_access_code', {})
    print("\n--- Exp1 Part1: RelationAccessCode (同对象不同关系的L0余弦) ---")
    for obj_key in list(rac.keys())[:2]:
        obj_data = rac[obj_key]
        avg_diffs = obj_data.get('avg_relation_diff_per_layer', {})
        best = obj_data.get('best_relation_split_layer', '?')
        sample_layers = list(avg_diffs.keys())[::2]
        print(f"  {obj_key}: best_split={best}, sample_diffs={[f'{k}={v}' for k,v in list(avg_diffs.items())[::3]]}")
    
    # Part2: Class Shared/Private
    csp = e1.get('class_shared_private', {})
    print("\n--- Exp1 Part2: ClassShared vs Private (同类别内vs跨类别余弦) ---")
    for cat in csp:
        cd = csp[cat]
        within = cd.get('within_class_cosine', {})
        across = cd.get('across_class_cosine', {})
        best = cd.get('best_separation_layer', '?')
        score = cd.get('separation_scores', {}).get(best, '?')
        # 打印首中末层
        layers = sorted(within.keys(), key=lambda x: int(x[1:]))
        if layers:
            print(f"  {cat}: best_sep={best}(score={score})")
            for lk in [layers[0], layers[len(layers)//2], layers[-1]]:
                w = within.get(lk, '?')
                a = across.get(lk, '?')
                print(f"    {lk}: within={w}, across={a}")
    
    # Part3: SlotSubTypeCode
    ssc = e1.get('slot_subtype_code', {})
    print("\n--- Exp1 Part3: SlotSubTypeCode (is_a变体余弦) ---")
    for obj_key in list(ssc.keys())[:2]:
        cd = ssc[obj_key]
        best = cd.get('best_slot_split_layer', '?')
        cat_diffs = cd.get('category_vs_kindof_diff', {})
        sample = list(cat_diffs.items())[::3]
        print(f"  {obj_key}: best_slot={best}, category_diffs_sample={sample}")
    
    # PCA
    pca = e1.get('pca_decomposition', {})
    if pca and 'top_eigenvalues' in pca:
        print(f"\n  PCA: top-3 eigenvalues={pca['top_eigenvalues'][:3]}, "
              f"class_corr={list(pca.get('pc_class_correlation',{}).values())[:3]}")
    
    # ===== Exp2: Recombination =====
    e2 = d.get('exp2_recombination', {})
    recomb = e2.get('recombination', {})
    print("\n--- Exp2: Shared/Private Recombination (方向注入效果) ---")
    for key in sorted(recomb.keys()):
        rd = recomb[key]
        fc = rd.get('fruit_logit_change', 0)
        tc = rd.get('tool_logit_change', 0)
        ac = rd.get('all_fam_logits', {})
        # 只打印关键变化
        if abs(fc) > 0.05 or abs(tc) > 0.05:
            print(f"  {key}: fruit_Δ={fc}, tool_Δ={tc}")
    
    # ===== Exp3: Multi-Hop Patch =====
    e3 = d.get('exp3_multihop_patch', {})
    print("\n--- Exp3: Multi-Hop Patch Causal ---")
    for pn, pd in e3.items():
        if not isinstance(pd, dict): continue
        analysis = pd.get('_analysis', {})
        if not analysis: continue
        print(f"  {pn}: 2hop={analysis.get('2hop_margin','?')}, 0hop={analysis.get('0hop_margin','?')}, "
              f"2vs0={analysis.get('2hop_vs_0hop','?')}, replace_effective={analysis.get('replace_effective','?')}")
        # 打印替换效果
        for k, v in pd.items():
            if k.startswith('replace_') and isinstance(v, dict):
                mc = v.get('margin_change', 0)
                if abs(mc) > 0.5:
                    print(f"    {k}: margin_change={mc}")
    
    # ===== Exp4: Negation Tracing =====
    e4 = d.get('exp4_negation_tracing', {})
    print("\n--- Exp4: Negation Layer Tracing ---")
    for cat in ['fruit', 'animal']:
        if cat not in e4: continue
        cd = e4[cat]
        summary = cd.get('_summary', {})
        peak = summary.get('peak_negation_layer', '?')
        avg_norms = summary.get('avg_neg_delta_norms', {})
        sample = list(avg_norms.items())[::3]
        print(f"  {cat}: peak_negation={peak}, delta_norms_sample={sample}")
    
    # ===== Exp5: Syntax Extended =====
    e5 = d.get('exp5_syntax_extended', {})
    vp = e5.get('verb_patient_routing', {})
    print(f"\n--- Exp5: Syntax Role Extended ---")
    print(f"  verb→patient match rate: {vp.get('match_rate','?')}")
    
    # Voice comparison
    voice = e5.get('voice_comparison', {})
    for vk in list(voice.keys())[:2]:
        vd = voice[vk]
        diff = vd.get('difference', {})
        print(f"  {vk}: class_diff={diff}")
    
    # Swap residual
    swap = e5.get('swap_residual', {})
    print(f"  Swap residual cosines:")
    for sk, sv in swap.items():
        cos = sv.get('layer_cosines', {})
        first_l = list(cos.keys())[0] if cos else '?'
        last_l = list(cos.keys())[-1] if cos else '?'
        print(f"    {sk}: {first_l}={cos.get(first_l,'?')}, {last_l}={cos.get(last_l,'?')}")
    
    # ===== Exp6: Cross-Language =====
    e6 = d.get('exp6_cross_language', {})
    print(f"\n--- Exp6: Cross-Language Invariance ---")
    summary6 = e6.get('_summary', {})
    print(f"  Best invariance layer: {summary6.get('best_invariance_layer','?')}")
    for sk in list(e6.keys())[:3]:
        if sk.startswith('_'): continue
        sd = e6[sk]
        cos = sd.get('layer_cosines', {})
        if cos:
            best_l = max(cos, key=cos.get)
            print(f"  {sk}: best_cos={cos[best_l]}@{best_l}, EN_fam={sd.get('en_fam_logits',{})}, ZH_fam={sd.get('zh_fam_logits',{})}")
    
    # ===== Exp7: Code Synthesis =====
    e7 = d.get('exp7_code_synthesis', {})
    print(f"\n--- Exp7: Artificial Code Synthesis ---")
    for key in sorted(e7.keys()):
        if key.startswith('_'): continue
        rd = e7[key]
        fam_changes = rd.get('fam_changes', {})
        desc = rd.get('desc', '?')
        # 只打印最终层
        if 'L35' in key or 'L39' in key or 'L27' in key:
            print(f"  {key}: {desc}")
            print(f"    fam_changes={fam_changes}")
