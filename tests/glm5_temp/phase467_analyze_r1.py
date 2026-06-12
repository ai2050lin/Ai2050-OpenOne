"""Phase 467 R1 Results Analyzer"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'results/glm5/phase467_{model}_r1.json'
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        
        print(f'\n===== {model} (n_layers={d["n_layers"]}, d_model={d["d_model"]}) =====')
        
        # Exp1: PC1 Attribution
        e1 = d.get('exp1_pc1_attribution', {})
        print('\n--- Exp1: PC1 Attribution ---')
        for layer_key in sorted(e1.keys()):
            lr = e1[layer_key]
            if isinstance(lr, dict) and 'pc_eigenvalues' in lr:
                pe = lr['pc_eigenvalues']
                pos_corr = lr.get('pc1_vs_position', {}).get('correlation', 0) if isinstance(lr.get('pc1_vs_position'), dict) else 0
                print(f'  {layer_key}: pc1_ratio={pe.get("pc1_ratio",0):.4f}, '
                      f'cat_spread={lr.get("pc1_category_spread",0):.4f}, '
                      f'norm_corr={lr.get("pc1_vs_norm_correlation",0):.4f}, '
                      f'pos_corr={pos_corr:.4f}, '
                      f'ent_corr={lr.get("pc1_vs_entropy_correlation",0):.4f}')
                wu = lr.get('pc1_vs_W_U_alignment', {})
                if isinstance(wu, dict) and 'cos_pc1_WU_pc1' in wu:
                    print(f'    W_U: cos_pc1={wu["cos_pc1_WU_pc1"]:.4f}, cos_pc2={wu.get("cos_pc1_WU_pc2",0):.4f}')
                npc = lr.get('no_pc1_effects', {})
                for cat in ['fruit','animal','vehicle','tool','furniture']:
                    if cat in npc:
                        print(f'    {cat}: cos_diff_pc1={npc[cat].get("cos_diff_pc1",0):.4f}, cos_raw_nopc1={npc[cat].get("cos_raw_nopc1",0):.4f}')
        
        # Exp2
        e2 = d.get('exp2_whitened_new_directions', {})
        print('\n--- Exp2: Whitened New Directions ---')
        for layer_key in sorted(e2.keys()):
            lr = e2[layer_key]
            if isinstance(lr, dict):
                for cat in ['animal', 'vehicle', 'fruit']:
                    if cat in lr:
                        cr = lr[cat]
                        dcos = cr.get('direction_cosines', {})
                        raw_s = cr.get('raw', {}).get('selectivity', '?')
                        white_s = cr.get('whitened_new', {}).get('selectivity', '?')
                        nopc1_s = cr.get('no_pc1', {}).get('selectivity', '?')
                        print(f'  {layer_key}/{cat}: cos_raw_white={dcos.get("raw_vs_whitened_new","?")}, '
                              f'raw_sel={raw_s}, white_sel={white_s}, nopc1_sel={nopc1_s}')
        
        # Exp4: Combined
        e4 = d.get('exp4_combined_directions', {})
        print('\n--- Exp4: Combined Directions ---')
        for layer_key in sorted(e4.keys()):
            lr = e4[layer_key]
            if isinstance(lr, dict):
                for cat in ['vehicle', 'furniture', 'animal']:
                    if cat in lr:
                        cr = lr[cat]
                        parts = []
                        for m in ['raw','no_pc1','disentangle','no_pc1+disentangle','no_top3pc+disentangle','random']:
                            if m in cr:
                                parts.append(f'{m}={cr[m]["selectivity"]:.4f}')
                        print(f'  {layer_key}/{cat}: ' + ', '.join(parts))
        
        # Exp5: Generation quality
        e5 = d.get('exp5_generation_systematic', {})
        print('\n--- Exp5: Generation Quality ---')
        for cat in ['fruit','animal','vehicle','furniture']:
            if cat in e5:
                cr = e5[cat]
                base = cr.get('base_gen', '')[:60]
                print(f'  {cat} base: {base}')
                for dir_name in ['raw', 'no_pc1']:
                    if dir_name in cr:
                        for ratio_key in sorted(cr[dir_name].keys()):
                            r = cr[dir_name][ratio_key]
                            q = r.get('quality', {})
                            gt = r.get('gen_text', '')[:55]
                            print(f'    {dir_name}/{ratio_key}: quality={q.get("overall","?")}, gen="{gt}"')
        
        # Exp3: Sensitivity map - key ratios for DS7B
        e3 = d.get('exp3_sensitivity_map', {})
        if e3:
            print('\n--- Exp3: Sensitivity Map ---')
            for layer_key in sorted(e3.keys()):
                lr = e3[layer_key]
                if isinstance(lr, dict):
                    for cat in ['fruit','animal','vehicle']:
                        if cat in lr:
                            cr = lr[cat]
                            parts = []
                            for rk in sorted(cr.keys()):
                                r = cr[rk]
                                s = r.get('selectivity', 0)
                                q = r.get('gen_quality', {}).get('overall', '?')
                                parts.append(f'{rk}:sel={s:.3f}/q={q}')
                            print(f'  {layer_key}/{cat}: ' + ', '.join(parts))
    
    except FileNotFoundError:
        print(f'  {model}: file not found')
    except Exception as ex:
        print(f'  {model}: error: {ex}')
        import traceback; traceback.print_exc()
