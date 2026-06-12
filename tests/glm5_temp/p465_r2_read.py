"""Phase 465 R2 结果提取"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')

# R2 Exp5 writability comparison
print('### Exp5 R2: 残差可写性 (8对象, train=4 test=4) ###')
for model in ['qwen3', 'deepseek7b', 'glm4']:
    path = f'results/glm5/phase465_{model}_r2.json'
    if not os.path.exists(path): continue
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    exp5 = d.get('exp5_writability', {})
    print(f'\n{model}:')
    for cat in ['fruit','animal','vehicle','clothing','furniture','tool']:
        if cat not in exp5: continue
        for lk in sorted(exp5[cat].keys()):
            dd = exp5[cat][lk]
            avg = dd.get('avg_selectivity', {})
            b5 = avg.get('beta_5.0', 0)
            b10 = avg.get('beta_10.0', 0)
            n = dd.get('n_test', 0)
            print(f'  {cat:<10} {lk:<5} b5={b5:.4f} b10={b10:.4f} n_test={n}')

# Exp1 manifold R2
print('\n\n### Exp1 R2: norm_ratio vs selectivity (beta=5,10) ###')
for model in ['qwen3', 'deepseek7b', 'glm4']:
    path = f'results/glm5/phase465_{model}_r2.json'
    if not os.path.exists(path): continue
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    exp1 = d.get('exp1_manifold', {})
    print(f'\n{model}:')
    for lk in sorted(exp1.keys()):
        ld = exp1[lk]
        for cat in ['animal', 'vehicle']:
            if cat not in ld: continue
            for bk in ['beta_5.0', 'beta_10.0']:
                if bk not in ld[cat]: continue
                dd = ld[cat][bk]
                nr = dd['norm_ratio']
                kl = dd['kl_div']
                sel = dd['selectivity']
                print(f'  {lk} {cat:<10} {bk}: norm_ratio={nr:.3f} KL={kl:.4f} sel={sel:.4f}')

# Exp2 axis verification R2
print('\n\n### Exp2 R2: 一维轴验证 ###')
for model in ['qwen3', 'deepseek7b', 'glm4']:
    path = f'results/glm5/phase465_{model}_r2.json'
    if not os.path.exists(path): continue
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    exp2 = d.get('exp2_axis_verify', {})
    print(f'\n{model}:')
    for lk in sorted(exp2.keys()):
        dd = exp2[lk]
        raw = dd.get('eff_rank_raw', 0)
        top1 = dd.get('top1_ratio', 0)
        wh = dd.get('eff_rank_whitened', -1)
        rm1 = dd.get('remove_top_k', {}).get('remove_top_1', {}).get('eff_rank', 0)
        rm3 = dd.get('remove_top_k', {}).get('remove_top_3', {}).get('eff_rank', 0)
        print(f'  {lk}: eff_rank={raw:.3f} top1={top1:.4f} whitened={wh:.3f} rm_top1={rm1:.3f} rm_top3={rm3:.3f}')

# Exp3 vehicle R2
print('\n\n### Exp3 R2: vehicle反向分析 ###')
for model in ['qwen3', 'deepseek7b', 'glm4']:
    path = f'results/glm5/phase465_{model}_r2.json'
    if not os.path.exists(path): continue
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    exp3 = d.get('exp3_vehicle', {})
    print(f'\n{model}:')
    for lk in sorted(exp3.keys()):
        dd = exp3[lk]
        cl = dd.get('cross_lang_vehicle_cos', 'N/A')
        vt = dd.get('cat_diff_cos', {}).get('vehicle_vs_tool', 'N/A')
        vf = dd.get('cat_diff_cos', {}).get('vehicle_vs_furniture', 'N/A')
        veh_rc = dd.get('vehicle_readout_cos', {})
        avg_rc = sum(veh_rc.values())/len(veh_rc) if veh_rc else 0
        cl_str = cl if isinstance(cl, str) else f"{cl:.4f}"
        vt_str = vt if isinstance(vt, str) else f"{vt:.4f}"
        vf_str = vf if isinstance(vf, str) else f"{vf:.4f}"
        print(f"  {lk}: cross_lang={cl_str}, v_vs_tool={vt_str}, v_vs_furniture={vf_str}, W_U_veh_cos_avg={avg_rc:.4f}")

print('\nDone!')
