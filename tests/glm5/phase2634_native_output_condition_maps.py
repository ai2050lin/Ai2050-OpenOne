"""Basic all-coordinate path summaries and exact full-matrix gradient comparisons."""
import itertools,json
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2632_fulltoken_native_adjoints import LAYERS,MODULES,INPUT_KEY

SOURCE=RESULT/'phase2632_fulltoken_native_adjoints';OUT=RESULT/'phase2634_native_output_condition_maps'

def matrix_cosine(pack_a,pack_b,layer,module,norm_a,norm_b):
    key=f'L{layer}_{module}';input_key=f'L{layer}_{INPUT_KEY[module]}'
    # Trace identity evaluates the complete matrix inner product, no dimension truncation.
    ga=pack_a[key+'__g'].astype('float64');gb=pack_b[key+'__g'].astype('float64')
    xa=pack_a[input_key].astype('float64');xb=pack_b[input_key].astype('float64')
    denominator=norm_a*norm_b
    if denominator==0:return None
    return float(np.sum((ga@gb.T)*(xa@xb.T))/denominator)

def main():
    records=read(SOURCE/'analysis/records.json');frames=read(SOURCE/'material/frames.json');site_summary={}
    for l in LAYERS:
        for name in MODULES:
            key=f'L{l}/{name}';site_summary[key]={}
            for cls in ('all','eos','non_eos'):
                selected=[s for r in records if cls=='all' or r['eos']==(cls=='eos') for s in r['sites'] if s['layer']==l and s['module']==name]
                values=[s['last_token_only_relative_l2_error'] for s in selected if s['last_token_only_relative_l2_error'] is not None]
                nonboundary=[s['non_boundary_adjoint_energy_fraction'] for s in selected if s['non_boundary_adjoint_energy_fraction'] is not None]
                site_summary[key][cls]={'n':len(values),'mean_last_token_only_error':float(np.mean(values)) if values else None,
                    'max_last_token_only_error':max(values) if values else None,'mean_nonboundary_adjoint_energy_fraction':float(np.mean(nonboundary)) if nonboundary else None}
    groups=sorted({r['family']+'/'+r['language'] for r in records});low={}
    for group in groups:
        rr=[r for r in records if r['family']+'/'+r['language']==group and not r['eos']]
        low[group]={'n_frames':len(rr),'hidden_quartile_energy_by_layer':np.mean([r['hidden_amplitude_quartile_gradient_energy'] for r in rr],0).tolist() if rr else None,
            'mlp_quartile_energy_by_layer':np.mean([r['mlp_amplitude_quartile_gradient_energy'] for r in rr],0).tolist() if rr else None}
    candidates={key:[] for key in ('same_output_different_family','same_family_same_output','same_family_different_output')}
    initial=[r for r in records if r['step']==0 and not r['eos']]
    for a,b in itertools.combinations(initial,2):
        same_family=(a['family'],a['language'])==(b['family'],b['language']);same_output=(a['chosen_id'],a['runnerup_id'])==(b['chosen_id'],b['runnerup_id'])
        if a['index']==b['index']:continue
        cls='same_family_same_output' if same_family and same_output else 'same_family_different_output' if same_family else 'same_output_different_family' if same_output else None
        if cls:candidates[cls].append((a,b))
    pair_records=[]
    # Fixed deterministic cap on pairs, not coordinates. All candidate counts are reported.
    for cls,pairs in candidates.items():
        chosen=sorted(pairs,key=lambda ab:(ab[0]['frame_id']*997+ab[1]['frame_id']*193)%1000003)[:24]
        for pi,(a,b) in enumerate(chosen):
            with np.load(SOURCE/f'field/factors/frame_{a["frame_id"]:04d}.npz') as pa,np.load(SOURCE/f'field/factors/frame_{b["frame_id"]:04d}.npz') as pb:
                by_site=[]
                for sa,sb in zip(a['sites'],b['sites']):
                    assert (sa['layer'],sa['module'])==(sb['layer'],sb['module'])
                    cosine=matrix_cosine(pa,pb,sa['layer'],sa['module'],sa['full_parameter_gradient_l2'],sb['full_parameter_gradient_l2'])
                    by_site.append({'layer':sa['layer'],'module':sa['module'],'fullmatrix_cosine':cosine})
            pair_records.append({'class':cls,'frame_a':a['frame_id'],'frame_b':b['frame_id'],'family_a':a['family'],'family_b':b['family'],'output_a':[a['chosen_id'],a['runnerup_id']],'output_b':[b['chosen_id'],b['runnerup_id']],'sites':by_site})
        print('full matrix matched-output comparisons',cls,len(chosen),flush=True)
    comparison={}
    for cls in candidates:
        rr=[r for r in pair_records if r['class']==cls]
        comparison[cls]={'eligible_pairs':len(candidates[cls]),'selected_pairs':len(rr),'sites':{f'L{l}/{m}':float(np.mean([s['fullmatrix_cosine'] for r in rr for s in r['sites'] if s['layer']==l and s['module']==m and s['fullmatrix_cosine'] is not None])) if any(s['layer']==l and s['module']==m and s['fullmatrix_cosine'] is not None for r in rr for s in r['sites']) else None for l in LAYERS for m in MODULES}}
    save(OUT/'analysis/fullmatrix_pair_comparisons.json',pair_records);save(OUT/'analysis/low_magnitude_allcoordinate.json',low)
    ties=[{'frame_id':r['frame_id'],'case_id':r['case_id'],'bf16_head_margin':r['bf16_head_margin'],'fp32_margin':r['fp32_loss']} for r in records if not r['autograd_same_native_argmax']]
    summary={'all28_site_summaries':site_summary,'output_condition_comparisons':comparison,'native_argmax_disagreement_cases':ties,
        'tie_policy':'phase2631 chooses first of torch.topk(2); at equal BF16 maxima torch.argmax may pick another index. Both are greedy maxima; FP32 margin equality audit distinguishes this from changed fields.'}
    result={'provenance':str(Path(__file__)),'summary':summary,'checks':{'all28_operator_sites':len(site_summary)==28,'all16_groups_low_coordinate_audit':len(low)==16,
        'all_pair_cosines_finite_or_unavailable':all(s['fullmatrix_cosine'] is None or np.isfinite(s['fullmatrix_cosine']) for r in pair_records for s in r['sites']),
        'argmax_disagreements_are_bf16_ties':all(t['bf16_head_margin']==0 for t in ties)}}
    finish(2634,'全参数跨位置误差、输出身份控制及低幅值坐标图谱',OUT,result,
        '对四层七矩阵按内容/EOS分别汇总完整参数梯度与只看末token的误差；全部低幅值坐标参加。比较严格相同原生输出token对与不同输出token对，完整矩阵内积通过精确迹恒等式计算，不做压缩或坐标截断。',
        r'\langle G_a,G_b\rangle_F=\sum_{t,s}(\bar Y_{a,t}^T\bar Y_{b,s})(X_{a,t}^TX_{b,s}),\quad G_a=\bar Y_a^TX_a.',
        '全部415已采样决策及28矩阵，EOS分层；三类配对各最多24，按固定hash排序选取并公开全部合格对数；输出对比条件是观测性匹配，不是随机因果实验。',
        '把“矩阵物理梯度相似”拆成语言条件和输出身份两个来源，防止把固定读出方向误认通用语义主干。早层共享参数对源位置的依赖可量化为末位置近似误差，而不是差分搬运是否成功。',
        '相同输出token对样本覆盖可能有限，多步同prompt不是独立语义样本；语义、模板、语言仍有混杂。梯度对比不等于注意力自然必要性或闭合传递。个别topk/argmax差异为BF16并列最大值的tie-break，原轨迹仍为greedy最大值。',
        '围绕真实单参数前向检验支持的完整token求和继续扩大确认，并发布可按token逐项查看的参数作用查询；不把工程恒等式升级成语言定理。')

if __name__=='__main__':main()
