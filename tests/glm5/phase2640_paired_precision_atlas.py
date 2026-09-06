"""Whole-coordinate precision differences and exact all-parameter comparison, then delivery."""
import argparse
import itertools
import math
import shutil
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2628_native_atlas_delivery import ASSET,panel,row
from phase2632_fulltoken_native_adjoints import LAYERS,MODULES,INPUT_KEY
from phase2636_precision_engine import summarize

OUT=RESULT/'phase2640_paired_precision_atlas'
INITIAL={'bf16':RESULT/'phase2637_bf16_native_numeric_control','fp32':RESULT/'phase2638_fp32_native_numeric_control'}
EXPANDED={p:RESULT/'phase2639_expanded_paired_precision_control'/p for p in INITIAL}

def operator_inner(a,b,l,name):
    key=f'L{l}_{name}';inp=f'L{l}_{INPUT_KEY[name]}'
    ga=a[key+'__g'].astype('float64');gb=b[key+'__g'].astype('float64')
    xa=a[inp].astype('float64');xb=b[inp].astype('float64')
    return float(np.sum((ga@gb.T)*(xa@xb.T)))

def pair_controls(sources):
    data={p:read(s/'analysis/conditions.json') for p,s in sources.items()}
    key=lambda r:(r['frame_id'],r.get('layer'),r.get('module'),r.get('selector'),r.get('scale'),r.get('sign'),r['kind'])
    a={key(r):r for r in data['bf16']};b={key(r):r for r in data['fp32']}
    assert a.keys()==b.keys()
    checks={'same_condition_keys':a.keys()==b.keys(),
            'same_numerical_parameter_changes':all(a[k]['actual_delta']==b[k]['actual_delta'] and a[k]['target_weight']==b[k]['target_weight'] for k in a if a[k]['kind']=='shared_weight')}
    summary={p:summarize(rr) for p,rr in data.items()}
    ratios={}
    for group in summary['bf16']:
        numerator=summary['bf16'][group]['mean_abs_effect'];denominator=summary['fp32'][group]['mean_abs_effect']
        ratios[group]=numerator/denominator if denominator else None
    return {'summary':summary,'bf16_over_fp32_mean_abs_effect':ratios,'checks':checks}

def analyze():
    initial=pair_controls(INITIAL);expanded=pair_controls(EXPANDED)
    map_acc={};counts={};rows=[];cosines=[];rank_changes=[];max_same_weight_check=True
    for label,sources in [('initial',INITIAL),('expanded',EXPANDED)]:
        frames=read(sources['bf16']/'material/frames.json');records={p:{r['frame_id']:r for r in read(s/'analysis/records.json')} for p,s in sources.items()}
        for i,f in enumerate(frames):
            fid=f['frame_id'];group=f['family']+'/'+f['language']
            with np.load(sources['bf16']/f'field/frame_{fid:04d}.npz') as a,np.load(sources['fp32']/f'field/frame_{fid:04d}.npz') as b:
                metrics={}
                for name in ('hidden_boundary','hidden_adjoint_boundary','mlp_boundary','mlp_adjoint_boundary'):
                    aa=a[name].astype('float64');bb=b[name].astype('float64');difference=bb-aa
                    metrics[name]={'relative_l2_by_layer':(np.linalg.norm(difference,axis=-1)/(np.linalg.norm(bb,axis=-1)+1e-30)).tolist(),
                                   'cosine_by_layer':(np.sum(aa*bb,axis=-1)/(np.linalg.norm(aa,axis=-1)*np.linalg.norm(bb,axis=-1)+1e-30)).tolist()}
                    mapkey=group+'/'+name
                    map_acc[mapkey]=map_acc.get(mapkey,0)+difference*difference;counts[mapkey]=counts.get(mapkey,0)+1
                    if name=='hidden_boundary':assert np.array_equal(aa[0],bb[0])
                if label=='initial':
                    for l in LAYERS:
                        for name in MODULES:
                            aa=operator_inner(a,a,l,name);bb=operator_inner(b,b,l,name);ab=operator_inner(a,b,l,name)
                            cosines.append({'frame_id':fid,'family':f['family'],'language':f['language'],'layer':l,'module':name,
                                'cosine':ab/math.sqrt(aa*bb) if aa>0 and bb>0 else None,
                                'relative_l2_difference':math.sqrt(max(aa+bb-2*ab,0)/bb) if bb>0 else None})
                rows.append({'set':label,'frame_id':fid,'case_id':f['case_id'],'metrics':metrics})
            ar=records['bf16'][fid];br=records['fp32'][fid]
            rank_changes.append({'set':label,'frame_id':fid,'case_id':f['case_id'],'bf16_head32_top2':ar['common_fp32_head_top2'],
                'fp32_head32_top2':br['common_fp32_head_top2'],'argmax_changed':ar['common_fp32_head_top2'][0]!=br['common_fp32_head_top2'][0],
                'bf16_margin':ar['margin'],'fp32_margin':br['margin']})
            if (i+1)%8==0:print('paired all-coordinate analysis',label,i+1,'/',len(frames),flush=True)
    maps={k:np.sqrt(v/counts[k]).astype('float32') for k,v in map_acc.items()}
    (OUT/'field').mkdir(parents=True,exist_ok=True);np.savez(OUT/'field/allcoordinate_precision_rms.npz',**maps)
    save(OUT/'analysis/per_frame_coordinate_differences.json',rows);save(OUT/'analysis/full_parameter_gradient_cosines.json',cosines)
    save(OUT/'analysis/paired_rank_and_margin.json',rank_changes)
    site_cos={f'L{l}/{n}':{'n':len(rr),'mean_cosine':float(np.mean([r['cosine'] for r in rr if r['cosine'] is not None]))} for l in LAYERS for n in MODULES for rr in [[r for r in cosines if r['layer']==l and r['module']==n]]}
    summary={'initial_control':initial,'expanded_control':expanded,'mapped_frames':len(rows),'groups':len(map_acc)//4,
        'initial_full_parameter_cosines':site_cos,'common_fp32_head_argmax_changes':sum(r['argmax_changed'] for r in rank_changes),
        'all_frame_baseline_count':len(rank_changes),'precision_boundary':'same numerical weights and perturbations, but baseline hidden states change with arithmetic precision; no claim of identical behavior or recovery of original training precision'}
    checks={**{'initial_'+k:v for k,v in initial['checks'].items()},**{'expanded_'+k:v for k,v in expanded['checks'].items()},
        'all48_coordinate_maps':len(rows)==48,'all448_initial_full_parameter_cosines':len(cosines)==448,
        'finite_maps':all(np.isfinite(v).all() for v in maps.values()),'four_prior_phases_ready':all(read(next(RESULT.glob(f'phase{p}_*/analysis/final.json')))['all_checks_passed'] for p in (2636,2637,2638,2639))}
    save(OUT/'analysis/paired_analysis.json',{'summary':summary,'checks':checks,'all_checks_passed':all(checks.values())})
    print(json.dumps({'checks':checks,'native_diagnostic_argmax_changes':summary['common_fp32_head_argmax_changes']}),flush=True)

def publish():
    analysis=read(OUT/'analysis/paired_analysis.json');assert analysis['all_checks_passed']
    spec=[('hidden_boundary',2560,(0,1,6,18,36),'BF16 vs FP32: all-coordinate HiddenState discrepancy'),
          ('hidden_adjoint_boundary',2560,(0,1,6,18,36),'BF16 vs FP32: all-coordinate HiddenState adjoint discrepancy'),
          ('mlp_boundary',9728,(0,5,17,35),'BF16 vs FP32: every MLP neuron activation discrepancy'),
          ('mlp_adjoint_boundary',9728,(0,5,17,35),'BF16 vs FP32: every MLP neuron adjoint discrepancy')]
    new=[]
    with np.load(OUT/'field/allcoordinate_precision_rms.npz') as maps:
        for name,d,layers,title in spec:
            rows=[]
            for key in sorted(maps.files):
                if not key.endswith('/'+name):continue
                for l in layers:rows.append(row(f'{key}/L{l}/RMS(FP32-BF16)',maps[key][l],2640,'allcoordinate_precision_discrepancy',l))
            new.append(panel('phase2640_'+name,title,d,rows,'All native physical coordinates; groupwise RMS over 3 prefixes (one initial + two expanded variants), descriptive not semantic feature importance.'))
    payload=read(ASSET);old=sha(ASSET)
    payload['models']=[p for p in payload['models'] if not p['key'].startswith('phase2640_')]+new;payload['phase']=2640
    payload['claim_boundary']='Full native coordinates, actual weights and precision-labelled adjoints. Same-weight FP32 numerical controls must not be conflated with BF16 natural behavior or semantic mechanism closure.'
    payload['summary']['model_rows']={p['key']:len(p['rows']) for p in payload['models']};payload['summary']['total_rows']=sum(payload['summary']['model_rows'].values())
    ASSET.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    frames=read(INITIAL['bf16']/'material/frames.json');save(OUT/'material/published_frames.json',frames)
    cases={r['case_id']:r for r in read(RESULT/'phase2631_native_generation_paths/material/cases.json')}
    assert all(f['prefix_ids']==cases[f['case_id']]['prompt_ids'] for f in frames)
    save(OUT/'material/token_strings.json',{f['frame_id']:cases[f['case_id']]['token_strings'] for f in frames})
    publication={'previous_asset_sha256':old,'asset_sha256':sha(ASSET),'asset_bytes':ASSET.stat().st_size,
        'panels':[{'key':p['key'],'width':p['coordinate_count'],'rows':len(p['rows'])} for p in new],
        'published_frame_ids':[f['frame_id'] for f in frames],'retained':'all16 initial prefixes in both precisions, all28 operators and their tokenwise factors; expanded32 both-precision raw packs cleaned only after analysis/QA'}
    save(OUT/'analysis/publication.json',publication);print(json.dumps(publication),flush=True)

def cleanup():
    assert read(OUT/'analysis/paired_analysis.json')['all_checks_passed'] and read(OUT/'analysis/delivery_checks.json')['all_checks_passed']
    paths=[];kept=[]
    for sources,preserve in ((INITIAL,True),(EXPANDED,False)):
        for precision,source in sources.items():
            for r in read(source/'analysis/raw_manifest.json'):
                p=Path(r['path']).resolve()
                if p.parent!=(source/'field').resolve() or not p.is_relative_to(RESULT.resolve()) or not re.fullmatch(r'frame_\d{4}\.npz',p.name):raise RuntimeError('unexpected cleanup path')
                assert p.is_file() and p.stat().st_size==r['bytes']
                entry={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
                (kept if preserve else paths).append(entry)
    assert len(kept)==32 and len(paths)==64
    audit={'targets':paths,'kept':kept,'deleted_files':len(paths),'deleted_bytes':sum(r['bytes'] for r in paths),
           'recoverability':'not in Recycle Bin; regenerated from local model and saved frames/scripts','before_free_bytes':shutil.disk_usage(OUT).free}
    save(OUT/'analysis/cleanup_plan.json',audit)
    for r in paths:Path(r['path']).unlink()
    audit.update(after_free_bytes=shutil.disk_usage(OUT).free,all_deleted=all(not Path(r['path']).exists() for r in paths),all_published_retained=all(Path(r['path']).is_file() for r in kept))
    save(OUT/'analysis/cleanup_completed.json',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ('targets','kept')}),flush=True)

def numerical_audit():
    result={}
    for label,sources in [('initial',INITIAL),('expanded',EXPANDED)]:
        data={p:[r for r in read(s/'analysis/conditions.json') if r['kind']=='shared_weight'] for p,s in sources.items()}
        key=lambda r:(r['frame_id'],r['layer'],r['module'],r['selector'],r['scale'],r['sign'])
        a={key(r):r for r in data['bf16']};b={key(r):r for r in data['fp32']};by_site={};by_family={}
        for l in (0,5,17,35):
            for scale in (.02,.2,1.):
                keys=[k for k in a if k[1:3]==(l,'v_proj') and k[4]==scale]
                den=sum(abs(b[k]['effect']) for k in keys)
                by_site[f'L{l}/v_proj/scale{scale}']={'n':len(keys),
                    'bf16_gradient_predicting_fp32_effect_l1_error':sum(abs(b[k]['effect']-a[k]['predicted_full']) for k in keys)/den if den else None,
                    'fp32_gradient_predicting_fp32_effect_l1_error':sum(abs(b[k]['effect']-b[k]['predicted_full']) for k in keys)/den if den else None}
        for group in sorted({r['family']+'/'+r['language'] for r in data['bf16']}):
            by_family[group]={}
            for precision,rows in data.items():
                rr=[r for r in rows if r['family']+'/'+r['language']==group and r['layer']==0 and r['module']=='v_proj' and r['scale']==.2]
                den=sum(abs(r['effect']) for r in rr)
                by_family[group][precision]={'n':len(rr),'full_l1_error':sum(abs(r['effect']-r['predicted_full']) for r in rr)/den if den else None}
        centered={}
        for precision,rr in data.items():
            index={key(r):r for r in rr};by_group={}
            for k,r in index.items():
                if r['sign']!=1:continue
                other=index[k[:-1]+(-1,)]
                observed=r['effect']-other['effect'];predicted=r['gradient_full']*(r['actual_delta']-other['actual_delta'])
                group=f'L{r["layer"]}/{r["module"]}/scale{r["scale"]}'
                by_group.setdefault(group,[]).append((observed,predicted))
            centered[precision]={g:{'paired_parameters':len(v),'central_contrast_l1_error':sum(abs(x-y) for x,y in v)/sum(abs(x) for x,y in v) if sum(abs(x) for x,y in v) else None} for g,v in by_group.items()}
        result[label]={'cross_precision_gradient_predictions':by_site,'early_v_scale0.2_by_family':by_family,'central_two_sided_control':centered}
    result['boundary']='Central two-sided scalar parameter changes cancel the common baseline but do not remove BF16 internal rounding. Error ratios are aggregate L1, not accuracy rates or causal information fractions.'
    frames=[f for f in read(RESULT/'phase2632_fulltoken_native_adjoints/material/frames.json') if f['step']==0 and not f['eos']]
    matches=[{'frame_a':a['frame_id'],'frame_b':b['frame_id'],'case_a':a['case_id'],'case_b':b['case_id'],
              'families':[a['family'],b['family']],'same_material_index':a['index']==b['index'],
              'native_output_pair':[a['chosen_id'],a['runnerup_id']]}
             for a,b in itertools.combinations(frames,2) if a['family']!=b['family'] and (a['chosen_id'],a['runnerup_id'])==(b['chosen_id'],b['runnerup_id'])]
    result['next_output_matching_audit']={'crossfamily_same_native_output_pairs':len(matches),
        'same_material_index_pairs':sum(r['same_material_index'] for r in matches),'different_material_index_pairs':sum(not r['same_material_index'] for r in matches),
        'pairs':matches,'boundary':'Phase2634 deliberately excluded identical material indices. Its4 pairs concern a different-index subset, not all available matched-output pairs. Same-index matched-operation comparison and independent-index generalization answer different questions; a material index alone is not proof of identical semantic input.'}
    save(OUT/'analysis/numerical_audit.json',result)
    print(json.dumps({'expanded_L0_V_cross_gradient':result['expanded']['cross_precision_gradient_predictions']['L0/v_proj/scale0.2'],
        'expanded_L0_V_centered':{p:v['L0/v_proj/scale0.2'] for p,v in result['expanded']['central_two_sided_control'].items()}}),flush=True)
    return result

def finalize():
    analysis=read(OUT/'analysis/paired_analysis.json');publication=read(OUT/'analysis/publication.json');clean=read(OUT/'analysis/cleanup_completed.json')
    audit=numerical_audit()
    summary={**analysis['summary'],'client':publication,'cleanup':{k:clean[k] for k in ('deleted_files','deleted_bytes','recoverability')},
        'additional_numeric_audit':'analysis/numerical_audit.json: by-family early V, central signed parameter contrasts and BF16-gradient prediction of FP32 finite effects; these do not depend on retained raw fields',
        'next_output_matching_material':{k:v for k,v in audit['next_output_matching_audit'].items() if k!='pairs'},
        'next_same_goal':'expand strict output-matched language families, keep native and externally specified common readouts distinct; use validated FP32 native-coordinate algorithms for small-effect numerical interpretation, retain BF16 actual behavior as separate object'}
    checks={**analysis['checks'],'api_build_checks':read(OUT/'analysis/delivery_checks.json')['all_checks_passed'],
        'post_cleanup_checks':read(OUT/'analysis/post_cleanup_checks.json')['all_checks_passed'],'all_published_retained':clean['all_published_retained'],'all_manifested_cleanup':clean['all_deleted']}
    assert all(checks.values())
    finish(2640,'双精度全坐标图谱、早层单参数测量纠错与完整交付',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '严格逐条配对BF16与FP32实际参数数值、同前缀、同坐标及同剂量，比较有限误差；全坐标汇总48前缀的状态/MLP及其伴随差异。初测16前缀全部28矩阵的全参数内积以精确因子恒等式计算，无坐标裁剪或低秩近似。',
        r'\langle G^{16},G^{32}\rangle_F=\sum_{t,s}(\bar Y^{16}_t\cdot\bar Y^{32}_s)(X^{16}_t\cdot X^{32}_s);\quad R_{l,j}=\sqrt{\operatorname{mean}_x(H^{32}_{x,l,j}-H^{16}_{x,l,j})^2}.',
        '16初测+32扩大，共48前缀，两精度各73条件/例，共7008条件（6912真实单权重扰动+96 no-op），另有96带伴随基线。全37个隐藏检查点与36层MLP全部坐标；28矩阵全token因子。客户端4张全坐标精度差异图和16前缀双精度逐参数查询；扩大原场完成分析后按清单清理。',
        '扩大32前缀的L0/V在0.2倍矩阵RMS下，完整导数汇总L1误差由BF16的0.98805降至FP32的0.004258；初测对应1.00996与0.003171。28矩阵初测完整梯度平均余弦均超过0.990，意味着局部伴随方向整体相近，但BF16有限效应不能同等预测。前轮早层阴性不足以否定原生单参数路径，低精度传播必须先排查。全token参数求和是基础链式结构；数值验证不等于已提取通用语言编码规则。共同FP32头的48基线比较有1例首位改变，是precision改变基线的明示边界，不冒称全部自然生成相同。',
        'FP32改变了整个前向基线，不是保持自然BF16状态的单点精度修复；不能断言所有残差全由某个舍入点造成，也不能拿FP32结果覆盖BF16真实行为。小效应受FP32测量下限影响，较大剂量仍有曲率。只有一个模型、受控八族、复用词表，且实际干预六矩阵；不是完整自然生成闭环。',
        '完整合同2636—2640已完成。下一阶段仍同目标，回到语言模式族的普遍性：本材料跨族同原生输出对共有31，其中27共享材料index、4不同index；2634只报告不同index子集，不代表全部匹配覆盖。固定材料比较操作与独立材料验证泛化必须分层，避免把匹配控制一并过滤。扩大词表和句式，分开原生与共同任务读出；用已校准的单参数算法沿真实计算追踪复用/差异，而不是因BF16导数门失败再换一个搬运方向。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analyze','publish','cleanup','finalize','numerical_audit']);args=p.parse_args();globals()[args.action]()
