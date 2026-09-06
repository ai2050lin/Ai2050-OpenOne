"""Natural BF16 atlas audit, matched-cohort interpretation, client and bounded cleanup."""
import argparse,itertools,shutil
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2628_native_atlas_delivery import ASSET,panel,row
from phase2643_matched_dual_adjoint_engine import MATERIAL,BF,INITIAL,CONFIRM,LAYERS
from phase2644_matched_coordinate_maps import cosine

OUT=RESULT/'phase2647_matched_operation_delivery'
MAPS={'initial':RESULT/'phase2644_matched_coordinate_maps','confirmation':RESULT/'phase2646_matched_single_parameter'}

def audit():
    cases=[r for r in read(MATERIAL/'material/cases.json') if r['field_set']!='behavior_only'];records={r['case_index']:r for source in (INITIAL,CONFIRM) for r in read(source/'analysis/records.json')}
    table={(r['family'],r['language'],r['unit'],r['form'],r['target_index'],r['mention_order']):r for r in cases}
    groups=sorted({(r['family'],r['language']) for r in cases});units=sorted({r['unit'] for r in cases});identities=[];summaries={};total_tokens=0
    OUT.joinpath('field').mkdir(parents=True,exist_ok=True)
    for fam,lang in groups:
        accum={};counts={};tokens=0;bfhs=None;bfhq=None;fphq=None;diffq=None
        for unit in units:
            cube={}
            for form,v,o in itertools.product((0,1),repeat=3):
                r=table[(fam,lang,unit,form,v,o)];ci=r['case_index'];rec=records[ci];source=INITIAL if r['field_set']=='initial' else CONFIRM
                pos=rec['positions']
                with np.load(BF/f'field/case_{ci:04d}.npz',allow_pickle=False) as b,np.load(source/f'field/case_{ci:04d}.npz',allow_pickle=False) as f:
                    bh=b['hidden'].astype('float64');fh=f['hidden'].astype('float64');tokens+=bh.shape[1];total_tokens+=bh.shape[1]
                    assert np.array_equal(bh[0],fh[0])
                    for name,term in [('bfhs',bh.sum(1)),('bfhq',(bh*bh).sum(1)),('fphq',(fh*fh).sum(1)),('diffq',((bh-fh)**2).sum(1))]:
                        if name=='bfhs':bfhs=term if bfhs is None else bfhs+term
                        elif name=='bfhq':bfhq=term if bfhq is None else bfhq+term
                        elif name=='fphq':fphq=term if fphq is None else fphq+term
                        else:diffq=term if diffq is None else diffq+term
                    cube[(form,v,o)]={'h':bh[:,pos],'mlp':b['mlp_positions'].astype('float64')}
                    identity=rec['native_common_identity']
                    if identity in ('same','opposite'):
                        sign=1 if identity=='same' else -1;err=0.
                        for suffix in ('hidden_adjoint_positions','mlp_adjoint_positions')+tuple(f'L{l}_v_g' for l in LAYERS):
                            err=max(err,float(np.max(np.abs(f['native__'+suffix]-sign*f['common__'+suffix]))))
                        identities.append({'case_index':ci,'identity':identity,'maximum_absolute_error':err})
                        assert err==0
                del bh,fh
            for kind,axis in [('target',1),('order',2),('form',0)]:
                for a in itertools.product((0,1),repeat=3):
                    if a[axis]!=0:continue
                    b=list(a);b[axis]=1;b=tuple(b)
                    for metric in ('h','mlp'):
                        d=cube[a][metric]-cube[b][metric];key=kind+'__'+metric
                        if key not in accum:accum[key]=[np.zeros_like(d),np.zeros_like(d)] ;counts[key]=0
                        accum[key][0]+=d;accum[key][1]+=d*d;counts[key]+=1
            del cube
        maps={'bf_h_alltoken_mean':(bfhs/tokens).astype('float32'),'bf_h_alltoken_rms':np.sqrt(bfhq/tokens).astype('float32'),
              'fp_h_alltoken_rms':np.sqrt(fphq/tokens).astype('float32'),'bf_fp_alltoken_difference_rms':np.sqrt(diffq/tokens).astype('float32')}
        ss={}
        with np.load(MAPS['initial']/f'field/{fam}_{lang}_fullcoordinate_maps.npz') as a,np.load(MAPS['confirmation']/f'field/{fam}_{lang}_fullcoordinate_maps.npz') as b:
            for key,(s,q) in accum.items():
                n=counts[key];maps[key+'__mean']=(s/n).astype('float32');maps[key+'__rms']=np.sqrt(q/n).astype('float32')
                fp_rms=np.sqrt((a[key+'__rms'].astype('float64')**2+b[key+'__rms'].astype('float64')**2)/2)
                cc,valid=cosine(maps[key+'__rms'],fp_rms)
                ss[key]={'n':n,'rms_by_layer_position':np.sqrt(np.mean(q/n,axis=-1)).tolist(),'bf_fp_response_rms_cosine':cc.tolist(),'valid':valid.tolist()}
        np.savez(OUT/f'field/{fam}_{lang}_natural_maps.npz',**maps)
        summaries[fam+'/'+lang]={'all_tokens':tokens,'response':ss,'alltoken_hidden_precision_relative_l2_by_layer':np.sqrt(diffq.sum(-1)/np.maximum(bfhq.sum(-1),1e-30)).tolist()}
        print('BF16 natural maps audit',fam,lang,flush=True)
    cohorts={}
    for label,path in MAPS.items():
        pairs=read(path/'analysis/crossfamily_pairs.json');matched=[r for r in pairs if r['native_ids_match']]
        for group,pp in [('all_pairs',pairs),('native_id_matched',matched),('native_id_matched_readout_not_identical',
                [r for r in matched if records[r['case_a']]['native_common_identity']=='different' or records[r['case_b']]['native_common_identity']=='different'])]:
            cohorts[label+'/'+group]={}
            for l,obj in itertools.product(LAYERS,('native','common')):
                vals=[r['V'][f'L{l}/{obj}'] for r in pp if r['V'].get(f'L{l}/{obj}') is not None]
                cohorts[label+'/'+group][f'L{l}/{obj}']={'n':len(vals),'mean':float(np.mean(vals)) if vals else None}
    save(OUT/'analysis/natural_fullcoordinate_audit.json',summaries);save(OUT/'analysis/dual_readout_identity_checks.json',identities);save(OUT/'analysis/matched_cohort_audit.json',cohorts)
    checks={'all1024_raw_cases_audited':len(cases)==1024,'all16_natural_map_groups':len(summaries)==16,'same_or_opposite_readout_gradients_exact':bool(identities) and all(r['maximum_absolute_error']==0 for r in identities)}
    result={'summary':{'raw_cases':len(cases),'all_token_occurrences':total_tokens,'identical_or_opposite_readout_cases':len(identities),'cohorts':cohorts},'checks':checks,'all_checks_passed':all(checks.values())}
    save(OUT/'analysis/science_audit.json',result)

def publish():
    assert read(OUT/'analysis/science_audit.json')['all_checks_passed']
    allcases=read(MATERIAL/'material/cases.json');records={r['case_index']:r for source in (INITIAL,CONFIRM) for r in read(source/'analysis/records.json')}
    behavior={r['case_index']:r for r in (json.loads(s) for s in (BF/'behavior/greedy.jsonl').read_text(encoding='utf-8').splitlines())}
    published=[{**r,**records[r['case_index']],'generated':behavior[r['case_index']]['generated'],'name_content_correct':behavior[r['case_index']]['name_content_correct']}
               for r in allcases if (r['unit'],r['form'],r['target_index'],r['mention_order']) in ((0,0,0,0),(12,1,1,1))]
    assert len(published)==32;save(OUT/'material/published_cases.json',published)
    rr={k:[] for k in ('bf_hidden','bf_mlp','response_h','response_mlp','native_common_hg','V_parameter_row')}
    for r in published:
        ci=r['case_index'];source=INITIAL if r['field_set']=='initial' else CONFIRM
        with np.load(BF/f'field/case_{ci:04d}.npz') as b,np.load(source/f'field/case_{ci:04d}.npz') as f:
            h=b['hidden'];mlp=b['mlp_positions']
            for l in (0,1,18,36):rr['bf_hidden'].append(row(f'{r["case_id"]}/BF16/boundary/checkpoint{l}',h[l,-1],2647,'unmodified_native_hidden',l))
            rr['bf_mlp'].append(row(f'{r["case_id"]}/BF16/MLP35',mlp[35,2],2647,'actual_mlp_neuron',35))
            for l,obj in itertools.product((0,35),('native','common')):
                rr['native_common_hg'].append(row(f'{r["case_id"]}/{obj}/FP32/H{l+1}',f[f'{obj}__hidden_adjoint_positions'][l+1,2],2647,'output_labelled_native_adjoint',l))
                j=57 if l==0 else 137;x=f[f'L{l}_v_x'].astype('float64');g=f[f'{obj}__L{l}_v_g'][:,j].astype('float64')
                rr['V_parameter_row'].append(row(f'{r["case_id"]}/{obj}/V{l}/j{j}/all_k',(g[:,None]*x).sum(0),2647,'exact_alltoken_shared_weight_row',l))
    for p in sorted((OUT/'field').glob('*_natural_maps.npz')):
        with np.load(p) as m:
            for kind in ('target','order','form'):
                rr['response_h'].append(row(f'{p.stem}/{kind}/BF16/H36/RMS',m[kind+'__h__rms'][36,2],2647,'natural_crossed_condition_response',36))
                rr['response_mlp'].append(row(f'{p.stem}/{kind}/BF16/MLP35/RMS',m[kind+'__mlp__rms'][35,2],2647,'natural_crossed_condition_response',35))
    semantics='All native physical coordinate columns. Display examples and selected checkpoints are not a Top-K algorithm; all-layer/all-coordinate primary maps retained. Derivatives depend on output IDs and precision, not semantic necessity.'
    names={'bf_hidden':('Natural BF16 embeddings and raw HiddenState',2560),'bf_mlp':('Natural BF16 all final MLP neurons',9728),
           'response_h':('Target vs mention-order vs form: every hidden coordinate',2560),'response_mlp':('Target vs mention-order vs form: every MLP unit',9728),
           'native_common_hg':('Native-ID vs external A/B: full hidden adjoints',2560),'V_parameter_row':('One actual V weight row: all input coordinates, all tokens',2560)}
    new=[panel('phase2647_'+k,names[k][0],names[k][1],values,semantics) for k,values in rr.items()]
    with np.load(OUT/'field/allcoordinate_response_envelopes.npz') as a,np.load(OUT/'third_field_confirmation/field/coordinate_group_coverage.npz') as b:
        for metric,d,layers in [('h',2560,(1,6,18,36)),('mlp',9728,(0,5,17,35))]:
            values=[]
            for l in layers:
                for suffix in ('dominant','dominant_same_sign'):
                    values.append(row(f'{metric}/L{l}/initial+confirmation/{suffix}/groupcount0..16',a[f'{metric}__both_{suffix}'][l,2],2647,'fullcoordinate_response_envelope',l))
                    values.append(row(f'{metric}/L{l}/third1024/{suffix}/groupcount0..16',b[f'{metric}__{suffix}'][l,2],2647,'prospective_fullcoordinate_envelope',l))
            new.append(panel('phase2647_envelope_'+metric,'All-coordinate response envelope and prospective third-set check: '+metric,d,values,
                'Counts of16 family/language groups, noTopK. Target amplitude exceeds both order/form controls; signed means initial/third sign agreement WITHIN each group, NOT one universal cross-family semantic direction.'))
    payload=read(ASSET);old=sha(ASSET);prior=[p['key'] for p in payload['models'] if not p['key'].startswith('phase2647_')]
    payload['models']=[p for p in payload['models'] if not p['key'].startswith('phase2647_')]+new;payload['phase']=2647
    payload['claim_boundary']='Matched operations, native h / actual MLP a / scalar theta, and output-labelled derivatives. Common output geometry and local numeric prediction are not universal semantic mechanisms.'
    payload['summary']['model_rows']={p['key']:len(p['rows']) for p in payload['models']};payload['summary']['total_rows']=sum(payload['summary']['model_rows'].values())
    ASSET.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    proof={'previous_asset_sha256':old,'asset_sha256':sha(ASSET),'asset_bytes':ASSET.stat().st_size,'prior_panel_keys':prior,
           'panels':[{'key':p['key'],'width':p['coordinate_count'],'rows':len(p['rows'])} for p in new],'published_case_indices':[r['case_index'] for r in published]}
    save(OUT/'analysis/publication.json',proof);print(json.dumps({'new_panels':len(new),'published_cases':len(published),'asset_bytes':proof['asset_bytes']}),flush=True)

def cleanup():
    assert read(OUT/'analysis/science_audit.json')['all_checks_passed'] and read(OUT/'analysis/delivery_checks.json')['all_checks_passed']
    assert read(RESULT/'phase2646_matched_single_parameter/analysis/final.json')['all_checks_passed']
    assert read(OUT/'third_field_confirmation/analysis/completion.json')['all_checks_passed']
    published=set(read(OUT/'analysis/publication.json')['published_case_indices']);targets=[];kept=[]
    for source in (BF,INITIAL,CONFIRM):
        for r in read(source/'analysis/raw_manifest.json'):
            p=Path(r['path']).resolve()
            assert p.parent==(source/'field').resolve() and p.is_relative_to(RESULT.resolve()) and re.fullmatch(r'case_\d{4}\.npz',p.name)
            assert p.is_file() and p.stat().st_size==r['bytes']
            entry={'path':str(p),'bytes':p.stat().st_size,'case_index':r['case_index'],'sha256':sha(p)}
            (kept if r['case_index'] in published else targets).append(entry)
    assert len(kept)==64 and len(targets)==1984 and len({r['path'] for r in targets+kept})==2048
    audit={'targets':targets,'kept':kept,'deleted_files':len(targets),'deleted_bytes':sum(r['bytes'] for r in targets),'before_free_bytes':shutil.disk_usage(OUT).free,
        'recoverability':'not in Recycle Bin; regenerate from2641 revised frozen material,2642 BF16 behavior first decisions and2643/2645 FP32 engine. Complete behavior, coordinate-derived maps, model weights, scripts and32x2 published packs retained.'}
    save(OUT/'analysis/cleanup_plan.json',audit)
    for r in targets:
        p=Path(r['path']);assert p.stat().st_size==r['bytes'];p.unlink()
    audit.update(after_free_bytes=shutil.disk_usage(OUT).free,all_deleted=all(not Path(r['path']).exists() for r in targets),all_published_retained=all(Path(r['path']).is_file() for r in kept))
    save(OUT/'analysis/cleanup_completed.json',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ('targets','kept')}),flush=True)

def finalize():
    audit=read(OUT/'analysis/science_audit.json');clean=read(OUT/'analysis/cleanup_completed.json');pub=read(OUT/'analysis/publication.json')
    phase_results={p:read(next(RESULT.glob(f'phase{p}_*/analysis/final.json'))) for p in range(2641,2647)}
    envelopes=read(OUT/'analysis/response_envelopes.json');third=read(OUT/'third_field_confirmation/analysis/completion.json')
    checks={**audit['checks'],'all_six_prior_phases':all(r['all_checks_passed'] for r in phase_results.values()),
        'client_api_and_build':read(OUT/'analysis/delivery_checks.json')['all_checks_passed'],'post_cleanup_checks':read(OUT/'analysis/post_cleanup_checks.json')['all_checks_passed'],
        'only_manifested_raw_deleted':clean['all_deleted'],'all64_published_raw_packs_retained':clean['all_published_retained'],'prospective1024_internal_fields_checked':third['all_checks_passed']}
    assert all(checks.values())
    summary={'behavior':read(BF/'analysis/factorial_audit.json'),'atlas':audit['summary'],'numeric':phase_results[2646]['summary']['scalar_numeric'],
        'coordinate_envelope_discovery':envelopes['summary'],'prospective_internal_field_confirmation':third['summary'],
        'client':pub,'cleanup':{k:clean[k] for k in ('deleted_files','deleted_bytes','recoverability')},
        'next_same_goal':True,'next_scope':'Cross operation with output function: named-person selection versus natural yes/no truth judgment and factual continuation, separate fixed output-head rows from varying entities. Freeze full-coordinate amplitude/signed envelope rules before new fields, retain all coordinates and heldout forms/entities. Trace calibrated scalar paths instead of repeating generic matched-answer cosine.'}
    finish(2647,'完整自然全坐标图谱、同输出混杂审计与客户端存储交付',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '补齐全部1024 BF16自然原场的全token逐坐标RMS与FP32差异，以及目标/顺序/句式原生响应。相同/相反输出向量逐坐标验证伴随恒等，匹配相同样本对再比较原生和共同读出，避免不同样本集混比。',
        r'R_{l,j}^{16}=\sqrt{\frac{\sum_{x,t}(H^{16}_{x,l,t,j})^2}{\sum_xT_x}};\quad G_{jk}=\sum_t\bar V_{t,j}X_{t,k};\quad I^{s}_{g,l,j}=\mathbf1[R^{s,target}_{g,l,j}>\max(R^{s,order}_{g,l,j},R^{s,form}_{g,l,j})];\quad C_{l,j}=\sum_{g=1}^{16}I^{initial}_{g,l,j}I^{confirmation}_{g,l,j}I^{third}_{g,l,j}.',
        '完整合同2641—2647：4096自然生成、初始512和独立扩大512、37隐藏检查点与36层全部MLP、四层V全参数因子、1088数值条件。观察后增加全坐标目标幅度包络提取，冻结后再测1024未采内部场条件（单位2..9，三个锚点全部坐标流式汇总，无逐例原包）。客户端8新热力图和32示例逐参数查询；保留64双精度原包，清理1984未展示原包。',
        '可复用拼图是：精确单参数全token导数算法、八族条件对照地图、读出身份受控的跨操作末层公共纹理。自然BF16末层目标响应在16组均大于顺序/句式响应；其逐坐标RMS与FP32余弦0.999831—0.999959，注意RMS不保留符号。冻结后第三批1024条件，隐藏幅度候选759/886=85.67%通过全部16组，MLP1186/1570=75.54%；附加逐组跨实体集合的同号要求，只保留8/104和16/152。幅度包络比固定方向稳定，是候选条件编码拼图，不是固定词义神经元。共同读出的稳定性与早中层差异应并存描述，不能把输出端相似叫作语言语义齿轮。完整名字识别3996/4096=97.5586%，严格格式2996/4096=73.1445%；所有真实错误保留。单权重八个层/读出组的汇总相对L1误差0.3896%—2.3014%，末token近似98.95%—99.91%；这是数值预测，不是语言准确率。',
        '只有一个模型，八族均要求选人名，中英各32对实体在多个条件复用；没有覆盖无限组合语言。512扩大是4新实体对×双语等重复，不是512独立语义样本。包络规则为观察后提出，第三批1024才是冻结后的内部场复核；其BF16行为此前已测，不伪称完全未知语料。包络同号是每个组内部跨实体集合同号，不是跨族统一方向，也不是语义主干。解析导数是架构已有链式规律，不是已破解编码机制。客户端已做API精确值和构建验证，未做浏览器交互视觉验收。',
        '本大阶段全部完成。下一阶段目标相同，继续既有自动续研：交叉人名选择、自然是/否判断与事实续写，固定输出头行和改变实体词汇分别控制，检验幅度包络与方向分化的来源。一个候选第一性原理是“共享参数反复作用于位置、上下文条件决定响应”，但其语言组织规律仍待提取；无需先宣称新数学，亦不因单个删除失败否定分布式编码。有限上下文模型的组合泛化不等于已证明字面无限能力。先可判定广材料与自然行为，再原坐标地图、独立确认和适量真实标量验证，不只增加同输出余弦。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','publish','cleanup','finalize']);a=p.parse_args();globals()[a.action]()
