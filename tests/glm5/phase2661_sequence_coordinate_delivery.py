"""Whole-campaign scientific accounting, exact scalar publication and safe raw retention."""
import argparse,os,shutil
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2628_native_atlas_delivery import ASSET,panel,row
from phase2655_truth_answer_contract import OUT as MATERIAL
from phase2656_truth_answer_behavior import OUT as BF
from phase2657_truth_answer_maps import OUT as MAPS
from phase2658_sequence_parameter_engine import OUT as FP,LAYERS
from phase2660_qwen14_truth_answer_replication import OUT as Q14

OUT=RESULT/'phase2661_sequence_coordinate_delivery'


def audit():
    phases={p:read(next(RESULT.glob(f'phase{p}_*/analysis/final.json'))) for p in range(2655,2661)}
    material={r['case_index']:r for r in read(MATERIAL/'material/cases.json')};bf=read(BF/'analysis/records.json');fp=read(FP/'analysis/records.json');q14=read(Q14/'analysis/records.json')
    behavior={}
    for lang in ('en','zh'):
        for q in (0,1):
            for m in (0,1):
                rr=[r for r in bf if (r['language'],r['polarity'],r['mapping'])==(lang,q,m)]
                behavior[f'{lang}/q{q}/m{m}']={'n':len(rr),'content_correct':sum(r['content_correct'] for r in rr),'leading_correct':sum(r['decision'] is not None and r['decision']['answer_yes']==r['expected_yes'] for r in rr),
                    'no_recognizable_answer':sum(r['decision'] is None for r in rr),'recognized_after_first':sum(r['decision'] is not None and r['decision']['step']>0 for r in rr),'eos':sum(r['eos'] for r in rr)}
    sequence={}
    for m in (0,1):
        rr=[r for r in fp if r['mapping']==m];sequence[f'mapping{m}']={'n':len(rr),
            'canonical_first_score_expected_sign':sum((1 if material[r['case_index']]['expected_yes'] else -1)*r['first_token_contrast']>0 for r in rr),
            'canonical_complete_score_expected_sign':sum((1 if material[r['case_index']]['expected_yes'] else -1)*r['contrast']>0 for r in rr),
            'first_complete_sign_differ':int(sum(np.sign(r['first_token_contrast'])!=np.sign(r['contrast']) for r in rr)),
            'eos_absolute_contribution_mean':float(np.mean([abs(r['eos_contrast']) for r in rr]))}
    checks={'six_prior_phases_complete':all(r['all_checks_passed'] for r in phases.values()),'8192_natural':len(bf)==8192,'256_sequence_gradients':len(fp)==256,'256_second_model':len(q14)==256,
        'preserved_phase2654_asset':sha(ASSET)==read(RESULT/'phase2654_output_function_delivery/analysis/publication.json')['asset_sha256'],
        'allold_frozen_masks_unchanged':sha(MATERIAL/'maps/frozen_previous_coordinates.npz')==read(MATERIAL/'protocol/frozen.json')['frozen_coordinate_sha256']}
    # Idempotent prepublication audit remains valid after this phase's own asset revision.
    if (OUT/'analysis/publication.json').exists():checks['preserved_phase2654_asset']=read(OUT/'analysis/publication.json')['previous_asset_sha256']==read(RESULT/'phase2654_output_function_delivery/analysis/publication.json')['asset_sha256']
    summary={'behavior_qwen4':behavior,'old_coordinate_confirmation_and_factors':phases[2657]['summary'],'sequence_probabilities':sequence,
        'sequence_engine':phases[2658]['summary'],'single_parameter_numeric':phases[2659]['summary'],'qwen14':phases[2660]['summary'],
        'independence_boundary':'8192 prompts share8 entitypairs/language and reused catalogs/templates. Scalar128 prefixes use one heldout entitypair/language across16family/language groups and8 probe/polarity/mapping cells; not128 independent semantic problems.',
        'numerical_boundary':'The derivative identity is standard chain rule, not a newly discovered language code. At8 frozen V coordinates, whole-sequence alltoken finite-change prediction extends the measurement tool; semantic interpretation still needs condition-specific structure.',
        'runtime_adaptations':{'phase2658':phases[2658].get('recording_recovery'),'phase2660':read(Q14/'protocol/runtime_adaptation.json')},
        'separation_boundary':'Correct natural reverse-label performance is required before sign invariance can distinguish truth vs answer preparation. Canonical two-string likelihood is not total natural answer mass. Qwen14 own coordinate maps are not a Qwen4 index alignment or full old amplitude-rule replication.'}
    assert all(checks.values());save(OUT/'analysis/science_audit.json',{'summary':summary,'checks':checks,'all_checks_passed':True})


def publish():
    assert read(OUT/'analysis/science_audit.json')['all_checks_passed']
    first={r['case_index']:r for r in read(FP/'analysis/records.json')};behavior={r['case_index']:r for r in read(BF/'analysis/records.json')}
    published=[{**r,**first[r['case_index']],'generated':behavior[r['case_index']]['generated'],'content_correct':behavior[r['case_index']]['content_correct']} for r in read(MATERIAL/'material/cases.json') if r['published']]
    save(OUT/'material/published_cases.json',published);rr={k:[] for k in ('qwen4_h','qwen4_mlp','sequence_parameter_row')}
    for r in published:
        ci=r['case_index']
        with np.load(BF/f'field/case_{ci:04d}.npz') as b,np.load(FP/f'field/case_{ci:04d}.npz') as f:
            for l in (0,18,36):
                rr['qwen4_h'].append(row(f'{r["case_id"]}/BF16/H{l}/prompt',b['hidden_fulltoken'][l,-1],2661,'native_hidden_coordinate',l))
                if 'decision_hidden_fulltoken' in b:rr['qwen4_h'].append(row(f'{r["case_id"]}/BF16/H{l}/observable_answer_time',b['decision_hidden_fulltoken'][l,-1],2661,'native_hidden_coordinate',l))
            rr['qwen4_mlp'].append(row(f'{r["case_id"]}/BF16/MLP35',b['mlp_boundary'][35],2661,'actual_mlp_neuron',35))
            for l,j in ((0,57),(35,137)):
                gradients=[]
                for label in ('Y','N'):
                    x=f[f'{label}__L{l}_v_x'].astype('float64');g=f[f'{label}__L{l}_v_g'][:,j].astype('float64');gradients.append((x*g[:,None]).sum(0))
                rr['sequence_parameter_row'].append(row(f'{r["case_id"]}/sequence_logprob/L{l}/Vrow{j}/all_k',gradients[0]-gradients[1],2661,'actual_scalar_sequence_logprob_derivative',l))
    meanings={'qwen4_h':('Natural original and observed-answer-time full hidden coordinates',2560),'qwen4_mlp':('All actual final MLP neurons: polarity and response convention',9728),
        'sequence_parameter_row':('Actual V row: full canonical answer-plus-EOS parameter derivatives',2560)}
    boundary='Every physical coordinate retained, noTopK analysis. BF16 natural behavior versus FP32 numeric fields. Full answer means only two canonical answer+EOS strings with teacher forcing, not all valid language responses.'
    new=[panel('phase2661_'+key,meanings[key][0],meanings[key][1],values,boundary) for key,values in rr.items()]
    with np.load(MAPS/'maps/allcoordinate_factor_sign_counts.npz') as z:
        for metric,dim,ll in [('h',2560,(0,6,18,36)),('mlp',9728,(0,5,17,35))]:
            values=[]
            for fold in ('initial','confirmation'):
                for hypothesis in ('truth_invariant','question_affirmative','answer_label'):
                    for l in ll:values.append(row(f'{fold}/{metric}{l}/{hypothesis}/groups0..32',z[f'{fold}__{metric}__{hypothesis}'][l],2661,'native_coordinate_factor_sign_count',l))
            for kind in ('amplitude','signed'):
                for l in ll:values.append(row(f'oldBFsign_fullcoordinate_confirmation/{metric}{l}/{kind}/groups0..32',z[f'old__bf__{metric}__{kind}'][l],2661,'old_sign_allcoordinate_confirmation_before_candidate_mask',l))
            new.append(panel('phase2661_factor_'+metric,'Truth / question / answer sign alternatives, all native '+metric,dim,values,'Counts of32 family/language/probe groups. Simple descriptive direction alternatives; reverse-mapping behavioral failure must be checked. No semantic-core claim.'))
    qcases=[r for r in read(Q14/'material/cases.json') if r['published']];qinfo=read(Q14/'protocol/model.json');qh=[];qa=[]
    for r in qcases:
        with np.load(Q14/f'field/case_{r["case_index"]:04d}.npz') as z:
            for l in (0,qinfo['dimensions']['layers']//2,qinfo['dimensions']['layers']):qh.append(row(f'{r["case_id"]}/Qwen14/H{l}',z['hidden_fulltoken'][l,-1],2661,'qwen14_native_coordinate',l))
            qa.append(row(f'{r["case_id"]}/Qwen14/finalMLP',z['mlp_boundary'][-1],2661,'qwen14_actual_mlp_unit',qinfo['dimensions']['layers']-1))
    new.append(panel('phase2661_qwen14_h','Qwen14B own physical embedding and hidden coordinates',qinfo['dimensions']['hidden'],qh,'BF16 nonquantized, model-local coordinates, no coordinate-index alignment withQwen4.'))
    new.append(panel('phase2661_qwen14_mlp','Qwen14B all actual final MLP units',qinfo['dimensions']['mlp'],qa,'BF16 nonquantized, model-local actual MLP intermediate units, not residual coordinates or independent semantic neurons.'))
    with np.load(Q14/'maps/allcoordinate_sign_group_counts.npz') as z:
        for metric,dim in [('hidden_boundary',qinfo['dimensions']['hidden']),('mlp_boundary',qinfo['dimensions']['mlp'])]:
            vals=[row(f'Qwen14/{metric}/{hyp}/L{l}/groups0..16',z[metric+'__'+hyp][l],2661,'qwen14_native_factor_sign_count',l) for hyp in ('statement_truth','question_affirmative','answer_label') for l in (0,5,17,z[metric+'__'+hyp].shape[0]-1)]
            new.append(panel('phase2661_qwen14_factor_'+metric,'Qwen14B native factor sign atlas '+metric,dim,vals,'Oneentitypair/language,16family/languagegroups; not old target>order/form amplitude-rule replication and no universal mechanism.'))
    payload=read(ASSET);original=sha(ASSET);prior=[p['key'] for p in payload['models'] if not p['key'].startswith('phase2661_')]
    previous_own={p['key']:p for p in payload['models'] if p['key'].startswith('phase2661_')}
    same_numeric_rows=bool(previous_own) and all(p['key'] in previous_own and p['coordinate_count']==previous_own[p['key']]['coordinate_count'] and [r['values'] for r in p['rows']]==[r['values'] for r in previous_own[p['key']]['rows']] for p in new)
    if previous_own:assert same_numeric_rows,'Republishing this phase may correct labels but must not silently change scientific values'
    payload['models']=[p for p in payload['models'] if not p['key'].startswith('phase2661_')]+new;payload['phase']=2661;payload['claim_boundary']=boundary
    payload['summary']['model_rows']={p['key']:len(p['rows']) for p in payload['models']};payload['summary']['total_rows']=sum(payload['summary']['model_rows'].values())
    temp=ASSET.with_suffix('.phase2661.tmp');assert not temp.exists();temp.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8');os.replace(temp,ASSET)
    old=read(OUT/'analysis/publication.json') if (OUT/'analysis/publication.json').exists() else None
    proof={'previous_asset_sha256':old['previous_asset_sha256'] if old else original,'asset_sha256':sha(ASSET),'asset_bytes':ASSET.stat().st_size,'prior_panel_keys':prior,
        'label_revision':{'previous_own_asset_sha256':original,'all_numeric_rows_unchanged':same_numeric_rows} if previous_own else None,
        'panels':[{'key':p['key'],'width':p['coordinate_count'],'rows':len(p['rows'])} for p in new],'qwen4_published':[r['case_index'] for r in published],'qwen14_published':[r['case_index'] for r in qcases]}
    save(OUT/'analysis/publication.json',proof);print(json.dumps({'panels':len(new),'qwen4_cases':len(published),'qwen14_cases':len(qcases),'asset_bytes':proof['asset_bytes']}),flush=True)


def next_plan():
    """Freeze confirmed coordinates for the next campaign without calling it executed."""
    masks={};indices={}
    with np.load(MATERIAL/'maps/frozen_previous_coordinates.npz') as old,np.load(MAPS/'maps/allcoordinate_factor_sign_counts.npz') as z:
        for precision in ('bf','fp'):
            for metric in ('h','mlp'):
                om=('bf_' if precision=='bf' else '')+metric;mask=old[om+'__truth_oriented_opposite']
                if metric=='h':mask=mask[:,2]
                key=precision+'_'+metric;masks[key]=(mask.astype(bool)&(z[f'old__{precision}__{metric}__signed']==32)).astype('uint8')
                indices[key]=np.flatnonzero(masks[key][-1]).tolist()
    assert len(indices['bf_h'])==6 and len(indices['bf_mlp'])==14
    OUT.joinpath('maps').mkdir(parents=True,exist_ok=True);path=OUT/'maps/confirmed_native_coordinate_masks.npz';np.savez_compressed(path,**masks)
    plan={2662:'复审并冻结已确认原坐标；建立正常/反向指令同复杂度、按分词量核验的基础合同及独立校准/确认集合',
        2663:'至少1024预定校准行为，覆盖多种明确真假问法、对称标签映射、无演示与少量演示；只用校准集冻结后续问法，不把其分数算留出结果',
        2664:'冻结问法后8192新材料自然行为和全原坐标场；八族双语、多个实体/句式/目标/顺序/探问/极性/映射，未通过也完整记录',
        2665:'全层H/MLP与源位置到输出边界条件图；旧候选扩大确认与全坐标真值/问题/标签方向图，成功配对只另账展示',
        2666:'256多token规范答案的全序列逐参数因子；内容、标点与EOS逐步分账，分支形状/掩码数值检查，不冒充自由生成',
        2667:'冻结真实标量在留出材料上扩大至少2048改动验证；保留全部位置和低值坐标，不用搬运作核心',
        2668:'Qwen14B串行非量化模型内复验，预定1024条件并覆盖至少双实体对/双问法；显式注明是否复制幅度控制，不硬对齐模型坐标',
        2669:'科学复核、客户端真实参数与全坐标交付、明确清单清理、完整MEMO记录和同目标后续决策'}
    report={'status':'planned_not_executed','same_goal':True,'start_after_completed_phase':2661,'plan':plan,'confirmed_mask_sha256':sha(path),'last_layer_indices':indices,
        'precision_boundary':'bf keys: BF16 candidate and BF16 confirmation. fp keys: old FP32-derived masks tested on NEW BF16, not new full-FP32 confirmation.',
        'decision_basis':'4B reverse mapping weak; 14B normal q0 content32/32 per language, q1normal28/32 EN27/32 ZH, reverse q0 16/32 EN9/32 ZH and q1 3/32 EN8/32 ZH. Do not equate failure to absent truth structure; first remove asymmetric instruction complexity.',
        'priority':'Language-coordinate regularities first. Exact chain rule is a measurement algorithm, not semantic decoding. Frozen confirmed candidates guide inquiry but all coordinates remain primary.',
        'constraints':'New datasets/family instructions are frozen before heldout forward. Do not reuse cleaned unshown packs or rewrite completed phases. If data contract changes, record before execution and retain limits.'}
    save(OUT/'analysis/next_campaign.json',report);print(json.dumps({'same_goal':True,'planned_phases':list(plan),'last_layer_indices':indices}),flush=True)


def cleanup():
    assert not (OUT/'analysis/cleanup_completed.json').exists();assert read(OUT/'analysis/delivery_checks.json')['all_checks_passed'];assert read(OUT/'analysis/science_audit.json')['all_checks_passed']
    assert read(OUT/'analysis/scientific_checks.json')['all_checks_passed']
    pub=read(OUT/'analysis/publication.json');kept=[];targets=[]
    for source,key in ((BF,'qwen4_published'),(FP,'qwen4_published'),(Q14,'qwen14_published')):
        for r in read(source/'analysis/raw_manifest.json'):
            p=Path(r['path']).resolve();assert p.parent==(source/'field').resolve() and p.is_relative_to(RESULT.resolve()) and re.fullmatch(r'case_\d{4}\.npz',p.name)
            assert p.is_file() and p.stat().st_size==r['bytes'];entry={**r,'path':str(p),'sha256':sha(p)};(kept if r['case_index'] in pub[key] else targets).append(entry)
    assert len(kept)==144 and len(targets)==8560 and len({r['path'] for r in kept+targets})==8704
    report={'targets':targets,'kept':kept,'deleted_files':len(targets),'deleted_bytes':sum(r['bytes'] for r in targets),'before_free_bytes':shutil.disk_usage(OUT).free,
        'recoverability':'Direct removal, not Recycle Bin. Allbehavior/material/token/readout/EOS definitions, model/runtime records, scripts, derivedallcoordinate maps and144published packs retained. Recompute into new directory; no completedPhase/MEMO overwrite.'}
    save(OUT/'analysis/cleanup_plan.json',report)
    for r in targets:
        p=Path(r['path']);assert p.stat().st_size==r['bytes'];p.unlink()
    report.update(all_deleted=all(not Path(r['path']).exists() for r in targets),all_kept=all(sha(r['path'])==r['sha256'] for r in kept),after_free_bytes=shutil.disk_usage(OUT).free)
    save(OUT/'analysis/cleanup_completed.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('targets','kept')}),flush=True)


def finalize():
    a=read(OUT/'analysis/science_audit.json');clean=read(OUT/'analysis/cleanup_completed.json');pub=read(OUT/'analysis/publication.json')
    checks={**a['checks'],'independent_scientific_recomputation':read(OUT/'analysis/scientific_checks.json')['all_checks_passed'],'api_and_build':read(OUT/'analysis/delivery_checks.json')['all_checks_passed'],'postcleanup_api':read(OUT/'analysis/post_cleanup_checks.json')['all_checks_passed'],
        'only8560_manifested_raw_deleted':clean['all_deleted'],'all144_published_retained':clean['all_kept']};assert all(checks.values())
    summary={**a['summary'],'client':pub,'cleanup':{k:clean[k] for k in ('deleted_files','deleted_bytes','recoverability')},'next_same_goal':True,'next_campaign':read(OUT/'analysis/next_campaign.json')}
    finish(2661,'完整答案逐参数与事实—问题—标签全坐标研究交付',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '完整执行2655—2661的自然行为、旧候选冻结确认、新因素全坐标图、完整规范序列概率导数、真实单参数数值检验、第二模型观察、客户端和清理。原始H、MLP中间单位和真实权重分开；证据按行为与数值边界分别解释。',
        r'\log\frac{P(Y,EOS|x)}{P(N,EOS|x)}=\log\frac{P(Y|x)}{P(N|x)}+\log\frac{P(EOS|x,Y)}{P(EOS|x,N)};\quad G_{jk}=\sum_t\bar V^Y_{t,j}X^Y_{t,k}-\sum_t\bar V^N_{t,j}X^N_{t,k}.',
        '8192 Qwen4B未干预BF16完整生成和全部原坐标边界；256初始/确认FP32规范答案序列因素；128独立前缀×17=2176数值条件，2048次真实单参数改动；Qwen14B非量化CPU/GPU串行256自然条件。9新面板、64 Qwen4逐参数实例、16 Qwen14实例；保留144原包，清理8560未展示包。',
        '冻结旧BF16候选在新实体集合中仍有6/11个末层H坐标和14/32个MLP单元通过全部组的幅度及方向条件，构成有限的稳定复用拼图，而非纯真值单元。4B反向规则/否定提问行为下降，使三种方向假说未能得到可判定的全面分离。参数工具方面，标准链式法则把跨token、跨候选上下文和结束概率纳入同一个真实共享参数导数；2048次单权重改动的四层汇总相对L1误差0.5943%—1.5694%，末位置错误近似约97%—101%。这是局部数值工具进展，不是新数学定律或语义必要性。7/256前缀首词与答案加EOS的候选排名反转，说明内容偏好与终止选择必须分账。',
        'Qwen4词表/模板复用；128个参数验证前缀只有每语言1个留出实体对，跨八族和八条件重复，不是128个独立语义问题。Qwen14每语言1实体对、1句式和顺序，未复制旧幅度控制门。规范答案加EOS是两条teacher-forced序列，不是所有自然答案总概率；EOS改变排名不代表语义推理改变。基底内全部组同号是严格描述性条件，失败不能排除混合或非线性编码。API精确值与构建不是浏览器视觉验收。原生导数算法已更细，但语言编码规律仍只积累有限拼图。',
        '本大阶段全部完成。下一目标仍是条件原坐标如何复用、分化并编译为输出。应基于本批实际自然胜任度挑选可判定的新因素合同，而非把反向映射失败简单判作无真值结构；同时区分答案词内容与终止/格式决策，在更长真实答案上验证全序列标量算法。保持全坐标、冻结后扩大和模型内基底原则，沿既有自动续研继续。')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','publish','next_plan','cleanup','finalize']);a=p.parse_args();globals()[a.action]()
