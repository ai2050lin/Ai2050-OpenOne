"""Complete campaign audit, exact coordinate publication and allowlisted raw cleanup."""
import argparse,os,shutil
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2628_native_atlas_delivery import ASSET,panel,row
from phase2662_symmetric_mapping_contract import OUT as CONTRACT,leading_answer
from phase2664_symmetric_native_field import OUT as SHORT
from phase2665_symmetric_coordinate_maps import OUT as MAPS
from phase2666_multitoken_parameter_engine import OUT as FP,PARTS,LAYERS
from phase2668_qwen14_symmetric_replication import OUT as Q14

OUT=RESULT/'phase2669_symmetric_multitoken_delivery';BF=FP/'natural'


def confusion(records):
    groups={}
    for r in records:groups.setdefault(f'{r["language"]}/q{r["polarity"]}/m{r["mapping"]}',[]).append(r)
    result={}
    for k,rr in groups.items():
        matrix={f'expected{truth}/observed{answer}':0 for truth in (False,True) for answer in (False,True,None)}
        for r in rr:matrix[f'expected{r["expected_yes"]}/observed{leading_answer(r["generated"],r["language"])}']+=1
        result[k]={'n':len(rr),'content_correct':sum(r['content_correct'] for r in rr),'strict_correct':sum(r['strict_correct'] for r in rr),'leading_label_counts_not_semantic_parse':matrix}
    pairs={}
    for r in records:pairs.setdefault(tuple(r[k] for k in ('family','language','unit','form','target_index','mention_order','probe_index','polarity','style','shots')),[]).append(r)
    return {'groups':result,'mapping_pairs':len(pairs),'both_mappings_content_correct':sum(len(rr)==2 and all(r['content_correct'] for r in rr) for rr in pairs.values()),
        'boundary':'Leading-code counts do not interpret explanations; full-content scoring remains primary. Paired mapping success requires both opposite expected responses, not just50% marginal score.'}


def audit():
    phases={p:read(next(RESULT.glob(f'phase{p}_*/analysis/final.json'))) for p in range(2662,2669)};short=read(SHORT/'analysis/records.json');q14=read(Q14/'analysis/records.json');fp=read(FP/'analysis/records.json')
    with np.load(MAPS/'maps/confirmed_masks.npz') as z:survivors={k:np.argwhere(z[k]).tolist() for k in z.files}
    checks={'seven_prior_phases_complete':all(r['all_checks_passed'] for r in phases.values()),'8192_short':len(short)==8192,'256_multitoken':len(fp)==256,'1024_qwen14':len(q14)==1024,
        'old_mask_frozen':sha(CONTRACT/'maps/frozen_masks.npz')==read(CONTRACT/'protocol/frozen.json')['frozen_mask_sha256']}
    summary={'short_confusions':confusion(short),'calibration':phases[2663]['summary'],'native_maps':phases[2665]['summary'],'surviving_native_coordinates':survivors,
        'multitoken_engine':phases[2666]['summary'],'actual_single_parameters':phases[2667]['summary'],'qwen14':phases[2668]['summary'],
        'qwen14_confusions_by_style':{str(s):confusion([r for r in q14 if r['style']==s]) for s in (0,1)},
        'candidate_scope_audit':read(OUT/'analysis/candidate_scope_summary.json'),
        'next_mlp_formula_synthetic_preflight_not_model_result':read(OUT/'analysis/mlp_formula_preflight.json'),
        'claim_corrections':[
            '几千个Phase没有闭合，提示材料、测量对象、对照和理论接口需要调整；这本身既不证明整个范式致命错误，也不证明必须发明新数学。',
            '固定参数基底中的可重复单坐标/单元响应是真实可测对象，但单坐标不是天然独立语义符号；条件共同响应不等于必要性或充分性。',
            '问题或映射失败限制语义解释，不能作为关闭中层全场路线的唯一依据；失败中仍保留纹理，但不以稳定纹理替代行为胜任。',
            '无限组合按研究目标理解为组合泛化，有限精度与有限上下文并不提供无限个可区分内部状态；有限实验不能证明全部语言组合的完备机制。'],
        'limitations':['Midlayer survivors are candidate conditional textures, not pure semantic units. The old-mask confirmation tests only q0/m0, all32 family/language/probe groups in both new entity sets; it does NOT require survival across reversed questions or mappings. Whole maps are primary; selection does not remove other coordinates.',
            'Equal tokenizer length alone does not equalize instruction difficulty; demos may introduce scaffolding and recency effects.',
            'Each canonical answer is4tokens including fixedformat plusEOS; this is not a long natural explanation or all answer-class probability.',
            'Exact chain rule and finite scalar predictions are measurement controls, not language mechanism closure.',
            'Scalar total-score relative L1 errors are0.59%-1.78%; content and format effects are measurable, but mean absolute EOS effects are only about5e-10 to5e-9 and relative errors exceed100%. EOS finite-change sensitivity is numerically unresolved here, not a successful fine-grained prediction.']}
    assert all(checks.values());save(OUT/'analysis/science_audit.json',{'summary':summary,'checks':checks,'all_checks_passed':True})
    next_campaign(summary)


def next_campaign(summary):
    with np.load(Q14/'maps/old_coordinate_reconfirmation.npz') as z:q14_sites={k:np.argwhere(z[k]).tolist() for k in z.files}
    plan=[
        {'phase':2670,'task':'冻结中层H24[2355]、H27[1217]与MLP23[6197]、26[3594]、27[3221]、28[5952,8513]及全部背景坐标，复审其仅q0/m0旧方向确认范围；设计新的实体、句式、位置和任务格式交叉，不将旧失效条件删除。'},
        {'phase':2671,'task':'至少8192条Qwen4B非量化自然行为与全坐标gate/up/SiLU乘积/down测绘：8族×2语言×4实体对×2独立内容实例×2形式×2顺序×2目标×2探问×2极性×2映射。实体与词义/物品/数值实例解除绑定，2实体对发现+2确认。保留反问/映射失败条件另账。正文和任务边界的原始H及所有MLP单元全量，所有token逐坐标流式记录矩和指定展示原轨迹，不用TopK定义主线；先精确预估硬盘，不以未执行数据凑数。'},
        {'phase':2672,'task':'按冻结候选追踪每个真实MLP单元全部输入k的Wgate[j,k]*x[k]和Wup[j,k]*x[k]、以及该单元全部输出i的Wdown[i,j]*a[j]；同时保存全MLP分支实测输出和全坐标背景图，区分单元、残差坐标和参数。'},
        {'phase':2673,'task':'在新材料确认集检验门/上支路/下投影的条件复用和分化；使用逐坐标有限变化的精确乘积展开并单列SiLU非线性余项、RMSNorm影响、Attention/残差旁路。分解是记账，不将标准恒等式当作新语言规律；不搬运donor场。'},
        {'phase':2674,'task':'围绕仍通过的单元，在至少128条件核对小剂量真实gate/up/down标量变化、数值精度、完整内容/格式/EOS分数及联合冗余；每次恢复整矩阵并核对哈希。候选和匹配低值/普通坐标对照事前冻结，单参数阴性不关闭主线。'},
        {'phase':2675,'task':'顺序加载本地Qwen14B、GLM4、DS7B做同任务行为与模型内全坐标复验；每模型至少512条件，先明确实际gated-MLP结构和维度，禁止跨模型同下标对齐。BF16非量化、auto内存分配，加载异常先安全排查；不得将未运行结果记作完成。'},
        {'phase':2676,'task':'整合全部语言族和失败条件，扩大复测重要阳性；发布真实词嵌入/H/MLP及gate/up/down参数热力图和精确查询，完整审计与留存白名单清理；逐Phase追加MEMO。依据通过规律安排下一整批同目标研究。'}]
    save(OUT/'analysis/next_campaign.json',{'frontier':2669,'same_goal':True,'plan':plan,'frozen_survivors':summary['surviving_native_coordinates'],
        'q14_native_survivors':q14_sites,'q14_mask_sha256':sha(Q14/'maps/old_coordinate_reconfirmation.npz'),
        'q14_scope':'62H and125MLP old-candidate intersections retain statement-truth sign patterns across64new family/language/entity/style groups. H26:18/old154,MLP24:42/old297. No old mean-sign requirement or target>form/order gate. Freeze ALL survivors prospectively in2670, not just the4B sites.',
        'priority':'观察全坐标语言规律→追踪真实MLP计算→条件确认→局部数值/因果校验。不要把4096条件写成4096独立语义，也不要用必要性阴性终止路线。',
        'minimum_sufficiency':'Each phase includes multiple related controls; do not mark unexecuted work complete. Hold model weights and library files immutable; store only owned raw arrays for eventual allowlisted cleanup.'})


def prepare_examples():
    fm={r['case_index']:r for r in read(FP/'analysis/records.json')};bm={r['case_index']:r for r in read(BF/'analysis/records.json')}
    cases=[{**r,**fm[r['case_index']],'generated':bm[r['case_index']]['generated'],'content_correct':bm[r['case_index']]['content_correct']} for r in read(FP/'material/cases.json') if r['published']]
    assert len(cases)==64;save(OUT/'material/published_cases.json',cases);return cases


def publish():
    assert read(OUT/'analysis/science_audit.json')['all_checks_passed'];cases=prepare_examples()
    short=[r for r in read(SHORT/'material/cases.json') if r['published']];qcases=[r for r in read(Q14/'material/cases.json') if r['published']];new=[]
    h=[];a=[];mh=[];parameter=[]
    for r in short:
        with np.load(SHORT/f'field/case_{r["case_index"]:04d}.npz') as z:
            for l in (0,24,27,36):
                for pos,name in ((0,'sourceA'),(1,'sourceB'),(2,'output')):h.append(row(f'{r["case_id"]}/H{l}/{name}',z['hidden_anchor'][l,pos],2669,'native_hidden_coordinate',l))
            for l in (23,26,27,28,35):a.append(row(f'{r["case_id"]}/MLP{l}',z['mlp_boundary'][l],2669,'actual_mlp_neuron',l))
    for r in cases:
        with np.load(BF/f'field/case_{r["case_index"]:04d}.npz') as b,np.load(FP/f'field/case_{r["case_index"]:04d}.npz') as f:
            for l in (0,24,27,36):mh.append(row(f'{r["case_id"]}/multitoken_condition/BF16/H{l}',b['hidden_boundary'][l],2669,'native_hidden_coordinate',l))
            for l,j in ((0,57),(35,137)):
                for part in ('all',)+PARTS:
                    suffix='' if part=='all' else '_'+part;gg=[]
                    for label in ('Y','N'):gg.append((f[f'{label}__L{l}_v_x'].astype('float64')*f[f'{label}__L{l}_v_g{suffix}'][:,j].astype('float64')[:,None]).sum(0))
                    parameter.append(row(f'{r["case_id"]}/L{l}/Vrow{j}/{part}/all_k',gg[0]-gg[1],2669,'actual_scalar_multitoken_derivative',l))
    boundary='All physical columns retained. NativeH,actualMLPunits and scalarweights are distinct. Midlayer candidates are not semantic necessity. NaturalBF16 and same-valuedFP32 numeric scores separate.'
    for key,title,dim,rows in [('short_h','Symmetric rules: source and output full H coordinates',2560,h),('short_mlp','Symmetric rules: all actual mid/late MLP units',9728,a),
        ('multi_h','Multi-token answer instruction: raw BF16 H coordinates',2560,mh),('parameter_parts','Real V row: all-coordinate content/format/EOS derivatives',2560,parameter)]:new.append(panel('phase2669_'+key,title,dim,rows,boundary))
    with np.load(MAPS/'maps/allcoordinate_factor_counts.npz') as z,np.load(MAPS/'maps/confirmed_masks.npz') as masks:
        for metric,dim,ll in [('h',2560,(0,24,27,36)),('mlp',9728,(0,23,26,27,28,35))]:
            rows=[row(f'{fold}/{metric}{l}/{hyp}/groups0..32',z[f'{fold}__{metric}__{hyp}'][l],2669,'native_direction_group_count',l) for fold in ('initial','confirmation') for hyp in ('truth_invariant','question_affirmative','answer_label') for l in ll]
            rows.extend(row(f'old_confirmed/{metric}{l}/mask0or1',masks[metric][l],2669,'frozen_candidate_membership_not_sparse_basis',l) for l in ll)
            new.append(panel('phase2669_factor_'+metric,'All-coordinate sign alternatives and candidate confirmation '+metric,dim,rows,boundary))
    qh=[];qa=[];info=read(Q14/'protocol/model.json')
    for r in qcases:
        with np.load(Q14/f'field/case_{r["case_index"]:04d}.npz') as z:
            for l in (0,26,40):qh.append(row(f'{r["case_id"]}/Qwen14/H{l}',z['hidden_boundary'][l],2669,'qwen14_model_local_hidden',l))
            for l in (24,39):qa.append(row(f'{r["case_id"]}/Qwen14/MLP{l}',z['mlp_boundary'][l],2669,'qwen14_model_local_mlp',l))
    new.append(panel('phase2669_q14_h','Qwen14B all physical embedding/hidden coordinates',5120,qh,'Model-local BF16 coordinates; no crossmodel index alignment.'))
    new.append(panel('phase2669_q14_mlp','Qwen14B all actual mid/late MLP units',17408,qa,'Model-local BF16 actualMLP units; not independent semantic neurons.'))
    with np.load(Q14/'maps/allcoordinate_sign_counts.npz') as z:
        for metric,dim in [('hidden_boundary',5120),('mlp_boundary',17408)]:
            rows=[row(f'Qwen14/{metric}/{hyp}/L{l}/groups0..64',z[metric+'__'+hyp][l],2669,'qwen14_native_direction_count',l) for hyp in ('statement_truth','question_affirmative','answer_label') for l in range(z[metric+'__'+hyp].shape[0])]
            new.append(panel('phase2669_q14_factor_'+metric,'Qwen14B everylayer all-coordinate direction counts '+metric,dim,rows,'64family/language/entity/style groups, sign pattern only; no target>form/order amplitude gate.'))
    native_panel_dir=OUT/'maps/client_panels';native_panel_dir.mkdir(parents=True,exist_ok=True);panel_catalog=[]
    for p in new:
        matrix=np.asarray([r['values'] for r in p['rows']],dtype='float64');assert np.isfinite(matrix).all() and matrix.shape==(len(p['rows']),p['coordinate_count'])
        np.savez_compressed(native_panel_dir/(p['key']+'.npz'),values=matrix)
        panel_catalog.append({'key':p['key'],'title':p.get('model',p['key']),'coordinate_count':p['coordinate_count'],'rows':[{k:v for k,v in r.items() if k!='values'} for r in p['rows']],
            'matrix_sha256':sha(native_panel_dir/(p['key']+'.npz'))})
    save(OUT/'material/client_panel_catalog.json',{'phase':2669,'panels':panel_catalog,'boundary':boundary,'display':'Every physical coordinate stored losslessly as float64. Row paging is display only; no TopK, averaging or dimensional compression.'})
    payload=read(ASSET);oldsha=sha(ASSET);prior=[p['key'] for p in payload['models'] if not p['key'].startswith('phase2669_')]
    payload['models']=[p for p in payload['models'] if not p['key'].startswith('phase2669_')]+new;payload['phase']=2669;payload['claim_boundary']=boundary;payload['summary']['model_rows']={p['key']:len(p['rows']) for p in payload['models']};payload['summary']['total_rows']=sum(payload['summary']['model_rows'].values())
    temp=ASSET.with_suffix('.phase2669.tmp');assert not temp.exists();temp.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8');os.replace(temp,ASSET)
    save(OUT/'analysis/publication.json',{'previous_asset_sha256':oldsha,'asset_sha256':sha(ASSET),'asset_bytes':ASSET.stat().st_size,'prior_panel_keys':prior,
        'panels':[{'key':p['key'],'width':p['coordinate_count'],'rows':len(p['rows'])} for p in new],
        'short_published':[r['case_index'] for r in short],'multi_published':[r['case_index'] for r in cases],'q14_published':[r['case_index'] for r in qcases]});print('Published',len(new),'panels',flush=True)


def cleanup():
    assert not (OUT/'analysis/cleanup_completed.json').exists();assert read(OUT/'analysis/delivery_checks.json')['all_checks_passed'];assert read(OUT/'analysis/scientific_checks.json')['all_checks_passed']
    pub=read(OUT/'analysis/publication.json');kept=[];targets=[]
    for source,key in ((SHORT,'short_published'),(BF,'multi_published'),(FP,'multi_published'),(Q14,'q14_published')):
        for r in read(source/'analysis/raw_manifest.json'):
            p=Path(r['path']).resolve();assert p.parent==(source/'field').resolve() and p.is_relative_to(RESULT.resolve()) and re.fullmatch(r'case_\d{4}\.npz',p.name) and p.stat().st_size==r['bytes']
            entry={**r,'path':str(p),'sha256':sha(p)};(kept if r['case_index'] in pub[key] else targets).append(entry)
    assert len(kept)==224 and len(targets)==9504 and len({r['path'] for r in kept+targets})==9728
    proof={'kept':kept,'targets':targets,'deleted_files':len(targets),'deleted_bytes':sum(r['bytes'] for r in targets),'before_free_bytes':shutil.disk_usage(OUT).free,
        'recoverability':'Direct deletion notRecycleBin. Keep224published packs, models, allbehavior/material/token/category/mask definitions, derivedfullcoordinate maps, scalarconditions, code and hashes. Recompute onlyinto newdirectory, do notoverwrite MEMO/completedPhase.'};save(OUT/'analysis/cleanup_plan.json',proof)
    for r in targets:
        p=Path(r['path']);assert p.stat().st_size==r['bytes'];p.unlink()
    proof.update(all_deleted=all(not Path(r['path']).exists() for r in targets),all_kept=all(sha(r['path'])==r['sha256'] for r in kept),after_free_bytes=shutil.disk_usage(OUT).free);save(OUT/'analysis/cleanup_completed.json',proof);print(json.dumps({k:v for k,v in proof.items() if k not in ('targets','kept')}),flush=True)


def finalize():
    a=read(OUT/'analysis/science_audit.json');pub=read(OUT/'analysis/publication.json');clean=read(OUT/'analysis/cleanup_completed.json');checks={**a['checks'],
        'scientific_checks':read(OUT/'analysis/scientific_checks.json')['all_checks_passed'],'client_build':read(OUT/'analysis/delivery_checks.json')['all_checks_passed'],
        'postcleanup_client':read(OUT/'analysis/post_cleanup_checks.json')['all_checks_passed'],'only_manifested_raw_deleted':clean['all_deleted'],'all224_retained':clean['all_kept'],
        'real_http':read(OUT/'analysis/live_api_checks.json')['all_checks_passed'],'actual_browser':read(OUT/'analysis/browser_checks.json')['all_checks_passed'],
        'candidate_scope':read(OUT/'analysis/candidate_scope_summary.json')['all_checks_passed'],'synthetic_formula_preflight_not_model_evidence':read(OUT/'analysis/mlp_formula_preflight.json')['all_checks_passed']};assert all(checks.values())
    summary={**a['summary'],'publication':pub,'cleanup':{k:clean[k] for k in ('deleted_files','deleted_bytes','recoverability')},'next_same_goal':True}
    finish(2669,'对称协议全坐标、中层候选与多token逐参数完整交付',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '完整校准—冻结—新材料—原坐标确认—多token分数—真实参数—14B复验—客户端和清理证据链。先积累可复用的语言条件响应，再审查数值算法，阴性不关闭分布式路线。',
        r'g=W_gx,\quad u=W_ux,\quad a=SiLU(g)\odot u,\quad r=W_da;\quad \partial L/\partial W_{g,jk}=\sum_t\bar a_{tj}u_{tj}SiLU^{\prime}(g_{tj})x_{tk};\quad G^{all}=G^{content}+G^{format}+G^{EOS}.',
        '2048校准、8192新实体短码条件及全原坐标、256多token自然/FP32序列、2176数值条件含2048真实单参数改动、1024非量化14B条件。10新面板；64短码、64多token、32大模型展示实例；保留224原包。另有1792候选条件范围逐值核对和16个CPU/float64合成门控MLP公式检查；后者不是LLM实验或语言规律证据，真实MLP逐参数确认属于下一批。',
        '重要的主线变化是末层候选未跨新协议，而中层仍有通过者：H24[2355]、H27[1217]；MLP23[6197]、26[3594]、27[3221]、28[5952,8513]，均为零起始层/坐标。该旧掩码确认仅检查q0/m0在双新实体集合全部32族语言/探问组的方向与幅度条件，并不代表反问或反向映射也通过。应沿具体单元追踪实际gate、up和down全输入/输出坐标，不把标准链式法则称为新语言理论。',
        '同token长度并不保证同任务难度，演示带来支架/近因干扰；实体和协议一起变化不能单因归因。多token只是4token规范短格式，不是长句自主内容生成。14B每语言2实体对且没有正文形式/顺序全交叉。尚未破解通用编码机制。',
        '本大阶段全部完成，下一目标仍相同：冻结本批仍通过的中层坐标/MLP单元，围绕真实SiLU(g)*u和全部输入/输出权重逐坐标追踪跨语言操作的条件复用与分化，同时保留全场对照；不要只重复标签反转或必要性门。2670—2676完整方案已保存到analysis/next_campaign.json，调度遵循用户当前暂停/启用设置；尚未执行的下一批不记为完成。')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['audit','prepare_examples','publish','cleanup','finalize']);a=ap.parse_args();globals()[a.action]()
