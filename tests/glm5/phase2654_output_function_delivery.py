"""Complete output-function campaign: bounded claims, exact client and manifested retention."""
import argparse,os,shutil
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2628_native_atlas_delivery import ASSET,panel,row
from phase2650_output_function_adjoints import MATERIAL,BF,INITIAL,CONFIRM,LAYERS

OUT=RESULT/'phase2654_output_function_delivery'
MAPS=RESULT/'phase2651_output_function_maps'
SCALAR=RESULT/'phase2653_output_function_scalar_validation'
MODES=('name','cloze','truth_a','truth_b')


def audit():
    cases=read(MATERIAL/'material/cases.json');initial=read(INITIAL/'analysis/records.json');confirmation=read(CONFIRM/'analysis/records.json')
    interface=read(BF/'analysis/output_interface_audit.json');envelopes=read(CONFIRM/'analysis/envelope_confirmation.json')
    scalar=read(SCALAR/'analysis/final.json');storage=read(CONFIRM/'analysis/lossless_storage_transition.json');reuse=read(OUT/'analysis/truth_query_coordinate_reuse.json')
    sources={label:read(p/'analysis/cross_function_source_traces.json') for label,p in [('initial',MAPS),('confirmation',CONFIRM)]}
    phase_checks={str(p):read(next(RESULT.glob(f'phase{p}_*/analysis/final.json')))['all_checks_passed'] for p in range(2648,2654)}
    identity=[r for r in initial+confirmation if r['native_common_identity']!='different']
    checks={'8192_material_cases':len(cases)==8192,'4096_field_conditions':len(initial)+len(confirmation)==4096,
        'disjoint_initial_confirmation':not({r['case_index'] for r in initial}&{r['case_index'] for r in confirmation}),
        'all64_published_fulltoken_examples':sum(r['published'] for r in cases)==64,
        'all6144_crossfunction_comparisons':sum(map(len,sources.values()))==6144,
        'output_interface_covers8192':sum(r['n'] for r in interface['groups'].values())==8192,
        'all_readout_identities_exact':bool(identity) and all(r['identity_max_absolute_error']==0 for r in identity),
        'all_embeddings_exact':all(r['embedding_exact'] for r in initial+confirmation),
        'frozen_envelopes_complete':all(envelopes['checks'].values()),'all_six_prior_phases':all(phase_checks.values()),
        'lossless_arrays_bitwise_identical':storage['all_arrays_bitwise_identical'],'all16_truth_query_reuse_groups':len(reuse['groups'])==16}
    summary={'behavior_output_interface':interface['groups'],'coordinate_envelopes':envelopes['summary'],
        'source_traces':{label:read(p/'analysis/final.json')['summary'] if label=='initial' else read(p/'analysis/final.json')['summary']['maps'] for label,p in [('initial',MAPS),('confirmation',CONFIRM)]},
        'numeric':scalar['summary'],'readout_identity_cases':len(identity),'post_confirmation_truth_query_reuse':reuse,
        'storage_container_only':{'files':len(storage['files']),'old_bytes':storage['old_bytes'],'new_bytes':storage['new_bytes'],'saved_bytes':storage['old_bytes']-storage['new_bytes'],'arrays_bitwise_identical':True}}
    assert all(checks.values())
    save(OUT/'analysis/science_audit.json',{'summary':summary,'checks':checks,'all_checks_passed':True,
        'claim_boundary':['Natural BF16 behavior is not replaced by calibrated FP32 computation.',
        'Cloze has an experimenter-supplied sentence prefix; leading correct entity does not certify the whole continuation.',
        'Native first-token pairs often contrast formatting of one answer; fixed canonical cloze rows are an external control, not complete answer probability.',
        'Envelope comparisons use every physical coordinate; repetition across templates is not independent semantic population evidence.',
        'Source state belongs to the causal past; future-task changes in adjoints are future readout sensitivity, not retroactive rewriting.',
        'All-token scalar differentiation is a verified calculation algorithm, not discovery of a universal semantic gear.']})


def publish():
    assert read(OUT/'analysis/science_audit.json')['all_checks_passed']
    records={r['case_index']:r for r in read(INITIAL/'analysis/records.json')}
    behavior={r['case_index']:r for r in map(json.loads,(BF/'behavior/greedy.jsonl').read_text(encoding='utf-8').splitlines())}
    published=[{**r,**records[r['case_index']],'generated':behavior[r['case_index']]['generated'],'content_correct':behavior[r['case_index']]['content_correct']}
        for r in read(MATERIAL/'material/cases.json') if r['published']]
    assert len(published)==64;save(OUT/'material/published_cases.json',published)
    rr={key:[] for key in ('bf_hidden','bf_mlp','source_V','source_common_adjoint','V_parameter_row')}
    for r in published:
        ci=r['case_index'];pos=r['positions'][0]
        with np.load(BF/f'field/case_{ci:04d}.npz',allow_pickle=False) as b,np.load(INITIAL/f'field/case_{ci:04d}.npz',allow_pickle=False) as f:
            for checkpoint in (0,18,36):
                rr['bf_hidden'].append(row(f'{r["case_id"]}/BF16/sourceA/H{checkpoint}',b['hidden_fulltoken'][checkpoint,pos],2654,'native_hidden_coordinate',checkpoint))
                rr['bf_hidden'].append(row(f'{r["case_id"]}/BF16/boundary/H{checkpoint}',b['hidden_fulltoken'][checkpoint,-1],2654,'native_hidden_coordinate',checkpoint))
            rr['bf_mlp'].append(row(f'{r["case_id"]}/BF16/MLP35',b['mlp_boundary'][35],2654,'actual_mlp_neuron',35))
            for l in (0,35):
                rr['source_V'].append(row(f'{r["case_id"]}/FP32/L{l}/sourceA/V',f[f'L{l}_v_value'][pos],2654,'actual_V_output_coordinate',l))
                rr['source_common_adjoint'].append(row(f'{r["case_id"]}/fixed_readout/L{l}/sourceA/dm_dV',f[f'common__L{l}_v_g'][pos],2654,'output_conditioned_source_adjoint',l))
                j=57 if l==0 else 137;x=f[f'L{l}_v_x'].astype('float64');g=f[f'common__L{l}_v_g'][:,j].astype('float64')
                rr['V_parameter_row'].append(row(f'{r["case_id"]}/fixed_readout/L{l}/Vrow{j}/all_k',(x*g[:,None]).sum(0),2654,'alltoken_actual_scalar_weight_derivative',l))
    names={'bf_hidden':('Output function: natural embeddings and full-coordinate hidden states',2560),
        'bf_mlp':('Output function: all final actual MLP neurons',9728),'source_V':('Same causal source: all V coordinates',1024),
        'source_common_adjoint':('Future output condition: source sensitivity at every V coordinate',1024),
        'V_parameter_row':('Fixed historical V row: all scalar input weights, all token terms',2560)}
    semantics='Native coordinates, no Top-K or dimension reduction. Displayed examples/checkpoints do not replace all-layer primary maps. Fixed output IDs and native formatting competition are different objectives; no semantic closure.'
    new=[panel('phase2654_'+k,names[k][0],names[k][1],v,semantics) for k,v in rr.items()]
    with np.load(CONFIRM/'maps/initial_confirmation_envelopes.npz',allow_pickle=False) as z,np.load(OUT/'maps/truth_query_fullcoordinate_reuse.npz',allow_pickle=False) as reuse:
        for metric,dim,layers in [('h',2560,(0,6,18,36)),('mlp',9728,(0,5,17,35))]:
            values=[]
            for mode in MODES:
                for precision,prefix in [('fp32',''),('bf16','bf_')]:
                    for l in layers:
                        for kind in ('initial','both','both_signed'):
                            a=z[f'{mode}__{prefix}{metric}__{kind}'];v=a[l,2] if metric=='h' else a[l]
                            values.append(row(f'{mode}/{precision}/{metric}{l}/{kind}/groups0..16',v,2654,'allcoordinate_frozen_envelope',l))
            for prefix in ('','bf_'):
                for l in layers:
                    for kind in ('both_modes_stable','truth_oriented_opposite','semantic_target_same'):
                        a=reuse[f'{prefix}{metric}__{kind}'];v=a[l,2] if metric=='h' else a[l]
                        values.append(row(f'truth_query_reuse/{prefix}{metric}{l}/{kind}/groups0..16',v,2654,'postconfirmation_fullcoordinate_sign_accounting',l))
            new.append(panel('phase2654_envelope_'+metric,'Four output functions: all-coordinate frozen confirmation '+metric,dim,values,
                'Count of16 family/language groups. targetRMS>orderRMS and formRMS; both_signed means within-group initial/heldout sign agreement, not a universal cross-family direction. truth_query_reuse is a post-confirmation descriptive extension, not a third independent test. Every column retained.'))
    payload=read(ASSET);old=sha(ASSET);prior=[p['key'] for p in payload['models'] if not p['key'].startswith('phase2654_')]
    payload['models']=[p for p in payload['models'] if not p['key'].startswith('phase2654_')]+new;payload['phase']=2654
    payload['claim_boundary']=semantics;payload['summary']['model_rows']={p['key']:len(p['rows']) for p in payload['models']};payload['summary']['total_rows']=sum(payload['summary']['model_rows'].values())
    temp=ASSET.with_suffix('.phase2654.tmp');assert not temp.exists()
    temp.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8');os.replace(temp,ASSET)
    prior_proof=read(OUT/'analysis/publication.json') if (OUT/'analysis/publication.json').exists() else None
    proof={'previous_asset_sha256':prior_proof['previous_asset_sha256'] if prior_proof else old,'replaced_revision_sha256':old,'asset_sha256':sha(ASSET),'asset_bytes':ASSET.stat().st_size,'prior_panel_keys':prior,
        'panels':[{'key':p['key'],'width':p['coordinate_count'],'rows':len(p['rows'])} for p in new],'published_case_indices':[r['case_index'] for r in published]}
    save(OUT/'analysis/publication.json',proof);print(json.dumps({'panels':len(new),'published_cases':len(published),'bytes':proof['asset_bytes']}),flush=True)


def cleanup():
    assert not (OUT/'analysis/cleanup_completed.json').exists()
    assert read(OUT/'analysis/science_audit.json')['all_checks_passed'] and read(OUT/'analysis/delivery_checks.json')['all_checks_passed']
    assert read(SCALAR/'analysis/final.json')['all_checks_passed']
    pub=set(read(OUT/'analysis/publication.json')['published_case_indices']);transition={str(Path(r['path']).resolve()):r for r in read(CONFIRM/'analysis/lossless_storage_transition.json')['files']}
    kept=[];targets=[]
    for source in (BF,INITIAL,CONFIRM):
        for r in read(source/'analysis/raw_manifest.json'):
            p=Path(r['path']).resolve();assert p.parent==(source/'field').resolve() and p.is_relative_to(RESULT.resolve()) and re.fullmatch(r'case_\d{4}\.npz',p.name)
            assert p.is_file();digest=sha(p);size=p.stat().st_size;t=transition.get(str(p))
            if t:assert t['old_bytes']==r['bytes'] and t['new_bytes']==size and t['new_file_sha256']==digest and t['bitwise_identical']
            else:assert size==r['bytes']
            entry={'path':str(p),'bytes':size,'case_index':r['case_index'],'sha256':digest}
            (kept if r['case_index'] in pub and source in (BF,INITIAL) else targets).append(entry)
    assert len(kept)==128 and len(targets)==8064 and len({r['path'] for r in kept+targets})==8192
    report={'targets':targets,'kept':kept,'deleted_files':len(targets),'deleted_bytes':sum(r['bytes'] for r in targets),'before_free_bytes':shutil.disk_usage(OUT).free,
        'recoverability':'Directly removed, not in Recycle Bin. Recompute into a NEW directory with preserved2648 material/token IDs,2649 BF16 first decisions,2650 capture engine,model and runtime protocols; completed Phase mains must not be rerun over MEMO. Preserve all derived maps, behavior, manifest/hash chains, code, models and128 published raw packs.'}
    save(OUT/'analysis/cleanup_plan.json',report)
    for r in targets:
        p=Path(r['path']);assert p.stat().st_size==r['bytes'];p.unlink()
    report.update(after_free_bytes=shutil.disk_usage(OUT).free,all_deleted=all(not Path(r['path']).exists() for r in targets),all_published_retained=all(Path(r['path']).is_file() and sha(r['path'])==r['sha256'] for r in kept))
    save(OUT/'analysis/cleanup_completed.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('targets','kept')}),flush=True)


def finalize():
    a=read(OUT/'analysis/science_audit.json');clean=read(OUT/'analysis/cleanup_completed.json');pub=read(OUT/'analysis/publication.json')
    checks={**a['checks'],'client_api_and_build':read(OUT/'analysis/delivery_checks.json')['all_checks_passed'],'post_cleanup_api':read(OUT/'analysis/post_cleanup_checks.json')['all_checks_passed'],
        'all8064_manifested_unshown_deleted':clean['all_deleted'],'all128_published_raw_retained':clean['all_published_retained']}
    assert all(checks.values())
    summary={**a['summary'],'client':pub,'cleanup':{k:clean[k] for k in ('deleted_files','deleted_bytes','recoverability')},'next_same_goal':True,
        'next_scope':'Freeze the truth-task signed native-coordinate candidates and test a third entity/material set. Cross factual relation, queried entity and reversed output labels to separate truth from answer-word preparation. Add semantic decision-time atlas: distinguish output-format prefixes from actual answer alternatives; naturally generated token-by-token fields and exact conditional answer-string scores with frozen multi-token/tokenization controls. Confirm strong new results on heldout materials and sequential nonquantized second model, not more common-head cosine alone.'}
    finish(2654,'输出功能全坐标研究完整交付、接口纠错与下一决策时刻路线',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '先保留8192自然BF16行为，再以初始和独立2048条件各自的全坐标场比较输出功能，最后检验真实单权重而非donor搬运。源状态与未来任务敏感度、自然输出与外部固定读出分别记账。无损NPZ转换逐数组形状、dtype、字节哈希完全一致，只有容器编码变化。',
        r'G^r_{l,jk}=\sum_t\frac{\partial m_r}{\partial V_{l,t,j}}X_{l,t,k};\quad I^s_{g,l,j}=\mathbf1[R^{s,target}_{g,l,j}>\max(R^{s,order}_{g,l,j},R^{s,form}_{g,l,j})];\quad C_{l,j}=\sum_g I^{initial}_{g,l,j}I^{heldout}_{g,l,j}.',
        '2648—2654全部合同：八族×双语×16实体对×双句式/目标/顺序×四输出模式共8192行为；4096内部场条件；初始/确认各64完整坐标组及3072跨功能源比较；1088单参数数值条件和256等形状数值前向；7新面板、64逐参数实例、128保留原包。主分析不删低值坐标，不按Top-K找核心。',
        '新增拼图是把相同事实的源状态、输出功能导致的边界响应以及同一真实参数的输出敏感度分开测量。冻结独立确认中FP32末层H，name有881个幅度候选保持全部16组，仅86个同时逐组保持符号；truth_a有105个幅度候选全部保持符号，truth_b有38个全部保持符号；cloze只有3个幅度候选、0同号。BF16对应name888/86、truth_a106/106、truth_b29/29、cloze2/0。方向稳定依赖功能的迹象值得继续，但固定真假答案、问题措辞、边界位置均可能解释差异，不能声称因果已分离。单参数各层/读出汇总相对L1误差0.4491%—2.1333%，错误末token近似99.03%—100.19%，不是完整答案准确率。源状态原有最大0.001556数值差异，在64个等形状屏蔽填充条件（38个独特对）全部消失到位级一致，支持本批差异为计算形状数值效应，不能解释为未来问题改变过去状态。模型原生首位竞争中存在大量同答案格式变体，尤其英文cloze原生首位全部不在独立编码的人名canonical二选一中，故固定读出只是诊断控制。中文cloze完整目标判定42/1024，但首个名字正确929/1024，其中655达到16token截断上限；前者不能推出语义崩解，后者也不能证明后续整句正确。公式精确预测只建立局部真实参数算法，不建立独立语义齿轮。',
        '单模型Qwen4、模板与实体复用、仅4实体对/语言每个内部集合；部分词表也随集合变化。追加两类真假提问的跨模式同坐标复用：FP32末层H20个、MLP37个同时在两模式两集合全部16组稳定，全部在两提问之间反号；BF16分别11个和32个，同样全部反号。这是看完确认集后对已有数据的描述性整理，不是第三次独立验证，不得将20/37称为纯真值神经元；回答词准备仍可解释。cloze由实验者给出句首，不是自主规划全文。原生首token和固定两行输出均非完整答案概率。显示选定实例/层但主分析全层全坐标；API精确值与前端构建验证不冒充浏览器视觉验收。零必要性、非零梯度、幅度包络及精确标量预测都不能单独定义语言机制。无限组合是研究动机，不是有限上下文模型已经证明的字面无限能力。',
        '本大阶段全部任务完成。下一阶段目标相同，继续既有自动续研：先冻结truth任务的同号原坐标候选，第三批扩大材料，交叉事实关系、被询问实体和反向答案标签，检验稳定性属于真值还是答案词准备；同时转向自然生成中实际答案分叉的决策时刻，把格式token、语义答案字符串及完整条件序列评分拆开，观察同一真实源坐标如何在不同输出时刻被使用。冻结多token对照并扩大材料，在新证据强时串行复验第二个非量化模型。先积累可复用条件规律，不再把相同输出头的末层相似度当作突破，也不因单零件必要性失败放弃分布式路线。')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','publish','cleanup','finalize']);a=p.parse_args();globals()[a.action]()
