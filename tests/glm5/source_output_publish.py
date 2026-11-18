"""Audit and publish the real branch-gated source/output experiment locally."""
import argparse,base64,hashlib,io,json,tarfile,time
import numpy as np
import source_output_core as S
from source_output_core import C
from source_output_runner import check_analysis,admitted
from source_output_causal import selected
from deploy.research_library import Library,canonical

def audit():
    p=C.checked();check_analysis();start=time.monotonic();native=[];prefix_checks=0;maxq=0.;loaders={};weight_identity=None
    for stage in ['behavior_calibration','native_calibration','behavior_confirm','native_confirm','causal']:
        path=C.OUT/f'loader-{stage}.json'
        if not path.exists():continue
        d=C.read(path);assert d['model_snapshot']==C.SNAP and d['plan_sha256']==C.sha(C.OUT/'plan.json')
        if weight_identity is None:weight_identity=d['checks']
        else:assert weight_identity==d['checks']
        loaders[stage]={'sha256':C.sha(path),'parameter_tensors':len(d['checks'])}
    assert 'behavior_calibration' in loaders and 'behavior_confirm' in loaders
    for split,nf in p['families'].items():
        rows=C.read(C.OUT/(split+'-materials.json'))['rows'];scores=[];active=admitted(split)
        for f in range(nf):
            b=C.read(C.OUT/f'behavior-{split}-{f:04d}.json');assert b['plan_sha256']==C.sha(C.OUT/'plan.json');scores+=b['scores']
            if not active:continue
            meta=C.read(C.OUT/f'native-{split}-{f:04d}.json');assert meta['plan_sha256']==C.sha(C.OUT/'plan.json')
            assert meta['analysis_plan_sha256']==C.sha(C.OUT/'analysis-plan.json') and meta['prefix_control_bitwise'] and meta['behavior_scores_bitwise']
            assert len(meta['records'])==len(active)*4;prefixes={}
            for item in meta['records']:
                row=rows[f*24+item['variant']];assert row['id']==item['id'] and S.branch(row) in active
                path=C.OUT/item['path'];assert C.sha(path)==item['sha256'];native.append({**item,'bytes':path.stat().st_size})
                with np.load(path) as z:
                    assert z['hidden'].shape==(37,2,2560) and z['units'].shape==z['gate'].shape==z['value'].shape==(2,9728)
                    assert z['head_context'].shape==(2,32,128) and z['selected_source_context'].shape==(2,2,4,128)
                    np.testing.assert_allclose(z['source_mass'].sum(-1),1,atol=2e-6)
                    assert np.count_nonzero(z['attention_prob'][0,:,row['positions'][0]+1:])==0
                    key=(row['task'],row['status_position']);h=z['hidden'][:,0]
                    if key in prefixes:np.testing.assert_array_equal(h,prefixes[key]);prefix_checks+=1
                    else:prefixes[key]=h.copy()
                    maxq=max(maxq,float(z['qkv_relative_error'].max()))
            if (f+1)%24==0:C.log('audit_'+split,f+1,nf,start)
        assert S.behavior(rows,scores,p)['branches']==C.read(C.OUT/f'behavior-{split}.json')['branches']
        assert C.read(C.OUT/f'native-{split}-complete.json')['n']==nf*len(active)*4
    rows=selected(C.read(C.OUT/'confirm-materials.json')['rows'],admitted('confirm'));full=0;maxkl=0.;invalid=0
    for i,row in enumerate(rows):
        item=C.read(C.OUT/f'causal-{i:04d}.json');assert item['sample_id']==row['id'] and item['restoration_bitwise']
        assert item['analysis_plan_sha256']==C.sha(C.OUT/'analysis-plan.json')
        original=C.read(C.OUT/f"behavior-confirm-{row['family']:04d}.json")['scores'][row['variant']]
        for key,value in original.items():
            if key!='id':assert item['baseline'][key]==value
        invalid+=sum(not c['valid'] for e in item['edits'] for c in e['controls'])
        if i%16==0:
            path=C.OUT/f'causal-logits-{i:04d}.npz';assert C.sha(path)==item['audit_logits_sha256']
            with np.load(path) as z:
                base=C.unbits(z['base']).astype(float);lb=base-base.max();lb-=np.log(np.exp(lb).sum())
                for name in ['base',*[c['id'] for c in p['fixed_candidates']],'joint']:
                    raw=z[name];logits=C.unbits(raw).astype(float);sc=item['baseline'] if name=='base' else item['joint'] if name=='joint' else next(e['edited'] for e in item['edits'] if e['candidate']==name)
                    assert hashlib.sha256(raw.tobytes()).hexdigest()==sc['logits_sha256']
                    assert logits[p['answer_ids'][row['alphabet']]].tolist()==sc['pair_logits'] and int(logits.argmax())==sc['argmax']
                    lp=logits-logits.max();lp-=np.log(np.exp(lp).sum());kl=float((np.exp(lp)*(lp-lb)).sum());maxkl=max(maxkl,abs(kl-sc['KL_edit_base']));full+=1
    assert maxq<.02 and maxkl<2e-5
    old=C.read(S.OLD/'plan.json')
    for name,digest in old['dependencies'].items():assert C.sha(C.ROOT/name)==digest
    for name,digest in old['materials'].items():assert C.sha(S.OLD/name)==digest
    assert C.sha(S.OLD/'selection.json')==p['old_selection_sha256']
    assert C.sha(S.OLD/'report.json')==C.read(S.OLD/'audit.json')['report_sha256']
    C.save('audit.json',{'schema':'source_output_audit.v1','passed':True,'native_inputs':len(native),'native_manifest':native,'prefix_array_comparisons':prefix_checks,
        'maximum_QKV_relative_error':maxq if native else None,'weight_loaders':loaders,'causal_inputs':len(rows),'full_vocabulary_arrays_checked':full,
        'maximum_CPU64_KL_error':maxkl if full else None,'invalid_native_controls_preserved':invalid,'all_baselines_match_behavior':True if rows else None,'all_restorations_bitwise':True if rows else None,
        'old_frozen_sources_materials_selection_report_preserved':True,'no_family_filtering':True,'formal_promotion':False,
        'scope':'Same-node technical audit, not independent scientific replication'})

def publish():
    p=C.checked();check_analysis();auditdoc=C.read(C.OUT/'audit.json');assert auditdoc['passed'];active=admitted('confirm')
    obs=C.read(C.OUT/'observations.json');effects=C.read(C.OUT/'effects.json');behavior={s:C.read(C.OUT/f'behavior-{s}.json') for s in ['calibration','confirm']}
    libs=[Library(C.ROOT/x) for x in ['distributed_data/research_library','tests/glm5/result/mlp_tail_acceptance/center/research_library']];objects={}
    def put(name,body):
        d={**body,'run_id':C.RUN,'model_snapshot':C.SNAP,'kind':'real','provenance':p['provenance']};raw=canonical(d)
        digests={lib.ingest(d,C.RUN+'/'+name,kind='local_report')['sha256'] for lib in libs};assert len(digests)==1;sha=digests.pop();objects[sha]=raw;return sha
    artifacts={name:put(file,C.read(C.OUT/file)) for name,file in [('原冻结方案','plan.json'),('资源修订','resource-amendment.json'),('冻结分析和作用规则','analysis-plan.json'),
        ('留出前冻结的预测器','predictor-freeze.json'),('完整校准行为','behavior-calibration.json'),('完整留出行为','behavior-confirm.json'),('整支准入与停止','admission-confirm.json'),
        ('原模型作用与负对照','effects.json'),('留出预测与响应','observations.json'),('同节点技术复算','audit.json')]}
    for name,file in [('二值副作用的预先功效边界','secondary-power-freeze.json'),('固定单映射的二值副作用','secondary-binary-results.json'),
        ('行为失败后的资源缩减','resource-amendment-behavior-only.json'),('整支停止登记的执行修复','stop-registration-recovery.json'),('首词格式和未映射编号诊断','failure-diagnostics.json')]:
        artifacts[name]=put(file,C.read(C.OUT/file))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    failures=[];parts=[]
    for split in ['calibration','confirm']:
        rows=C.read(C.OUT/(split+'-materials.json'))['rows'];scores=[]
        for f in range(len(rows)//24):scores+=C.read(C.OUT/f'behavior-{split}-{f:04d}.json')['scores']
        for row,score in zip(rows,scores):
            if not score['strict_correct'] or not score['first_top1']:
                labels=['1','2'] if row['alphabet']=='digits' else ['A','B'];failures.append({'sample':row,'score':score,'answer':labels[0 if row['output_sign']==1 else 1],
                    'predicted_text':tokenizer.decode([score['argmax']]),'failure_type':'candidate_choice' if not score['strict_correct'] else 'first_token_format_or_other_answer'})
        for i in range(0,len(rows),192):parts.append(put(f'inputs-{split}-{i}',{'schema':'source_output_inputs.v1','split':split,'rows':rows[i:i+192],'scores':scores[i:i+192]}))
    artifacts['全部输入与逐条行为']=put('input-index',{'schema':'source_output_index.v1','parts':parts})
    n=effects['causal_samples'];parts=[]
    for i in range(0,n,24):parts.append(put(f'causal-{i}',{'schema':'source_output_causal_records.v1','rows':[C.read(C.OUT/f'causal-{j:04d}.json') for j in range(i,min(n,i+24))]}))
    artifacts['全部干预与无效对照登记']=put('causal-index',{'schema':'source_output_index.v1','parts':parts})
    coordinates={};cal=np.load(C.OUT/'calibration-coordinates.npz');conf=np.load(C.OUT/'confirm-coordinates.npz')
    def vector(branch,field,index,value):
        value=np.asarray(value,dtype='<f4');return put(f'{branch}-{field}-{index}',{'schema':'source_output_coordinates.v1','branch':branch,'field':field,'index':index,
            'scope':'All samples of the admitted complete branch, original native order; raw signed conditional means',
            'array':{'dtype':'<f4','encoding':'base64','shape':list(value.shape),'data':base64.b64encode(value.tobytes()).decode('ascii')}})
    for branch in active:
        key=branch.replace(':','_')+'__';coordinates[branch]={'hidden':{},'units':{},'heads':{}}
        for layer in range(37):
            value=np.stack([np.stack([z[key+'hidden_'+factor][layer] for factor in ['source','output']]) for z in [cal,conf]])
            coordinates[branch]['hidden'][str(layer)]=vector(branch,'hidden',layer,value)
        value=np.stack([np.stack([z[key+'unit_'+factor] for factor in ['source','output']]) for z in [cal,conf]])
        coordinates[branch]['units']['26']=vector(branch,'units',26,value)
        value=np.stack([z[key+'head_route'][None] for z in [cal,conf]]);coordinates[branch]['heads']['29']=vector(branch,'heads',29,value)
    summary=[f"完整行为分支通过 {len(active)}/6；内部分析只沿用通过的整支，不剔除其中的个别失败样本。"]
    if obs['branches']:
        wins=sum(o['native_over_parsed']<1 for o in obs['branches']);summary.append(f"固定原生组读出在 {wins}/{len(active)} 个分支的均方误差低于文本规则参照；是否稳定改善还须查看句式区间。")
        supported=sum(e.get('supported',False) for e in effects['results']);summary.append(f"{supported} 项单组作用比较达到预设门；无效的幅度对照、未通过项目和联合效应均单独登记。")
    else:summary.append('当前只能保留行为失败与任务分解；没有合格分支支持继续作内部机制结论。')
    limitations=['本轮是人工模板与两个候选答案的首词行为，未检验自然自由生成、开放实体或通用语言能力。',
        '每题只有两条记录，日期分别来自2022年与2025年；不能据此确认任意日期排序、多次更新或复杂撤销规则。',
        '独立性指相对候选选择的新实体、新句式和新映射；不是独立节点或独立模型复现。',
        '完整分支的行为准入限定了适用范围；失败分支缺失的内部数据不等于机制不存在。',
        '行为和全场在同一家族覆盖两套符号；作用阶段为控制成本把两套符号分配给不同家族，未直接完成跨符号的家族内因果等效检验。',
        '各分支分别拟合线性读出；共享的是原生单元索引，并未证明同一个固定读出器能跨任务和符号直接迁移。',
        's×状态和s×m×状态是平衡设计的条件统计量；可分离读出不能证明原模型采用可分离模块。',
        '记录来源的注意力质量还受文本跨度长度与位置影响；来源质量差和局部V/O贡献不能单独证明主动排除过时信息。',
        '只检验前一轮的B26单元组和B29头组，不能将失败推广为其他层或所有原生特征都无作用。',
        '原生8变量预测器读入真实激活；文本规则参照已知任务类型，两者都不构成从文本独立预测完整下一词分布的图谱。',
        '12个留出句式簇限制了不确定性的外推；预注册功效针对0.2 nats均值差，不代表能检测稀少的答案翻转。',
        'α=.05/6按每个候选的六个任务分支校正；未保证两个候选和全部辅助指标合在一起的全局错误率。范围内通过不是全局多重检验后的正式确认。',
        '随机方向和缩放原生方向都是局部反事实；尺度上限触发时保留失败，不拿有效子集补过门。',
        'B26在B29之前，联合效应不能证明B29到B26的中介链；没有跨层完整计算闭合。',
        '未训练SAE或transcoder，也未做跨模型比较；本轮不能证明需要新的数学理论。']
    report={'schema':'source_output_report.v1','plan_sha256':C.sha(C.OUT/'plan.json'),'analysis_plan_sha256':C.sha(C.OUT/'analysis-plan.json'),
        'unique_inputs':p['unique_inputs'],'active_branches':active,'behavior':behavior,'summary':summary,'failures':failures,'candidates':p['fixed_candidates'],
        'observations':obs['branches'],'effects':effects['results'],'specificity':effects['specificity'],'causal_samples':n,
        'coordinate_artifacts':coordinates,'artifacts':artifacts,'limitations':limitations,'formal_promotion':False,
        'failure_diagnostics':C.read(C.OUT/'failure-diagnostics.json'),
        'audit_summary':{k:v for k,v in auditdoc.items() if k!='native_manifest'}}
    C.save('report.json',report);ref=put('report',report);cards=[];edges=[]
    for c in p['fixed_candidates'] if active else []:
        cid=C.RUN+':'+c['id'];cards.append({'id':cid,'run_id':C.RUN,'layer':c['block'],'boundary':'attention.o_proj.input' if c['kind']=='heads' else 'mlp.down_proj.input',
            'label':c['id']+' · 记录来源/输出符号的冻结候选检验','index':{'space':'attention' if c['kind']=='heads' else 'mlp','indices':c['indices'],'group':True,'anchor_only':False},
            'method':'fixed previous candidates; crossed source rule/output codebook; whole-branch held-out gates; matched native direction controls',
            'evidence':{'level':'observation','artifact_sha256':ref},'scope':{'active_branches':active,'formal_promotion':False,'causal_chain_closed':False},
            'counterexamples':[{'artifact_sha256':ref,'field':'failures'},{'artifact_sha256':ref,'field':'effects'}]})
        edges.append({'id':cid+':mapping','source':'language:source_selection_vs_output_code','target':cid,'relation':'通过整支行为门后的响应与作用检验；非完整机制',
            'run_id':C.RUN,'evidence':{'level':'observation','artifact_sha256':ref}})
    graph=put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges})
    binding=put('binding',{'schema':'research_space_binding.v1','report_sha256':ref,'graph_sha256':graph,'traces':[],
        'context':{'study_type':'source_output','native_bank':True,'model_dimensions':{'n_layers':36,'d_model':2560,'d_ff':9728,'n_heads':32}}})
    work=put('workpackage',{'schema':'research_workpackage.v1','title':'Qwen3-4B · 记录来源与输出符号的独立检验',
        'stages':{'behavior':{'status':'done','passed':len(active)==6},'prediction':{'status':'done' if active else 'not_tested','active_branches':active},'native_effect':{'status':'done' if n else 'not_tested','samples':n},
            'registration_validated':{'status':'done','report_sha256':ref,'graph_sha256':graph}}})
    reg={'run_id':C.RUN,'model_snapshot':C.SNAP,'report_sha256':ref,'graph_sha256':graph,'binding_sha256':binding,'workpackage_sha256':work}
    bundle=C.OUT/'ai2050-source-output-v1.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=1) as tar:
        def add(name,raw):
            item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
        add('MANIFEST.json',canonical({'schema':'ai2050_research_upload.v1','objects':[{'sha256':h} for h in objects],'studies':[reg]}))
        for sha,raw in objects.items():add(f'objects/{sha}.json',raw)
    C.save('registration.json',{**reg,'bundle':str(bundle),'bundle_sha256':C.sha(bundle),'objects':len(objects),'cards':len(cards),'edges':len(edges),
        'coordinate_artifacts':sum(sum(len(v) for v in d.values()) for d in coordinates.values()),'local_libraries':[str(x.root) for x in libs],'remote_receipt':None})
    print(json.dumps({'report_sha256':ref,'active_branches':active,'objects':len(objects),'failures':len(failures),'causal_samples':n}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','publish']);globals()[parser.parse_args().stage]()
