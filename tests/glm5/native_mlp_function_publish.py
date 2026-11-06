"""Archive the complete real study, including falsifications and native 3D bindings."""
import sys,json,collections,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from shared.dist_protocol.mechanism_trace import TraceStore,file_sha,digest
from shared.dist_protocol.research_registry import Registry
from deploy.research_library import Library

def main():
    lib=Library(P.ROOT/'distributed_data/research_library');reg=Registry(lib);store=TraceStore()
    obs=P.read(P.OUT/'observations.json');prov=obs['provenance'];snap=obs['model_snapshot'];freeze=P.read(P.OUT/'execution-freeze.json')
    assert all(file_sha(P.ROOT/n)==h for n,h in freeze['dependencies'].items()),'Frozen dependency changed'
    def archive(name,doc,run=P.RUN):
        body={'schema':'research_native_evidence.v1','kind':'real','run_id':run,'model_snapshot':snap,'provenance':prov,**doc}
        path=P.PUBLIC/(name+'.research.json');P.save(path,body)
        return lib.ingest(body,str(path.relative_to(P.ROOT)),kind='local_report')['sha256']
    def chunks(name,rows,key='rows',size=32,run=P.RUN):
        return [archive(f'{name}-{i//size:04d}',{key:rows[i:i+size]},run) for i in range(0,len(rows),size)]
    rev=P.read(P.OUT/'visibility-revisions.v3.json');assert len(rev['corrections'])==4358
    mapping={r['old_trace_id']:r['trace_id'] for r in rev['corrections']}
    def remap(x):
        if isinstance(x,dict):return {k:remap(v) for k,v in x.items()}
        if isinstance(x,list):return [remap(v) for v in x]
        return mapping.get(x,x) if isinstance(x,str) else x
    refs={};c=P.read(P.OUT/'corpus.json');a=remap(P.read(P.OUT/'analysis.json'));coord=P.read(P.OUT/'coordinate-results.json');dose=P.read(P.OUT/'dose-result.json');cap=P.read(P.OUT/'capability-result.json')
    for name in ['execution-freeze','coordinate-selection','coordinate-results','selection','discovery','dose-contract','dose-groups','capability-contract','audit-contract','factorial-results']:
        refs[name]=archive(name+'-archived',{'source_file_sha256':file_sha(P.OUT/(name+'.json')),'data':P.read(P.OUT/(name+'.json'))})
    for p in sorted(P.PUBLIC.glob('*prereg*.research.json'))+sorted(P.PUBLIC.glob('*amendment*.research.json')):
        refs[p.stem]=lib.ingest(P.read(p),str(p.relative_to(P.ROOT)),kind='local_report')['sha256']
    refs['inputs']=chunks('inputs',c['items'],'items')
    refs['observations']=chunks('observations',remap(obs['rows']))
    refs['primary-analysis']=archive('primary-analysis',a)
    refs['dose']=archive('dose-archived',dose);refs['capability']=archive('capability-archived',cap)
    qs=P.read(P.ROOT/'tests/glm5/result/phase2773_w6_v2/capability-inputs.json')
    refs['capability_inputs']=chunks('capability-inputs',qs,'questions',100)
    for name in ['base','native','coordinate']:refs['capability-'+name]=chunks('capability-'+name,P.read(P.OUT/f'capability-{name}.json'),size=100)
    polarities={}
    for name,folder in [('universal',P.OUT),('name_balanced_place',P.OUT/'place-polarity')]:
        r=remap(P.read(folder/'polarity-result.json'));polarities[name]={k:v for k,v in r.items() if k!='rows'}
        refs[name+'_inputs']=chunks(name+'-inputs',P.read(folder/'polarity-corpus.json')['items'],'items')
        refs[name+'_result']=archive(name+'-polarity',r,run=r['run_id'])
        for p in sorted((P.PUBLIC/('place-polarity' if name=='name_balanced_place' else '')).glob('*contract.research.json')):
            refs[name+'-'+p.stem]=lib.ingest(P.read(p),str(p.relative_to(P.ROOT)),kind='local_report')['sha256']
    refs['audit']=archive('audit-replays',remap(P.read(P.OUT/'audit-traces.json')))
    refs['operations']=archive('operation-replays',remap(P.read(P.OUT/'operation-traces.json')))
    profiles=P.read(P.OUT/'profile-index.json');refs['profiles']=archive('profile-index-archived',profiles)
    # Each original immutable manifest and corrected annotation is retained; never replace arrays.
    native_bytes=0;native_files=0;manifests=[];cases=[];byrun=collections.defaultdict(list);started=time.time()
    for i,r in enumerate(rev['corrections']):
        d,folder=store.manifest(r['trace_id']);original=P.read(folder/'manifest.json')
        for f in d['files']:
            p=folder/f['path'];assert p.stat().st_size==f['bytes'] and file_sha(p)==f['sha256'];native_bytes+=f['bytes'];native_files+=1
        manifests.append({'corrected':d,'original':original,'corrected_manifest_file_sha256':file_sha(P.ROOT/r['path'])})
        case={k:r[k] for k in ['trace_id','run_id','sample_id','deep']};case.update(n_tokens=d['n_tokens'],readout_annotations=d['readout_annotations'],layer=24,unit=8443)
        cases.append(case);byrun[d['run_id']].append(case)
        if (i+1)%256==0:print(json.dumps({'verified_native_cases':i+1,'bytes':native_bytes}),flush=True)
    refs['manifests']=chunks('native-manifests',manifests,'manifests',16)
    refs['visibility']=archive('visibility-erratum',{**rev,'corrections':remap(rev['corrections']),'note':'Original trace IDs remain valid; final case index uses v3 annotation IDs. No new inference or independent data.'})
    case_index=archive('case-index',{'schema':'research_native_case_index.v1','cases':cases,'original_to_corrected':mapping})
    refs['case_index']=case_index
    raw=[]
    for folder in [P.OUT,P.OUT/'place-polarity']:
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix in ['.npy','.npz']:raw.append({'path':str(p.relative_to(P.ROOT)),'bytes':p.stat().st_size,'sha256':file_sha(p)})
    native={'cases':len(cases),'deep_cases':sum(r['deep'] for r in cases),'tensor_files':native_files,'trace_bytes':native_bytes,'derived_files':raw,
        'total_bytes':native_bytes+sum(r['bytes'] for r in raw),'all_tensor_hashes_verified':True,'snapshot_unchanged':obs['snapshot_unchanged'],
        'verification_seconds':time.time()-started,'native_local_only':True,'full_module_arrays_at_declared_readouts':True,'full_token_QKV_only_deep':True}
    refs['native']=archive('native-file-verification',native)
    calibrations=[]
    for v in [1,2,3]:
        folder=P.ROOT/f'tests/glm5/result/native_mlp_function_v{v}';pilot=P.read(folder/'pilot-result.json');run=f'NMLP-FUNCTION-20261104-v{v}'
        cc=P.read(folder/'corpus.json');idx={r['id']:r for r in cc['items']};used=[idx[r['id']] for r in pilot['rows']]
        cp={'version':f'v{v}','passed':pilot['passed'],'cells':pilot['cells'],'n_inputs':len(pilot['rows']),'revision':cc.get('revision_reason','initial measurement')}
        calibrations.append(cp);cr=chunks(f'calibration-v{v}-inputs',used,'items',32,run);pr=archive(f'calibration-v{v}-raw',pilot,run)
        examples=[{**t,'url':f'?lens=spatial&view=live&trace={t["trace_id"]}&tblock=24&tunit=8443'} for t in byrun[run]]
        rr=archive(f'calibration-v{v}-report',{'schema':'research_native_function_report.v1','mode':'calibration_only',**cp,'examples':examples,'artifact_refs':{'inputs':cr,'result':pr}},run)
        gr=archive(f'calibration-v{v}-graph',{'schema':'research_graph.v1','feature_cards':[],'graph_edges':[]},run)
        register_binding(archive,store,run,rr,gr,examples,{'study_type':'native_mlp_calibration','parent_run_id':P.RUN},f'校准v{v}：保留全部错误与测量修订')
    eligible={r['family']:r['behavior_gate'] for r in a['findings']}
    coordinates=[{**r,'eligible_supported':eligible[r['family']] and r['heldout_response_supported']} for r in coord['candidates']]
    examples=[]
    sample_by={r['sample_id']:r for r in byrun[P.RUN]}
    for family,l,u in [('location',24,8443),('conditional',28,4371),('universal',32,3952)]:
        for sem in [0,1]:
            t=sample_by[f'{family}:288:{sem}'];examples.append({**t,'layer':l,'unit':u,'label':family+' / lockbox / '+str(sem),'url':f'?lens=spatial&view=live&trace={t["trace_id"]}&tblock={l}&tunit={u}&ttab=evidence'})
    errors=remap(a['counterexamples'])
    for r in errors:r['reason']='固定主假设原生A排序未复现；请结合本格行为门解释'
    for name,r in polarities.items():
        rr=remap(P.read((P.OUT if name=='universal' else P.OUT/'place-polarity')/'polarity-result.json'))
        for row in rr['rows'][:4]:errors.append({**row,'family':name,'reason':'反转问法后事实不变、答案方向改变：固定属性命名的反例','positive_trace':row['trace_id']})
    operations=remap(P.read(P.OUT/'operation-traces.json'))['rows']
    errors.extend({**r,'reason':'能力变化的原始批次几何复采；不是新增独立题目','positive_trace':r['trace_id']} for r in operations if 'question_id' in r)
    unit_cards={};features=[];edges=[]
    allunits=sorted({tuple(x) for x in c['observed_units']}|{(r['layer'],r['unit']) for r in coordinates})
    # observed_units is a list of [block,unit] in the frozen corpus.
    for l,u in allunits:
        fs=[{'scope':name,**f} for name,r in polarities.items() for f in r['findings'] if (f['layer'],f['unit'])==(l,u)]
        primary=[f for f in a['findings'] if tuple(f['unit'])==(l,u)];cs=[r for r in coordinates if (r['layer'],r['unit'])==(l,u)]
        isanswer=any(f['answer_supported'] for f in fs);label='限定任务中的答案方向响应' if isanswer else '自然极点提名：功能解释未确认'
        ex=[{**e,'label':label} for e in examples[:2]]
        card={'label':label,'layer':l,'unit':u,'boundary':'question_end / assistant_boundary separately','findings':fs,'original_hypotheses':primary,'coordinate_results':cs,
            'examples':ex,'counterexamples':[{'reason':'current-query joint ablation below registered effect threshold; not individual necessity test','dose_sha256':refs['dose']},
                {'reason':'six coordinates track answer after question inversion; B24/U8443 only question-end answer-aligned, assistant boundary not disambiguated','polarity_refs':[refs['universal_result'],refs['name_balanced_place_result']]}],
            'natural_discovery':refs['discovery'],'individual_necessity':'not_tested','semantic_monosemanticity':'not_established'}
        sha=archive(f'card-B{l:02d}-U{u:04d}',{'schema':'research_native_function_card.v1','card':card});unit_cards[f'{l}:{u}']=sha
        fid=f'{snap}:B{l}:U{u}:{P.RUN}'
        scope={'trace_id':examples[0]['trace_id'],'position':1023,'readout_roles':['question_end','assistant_boundary'],'semantic_function':label,'card_sha256':sha}
        features.append({'id':fid,'label':f'B{l}/U{u} · {label}','layer':l,'boundary':'mlp.down_proj.input','index':{'space':'mlp','unit':u},'run_id':P.RUN,'method':'discovery-only coordinate nomination; frozen paired holdouts; fresh fact × question polarity; no SAE claim','scope':scope,'evidence':{'level':'heldout' if isanswer else 'observation','artifact_sha256':sha},'counterexamples':card['counterexamples']})
        edges.append({'id':fid+':language','source':'language:conditioned_yes_no_answer' if isanswer else 'language:natural_context_candidate','target':fid,'relation':'scope_limited_answer_order_response' if isanswer else 'function_unconfirmed','run_id':P.RUN,'evidence':{'level':'heldout' if isanswer else 'observation','artifact_sha256':sha}})
    # Native embedding identity baseline and direct arithmetic sources are separately typed.
    from server.mechanism_trace_service import contributions
    t=examples[0];r=contributions(t['trace_id'],24,1023,0,8443,0);assert r['verified_full_snapshot']
    refs['parameter_sources']=archive('actual-parameter-sources',r);head=int(np.argmax(np.abs(r['head_gate_terms'])));hid=f'{snap}:B24:H{head}:{P.RUN}'
    features.append({'id':hid,'label':f'B24/H{head} · 实际Gate算术来源','layer':24,'boundary':'self_attn.o_proj.input','index':{'space':'attention','head':head},'run_id':P.RUN,'method':'all32 source terms from actual parameter slices with actual RMS held fixed','scope':{'trace_id':t['trace_id'],'position':1023},'evidence':{'level':'parameter_source','artifact_sha256':refs['parameter_sources']},'counterexamples':[]})
    edges.append({'id':hid+':gate','source':hid,'target':f'{snap}:B24:U8443:{P.RUN}','relation':'fixed_RMS_direct_arithmetic_source_not_total_causality','run_id':P.RUN,'evidence':{'level':'parameter_source','artifact_sha256':refs['parameter_sources']}})
    ef=f'{snap}:embedding:0:{P.RUN}';emb=archive('embedding-baseline',{'early_prefix_all_hidden_max_abs':0,'final_shared_token_embedding_max_abs':0,'passed':a['negative_control_early_pass'],'causal_scope':'selector token identity differs; later shared token embedding does not acquire context in fixed weights'})
    features.append({'id':ef,'label':'共同末token的embedding身份基线','layer':0,'boundary':'embedding','index':{'space':'residual','coordinate':0},'run_id':P.RUN,'method':'all2560coordinates exact baseline, selected coordinate merely a native locator','scope':{'trace_id':t['trace_id'],'position':1023,'all_coordinates':2560},'evidence':{'level':'observation','artifact_sha256':emb},'counterexamples':[]})
    limitations=['仅单模型、单节点、固定快照；尚无跨节点独立复算或跨模型复现。','主实验部分条件/全称留出句式行为未过门，不能将可读出当成模型掌握对应行为。',
        '两个新极性队列支持指定Yes/No任务的答案方向解释；未区分答案词、抽象决策或任务策略，未证明单义语义神经元。','主实验Amber/Birch与事实相关；独立的地点极性队列平衡cobalt/umber后仍出现答案方向响应。',
        '只在已声明位置保存完整模块数组，只有8条深采保留全token QKV/隐态；未采集的位置明确不可用。','6/8单元的联合消融只作用最后回答query；问题结束与历史位置没有被消融，负结果不能否定历史路径。',
        '2200题能力池已在历史试验中曝光；非劣效仅限当前操作与选择题监测口径，不代表自然生成、所有能力或长期安全。',
        '全场方向是监督均值差比较器，不是自主提取的字典、完整图谱预测器，也未完成与SAE/transcoder同预算比较。',
        '读出注释追加v3保留原manifest与v2；数值、坐标、样本身份未改变。原始张量在本机，离线数据包包含索引与证据不包含近百GiB原场。',
        '3D层高/阵列位置用于原生索引定位，图距离不是语义距离；参数算术来源不是因果必要性；数量不代表破解百分比。']
    sources=sorted((P.ROOT/'tests/glm5').glob('native_mlp_function*.py'))+[P.ROOT/'shared/dist_protocol/mechanism_trace.py']
    refs['sources']=archive('sources',{'schema':'research_source_bundle.v1','files':[{'path':str(p.relative_to(P.ROOT)),'sha256':file_sha(p),'text':p.read_text(encoding='utf-8')} for p in sources]})
    report={'schema':'research_native_function_report.v1','analysis':a,'coordinate_candidates':coordinates,'calibrations':calibrations,'polarity_controls':polarities,
        'dose':{k:v for k,v in dose.items() if k!='rows'},'capability':cap['groups']['coordinate'],'capability_groups':cap['groups'],'examples':examples,'counterexamples':errors,
        'native':native,'preregistration':P.read(P.PUBLIC/'prereg.research.json'),'amendment':P.read(P.PUBLIC/'prospective-amendment.research.json'),
        'artifact_refs':refs,'case_index_sha256':case_index,'limitations':limitations,'decisive_interpretation':'response follows conditioned answer in polarity tests; fixed property and individual switch unproven'}
    rr=archive('report',report);gr=archive('graph',{'schema':'research_graph.v1','feature_cards':features,'graph_edges':edges,'mechanisms_fully_closed':0})
    context={'study_type':'native_mlp_function','unit_cards':unit_cards,'family_profile_refs':profiles['refs'],'case_index_sha256':case_index,'profile_scope':'main paired cohort; do not treat as polarity-cohort averages','report_summary':{'primary_supported':1,'coordinate_answer_aligned':6,'joint_dose_supported':False}}
    register_binding(archive,store,P.RUN,rr,gr,examples,context,'原生MLP：功能假设、答案方向反例与全层真实3D',native=native)
    for run in [P.RUN+':place-polarity',P.RUN+':capability-audit',P.RUN+':operation-audit']:
        cc={**context,'parent_run_id':P.RUN}
        if ':capability-' in run:cc.pop('family_profile_refs')
        register_binding(archive,store,run,rr,gr,byrun[run][:8],cc,run.split(':')[-1]+'：原始操作/反例复采')
    did=P.RUN+':controlled-long-text'
    reg.register({'schema':'dataset_version.v1','kind':'real','id':did,'dataset_id':'native-mlp-controlled-long-text','version':'4','content_sha256':digest(c),'source':'frozen generated catalogue pairs over pinned natural backgrounds; complete token IDs archived','license':'generated material plus per-source natural text license/lineage','label_status':'operational fact, selector, question, expected Yes/No; not internal ontology','role':'controlled_mechanism','source_groups':sorted({r['background_doc_id'] for r in c['items']}),'sample_count':3264,'provenance':prov})
    for name,status,scope in [('conditioned_answer_response','supported_in_scope','six selected coordinates under two fresh fact×question cohorts; readout roles separately'),('fixed_property_neuron','not_supported','polarity/name-balanced counterexample narrows fixed-property interpretation'),('joint_query_switch','not_supported','six/eight units finalassistantquery only; no correctness changes')]:
        reg.register({'schema':'research_claim.v1','kind':'real','id':P.RUN+':'+name,'canonical_feature_id':P.RUN+':'+name,'model_snapshot':snap,'external_pattern_id':'language:conditioned_yes_no_answer','outcome_metric':name,'scope':scope,'status':status,'dataset_versions':[did],'artifact_sha256':rr,'run_id':P.RUN,'provenance':prov})
    # Existing native studies get discoverable bindings; other archives remain metadata-only.
    for rel in ['research/phase2793/unified-evidence.research.json','research/phase2794/unified-evidence.research.json','research/native_mlp_long/v1/unified-evidence.research.json']:
        p=P.ROOT/rel
        if not p.exists():continue
        d=P.read(p);run=d['run_id'];oldexamples=d.get('examples',[])[:6]
        if not oldexamples:continue
        register_binding(archive,store,run,d['report_sha256'],d['graph_sha256'],oldexamples,d,'历史原生研究 · '+run)
    out={'report_sha256':rr,'graph_sha256':gr,'case_index_sha256':case_index,'feature_cards':len(features),'edges':len(edges),'native':native,
        'space_url':f'http://127.0.0.1:5173/workbench?lens=spatial&view=research&run={P.RUN}','trace_url':'http://127.0.0.1:5173/workbench'+examples[0]['url']}
    P.save(P.OUT/'client-registration.json',out);print(json.dumps({k:v for k,v in out.items() if k!='native'},ensure_ascii=False),flush=True)

def register_binding(archive,store,run,report,graph,traces,context,title,native=None):
    proof=[]
    for t in traces[:16]:
        d,_=store.manifest(t['trace_id']);sha=archive('trace-proof-'+t['trace_id'],d,run);proof.append({**t,'manifest_sha256':sha})
    archive('binding-'+digest(run)[:12],{'schema':'research_space_binding.v1','run_id':run,'report_sha256':report,'graph_sha256':graph,'traces':proof,'context':context},run)
    archive('workpackage-'+digest(run)[:12],{'schema':'research_workpackage.v1','run_id':run,'parent_run_id':P.RUN if run!=P.RUN else None,'title':title,
        'stages':{'collection':{'status':'done','native':native},'registration_validated':{'status':'done','report_sha256':report,'graph_sha256':graph},
            'interpretation':{'status':'scope_limited','note':'Answer alignment is not fixed property, unique mechanism or global deciphering percentage; failed gates and negative operations retained.'}}},run)

if __name__=='__main__':main()
