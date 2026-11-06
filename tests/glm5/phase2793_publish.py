"""Recompute study metrics, retain failures, migrate explicit historical sources."""
from pathlib import Path
import collections
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2793_materials as P
import phase2793_run as R
from shared.dist_protocol.mechanism_trace import TraceStore,digest,file_sha,unbits
from shared.dist_protocol.research_registry import Registry
from deploy.research_library import Library


def ci(v):
    x=np.asarray(v,float);rng=np.random.default_rng(P.SEED+1);m=x[rng.integers(0,len(x),(20000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(m,[.025,.975]).tolist(),'n_families':len(x),'method':'entity-cluster bootstrap'}


def main():
    out=P.OUT;root=P.ROOT;pub=P.PUBLIC;c=P.W.read(out/'corpus.json');pr=P.W.read(pub/'prereg.research.json')
    obs=P.W.read(out/'observations.json');rows=obs['rows'];prov=obs['provenance'];model=c['model_snapshot'];lib=Library(root/'distributed_data/research_library');registry=Registry(lib);store=TraceStore()
    freeze=P.W.read(out/'execution-freeze.json');assert digest(c)==freeze['corpus_sha256']
    assert all(file_sha(root/p)==h for p,h in freeze['dependencies'].items())
    def archive(name,doc):
        path=pub/(name+'.research.json');P.save(path,doc,True);return lib.ingest(doc,str(path.relative_to(root)),kind='local_report')['sha256']
    historical=[]
    sources=[('实体共享属性','research/entity_atlas/EA-v2-qwen3-4b/report.json'),('属性写入窗口','research/entity_atlas/EA-v4-qwen3-4b/report.json'),
        ('知识与上下文仲裁','research/arbitration/CF-v2-qwen3-4b/report.json'),('注意力仲裁路径','research/path_scan/PS-v1-qwen3-4b/report.json'),
        ('代码门控参数通道','research/weight_space/WS-v1-qwen3-4b/report.json'),('同模式QK','research/weight_space/WS-v4-qwen3-4b/report.json'),
        ('GLM-bf16门控复现','research/weight_space/GL-v1-glm4-9b-chat-hf/report.json'),('第30层字典','research/transcoder/TC-v1-qwen3-4b-L30/report.json'),
        ('受控图谱拼装预测','research/assembly/AM-v1-qwen3-4b/report.json')]
    for title,name in sources:
        path=root/name;assert path.exists();nonfinite=[]
        def mark(v):nonfinite.append(v);return {'$nonfinite':v}
        original=path.read_text(encoding='utf-8');d=json.loads(original,parse_constant=mark);snap=d.get('model_snapshot')
        if isinstance(snap,dict):snap=snap.get('sha256')
        wrapper={'schema':'research_historical_evidence.v1','kind':'real','title':title,'source_path':name,'source_file_sha256':file_sha(path),
            'model_snapshot':snap,'precision':d.get('precision','not_declared_in_original_report'),'original_report':d,
            'migration_status':'original artifact preserved; not recomputed or automatically promoted',
            'provenance':prov,'provenance_scope':'import operation only, not retroactive execution provenance'}
        if nonfinite:wrapper.update(original_utf8=original,nonfinite_original=nonfinite,note='Original nonfinite values retained as tags and source text, not replaced by valid statistics.')
        if snap and snap!=model:wrapper['provenance']={**prov,'model_revision':snap,'model_path':'historical-report:'+name,'seed':None}
        ref=archive('history/'+title,wrapper);historical.append({'title':title,'artifact_sha256':ref,'model_snapshot':snap,'precision':wrapper['precision'],'migration_status':wrapper['migration_status']})
    # Pair equality / causal visibility and native byte/hash audit, independently recomputed.
    pairs=collections.defaultdict(dict);lookup={r['id']:r for r in rows};byid={x['id']:x for x in c['items']}
    manifests=[];native_bytes=0;files=0;max_error=0;early_max=0.;hs0_max=0.
    hidden={};mlp={};headz={}
    for index,item in enumerate(c['items']):
        r=lookup[item['id']];d,path=store.manifest(r['trace_id']);manifests.append(d)
        assert d['n_tokens']==item['length']==len(item['token_ids']) and d['model_snapshot']==model
        for meta in d['files']:
            p=store.native_path(d['trace_id'],meta['path']);native_bytes+=meta['bytes'];files+=1
        max_error=max(max_error,max(x['attention_relative_l2'] for x in d['reconstruction'].values()))
        h=np.load(path/'hidden_readouts.npy');hidden[item['id']]=unbits(h)
        finalA=[];finalZ=[]
        for l in range(36):
            with np.load(path/f'block_{l:02d}.npz') as z:finalA.append(unbits(z['activation'][-1]));finalZ.append(unbits(z['head_z'][-1]))
        mlp[item['id']]=np.stack(finalA);headz[item['id']]=np.stack(finalZ)
        pairs[(item['family_id'],item['frame'],item['length'])][item['scope']]=item
    for pair in pairs.values():
        a,b=pair['文档'],pair['现实'];assert a['length']==b['length'] and a['source_text']==b['source_text']
        assert a['text'].replace(a['scope_question'],'')==b['text'].replace(b['scope_question'],'')
        assert a['readout_positions']==b['readout_positions']
        ha,hb=hidden[a['id']],hidden[b['id']];early_max=max(early_max,float(np.max(np.abs(ha[:,:-1]-hb[:,:-1]))));hs0_max=max(hs0_max,float(np.max(np.abs(ha[0]-hb[0]))))
    assert early_max==0 and hs0_max==0 and max_error<.05
    held=[r for r in rows if r['split']=='heldout_entity_frame'];families=sorted({r['family_id'] for r in held});values=[]
    for f in families:
        g=[r for r in held if r['family_id']==f];values.append(np.mean([r['document_choice'] for r in g if r['scope']=='文档'])-np.mean([r['document_choice'] for r in g if r['scope']=='现实']))
    scope_effect=ci(values);gates={s:float(np.mean([r['correct'] for r in held if r['scope']==s])) for s in ['文档','现实']};gate=min(gates.values())>=.8
    h1=scope_effect['mean']>=.5 and scope_effect['ci95'][0]>.2 and gate
    interventions=P.W.read(out/'head-interventions.json');base={r['id']:r for r in rows};loss={}
    for condition in ['selected_heads','matched_random_heads']:
        g=[r for r in interventions['rows'] if r['condition']==condition];loss[condition]=ci([int(base[r['sample_id']]['correct'])-int(r['correct']) for r in g])
    h2=gate and loss['selected_heads']['mean']>=.2 and loss['selected_heads']['ci95'][0]>0 and loss['selected_heads']['mean']>loss['matched_random_heads']['mean']
    # Observational full-coordinate profiles; selected on discovery only, not causal feature proof.
    profile={};direction_disc=[];direction_held=[];representative={}
    entity_diffs={'discovery':[],'heldout_entity_frame':[]};entity_hidden={'discovery':[],'heldout_entity_frame':[]}
    for e in c['entities']:
        ps=[pair for (fid,_,_),pair in pairs.items() if fid==e['id']]
        entity_diffs[e['split']].append(np.mean([mlp[p['文档']['id']]-mlp[p['现实']['id']] for p in ps],axis=0))
        entity_hidden[e['split']].append(np.mean([hidden[p['文档']['id']][:,-1]-hidden[p['现实']['id']][:,-1] for p in ps],axis=0))
    ad=np.asarray(entity_diffs['discovery']);ah=np.asarray(entity_diffs['heldout_entity_frame'])
    hd=np.asarray(entity_hidden['discovery']);hh=np.asarray(entity_hidden['heldout_entity_frame'])
    disc=ad.mean(0);conf=ah.mean(0);scale=np.maximum(ad.std(0,ddof=1),1e-4);score=disc/scale
    np.savez(out/'native-profiles.npz',mlp_discovery_delta=disc,mlp_heldout_delta=conf,mlp_discovery_standardized=score,
        hidden_discovery_delta=hd.mean(0),hidden_heldout_delta=hh.mean(0))
    for l in range(36):
        unit=int(np.argmax(np.abs(score[l])));representative[str(l)]=unit
        profile[str(l)]={'unit':unit,'discovery_delta':float(disc[l,unit]),'heldout_delta':float(conf[l,unit]),
            'same_sign':bool(disc[l,unit]*conf[l,unit]>0),'discovery_standardized':float(score[l,unit]),
            'scope':'scope-prompt response including incorrect arbitration; no FWER-controlled claim or causality'}
    prior=P.W.read(out/'source-free-priors.json');reference_reviews=[]
    blind=P.W.read(out/'reference-quality-blind.json');expected={x['id'] for x in blind['records']};quality_maps=[]
    for suffix in ['astra','sol']:
        p=out/f'reference-quality-{suffix}.json';d=P.W.read(p);assert d['input_sha256']==file_sha(out/'reference-quality-blind.json')
        assert len(d['ratings'])==48 and {x['id'] for x in d['ratings']}==expected
        quality_maps.append({r['id']:r for r in d['ratings']});d.update(kind='real',provenance={**prov,'model_path':'collaboration:'+d['rater_model'],'model_revision':'provider_alias:'+d['rater_model'],'seed':None},
            status='posthoc AI reference-quality audit; primary labels unchanged; no human gold')
        reference_reviews.append({'rater_model':d['rater_model'],'counts':dict(collections.Counter(r['quality'] for r in d['ratings'])),'artifact_sha256':archive('reference-quality-'+suffix,d)})
    reference_summary={'n':48,'both_typical':sum(all(q[k]['quality']=='typical_unique' for q in quality_maps) for k in expected),
        'disagreements':sum(quality_maps[0][k]['quality']!=quality_maps[1][k]['quality'] for k in expected),'reviews':reference_reviews,
        'status':'posthoc, not a change to preregistered thresholds or labels'}
    counter=[]
    for r in rows:
        if not r['correct']:counter.append({'sample_id':r['id'],'reason':'reference_colour_not_selected','entity':r['entity'],'scope':r['scope'],'expected':r['expected'],
            'predicted':r['palette_choice'],'prior':r['prior'],'document':r['document'],'trace_id':r['trace_id']})
    for r in interventions['rows']:
        if base[r['sample_id']]['correct'] and not r['correct']:counter.append({'sample_id':r['sample_id'],'reason':'head_ablation_regression','condition':r['condition'],**r})
    cap=P.W.read(out/'capability-result.json');cb=P.W.read(out/'capability-base.json');ce=P.W.read(out/'capability-selected_heads.json')
    for b,e in zip(cb,ce):
        if b['correct'] and not e['correct']:counter.append({'sample_id':b['id'],'reason':'keyed_capability_regression' if b['gate_eligible'] else 'reference_only_change','base':b,'edited':e})
    stats=[]
    for split in ['discovery','heldout_entity_frame']:
        for scope in ['文档','现实']:
            for length in [512,1024,2048]:
                g=[r for r in rows if (r['split'],r['scope'],r['length'])==(split,scope,length)]
                stats.append({'split':split,'scope':scope,'length':length,'n':len(g),'reference_accuracy':float(np.mean([r['correct'] for r in g])),
                    'document_choice_rate':float(np.mean([r['document_choice'] for r in g]))})
    audit={'kind':'real','run_id':P.RUN,'model_snapshot':model,'native_files_verified':files,'native_bytes':native_bytes,
        'paired_inputs':len(pairs),'all_equal_length':True,'early_positions_scope_delta_max':early_max,'embedding_delta_max':hs0_max,
        'max_attention_reconstruction_relative_l2':max_error,'weights_unchanged':True,'provenance':prov}
    audit_ref=archive('native-audit',audit)
    obs_ref=archive('observations',obs);cap_ref=archive('capability-monitor',{'schema':'research_capability_monitor.v1','kind':'real','run_id':P.RUN,
        'model_snapshot':model,'base':cb,'edited':ce,'result':cap,'contract':P.W.read(out/'capability-contract.json'),'provenance':prov})
    input_refs=[]
    for split in ['discovery','heldout_entity_frame']:
        input_refs.append(archive('inputs-'+split,{'schema':'research_frozen_inputs.v1','kind':'real','run_id':P.RUN,'model_snapshot':model,'items':[r for r in c['items'] if r['split']==split],'provenance':prov}))
    source_paths=[root/p for p in freeze['dependencies']]+[Path(__file__),root/'tests/glm5/phase2793_capability.py']
    source_ref=archive('sources-'+file_sha(Path(__file__))[:12],{'schema':'research_source_bundle.v1','kind':'real','run_id':P.RUN,'model_snapshot':model,'input_artifacts':input_refs,
        'execution_freeze':freeze,'files':[{'path':str(p.relative_to(root)),'sha256':file_sha(p),'text':p.read_bytes().decode('utf-8')} for p in source_paths],
        'execution':'reproduction data only; never auto-execute downloaded source','provenance':prov})
    selection=P.W.read(out/'head-selection.json');examples=[]
    for gen in obs['generations']:
        item=byid[gen['sample_id']];r=lookup[item['id']];block=selection['heads'][0][0];unit=representative[str(block)]
        examples.append({'sample_id':item['id'],'entity':item['entity'],'scope':item['scope'],'length':item['length'],'expected':item['expected'],'actual':r['palette_choice'],
            'generation':gen,'trace_id':r['trace_id'],'url':f'?lens=spatial&view=live&trace={r["trace_id"]}&tblock={block}&thead={selection["heads"][0][1]}&tunit={unit}'})
    limitations=['参考标签来自项目常识词表，不是所有品种/状态都唯一；事后两AI审计不修改主判。','现实范围行为门失败，不把响应选择性或DLA当正确仲裁机制。',
        '受控长材料含重复背景，长度增加同时增加内容；不是自然长语料。','主判是11色字候选分布，自由生成仅8固定深采例、每例最多24token。',
        'Head选择只在16发现实体，32留出实体/新句式；8头消融不证明单个头必要或全部仲裁路径。','单元与残差方向是观察性范围提示响应，未控制全部局部n-gram、未做全单位族错误检验。',
        '参数乘积是局部计算来源，固定RMS归因不等于下游因果效应；原生/重算误差单列。','全部坐标在声明的读出保留；只有8前缀深采全token隐态/QKV，不是所有576输入全token全模块。',
        '新能力监控在仲裁结果之后登记，属额外探索；2200题是重复题池、医学占比高，140镜像参考题排除正式门。','只有1台物理节点；无跨模型或独立第二节点复算。']
    report={'schema':'research_entity_arbitration_report.v1','kind':'real','run_id':P.RUN,'title':'长文本实体属性仲裁与真实三模块路径','model_snapshot':model,
        'n_inputs':576,'n_entities':48,'n_discovery_entities':16,'n_heldout_entities':32,'behavior_gate':gate,'scope_accuracy':gates,
        'scope_effect':scope_effect,'H1_supported':bool(h1),'H2_supported':bool(h2),'head_ablation':loss,'head_selection':selection,
        'source_free_reference_accuracy':float(np.mean([r['correct'] for r in prior['rows']])),'reference_quality':reference_summary,'cell_statistics':stats,
        'profiles':profile,'profile_note':'full-coordinate descriptive effects, discovery selection only; no semantic atom or necessity claim',
        'capability':cap,'examples':examples,'counterexamples':counter,'historical_evidence':historical,'native_audit':audit,
        'artifact_refs':{'observations':obs_ref,'audit':audit_ref,'inputs':input_refs,'sources':source_ref,'capability':cap_ref},
        'preregistration':pr,'execution':P.W.read(out/'execution.json'),'limitations':limitations,'provenance':prov}
    report['supersedes_scope']='Adds independently registered exploratory capability monitor and posthoc AI reference audit; original primary verdict unchanged.'
    rr=archive('report-unified-v2',report);cards=[];edges=[]
    for l in range(36):
        unit=representative[str(l)];fid=f'{model}:B{l}:scope-response:U{unit}:P2793'
        cards.append({'id':fid,'label':f'范围提示响应 B{l}/U{unit}（含错误仲裁）','layer':l,'boundary':'mlp.down_proj.input',
            'index':{'space':'mlp','unit':unit},'method':'full native units; discovery-selected standardized document-minus-reality response, heldout descriptive check',
            'run_id':P.RUN,'scope':{'readout':'final','representative_unit_only':True,'baseline_behavior_gate':gate,'individual_causality':'not_tested'},
            'evidence':{'level':'observation','artifact_sha256':rr},'counterexamples':counter[:12],'summary':profile[str(l)]})
        edges.append({'id':fid+':mapping','source':'language:source-scope-prompt','target':fid,'relation':'observed_conditioned_response_not_correct_arbitration',
            'run_id':P.RUN,'evidence':{'level':'observation','artifact_sha256':rr}})
    for l,h in selection['heads']:
        fid=f'{model}:B{l}:Head{h}:P2793';cards.append({'id':fid,'label':f'仲裁候选 Head B{l}/H{h} · 必要性未支持','layer':l,
            'boundary':'self_attn.o_proj.input','index':{'space':'attention','head':h},'method':'discovery-selected DLA and preregistered8-head joint ablation',
            'run_id':P.RUN,'scope':{'joint_group':P.RUN+':heads8','individual_causality':'not_tested','activation_ablation_not_parameter_edit':True},
            'evidence':{'level':'negative','artifact_sha256':rr},'counterexamples':counter[:12]})
        edges.append({'id':fid+':test','source':'language:entity-colour-arbitration','target':fid,'relation':'joint_necessity_test_not_supported',
            'run_id':P.RUN,'evidence':{'level':'negative','artifact_sha256':rr}})
    gr=archive('graph-unified-v2',{'schema':'research_graph.v1','kind':'real','run_id':P.RUN,'model_snapshot':model,'feature_cards':cards,'graph_edges':edges,
        'independent_intervention_tests':1,'independent_response_profiles':1,'provenance':prov})
    # Existing primary dataset/claim identities remain canonical. This report supplements
    # their evidence; it is not a second dataset or a second independent H1/H2 test.
    did=P.RUN+':corpus'
    context={'schema':'unified_research_context.v1','kind':'real','run_id':P.RUN,'model_snapshot':model,'historical_evidence':historical,
        'report_sha256':rr,'graph_sha256':gr,'report_summary':{k:report[k] for k in ['scope_accuracy','scope_effect','behavior_gate','H1_supported','H2_supported','source_free_reference_accuracy','reference_quality']},
        'profiles':profile,'examples':examples,'limitations':limitations,'dataset_version_id':did,
        'claim_ids':[P.RUN+':paired_scope_behaviour',P.RUN+':selected_head_group_effect'],'provenance':prov}
    archive('unified-evidence',context)
    registration={'report_sha256':rr,'graph_sha256':gr,'client_url':f'http://127.0.0.1:5173/workbench?lens=process&station=plan&run={P.RUN}',
        'trace_url':'http://127.0.0.1:5173/workbench'+examples[-1]['url'],'native_bytes':native_bytes,'native_files':files,'history_sources':len(historical)}
    P.save(out/'client-registration-v2.json',registration,True);R.status('registration_validated',**registration)
    print(json.dumps({'scope_accuracy':gates,'scope_effect':scope_effect,'H1':bool(h1),'H2':bool(h2),'ablation':loss,'reference_quality':reference_summary,
        'capability':{k:cap[k] for k in ['n','status','loss_interval']},'counterexamples':len(counter),'registration':registration},ensure_ascii=False))


if __name__=='__main__':main()
