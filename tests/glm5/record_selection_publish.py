"""Local evidence audit and content-addressed client publication."""
import base64,hashlib,io,json,tarfile,time
import numpy as np
import record_selection_core_v3 as R
from record_selection_core_v3 import C
from record_selection_analysis import checked,admission
from deploy.research_library import Library,canonical

def audit():
    p,a=checked();r=C.read(C.OUT/'report.json');selection=C.read(C.OUT/'selection.json');start=time.monotonic()
    assert r['plan_sha256']==C.sha(C.OUT/'plan.json') and r['selection_sha256']==C.sha(C.OUT/'selection.json')
    assert r['atlas_sha256']==C.sha(C.OUT/'confirm-atlas.npz')
    assert not r['formal_evidence_promotion'] and all(not e['scope_use_supported'] for e in r['effects'])
    assert selection['semantic_discovery_admitted'] is False
    manifest=[];maxq=0;prefix_checks=0;sample_ids=set();loader=None;loaders={}
    for stage in ['pilot','calibration_full','restricted_discovery','restricted_confirm','causal']:
        path=C.OUT/f'loader-{stage}.json';doc=C.read(path)
        assert doc['model_snapshot']==C.SNAP and doc['plan_sha256']==C.sha(C.OUT/'plan.json')
        if loader is None:loader=doc['checks']
        else:assert loader==doc['checks']
        loaders[stage]={'sha256':C.sha(path),'parameter_tensors':len(doc['checks'])}
    for split,n in [('discovery',128),('confirm',384)]:
        rows=C.read(C.OUT/(split+'-materials.json'))['rows'];scores=[]
        for f in range(n):
            meta=C.read(C.OUT/f'{split}-{f:04d}.json');assert meta['prefix_control_bitwise']
            assert meta['plan_sha256']==C.sha(C.OUT/'plan.json') and meta['analysis_plan_sha256']==C.sha(C.OUT/'analysis-plan.json')
            prefix=None
            for i,item in enumerate(meta['native']):
                path=C.OUT/item['path'];assert C.sha(path)==item['sha256'];assert item['sample_id']==rows[f*8+i]['id']
                assert item['sample_id'] not in sample_ids;sample_ids.add(item['sample_id']);maxq=max(maxq,item['qkv_error_max'])
                manifest.append({**item,'bytes':path.stat().st_size})
                if f%32==0:
                    with np.load(path) as z:
                        assert z['hidden'].shape==(37,4,2560) and z['units'].shape==z['gate'].shape==z['value'].shape==(36,9728)
                        assert z['head_context'].shape==(36,32,128)
                        np.testing.assert_allclose(z['source_mass'].sum(-1),1,atol=2e-6)
                        if prefix is None:prefix=z['hidden'][:,0].copy()
                        else:np.testing.assert_array_equal(prefix,z['hidden'][:,0]);prefix_checks+=1
            scores+=meta['scores']
            if (f+1)%64==0:C.log('audit_'+split,f+1,n,start)
        gate=R.behavior(scores,p);stored=C.read(C.OUT/(split+'-behavior.json'))
        assert gate['cells']==stored['cells'] and gate['passed']==stored['passed']
        assert np.where(admission(split))[0].tolist()==C.read(C.OUT/(split+'-admission.json'))['eligible_families']
    assert maxq<.02
    amend=C.read(C.OUT/'controls-amendment.json')
    assert amend['selection_sha256']==C.sha(C.OUT/'selection.json') and amend['analysis_plan_sha256']==C.sha(C.OUT/'analysis-plan.json')
    for name,digest in amend['sources'].items():assert C.sha(C.ROOT/name)==digest
    assert C.read(C.OUT/'causal-admission.json')['controls_amendment_sha256']==C.sha(C.OUT/'controls-amendment.json')
    maxkl=0.;maxnorm=0.;maxnative=0.;count=0
    for f in range(384):
        item=C.read(C.OUT/f'causal-{f:04d}.json');assert item['selection_sha256']==C.sha(C.OUT/'selection.json')
        assert item['restoration_bitwise'] and item['scope']=='observed_choice_diagnostic_after_behavior_failure'
        path=C.OUT/f'causal-logits-{f:04d}.npz';assert C.sha(path)==item['native_logits_sha256']
        original=next(s for s in C.read(C.OUT/f'confirm-{f:04d}.json')['scores'] if s['id']==item['sample']['id'])
        for k,v in item['baseline'].items():assert v==original[k],(f,k)
        with np.load(path) as z:
            base=C.unbits(z['base']).astype(float);lb=base-base.max();lb-=np.log(np.exp(lb).sum())
            for c in [None,*selection['candidates']]:
                name='base' if c is None else c['id'];logits=C.unbits(z[name]).astype(float)
                rec=None if c is None else next(x for x in item['records'] if x['candidate']==name and x['method']=='ablation' and x['dose']==1)
                score=item['baseline'] if c is None else rec['edited'];vals=[float(logits[p['record_ids'][str(i)]]) for i in [1,2]]
                assert vals==score['choice_logits'] and int(logits.argmax())==score['argmax'];assert vals[0]-vals[1]==score['record_margin'];count+=1
                if rec:
                    lp=logits-logits.max();lp-=np.log(np.exp(lp).sum());kl=float((np.exp(lp)*(lp-lb)).sum())
                    maxkl=max(maxkl,abs(kl-score['KL_edit_base']));maxnorm=max(maxnorm,rec['norm_relative_error'])
                    maxnative=max(maxnative,max(v['matched_norm_relative_error'] for v in rec['group_controls']))
        if (f+1)%64==0:C.log('audit_causal',f+1,384,start)
    assert maxkl<2e-5 and maxnorm<=.02 and maxnative<=.02 and len(sample_ids)==4096
    C.save('audit.json',{'schema':'record_selection_audit.v1','passed':True,'report_sha256':C.sha(C.OUT/'report.json'),
        'native_inputs':4096,'native_manifest':manifest,'weight_loaders':loaders,'maximum_QKV_relative_error':maxq,
        'prefix_arrays_independently_compared':prefix_checks,'original_prefix_checks_all512_families':True,
        'full_vocabulary_arrays_recomputed':count,'maximum_CPU64_KL_error':maxkl,'maximum_actual_norm_control_error':maxnorm,'maximum_native_direction_norm_error':maxnative,
        'causal_baselines_match_original':384,'restored_outputs_bitwise':384,'discovery_eligible':int(admission('discovery').sum()),
        'confirm_eligible':int(admission('confirm').sum()),'sources_and_materials_unchanged':True,'formal_promotion':False,
        'scope':'same-node independent numeric implementation checks; not independent-node scientific replication'})

def publish():
    p,a=checked();r=C.read(C.OUT/'report.json');auditdoc=C.read(C.OUT/'audit.json');assert auditdoc['passed'] and auditdoc['report_sha256']==C.sha(C.OUT/'report.json')
    libs=[Library(C.ROOT/v) for v in ['distributed_data/research_library','tests/glm5/result/mlp_tail_acceptance/center/research_library']];objects={}
    def put(name,body):
        doc={**body,'kind':'real','run_id':C.RUN,'model_snapshot':C.SNAP,'provenance':p['provenance']};raw=canonical(doc)
        hashes={lib.ingest(doc,f'{C.RUN}/{name}',kind='local_report')['sha256'] for lib in libs};assert len(hashes)==1
        digest=hashes.pop();objects[digest]=raw;return digest
    artifacts={label:put(name,C.read(C.OUT/name)) for label,name in [('原冻结方案','plan.json'),('失败校准后的受限分析协议','analysis-plan.json'),('候选组与留出前冻结','selection.json'),('同节点技术审计','audit.json'),('留出行为与失败格','confirm-behavior.json'),('配对排除登记','confirm-admission.json'),('作用诊断准入','causal-admission.json')]}
    artifacts['同层对照幅度不足后的补强协议']=put('controls-amendment',C.read(C.OUT/'controls-amendment.json'))
    embedding=C.read(C.OUT/'embedding-baseline-freeze.json');artifacts['嵌入层常数基线']=put('embedding-baseline',embedding)
    truth=np.load(C.OUT/'confirm-atlas.npz')['prediction_truth'];keep=admission('confirm')
    embedding_mse=((truth-embedding['prediction'])**2).reshape(384,8).mean(1)[keep]
    r['baselines']['mse']['embedding_constant']=float(embedding_mse.mean())
    r['baselines']['embedding_scope']=embedding
    matched=C.read(C.OUT/'matched-control-report.json');artifacts['真实对照方向的等幅度检验']=put('matched-controls',matched)
    r['matched_native_controls']=matched['results']
    failure=C.read(C.OUT/'behavior-failure-strata.json');artifacts['行为错误的条件分组诊断']=put('failure-strata',failure)
    r['behavior_error_strata']=next(x['strata'] for x in failure['results'] if x['split']=='confirm')
    r['limitations']+=['Native-group original write energies were only .25-.85 of candidate proxies; original group controls are not magnitude matched. Supplemental actual residual-direction controls correct amplitude but may extrapolate native edits.',
        'B26 unit group precedes B29 head group: no B29-to-B26 causal chain is possible in this forward pass.',
        'The numeric output contrast may reflect general 1/2 choice or position reading; unrelated numeric tasks and output-label remapping were not tested.',
        'Behavior-based family filtering on confirm yields a conditional diagnostic subset, not unbiased full-domain confirmation.',
        'B29 direct source QK masses and whole head outputs are available; source-specific V/O vectors were archived only for B18-26.',
        'Sixteen record-frame bootstrap clusters share four query frames per language; intervals do not establish broad population uncertainty.']
    history=[]
    for version,name,label in [(1,'pilot-behavior.json','v1 颜色提取小试'),(2,'pilot-behavior.json','v2 明确状态指令的颜色小试'),(3,'pilot-behavior.json','v3 记录编号小试'),(3,'calibration-full-behavior.json','v3 完整发现集行为校准'),(3,'confirm-behavior.json','v3 全部独立留出行为')]:
        root=C.OUT.parent/f'record_selection_v{version}';gate=C.read(root/name)
        history.append({'id':f'{version}-{name}','label':label,'n':gate['n'],'accuracy':gate['strict_accuracy'],'passed':gate['passed'],'source_sha256':C.sha(root/name)})
    old=[]
    for version in [1,2]:
        root=C.OUT.parent/f'record_selection_v{version}';material=C.read(root/'discovery-materials.json')['rows'];byid={v['id']:v for v in material}
        for path in sorted(root.glob('pilot-[0-9]*.json')):
            for score in C.read(path)['scores']:old.append({'version':version,'original_run_id':C.read(root/'plan.json')['run_id'],
                'original_plan_sha256':C.sha(root/'plan.json'),'original_pilot_metadata_sha256':C.sha(path),'sample':byid[score['id']],'score':score})
    artifacts['两轮颜色提取失败原始记录']=put('failed-calibrations',{'schema':'record_selection_calibration_history.v1','history':history,'old_pilot_rows':old,'scope':'v1/v2 color scores, distinct from v3 numeric source scores'})
    parts=[]
    for split,n in [('discovery',128),('confirm',384)]:
        rows=C.read(C.OUT/(split+'-materials.json'))['rows']
        for f in range(0,n,32):parts.append(put(f'materials-{split}-{f}',{'schema':'record_selection_materials.v1','split':split,'rows':rows[f*8:(f+32)*8]}))
    artifacts['全部输入与token位置']=put('materials-index',{'schema':'record_selection_index.v1','parts':parts})
    parts=[]
    for f in range(0,384,16):parts.append(put(f'causal-{f}',{'schema':'record_selection_causal_evidence.v1','rows':[C.read(C.OUT/f'causal-{i:04d}.json') for i in range(f,f+16)]}))
    artifacts['全部干预及负对照']=put('causal-index',{'schema':'record_selection_index.v1','parts':parts})
    disc=np.load(C.OUT/'discovery-atlas.npz');conf=np.load(C.OUT/'confirm-atlas.npz');coordinates={'heads':{},'units':{}}
    def vector(kind,index,value):
        value=value.astype('<f4');return put(f'coordinates-{kind}-{index}',{'schema':'record_selection_coordinates.v1','field':kind,'index':index,
            'readout':'answer_prefix','scope':'behavior-eligible whole paired families; original native order',
            'array':{'dtype':'<f4','encoding':'base64','shape':list(value.shape),'data':base64.b64encode(value.tobytes()).decode('ascii')}})
    coordinates['heads']['all']=vector('heads','all',np.stack([z['head_validity_by_position'].reshape(2,1152) for z in [disc,conf]]))
    for b in range(36):coordinates['units'][str(b)]=vector('units',b,np.stack([np.stack([z['unit_choice_contrast'][b],z['unit_dla_contrast'][b]]) for z in [disc,conf]]))
    r.update(coordinate_artifacts=coordinates,artifacts=artifacts,calibration_history=history,audit_summary={k:v for k,v in auditdoc.items() if k!='native_manifest'},diagnostic_only=True)
    ref=put('report',r);cards=[];edges=[]
    for c in r['candidates']:
        cid=C.RUN+':'+c['id'];space='attention' if c['kind']=='heads' else 'mlp'
        cards.append({'id':cid,'run_id':C.RUN,'layer':c['block'],'boundary':'attention.o_proj.input' if space=='attention' else 'mlp.down_proj.input',
            'label':c['id']+' · 有效记录选择的受限组候选','index':{'space':space,'indices':c['indices'],'group':True,'anchor_only':False},
            'method':'train-frame selection; validation maxT family swaps; frozen new-frame diagnostics; native group and matched-norm controls',
            'evidence':{'level':'observation','artifact_sha256':ref},'scope':{'formal_promotion':False,'full_scope_behavior_failed':True,'causal_chain_closed':False},
            'counterexamples':[{'artifact_sha256':ref,'field':'failures'},{'artifact_sha256':ref,'field':'effects'}]})
        edges.append({'id':cid+':mapping','source':'language:valid_record_selection','target':cid,'relation':'行为合格条件下的组候选；完整范围未过门',
            'run_id':C.RUN,'evidence':{'level':'observation','artifact_sha256':ref}})
    graph=put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges})
    binding=put('binding',{'schema':'research_space_binding.v1','report_sha256':ref,'graph_sha256':graph,'traces':[],
        'context':{'study_type':'record_selection','native_bank':True,'model_dimensions':{'n_layers':36,'d_model':2560,'d_ff':9728,'n_heads':32}}})
    work=put('workpackage',{'schema':'research_workpackage.v1','title':'Qwen3-4B · 有效与过时记录：头与单元组检验',
        'stages':{'behavior':{'status':'done','passed':False},'prediction':{'status':'done','scope':'conditional diagnostics'},
            'native_effect':{'status':'done','families':384,'formal_promotion':False},'registration_validated':{'status':'done','report_sha256':ref,'graph_sha256':graph}}})
    reg={'run_id':C.RUN,'model_snapshot':C.SNAP,'report_sha256':ref,'graph_sha256':graph,'binding_sha256':binding,'workpackage_sha256':work}
    bundle=C.OUT/'ai2050-record-selection-v3.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=1) as tar:
        def add(name,raw):
            info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
        add('MANIFEST.json',canonical({'schema':'ai2050_research_upload.v1','objects':[{'sha256':h} for h in objects],'studies':[reg]}))
        for digest,raw in objects.items():add(f'objects/{digest}.json',raw)
    reg.update(bundle=str(bundle),bundle_sha256=C.sha(bundle),objects=len(objects),cards=len(cards),edges=len(edges),local_libraries=[str(lib.root) for lib in libs],remote_receipt=None)
    C.save('registration.json',reg);C.save('client-report.json',r);print(json.dumps(reg,ensure_ascii=False),flush=True)

if __name__=='__main__':
    import sys
    {'audit':audit,'publish':publish}[sys.argv[-1]]()
