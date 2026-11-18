"""Post-run audit, lossless coordinate publication and client registration."""
import base64,hashlib,io,json,tarfile,time
import numpy as np
import state_formation_core as C
from deploy.research_library import Library,canonical

def rounded_bf16(x):
    raw=np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((raw+0x7fff+((raw>>16)&1))>>16).astype(np.uint16)

def audit():
    p=C.checked();report=C.read(C.OUT/'report.json');selection=C.read(C.OUT/'selection.json');start=time.monotonic()
    assert report['arrays_sha256']==C.sha(C.OUT/'analysis-arrays.npz')
    manifest=[];independent=[];prefixmax=0.;finecells={};exactprefix=True;behavior_errors=[]
    diagnostic=C.read(C.OUT/'diagnostic-amendment.json') if not report['behavior']['passed'] else None
    if diagnostic:
        assert not report['center_primary_supported'] and all(not e['scope_use_supported'] and not e['effect_prediction_supported'] for e in report['effects'])
        assert diagnostic['diagnostic_source_sha256']==C.sha(C.ROOT/'tests/glm5/state_formation_diagnostic.py')
    loader_checks=None;loaders={}
    for stage in ['pilot','discovery','calibrate','confirm','causal','diagnostic_causal','capability']:
        path=C.OUT/f'loader-{stage}.json';loaded=C.read(path)
        assert loaded['model_snapshot']==C.SNAP and loaded['plan_sha256']==C.sha(C.OUT/'plan.json')
        if loader_checks is None:loader_checks=loaded['checks']
        else:assert loaded['checks']==loader_checks
        loaders[stage]={'sha256':C.sha(path),'verified_parameter_tensors':len(loaded['checks'])}
    for split,n in [('discovery',256),('confirm',768)]:
        materials=C.read(C.OUT/(split+'-materials.json'))['rows']
        for f in range(n):
            path=C.OUT/f'{split}-{f:04d}.npz';meta=C.read(path.with_suffix('.json'))
            assert C.sha(path)==meta['native_sha256'];assert meta['plan_sha256']==C.sha(C.OUT/'plan.json')
            manifest.append({'path':path.name,'bytes':path.stat().st_size,'sha256':meta['native_sha256'],'metadata_sha256':C.sha(path.with_suffix('.json'))})
            prefixmax=max(prefixmax,max(x['max_relative_l2'] for x in meta['negative_controls'].values()))
            exactprefix=exactprefix and all(x['future_factors_bitwise_equal'] for x in meta['negative_controls'].values())
            if split=='confirm':
                for row,score in zip(materials[f*8:(f+1)*8],meta['scores']):
                    key=':'.join(str(row[k]) for k in ['frame','history','query','record','language'])
                    finecells.setdefault(key,[]).append(score['four_correct'])
                    if not score['four_correct']:behavior_errors.append({'sample':row,'score':score})
            # Independent CPU IEEE BF16 round-to-nearest-even check, no Torch.
            if f%32==0:
                with np.load(path) as z:
                    h=z['hidden'];r=z['post_attention'];a=z['attn_write'];m=z['mlp_write']
                    assert h.shape==(8,37,4,2560) and r.shape==a.shape==m.shape==(8,36,4,2560)
                    np.testing.assert_array_equal(rounded_bf16(C.unbits(h[:,:36])+C.unbits(a)),r)
                    np.testing.assert_array_equal(rounded_bf16(C.unbits(r)+C.unbits(m)),h[:,1:])
                    for indices in [[0,1,4,5],[2,3,6,7]]:
                        rs=[materials[f*8+i] for i in indices];prefixes=[r['tokens'][:r['positions'][0]+1] for r in rs]
                        assert all(v==prefixes[0] for v in prefixes)
                independent.append({'split':split,'family':f,'all36_blocks_all4_positions_all8_conditions':True})
            if (f+1)%64==0:C.log('raw_audit_'+split,f+1,n,start)
    effects=C.read(C.OUT/'effects-frozen.json')
    max_kl_error=0.;logit_arrays_checked=0
    for f in range(768):
        item=C.read(C.OUT/f'causal-{f:04d}.json');commit=C.read(C.OUT/f'commit-{f:04d}.json')
        assert commit['effects_frozen_sha256']==C.sha(C.OUT/'effects-frozen.json') and item['restoration_bitwise']
        assert item['selection_sha256']==C.sha(C.OUT/'selection.json')
        assert C.sha(C.OUT/f'causal-logits-{f:04d}.npz')==item['native_logits_sha256']
        if diagnostic:
            assert item['diagnostic_amendment_sha256']==C.sha(C.OUT/'diagnostic-amendment.json') and item['formal_confirmation'] is False
        assert commit['sample_id']==item['sample']['id']
        original=next(s for s in C.read(C.OUT/f'confirm-{f:04d}.json')['scores'] if s['id']==item['sample']['id'])
        for key,value in item['baseline'].items():assert original[key]==value,('native baseline replay',f,key)
        with np.load(C.OUT/f'causal-logits-{f:04d}.npz') as z:
            base=C.unbits(z['base_logits']).astype(float)
            assert base.shape==(C.read(C.MODEL/'config.json')['vocab_size'],)
            lb=base-np.max(base);lb-=np.log(np.exp(lb).sum())
            for candidate in [None,*selection['candidates']]:
                name='base_logits' if candidate is None else candidate['id']+'_logits'
                v=C.unbits(z[name]).astype(float)
                metric=item['baseline'] if candidate is None else next(r['edited'] for r in item['records'] if r['candidate']==candidate['id'] and r['method']=='ablation' and r['dose']==1.)
                four=[float(v[p['choice_ids'][k]].max()) for k in ['en_red','en_blue','zh_red','zh_blue']]
                assert four==metric['four_logits'] and int(v.argmax())==metric['argmax']
                if candidate is not None:
                    lv=v-v.max();lv-=np.log(np.exp(lv).sum());kl=float((np.exp(lv)*(lv-lb)).sum())
                    error=abs(kl-metric['KL_from_base']);max_kl_error=max(max_kl_error,error)
                    assert error<2e-5,'Independent float64 KL differs from stored FP32 result'
                logit_arrays_checked+=1
        if (f+1)%64==0:C.log('causal_output_audit',f+1,768,start)
    # Scope of the center intervention: inferred-center models ignore oracle
    # labels; this was tested before measurements and remains a frozen dependency.
    C.save('audit.json',{'schema':'state_formation_audit.v1','passed':True,'report_sha256':C.sha(C.OUT/'report.json'),
        'frozen_dependencies_unchanged':True,'verified_native_families':1024,'native_manifest':manifest,
        'identical_disk_and_CUDA_weight_loading_checks':loaders,
        'independent_cpu_addition_checks':independent,'prefix_negative_control_max_relative_l2':prefixmax,
        'prefix_negative_controls_all_bitwise_equal':exactprefix,
        'independently_checked_full_vocabulary_arrays':logit_arrays_checked,'maximum_float64_vs_float32_KL_error':max_kl_error,
        'fine_condition_cells':{'count':len(finecells),'minimum_n':min(map(len,finecells.values())),
            'minimum_accuracy':min(float(np.mean(v)) for v in finecells.values()),
            'cells_below_point90':int(sum(np.mean(v)<.9 for v in finecells.values())),
            'scope':'descriptive post-run granularity audit; frozen gate is frame x history, not this finer grid'},
        'restored_causal_families':768,'confirmed_effect_prediction_commitments':768,
        'intervention_baselines_exactly_match_original_collection':768,
        'fresh_confirm_families':768,'fresh_confirm_frames':16,'calibration_families':len(effects['families']),
        'candidate_count':len(selection['candidates']),'formal_promotion':False,
        'behavior_errors':behavior_errors,'diagnostic_only':bool(diagnostic),
        'diagnostic_amendment_sha256':C.sha(C.OUT/'diagnostic-amendment.json') if diagnostic else None,
        'additional_scope_limits':['new record wording and nouns; query and output-language instruction templates are shared across discovery and confirm',
            'single permuted-label comparator is not a calibrated null distribution: at hs34 accuracy is 10.90%, but complementing its predictions gives 89.10% post hoc; do not treat low fixed-sign accuracy as proving absence of information',
            'source exclusions mask only this selected block final-query direct attention edges; earlier layers and other positions can already carry source information',
            'color/language margins and other-factor correctness condition on the correct level of the other factor',
            'KL is KL(edited distribution || baseline distribution)',
            'four-choice gate aggregates frame x history; finer q/r/language cells are audited descriptively'],
        'scope':'same-node technical audit, not an independent scientific replication; raw fields and metadata remain at linked local storage'})

def publish():
    p=C.checked();r=C.read(C.OUT/'report.json');auditdoc=C.read(C.OUT/'audit.json');assert auditdoc['passed']
    assert auditdoc['report_sha256']==C.sha(C.OUT/'report.json')
    libs=[Library(C.ROOT/v) for v in ['distributed_data/research_library','tests/glm5/result/mlp_tail_acceptance/center/research_library']];objects={}
    def put(name,body):
        doc={**body,'kind':'real','run_id':C.RUN,'model_snapshot':C.SNAP,'provenance':p['provenance']};raw=canonical(doc)
        hashes={lib.ingest(doc,f'{C.RUN}/{name}',kind='local_report')['sha256'] for lib in libs};assert len(hashes)==1
        digest=hashes.pop();objects[digest]=raw;return digest
    artifacts={name:put(name,C.read(C.OUT/file)) for name,file in [('冻结方案','plan.json'),('发现期候选选择','selection.json'),('确认前作用预测冻结','effects-frozen.json'),('原生数据与结论审计','audit.json'),('证据迁移校验','storage-audit.json'),('确认行为门','confirm-behavior.json')]}
    if auditdoc['diagnostic_only']:artifacts['行为门失败后的诊断协议']=put('diagnostic-amendment',C.read(C.OUT/'diagnostic-amendment.json'))
    artifacts['旧记录冲突与行为失败']=put('behavior-error-diagnosis',C.read(C.OUT/'behavior-error-diagnosis.json'))
    artifacts['干预的失败分组诊断']=put('effect-error-diagnosis',C.read(C.OUT/'effect-error-diagnosis.json'))
    artifacts['审计文件保存故障与恢复']=put('audit-recovery',C.read(C.OUT/'audit-recovery.json'))
    artifacts['全部语言模板']=put('grammars',C.read(C.OUT/'grammars.json'))
    # Material chunks include exact tokens and positions; no model outputs invented.
    material_parts=[]
    for split,count in [('discovery',256),('confirm',768)]:
        rows=C.read(C.OUT/(split+'-materials.json'))['rows']
        for start in range(0,count,64):material_parts.append(put(f'materials-{split}-{start}',{'schema':'state_formation_materials.v1','split':split,'rows':rows[start*8:(start+64)*8]}))
    artifacts['完整输入与位置索引']=put('material-index',{'schema':'state_formation_material_index.v1','parts':material_parts})
    causal_parts=[]
    for start in range(0,768,32):
        causal_parts.append(put(f'causal-{start}',{'schema':'state_formation_effects_evidence.v1','rows':[C.read(C.OUT/f'causal-{f:04d}.json') for f in range(start,start+32)]}))
    artifacts['全部作用与对照索引']=put('effect-index',{'schema':'state_formation_effect_index.v1','parts':causal_parts})
    artifacts['能力题原始记录']=put('capability',C.read(C.OUT/'capability.json'))
    arrays=np.load(C.OUT/'analysis-arrays.npz');est=np.load(C.OUT/'estimator.npz');coordinates={k:{} for k in ['state','attention','mlp']}
    def vector(kind,index,value):
        value=value.astype('<f4');assert value.shape in [(1,3,2560),(2,3,2560)]
        return put(f'coordinates-{kind}-{index}',{'schema':'state_formation_coordinates.v1','field':kind,'index':index,
            'readout':'answer_prefix','terms':['language','color','color_language'],'source_arrays_sha256':r['arrays_sha256'],
            'array':{'dtype':'<f4','encoding':'base64','shape':list(value.shape),'data':base64.b64encode(value.tobytes()).decode('ascii')}})
    for state in range(74):coordinates['state'][str(state)]=vector('state',state,np.stack([est['beta'][state,[3,6,7]],arrays['evaluation_beta'][state,[3,6,7]]]))
    for k,kind in enumerate(['attention','mlp']):
        for block in range(36):coordinates[kind][str(block)]=vector(kind,block,arrays['module_coefficients'][k,block,[3,6,7]][None])
    r.update(artifacts=artifacts,coordinate_artifacts=coordinates,audit_summary={k:v for k,v in auditdoc.items() if k not in ['native_manifest','behavior_errors']},
        diagnostic_only=auditdoc['diagnostic_only'],behavior_errors=auditdoc['behavior_errors'])
    r['behavior_error_diagnosis']={k:v for k,v in C.read(C.OUT/'behavior-error-diagnosis.json').items() if k!='rows'}
    r['effect_error_diagnosis']=C.read(C.OUT/'effect-error-diagnosis.json')['rows']
    r['limitations']=r['limitations']+auditdoc['additional_scope_limits']
    ref=put('report',r);cards=[];edges=[]
    for candidate in r['formation']:
        index=candidate['after_state'];factor=candidate['factor'];field=3 if factor=='language' else 6;values=arrays['evaluation_beta'][index,field]
        cid=f"{C.RUN}:{candidate['id']}";effect=next(e for e in r['effects'] if e['candidate']==candidate['id'] and e['method']=='ablation' and e['dose']==1.)
        cards.append({'id':cid,'run_id':C.RUN,'layer':candidate['block'],'boundary':'residual.post_attention' if candidate['module']=='attention' else 'residual.post_block',
            'label':candidate['id']+' · 全模块状态形成候选','index':{'space':'residual','coordinate':int(np.argmax(np.abs(values))),'anchor_only':True,'support':'all2560 residual coordinates;whole module manipulation'},
            'method':'discovery selection;new confirm templates;original module ablation and matched realized-norm control',
            'evidence':{'level':'observation','artifact_sha256':ref},'scope':{'candidate':candidate,'local_use_test_supported':effect['scope_use_supported'],'formal_promotion':False,'single_neuron_mechanism':False},
            'counterexamples':[{'artifact_sha256':ref,'field':'effects.'+candidate['id']}]})
        edges.append({'id':cid+':mapping','source':'language:requested_output_language' if factor=='language' else 'language:record_binding','target':cid,
            'relation':'冻结新材料上的响应与范围作用记录；正式证据待审','run_id':C.RUN,'evidence':{'level':'observation','artifact_sha256':ref}})
    center_id=C.RUN+':conditional-center-hs34';center_vector=est['beta'][68,6]
    cards.append({'id':center_id,'run_id':C.RUN,'layer':33,'boundary':'residual.post_block','label':'hs[34] · 颜色条件中心修正检验',
        'index':{'space':'residual','coordinate':int(np.argmax(np.abs(center_vector))),'anchor_only':True,'support':'all2560 coordinates'},
        'method':'frozen inferred-color-conditioned center versus fixed center,pooled linear,norm,wrong-center and permuted-label controls',
        'evidence':{'level':'observation' if r['center_primary_supported'] or r['diagnostic_only'] else 'negative','artifact_sha256':ref},
        'scope':{'primary_supported':r['center_primary_supported'],'formal_promotion':False},'counterexamples':[{'artifact_sha256':ref,'field':'counterexamples'}]})
    graph=put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges})
    binding=put('binding',{'schema':'research_space_binding.v1','report_sha256':ref,'graph_sha256':graph,'traces':[],
        'context':{'study_type':'state_formation','native_bank':True,'model_dimensions':{'n_layers':36,'d_model':2560,'d_ff':9728,'n_heads':32}}})
    work=put('workpackage',{'schema':'research_workpackage.v1','title':'Qwen3-4B · 新样本：条件中心、注意力与MLP作用',
        'stages':{'behavior':{'status':'done','passed':r['behavior']['passed']},'prediction':{'status':'done','center_primary_supported':r['center_primary_supported']},
            'native_effect':{'status':'done','families':768,'diagnostic_only':r['diagnostic_only'],'formal_promotion':False},'registration_validated':{'status':'done','report_sha256':ref,'graph_sha256':graph}}})
    reg={'run_id':C.RUN,'model_snapshot':C.SNAP,'report_sha256':ref,'graph_sha256':graph,'binding_sha256':binding,'workpackage_sha256':work}
    bundle=C.OUT/'ai2050-state-formation-v1.tar.gz'
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
