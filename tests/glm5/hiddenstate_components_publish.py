"""Independent arithmetic checks and local publication of retrospective state fields.

The frozen estimator is never modified. This audit is post hoc and adds no
independent scientific confirmation; direction/offset diagnostics are descriptive.
"""
import base64,io,json,shutil,tarfile
import numpy as np
import hiddenstate_components as H
from deploy.research_library import Library,canonical


def audit():
    p=H.checked();r=H.read(H.OUT/'report.json');est=H.read(H.OUT/'estimator.json')
    assert r['plan_sha256']==H.sha(H.OUT/'plan.json')
    assert r['estimator_sha256']==H.sha(H.OUT/'estimator.json')
    assert r['arrays_sha256']==H.sha(H.OUT/'analysis-arrays.npz')
    assert est['coefficients_sha256']==H.sha(H.OUT/'frozen-coefficients.npz')
    assert len(r['rows'])==74 and r['unique_inputs']==5120 and r['verified_native_files']==640
    z=np.load(H.OUT/'analysis-arrays.npz');f=np.load(H.OUT/'frozen-coefficients.npz')
    b=f['beta'].astype(float);var=f['variance'].astype(float);observed=z['evaluation_beta'].astype(float)
    assert b.shape==observed.shape==(37,2,8,2560)
    mats=H.read(H.SOURCE/'holdout-materials.json')
    frames={}
    for i in range(512):
        row=mats[i*8];frames.setdefault(i%4,set()).add(str(row['frame']))
    assert all(len(v)==1 for v in frames.values()) and len({next(iter(v)) for v in frames.values()})==4
    checks=[]
    for family in [0,22,175,511]:
        with np.load(H.SOURCE/f'holdout-{family:04d}.npz') as native:raw=H.unbits(native['hidden']).astype(float)
        rows=mats[family*8:family*8+8]
        for layer in [0,15,24,34,35,36]:
            for ro in [0,1]:
                h=raw[:,layer,ro];coeff=np.zeros((8,2560))
                terms=[]
                for j,row in enumerate(rows):
                    q,v,lang=row['query'],row['record'],row['language'];terms.append([1,q,v,lang,q*lang,v*lang,q*v,q*v*lang])
                    for k,t in enumerate(terms[-1]):coeff[k]+=t*h[j]/8
                # Parseval identity checks both sign order and orthogonal decomposition.
                reconstructed=np.array([sum(t*coeff[k] for k,t in enumerate(x)) for x in terms])
                np.testing.assert_allclose(reconstructed,h,atol=1e-10)
                np.testing.assert_allclose(np.mean(h*h),np.sum(coeff*coeff)/2560,atol=1e-10)
                for method,columns in enumerate([[0],[0,1,2,3],[0,3,6,7],list(range(8))]):
                    predicted=np.array([sum(x[k]*b[layer,ro,k] for k in columns) for x in terms])
                    error=np.mean((h-predicted)**2)
                    np.testing.assert_allclose(error,z['family_prediction_mse'][family,layer,ro,method],rtol=3e-6,atol=1e-8)
        checks.append({'family':family,'layers':[0,15,24,34,35,36],'readouts':2,'parseval':True,'prediction_mse':True})
    for row in r['rows']:
        l=row['hidden_index'];ro=H.READOUTS.index(row['readout'])
        if ro==0:assert row['probes']['color']==row['probes']['language']==.5
        if l==0 and ro==1:assert all(v==0 for v in row['prediction'].values())
        if l in [35,36]:assert row['step'] is None
        if l==36:assert all(c['contrast_increment_norm'] is None for c in row['components'].values())
        for key,value in zip(['constant','additive','four_state','eight_cell'],z['family_prediction_mse'][:,l,ro].astype(float).mean(0)):
            np.testing.assert_allclose(row['prediction'][key],value,rtol=1e-6,atol=1e-10)
    # Cross-color language decoder fails if a transferred origin shifts the score
    # past its threshold; failure does not imply absent information or reversed axes.
    diagnostics=[]
    for l in [24,25,32,33,34,35,36]:
        red=b[l,1,3]+b[l,1,7];blue=b[l,1,3]-b[l,1,7]
        for sign in [-1,1]:
            direction=b[l,1,3]+sign*b[l,1,7];weight=direction/np.maximum(var[l,1],1e-6);weight/=direction@weight
            source_center=b[l,1,0]+sign*b[l,1,6]
            target_center=observed[l,1,0]-sign*observed[l,1,6]
            target_direction=observed[l,1,3]-sign*observed[l,1,7]
            offset=float((target_center-source_center)@weight);amplitude=float(target_direction@weight)
            diagnostics.append({'hidden_index':l,'trained_color':sign,'evaluated_color':-sign,
                'direction_cosine_red_blue':float(red@blue/(np.linalg.norm(red)*np.linalg.norm(blue))),
                'mean_target_offset':offset,'mean_target_language_amplitude':amplitude,
                'mean_english_score':offset+amplitude,'mean_chinese_score':offset-amplitude,
                'scope':'post-hoc means, not corrected accuracy; pooled discovery variance used for all conditional probes'})
    # Original score sidecars were not explicitly hashed in the new pre-analysis
    # manifest. Preserve that gap; record them now, never backdate the freeze.
    score_manifest=[]
    for i in range(512):
        path=H.SOURCE/f'holdout-{i:04d}.json';doc=H.read(path)
        assert doc['plan_sha256']==H.sha(H.SOURCE/'plan.json')
        score_manifest.append({'path':path.name,'sha256':H.sha(path)})
    body={'schema':'hiddenstate_components_audit.v1','kind':'real','run_id':H.RUN,'model_snapshot':p['model_snapshot'],
        'plan_sha256':H.sha(H.OUT/'plan.json'),'report_sha256':H.sha(H.OUT/'report.json'),
        'script_sha256':H.sha(__file__),'frozen_dependencies_unchanged':True,'verified_native_files':640,
        'independent_arithmetic_spot_checks':checks,'checks_passed':True,'source_frame_mapping':{str(k):list(v) for k,v in frames.items()},
        'language_transfer_diagnostics':diagnostics,'posthoc_score_sidecars':score_manifest,
        'limits':['Same node arithmetic audit, not independent replication.',
            'Shared variance uses both colors/languages in discovery; conditional probes transfer direction/center, not a completely unexposed condition.',
            'Mean score offsets diagnose aggregate geometry, not individual-example corrected classification.',
            '512 old score JSON sidecars were not frozen by individual hashes in this analysis; endpoint calibration is retrospective and these hashes are post hoc.',
            'No confidence intervals or significance claims across only four confounded sentence frames.',
            'Original acquisition admissibility gaps remain. New formal confirmation count is zero.'],
        'formal_evidence_promotion':False}
    H.save('audit.json',body,p['model_snapshot']);print(json.dumps({'audit':'passed','diagnostics':diagnostics},ensure_ascii=False),flush=True)


def publish():
    p=H.checked();r=H.read(H.OUT/'report.json');auditdoc=H.read(H.OUT/'audit.json');assert auditdoc['checks_passed']
    assert auditdoc['report_sha256']==H.sha(H.OUT/'report.json')
    assert shutil.disk_usage(H.ROOT).free>48*2**20
    libs=[Library(H.ROOT/v) for v in ['distributed_data/research_library','tests/glm5/result/mlp_tail_acceptance/center/research_library']]
    objects={}
    def put(name,body):
        doc={**body,'kind':'real','run_id':H.RUN,'model_snapshot':p['model_snapshot'],'provenance':p['provenance']}
        raw=canonical(doc);hashes={lib.ingest(doc,f'{H.RUN}/{name}',kind='local_report')['sha256'] for lib in libs};assert len(hashes)==1
        digest=hashes.pop();objects[digest]=raw;return digest
    artifacts={name:put(name,H.read(H.OUT/file)) for name,file in [('冻结方案','plan.json'),('发现集拟合冻结','estimator.json'),('数值与结论审计','audit.json')]}
    artifacts['原始材料与旧档案索引']=put('source-index',{'schema':'hiddenstate_components_sources.v1',
        'source_run':r['source_run'],'source_registration':H.read(H.SOURCE/'registration.json'),
        'source_plan_sha256':H.sha(H.SOURCE/'plan.json'),'source_audit_sha256':H.sha(H.SOURCE/'audit.json'),
        'materials':p['sources'],'raw_native_index':'冻结方案.native_files','local_only':True})
    arrays=np.load(H.OUT/'analysis-arrays.npz');coordinates={}
    for layer in range(37):
        value=np.stack([arrays[k][layer][:,[3,6,7],:] for k in ['discovery_beta','evaluation_beta']]).astype('<f4')
        assert value.shape==(2,2,3,2560)
        coordinates[str(layer)]=put(f'coordinates-{layer}',{'schema':'hiddenstate_components_coordinates.v1','hidden_index':layer,
          'axis_order':['discovery_or_exposed_evaluation','record_end_or_answer_prefix','language_or_color_or_interaction','native_coordinate'],
          'source_arrays_sha256':r['arrays_sha256'],
          'array':{'encoding':'base64','dtype':'<f4','shape':list(value.shape),'data':base64.b64encode(value.tobytes()).decode('ascii')}})
    r.update(artifacts=artifacts,coordinate_artifacts=coordinates,audit=auditdoc,
        interpretation='同一末端位置上，输出语言从hs[15]、查询颜色从hs[24]达到预定可读出标准；尾层交互增强，但固定阈值的跨颜色语言读出会失败。可读出、完整状态预测与原模型作用分开记账。',
        limits=[*r['limits'],*auditdoc['limits']])
    ref=put('report',r);cards=[];edges=[]
    for factor,layers in [('language',[15,33,34,35]),('color',[24,25,33,35]),('color_language',[25,33,35])]:
        for layer in layers:
            row=next(row for row in r['rows'] if row['hidden_index']==layer and row['readout']=='answer_prefix');c=row['components'][factor]
            cid=f'{H.RUN}:hs{layer}:{factor}'
            cards.append({'id':cid,'run_id':H.RUN,'layer':layer-1,'boundary':'residual.post_block',
                'label':f'hs[{layer}] · '+{'color':'查询颜色','language':'输出语言','color_language':'颜色×语言交互'}[factor]+' · 全坐标对照',
                'index':{'space':'residual','coordinate':c['dominant_native_coordinates'][0],'anchor_only':True,'support':'all 2560 coordinates'},
                'method':'balanced factorial contrast, discovery-fitted decoding, previously exposed evaluation',
                'evidence':{'level':'observation','artifact_sha256':ref},
                'scope':{'hidden_index':layer,'readout':'answer_prefix','vector_sha256':coordinates[str(layer)],'factor':factor,'formal_promotion':False,'single_coordinate_causality':False},
                'counterexamples':[{'artifact_sha256':ref,'field':'audit.language_transfer_diagnostics' if factor=='language' else 'counterexamples'}]})
            edges.append({'id':cid+':association','source':'language:requested_output_language' if factor=='language' else 'language:record_binding','target':cid,
                'relation':'八条件响应关联；非因果边','run_id':H.RUN,'evidence':{'level':'observation','artifact_sha256':ref}})
    graph=put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges})
    binding=put('binding',{'schema':'research_space_binding.v1','report_sha256':ref,'graph_sha256':graph,'traces':[],
        'context':{'study_type':'hiddenstate_components','native_bank':True,'model_dimensions':{'n_layers':36,'d_ff':9728,'d_model':2560,'n_heads':32}}})
    work=put('workpackage',{'schema':'research_workpackage.v1','title':'Qwen3-4B · 全层HiddenState：成分形成与状态更新',
        'stages':{'retrospective_analysis':{'status':'done','native_files':640,'new_blind_confirmations':0},
            'native_effect':{'status':'not_tested'},'registration_validated':{'status':'done','report_sha256':ref,'graph_sha256':graph}}})
    reg={'run_id':H.RUN,'model_snapshot':p['model_snapshot'],'report_sha256':ref,'graph_sha256':graph,'binding_sha256':binding,'workpackage_sha256':work}
    assert sum(map(len,objects.values()))<10*2**20
    bundle=H.OUT/'ai2050-hiddenstate-components-v1.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=5) as tar:
        def add(name,raw):
            info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
        add('MANIFEST.json',canonical({'schema':'ai2050_research_upload.v1','objects':[{'sha256':h} for h in objects],'studies':[reg]}))
        for digest,raw in objects.items():add(f'objects/{digest}.json',raw)
    reg.update(bundle=str(bundle),bundle_sha256=H.sha(bundle),objects=len(objects),cards=len(cards),edges=len(edges),
        local_libraries=[str(lib.root) for lib in libs],remote_receipt=None,formal_evidence_promotion=False)
    H.save('client-report.json',r,p['model_snapshot']);H.save('registration.json',reg,p['model_snapshot'])
    print(json.dumps(reg,ensure_ascii=False),flush=True)


if __name__=='__main__':
    import sys
    {'audit':audit,'publish':publish}[sys.argv[-1]]()
