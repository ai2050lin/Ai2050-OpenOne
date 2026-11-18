"""Publish immutable condition cards, full selected evidence and research bindings."""
import io
import base64
import json
from pathlib import Path
import tarfile
import numpy as np
import mlp_condition_design as D
from deploy.research_library import Library,canonical
from shared.dist_protocol.mechanism_trace import unbits
from tools import provenance

def main():
    plan=D.checked();report=D.READ(D.OUT/'report.json');objects={};revision=D.RUN.rsplit('-',1)[-1];public=D.ROOT/f'research/mlp_tail/conditional_{revision}'
    libraries=[Library(D.ROOT/p) for p in ('distributed_data/research_library','tests/glm5/result/mlp_tail_acceptance/center/research_library')]
    def put(name,body):
        doc={**body,'kind':'real','run_id':D.RUN,'model_snapshot':D.SNAPSHOT,'provenance':plan['provenance']}
        raw=canonical(doc);assert len(raw)<16*2**20,(name,len(raw))
        shas={lib.ingest(doc,f'{D.RUN}/{name}',kind='local_report')['sha256'] for lib in libraries};assert len(shas)==1
        sha=shas.pop();objects[sha]=raw
        # Keep only the concise registration and report beside code; CAS stores the other objects.
        return sha
    inputs=[]
    for split in ('discovery','holdout'):
        fs=D.families(plan,split)
        for offset in range(0,len(fs),16):
            batch=fs[offset:offset+16];evidence=[]
            for f in batch:
                path=D.OUT/f'{split}-{f["family"]:04d}.npz';meta=D.READ(path.with_suffix('.json'))
                assert meta['sha256']==D.HASH(path)
                with np.load(path) as raw:values=unbits(raw['selected']).tolist()
                evidence.append({'family':f,'selected_GVA':values,'observations':meta,
                                 'layout':'variant × [B29/U17,B30/U2,B33/U1521,B34/U6097] × [G,V,A] × 6 prefix positions'})
            inputs.append(put(f'{split}-{offset:04d}',{'schema':'mlp_condition_samples.v1','split':split,'families':evidence}))
    relative=D.ROOT/'tests/glm5/result/mlp_tail_relative_encoding_v1'
    summary=D.READ(relative/'report.json');values=np.load(relative/'per_unit.npz');layers={}
    for b in range(28,36):
        a=values[f'B{b}'];tested={}
        for label,src,u,dst,v in D.PAIRS:
            if b in (src,dst):
                obs=report['observations'][label];tested[str(u if b==src else v)]={
                    'family':label,'role':'source' if b==src else 'target','prediction_passed':obs['conditional']['passed'],
                    'rare_families':obs['rare']['families'],'rare_powered':obs['rare']['powered']}
        assert a.shape==(9728,5) and np.isfinite(a).all()
        layers[str(b)]=put(f'condition-layer-{b}',{'schema':'mlp_condition_layer.v1','block':b,'width':9728,
          'arrays':{key:a[:,i].tolist() for i,key in enumerate(('within_token_variance_fraction','token_predictor_fvu','sign_mix','negative_gate_energy','double_negative_energy'))},
          'tested_units':tested,'source_report_sha256':D.HASH(relative/'report.json'),'source_array_sha256':D.HASH(relative/'per_unit.npz'),
          'scope':'旧归档的探索性条件响应：同token方差、FVU与符号混合用57种重复token、2185个旧留出位置、509篇文档；负门控和双负A²占比用完整旧留出512篇×16位置。上下文还包含主题、位置和范数等因素。新受控实验仅涉及明确标出的4个单元；其他单元没有因此获得语义确认。'})
    params={}
    for _,src,u,dst,v in D.PAIRS:
        for b,j in ((src,u),(dst,v)):
            folder=D.ROOT/('tests/glm5/result/mlp_tail_goodfire_v1' if b>=32 else 'tests/glm5/result/mlp_tail_b28_b31_v1')
            assert D.READ(folder/'manifest.json')['model_snapshot']==D.SNAPSHOT
            params[f'B{b}/U{j}']={}
            for field in ('gate','up','down'):
                p=folder/f'B{b}-W-{field}.npy';a=np.load(p,mmap_mode='r')
                params[f'B{b}/U{j}'][field]={'values':unbits(a[:,j] if field=='down' else a[j]).tolist(),'source_sha256':D.HASH(p)}
    parameter_sha=put('native-parameters',{'schema':'mlp_condition_parameters.v1','parameters':params,'scope':'完整原生 gate行、up行、down列；每条2560个BF16参数，无投影压缩'})
    causal=[]
    for offset in range(0,D.HOLDOUT,32):
        records=[]
        for i in range(offset,min(offset+32,D.HOLDOUT)):
            row=D.READ(D.OUT/f'causal-{i:04d}.json')
            row['prediction_commits']={label:D.READ(D.OUT/f'prediction-{i:04d}-{label}.json') for label,*_ in D.PAIRS}
            records.append(row)
        causal.append(put(f'causal-{offset:04d}',{'schema':'mlp_condition_causal_evidence.v1','records':records}))
    calibration=np.load(D.OUT/'discovery-calibration.npz');calrefs=[]
    for p,(label,*_) in enumerate(D.PAIRS):
        # Split each full vocabulary vector to keep every immutable object bounded.
        for stratum in range(4):
            vector=calibration['logit_slope'][p,stratum].astype('<f4')
            encoded=base64.b64encode(vector.tobytes()).decode('ascii')
            assert np.array_equal(np.frombuffer(base64.b64decode(encoded),dtype='<f4'),vector)
            calrefs.append(put(f'predictor-{label}-{stratum}',{'schema':'mlp_condition_output_predictor.v1','pair':label,'stratum':stratum,
                'logit_slope':{'encoding':'base64','dtype':'<f4','shape':list(vector.shape),'data':encoded},
                'target_slope':float(calibration['target_slope'][p,stratum])}))
    prereg=put('preregistration',plan);selection=put('selection',{'schema':'mlp_condition_selection.v1','selection':D.READ(D.OUT/'selection.json')})
    loader=put('loader-audit',D.READ(D.OUT/'loader-audit.json'))
    controls=put('matched-control-calibration',D.READ(D.OUT/'matched-control-calibration.json'))
    cap_power=put('capability-power-addendum',D.READ(D.OUT/'capability-power-addendum.json'))
    cap=put('capability-evidence',{'schema':'mlp_condition_capability.v1','measurements':D.READ(D.OUT/'capability.json'),
        'materials':D.READ(D.OUT/'capability-materials.json')})
    manifest=[]
    for path in sorted(D.OUT.iterdir()):
        if path.is_file() and path.suffix in ('.json','.npz','.jsonl','.log') and path.name not in ('report.json','registration.json','local-acceptance.json','server-acceptance.json') and not path.name.startswith('publish'):
            manifest.append({'path':path.name,'bytes':path.stat().st_size,'sha256':D.HASH(path)})
    relative_ref=put('historical-relative-observations',{'schema':'mlp_condition_prior_observations.v1','summary':summary,
        'source_sha256':D.HASH(relative/'report.json'),'scope':'retrospective archived observations, not new heldout confirmation'})
    prior_ref=None
    if plan.get('prior_failure'):
        previous=D.OUT.with_name('mlp_condition_v1');prior=D.READ(previous/'selection.json');examples=[];seen=set()
        for f in D.READ(previous/'discovery-materials.json')['families']:
            meta=D.READ(previous/f'discovery-{f["family"]:04d}.json')
            for row,record in zip(f['rows'],meta['rows']):
                key=(row['frame'],row['role'],row['mapping'])
                if np.argmax(record['scores'])!=row['gold'] and key not in seen:
                    seen.add(key);examples.append({'material':row,'observation':record})
        prior_ref=put('prior-behavior-failure',{'schema':'mlp_condition_prior_failure.v1','selection':prior,'examples':examples,
            'holdout_unopened':not list(previous.glob('holdout-[0-9]*.npz')),'scope':'v1 stopped before confirmation; v2 uses newly authored materials'})
    report.update(condition_layers=layers,material_artifacts=inputs,causal_artifacts=causal,
                  output_predictor_artifacts=calrefs,native_parameter_sha256=parameter_sha,
                  capability_evidence_sha256=cap,loader_audit_sha256=loader,selection_artifact_sha256=selection,
                  control_calibration_sha256=controls,capability_power_sha256=cap_power,
                  historical_observations_sha256=relative_ref,prior_failure_sha256=prior_ref,raw_manifest=manifest)
    report['interpretation_limits']=[
        '512 authored text families use four heldout role-assignment frames and repeated background paragraphs; they are not 512 independent natural documents or broad semantic tasks.',
        'The frozen numerical baseline includes identity/role/position/mapping, readout status and discovery token indicators. The plan prose also mentions frame indicators, but the frozen implementation does not include them. Numeric gates describe this actual baseline; no formal semantic or causal evidence promotion is granted from these gates.',
        'The same source activation is observed at prediction time. Intervention forecasts also use baseline logits and state; neither result is graph-only independent generation.',
        'Only final-token source writes and recomputed suffixes are tested. This experiment does not establish earlier-token KV-history mediation.',
        'Capability preservation covers the primary 1-SD dose, both signs, on the existing 2200-item keyed battery; it does not certify 2-SD edits or general free generation.',
        'One physical node performed these runs; independent replication and external semantic review remain pending.'
    ]
    report['formal_evidence_promotion']=False
    report_ref=put('report',report);cards=[];edges=[]
    for label,src,u,dst,v in D.PAIRS:
        obs=report['observations'][label]
        for b,j in ((src,u),(dst,v)):
            cards.append({'id':f'{D.RUN}:B{b}:U{j}','run_id':D.RUN,'layer':b,'boundary':'mlp.down_proj.input',
              'label':f'B{b}/U{j} · 条件响应卡','index':{'space':'mlp','unit':j},'method':'预注册16条件交叉、6个可见前缀位置、固定原生单元',
              'evidence':{'level':'observation','artifact_sha256':report_ref},
              'scope':{'family':label,'conditions':['word identity','role','evidence position','answer mapping'],
                       'layer_artifact_sha256':layers[str(b)],'parameter_artifact_sha256':parameter_sha},
              'counterexamples':[{'artifact_sha256':report_ref,'field':'prediction_counterexamples'}]})
        primary=next(c for c in report['dose_curve'] if c['pair']==label and c['dose']==D.PRIMARY)
        edges.append({'id':f'{D.RUN}:{label}:prediction','source':f'{D.RUN}:B{src}:U{u}','target':f'{D.RUN}:B{dst}:U{v}',
              'run_id':D.RUN,'relation':'条件预测与有限写入检验（分别计分）',
              'evidence':{'level':'observation' if obs['conditional']['passed'] else 'negative','artifact_sha256':report_ref},
              'scope':{'prediction':obs,'local_intervention':primary,'behavior_mediation_passed':primary['mediation']['passed'],
                       'interpretation':'预测关联不是因果边；原模型响应、行为翻转和救援按各自判据回查'}})
    graph=put('graph',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':edges})
    binding=put('binding',{'schema':'research_space_binding.v1','report_sha256':report_ref,'graph_sha256':graph,'traces':[],
        'context':{'study_type':'mlp_condition','native_bank':True,'model_dimensions':{'n_layers':36,'d_ff':9728,'d_model':2560,'n_heads':32}}})
    work=put('workpackage',{'schema':'research_workpackage.v1','title':'Qwen3-4B · 条件响应卡、广泛关联与稀有强响应',
        'stages':{'condition_cards':{'status':'done','n':77824,'semantic_confirmations_added':0},
                  'preregistration':{'status':'done','artifact_sha256':prereg},'new_heldout':{'status':'done','families':512,'conditions_per_family':16},
                  'prediction_and_intervention':{'status':'done','gates':{k:v['conditional']['passed'] for k,v in report['observations'].items()}},
                  'registration_validated':{'status':'done','report_sha256':report_ref,'graph_sha256':graph}}})
    registry={'run_id':D.RUN,'model_snapshot':D.SNAPSHOT,'report_sha256':report_ref,'graph_sha256':graph,'binding_sha256':binding,'workpackage_sha256':work}
    bundle=D.ROOT/f'release/ai2050-research-conditional-{revision}.tar.gz'
    assert sum(len(raw)+1024 for raw in objects.values())+16384<128*2**20,'Client upload expanded size limit'
    with tarfile.open(bundle.with_suffix('.tmp'),'w:gz',compresslevel=5) as tar:
        def add(name,raw):
            info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
        add('MANIFEST.json',canonical({'schema':'ai2050_research_upload.v1','objects':[{'sha256':s} for s in objects],'studies':[registry]}))
        for sha,raw in objects.items():add(f'objects/{sha}.json',raw)
    bundle.with_suffix('.tmp').replace(bundle)
    assert bundle.stat().st_size<=64*2**20,'Client upload compressed size limit'
    provenance._git=D.P.git
    provenance.model_revision=lambda _:D.SNAPSHOT
    audit=provenance.write_result('mlp_conditional_paths',{'registry':registry,'bundle_sha256':D.HASH(bundle),'plan_sha256':D.HASH(D.OUT/'prereg.json'),
             'scope':'one real node; remote receipt tracked separately'},model_path=D.MODEL,seed=D.SEED,tool='mlp_condition_publish')
    registry.update(provenance=plan['provenance'],condition_layers=layers,bundle=str(bundle.relative_to(D.ROOT)),bundle_sha256=D.HASH(bundle),
                    objects=len(objects),cards=77824,controlled_units=4,audit=str(audit.relative_to(D.ROOT)))
    D.SAVE(public/'registration.json',registry);D.SAVE(public/'report.research.json',report);D.SAVE(D.OUT/'registration.json',registry)
    print(registry,flush=True)

if __name__=='__main__':main()
