"""Full campaign provenance audit, with restartable hash checks and honest completion state."""
import hashlib,sys
from rdc_continuity_common import *

def main():
    import psutil
    assert read(CAMPAIGN/'serial_tail.json')['state']=='complete'
    active=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        cmd=' '.join(p.info['cmdline'] or [])
        if 'python' in (p.info['name'] or '').lower() and any(s in cmd for s in ('phase2702_rdc_serial_scale.py','phase2701_rdc_current_token_capture.py','phase2699_rdc_equalshape_capture.py')):active.append(p.pid)
    assert not active,active
    prefix=read(CAMPAIGN/'memo_prefix.json');memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    with memo.open('rb') as f:actual=hashlib.sha256(f.read(prefix['bytes'])).hexdigest()
    assert actual==prefix['sha256'],'Append-only memo prefix changed'
    assert read(CAMPAIGN/'client_api_audit.json')['passed']
    material_rows=read(CAMPAIGN/'e_confirmation/material.json')
    base_splits={}
    for r in material_rows:
        base_splits.setdefault(r['base_id'],set()).add(r['word_split'])
        assert r['expected_yes']==(r['fact_truth']!=r['negative_query'])
    assert all(len(v)==1 for v in base_splits.values())
    assert len({r['prompt'] for r in material_rows})==1024
    material_audit={'group_count':len(base_splits),'cross_split_base_groups':0,'unique_prompts':1024,
      'scope':'Checks data identity and generated-label bookkeeping, not independent human semantic adjudication.',
      'limits':['The punctuation claim says final punctuation without repeating of the evidence; interpretation as full-prompt punctuation is a potential ambiguity.','Scaleform1 jointly shifts wording and some voice/depth; not independent factor attribution.','English/Chinese counterparts and alltruth/query variants stay in their base split.']}
    groups={}
    for key,count,hlayers,width in [('e_confirmation',1024,37,2560),('g_generation',975,37,2560),('h_scale/qwen14',256,41,5120),('h_scale/glm4',256,41,4096)]:
        out=CAMPAIGN/key;rows=read(out/'material.json');assert len(rows)==count
        commits=list((out/'commits').glob('*.json'));assert len(commits)==count
        totals={'samples':count,'raw_bytes':0,'h_scalars':0,'finite_float_scalars':0,'hashed_files':0,'capture_work_seconds':0.0}
        for i,r in enumerate(rows):
            c=read(out/f'commits/{r["sample_id"]}.json')
            if 'protocol_sha' in c:assert c['protocol_sha']==sha(out/'protocol.json')
            for rel,digest in c['files'].items():
                p=(out/rel).resolve();assert p.is_relative_to(out.resolve())
                assert sha(p)==digest,str(p);totals['hashed_files']+=1
            p=out/f'fields/{r["sample_id"]}.npz';totals['raw_bytes']+=p.stat().st_size
            with np.load(p,allow_pickle=False) as z:
                h=z['h'];assert h.shape[0]==hlayers and h.shape[2]==width
                if key=='g_generation':assert h.shape[1]==1
                else:assert h.shape[1]==len(r['prompt_ids'])
                totals['h_scalars']+=h.size
                for name in z.files:
                    a=z[name]
                    if a.dtype==np.uint16:a=unbits(a)
                    if np.issubdtype(a.dtype,np.floating):
                        assert np.isfinite(a).all(),(key,r['sample_id'],name)
                        totals['finite_float_scalars']+=a.size
            totals['capture_work_seconds']+=c.get('elapsed_seconds',0)
            if (i+1)%64==0:print('AUDIT',key,i+1,count,flush=True)
        groups[key]=totals
        save(CAMPAIGN/'delivery_audit_progress.json',{'timestamp':stamp(),'completed_groups':groups})
    prefixes=read(CAMPAIGN/'g_generation/prefixes.json');assert len(prefixes)==320
    pcommits=[read(p) for p in (CAMPAIGN/'g_generation/prefix_commits').glob('*.json')];assert len(pcommits)==320
    ids=[s for p in pcommits for s in p['steps']]
    assert len(ids)==len(set(ids))==975
    assert set(ids)=={r['sample_id'] for r in read(CAMPAIGN/'g_generation/material.json')}
    groups['g_generation']['capture_work_seconds']=sum(p['elapsed_seconds'] for p in pcommits)
    for sid in ids:
        assert (CAMPAIGN/f'g_generation/accounts/{sid}.json').exists()
        assert (CAMPAIGN/f'g_generation/ledgers/{sid}.npz').exists()
    model_metadata={}
    for key in ('qwen3-4b','Qwen3-14B','glm4-9b-chat-hf'):
        out=ROOT/'models/hf'/key;idx=read(out/'model.safetensors.index.json')
        model_metadata[key]={'index_sha256':sha(out/'model.safetensors.index.json'),'config_sha256':sha(out/'config.json'),
            'shards':[{'name':name,'bytes':(out/name).stat().st_size,'mtime_ns':(out/name).stat().st_mtime_ns} for name in sorted(set(idx['weight_map'].values()))],
            'note':'Shard size/mtime inventory, not a new checksum of every checkpoint byte; live scalar slices checked against checkpoint bits.'}
    sources=[p for p in (ROOT/'tests/glm5').glob('phase*.py') if any(p.name.startswith(f'phase{i}_') for i in (2699,2700,2701,2702))]
    sources += [ROOT/p for p in ('tests/glm5/rdc_continuity_common.py','tests/glm5/rdc_continuity_material.py','tests/glm5/rdc_feature_extractors.py','server/rdc_feature_service.py','frontend/src/components/app/RdcFeatureAtlas.jsx','frontend/src/components/app/RdcFeatureAtlas.css','tests/glm5_temp/rdc_continuity_client_test.cjs','tests/glm5_temp/rdc_continuity_serial_tail.py')]
    report={'timestamp':stamp(),'passed':True,'groups':groups,'total_raw_bytes':sum(v['raw_bytes'] for v in groups.values()),
      'total_h_scalars':sum(v['h_scalars'] for v in groups.values()),'memo_prefix':dict(prefix,verified=True),
      'sources':{str(p.relative_to(ROOT)):sha(p) for p in sources},'model_metadata':model_metadata,
      'material_audit':material_audit,
      'owned_model_capture_processes':active,'deleted_files':0,'raw_retention':'All native fields retained for client, repeat analysis and evidence; no cleanup performed.',
      'scientific_goal_solved':False,'limits':['No universal language closure.','No cross-model native-coordinate isomorphism.','Long-form rewriting and first shape-difference operator not tested in this bounded plan.','Old Phase2691 four-protocol work remains deferred.']}
    save(CAMPAIGN/'delivery_audit.json',report)
    print('AUDIT_COMPLETE',report['total_raw_bytes'],report['total_h_scalars'],flush=True)

if __name__=='__main__':main()
