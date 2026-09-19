"""Read-only integrity, finite-value, coverage, and append-only audit. No model execution/deletion."""
import argparse,os,re,gc
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from rdc_conditional_common import *


def finite_npz(path):
    shapes={};count=0
    with np.load(path,allow_pickle=False) as z:
        for key in z.files:
            a=z[key]
            if a.dtype==np.uint16:assert not np.any((a&np.uint16(0x7f80))==np.uint16(0x7f80)),(path,key,'BF16 nonfinite')
            elif np.issubdtype(a.dtype,np.floating):assert np.isfinite(a).all(),(path,key,'nonfinite')
            shapes[key]=list(a.shape);count+=a.size
    return shapes,count


def committed(run,expected,scan=True):
    out=CAMPAIGN/run;commits=sorted((out/'commits').glob('*.json'));manifest=hashlib.sha256();files=0;scalars=0;bytes_checked=0;panels=0;key_counts={}
    assert len(commits)==expected,(run,len(commits),expected)
    protocol=sha(out/'protocol.json');retained={}
    for i,path in enumerate(commits):
        c=read(path);assert c['protocol_sha']==protocol,(path,'protocol changed')
        for rel,digest in c['files'].items():
            p=(out/rel).resolve();assert p.is_relative_to(out.resolve()),(path,'outside run')
            assert p.is_file() and sha(p)==digest,(path,rel,'hash mismatch')
            files+=1;bytes_checked+=p.stat().st_size;manifest.update((str(p.relative_to(CAMPAIGN))+'\0'+digest+'\n').encode())
            if p.suffix=='.npz' and scan:
                shapes,n=finite_npz(p);scalars+=n
                if p.parent.name=='fields':
                    panels+=int('h' in shapes or 'h_prefill' in shapes)
                    for k in shapes:key_counts[k]=key_counts.get(k,0)+1
        if i%512==0:print('INTEGRITY',run,i,len(commits),flush=True)
    material=read(out/'material.json');assert len(material)==expected,(run,'material count')
    assert len({r['sample_id'] for r in material})==len(material)
    entity_splits={}
    for r in material:
        entity_splits.setdefault(r['unit'],set()).add(r['word_split'])
    assert all(len(v)==1 for v in entity_splits.values()),(run,'entity split leakage')
    prefixfiles=sorted((out/'prefix_commits').glob('*.json'));corrected=0;linked_steps=[]
    if run!='i_factorial':assert len(prefixfiles)==(128 if run=='k_long' else 288 if run=='m_order' else 512 if run=='o_generalization' else 256),(run,'prefix coverage')
    for p in prefixfiles:
        c=read(p);sid=c.get('sample_id',c.get('prefix_id'));b=out/f'behavior/{sid}.json'
        assert sha(b)==c['behavior_sha'],(p,'behavior changed')
        assert all((out/f'commits/{s}.json').exists() for s in c['steps'])
        linked_steps.extend(c['steps'])
        sb=out/f'behavior_scored/{sid}.json'
        if sb.exists():
            score=read(sb);score_protocol='scoring_alignment_protocol_v2.json' if run=='m_order' else 'scoring_protocol_v2.json'
            assert score['original_behavior_sha']==sha(b) and score['scoring_protocol_sha']==sha(out/score_protocol);corrected+=1
    if prefixfiles:assert len(linked_steps)==expected and set(linked_steps)=={r['sample_id'] for r in material},(run,'prefix/state linkage')
    if run in ('k_long','m_order'):assert corrected==len(prefixfiles),(run,'corrected score coverage')
    if run=='m_order':
        scored=read(out/'material_scored.json');assert [r['sample_id'] for r in scored]==[r['sample_id'] for r in material]
        selected={r['sample_id'] for r in scored if r.get('result_field_onset')}
        scores=[read(out/f'behavior_scored/{p.stem}.json') for p in prefixfiles]
        assert selected=={b['result_boundary_state'] for b in scores if b['result_boundary_state'] is not None}
        assert all((out/f'accounts/{sid}.json').exists() and (out/f'ledgers/{sid}.npz').exists() for sid in selected)
    return {'run':run,'commits':len(commits),'prefix_commits':len(prefixfiles),'checked_file_references':files,'bytes_hashed':bytes_checked,
      'finite_array_scalars':scalars,'fulltoken_panel_files':panels,'field_key_counts':key_counts,'file_reference_manifest_sha256':manifest.hexdigest(),
      'material_sha':sha(out/'material.json'),'grouped_entity_splits':{str(k):list(v)[0] for k,v in entity_splits.items()},'corrected_behavior_links':corrected,'passed':True}


def prefix_audit():
    p=CAMPAIGN/'memo_prefix.json';original=read(p);memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';h=hashlib.sha256();remaining=original['bytes']
    with open(memo,'rb') as stream:
        while remaining:
            chunk=stream.read(min(8*1024**2,remaining));assert chunk;h.update(chunk);remaining-=len(chunk)
        suffix=stream.read().decode('utf-8')
    assert h.hexdigest()==original['sha256'],'Precampaign MEMO prefix changed'
    phases=[int(x) for x in re.findall(r'^## Phase ([0-9]+):',suffix,re.M)];assert phases==list(range(2703,2703+len(phases))),phases
    return {'initial_bytes':original['bytes'],'initial_sha256':h.hexdigest(),'unchanged':True,'appended_phases':phases,'current_bytes':memo.stat().st_size}


def main(partial):
    roots=['i_factorial','k_long'] if partial else ['i_factorial','k_long','l_aligned/qwen4','l_aligned/qwen14','l_aligned/glm4','m_order','o_generalization']
    reports=[]
    for run in roots:
        expected=4096 if run=='i_factorial' else len(read(CAMPAIGN/run/'material.json'))
        reports.append(committed(run,expected));gc.collect()
    assert reports[0]['fulltoken_panel_files']==512 and reports[1]['fulltoken_panel_files']==16
    derived=[]
    for run in (['i_factorial','j_predictive_gates','k_long'] if partial else ['i_factorial','j_predictive_gates','k_long','l_aligned','l_aligned/qwen4','l_aligned/qwen14','l_aligned/glm4','m_order','n_cached_attention','o_generalization','p_token_conditioned']):
        out=CAMPAIGN/run;count=0;scalar=0;manifest=hashlib.sha256()
        paths=[p for folder in ('models','predictions','unit_errors','features','ledgers','figures') for p in (out/folder).rglob('*.npz')]
        paths.extend(out.glob('*.npz'))
        for p in sorted(set(paths)):
            _,n=finite_npz(p);count+=1;scalar+=n;manifest.update((str(p.relative_to(out))+'\0'+sha(p)+'\n').encode())
        derived.append({'run':run,'finite_derived_files':count,'scalars':scalar,'manifest_sha256':manifest.hexdigest()})
    sources={}
    for pattern in ('phase2703*.py','phase2704*.py','phase2705*.py','phase2706*.py','phase2707*.py','phase2708*.py','phase2709*.py','phase2710*.py','rdc_conditional*.py','rdc_long_material.py','rdc_order_material.py','rdc_attention_transfer_material.py'):
        for p in (ROOT/'tests/glm5').glob(pattern):sources[str(p.relative_to(ROOT))]=sha(p)
    for p in [ROOT/'server/rdc_feature_service.py',ROOT/'server/server.py',ROOT/'frontend/src/components/app/RdcFeatureAtlas.jsx',ROOT/'frontend/src/components/app/RdcFeatureAtlas.css',ROOT/'tests/glm5_temp/rdc_conditional_client_test.cjs',ROOT/'AGENTS.md']:sources[str(p.relative_to(ROOT))]=sha(p)
    if not partial:
        frozen=read(CAMPAIGN/'o_generalization/frozen_predictors.json')
        for rel,digest in frozen['files'].items():assert sha(CAMPAIGN/'n_cached_attention'/rel)==digest
        assert read(CAMPAIGN/'n_cached_attention/cache_input_audit.json')['passed']
        assert read(CAMPAIGN/'o_generalization/serialization_audit.json')['passed']
        assert read(CAMPAIGN/'p_token_conditioned/embedding_identity_audit.json')['passed']
        original=read(CAMPAIGN/'o_generalization/features/selected_rows.json');p_rows=read(CAMPAIGN/'p_token_conditioned/selected_rows.json')
        assert len(original)==len(p_rows)==1536
        for a,b in zip(original,p_rows):
            assert a['sample_id']==b['sample_id'] and a['word_split']==b['origin_word_split']=='test'
            assert b['word_split']==('train' if b['unit']<8 else 'validation' if b['unit']<12 else 'test')
        assert read(CAMPAIGN/'client_verification.json')['passed']
    model_identity=[]
    for name in ('qwen3-4b','Qwen3-14B','glm4-9b-chat-hf'):
        root=ROOT/'models/hf'/name;index=root/'model.safetensors.index.json'
        model_identity.append({'model':name,'config_sha':sha(root/'config.json'),'tokenizer_json_sha':sha(root/'tokenizer.json'),'weight_index_sha':sha(index),
          'checkpoint_shards':[{'file':file,'bytes':(root/file).stat().st_size,'mtime_ns':(root/file).stat().st_mtime_ns} for file in sorted(set(read(index)['weight_map'].values()))],
          'limit':'Shard size/mtime inventory is not a full cryptographic hash of every model weight. Config/tokenizer/index hashes and actual sampled native rows/code versions separately recorded.'})
    total=sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file());free=__import__('shutil').disk_usage(CAMPAIGN).free
    assert total<30*1024**3 and free>8*1024**3,(total,free)
    report={'timestamp':stamp(),'partial':partial,'passed':True,'runs':reports,'derived_arrays':derived,'memo_prefix':prefix_audit(),'source_files':sources,'models':model_identity,
      'campaign_bytes_at_audit':total,'free_bytes_at_audit':free,'resource_limits':{'campaign_bytes':30*1024**3,'free_floor':8*1024**3},
      'scope':'All committed field hashes/finite values, prefix behavior links, split identities, derived arrays, append-only prefix; not a proof of scientific theory.'}
    save(CAMPAIGN/('partial_integrity_audit.json' if partial else 'delivery_audit.json'),report)
    print('DELIVERY_INTEGRITY_PASS','partial' if partial else 'full',total,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
