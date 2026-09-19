"""Independent artifact/identity/axis audit after scientific execution, before delivery."""
from collections import Counter,defaultdict
from rdc_query_common import *

OUT=BASE/'verification'


def main():
    start=time.monotonic();assert read(BASE/'science_queue/status.json')['all_passed'];checks=[]
    required=['contract.json','material/result.json','algebra/result.json','atlas/result.json','events/result.json',
      'rules/capture_result.json','rules/fit_result.json','rules/vocabulary_result.json','pairs/result.json',
      'transfer/capture_result.json','transfer/fit_result.json','transfer/vocabulary_result.json','transfer/injection/result.json',
      'formation/result.json','followup/result.json','analysis/phase2740.json','analysis/phase2741.json','analysis/phase2742.json','analysis/query_identity_control.json','analysis/query_language_control.json']
    required += ['late/'+b+'/result.json' for b in ['native','fixed128_digit','entropy_digit','entropy_letter','terminal_marker_digit']]
    required += ['scale/'+m+'/result.json' for m in ['qwen4','qwen14','glm4']]
    for p in required:
        r=read(BASE/p)
        if p!='contract.json':assert r['all_passed'],p
    checks.append({'check':'Every required experiment completed; execution success kept distinct from hypothesis gates','artifacts':len(required)})
    rows=gzread(BASE/'material/natural.json.gz');lookup={r['sample_id']:r for r in rows};assert len(lookup)==len(rows)==10000
    groups=defaultdict(set)
    for r in rows:groups[r['source_group']].add(r['split'])
    assert len(groups)==2777 and all(len(s)==1 for s in groups.values());checks.append({'check':'Natural source documents do not cross train/validation/test splits','documents':len(groups)})
    probes=read(BASE/'probes/protocol.json')['probes'];assert len(probes)==100 and len({p['probe_id'] for p in probes})==100
    assert Counter(p['split'] for p in probes)=={'train_query':60,'validation_query':20,'unseen_query':20}
    assert Counter(p['language'] for p in probes)=={'en':50,'zh':50};checks.append({'check':'All100prospective query identities and60/20/20target splits','queries':100})
    detail=set(read(BASE/'material/protocol.json')['detailed_prefix_ids']);fixture=set(read(BASE/'material/protocol.json')['full_layer_all_token_fixture_ids'])
    assert len(detail)==576 and len(fixture)==9;cache_bits=0
    for i,r in enumerate(rows):
        sid=r['sample_id'];cp=read(BASE/'capture/commits'/f'{sid}.json');p=BASE/'capture/fields'/f'{sid}.npz'
        assert cp['sample_id']==sid and cp['source_group']==r['source_group'] and cp['array_sha256']==sha(p)
        assert cp['source_cache_tensors_not_aliased'] and cp['source_cache_lengths_unchanged_all_layers']
        assert len(cp['full_layer_prefix_identities'])==37 and sorted(q for batch in cp['exact_suffix_batch_indices'] for q in batch)==list(range(100))
        cache_bits+=cp['source_cache_bits_checked']
        with np.load(p) as z:
            assert z['prefix_layers'].shape==(37,2560) and z['postnorm'].shape==(100,2560) and z['full_vocabulary_statistics'].shape==(100,6)
            assert ('query_H12_H24_rawH36' in z.files)==(sid in detail)
            if sid in detail:
                assert z['query_H12_H24_rawH36'].shape==(100,3,2560)
                assert z['prefix_H12_sources'].shape==(len(r['prompt_ids']),2560)
                assert z['prefix_block12_keys'].shape==z['prefix_block12_values'].shape==(8,len(r['prompt_ids']),128)
            assert ('full_prefix_layers' in z.files)==(sid in fixture)
            if sid in fixture:assert z['full_prefix_layers'].shape==(37,len(r['prompt_ids']),2560)
        if (i+1)%1000==0:print('QUERY_INTEGRITY_NATIVE',i+1,10000,flush=True)
    assert cache_bits==100;checks.append({'check':'Every native capture hash, all declared axes and suffix IDs','prefixes':10000,'whole_cache_byte_audit_prefixes':cache_bits})
    proto=read(BASE/'prototypes/result.json');assert sha(BASE/'prototypes/qwen4.npz')==proto['archive_sha256']
    rc=read(BASE/'rules/capture_result.json');assert rc['merge_max_abs_error']<1e-10 and len(rc['prototype_reconstruction'])==100
    for c in rc['prototype_reconstruction']:assert c['Q_bit_equal'] and c['K_rope_bit_equal'] and c['V_bit_equal']
    for sid in detail:
        rec=read(BASE/'rules/commits'/f'{sid}.json');assert rec['sha256']==sha(BASE/'rules/features'/f'{sid}.npz')
        assert rec['input_capture_sha256']==read(BASE/'capture/commits'/f'{sid}.json')['array_sha256']
        with np.load(BASE/'rules/features'/f'{sid}.npz') as z:assert z['candidate_H13'].shape==(5,100,2560)
    assert sha(BASE/'rules/decoder.npz')==read(BASE/'rules/fit_result.json')['decoder_sha256']==read(BASE/'followup/protocol.json')['unchanged_main_decoder_sha256']
    checks.append({'check':'Full candidate vectors, known-query Q/K/V bit reconstruction, stable fixed-query merge, immutable confirmation decoder','detail_prefixes':576})
    event_material=gzread(BASE/'events/material.json.gz')['trajectories'];steps=0;anchors=0
    for item in event_material:
        sid=item['row']['sample_id'];native=read(ROOT/item['native_record']);r=read(BASE/'events/commits'/f'{sid}.json')
        assert [s['token_id'] for s in r['steps']]==native['generated_ids'];steps+=len(r['steps']);anchors+=len(r['anchors'])
        assert r['anchors']==item['anchors'] and sha(BASE/'events/fields'/f'{sid}.npz')==r['archive_sha256']
        with np.load(BASE/'events/fields'/f'{sid}.npz') as z:
            assert z['H'].shape==(len(r['anchors']),37,2560) and z['all_units'].shape==(len(r['anchors']),6,9728)
            assert z['dynamic_query_postnorm'].shape==(len(r['anchors']),100,2560)
    assert len(event_material)==32 and anchors==352
    paths=read(BASE/'events/paths_index.json');assert len(paths)==44
    for p in paths:assert sha(BASE/p['path'])==p['sha256'] and all(a['all_unit_sum_relative_error']<1e-5 for a in p['audits'])
    checks.append({'check':'Every old native token reproduced and every declared actual-time/ordered-source path committed','native_steps':steps,'anchors':anchors,'path_records':len(paths)})
    transfer=gzread(BASE/'transfer/material.json.gz');assert len(transfer)==768
    for r in transfer:
        rec=read(BASE/'transfer/commits'/f"{r['sample_id']}.json");assert rec['source_group']==r['source_group'] and sha(BASE/'transfer/fields'/f"{r['sample_id']}.npz")==rec['sha256']
    formation=gzread(BASE/'formation/material.json.gz')
    for r in [*formation['train'],*[r for r in formation['panel'] if r['kind']=='natural_content']]:
        original=lookup[r['sample_id'].rsplit('_content',1)[0]];pos=r['position']
        assert r['ids']==original['prompt_ids'][:pos+1] and r['target']==original['prompt_ids'][pos+1]
    assert len(formation['train'])==576 and len(formation['panel'])==414
    p=read(BASE/'formation/protocol.json')
    for seed in p['seeds']:
      for condition in p['conditions']:
        out=BASE/'formation'/f'{condition}_{seed}';r=read(out/'result.json');assert len(r['trace'])==32 and sha(out/'parameter_delta_FP32.npz')==r['delta_sha256']
        with np.load(out/'parameter_delta_FP32.npz') as z:assert sum(z[k].size for k in z.files)==74711040 and all(z[k].dtype==np.float32 for k in z.files)
    checks.append({'check':'Paired768expression identities and actual natural-content truncated training/scoring inputs, all74711040updated scalar deltas','native_training_runs':4})
    from rdc_query_scoring import score
    for mode,branches,expected in [('late',['native','fixed128_digit','entropy_digit','entropy_letter','terminal_marker_digit'],96),('injection',['native','code_identity','mapped_code'],32)]:
        material=gzread(BASE/'late/material.json.gz') if mode=='late' else transfer;ml={r['sample_id']:r for r in material}
        for b in branches:
            folder=BASE/'late'/b/'commits' if mode=='late' else BASE/'transfer/injection/commits'/b;files=list(folder.glob('*.json'));assert len(files)==expected
            native_folder=BASE/'late/native/commits' if mode=='late' else BASE/'transfer/injection/commits/native'
            for f in files:
                r=read(f);assert r['branch']==b and len(r['steps'])==len(r['generated_ids'])<=1024
                assert [s['token_id'] for s in r['steps']]==r['generated_ids']
                # Stop tokens are from the original, read-only native configuration.
                cfg=read(ROOT/'models/hf/qwen3-4b/generation_config.json');stop=cfg['eos_token_id'];stop=set(stop if isinstance(stop,list) else [stop])
                assert score(ml[r['sample_id']],r['generated_text'],r['generated_ids'],stop,1024)==r['answer_scoring']
                if mode=='late' and b!='native':
                    n=read(native_folder/f.name);fire=r['intervention']
                    if fire is None:assert r['generated_ids']==n['generated_ids']
                    else:
                        assert fire['all_same_history_KV_bit_equal']
                        div=r['first_token_divergence_from_native'];assert div is None or div>=fire['step']
                if mode=='injection':assert r['cache_audit']['same_current_history_KV_exact'] and not r['cache_audit']['used_correct_label']
    checks.append({'check':'All576formal own-history trajectories independently rescored, IDs and intervention-time boundaries preserved','trajectories':576})
    for model,D in [('qwen4',2560),('qwen14',5120),('glm4',4096)]:
        r=read(BASE/'scale'/model/'result.json');assert r['width']==D and r['native_dtype']=='torch.bfloat16' and not r['quantized'] and r['sources'] in [16,32,64]
        assert r['native_QK_RoPE_attention_checks'] and max(c['max_abs_error'] for c in r['native_QK_RoPE_attention_checks'])<.005
        for f in (BASE/'scale'/model/'commits').glob('*.json'):
            rec=read(f);assert sha(BASE/'scale'/model/'fields'/f.with_suffix('.npz').name)==rec['sha256']
    fresh=gzread(BASE/'followup/material.json.gz');assert len(fresh)==96 and len({r['source_group'] for r in fresh})==96
    assert not set(groups)&{r['source_group'] for r in fresh}
    for r in fresh:
        rec=read(BASE/'followup/capture/commits'/f"{r['sample_id']}.json");assert rec['all_prefix_KV_unchanged'] and sha(BASE/'followup/capture/fields'/f"{r['sample_id']}.npz")==rec['sha256']
    checks.append({'check':'Own-native three-model dimensions/precision/QK and96reserved-document automatic confirmation','fresh_documents':96})
    if (BASE/'identifiability/analysis/result.json').exists():
        from phase2744_rdc_query_integrity import audit
        identity=audit();save(BASE/'identifiability/verification.json',identity);checks.extend(identity['checks'])
    # Inspect every retained NPZ, including all low-magnitude coordinates and full parameter deltas.
    archives=[];scalars=0
    for i,p in enumerate(sorted(BASE.rglob('*.npz'))):
        assert not p.name.endswith('.tmp.npz'),('Uncommitted temporary numerical archive',str(p));arrays=[]
        with np.load(p) as z:
            for name in z.files:
                a=z[name];assert a.dtype.kind in 'buif',('Unexpected nonnumeric archive',str(p),name)
                flat=a.reshape(-1)
                for j in range(0,len(flat),2_000_000):
                    chunk=unbits(flat[j:j+2_000_000]) if a.dtype==np.uint16 else flat[j:j+2_000_000]
                    assert np.isfinite(chunk).all(),(str(p),name,j)
                arrays.append({'name':name,'shape':list(a.shape),'dtype':str(a.dtype)});scalars+=a.size
        archives.append({'path':p.relative_to(BASE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'arrays':arrays})
        if (i+1)%1000==0:print('QUERY_INTEGRITY_ALL_ARCHIVES',i+1,flush=True)
    compressed(OUT/'all_numerical_archives.json.gz',archives)
    checks.append({'check':'Every retained native-bit and numerical array scanned for finite values and original axes','archives':len(archives),'saved_scalar_elements':scalars})
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':checks,'required_artifact_sha256':{p:sha(BASE/p) for p in required},
      'numerical_archive_manifest_sha256':sha(OUT/'all_numerical_archives.json.gz'),'seconds':time.monotonic()-start,
      'scope':'Numerical/identity/reproducibility checks do not turn a passed implementation into a universally supported mechanism. All-stage hypothesis failures remain recorded.'}
    save(OUT/'scientific_integrity.json',result);ledger('query_scientific_artifact_integrity',result['seconds']);print('QUERY_SCIENTIFIC_INTEGRITY_PASS',len(checks),result['seconds'],flush=True)


if __name__=='__main__':main()
