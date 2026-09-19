"""Complete saved-array and source-identity audit, with bounded resident memory."""
from collections import Counter
from rdc_binding_common import *

def main():
    start=time.monotonic();guard()
    assert read(BASE/'format_content/suite_result.json')['all_passed']
    contract=read(BASE/'contract.json');memo=MEMO.read_bytes()
    assert hashlib.sha256(memo[:contract['memo_prefix_bytes']]).hexdigest()==contract['memo_prefix_sha']
    assert sha(LAW/'verification/final.json')==contract['prior_final_sha']
    for r in contract['attachments']:
        assert sha(Path(r['path']))==r['sha256']
        assert sha(BASE/r['snapshot'])==r['sha256']
    material=read(BASE/'material_frozen.json')
    for name,digest in material['hashes'].items():assert sha(BASE/name)==digest,name
    for r in material['sources']:assert sha(ROOT/r['path'])==r['archive_sha']
    frozen=read(BASE/'prediction/frozen.json')
    assert frozen['material_sha']==sha(BASE/'natural_discovery.json.gz')
    assert frozen['kernels_sha']==sha(BASE/'prediction/kernels.npz')
    assert read(BASE/'confirmation/result.json')['prediction_freeze_sha']==sha(BASE/'prediction/frozen.json')
    future_protocol=read(BASE/'format_content/protocol.json')
    assert future_protocol['new_material_sha']==sha(BASE/'format_content/prospective_material.json.gz')
    recovery_precision=read(BASE/'format_content/numerical_recovery/failed_attempt.json')
    assert recovery_precision['unchanged_protocol_sha256']==sha(BASE/'format_content/protocol.json')
    assert read(BASE/'format_content/numerical_recovery/precision_diagnostic.json')['stable_variant_passed']
    preserved=recovery_precision['preserved_initial_scores'];assert sha(BASE/preserved['file'])==preserved['sha256']
    projection_audit=read(BASE/'format_content/projection_condition_audit.json')
    assert projection_audit['input_gram_sha256']==sha(BASE/'format_content/all_parameter_gram_decomposition.npz')
    assert projection_audit['all_declared_features_and_gram_entries_preserved'] and not projection_audit['new_direction_fitted_or_deployed']
    full_export=read(BASE/'format_content/full_factor_export.json');assert full_export['all_passed']
    assert full_export['factor_sha256']==sha(BASE/'format_content/full_gradient_factors.npz')
    discovery=gzread(BASE/'natural_discovery.json.gz');natural=gzread(BASE/'natural_confirmation.json.gz')
    programs=gzread(BASE/'program_material.json.gz');future=gzread(BASE/'format_content/prospective_material.json.gz')
    assert tuple(map(len,(discovery,natural,programs,future)))==(512,128,768,128)
    signed=signed_rows();identity_audit=read(BASE/'signed_source/identity_recovery/result.json')
    assert identity_audit['all_passed'] and identity_audit['native_forward_count']==127
    assert identity_audit['resolved_material_sha256']==sha(BASE/'signed_source/identity_recovery/resolved_material.json.gz')
    allrows=discovery+natural+programs+future+signed;lookup={r['sample_id']:r for r in allrows}
    assert len(signed)==128 and len(lookup)==1664
    assert len({tuple(r['prompt_ids']) for r in signed})==127
    signed_freeze=read(BASE/'signed_source/frozen.json')
    assert signed_freeze['material_sha256']==sha(BASE/'signed_source/natural_material.json.gz')
    assert signed_freeze['features_sha256']==sha(BASE/'signed_source/all_discovery_signed_features.npz')
    for name,digest in signed_freeze['banks_sha256'].items():assert sha(BASE/'signed_source/banks'/name)==digest
    assert read(BASE/'signed_source/math.json')['all_passed']
    assert read(BASE/'verification/source_moment_collision/result.json')['all_passed']
    from phase2732_rdc_binding_material import connected
    train=[r for r in discovery if r['split']=='train']
    assert len(train)==320 and all(r['retrospective_ud'] and not connected(r) for r in train)
    old=gzread(LAW/'material.json.gz')+gzread(LAW/'confirmation_material.json.gz')
    oldcomponents={c for r in old for c in r.get('component_ids',[])}
    freshcomponents=[c for r in natural for c in r['component_ids']]
    assert not oldcomponents.intersection(freshcomponents)
    assert len(freshcomponents)==len(set(freshcomponents))
    signed_components=[c for r in signed for c in r['component_ids']]
    assert len(signed_components)==len(set(signed_components))
    assert not (oldcomponents|set(freshcomponents)).intersection(signed_components)
    groups={}
    for r in programs+future:
        groups.setdefault(r['source_group'],set()).add(r['split'])
        assert len(r['target_ids'])==1 and len(r['relations'])==r['depth']
    assert len(groups)==224 and all(len(v)==1 for v in groups.values())
    assert all(r['depth']==6 and r['split']=='prospective_depth6' for r in future)
    expected={};fixtures=[];stop=set(read(ROOT/'models/hf/qwen3-4b/generation_config.json')['eos_token_id'])
    for r in natural+programs:
        sid=r['sample_id'];kind='natural' if r['kind']=='natural' else 'program'
        commit=read(BASE/'capture/commits'/f'{sid}.json')
        if kind=='program':assert commit['eos']==any(t in stop for t in commit['generated_ids'])
        expected[f'capture/{kind}/{sid}.npz']=commit['array_sha']
        with np.load(BASE/f'capture/{kind}/{sid}.npz') as z:
            assert z['H'].shape==(37,len(r['anchors']),2560)
            assert z['H12_sources'].shape==(len(r['prompt_ids']),2560)
            assert z['token_ids'].tolist()==r['prompt_ids']
            if commit['full_fixture']:
                with np.load(BASE/'capture/full_fields'/f'{sid}.npz') as full:
                    assert full['H'].shape==(37,len(r['prompt_ids']),2560)
                    assert np.array_equal(full['H'][:,r['anchors']],z['H'])
                    assert np.array_equal(full['H'][12],z['H12_sources'])
                fixtures.append(sid)
    assert len(fixtures)==12
    for r in signed:
        sid=r['sample_id'];commit=read(BASE/'signed_source/commits'/f'{sid}.json')
        expected[f'signed_source/fields/{sid}.npz']=commit['array_sha']
        with np.load(BASE/'signed_source/fields'/f'{sid}.npz') as z:
            assert z['H'].shape==(37,len(r['anchors']),2560) and z['token_ids'].tolist()==r['prompt_ids']
            if commit['full_fixture']:
                with np.load(BASE/'signed_source/full_fields'/f'{sid}.npz') as full:
                    assert np.array_equal(full['H'][:,r['anchors']],z['H'])
                    assert np.array_equal(full['H'][12],z['H12_sources'])
                fixtures.append(sid)
    assert len(fixtures)==16
    for model in ('qwen4','qwen14','glm4'):
        root=BASE/'scale'/model;result=read(root/'result.json');runtime=result['runtime']
        assert result['natural_rows']==32 and result['program_rows']==96
        assert not runtime['quantized'] and runtime['dtype']=='torch.bfloat16'
        commits=list((root/'commits').glob('*.json'));assert len(commits)==128
        for path in commits:
            r=read(path);key=f'scale/{model}/fields/{r["sample_id"]}.npz'
            expected[key]=r['array_sha']
            with np.load(BASE/key) as z:
                assert z['H'].shape==(runtime['depth']+1,len(r['positions']),runtime['width'])
                assert z['H_early_sources'].shape==(r['tokens'],runtime['width'])
            if model=='qwen4':assert r['independent_Q4_replay_exact']
    recovery=read(BASE/'scale/qwen14/recovery_verification.json')
    assert recovery['all_passed'] and all(recovery['profile_comparison'].values())
    for r in future:
        sid=r['sample_id'];key=f'format_content/prospective_fields/{sid}.npz'
        expected[key]=read(BASE/'format_content/native_commits'/f'{sid}.json')['array_sha']
        with np.load(BASE/key) as z:
            assert z['H'].shape==(37,1,2560) and z['token_ids'].tolist()==r['prompt_ids']
            assert z['H12_sources'].shape==(len(r['prompt_ids']),2560)
    native={p.stem:read(p) for p in (BASE/'format_content/native_commits').glob('*.json')}
    assert len(native)==192
    for sid in future_protocol['long_baseline_old_ids']:
        prior=read(BASE/'capture/commits'/f'{sid}.json')
        assert native[sid]['original8token_prefix_exact']
        assert native[sid]['generated_ids'][:len(prior['generated_ids'])]==prior['generated_ids']
    trajectory_counts={}
    for area,expected_total in [('binding_live',80),('format_content/autonomous',336)]:
        count=Counter()
        for path in (BASE/area/'commits').glob('*/*.json'):
            r=read(path);count[r['branch']]+=1
            assert r['sample_id'] in lookup and len(r['steps'])==len(r['generated_ids'])
            assert 0<len(r['generated_ids'])<=(32 if area=='binding_live' else 64)
            if area.endswith('autonomous') and r['branch']=='native':assert r['long_native_prefix_exact']
        assert sum(count.values())==expected_total
        trajectory_counts[area]=dict(count)
    manifest=[];array_count=0;element_count=0;undefined=[]
    for i,path in enumerate(sorted(BASE.rglob('*.npz'))):
        key=path.relative_to(BASE).as_posix();digest=sha(path);entries=[]
        if key in expected:assert digest==expected[key],key
        with np.load(path,allow_pickle=False) as archive:
          for name in archive.files:
            a=archive[name];assert a.dtype.kind in 'buif',(key,name,str(a.dtype))
            v=unbits(a) if a.dtype==np.uint16 else a
            # No silent nan_to_num: unexpected nonfinite values fail the audit.
            assert np.isfinite(v).all(),(key,name,'Unexpected nonfinite saved values')
            entries.append({'array':name,**identity(a)})
            array_count+=1;element_count+=int(a.size);del a,v
        manifest.append({'file':key,'bytes':path.stat().st_size,'sha256':digest,'arrays':entries})
        if i<2 or (i+1)%256==0:print('BINDING_FULL_ARRAY_AUDIT',i+1,round(time.monotonic()-start,1),flush=True)
    compressed(BASE/'verification/all_array_identities.json.gz',manifest)
    import sys
    sys.path.insert(0,str(ROOT))
    from server.rdc_binding_service import index
    assert set(index())=={r['file'] for r in manifest}
    assert read(BASE/'verification/model_checkpoint_fingerprints.json')['all_passed']
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'all_passed':True,
      'original_memo_prefix_unchanged_bytes':contract['memo_prefix_bytes'],'prior_evidence_unchanged':True,
      'material_rows':len(allrows),'unique_material_ids':len(lookup),
      'new_native_Q4_material_occurrences':len(natural)+len(programs)+len(future)+len(signed),
      'new_native_Q4_capture_forwards':len(natural)+len(programs)+len(future)+127,
      'signed_source_identity_audit':identity_audit,
      'automatic_followup_new_natural_rows':len(signed),'automatic_followup_natural_components_independent':True,
      'strict_natural_fit_rows':320,'strict_training_connected_holdout_absence':True,
      'new_natural_components_absent_from_prior_two_material_sets':True,'program_semantic_groups':len(groups),
      'full_layer_all_token_fixtures':fixtures,'committed_capture_archive_sha_checks':len(expected),
      'all_npz_files':len(manifest),'all_arrays':array_count,'all_array_elements_checked':element_count,
      'every_saved_npz_registered':True,'undefined_values':undefined,'trajectories':trajectory_counts,
      'all_original_short_long_native_prefixes_exact':True,'seconds':time.monotonic()-start,
      'qwen14_preserved_first_attempt_rows':recovery['prior_commits_and_arrays_unchanged'],
      'qwen14_same_native_precision_and_device_map_after_resume':True,
      'retention':'All captured arrays remain connected to client queries or further analysis. Nothing deleted; original data and checkpoint files preserved.',
      'limits':'Hash, finite-value and split checks do not prove semantic causality or exclude historical model-pretraining contamination. No all-token/all-layer retention outside declared fixtures claimed.'}
    save(BASE/'verification/scientific_integrity.json',result);ledger('binding_full_array_integrity',result['seconds']);guard()
    print('BINDING_INTEGRITY_PASS',len(manifest),array_count,flush=True)

if __name__=='__main__':main()
