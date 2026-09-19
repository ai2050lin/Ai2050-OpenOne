"""Complete coordinate archives, source identities, frozen fits and rollouts audit."""
from collections import Counter,defaultdict
from rdc_update_common import *


def frozen_inputs():
    contract=read(BASE/'contract.json');memo=MEMO.read_bytes()
    assert hashlib.sha256(memo[:contract['memo_prefix_bytes']]).hexdigest()==contract['memo_prefix_sha256']
    for name,digest in contract['prior_evidence_sha256'].items():assert sha(PRIOR/name)==digest,name
    for item in contract['attachments']:
        assert sha(Path(item['path']))==sha(BASE/item['snapshot'])==item['sha256']
    assert sha(BASE/'plan.json')==contract['plan_sha256']
    material=read(BASE/'material_frozen.json')
    for name,key in [('natural_material.json.gz','natural_sha256'),('program_material.json.gz','program_sha256')]:
        assert sha(BASE/name)==material[key]
    for item in material['raw_sources']:assert sha(ROOT/item['path'])==item['archive_sha']
    assert sha(BASE/'language_material.json.gz')==read(BASE/'language_material_frozen.json')['material_sha256']
    for item in read(BASE/'fresh_graph/material_audit.json')['source_manifest']:assert sha(ROOT/item['path'])==item['archive_sha']
    for area,material_path in [('graph','natural_material.json.gz'),('fresh_graph','fresh_graph/material.json.gz')]:
        frozen=read(BASE/area/'frozen.json')
        assert sha(BASE/material_path)==frozen['material_sha256']
        assert sha(PRIOR/'natural_discovery.json.gz')==frozen['fit_material_sha256']
        assert sha(BASE/'graph/head_mapping.npz')==frozen['head_mapping_sha256']
        for name,digest in frozen['banks_sha256'].items():assert sha(BASE/area/'banks'/name)==digest,name
    learning=read(BASE/'learning/frozen.json')
    assert learning['material_sha256']==sha(BASE/'program_material.json.gz')
    assert learning['forecast_sha256']==sha(BASE/'learning/frozen_forecast.npz')
    for name,digest in learning['direction_sha256'].items():assert sha(BASE/'learning/directions'/name)==digest
    return contract


def main():
    start=time.monotonic();guard();contract=frozen_inputs()
    natural=gzread(BASE/'natural_material.json.gz');program=gzread(BASE/'program_material.json.gz')
    language=gzread(BASE/'language_material.json.gz');fresh=gzread(BASE/'fresh_graph/material.json.gz')
    prior=gzread(PRIOR/'natural_discovery.json.gz');rows=natural+program+language+fresh
    assert list(map(len,(natural,program,language,fresh,prior)))==[128,768,640,128,512]
    lookup={r['sample_id']:r for r in rows+prior};assert len(lookup)==2176
    assert len({tuple(r['prompt_ids']) for r in rows})==len(rows),'New input alias needs explicit audit'
    groups=defaultdict(set)
    for r in program+language:groups[r['source_group']].add(r['split'])
    assert all(len(v)==1 for v in groups.values())
    old_inventory=prior_natural()+gzread(LAW/'material.json.gz')+gzread(LAW/'confirmation_material.json.gz')
    old_components={c for r in old_inventory for c in r.get('component_ids',[])}
    c36={c for r in natural for c in r['component_ids']};c39={c for r in fresh for c in r['component_ids']}
    assert not (old_components&c36 or old_components&c39 or c36&c39)
    assert len(old_inventory+natural)==read(BASE/'fresh_graph/material_audit.json')['components_disjoint_against']
    train_docs={r['source_group'] for r in old_inventory if r['split']=='train'}
    assert not train_docs&{r['source_group'] for r in fresh}
    expected={};fixtures=[];alltoken_scanned=0
    for area,rr,field_dir in [('capture',natural+program,'qwen4'),('language_capture',language,'fields'),('fresh_graph',fresh,'fields')]:
        assert len(list((BASE/area/'commits').glob('*.json')))==len(rr)
        for row in rr:
            sid=row['sample_id'];commit=read(BASE/area/'commits'/f'{sid}.json');path=BASE/area/field_dir/f'{sid}.npz'
            expected[path.relative_to(BASE).as_posix()]=commit['array_sha256']
            with np.load(path,allow_pickle=False) as z:
                assert z['H'].shape==(37,len(row['anchors']),2560)
                assert z['H12_sources'].shape==(len(row['prompt_ids']),2560)
                assert z['token_ids'].tolist()==row['prompt_ids']
                assert np.array_equal(z['H'][12],z['H12_sources'][row['anchors']])
                identities=commit['alltoken_layer_identities'];assert set(identities)=={str(i) for i in range(37)}
                for layer,entry in identities.items():
                    assert entry['shape']==[len(row['prompt_ids']),2560] and entry['dtype']=='uint16'
                    alltoken_scanned+=len(row['prompt_ids'])*2560
                assert identity(z['H12_sources'])==identities['12']
                if commit.get('full_fixture'):
                    fp=BASE/area/'full_fields'/f'{sid}.npz'
                    with np.load(fp,allow_pickle=False) as f:
                        assert f['H'].shape==(37,len(row['prompt_ids']),2560)
                        assert np.array_equal(f['H'][:,row['anchors']],z['H'])
                        for layer in range(37):assert identity(f['H'][layer])==identities[str(layer)]
                    fixtures.append(fp.relative_to(BASE).as_posix())
    native_paths=read(BASE/'native_paths/result.json');assert native_paths['rows']==24
    for r in native_paths['reports']:
        key=f'native_paths/fields/{r["sample_id"]}.npz';expected[key]=r['archive_sha256']
        assert read(BASE/'native_paths/commits'/f'{r["sample_id"]}.json')==r
        original=lookup[r['sample_id']]
        with np.load(BASE/key,allow_pickle=False) as z:
            assert z['token_ids'].tolist()==original['prompt_ids'] and z['positions'].tolist()==original['anchors']
            for block in (16,35):
                a=z[f'L{block}_native_A'];writes=z[f'L{block}_source_attention_write']
                assert a.shape==(32,len(r['positions']),r['tokens'])
                assert writes.shape==(len(r['positions']),r['tokens'],2560)
                assert z[f'L{block}_source_gate_read'].shape==(len(r['positions']),r['tokens'],9728)
                for i,p in enumerate(r['positions']):
                    assert np.count_nonzero(a[:,i,p+1:])==0 and np.count_nonzero(writes[i,p+1:])==0
    shape_controls=[]
    for model in ('qwen4','qwen14','glm4'):
        area=BASE/'scale'/model;result=read(area/'result.json');runtime=result['runtime']
        assert result['rows']==128 and result['generation_rows']==36
        assert not runtime['quantized'] and runtime['dtype']=='torch.bfloat16'
        commits=list((area/'commits').glob('*.json'));assert len(commits)==128
        generation_count=0
        for p in commits:
            rec=read(p);key=f'scale/{model}/fields/{rec["sample_id"]}.npz';expected[key]=rec['array_sha256']
            with np.load(BASE/key,allow_pickle=False) as z:
                assert z['token_ids'].tolist()==rec['prompt_ids']
                assert z['H'].shape==(runtime['depth']+1,len(rec['positions']),runtime['width'])
                assert z['H_early_sources'].shape==(rec['tokens'],runtime['width'])
                assert np.array_equal(z['H'][runtime['early']],z['H_early_sources'][rec['positions']])
            if model=='qwen4':assert rec['original_capture_replay_exact']
            if 'generated_ids' in rec:
                generation_count+=1;ids=rec['generated_ids']
                assert 0<len(ids)<=128 and len(rec['steps'])==len(ids)
                assert all(step['step']==i and step['chosen']==ids[i] for i,step in enumerate(rec['steps']))
                assert rec['generation_prompt_ids']==rec['prompt_ids'][:len(rec['generation_prompt_ids'])]
                if model!='qwen4':assert rec['generation_execution_batch']['native_own_row_history']
        assert generation_count==36
        if model!='qwen4':
            assert runtime['execution_shape_protocol']['prefill_batch_max']==8
            shape_controls.extend(result['execution_shape_audit'])
    shadow=BASE/'scale_batch/qwen4';shadow_result=read(shadow/'result.json')
    assert shadow_result['rows']==128 and shadow_result['generation_rows']==0
    assert len(list((shadow/'commits').glob('*.json')))==128
    for p in (shadow/'commits').glob('*.json'):
        r=read(p);key=f'scale_batch/qwen4/fields/{r["sample_id"]}.npz';expected[key]=r['array_sha256']
        original=read(BASE/'scale/qwen4/commits'/p.name)
        assert r['prompt_ids']==original['prompt_ids'] and r['positions']==original['positions']
        assert r['execution_batch']['position_origin']==0 and r['execution_batch']['mask_excludes_padding']
        with np.load(BASE/key,allow_pickle=False) as z:
            assert z['H'].shape==(37,len(r['positions']),2560)
            assert np.array_equal(z['H'][12],z['H_early_sources'][r['positions']])
            assert z['token_ids'].tolist()==r['prompt_ids']
    shape_controls.extend(shadow_result['execution_shape_audit']);assert len(shape_controls)==18
    for model,folder in [('qwen4',shadow),('qwen14',BASE/'scale/qwen14'),('glm4',BASE/'scale/glm4')]:
        controls=read(folder/'shape_audit/result.json')['rows'];assert len(controls)==6
        for r in controls:
            with np.load(folder/'shape_audit'/f'{r["sample_id"]}.npz',allow_pickle=False) as z:
                difference=unbits(z['batch_H']).astype(float)-unbits(z['B1_H'])
                actual=float(np.linalg.norm(difference)/max(np.linalg.norm(unbits(z['B1_H'])),1e-30))
                assert abs(actual-r['all_H_relative_RMS'])<1e-12
                assert bool(np.array_equal(z['B1_H'],z['batch_H']))==r['batch_H_bit_exact']
    language_control=read(BASE/'language_identity/result.json')
    assert language_control['rows']==640 and language_control['material_sha256']==sha(BASE/'language_material.json.gz')
    assert len(language_control['comparisons'])==540
    for name,digest in language_control['unchanged_cosine_archives_sha256'].items():assert sha(BASE/name)==digest
    alignment=gzread(BASE/'language_identity/body_anchor_alignment.json.gz');assert len(alignment)==640
    assert sum(r['exact_character_endpoint'] for r in alignment)==language_control['exact_body_character_endpoints']
    for r in alignment:
        original=lookup[r['sample_id']];p=original['anchors'][0]
        assert r['anchor']==p and r['token_id']==original['prompt_ids'][p]
        assert r['actual_token_end']==original['token_offsets'][p][1]
    causal=read(BASE/'causal_anchor/result.json');assert causal['pairs']==320 and causal['semantic_groups']==160
    assert causal['native_position_config_sha256']==sha(ROOT/'models/hf/qwen3-4b/config.json')
    assert causal['all320_causal_prefix_IDs_exact'] and causal['all320_embedding_body_states_bit_exact']
    for name,digest in causal['unchanged_capture_archive_sha256'].items():assert expected[name]==digest
    pairs=gzread(BASE/'causal_anchor/pair_identity.json.gz');assert len(pairs)==320
    with np.load(BASE/'causal_anchor/all_pair_all_layer_differences.npz',allow_pickle=False) as z:
        assert z['relative_RMS'].shape==z['bit_equal'].shape==(320,37) and z['bit_equal'][:,0].all()
        assert int(z['bit_equal'][:,12].sum())==causal['H12_bit_exact_pairs']
    for r in pairs:
        a,b=lookup[r['direct_id']],lookup[r['explain_id']]
        assert a['prompt_ids'][:a['anchors'][0]+1]==b['prompt_ids'][:b['anchors'][0]+1]
    replay=read(BASE/'causal_replay/result.json');prefixes=gzread(BASE/'causal_replay/material.json.gz')
    assert replay['unique_prefixes']==len(prefixes)==80 and replay['semantic_groups']==40
    assert replay['native_forward_calls']==320 and replay['all_prefix_repeats_exact']
    assert replay['frozen_predictor_sha256']==sha(BASE/'graph/frozen.json')
    assert replay['causal_anchor_audit_sha256']==sha(BASE/'causal_anchor/result.json')
    prefix_lookup={r['sample_id']:r for r in prefixes};same_shape_count=0
    for r in replay['records']:
        row=prefix_lookup[r['sample_id']];original=lookup[r['sample_id']];p=original['anchors'][0]
        assert row['prompt_ids']==original['prompt_ids'][:p+1] and row['anchors']==[p]
        assert original['split']=='language_test' and original['answer_style']=='direct'
        key=f"causal_replay/fields/{r['sample_id']}.npz";expected[key]=r['array_sha256']
        with np.load(BASE/key,allow_pickle=False) as z:
            assert z['H'].shape==(37,1,2560) and z['H12_sources'].shape==(p+1,2560)
            assert z['token_ids'].tolist()==row['prompt_ids']
            same=(z['padded_direct_H']==z['padded_explain_H']).all((1,2))
            assert same.tolist()==r['same_length_suffix_exact_layers']
            assert bool(same.all())==r['same_length_suffix_all37_layers_exact'];same_shape_count+=int(same.all())
    assert same_shape_count==replay['same_length_future_suffix_all37_exact_pairs']
    trajectory_counts={}
    for area,total,cap in [('own_history',1056,128),('same_history',108,128),('long_answers',192,1024)]:
        result=read(BASE/area/'result.json');counts=Counter()
        for path in (BASE/area/'commits').glob('*/*.json'):
            r=read(path);sid=r['sample_id'];assert sid in lookup
            ids=r['generated_ids'];assert 0<len(ids)<=cap and len(r['steps'])==len(ids)
            counts[r['branch']]+=1
            if area=='same_history':
                original=read(BASE/'own_history/commits'/r['branch']/f'{sid}.json')
                assert r['same_main_rollout_IDs_exact'] and ids==original['generated_ids']
                assert r['prompt_ids']==original['prompt_ids']
            if area=='long_answers':
                original=read(BASE/'own_history/commits'/r['branch']/f'{sid}.json')
                assert r['short128_prefix_exact'] and ids[:len(original['generated_ids'])]==original['generated_ids']
            for i,s in enumerate(r['steps']):
                assert s['step']==i
                token=s.get('token_id',s.get('chosen',s.get('main_chosen')))
                if token is not None:assert token==ids[i]
        assert sum(counts.values())==total,(area,counts)
        trajectory_counts[area]=dict(counts)
    scoring=read(BASE/'behavior_analysis/result.json');assert scoring['final'] and scoring['trajectories']==1464
    terminal=read(BASE/'terminal_format_audit/result.json')
    assert terminal['all_passed'] and terminal['primary_scorer_unchanged'] and terminal['all_trajectory_records']==1464
    manual=read(BASE/'manual_terminal_audit/result.json')
    assert manual['all_passed'] and manual['unchanged_generation_count']==192
    assert manual['annotations_sha256']==sha(BASE/'manual_terminal_audit/annotations.json')
    assert manual['inventory_sha256']==sha(BASE/'manual_terminal_audit/inventory.json')
    for item in manual['adjudications']:assert sha(BASE/item['raw_record'])==item['raw_sha256']
    assert scoring['format_audit_protocol_sha256']==sha(BASE/'terminal_format_audit/protocol.json')
    for name,digest in scoring['original_commit_sha256'].items():assert sha(BASE/name)==digest
    assert read(BASE/'learning/autograd_audit.json')['all_passed']
    logic=read(BASE/'language_analysis/logic_audit.json')
    assert logic['semantic_cases']==160 and logic['expressions']==640
    english_truth={r['source_group']:r['truth'] for r in language if r['language']=='en'}
    assert len(logic['checks'])==160
    for r in logic['checks']:
        assert r['all4_expression_labels_agree']
        assert r['independent_English_text_logic']==english_truth[r['source_group']]
    assert read(BASE/'moment_boundary/result.json')['all_passed']
    assert read(BASE/'verification/model_checkpoint_fingerprints.json')['all_passed']
    # All source versions are permanent snapshots; current editable code may be newer.
    snapshots={}
    def visit(value):
        if isinstance(value,dict):
            if set(('snapshot','sha256')).issubset(value) and isinstance(value['snapshot'],str):
                path=BASE/value['snapshot']
                if path.exists():snapshots[str(path)]=value['sha256']
            for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    for path in BASE.rglob('*.json'):
        if 'verification' not in path.relative_to(BASE).parts:visit(read(path))
    for path,digest in snapshots.items():assert sha(Path(path))==digest,path
    manifest=[];arrays=0;elements=0
    for i,path in enumerate(sorted(BASE.rglob('*.npz'))):
        key=path.relative_to(BASE).as_posix();digest=sha(path);entries=[]
        if key in expected:assert digest==expected[key],key
        with np.load(path,allow_pickle=False) as z:
            for name in z.files:
                a=z[name];assert a.dtype.kind in 'buif',(key,name,str(a.dtype))
                v=unbits(a) if a.dtype==np.uint16 else a
                assert np.isfinite(v).all(),(key,name,'Nonfinite value; no implicit replacement allowed')
                entries.append({'array':name,**identity(a)});arrays+=1;elements+=a.size;del a,v
        manifest.append({'file':key,'bytes':path.stat().st_size,'sha256':digest,'arrays':entries})
        if i<2 or (i+1)%256==0:print('UPDATE_ARRAY_AUDIT',i+1,round(time.monotonic()-start,1),flush=True)
    compressed(BASE/'verification/all_array_identities.json.gz',manifest)
    import sys
    sys.path.insert(0,str(ROOT));from server.rdc_update_service import index
    assert set(index())=={r['file'] for r in manifest}
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
      'original_memo_prefix_unchanged_bytes':contract['memo_prefix_bytes'],'prior_frozen_evidence_unchanged':True,
      'client_material_ids':2176,'new_native_Q4_materials':1664,'controlled_semantic_groups':len(groups),
      'alltoken_alllayer_coordinate_values_scanned_at_capture':alltoken_scanned,
      'full_layer_all_token_fixtures':fixtures,'new_natural_components_disjoint':True,'fresh_documents_disjoint_from_fit':True,
      'committed_native_archive_hash_checks':len(expected),'code_snapshot_checks':len(snapshots),
      'matched_shape_Q4_extra_archives':128,'independent_native_B1_shape_controls':len(shape_controls),
      'shape_controls_assert_numerical_equality':False,'language_identity_position_controls':540,
      'same_body_causal_prefix_all37_layer_controls':320,
      'prefix_only_recovery_unique_prefixes':80,'prefix_only_recovery_native_calls':320,
      'matched_length_future_suffix_exact_pairs':same_shape_count,
      'all_npz_files':len(manifest),'all_arrays':arrays,'all_array_elements_checked':int(elements),
      'every_saved_npz_client_registered':True,'trajectory_counts':trajectory_counts,
      'formal_trajectory_records':scoring['trajectories'],'all_same_history_and_short_long_prefixes_exact':True,
      'secondary_terminal_format_recoveries':terminal['recovered_records'],'original_terminal_scorer_unchanged':True,
      'manual_posthoc_terminal_reviews':manual['reviewed_residual_outputs'],
      'seconds':time.monotonic()-start,
      'retention':'All saved arrays retained for client queries, mechanism evidence or continued analysis; no material deletion. Full per-token/per-layer values retained only in listed fixtures, other full scans have hashes and recomputation inputs.',
      'limits':'Finite/hash/split tests verify recorded computations, not semantic causality, original pretraining identity, or universal predictive sufficiency. Repeated trajectories are not independent samples.'}
    save(BASE/'verification/scientific_integrity.json',result);ledger('update_full_array_integrity',result['seconds'])
    print('UPDATE_INTEGRITY_PASS',len(manifest),arrays,flush=True)


if __name__=='__main__':main()
