"""Full saved-array, prediction freeze, native branch and read-only source audit."""
import hashlib
from rdc_law_common import *


def main():
    start=time.monotonic();guard();out=BASE/'verification'
    review=read(BASE/'review.json');checks=[]
    for r in review['evidence']:
        assert sha(ROOT/r['path'])==r['sha256'],r['path'];checks.append(r['path'])
    for r in review['attachments']:
        assert sha(Path(r['path']))==r['sha256'],r['path'];checks.append(r['path'])
    prefix=read(BASE/'memo_prefix.json');memo=MEMO.read_bytes()
    assert hashlib.sha256(memo[:prefix['bytes']]).hexdigest()==prefix['sha256']
    frozen=read(BASE/'prediction/frozen.json')
    assert sha(BASE/'material.json.gz')==frozen['material_sha']
    assert sha(BASE/'prediction/protocol.json')==frozen['protocol_sha']
    for name,digest in frozen['all_banks_sha'].items():assert sha(BASE/'prediction'/name)==digest,name
    assert sha(BASE/'prediction/frozen.json')==read(BASE/'confirmation/result.json')['frozen_predictors_sha']
    expected_files={};full_fields={};tokens=0;anchors=0
    material=gzread(BASE/'material.json.gz')+gzread(BASE/'confirmation_material.json.gz')
    for row in material:
        mode='confirmation' if row['split']=='confirmation' else 'main';folder=BASE/'capture'/mode;sid=row['sample_id']
        commit=read(folder/'commits'/f'{sid}.json');assert commit['sample_id']==sid
        for area,key in [('fields','field_sha'),('sources','source_sha'),('energies','energy_sha')]:
            expected_files[(folder/area/f'{sid}.npz').relative_to(BASE).as_posix()]=commit[key]
        with np.load(folder/'fields'/f'{sid}.npz') as z:assert z['H'].shape==(37,len(row['anchors']),2560)
        with np.load(folder/'sources'/f'{sid}.npz') as z:
            assert any(z[k].shape==(len(row['prompt_ids']),2560) for k in z.files)
        if commit['full_field']:full_fields[(folder/'full_fields'/f'{sid}.npz').relative_to(BASE).as_posix()]=commit['full_layer_all_token_identities']
        tokens+=len(row['prompt_ids']);anchors+=len(row['anchors'])
    assert (len(material),tokens,anchors,len(full_fields))==(1152,204097,3072,12)
    records=[];undefined=[];arrays=0;elements=0
    # Deliberately inspect every saved tensor. No sampled files, selected units,
    # coordinate sorting, numerical threshold or float16 reinterpretation shortcut.
    for i,path in enumerate(sorted(BASE.rglob('*.npz'))):
        key=path.relative_to(BASE).as_posix();digest=sha(path)
        if key in expected_files:assert digest==expected_files[key],key
        entries=[]
        with np.load(path,allow_pickle=False) as z:
          for name in z.files:
            a=z[name];assert a.dtype.kind in 'buif', (key,name,a.dtype)
            arrays+=1;elements+=a.size;v=unbits(a) if a.dtype==np.uint16 else a
            bad=~np.isfinite(v)
            if bad.any():
                n=int(bad.sum());valid=False;meaning=''
                if name in ('next_NLL','next_nll') and '/energies/' in '/'+key and v.ndim==1:
                    valid=n==1 and bool(np.isnan(v[-1]));meaning='No supplied nexttoken after last materialposition'
                if key=='formation/gradient_controls/conditional_permutations.npz' and name=='also_cohort_current_token':
                    valid=v.shape==(256,) and bool(np.isnan(v).all());meaning='Exactcohort/currenttoken matching leaves zero evaluable comparisons; all256 null statistics explicitly unavailable'
                assert valid,(key,name,np.argwhere(bad)[:5].tolist())
                undefined.append({'file':key,'array':name,'count':n,'meaning':meaning})
            ident=identity(a)
            if key in full_fields and name=='H':
                for layer in range(37):assert identity(a[layer])==full_fields[key][str(layer)],(key,layer)
            entries.append({'array':name,**ident});del a,v,bad
        records.append({'file':key,'bytes':path.stat().st_size,'sha256':digest,'arrays':entries})
        if i<2 or (i+1)%512==0:print('LAW_COMPLETE_ARRAY_AUDIT',i+1,'seconds',round(time.monotonic()-start,1),flush=True)
    compressed(out/'all_array_identities.json.gz',records)
    import sys
    sys.path.insert(0,str(ROOT))
    from server.rdc_law_service import arrays_index
    assert set(arrays_index())=={r['file'] for r in records}
    counts={};scale_fields=0
    for model in ('qwen4','qwen14','glm4'):
        folder=BASE/'scale'/model;r=read(folder/'result.json')
        assert r['natural_windows']==144 and r['natural_anchors']==432 and r['QA_questions']==24
        assert len(list((folder/'fields').glob('*.npz')))==144
        commits=list((folder/'qa/commits').glob('*.json'));assert len(commits)==24
        for p in commits:
            case=read(p);assert (folder/'qa/fields'/f"{case['sample_id']}.npz").exists()
            assert case['generated_ids'] and len(case['generated_ids'])<=32
        counts[model]=len(commits);scale_fields+=144
    deployment=read(BASE/'deployment/result.json');assert deployment['trajectories']==672
    assert deployment['original_rollout_bytes_preserved_on_resume']
    resume=read(BASE/'deployment/resume_manifest.json');assert resume['completed_rollout_commits']==672
    for name,digest in resume['files_sha256'].items():assert sha(BASE/name)==digest,name
    assert read(BASE/'deployment/scope_precision/result.json')['passed']
    assert len(list((BASE/'deployment/scope/matched_shape_commits').glob('*.json')))==24
    protocol=read(BASE/'deployment/protocol.json');trajectory_counts={}
    for branch in protocol['rollout_branches']:
        commits=list((BASE/'deployment/rollouts'/branch).glob('*.json'));assert len(commits)==96
        for p in commits:
            r=read(p);assert r['branch']==branch and 0<len(r['generated_ids'])<=48
            assert len(r['steps'])==len(r['generated_ids']) and (BASE/'deployment/rollout_fields'/branch/f"{r['sample_id']}.npz").exists()
        trajectory_counts[branch]=len(commits)
    own=read(BASE/'own_history/result.json');assert own['trajectories']==96 and own['all_main_rollouts_exact']
    for p in (BASE/'own_history/commits').glob('*/*.json'):
        r=read(p);original=read(BASE/'deployment/rollouts'/r['branch']/f"{r['sample_id']}.json")
        assert r['generated_ids']==original['generated_ids'] and r['main_rollout_IDs_exact']
        if r['branch']!='early_prediction_L16':assert all(c['all_bitwise_equal'] for c in r['KV_comparisons'])
    training=read(BASE/'formation/trajectories/result.json')
    assert training['all_target_draws_matched'] and not training['confirmation_used']
    overhead=training['seconds_total_wall']-sum(r['seconds'] for r in training['runs']);assert overhead>=0
    if not any(r['kind']=='training_common_setup_overhead' for r in read(BASE/'compute_ledger.json')):
        ledger('training_common_setup_overhead',overhead,scope='Measuredtotal59.87123619997874 minus four alreadybooked trainingruns; no doublecount.')
    for name in ('math','prediction_math','cache_math'):
        assert read(out/f'{name}.json')['passed'],name
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'passed':True,'seconds':time.monotonic()-start,
        'prior_evidence_and_attachment_SHA_checks':checks,'original_memo_prefix_bytes_unchanged':prefix['bytes'],
        'frozen_prediction_bank_SHA_checks':len(frozen['all_banks_sha']),'main_and_confirmation_rows':len(material),
        'actual_main_confirmation_tokens':tokens,'main_confirmation_anchors':anchors,'full_all_layer_token_fixtures':len(full_fields),
        'committed_core_archive_SHA_checks':len(expected_files),'all_npz_files':len(records),'all_arrays':arrays,'all_array_elements_checked':int(elements),
        'all_stored_npz_paths_registered':True,'undefined_values':undefined,'scale_natural_field_files':scale_fields,'native_scale_QA':counts,
        'main_trajectories':trajectory_counts,'own_history_trajectories':96,'all_own_history_trajectories_replay_exact':True,
        'all_arrays_manifest':'verification/all_array_identities.json.gz','training_common_setup_seconds_added_once':overhead,
        'retention':'Every saved numerical array remains accessible from read-onlyclient registry. Nooriginalfields/userfiles removed. Streamed fulltoken/alllayerfields outside12fixtures are not falsely claimed retained.',
        'limits':'Hashes and numeric integrity do not establish semantic causality, independent observations or elimination of originalpretraining contamination.'}
    save(out/'scientific_integrity.json',result);ledger('complete_law_array_integrity',result['seconds']);guard()
    print('LAW_SCIENTIFIC_INTEGRITY_PASS',len(records),arrays,flush=True)


if __name__=='__main__':main()
