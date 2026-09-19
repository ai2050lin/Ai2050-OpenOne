"""Repair an occurrence-ID collision without changing frozen inputs or fits.

The original material, protocol, frozen banks, and first evaluation survive.
An exact-input duplicate is an explicit deterministic field reuse, not a new
native forward or an independent language observation. Frozen shuffle seeds
remain unchanged and every recomputed numerical array must be bitwise equal.
"""
from collections import defaultdict
from rdc_binding_common import *

def main():
    start=time.monotonic();out=BASE/'signed_source';audit=out/'identity_recovery'
    if (audit/'result.json').exists():
        assert read(audit/'result.json')['all_passed'];return
    guard(100*1024**2)
    rows=gzread(out/'natural_material.json.gz');buckets=defaultdict(list)
    for i,r in enumerate(rows):buckets[r['sample_id']].append(i)
    duplicates={k:v for k,v in buckets.items() if len(v)>1}
    assert len(rows)==128 and len(buckets)==127 and len(duplicates)==1
    assert len({tuple(r['prompt_ids']) for r in rows})==127
    fixed=[dict(r) for r in rows];aliases=[]
    frozen_paths=[out/'natural_material.json.gz',out/'protocol.json',out/'frozen.json',*sorted((out/'banks').glob('*.npz'))]
    frozen_hashes={p.relative_to(BASE).as_posix():sha(p) for p in frozen_paths}
    for sid,indices in duplicates.items():
        first=rows[indices[0]]
        assert len(indices)==2
        for i in indices[1:]:
            row=rows[i]
            for key in ('text','prompt_ids','anchors','cohort','split','source_group','capture_mode'):
                assert row[key]==first[key],key
            assert not set(row['component_ids']).intersection(first['component_ids'])
            new='b2735_'+ranked('signed-occurrence/'+json.dumps([row['cohort'],row['source_group'],row['component_ids']],ensure_ascii=False))[:20]
            assert new not in buckets
            fixed[i].update(sample_id=new,frozen_sample_id=sid,frozen_feature_seed_id=sid,
                field_reuse={'source_sample_id':sid,'reason':'Identical complete native prompt_ids, anchors, text and shared runtime; distinct document occurrence; not a new forward.'})
            src=out/'fields'/f'{sid}.npz';dst=out/'fields'/f'{new}.npz';original=read(out/'commits'/f'{sid}.json')
            assert not original['full_fixture'] and sha(src)==original['array_sha']
            with np.load(src) as z:
                assert z['token_ids'].tolist()==row['prompt_ids'] and z['positions'].tolist()==row['anchors']
            if not dst.exists():shutil.copyfile(src,dst)
            assert sha(dst)==sha(src)
            save(out/'commits'/f'{new}.json',dict(original,sample_id=new,seconds=0.0,
                native_forward_executed=False,field_reuse=fixed[i]['field_reuse'],frozen_sample_id=sid))
            aliases.append({'row_index':i,'frozen_sample_id':sid,'resolved_sample_id':new,
                'component_ids':row['component_ids'],'source_group':row['source_group'],
                'prompt_token_count':len(row['prompt_ids']),'anchors':row['anchors'],'array_sha256':sha(src),
                'complete_model_inputs_and_capture_positions_equal':True,'original_native_record_unchanged':True})
    assert len({r['sample_id'] for r in fixed})==128
    compressed(audit/'resolved_material.json.gz',fixed)
    derivative=[out/'result.json',out/'capture_result.json',out/'connected_visibility.json.gz',out/'all_confirmation_signed_features.npz',*sorted((out/'confirmation_predictions').glob('*.npz'))]
    preserved=[]
    for path in derivative:
        dest=audit/'initial_evaluation'/path.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(path,dest)
        if path.exists():assert sha(path)==sha(dest)
        preserved.append({'original':path.relative_to(BASE).as_posix(),'preserved':dest.relative_to(BASE).as_posix(),'sha256':sha(dest)})
    # Move this exact derivative out of the completion-marker location, keeping
    # both a pre-audit copy and the original bytes. No frozen input is moved.
    if (out/'result.json').exists():(out/'result.json').rename(audit/'initial_result_original.json')
    if (out/'connected_visibility.json.gz').exists():
        (out/'connected_visibility.json.gz').rename(audit/'initial_visibility_original.json.gz')
    from phase2735_rdc_binding_signed import evaluate
    evaluate()
    exact=[]
    for item in preserved:
        if not item['original'].endswith('.npz'):continue
        with np.load(BASE/item['original']) as now,np.load(BASE/item['preserved']) as old:
            assert now.files==old.files
            for name in now.files:assert np.array_equal(now[name],old[name]),(item['original'],name)
            exact.append({'file':item['original'],'all_arrays_bitwise_equal':True})
    result=read(out/'result.json');before=read(audit/'initial_result_original.json')
    for key in ('reports','paired','selected','visibility_counts'):assert result[key]==before[key],key
    for path,digest in frozen_hashes.items():assert sha(BASE/path)==digest
    original=read(audit/'initial_evaluation/capture_result.json')
    save(out/'capture_result.json',dict(original,rows=128,unique_native_inputs=127,native_forward_count=127,
      audited_identity_aliases=1,identity_correction_timestamp=stamp(),identity_audit='signed_source/identity_recovery/result.json'))
    save(audit/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'all_passed':True,
      'frozen_material_rows':128,'original_unique_ids':127,'resolved_unique_ids':128,
      'native_forward_count':127,'unique_native_inputs':127,'explicit_field_alias_count':1,
      'aliases':aliases,'frozen_inputs_and_predictors_unchanged':frozen_hashes,
      'resolved_material_sha256':sha(audit/'resolved_material.json.gz'),'preserved':preserved,'exact_recomputed_arrays':exact,
      'all_statistical_results_unchanged':True,'frozen_shuffle_seed_ids_preserved':True,
      'limits':'Source occurrences are distinct but two complete model inputs are identical within one source document. Neither occurrence is an independent replicate; original source-cluster weighting is preserved, not relabelled as unique text evidence.',
      'seconds':time.monotonic()-start})
    ledger('binding_occurrence_identity_audit',time.monotonic()-start);guard()
    print('BINDING_OCCURRENCE_IDENTITY_PASS',128,127,len(exact),flush=True)

if __name__=='__main__':main()
