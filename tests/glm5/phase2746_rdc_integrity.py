"""New-phase integrity without rewriting the completed2745 audit receipts."""
from collections import Counter,defaultdict
from rdc_construction_common import *
from rdc_construction_storage import verify_storage
import phase2745_rdc_construction_integrity as existing

OUT=BASE/'phase2746/verification'


def main():
    start=time.monotonic();verify_storage();OUT.mkdir(parents=True,exist_ok=True)
    prior=BASE/'verification/archive_cache.json';cache=OUT/'archive_cache.json'
    if not cache.exists():
        shutil.copyfile(prior,cache)
        immutable(OUT/'prior_audit_source.json',{'timestamp':stamp(),'source':snapshot(__file__),'previous_cache_sha256':sha(prior),
            'previous_result_sha256':sha(BASE/'verification/archives.json'),'method':'Reuse earlier finite/CRC checks only after fresh SHA256 of each original archive. All new arrays fully streamed. Original2745receipts remain untouched.'})
    existing.OUT=OUT
    existing.checkpoints();coverage=existing.archives()
    inventory={r['path']:r for r in read(cache)};checks=[]
    contract=read(BASE/'protocol.json')
    assert hashlib.sha256(MEMO.read_bytes()[:contract['memo_original_bytes']]).hexdigest()==contract['memo_original_sha256']
    for r in contract['required_prior_artifacts']:assert sha(OLD/r['path'])==r['sha256']
    for p in (BASE/'sources').iterdir():assert sha(p).startswith(p.stem.rsplit('_',1)[-1]),p
    checks.append('Old14priorartifacts, completepre2745MEMOprefix and every immutable source snapshot preserved')
    for prefix,n in [('phase2746/runtime',896),('phase2746/history_prediction/confirmation/native',512)]:
        rr=gzread(BASE/prefix/'records.json.gz');assert len(rr)==n==len({r['sample_id'] for r in rr})
        for r in rr:
            assert inventory[r['field_path']]['sha256']==r['field_sha256']
            with np.load(BASE/r['field_path']) as z:
                steps=len(r['generated_ids']);assert z['generated_ids'].tolist()==r['generated_ids']
                assert z['hidden'].shape==(steps,37,2560) and z['units'].shape==(min(8,steps),36,3,9728)
                assert z['Q_before_RoPE'].shape==(min(8,steps),36,32,128)
                assert z['coordinates'].shape==(min(8,steps),36,5,2560)
                assert z['positions'].tolist()==list(range(len(r['actual_prompt_ids'])-1,len(r['actual_prompt_ids'])-1+steps))
        checks.append(prefix+': every committed sample/step/token/layer/unit/query identity verified')
    pdir=BASE/'phase2746/history_prediction';frozen=read(pdir/'frozen.json')
    assert sha(pdir/'validation/selected.json')==frozen['validation_result_sha256']
    points=gzread(pdir/'material.json.gz');assignments=defaultdict(set)
    for r in points:assignments[r['source_group']].add(r['split'])
    assert len(points)==2688 and all(len(v)==1 for v in assignments.values())
    for phase,n in [('test/heldout.json',816),('confirmation/prediction/result.json',1536)]:
        meta=read(pdir/phase);assert meta['points']==n and len(meta['routes'])==20
        assert inventory[meta['field_path']]['sha256']==meta['field_sha256']
    confirmation=read(pdir/'confirmation/protocol.json');rows=gzread(pdir/'confirmation/material.json.gz')
    old=gzread(pdir/'confirmation/excluded_inventory.json.gz');oldgroups={r['source_group'] for r in old};oldtokens={tuple(r['prompt_ids']) for r in old}
    assert len(rows)==512 and len({r['source_group'] for r in rows if r['kind']=='natural'})==192
    assert confirmation['frozen_predictor_sha256']==sha(pdir/'frozen.json')
    for r in rows:
        if r['kind']=='natural':assert r['source_group'] not in oldgroups and tuple(r['prompt_ids']) not in oldtokens
    pairs=defaultdict(list)
    for r in rows:
        if r['kind']=='controlled':pairs[r['pair_id']].append(r)
    assert len(pairs)==160
    for pair in pairs.values():
        assert len(pair)==2 and Counter(pair[0]['prompt_ids'])==Counter(pair[1]['prompt_ids'])
        assert pair[0]['target']!=pair[1]['target'] and pair[0]['question']==pair[1]['question']
    checks.append('Frozen fit selection, source splits, all1536new forecasts and every new-source/token-pair exclusion audited')
    own=gzread(pdir/'confirmation/autonomous/records.json.gz');assert len(own)==1536
    byid={r['sample_id']:r for r in rows};native={r['sample_id']:r for r in own if r['route']=='native_B1_cache'}
    for r in own:
        assert inventory[r['field_path']]['sha256']==r['field_sha256']
        init=r['shared_initialization'];assert inventory[init['field_path']]['sha256']==init['field_sha256']
        with np.load(BASE/r['field_path']) as z:
            assert z['generated_ids'].tolist()==r['generated_ids'];steps=len(r['generated_ids'])
            if r['route']!='native_B1_cache':
                assert r['post_initialization_teacher_cache_injections']==r['true_current_late_state_inputs']==0
                assert r['first_step_all_early_H_and_KV_bit_equal_native']
                assert z['predicted_H35_H36_Q35'].shape==(steps,9216) and z['early_H0_H12'].shape==(steps,13,2560)
                with np.load(BASE/native[r['sample_id']]['field_path']) as nz:assert np.array_equal(z['early_H0_H12'][0],nz['hidden'][0,:13])
        with np.load(BASE/init['field_path']) as z:
            row=byid[r['sample_id']];assert z['prompt_ids'].tolist()==row['prompt_ids']
            assert z['prefix_keys'].shape==z['prefix_values'].shape==(13,8,len(row['prompt_ids'])-1,128)
    checks.append('All1536ownhistories preserve IDs, past-only initialization and all1024same-shape initial early-H checks; no teacher refresh')
    ownanalysis=read(pdir/'confirmation/autonomous/analysis.json')
    assert ownanalysis['detailed_metrics_file']=='metrics_corrected.json.gz'
    corrected=gzread(pdir/'confirmation/autonomous'/ownanalysis['detailed_metrics_file'])
    pairs=gzread(pdir/'confirmation/autonomous'/ownanalysis['pair_metrics_file'])
    for pair in pairs:
        rr=[r for r in corrected if r.get('pair_id')==pair['pair_id'] and r['route']==pair['route']]
        assert pair['opposite_first_tokens']==(rr[0]['first_emitted_token_id']!=rr[1]['first_emitted_token_id'])
    checks.append('Corrected auxiliary pair label reflects actual output IDs; original frozen tables preserved in separate revisions')
    for file in ['phase2746/client/runtime/result.json','client/code_checks.json','phase2746/figures/index.json']:
        assert read(BASE/file)['all_passed']
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':checks,'archive_coverage':coverage,
        'original_checkpoint_bytes_rehashed':read(OUT/'model_checkpoint_fingerprints.json')['bytes_rehashed'],
        'science_boundary':'Integrity verifies actual computation and retained data, not semantic closure. Autonomous failures remain failures.',
        'seconds':time.monotonic()-start}
    save(OUT/'result.json',result);ledger('phase2746_integrity',result['seconds']);print('PHASE2746_INTEGRITY_COMPLETE',len(checks),result['seconds'],flush=True)


if __name__=='__main__':main()
