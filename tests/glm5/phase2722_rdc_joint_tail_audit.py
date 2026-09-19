"""Independent provenance and frozen event-feature implementation checks, no new model outputs."""
from scipy.special import expit
from rdc_joint_common import *
from phase2719_rdc_joint_material import old_identities


def main():
    out=BASE/'extension/tail_confirmation';material=json.loads(gzip.decompress((out/'material.json.gz').read_bytes()));prior=rows()+rows(True)+old_rows()
    for name in ('material_stratified.json','confirmation_material.json','full_source_history/fresh_material.json'):prior.extend(read(PREFIX/name))
    excluded=old_identities();docs=set();pud=set()
    for r in prior:
        excluded[0].update(r.get('normalized_texts',[]));excluded[1].update(r.get('numeric_families',[]));excluded[2].update(r.get('construction_families',[]))
        excluded[3].update((r['language'],sid) for sid in r.get('component_ids',[r['source_sentence_id']]))
        if r['language']=='en':docs.update(sid.rsplit('-',1)[0] for sid in r.get('component_ids',[r['source_sentence_id']]))
        if r.get('treebank')=='pud':pud.update(r['component_ids'])
    seen_pud=set();current_components=set();patterns=[set(),set(),set()]
    for r in material:
        assert not set((r['language'],s) for s in r['component_ids'])&(excluded[3]|current_components)
        current_components.update((r['language'],s) for s in r['component_ids'])
        for i,key in enumerate(('normalized_texts','numeric_families','construction_families')):
            assert not set(r[key])&(excluded[i]|patterns[i]),(r['sample_id'],key)
            patterns[i].update(r[key])
        if r['source_key']=='ewt_dev':assert r['document_id'] not in docs
        if r['treebank']=='pud':
            assert not set(r['component_ids'])&(pud|seen_pud)
            seen_pud.update(r['component_ids'])
    original_meta=json.loads(gzip.decompress((BASE/'extension/all_token_amplitudes.json.gz').read_bytes()))
    with np.load(BASE/'extension/event_forecast.npz') as z:fit={k:z[k] for k in z.files}
    with np.load(BASE/'extension/event_forecast_all_token_predictions.npz') as z:expected=z['full_H12_linear_logistic']
    groups={};offset=0
    for r in rows()+rows(True):groups[r['sample_id']]=(r,offset);offset+=len(r['prompt_ids'])
    checked=0;maxdiff=0.
    probe_ids={r['sample_id'] for r in original_meta if r['event']}|{r['sample_id'] for r in (rows()[:8]+rows(True)[:8])}
    for sid in sorted(probe_ids):
        r,offset=groups[sid];z=field(r,r['split']=='confirmation');x=unbits(z['h12']);xx=((x-fit['mean'])/fit['standard_deviation']).astype(float)
        p=expit(xx@fit['coefficient']+fit['intercept'][0]);diff=float(np.max(np.abs(p-expected[offset:offset+len(p)])));maxdiff=max(maxdiff,diff);checked+=len(p)
    assert maxdiff<6e-8,maxdiff
    protocol=read(out/'protocol.json');assert sha(out/'material.json.gz')==protocol['material_sha']
    assert all(sha(BASE/name)==digest for name,digest in protocol['frozen_files'].items())
    report={'timestamp':stamp(),'passed':True,'source':snapshot(Path(__file__)),'sources':len(material),'tokens':sum(len(r['prompt_ids']) for r in material),
        'old_and_within_new_components_text_numeric_skeleton_disjoint':True,'ewt_document_disjoint':True,'PUD_parallel_ids_disjoint':True,
        'predictor_source_checks':len(probe_ids),'predictor_token_checks':checked,'max_abs_frozen_prediction_difference':maxdiff,
        'new_target_responses_observed':(out/'result.json').exists(),'limits':'Provenance identity checks do not establish article independence, semantic paraphrase exclusion, or exclusion from pretraining.'}
    save(out/'material_and_predictor_audit.json',report);print('TAIL_PROVENANCE_AND_IMPLEMENTATION_PASS',report,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
