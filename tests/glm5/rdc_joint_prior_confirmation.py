"""Untuned application of old fitted rules to independent new natural materials."""
from rdc_joint_common import *
from rdc_joint_prior_rules import PriorRMS, rms_sources


def summarize(records, names, metrics):
    result = {}
    for split in ['all']+sorted({r['split'] for r in records}):
        rr = [r for r in records if split == 'all' or r['split'] == split]
        result[split] = {}
        for name in names:
            result[split][name] = {m: paired_summary([r['methods'][name][m] for r in rr], [r['source_group'] for r in rr]) for m in metrics}
    return result


def prior_confirmation(store):
    from phase2716_rdc_relation_probability import Readout
    old = PriorRMS()
    out = BASE/'prior_confirmation'
    snapshot(Path(__file__))
    with np.load(PREVIOUS/'boundary_compilation/first_source_affine.npz') as z:
        aa, bb = z['a'], z['b']
        common = (z['training_mean_h23']-z['training_mean_h12']).astype(np.float32)
    first_ids = {r['prompt_ids'][0] for r in old_rows() if r['split'] == 'train'}
    current, means, target, post, observed, meta, captures, first = [], [], [], [], [], [], [], []
    first_profiles = {}
    for r in store.material:
        z = store[r]
        h12, h23, h36 = (unbits(z[k]) for k in ('h12','h23','h36'))
        fp = {'identity': h12[0], 'common_increment': h12[0]+common, 'affine': h12[0]*aa+bb,
              'old_anchor_quadratic': old.quad(h12[0])[0, :2560]}
        frec = {k:r[k] for k in ('sample_id','source_group','split','language','genre')}
        frec.update(first_token_id=r['prompt_ids'][0], first_ID_seen_in_old_training=r['prompt_ids'][0] in first_ids,
                    target_energy=float(np.mean(h23[0].astype(float)**2)), methods={})
        for name, p in fp.items():
            e = (p.astype(float)-h23[0])**2
            frec['methods'][name] = {'MSE':float(e.mean())}
            first_profiles[name] = first_profiles.get(name, np.zeros(2560))+e
        first.append(frec)
        cp = read(BASE/'main/commits'/f'{r["sample_id"]}.json')
        for j,p in enumerate(r['anchors']):
            current.append(h12[p]); means.append(rms_sources(h12[:p]).mean(0))
            target.append(np.concatenate([h23[p],h36[p]])); post.append(unbits(z['postnorm'][2+2*j]))
            observed.append(r['prompt_ids'][p+1]); captures.append(cp)
            meta.append({k:r[k] for k in ('sample_id','source_group','split','language','genre')}|{'anchor':j,'position':p})
    current, means, target, post = [np.stack(v) for v in (current, means, target, post)]
    prediction = old(current, means)
    records, profiles = [], {}
    for name,p in prediction.items():
        profiles[name] = ((p.astype(float)-target)**2).mean(0).astype(np.float32)
    for i,m in enumerate(meta):
        methods = {name:{'H23_MSE':float(np.mean((p[i,:2560].astype(float)-target[i,:2560])**2)),
                         'H36_MSE':float(np.mean((p[i,2560:].astype(float)-target[i,2560:])**2))} for name,p in prediction.items()}
        records.append({**m,'methods':methods})
    compressed_json(out/'state_rows.json.gz', records)
    compressed_json(out/'first_rows.json.gz', first)
    npz(out/'all_coordinate_MSE.npz', **profiles, **{'first_'+k:(v/len(first)).astype(np.float32) for k,v in first_profiles.items()})
    state = summarize(records, prediction, ('H23_MSE','H36_MSE'))
    first_summary = summarize(first, first_profiles, ('MSE',))
    first_strata = {}
    for seen in (False, True):
        rr = [r for r in first if r['first_ID_seen_in_old_training'] == seen]
        first_strata[str(seen)] = {'sources':len(rr), 'methods':{k:paired_summary([r['methods'][k]['MSE'] for r in rr], [r['source_group'] for r in rr]) for k in first_profiles}}
    rd = Readout()
    probability = []
    try:
        for name,p in list(prediction.items())+[('actual_H36_FP32_floor',target)]:
            probability.append(rd.evaluate(p[:,2560:],post,observed,meta,out/'probability',name,'current','new_main_all',captures))
    finally:
        rd.close()
    gains = {}
    base = prediction['current']
    for name,p in prediction.items():
        gains[name] = {}
        for layer,a,b in [('H23',0,2560),('H36',2560,5120)]:
            d = np.mean((base[:,a:b].astype(float)-target[:,a:b])**2-(p[:,a:b].astype(float)-target[:,a:b])**2,1)
            gains[name][layer] = paired_summary(d,[m['source_group'] for m in meta])
    report = {'timestamp':stamp(),'independent_sources':len(store.material),'anchors':len(meta),
        'new_data_fitted_or_selected':False,'source_RMS_receipt':read(out/'reconstruction.json'),
        'state':state,'first':first_summary,'first_seen_strata':first_strata,'paired_current_minus_rule_MSE':gains,
        'probability':probability,
        'limits':'Independent relative to prior run materials, not unseen model pretraining. All new-main sources may confirm old frozen rules; new fits still use only train/validation. Fresh256 responses remain untouched. Mixed treebanks differ in genre/annotation/translation, not only language. No native block compilation tested here.'}
    save(out/'result.json',report)
    print('JOINT_PRIOR_CONFIRMED',state['all'],'first',first_summary['all'],flush=True)
    return report
