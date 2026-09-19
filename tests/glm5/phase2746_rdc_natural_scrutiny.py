"""Matched-depth and exact cohort-composition checks; no new model inference."""
from collections import Counter
from phase2746_rdc_natural_interactions import analyze, OUT as NATURAL
from rdc_construction_common import *

OUT = BASE/'phase2746/natural_scrutiny'


def main():
    start = time.monotonic()
    if (OUT/'result.json').exists():
        return
    protocol = {'timestamp':stamp(),'source':snapshot(__file__),
        'question':'Separate depth from sample composition, and within-cohort from between-cohort query interaction.',
        'status':'Prospective calculation after inspection of the pooled natural ANOVA; exploratory, not independent confirmation.',
        'matched_depth':'Use the exact same576registered detailed sources at H12/H24/H36 and postnorm.',
        'identity':'Global interaction energy = weighted within-cohort interaction energy + weighted energy of cohort-specific query effects minus the global query effect.',
        'weights':['equal_window','equal_document'], 'views':['raw','whole_vector_RMS'],
        'scope':'Cohorts are source-corpus metadata, not semantic modules. Full native coordinates; no top-k, causal percentage or new theorem.'}
    immutable(OUT/'protocol.json',protocol)
    rows = gzread(OLD/'material/natural.json.gz')
    probes = read(OLD/'probes/protocol.json')['probes']
    detailed = set(read(OLD/'material/protocol.json')['detailed_prefix_ids'])
    selected = [r for r in rows if r['sample_id'] in detailed]
    target = OUT/'matched_postnorm'
    if not (target/'layer_postnorm.json').exists():
        analyze(selected,probes,'postnorm',target)
    matched = read(target/'layer_postnorm.json')
    assert sha(target/'layer_postnorm.npz') == matched['archive_sha256']
    reports = [r for r in matched['reports'] if r['cohort']=='all']
    for layer in [12,24,36]:
        packet=read(NATURAL/f'layer_{layer}.json')
        assert [r['sample_id'] for r in packet['row_index']] == [r['sample_id'] for r in matched['row_index']]
        reports += [r for r in packet['reports'] if r['cohort']=='all']
    meta=read(NATURAL/'layer_postnorm.json')
    assert sha(NATURAL/'layer_postnorm.npz')==meta['archive_sha256']
    with np.load(NATURAL/'layer_postnorm.npz') as z:
        query=z['weighted_query_means'];grand=z['grand_means']
        energy=z['all_coordinate_energies'];weights=z['weights']
    decomposition=[];all_axes={}
    for weighting in ['equal_window','equal_document']:
        global_s=next(i for i,s in enumerate(meta['strata']) if s['cohort']=='all' and s['weighting']==weighting)
        for view in range(2):
            b=query[global_s,view]-grand[global_s,view]
            within=np.zeros(2560);between=np.zeros(2560);cohorts=[]
            for s,stratum in enumerate(meta['strata']):
                if stratum['cohort']=='all' or stratum['weighting']!=weighting:
                    continue
                mask=np.array([r['cohort']==stratum['cohort'] for r in rows])
                mass=float(weights[global_s,mask].sum())
                # Each source group belongs to exactly one registered cohort.
                # Hence the conditional global weights equal that cohort's own weights.
                assert np.allclose(weights[global_s,mask]/mass,weights[s,mask],rtol=1e-12,atol=1e-12)
                bc=query[s,view]-grand[s,view]
                local=energy[s,view,3]
                shift=((bc-b)**2).mean(0)
                within+=mass*local;between+=mass*shift
                cohorts.append({'cohort':stratum['cohort'],'weight_mass':mass,
                    'within_interaction':float(local.mean()),'query_effect_shift':float(shift.mean())})
            total=energy[global_s,view,3]
            error=float(abs(total-within-between).max())
            assert error<max(1e-9,float(total.max())*1e-9),error
            name=weighting+'__'+['raw','whole_vector_RMS'][view]
            all_axes[name]=np.stack([total,within,between])
            decomposition.append({'weighting':weighting,'view':['raw','whole_vector_RMS'][view],
                'global_interaction':float(total.mean()),'within_cohort_interaction':float(within.mean()),
                'between_cohort_query_effect':float(between.mean()),
                'within_fraction_of_global_interaction':float(within.mean()/total.mean()),
                'between_fraction_of_global_interaction':float(between.mean()/total.mean()),
                'all_coordinate_identity_max_error':error,'cohorts':cohorts})
    npz(OUT/'all_coordinate_cohort_decomposition.npz',**all_axes)
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
        'protocol':protocol,'matched_depth_reports':reports,'cohort_decomposition':decomposition,
        'original_natural_result_sha256':sha(NATURAL/'result.json'),
        'matched_postnorm_sha256':sha(target/'layer_postnorm.npz'),
        'all_coordinate_decomposition_sha256':sha(OUT/'all_coordinate_cohort_decomposition.npz'),
        'seconds':time.monotonic()-start,'new_native_endpoints':0}
    save(OUT/'result.json',result);ledger('phase2746_natural_scrutiny',result['seconds'])
    print('NATURAL_SCRUTINY_DONE',result['seconds'],flush=True)


if __name__=='__main__':
    main()
