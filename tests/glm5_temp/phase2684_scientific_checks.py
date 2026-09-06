"""Whole-campaign independent arithmetic, provenance and scope checks."""
import sys,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2684_source_campaign_delivery import OUT,CONTRACT,FIELD,SOURCE,PATHS,FRESH,SCALAR,CROSS


def main():
    checks={};phases={p:read(path/'analysis/final.json') for p,path in ((2677,CONTRACT),(2678,FIELD),(2679,SOURCE),(2680,PATHS),(2681,FRESH),(2682,SCALAR),(2683,CROSS))}
    checks['all7completed_phases']=all(r['all_checks_passed'] and not r['language_mechanism_closed'] for r in phases.values())
    numbers=[int(p) for p in re.findall(r'^## Phase (\d+):',MEMO.read_text(encoding='utf-8-sig'),re.M)]
    checks['recent_MEMO_continuous']=numbers[-7:]==list(range(2677,2684))
    checks['2678_independent_fullfield_audit']=read(FIELD/'analysis/independent_field_audit.json')['all_checks_passed']
    charts=0
    def counts(z,prefix,den):
        p=z[prefix+'all4_positive'];n=z[prefix+'all4_negative'];same=z[prefix+'all4_same_nonzero']
        assert np.array_equal(p+n,same)
        for key in ('all4_positive','all4_negative','all4_same_nonzero','any_zero','opposed'):
            a=z[prefix+key];assert np.isfinite(a).all() and np.min(a)>=0 and np.max(a)<=den
    with np.load(PATHS/'maps/all_native_four_function_global_counts.npz') as z:
        for metric in ('h','a'):counts(z,metric+'__',64);charts+=1
    for path in (FRESH/'maps').glob('fresh_*.npz'):
        with np.load(path) as z:
            for metric in ('h','a'):
                counts(z,metric+'__',8);lo=z[metric+'__minimum_abs_delta_sum'];hi=z[metric+'__maximum_abs_delta_sum']
                assert (0<=lo).all() and (lo<=hi).all() and np.isfinite(hi).all();charts+=1
    for model in ('qwen14','glm4','ds7','ds7_answer'):
        for path in (CROSS/model/'maps').glob('counts_*.npz'):
            with np.load(path) as z:
                for metric in ('h','a'):counts(z,metric+'__',4);charts+=1
    checks['all_native_count_identities_and_amplitudes']=charts==162
    # Published four-function body-prefix equality is a numerical control only.
    body_pairs=0
    for material,field in ((CONTRACT,FIELD),(FRESH,FRESH)):
        groups={}
        for r in read(material/'material/cases.json'):
            if r['published']:groups.setdefault((r['family'],r['language']),[]).append(r)
        assert len(groups)==16
        for rr in groups.values():
            assert len(rr)==4;base=None
            for r in rr:
                with np.load(field/f'field/case_{r["case_index"]:04d}.npz') as z:now={k:z[k][:,0].copy() for k in ('h','a')}
                if base is not None:assert all(np.array_equal(now[k],base[k]) for k in now);body_pairs+=1
                else:base=now
    checks['96_samebody_published_controls']=body_pairs==96
    records=read(SCALAR/'analysis/records.json');cases={r['case_index']:r for r in read(SCALAR/'material/cases.json')};plan=read(SCALAR/'protocol/frozen.json')
    checks['scalar_material_immutable']=sha(SCALAR/'material/cases.json')==plan['material_sha256']
    with np.load(FIELD/'weights/native_candidate_vectors.npz') as z:weights={k:z[k].astype(np.float64) for k in z.files}
    maxpred=0.;maxerr=0.;scalarconditions=0;formula_checks=0
    for r in records:
        ci=r['case_index'];rr=cases[ci];assert len(r['conditions'])==120
        with np.load(SCALAR/f'maps/case_{ci:04d}.npz') as z:maps=z['local_coordinate_sums']
        assert maps.shape==(120,3,2560) and np.isfinite(maps).all()
        raw=None
        if rr['published_numeric']:
            with np.load(SCALAR/f'field/case_{ci:04d}.npz') as z:raw={k:z[k] for k in z.files}
        for idx,c in enumerate(r['conditions']):
            scalarconditions+=1
            assert c['actual_delta']!=0 and c['all_input_coords_unchanged'] and c['all_other_units_unchanged']
            assert len(c['output_logprob_delta'])==len(r['observed_ids'])
            assert abs(c['output_full_delta']-c['output_nonEOS_delta']-c['output_EOS_delta'])<1e-10
            maxerr=max(maxerr,abs(maps[idx,2].sum()-c['local_error_l1']))
            if raw is not None:
                l,j,k=c['layer'],c['unit'],c['coordinate'];kind=c['kind'];delta=c['actual_delta'];stem=f'L{l}_J{j}_'
                g=raw[stem+'gate'].astype(np.float64);u=raw[stem+'up'].astype(np.float64);unit=raw[stem+'a'].astype(np.float64);x=raw[f'L{l}_x'].astype(np.float64)
                if kind=='gate':change=(g+delta*x[:,k])/(1+np.exp(-(g+delta*x[:,k])))-g/(1+np.exp(-g));da=change*u
                elif kind=='up':da=g/(1+np.exp(-g))*delta*x[:,k]
                else:da=np.zeros(len(g))
                expected=weights[stem+'down']*da.sum()
                if kind=='down':expected[k]=delta*unit.sum()
                maxpred=max(maxpred,float(np.abs(expected-maps[idx,1]).max()),float(np.abs(da-raw[f'C{idx:03d}_predicted_a']).max()));formula_checks+=1
    restoration=read(SCALAR/'analysis/restoration.json')
    checks.update(scalar_15360_actual_conditions=scalarconditions==15360,all12_matrices_restored=restoration['before']==restoration['after'],
        local_error_fullcoordinate_sums=maxerr<1e-8,published_1920_independent_local_predictions=formula_checks==1920 and maxpred<1e-8,
        no_scalar_noop_reinterpreted_as_closure=phases[2682]['summary']['full_output_prediction_claim'] is False)
    checks['DS_each_calibrated64']=all(read(CROSS/k/'protocol/generation.json')['calibration_cases']==64 for k in ('ds7','ds7_answer'))
    checks['crossmodel_protocol_each512']=all(read(CROSS/k/'analysis/completion.json')['cases']==512 for k in ('qwen14','glm4','ds7','ds7_answer'))
    candidate=read(CROSS/'qwen14/analysis/candidate_weight_audit.json')
    checks['Qwen14_actual_candidate_weights_and_checkpoint_embedding_audit']=candidate['all_checks_passed']
    with np.load(CROSS/'qwen14/weights/candidate_native_vectors.npz') as z:
        for key,info in candidate['vectors'].items():
            assert z[key].shape==(5120,) and hashlib.sha256(z[key].tobytes()).hexdigest()==info['vector_sha256']
        for link in candidate['direct_same_layer_links']:
            key=f'L{link["MLP_layer"]}_J{link["unit"]}_down'
            assert float(z[key][link['coordinate']])==link['actual_Wdown']
    checks={k:bool(v) for k,v in checks.items()}
    summary={'all_checks_passed':all(checks.values()),'checks':checks,'full_coordinate_chart_checks':charts,'samebody_controls':body_pairs,'scalar_conditions':scalarconditions,
             'independent_published_predictions':formula_checks,'max_coordinate_prediction_error':maxpred,'max_error_accounting_difference':maxerr,
             'boundary':'Basic exact-arithmetic/provenance checks; not statistical significance, semantic specificity or mechanism closure. All failures and partial counts retained.'}
    save(OUT/'analysis/scientific_checks.json',summary);print({k:v for k,v in summary.items() if k!='checks'});assert all(checks.values()),checks


if __name__=='__main__':main()
