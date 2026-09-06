"""Post-run independent scalar algebra from saved arrays; numpy only, no models."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import RESULT,read,save,sha

OUT=RESULT/'phase2682_resolved_scalar_paths';FIELD=RESULT/'phase2678_padded_source_field'


def main():
    assert read(OUT/'analysis/final.json')['all_checks_passed'];records=read(OUT/'analysis/records.json');cases={r['case_index']:r for r in read(OUT/'material/cases.json')}
    with np.load(FIELD/'weights/native_candidate_vectors.npz') as z:weights={k:z[k].astype(np.float64) for k in z.files}
    maxpred=0.;maxerr=0.;count=0;formula=0;outputsum=0.;zero_effects=0
    for r in records:
        ci=r['case_index'];raw=None
        if cases[ci]['published_numeric']:
            with np.load(OUT/f'field/case_{ci:04d}.npz') as z:raw={k:z[k] for k in z.files}
        with np.load(OUT/f'maps/case_{ci:04d}.npz') as z:maps=z['local_coordinate_sums']
        assert maps.shape==(120,3,2560) and np.isfinite(maps).all()
        for index,c in enumerate(r['conditions']):
            count+=1;assert c['actual_delta']!=0
            maxerr=max(maxerr,abs(maps[index,2].sum()-c['local_error_l1']))
            outputsum=max(outputsum,abs(sum(c['output_logprob_delta'])-c['output_full_delta']),abs(c['output_full_delta']-c['output_nonEOS_delta']-c['output_EOS_delta']))
            zero_effects+=c['output_full_delta']==0
            if raw is None:continue
            l,j,k=c['layer'],c['unit'],c['coordinate'];stem=f'L{l}_J{j}_';kind=c['kind'];delta=c['actual_delta']
            g=raw[stem+'gate'].astype(np.float64);u=raw[stem+'up'].astype(np.float64);x=raw[f'L{l}_x'].astype(np.float64);a=raw[stem+'a'].astype(np.float64)
            if kind=='gate':gg=g+delta*x[:,k];da=(gg/(1+np.exp(-gg))-g/(1+np.exp(-g)))*u
            elif kind=='up':da=g/(1+np.exp(-g))*delta*x[:,k]
            else:da=np.zeros(len(g))
            prediction=weights[stem+'down']*da.sum()
            if kind=='down':prediction[k]=delta*a.sum()
            maxpred=max(maxpred,float(np.abs(prediction-maps[index,1]).max()),float(np.abs(da-raw[f'C{index:03d}_predicted_a']).max()));formula+=1
    restored=read(OUT/'analysis/restoration.json');plan=read(OUT/'protocol/frozen.json')
    checks={'15360conditions':count==15360,'1920_independent_published_formula_checks':formula==1920 and maxpred<1e-8,
        'allcoordinate_error_sums':maxerr<1e-8,'alltoken_output_sum_identity':outputsum<1e-10,'12matrices_restored':restored['before']==restored['after'],
        'material_immutable':sha(OUT/'material/cases.json')==plan['material_sha256']}
    checks={k:bool(v) for k,v in checks.items()}
    result={'all_checks_passed':all(checks.values()),'checks':checks,'conditions':count,'independent_predictions':formula,'maximum_prediction_sum_difference':float(maxpred),
        'maximum_error_sum_difference':float(maxerr),'maximum_output_component_difference':float(outputsum),'zero_output_effects':zero_effects,
        'boundary':'Independent saved-array algebra, not rerunning15360forwards. Recordedactualoutputeffects are observed, not independentlypredictedfullnetwork probabilities. ZeroFP32outputeffects are retained, notproof ofsemanticirrelevance.'}
    save(OUT/'analysis/independent_audit.json',result);print(result);assert all(checks.values())


if __name__=='__main__':main()
