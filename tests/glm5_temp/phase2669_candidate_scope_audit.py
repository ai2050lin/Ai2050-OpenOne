"""Read every native map; report candidate behavior under all query/mapping cells."""
import itertools,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2665_symmetric_coordinate_maps import OUT as MAPS,OLD
from phase2669_symmetric_multitoken_delivery import OUT
from phase2662_symmetric_mapping_contract import FAMILIES
from phase2657_truth_answer_maps import filename


def main():
    with np.load(MAPS/'maps/confirmed_masks.npz') as z:sites={k:np.argwhere(z[k]).tolist() for k in ('h','mlp')}
    rows=[]
    for fam,lang,p in itertools.product(FAMILIES,('en','zh'),(0,1)):
        with np.load(OLD/'maps/initial'/filename(fam,lang,p,0,0)) as z:prior={metric:{(l,j):float(z[f'target__{metric}__mean'][l,j]) for l,j in ss} for metric,ss in sites.items()}
        for fold,q,m in itertools.product(('initial','confirmation'),(0,1),(0,1)):
            with np.load(MAPS/'maps'/fold/filename(fam,lang,p,q,m)) as z:
                for metric,ss in sites.items():
                    fields={k:z[k] for k in z.files if '__'+metric+'__' in k}
                    for l,j in ss:
                        vals={k:float(v[l,j]) for k,v in fields.items()};old=prior[metric][l,j];mean=vals[f'target__{metric}__mean'];target=vals[f'target__{metric}__rms'];form=vals[f'form__{metric}__rms'];order=vals[f'order__{metric}__rms']
                        rows.append({'family':fam,'language':lang,'probe':p,'fold':fold,'q':q,'m':m,'metric':metric,'layer':l,'coordinate':j,'values':vals,'old_mean':old,
                            'old_direction':bool(np.sign(mean)==np.sign(old) and old!=0),'target_exceeds_form_order':target>form and target>order,
                            'finite_correctpair_map':sum(k.startswith('correctpair') for k in vals)==2 and all(np.isfinite(v) for k,v in vals.items() if k.startswith('correctpair'))})
    grouped={}
    for metric,ss in sites.items():
        for l,j in ss:
            key=f'{metric}{l}[{j}]';grouped[key]={}
            for fold,q,m in itertools.product(('initial','confirmation'),(0,1),(0,1)):
                rr=[r for r in rows if (r['metric'],r['layer'],r['coordinate'],r['fold'],r['q'],r['m'])==(metric,l,j,fold,q,m)]
                grouped[key][f'{fold}/q{q}/m{m}']={'n':len(rr),'old_direction':sum(r['old_direction'] for r in rr),'target_exceeds_form_order':sum(r['target_exceeds_form_order'] for r in rr),
                    'both':sum(r['old_direction'] and r['target_exceeds_form_order'] for r in rr),'finite_correctpair_groups_not_pair_counts':sum(r['finite_correctpair_map'] for r in rr)}
    checks={'1792_scope_rows':len(rows)==1792,'q0m0_reproduced_32groups_bothfolds':all(v[f'{fold}/q0/m0']['both']==32 for v in grouped.values() for fold in ('initial','confirmation'))}
    save(OUT/'analysis/candidate_scope_rows.json',rows);save(OUT/'analysis/candidate_scope_summary.json',{'checks':checks,'all_checks_passed':all(checks.values()),'groups':grouped,'boundary':'Full maps retain allcoordinates; these seven frozen survivors are a focused reporting view, not a new sparse representation. q/m cells are not interchangeable and candidates were not required to pass nonbaseline cells.'})
    assert all(checks.values());print(json.dumps({'checks':checks,'groups':grouped},ensure_ascii=True))


if __name__=='__main__':main()
