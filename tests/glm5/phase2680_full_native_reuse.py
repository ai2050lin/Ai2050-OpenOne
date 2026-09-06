"""Four-function target contrasts on EVERY layer and EVERY native H/MLP coordinate."""
import numpy as np
from phase2620_native_coordinate_contract import RESULT,read,save
from phase2677_source_role_contract import OUT as CONTRACT,FIELD
from phase2671_native_mlp_field import unbits

FUNCTIONS=('truth','mapped_truth','name','cloze')


def sign_counts(d):
    """No fitted thresholds, projection, selected coordinates or success filter."""
    assert d.shape[0]==4 and np.isfinite(d).all()
    positive=(d>0).all(axis=0);negative=(d<0).all(axis=0)
    return {'all4_same_nonzero':positive|negative,'all4_positive':positive,'all4_negative':negative,
            'any_zero':(d==0).any(axis=0),'opposed':(d>0).any(axis=0)&(d<0).any(axis=0)}


def analyze(out):
    rows=[r for r in read(CONTRACT/'material/cases.json') if r['source_selected']];groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in ('family','language','unit','content_instance')), {})[(r['output_function'],r['target_index'])]=r['case_index']
    assert len(groups)==64 and all(len(g)==8 for g in groups.values())
    global_counts={};family_counts={}
    for group,indices in groups.items():
        for metric in ('h','a'):
            differences=[]
            for function in FUNCTIONS:
                pair=[]
                for target in (0,1):
                    with np.load(FIELD/f'field/case_{indices[function,target]:04d}.npz') as z:pair.append(unbits(z[metric]).astype(np.float64))
                differences.append(pair[0]-pair[1])
            counts=sign_counts(np.stack(differences))
            for name,v in counts.items():
                key=f'{metric}__{name}'
                if key not in global_counts:global_counts[key]=np.zeros_like(v,dtype=np.uint16)
                global_counts[key]+=v
                fk='_'.join(group[:2])+'__'+key
                if fk not in family_counts:family_counts[fk]=np.zeros_like(v,dtype=np.uint16)
                family_counts[fk]+=v
    out.joinpath('maps').mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'maps/all_native_four_function_global_counts.npz',**global_counts)
    np.savez_compressed(out/'maps/all_native_four_function_family_counts.npz',**family_counts)
    summary={metric:{name:{'all64_coordinate_counts_by_layer_body_task':(global_counts[f'{metric}__{name}']==64).sum(axis=-1).tolist(),
                           'maximum_group_count_by_layer_body_task':global_counts[f'{metric}__{name}'].max(axis=-1).tolist()}
                     for name in ('all4_same_nonzero','all4_positive','all4_negative')} for metric in ('h','a')}
    result={'all_checks_passed':True,'base_groups':64,'source_conditions':512,'functions':FUNCTIONS,'per_family_language_denominator':4,'global_denominator':64,
        'shapes':{k:list(v.shape) for k,v in global_counts.items()},'summary':summary,
        'axes':'layer,query(bodylast,real_tasklast),originalphysicalcoordinate; h0embedding,h36pre-finalRMSNorm; a0..a35actualMLPproductunits',
        'counts':'all4_same_nonzero=P+N withinbase; same-direction acrossoutputfunctions may still reverse acrossfamilies. all4_positive andall4_negative separate globalorientation. any_zero andopposed can overlap; they are not a partition.',
        'body_warning':'Same causalbodyprefix acrossfunctions makes bodyfunctionagreement an expected control, not task-invariant semantic abstraction.',
        'interpretation':'Exhaustive native-coordinate descriptive feature extraction, not TopK, fitted classifier, statistical populationclaim or causalclosure. Strictcounts and all partial counts are retained.'}
    save(out/'analysis/all_native_four_function_reuse.json',result);return result
