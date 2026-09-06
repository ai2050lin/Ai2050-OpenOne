"""All-coordinate cross-query sign accounting; descriptive extension, not a new lockbox."""
import numpy as np
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2654_output_function_delivery'
FIRST=RESULT/'phase2651_output_function_maps';SECOND=RESULT/'phase2652_output_function_confirmation'


def main():
    counts={};groups=sorted(p.name.removesuffix('_truth_a.npz') for p in (FIRST/'maps').glob('*_truth_a.npz'))
    assert len(groups)==16
    for group in groups:
        with np.load(FIRST/f'maps/{group}_truth_a.npz') as ia,np.load(FIRST/f'maps/{group}_truth_b.npz') as ib,np.load(SECOND/f'maps/{group}_truth_a.npz') as ca,np.load(SECOND/f'maps/{group}_truth_b.npz') as cb:
            for metric in ('h','mlp','bf_h','bf_mlp'):
                z=[ia,ib,ca,cb]
                sign=[np.sign(v['target__'+metric+'__mean']) for v in z]
                dominant=[(v['target__'+metric+'__rms']>v['order__'+metric+'__rms'])&(v['target__'+metric+'__rms']>v['form__'+metric+'__rms']) for v in z]
                acrosssets=(sign[0]==sign[2])&(sign[1]==sign[3])&(sign[0]!=0)&(sign[1]!=0)
                both=np.logical_and.reduce(dominant)&acrosssets
                for name,mask in [('both_modes_stable',both),('truth_oriented_opposite',both&(sign[0]==-sign[1])),('semantic_target_same',both&(sign[0]==sign[1]))]:
                    key=metric+'__'+name
                    if key not in counts:counts[key]=np.zeros_like(mask,dtype='int16')
                    counts[key]+=mask
    summary={}
    for metric in ('h','mlp','bf_h','bf_mlp'):
        h=metric.endswith('h');l=36 if h else 35
        boundary=lambda a:a[l,2] if h else a[l]
        summary[metric]={key:int((boundary(counts[metric+'__'+key])==16).sum()) for key in ('both_modes_stable','truth_oriented_opposite','semantic_target_same')}
        summary[metric]['by_layer_position']={key:(counts[metric+'__'+key]==16).sum(-1).tolist() for key in ('both_modes_stable','truth_oriented_opposite','semantic_target_same')}
    OUT.joinpath('maps').mkdir(parents=True,exist_ok=True);np.savez(OUT/'maps/truth_query_fullcoordinate_reuse.npz',**counts)
    result={'summary':summary,'groups':groups,'frozen_before_third_set':True,
        'boundary':'Post-confirmation descriptive extension using already-seen initial and heldout fields. No new out-of-sample evidence is claimed here. Acrosssets signs are tested within each group. truth_a and truth_b reverse truth when the target entity changes; opposite raw target contrast is consistent with truth/answer preparation, but does not identify its cause or isolate semantics from Yes/No output rows. All coordinates assessed, no Top-K.'}
    save(OUT/'analysis/truth_query_coordinate_reuse.json',result);print(json.dumps({m:{k:v for k,v in s.items() if k!='by_layer_position'} for m,s in summary.items()}),flush=True)


if __name__=='__main__':main()
