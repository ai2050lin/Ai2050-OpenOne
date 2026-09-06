"""CPU-only audit of completed scalar data while crossmodel CUDA work proceeds."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import read,save,RESULT
FP=RESULT/'phase2674_native_mlp_scalar';OUT=RESULT/'phase2676_native_mlp_delivery'


def main():
    rr=read(FP/'analysis/records.json');sites=read(FP/'protocol/frozen.json')['sites'];n=0;mx=0.;term_sum=0.;scalar_effects=0
    for r in rr:
        for e in r['effects']:
            for part in ('all','content','format','eos'):
                x=e if part=='all' else e['parts'][part];v=sum(d*r['gradients'][part][si] for si,d in zip(e['indices'],e['actual_deltas']));term_sum=max(term_sum,abs(x['predicted']-v));scalar_effects+=1
        if not r['published']:continue
        with np.load(FP/f'field/case_{r["case_index"]:04d}.npz') as z:
            for si,s in enumerate(sites):
                l,j,k,unit=s['layer'],s['j'],s['k'],s['unit'];kind=s['kind']
                for part in ('all','content','format','eos'):
                    result=[]
                    for branch in ('Y','N'):
                        x=z[f'{branch}__L{l}_J{unit}_a'] if kind=='down' else z[f'{branch}__L{l}_x'][:,k]
                        g=z[f'{branch}__L{l}_down_g_{part}'][:,j] if kind=='down' else z[f'{branch}__L{l}_J{unit}_{kind}_g_{part}']
                        result.append(sum(float(a)*float(b) for a,b in zip(x,g)))
                    mx=max(mx,abs(result[0]-result[1]-r['gradients'][part][si]));n+=1
    checks={'128complete':len(rr)==128,'1920raw_alltoken_derivatives':n==1920 and mx<1e-8,'37760_effect_part_predictions':scalar_effects==37760 and term_sum<1e-10,
        '128_fullmatrix_hashes_restored':all(r['whole_matrix_hashes']==read(FP/'protocol/before_hashes.json') for r in rr),'all_noops_zero':all(r['noop_error']==0 for r in rr)}
    result={'checks':checks,'all_checks_passed':all(checks.values()),'max_derivative_discrepancy':mx,'max_recomputed_prediction_discrepancy':term_sum,'parts_compared':scalar_effects,
        'boundary':'Independent reduction of stored raw input and downstream derivative factors, not an independent autograd implementation or proof that tiny finitechanges are resolved. NoTorch/model import.'}
    save(OUT/'analysis/scalar_independent.json',result);print(json.dumps(result));assert result['all_checks_passed']


if __name__=='__main__':main()
