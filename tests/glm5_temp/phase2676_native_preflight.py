"""New serializer, pairing, physical paths and GPU-side capture QA; not semantic evidence."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import OUT as CONTRACT,FIELD,SITES,LAYERS
from phase2671_native_mlp_field import unbits
from phase2673_native_mlp_confirmation import pairs,silu

OUT=RESULT/'phase2676_native_mlp_delivery'

def main():
    rows=read(CONTRACT/'material/cases.json');rr=rows[:256];checks={}
    for axis in ('unit','content_instance','form','mention_order','probe_index','polarity','mapping','target_index'):
        ij=pairs(rr,axis);checks['fullpair_'+axis]=ij.shape==(128,2) and all(rr[a][axis]!=rr[b][axis] for a,b in ij)
    a=np.arange(65536,dtype=np.uint16);f=unbits(a);checks['lossless_all65536bits']=np.array_equal((f.view('uint32')>>16).astype('uint16'),a)
    # Read only prepublished complete files while the main run is still working.
    chosen=[r for r in rows if r['published'] and (FIELD/f'field/case_{r["case_index"]:04d}.npz').exists()];reports=[]
    for r in chosen:
        with np.load(FIELD/f'field/case_{r["case_index"]:04d}.npz') as z:
            zz={k:unbits(z[k]) for k in z.files};T=len(r['prompt_ids'])
            checks[f'{r["case_index"]}_finite']=all(np.isfinite(v).all() for v in zz.values())
            checks[f'{r["case_index"]}_Hfull']=zz['full__h'].shape==(37,T,2560)
            checks[f'{r["case_index"]}_allgateup']=zz['gate'].shape==zz['up'].shape==(36,9728)
            checks[f'{r["case_index"]}_body_task_H']=np.array_equal(zz['h'],zz['full__h'][:,[r['body_end_token'],T-1]])
            checks[f'{r["case_index"]}_no_truncated_units']=zz['a'].shape==(36,2,9728)
            checks[f'{r["case_index"]}_selected_x']=zz['full__x'].shape==(4,T,2560) and np.array_equal(zz['full__x'][:,-1],zz['x'])
            checks[f'{r["case_index"]}_MLPboundary']=all(np.array_equal(zz['a'][l,-1],zz['full__a'][i,-1]) for i,l in enumerate(LAYERS))
            with np.load(FIELD/'weights/native_candidate_vectors.npz') as w:
                for l,j in SITES:
                    li=LAYERS.index(l);x=zz['full__x'][li,-1].astype('float64')
                    reports.append({'case_index':r['case_index'],'layer':l,'unit':j,'gate_sum_error':float(w[f'L{l}_J{j}_gate'].astype('float64')@x-zz['gate'][l,j]),
                        'up_sum_error':float(w[f'L{l}_J{j}_up'].astype('float64')@x-zz['up'][l,j])})
    checks['at_least_one_actual_published_sample']=len(chosen)>0
    checks['128scalar_prospective_prefixes']=len(read(RESULT/'phase2674_native_mlp_scalar/material/cases.json'))==128
    result={'checks':checks,'all_checks_passed':all(checks.values()),'native_samples':len(chosen),'native_accounting':reports,'scope':'Engineering checks on completed published raw packs only. Not full8192completion or scientific candidate confirmation.'}
    save(OUT/'analysis/early_native_preflight.json',result);print(json.dumps({'checks':checks,'samples':len(chosen)}));assert result['all_checks_passed']

if __name__=='__main__':main()
