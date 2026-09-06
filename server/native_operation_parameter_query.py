"""Published crossed-operation cases: native coordinates, neurons, one shared weight."""
import json,math
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import HTTPException
from .native_path_parameter_query import RESULT,SOURCE,LAYERS

DELIVERY=RESULT/'phase2647_matched_operation_delivery'
BF=RESULT/'phase2642_matched_operation_behavior'
FP={'initial':RESULT/'phase2643_matched_dual_adjoints','confirmation':RESULT/'phase2645_confirmation_dual_adjoints'}

@lru_cache(maxsize=1)
def metadata():
    try:return {r['case_index']:r for r in json.loads((DELIVERY/'material/published_cases.json').read_text(encoding='utf-8'))}
    except FileNotFoundError as exc:raise HTTPException(status_code=404,detail='matched operation atlas unavailable') from exc

def options():
    return {'cases':[{k:r[k] for k in ('case_index','case_id','field_set')} for r in metadata().values()],
        'layers':list(LAYERS),'module':'v_proj','objectives':['native','common'],'hidden_checkpoints':37,'hidden_coordinates':2560,'mlp_neurons':9728}

def query(case,layer,j,k,hj,ak,checkpoint,token):
    if layer not in LAYERS:raise HTTPException(status_code=400,detail='recorded V layers:0,5,17,35')
    if min(case,j,k,hj,ak,checkpoint)<0 or j>=1024 or k>=2560 or hj>=2560 or ak>=9728 or checkpoint>=37:
        raise HTTPException(status_code=400,detail='j<1024,k/hj<2560,ak<9728,checkpoint<37, nonnegative indices required')
    if case not in metadata():raise HTTPException(status_code=404,detail='only the32 published matched-operation cases retained')
    r=metadata()[case];T=len(r['prompt_ids']);token=T-1 if token is None else token
    if token<0 or token>=T:raise HTTPException(status_code=400,detail='token outside this prefix')
    fp=FP[r['field_set']]/f'field/case_{case:04d}.npz';bf=BF/f'field/case_{case:04d}.npz'
    if not fp.is_file() or not bf.is_file():raise HTTPException(status_code=404,detail='published raw pack unavailable')
    W=np.load(SOURCE/f'field/weights/L{layer}_v_proj.float32.npy',mmap_mode='r',allow_pickle=False)
    fields={};objectives={}
    with np.load(bf,allow_pickle=False) as b,np.load(fp,allow_pickle=False) as z:
        for precision,pack in [('bf16',b),('fp32',z)]:
            h=pack['hidden'];a=pack['mlp_positions']
            fields[precision]={'embedding_token_hj':float(h[0,token,hj]),'hidden_checkpoint_token_hj':float(h[checkpoint,token,hj]),
                'hidden_V_block_boundary_hj':float(h[layer+1,-1,hj]),'mlp_neuron_boundary_ak':float(a[layer,2,ak])}
        x=z[f'L{layer}_v_x'][:,k].astype('float64');value=z[f'L{layer}_v_value'][:,j]
        tokens=[{'position':t,'token_id':r['prompt_ids'][t],'token':r['token_strings'][t],'V_input_k':float(x[t]),'V_output_j':float(value[t])} for t in range(T)]
        for obj in ('native','common'):
            key=f'{obj}__L{layer}_v_g'
            if key not in z:objectives[obj]={'available':False};continue
            g=z[key][:,j].astype('float64');product=x*g
            objectives[obj]={'available':True,'output_ids':r[obj+'_ids'],'all_token_parameter_derivative':math.fsum(product),'last_token_only_derivative':float(product[-1]),
                'hidden_block_boundary_adjoint_hj':float(z[f'{obj}__hidden_adjoint_positions'][layer+1,2,hj]),
                'mlp_boundary_adjoint_ak':float(z[f'{obj}__mlp_adjoint_positions'][layer,2,ak])}
            for t in range(T):tokens[t][obj]={'adjoint_j':float(g[t]),'parameter_product':float(product[t])}
    return {'case_index':case,'case_id':r['case_id'],'text':r['text'],'target':r['target'],'generated':r['generated'],'name_content_correct':r['name_content_correct'],
        'native_common_identity':r['native_common_identity'],'layer':layer,'module':'v_proj','j':j,'k':k,'actual_weight':float(W[j,k]),
        'hj':hj,'ak':ak,'checkpoint':checkpoint,'token_position':token,'fields':fields,'objectives':objectives,'tokens':tokens,
        'boundary':'BF16 actual natural behavior and original fields; FP32 same-valued weights with different arithmetic. Native means frozen BF16 output IDs, common is experimenter supplied entityA/B. Every token enters shared scalar derivative. Coordinate h, MLP neuron a, learned weight theta are different. No semantic mechanism closure.'}
