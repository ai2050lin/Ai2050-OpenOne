"""Output-function study, exact published native scalar query, no model loading."""
import json,math
from functools import lru_cache
import numpy as np
from fastapi import HTTPException
from .native_path_parameter_query import RESULT,SOURCE,LAYERS

DELIVERY=RESULT/'phase2654_output_function_delivery'
BF=RESULT/'phase2649_output_function_behavior';FP=RESULT/'phase2650_output_function_adjoints'

@lru_cache(maxsize=1)
def metadata():
    try:return {r['case_index']:r for r in json.loads((DELIVERY/'material/published_cases.json').read_text(encoding='utf-8'))}
    except FileNotFoundError as exc:raise HTTPException(status_code=404,detail='output function atlas unavailable') from exc

def options():
    return {'cases':[{k:r[k] for k in ('case_index','case_id','mode')} for r in metadata().values()],'layers':list(LAYERS),'module':'v_proj','objectives':['native','common']}

def query(case,layer,j,k,hj,ak,checkpoint,token):
    if layer not in LAYERS or min(case,j,k,hj,ak,checkpoint)<0 or j>=1024 or max(k,hj)>=2560 or ak>=9728 or checkpoint>=37:
        raise HTTPException(status_code=400,detail='V layer0/5/17/35; j<1024,k/hj<2560,ak<9728,checkpoint<37, nonnegative indices')
    if case not in metadata():raise HTTPException(status_code=404,detail='only64 published full-token examples retained')
    r=metadata()[case];T=len(r['prompt_ids']);token=T-1 if token is None else token
    if token<0 or token>=T:raise HTTPException(status_code=400,detail='token outside prefix')
    paths=[s/f'field/case_{case:04d}.npz' for s in (BF,FP)]
    if not all(p.is_file() for p in paths):raise HTTPException(status_code=404,detail='published field missing')
    W=np.load(SOURCE/f'field/weights/L{layer}_v_proj.float32.npy',mmap_mode='r',allow_pickle=False);fields={};objectives={}
    with np.load(paths[0],allow_pickle=False) as b,np.load(paths[1],allow_pickle=False) as f:
        for precision,z in [('bf16',b),('fp32',f)]:
            h=z['hidden_fulltoken'];a=z['mlp_boundary'];fields[precision]={'embedding_token_hj':float(h[0,token,hj]),'hidden_checkpoint_token_hj':float(h[checkpoint,token,hj]),
                'hidden_V_block_boundary_hj':float(h[layer+1,-1,hj]),'mlp_neuron_boundary_ak':float(a[layer,ak])}
        x=f[f'L{layer}_v_x'][:,k].astype('float64');value=f[f'L{layer}_v_value'][:,j]
        tokens=[{'position':t,'token_id':r['prompt_ids'][t],'token':r['token_strings'][t],'V_input_k':float(x[t]),'V_output_j':float(value[t])} for t in range(T)]
        for obj in ('native','common'):
            g=f[f'{obj}__L{layer}_v_g'][:,j].astype('float64');product=x*g
            objectives[obj]={'available':True,'output_ids':r[obj+'_ids'],'all_token_parameter_derivative':math.fsum(product),'last_token_only_derivative':float(product[-1]),
                'hidden_block_boundary_adjoint_hj':float(f[f'{obj}__hidden_adjoint_boundary'][layer+1,hj]),'mlp_boundary_adjoint_ak':float(f[f'{obj}__mlp_adjoint_boundary'][layer,ak])}
            for t in range(T):tokens[t][obj]={'adjoint_j':float(g[t]),'parameter_product':float(product[t])}
    return {'case_index':case,'case_id':r['case_id'],'text':r['text'],'prefill':r['prefill'],'mode':r['mode'],'output_function':r['output_function'],
        'target':r['target'],'generated':r['generated'],'name_content_correct':r['content_correct'],'common_readout_words':r['common_readout_words'],
        'native_common_identity':r['native_common_identity'],'layer':layer,'module':'v_proj','j':j,'k':k,'actual_weight':float(W[j,k]),
        'hj':hj,'ak':ak,'checkpoint':checkpoint,'token_position':token,'fields':fields,'objectives':objectives,'tokens':tokens,
        'boundary':'Same-valuedFP32 is a numerical model, BF16 natural behavior separately recorded. Native IDs may contrast formatting of the SAME answer; common is external fixed rows. Cloze supplied prefix and standalone-name readout are controls, not autonomous generation or total natural answer probability. All shared-weight token terms included; no semantic closure.'}
