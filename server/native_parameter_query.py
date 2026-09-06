"""Read-only scalar access to an explicit native-coordinate result allowlist.

No model loads or arbitrary filesystem paths; large NPY arrays are memory mapped.
"""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from fastapi import HTTPException

RESULT=Path(__file__).resolve().parents[1]/'tests/glm5/result'
SOURCES={'qwen4':'phase2622_unmodified_native_fields','qwen14':'phase2625_qwen14_native_parameters',
         'glm4':'phase2626_glm4_native_parameters','ds7':'phase2627_ds7_native_parameters'}

@lru_cache(maxsize=4)
def metadata(model):
    root=RESULT/SOURCES[model]
    return (json.loads((root/'material/cases.json').read_text(encoding='utf-8')),
            json.loads((root/'protocol/model.json').read_text(encoding='utf-8')))

def query(model,case,j,k,checkpoint=None,token=None):
    if model not in SOURCES:raise HTTPException(status_code=400,detail='unknown model')
    root=RESULT/SOURCES[model]
    try:cases,info=metadata(model)
    except FileNotFoundError as exc:raise HTTPException(status_code=404,detail='native result not yet available') from exc
    n=len(cases);D=info['hidden_size'];K=info['intermediate_size']
    if not (0<=case<n and 0<=j<D and 0<=k<K):raise HTTPException(status_code=400,detail=f'indices must be case< {n}, j< {D}, k< {K}, all nonnegative')
    def load(name):return np.load(root/f'field/{name}.float32.npy',mmap_mode='r',allow_pickle=False)
    H=load('hidden_anchor_boundary');A=load('mlp_anchor_boundary');gh=float(load('gradient_h')[case,j]);a=float(A[case,-1,-1,k]);x=float(load('final_mlp_input')[case,j])
    extra={}
    if checkpoint is not None or token is not None:
        if checkpoint is None or token is None:raise HTTPException(status_code=400,detail='checkpoint and token must be provided together')
        rawpath=root/f'field/fulltoken/case_{case:04d}.float32.npy'
        if not rawpath.is_file():raise HTTPException(status_code=404,detail='full-token exemplar not retained for this case; boundary data remain available')
        full=np.load(rawpath,mmap_mode='r',allow_pickle=False)
        if not (0<=checkpoint<full.shape[0] and 0<=token<full.shape[1]):raise HTTPException(status_code=400,detail='checkpoint/token outside recorded field')
        extra={'full_token_coordinate':float(full[checkpoint,token,j]),'full_token_checkpoint':checkpoint,'full_token_position':token,'full_token_string':cases[case]['token_strings'][token]}
    return {'model':model,'case_index':case,'case_id':cases[case]['case_id'],'prompt':cases[case]['text'],**extra,
        'layer':info['layers']-1,'coordinate_j':j,'neuron_k':k,'counts':{'cases':n,'coordinates':D,'neurons':K},
        'values':{'embedding_anchor_j':float(H[case,0,0,j]),'hidden_final_boundary_j':float(H[case,-1,-1,j]),
            'mlp_neuron_k':a,'actual_down_weight_jk':float(load('final_down_weights')[j,k]),
            'down_weight_local_derivative_jk':gh*a,'gate_weight_local_derivative_kj':float(load('gradient_gate')[case,k])*x,
            'up_weight_local_derivative_kj':float(load('gradient_up')[case,k])*x,
            'single_neuron_delete_predicted_margin_change':float(load('neuron_delete_effect')[case,k]),
            'single_coordinate_delete_predicted_margin_change':float(load('coordinate_delete_effect')[case,j])},
        'boundary':'最终 MLP 的首 token 对比、实数延拓预测；BF16 会舍入。激活、神经元、权重是不同对象；不是语义必要性。'}
