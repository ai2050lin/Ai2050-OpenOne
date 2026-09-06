"""Same scalar, all token positions, two arithmetic precisions; fixed result allowlist."""
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import HTTPException
from .native_path_parameter_query import LAYERS, INPUT_KEY, RESULT, SOURCE

SOURCES={'bf16':RESULT/'phase2637_bf16_native_numeric_control','fp32':RESULT/'phase2638_fp32_native_numeric_control'}
DELIVERY=RESULT/'phase2640_paired_precision_atlas'

@lru_cache(maxsize=1)
def metadata():
    try:
        frames=json.loads((DELIVERY/'material/published_frames.json').read_text(encoding='utf-8'))
        strings=json.loads((DELIVERY/'material/token_strings.json').read_text(encoding='utf-8'))
        return {f['frame_id']:f for f in frames},strings
    except FileNotFoundError as exc:raise HTTPException(status_code=404,detail='paired precision result unavailable') from exc

def options():
    frames,_=metadata()
    return {'frames':[{k:f[k] for k in ('frame_id','case_id','step')} for f in frames.values()],
            'layers':list(LAYERS),'modules':list(INPUT_KEY),'precisions':list(SOURCES)}

def query(frame,layer,module,j,k,hj,ak):
    if layer not in LAYERS or module not in INPUT_KEY:raise HTTPException(status_code=400,detail='unrecorded layer or projection')
    if min(frame,j,k,hj,ak)<0 or hj>=2560 or ak>=9728:raise HTTPException(status_code=400,detail='nonnegative indices, hj<2560 and ak<9728 required')
    frames,strings=metadata()
    if frame not in frames:raise HTTPException(status_code=404,detail='only the16 published initial prefixes are available')
    W=np.load(SOURCE/f'field/weights/L{layer}_{module}.float32.npy',mmap_mode='r',allow_pickle=False)
    if j>=W.shape[0] or k>=W.shape[1]:raise HTTPException(status_code=400,detail=f'j<{W.shape[0]}, k<{W.shape[1]} required')
    key=f'L{layer}_{module}';values={};f=frames[frame]
    tokens=[{'position':i,'token_id':v,'token':strings[str(frame)][i]} for i,v in enumerate(f['prefix_ids'])]
    for precision,source in SOURCES.items():
        path=source/f'field/frame_{frame:04d}.npz'
        if not path.is_file():raise HTTPException(status_code=404,detail='published precision field missing')
        with np.load(path,allow_pickle=False) as pack:
            x=pack[f'L{layer}_{INPUT_KEY[module]}'][:,k].astype('float64');g=pack[key+'__g'][:,j].astype('float64')
            product=x*g
            values[precision]={'all_token_derivative':math.fsum(product),'last_token_derivative':float(product[-1]),
                'embedding_last_token_hj':float(pack['hidden_boundary'][0,hj]),
                'hidden_block_output_hj':float(pack['hidden_boundary'][layer+1,hj]),
                'hidden_adjoint_hj':float(pack['hidden_adjoint_boundary'][layer+1,hj]),
                'mlp_neuron_ak':float(pack['mlp_boundary'][layer,ak]),'mlp_adjoint_ak':float(pack['mlp_adjoint_boundary'][layer,ak])}
            for t,xx,gg,pp in zip(tokens,x,g,product):t[precision]={'input_k':float(xx),'output_adjoint_j':float(gg),'product':float(pp)}
    return {'frame':frame,'case_id':f['case_id'],'layer':layer,'module':module,'j':j,'k':k,'hidden_coordinate_hj':hj,'mlp_neuron_ak':ak,
        'actual_weight_jk_same_both_precisions':float(W[j,k]),'shape':list(W.shape),'values':values,'tokens':tokens,
        'boundary':'Same stored BF16 weight values, original native output pair and prefix. FP32 changes core arithmetic and baseline states, not training precision. Hidden/MLP values are current-boundary positions; parameter products cover every token. Local sensitivity is not semantic necessity.'}
