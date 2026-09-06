"""Read-only, all-token scalar derivatives at 28 recorded native projection sites."""
from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np
from fastapi import HTTPException

RESULT=Path(__file__).resolve().parents[1]/'tests/glm5/result'
SOURCE=RESULT/'phase2632_fulltoken_native_adjoints'
LAYERS=(0,5,17,35)
INPUT_KEY={'q_proj':'attn_x','k_proj':'attn_x','v_proj':'attn_x','o_proj':'o_x',
           'gate_proj':'mlp_x','up_proj':'mlp_x','down_proj':'down_x'}

@lru_cache(maxsize=1)
def metadata():
    try:
        frames=json.loads((SOURCE/'material/frames.json').read_text(encoding='utf-8'))
        manifest=json.loads((SOURCE/'analysis/raw_manifest.json').read_text(encoding='utf-8'))
        ids={r['frame_id'] for r in manifest if r['published']}
        return {f['frame_id']:{**f,'chosen_token':f['tokens'][0],'runnerup_token':f['tokens'][1]} for f in frames if f['frame_id'] in ids}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404,detail='native path results unavailable') from exc

def frame_options():
    return {'frames':[{k:f[k] for k in ('frame_id','case_id','step','eos','chosen_token','runnerup_token')} for f in metadata().values()],
            'layers':list(LAYERS),'modules':list(INPUT_KEY),'model':'Qwen3-4B BF16'}

def query(frame,layer,module,j,k):
    if layer not in LAYERS or module not in INPUT_KEY:
        raise HTTPException(status_code=400,detail='only layers 0,5,17,35 and the seven recorded projections are available')
    if j<0 or k<0 or frame<0:raise HTTPException(status_code=400,detail='indices must be nonnegative')
    frames=metadata()
    if frame not in frames:raise HTTPException(status_code=404,detail='frame not published; select a retained exemplar from native-path-frames')
    key=f'L{layer}_{module}'
    path=SOURCE/f'field/factors/frame_{frame:04d}.npz'
    if not path.is_file():raise HTTPException(status_code=404,detail='published factor pack unavailable')
    weight=np.load(SOURCE/f'field/weights/{key}.float32.npy',mmap_mode='r',allow_pickle=False)
    if j>=weight.shape[0] or k>=weight.shape[1]:
        raise HTTPException(status_code=400,detail=f'j < {weight.shape[0]}, k < {weight.shape[1]} required')
    # Load complete source arrays, return every token contribution. No Top-K selection.
    with np.load(path,allow_pickle=False) as pack:
        x=pack[f'L{layer}_{INPUT_KEY[module]}'][:,k].astype('float64')
        g=pack[key+'__g'][:,j].astype('float64')
        values=pack[key+'__value'][:,j].astype('float64')
    f=frames[frame];products=x*g
    token_labels_path=RESULT/'phase2635_expanded_native_path_confirmation/material/client_token_strings.json'
    labels=json.loads(token_labels_path.read_text(encoding='utf-8')).get(str(frame),[]) if token_labels_path.is_file() else []
    terms=[{'position':t,'token_id':token_id,'token':labels[t] if t<len(labels) else str(token_id),
            'input_k':float(x[t]),'projection_output_j':float(values[t]),'output_adjoint_j':float(g[t]),'product':float(products[t])}
           for t,token_id in enumerate(f['prefix_ids'])]
    total=math.fsum(float(v) for v in products)
    return {'frame':frame,'case_id':f['case_id'],'step':f['step'],'eos':f['eos'],
        'chosen_id':f['chosen_id'],'runnerup_id':f['runnerup_id'],'chosen_token':f['chosen_token'],'runnerup_token':f['runnerup_token'],
        'layer':layer,'module':module,'j':j,'k':k,'shape':list(weight.shape),'token_count':len(terms),
        'values':{'actual_weight_jk':float(weight[j,k]),'all_token_derivative':total,'last_token_only_derivative':float(products[-1]),
                  'earlier_token_sum':math.fsum(float(v) for v in products[:-1])},
        'tokens':terms,'boundary':'All token positions, native coordinates; FP64 sum of recorded BF16 adjoints and inputs. Local AD ignores rounding discontinuities; neither semantic necessity nor a finite-change guarantee.'}
