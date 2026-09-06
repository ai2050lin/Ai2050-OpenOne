"""Read-only exact full-answer sequence scalar factors; no inference or weight writes."""
import json,math
from functools import lru_cache
import numpy as np
from fastapi import HTTPException
from .native_path_parameter_query import RESULT,SOURCE,LAYERS

DELIVERY=RESULT/'phase2661_sequence_coordinate_delivery';BF=RESULT/'phase2656_truth_answer_behavior';FP=RESULT/'phase2658_sequence_parameter_engine'


@lru_cache(maxsize=1)
def metadata():
    try:return {r['case_index']:r for r in json.loads((DELIVERY/'material/published_cases.json').read_text(encoding='utf-8'))}
    except FileNotFoundError as exc:raise HTTPException(status_code=404,detail='sequence atlas unavailable') from exc


def options():return {'cases':[{k:r[k] for k in ('case_index','case_id')} for r in metadata().values()],'layers':list(LAYERS)}


@lru_cache(maxsize=2)
def published_pack(path,mtime_ns,size):
    """Two immutable packs (BF16/FP32) only; keys invalidate on file replacement."""
    with np.load(path,allow_pickle=False) as source:arrays={k:source[k] for k in source.files}
    for value in arrays.values():value.setflags(write=False)
    return arrays


def query(case,layer,j,k,hj,ak,checkpoint,token):
    if layer not in LAYERS or min(case,j,k,hj,ak,checkpoint)<0 or j>=1024 or max(k,hj)>=2560 or ak>=9728 or checkpoint>=37:raise HTTPException(status_code=400,detail='Invalid model-local coordinate; V layer0/5/17/35,j<1024,k/hj<2560,ak<9728,checkpoint<37.')
    if case not in metadata():raise HTTPException(status_code=404,detail='Only64 published cases retained')
    r=metadata()[case];T=len(r['prompt_ids']);token=T-1 if token is None else token
    if token<0 or token>=T:raise HTTPException(status_code=400,detail='token outside original prefix')
    paths=[p/f'field/case_{case:04d}.npz' for p in (BF,FP)]
    if not all(p.is_file() for p in paths):raise HTTPException(status_code=404,detail='Published raw pack missing')
    W=np.load(SOURCE/f'field/weights/L{layer}_v_proj.float32.npy',mmap_mode='r',allow_pickle=False);fields={};branches=[]
    b,f=[published_pack(str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths]
    for precision,z in [('bf16',b),('fp32',f)]:
        fields[precision]={'embedding':float(z['hidden_fulltoken'][0,token,hj]),'hidden':float(z['hidden_fulltoken'][checkpoint,token,hj]),'mlp':float(z['mlp_boundary'][layer,ak])}
        if 'decision_hidden_fulltoken' in z:fields[precision]['decision_hidden_boundary']=float(z['decision_hidden_fulltoken'][checkpoint,-1,hj]);fields[precision]['decision_mlp_boundary']=float(z['decision_mlp_boundary'][layer,ak])
    for i,label in enumerate(('Y','N')):
        x=f[f'{label}__L{layer}_v_x'][:,k].astype('float64');g=f[f'{label}__L{layer}_v_g'][:,j].astype('float64');v=f[f'{label}__L{layer}_v_value'][:,j];products=x*g;meta=r['branches'][i]
        tokens=[{'position':t,'token_id':idx,'token':r['token_strings'][t] if t<T else r['common_readout_words'][i],
            'x':float(x[t]),'value':float(v[t]),'adjoint':float(g[t]),'product':float(products[t])} for t,idx in enumerate(meta['input_ids'])]
        branches.append({'label':label,'answer':r['common_readout_words'][i],'total_logprob':meta['total_logprob'],'logprobs':meta['logprobs'],'target_ids':meta['target_ids'],
            'prediction_positions':meta['prediction_positions'],'derivative':math.fsum(products),'prompt_last_only':float(products[T-1]),'branch_last_only':float(products[-1]),
            'hidden_prompt_adjoint':float(f[f'{label}__hidden_adjoint_prompt_boundary'][layer+1,hj]),'mlp_prompt_adjoint':float(f[f'{label}__mlp_adjoint_prompt_boundary'][layer,ak]),'tokens':tokens})
    return {'case_index':case,'case_id':r['case_id'],'text':r['text'],'target':r['target'],'generated':r['generated'],'content_correct':r['content_correct'],
        'sequence_contrast':r['contrast'],'first_token_contrast':r['first_token_contrast'],'eos_contrast':r['eos_contrast'],'decision':r['decision'],
        'layer':layer,'j':j,'k':k,'hj':hj,'ak':ak,'checkpoint':checkpoint,'token_position':token,'actual_weight':float(W[j,k]),'fields':fields,'branches':branches,
        'parameter_derivative':branches[0]['derivative']-branches[1]['derivative'],
        'boundary':'仅两条规范答案+EOS的teacher-forced归一概率差，不是所有等价答案总概率，也不是自由生成成功率。BF16行为和FP32数值分别记录；两分支伴随属于各自序列计算。H坐标、MLP单元与学习权重不是同一个对象。'}
