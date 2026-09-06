"""Exact read-only multi-token native weights and content/format/EOS factors."""
import json,math
from functools import lru_cache
import numpy as np
from fastapi import HTTPException
from .native_path_parameter_query import RESULT,SOURCE,LAYERS
from .native_sequence_parameter_query import published_pack

DELIVERY=RESULT/'phase2669_symmetric_multitoken_delivery';FP=RESULT/'phase2666_multitoken_parameter_engine';BF=FP/'natural';PARTS=('content','format','eos')


@lru_cache(maxsize=1)
def metadata():
    try:return {r['case_index']:r for r in json.loads((DELIVERY/'material/published_cases.json').read_text(encoding='utf-8'))}
    except FileNotFoundError as e:raise HTTPException(status_code=404,detail='multitoken atlas unavailable') from e


def options():return {'cases':[{k:r[k] for k in ('case_index','case_id')} for r in metadata().values()],'layers':list(LAYERS),'parts':list(PARTS)}


def query(case,layer,j,k,hj,ak,checkpoint,token):
    if layer not in LAYERS or min(case,j,k,hj,ak,checkpoint)<0 or j>=1024 or max(k,hj)>=2560 or ak>=9728 or checkpoint>=37:raise HTTPException(status_code=400,detail='Invalid Qwen4 physical coordinate')
    if case not in metadata():raise HTTPException(status_code=404,detail='Only64 published multi-token cases retained')
    r=metadata()[case];T=len(r['prompt_ids']);token=T-1 if token is None else token
    if token<0 or token>=T:raise HTTPException(status_code=400,detail='token outside original multi-token-condition prompt')
    paths=[p/f'field/case_{case:04d}.npz' for p in (BF,FP)]
    if not all(p.is_file() for p in paths):raise HTTPException(status_code=404,detail='Published raw pack missing')
    b,f=[published_pack(str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths];W=np.load(SOURCE/f'field/weights/L{layer}_v_proj.float32.npy',mmap_mode='r',allow_pickle=False);fields={};branches=[]
    for precision,z in (('bf16',b),('fp32',f)):fields[precision]={'embedding':float(z['hidden_fulltoken'][0,token,hj]),'hidden':float(z['hidden_fulltoken'][checkpoint,token,hj]),'mlp':float(z['mlp_boundary'][layer,ak])}
    for i,label in enumerate(('Y','N')):
        x=f[f'{label}__L{layer}_v_x'][:,k].astype('float64');g=f[f'{label}__L{layer}_v_g'][:,j].astype('float64');value=f[f'{label}__L{layer}_v_value'][:,j];product=x*g;meta=r['branches'][i]
        gp={p:f[f'{label}__L{layer}_v_g_{p}'][:,j].astype('float64') for p in PARTS};tokens=[]
        for t,idx in enumerate(meta['input_ids']):
            text=r['token_strings'][t] if t<T else r['answer_token_strings'][i][t-T] if t<meta['actual_input_length'] else '[masked padding]'
            tokens.append({'position':t,'token_id':idx,'token':text,'padding':t>=meta['actual_input_length'],'x':float(x[t]),'value':float(value[t]),'adjoint':float(g[t]),'product':float(product[t]),
                'parts':{p:{'adjoint':float(gp[p][t]),'product':float(x[t]*gp[p][t])} for p in PARTS}})
        branches.append({'label':label,'answer':r['common_readout_words'][i],'total_logprob':meta['total_logprob'],'logprobs':meta['logprobs'],'target_ids':meta['target_ids'],'prediction_positions':meta['prediction_positions'],
            'categories':meta['categories'],'part_logprobs':meta['part_logprobs'],'part_derivatives':{p:math.fsum(x*gp[p]) for p in PARTS},'derivative':math.fsum(product),
            'prompt_last_only':float(product[T-1]),'branch_last_only':float(product[meta['actual_input_length']-1]),'hidden_prompt_adjoint':float(f[f'{label}__hidden_adjoint_prompt_boundary'][layer+1,hj]),
            'mlp_prompt_adjoint':float(f[f'{label}__mlp_adjoint_prompt_boundary'][layer,ak]),'tokens':tokens})
    return {'case_index':case,'case_id':r['case_id'],'text':r['text'],'target':r['target'],'generated':r['generated'],'content_correct':r['content_correct'],'layer':layer,'j':j,'k':k,'hj':hj,'ak':ak,'checkpoint':checkpoint,
        'token_position':token,'actual_weight':float(W[j,k]),'sequence_contrast':r['contrast'],'first_token_contrast':r['branches'][0]['logprobs'][0]-r['branches'][1]['logprobs'][0],
        'content_contrast':r['parts']['content'],'format_contrast':r['parts']['format'],'eos_contrast':r['parts']['eos'],'fields':fields,'branches':branches,
        'parameter_derivative':branches[0]['derivative']-branches[1]['derivative'],'part_parameter_derivatives':{p:branches[0]['part_derivatives'][p]-branches[1]['part_derivatives'][p] for p in PARTS},
        'boundary':'仅两条4token规范格式答案加EOS的teacher-forced概率差，非长句解释、所有等价回答总概率或自主生成成功率。本处H/MLP与导数都属于多token格式指令，不混用短码条件状态。全部V输入坐标和token保留；分数及其部分伴随为FP32，微小加总误差单列。H、MLP中间单元和真实标量权重不是同一对象。'}
