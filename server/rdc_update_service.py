"""Read-only relation/update/predictive-state atlas; never starts a model or training."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import importlib.util
import numpy as np
from fastapi import APIRouter,HTTPException,Query
from fastapi.responses import FileResponse
from server.rdc_feature_service import ROOT,read,decode
from server.rdc_relation_service import field_response
from server.rdc_joint_service import compressed,npz_headers,scalar_values

BASE=ROOT/'tests/glm5/result/rdc_update_campaign_20260913'
PRIOR=ROOT/'tests/glm5/result/rdc_binding_campaign_20260912'
LAW=ROOT/'tests/glm5/result/rdc_law_campaign_20260911'
router=APIRouter(prefix='/api/rdc-update',tags=['rdc-relation-update-evidence'])
_scoring_spec=importlib.util.spec_from_file_location('rdc_update_pure_scoring',ROOT/'tests/glm5/rdc_update_scoring.py')
_scoring=importlib.util.module_from_spec(_scoring_spec)
_scoring_spec.loader.exec_module(_scoring)
_terminal_spec=importlib.util.spec_from_file_location('rdc_update_format_audit',ROOT/'tests/glm5/rdc_update_terminal_audit.py')
_terminal=importlib.util.module_from_spec(_terminal_spec)
_terminal_spec.loader.exec_module(_terminal)

def materials():
    paths=[PRIOR/'natural_discovery.json.gz']+[BASE/p for p in ('natural_material.json.gz','program_material.json.gz','language_material.json.gz','fresh_graph/material.json.gz')]
    return [r for p in paths if p.exists() for r in compressed(p)]

def source(sid):
    r=next((r for r in materials() if r['sample_id']==sid),None)
    if r is None:raise HTTPException(404,'Unknown registered sample')
    return r

def paths(r):
    mode=r.get('capture_mode','main');sid=r['sample_id']
    if mode=='update':return BASE/'capture/qwen4'/f'{sid}.npz',None,BASE/'capture/full_fields'/f'{sid}.npz'
    if mode=='language':return BASE/'language_capture/fields'/f'{sid}.npz',None,BASE/'language_capture/full_fields'/f'{sid}.npz'
    if mode=='fresh_update':return BASE/'fresh_graph/fields'/f'{sid}.npz',None,None
    if mode=='binding':return PRIOR/'capture/natural'/f'{sid}.npz',None,PRIOR/'capture/full_fields'/f'{sid}.npz'
    return LAW/'capture'/mode/'fields'/f'{sid}.npz',LAW/'capture'/mode/'sources'/f'{sid}.npz',LAW/'capture'/mode/'full_fields'/f'{sid}.npz'

def modelspec(model,sid,r):
    if model=='qwen4':return paths(r)[0],{'width':2560,'depth':36,'early':12,'units':9728},r['anchors'],r['prompt_ids']
    if model not in ('qwen14','glm4'):raise HTTPException(422,'Unknown model')
    path=BASE/'scale'/model/'fields'/f'{sid}.npz';record=BASE/'scale'/model/'commits'/f'{sid}.json'
    if not path.exists() or not record.exists():raise HTTPException(409,'No committed capture for this model/sample; no other model substituted')
    meta=read(record);return path,read(BASE/'scale'/model/'runtime.json'),meta['positions'],meta['prompt_ids']

def index():return {p.relative_to(BASE).as_posix():p for p in BASE.rglob('*.npz') if p.is_file()}

def registered(area,file):
    key=(area+'/' if area else '')+file
    if '\\' in key or '..' in key.split('/') or key not in index():raise HTTPException(404,'Unknown registered array archive')
    p=index()[key]
    if not p.resolve().is_relative_to(BASE.resolve()):raise HTTPException(404,'Outside result registry')
    return p

@router.get('/overview')
def overview():
    parts={'contract':'contract.json','capture':'capture/result.json','graph':'graph/frozen.json','confirmation':'graph/confirmation.json',
      'analysis':'analysis/phase2736.json','learning':'learning/finite_result.json','forecast_audit':'learning/forecast_audit.json',
      'autograd_audit':'learning/autograd_audit.json','language_logic':'language_analysis/logic_audit.json',
      'language_prediction':'language_prediction/result.json','language_identity':'language_identity/result.json',
      'causal_anchor':'causal_anchor/result.json','causal_replay':'causal_replay/result.json',
      'answer_scoring':'behavior_analysis/result.json','long_answers':'long_answers/result.json','terminal_format_audit':'terminal_format_audit/result.json',
      'manual_terminal_audit':'manual_terminal_audit/result.json',
      'middle':'middle_training/result.json','language':'language_analysis/result.json','history':'own_history/result.json',
      'same_history':'same_history/result.json','native_paths':'native_paths/result.json','moment_boundary':'moment_boundary/result.json',
      'predictive_state':'predictive_state/result.json','fresh_confirmation':'fresh_graph/result.json','theory':'theory_snapshot.json',
      'scale_analysis':'scale_analysis/result.json',
      'integrity':'verification/final.json','resources':'resources.json'}
    result={k:read(BASE/p,{}) for k,p in parts.items()}
    result.update(scale={m:read(BASE/'scale'/m/'result.json',{}) for m in ('qwen4','qwen14','glm4')},
      figures=read(BASE/'figures/index.json',{}).get('figures',[]),
      completion={k:(BASE/p).exists() for k,p in parts.items()},
      status='观察、恒等式、有限预测、继续训练和自然生成分开记录；不存在“已经破解语言机制”的状态替代。')
    return result

@router.get('/samples')
def samples(model:str='qwen4'):
    if model not in ('qwen4','qwen14','glm4'):raise HTTPException(422,'Unknown model')
    result=[]
    for r in materials():
        if model!='qwen4' and r['sample_id'] not in set(read(BASE/'scale/protocol.json',{}).get('source_ids',[])):continue
        p=paths(r)[0] if model=='qwen4' else BASE/'scale'/model/'fields'/f'{r["sample_id"]}.npz'
        item=({k:r[k] for k in ('sample_id','source_group','cohort','split','kind','language','anchors')}|
          {'tokens':len(r['prompt_ids']),'captured':p.exists(),'family':r.get('family'),'native_path':(BASE/'native_paths/fields'/f'{r["sample_id"]}.npz').exists()})
        if model!='qwen4':
            commit=BASE/'scale'/model/'commits'/f'{r["sample_id"]}.json'
            if commit.exists():
                meta=read(commit);item.update(anchors=meta['positions'],tokens=meta['tokens'])
        result.append(item)
    return result

@router.get('/sample')
def sample(sample:str='',model:str='qwen4'):
    r=source(sample)
    if model not in ('qwen4','qwen14','glm4'):raise HTTPException(422,'Unknown model')
    if model!='qwen4':r=r|{'native_model_record':read(BASE/'scale'/model/'commits'/f'{sample}.json',{})}
    return r

@router.get('/field')
def field(sample:str='',model:str='qwen4',mode:str='all_layers',anchor:int=Query(0,ge=0,le=3),layer:int=Query(12,ge=0,le=80),view:str='raw'):
    r=source(sample);p,spec,positions,ids=modelspec(model,sample,r)
    if not p.exists():raise HTTPException(409,'Capture not committed')
    if anchor>=len(positions):raise HTTPException(422,'Anchor outside captured range')
    if mode=='all_layers':
        with np.load(p) as z:a=decode(z['H'][:,anchor])
        labels=[f'H{i} at own token{positions[anchor]}' for i in range(len(a))]
    elif mode=='sources':
        sp=(paths(r)[1] or p) if model=='qwen4' else p
        with np.load(sp) as z:a=decode(z['H12_sources' if model=='qwen4' else 'H_early_sources'])
        labels=[f'token{i}: ID{ids[i]}' for i in range(len(a))]
    elif mode=='fixture':
        if model!='qwen4':raise HTTPException(409,'Larger-model all-layer/all-token fixture not claimed; early full sources are available')
        fp=paths(r)[2]
        if fp is None or not fp.exists():raise HTTPException(409,'Not a predeclared full-field fixture')
        with np.load(fp) as z:
            if layer>=len(z['H']):raise HTTPException(422,'Layer outside original depth')
            a=decode(z['H'][layer])
        labels=[f'token{i}: ID{ids[i]}' for i in range(len(a))]
    else:raise HTTPException(422,'Unknown field scope')
    if view=='RMS':a=a/np.sqrt(np.mean(a*a,-1,keepdims=True)).clip(1e-8)
    elif view!='raw':raise HTTPException(422,'Unknown normalization')
    return field_response(a,labels,0,a.shape[-1],f'{model}: {view}; all own coordinates in original order. Cross-model indices are not functionally aligned.',sample_id=sample,model=model,early=spec['early'])

@lru_cache(maxsize=1)
def head_coefficients(path,mtime):
    with np.load(path) as z:return z['coefficients'].astype(np.float32)

@router.get('/graph')
def graph(sample:str='',anchor:int=Query(0,ge=0,le=3)):
    r=source(sample);p,sp,_=paths(r)
    if not p.exists():raise HTTPException(409,'Capture pending')
    if anchor>=len(r['anchors']):raise HTTPException(422,'Anchor absent')
    with np.load(sp or p) as z:h=decode(z['H12_sources'])[:r['anchors'][anchor]+1]
    file=BASE/'graph/head_mapping.npz';c=head_coefficients(str(file),file.stat().st_mtime_ns)
    h=h/np.sqrt(np.mean(h*h,-1,keepdims=True)).clip(1e-8);q=h@c[:-1]+c[-1];q/=np.sqrt(np.mean(q*q,-1,keepdims=True)).clip(1e-8)
    score=q@h.T/2560/.15;np.fill_diagonal(score,-np.inf);a=np.exp(score-score.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
    v=field_response(a,[f'dependent token{i}' for i in range(len(a))],0,len(a),
      'Train-only English H12 head readout, CPU FP32 replay. Rows/columns are TOKEN POSITIONS, not hidden coordinates. Not native attention or gold dependency edges.')
    v.update(sample_id=sample,query_position=r['anchors'][anchor],axes='row=dependent token; column=candidate head token; English-trained candidate scores, not calibrated probabilities');return v

@router.get('/scalar')
def scalar(sample:str='',block:int=16,anchor:int=Query(0,ge=0,le=3),unit:int=Query(0,ge=0,le=9727),input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    r=source(sample);p,_,_=paths(r)
    if block not in (6,16,35) or anchor>=len(r['anchors']):raise HTTPException(422,'Unsupported captured block/anchor')
    if not p.exists():raise HTTPException(409,'Capture pending')
    with np.load(p) as z:f={f'L{block}_{k}':z[f'L{block}_{k}'] for k in ('x','gate','up','activation','mlp')}
    f[f'L{block}_mlp_input']=f[f'L{block}_x']
    return scalar_values(f,block,anchor,unit,input_coordinate,output_coordinate)|{'sample_id':sample,'scope':'Actual scalar products within all native sums, not single-neuron concept localization.'}

@router.get('/native-path')
def native_path(sample:str='',block:int=16,anchor:int=Query(0,ge=0,le=3),source_position:int=Query(0,ge=0),unit:int=Query(0,ge=0,le=9727),output_coordinate:int=Query(0,ge=0,le=2559)):
    source(sample);p=BASE/'native_paths/fields'/f'{sample}.npz'
    if block not in (16,35):raise HTTPException(422,'Path blocks are16and35')
    if not p.exists():raise HTTPException(409,'Not among24predeclared native path cases, or capture pending')
    with np.load(p) as z:
        positions=z['positions'];n=len(z['token_ids'])
        if anchor>=len(positions) or source_position>positions[anchor]:raise HTTPException(422,'Anchor or source outside visible prefix')
        def a(name):return z[f'L{block}_{name}'][anchor]
        c=a('source_attention_write')[source_position];gs=a('source_gate_read')[source_position];us=a('source_up_read')[source_position]
        written=a('source_MLP_write')[source_position];g=a('gate');u=a('up')
        alloc=.5/(1+np.exp(-np.clip(g,-80,80)))*(gs*u+us*g)
        chain={'block':block,'anchor_token':int(positions[anchor]),'source_token':source_position,'source_token_id':int(z['token_ids'][source_position]),
          'MLP_unit':unit,'output_coordinate':output_coordinate,'source_gate_read':float(gs[unit]),'source_up_read':float(us[unit]),
          'native_gate':float(g[unit]),'native_up':float(u[unit]),'symmetric_source_activation_allocation':float(alloc[unit]),
          'source_MLP_output_coordinate':float(written[output_coordinate]),'native_MLP_output_coordinate':float(a('mlp')[output_coordinate]),
          'attention_weights_all32heads':z[f'L{block}_native_A'][:,anchor,source_position].tolist()}
    return {'sample_id':sample,'chain':chain,'attention_write':field_response(c[None],['source attention write'],0,2560,'All original residual coordinates; complete32-head Wo contraction.'),
      'unit_reads':field_response(np.stack([gs,us,alloc]),['source gate read','source up read','symmetric activation allocation'],0,9728,'All9728units. Symmetric bilinear allocation is one stated convention, not unique causal attribution.'),
      'MLP_write':field_response(written[None],['source MLP write'],0,2560,'All output coordinates; observed sigmoid and RMS denominator fixed, rounding remainders retained in archive.'),
      'scope':'Native computational provenance uses observed attention/gates. It is not an early predictor, causal intervention, or unique semantic pathway.'}

@router.get('/gradient')
def gradient(query:int=Query(0,ge=0,le=767),part:str='content',unit:int=Query(0,ge=0,le=9727),input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    if part not in ('full','content','format'):raise HTTPException(422,'Unknown objective')
    p=BASE/'learning'/f'{part}_factors.npz'
    if not p.exists():raise HTTPException(409,'Gradient not committed')
    with np.load(p) as z:f={k:z[k][query].astype(float) for k in ('x','a','s','bg','bu')}
    g=f['bg'][unit]*f['x'];u=f['bu'][unit]*f['x'];d=f['s'][output_coordinate]*f['a']
    return {'material':compressed(BASE/'program_material.json.gz')[query],'part':part,
      'scalars':{'gate':float(g[input_coordinate]),'up':float(u[input_coordinate]),'down':float(d[unit])},
      'input_terms':field_response(np.stack([g,u]),['gate parameter row','up parameter row'],0,2560,'Every input parameter coordinate.'),
      'output_terms':field_response(d[None],['down parameter row'],0,9728,'Every MLP unit.'),
      'scope':'Supervised current full-vocabulary/conditional/format gradients. Native-valued FP32 MLP/logits, FP64 probability scoring, FP32 adjoints. Content and format are generally nonorthogonal.'}

@router.get('/areas')
def areas():return [{'area':a,'files':n} for a,n in sorted(Counter(k.rsplit('/',1)[0] if '/' in k else '' for k in index()).items())]

@router.get('/files')
def files(area:str='graph'):
    return [{'file':p.name,'bytes':p.stat().st_size} for k,p in sorted(index().items()) if (k.rsplit('/',1)[0] if '/' in k else '')==area]

@router.get('/arrays')
def arrays(area:str='',file:str=''):
    p=registered(area,file);return npz_headers(str(p),p.stat().st_mtime_ns)

@router.get('/array')
def array(area:str='',file:str='',name:str='',row_start:int=Query(0,ge=0),row_count:int=Query(37,ge=1,le=128),start:int=Query(0,ge=0),count:int=Query(8192,ge=1,le=16384)):
    p=registered(area,file)
    with np.load(p) as z:
        if name not in z.files:raise HTTPException(404,'Unknown named tensor')
        a=z[name]
    if a.dtype.kind not in 'buif':raise HTTPException(422,'Non-numerical tensor')
    original=list(a.shape);a=decode(a) if a.dtype==np.uint16 else a
    if a.ndim<2:a=a.reshape(1,-1)
    v=a.reshape(-1,a.shape[-1]);end=min(len(v),row_start+row_count)
    if row_start>=len(v) or start>=v.shape[-1]:raise HTTPException(422,'Page outside original axes')
    selected=v[row_start:end,start:start+count]
    if not np.isfinite(selected).all():raise HTTPException(422,'Undefined diagnostic values in selected page; inspect original archive')
    labels=[str(tuple(map(int,np.unravel_index(i,a.shape[:-1])))) for i in range(row_start,end)]
    scope='Original last-axis order, explicitly paged leading axes. Coordinates, units, vocabulary and sample axes are distinct.'
    if area.startswith('moment_boundary'):scope+=' SYNTHETIC ambient-vector certificate, NOT native reachable language states.'
    result=field_response(v[row_start:end],labels,start,count,scope,tensor_shape=original,total_rows=len(v),row_start=row_start,row_end=end)
    result['whole_field_absmax']=float(abs(selected).max());return result

@router.get('/behavior-index')
def behavior_index(mode:str='own_history'):
    if mode in ('qwen4','qwen14','glm4'):folder=BASE/'scale'/mode/'commits';paths0=folder.glob('*.json')
    elif mode in ('own_history','same_history','long_answers'):paths0=(BASE/mode/'commits').glob('*/*.json')
    else:raise HTTPException(422,'Unknown trajectory collection')
    result=[]
    for p in sorted(paths0):
        r=read(p)
        if 'generated_ids' in r:result.append({k:r[k] for k in ('sample_id','cohort','split')}|{'branch':r.get('branch','native')})
    return result

@router.get('/behavior')
def behavior(mode:str='own_history',sample:str='',branch:str='native'):
    rows=behavior_index(mode)
    if not any(r['sample_id']==sample and r['branch']==branch for r in rows):raise HTTPException(404,'No exact committed trajectory identity')
    p=BASE/'scale'/mode/'commits'/f'{sample}.json' if mode in ('qwen4','qwen14','glm4') else BASE/mode/'commits'/branch/f'{sample}.json'
    record=read(p);material=source(sample)
    if 'answer_scoring' not in record:
        # Legacy own-history parser was a leading-answer diagnostic. Keep it
        # verbatim while exposing explicit terminal scoring as the formal view.
        ids=record['generated_ids'];stopped=bool(record.get('stopped_by_EOS',record.get('EOS',False)))
        record['answer_scoring']=_scoring.score(material,record['generated_text'],ids,{ids[-1]} if stopped and ids else set(),record.get('cap',128))
        record['legacy_parser_scope']='Original leading-answer fields retained for audit only; not final-answer accuracy.'
    manual=next((r for r in read(BASE/'manual_terminal_audit/result.json',{}).get('adjudications',[]) if r['raw_record']==p.relative_to(BASE).as_posix()),None)
    return record|{'material':material,'format_aware_scoring':_terminal.enrich_score(material,record['generated_text'],record['answer_scoring']),
      'manual_terminal_adjudication':manual}

@router.get('/download')
def download(area:str='',file:str=''):return FileResponse(registered(area,file),filename=file)

@router.get('/figure/{name}')
def figure(name:str):
    if name not in {r['path'] for r in read(BASE/'figures/index.json',{}).get('figures',[])}:raise HTTPException(404,'Unknown registered figure')
    return FileResponse(BASE/'figures'/name)
