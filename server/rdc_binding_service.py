"""Read-only source-binding, full native parameters and content/format evidence atlas."""
from collections import Counter
import numpy as np
from fastapi import APIRouter,HTTPException,Query
from fastapi.responses import FileResponse
from server.rdc_feature_service import ROOT,read,decode
from server.rdc_relation_service import field_response
from server.rdc_joint_service import compressed,npz_headers,scalar_values

BASE=ROOT/'tests/glm5/result/rdc_binding_campaign_20260912'
LAW=ROOT/'tests/glm5/result/rdc_law_campaign_20260911'
router=APIRouter(prefix='/api/rdc-binding',tags=['rdc-source-binding-training'])

def materials():
    signed='signed_source/identity_recovery/resolved_material.json.gz'
    if not (BASE/signed).exists():signed='signed_source/natural_material.json.gz'
    paths=['natural_discovery.json.gz','natural_confirmation.json.gz','program_material.json.gz','format_content/prospective_material.json.gz',signed]
    return [r for p in paths if (BASE/p).exists() for r in compressed(BASE/p)]

def source(sid):
    row=next((r for r in materials() if r['sample_id']==sid),None)
    if row is None:raise HTTPException(404,'Unknown registered source')
    return row

def paths(row):
    sid=row['sample_id']
    if row.get('capture_mode')=='signed':return BASE/'signed_source/fields'/f'{sid}.npz',None
    if row['split']=='prospective_depth6':return BASE/'format_content/prospective_fields'/f'{sid}.npz',None
    if row.get('capture_mode')=='main':return LAW/'capture/main/fields'/f'{sid}.npz',LAW/'capture/main/sources'/f'{sid}.npz'
    kind='natural' if row['kind']=='natural' else 'program'
    return BASE/'capture'/kind/f'{sid}.npz',None

def index():return {p.relative_to(BASE).as_posix():p for p in BASE.rglob('*.npz') if p.is_file()}

def registered(area,file):
    key=area+'/'+file
    if '\\' in key or '..' in key.split('/') or key not in index():raise HTTPException(404,'Unknown registered numerical archive')
    return index()[key]

@router.get('/overview')
def overview():
    return {'review':read(BASE/'contract.json',{}).get('corrections',[]),'material':read(BASE/'material_frozen.json',{}),
      'prediction':read(BASE/'prediction/frozen.json',{}),'confirmation':read(BASE/'confirmation/result.json',{}),
      'analysis':read(BASE/'analysis/result.json',{}),'native_bilinear':read(BASE/'native_bilinear/result.json',{}),
      'alpha':read(BASE/'alpha_natural/result.json',{}),'gamma':read(BASE/'gradient_span/result.json',{}),
      'middle':read(BASE/'middle_training/result.json',{}),'profiles':read(BASE/'atlas/result.json',{}),
      'scale':{m:read(BASE/'scale'/m/'result.json',{}) for m in ('qwen4','qwen14','glm4')},
      'live':read(BASE/'binding_live/result.json',{}),'followup':read(BASE/'format_content/decomposition_result.json',{}),
      'behavior_analysis':read(BASE/'analysis/behavior.json',{}),'autonomous':read(BASE/'format_content/autonomous/result.json',{}),
      'theory':read(BASE/'theory_snapshot.json',{}),'figures':read(BASE/'figures/index.json',{}).get('figures',[]),
      'signed_source':{'frozen':read(BASE/'signed_source/frozen.json',{}),'result':read(BASE/'signed_source/result.json',{}),
        'collision':read(BASE/'verification/source_moment_collision/result.json',{}),
        'identity_audit':read(BASE/'signed_source/identity_recovery/result.json',{})},
      'resources':read(BASE/'resources.json',{}),'integrity':read(BASE/'verification/final.json',{}),
      'status':'Evidence states are separate: observations, exact identities, limited forecasts and continued-training effects. No universal language mechanism or original pretraining reconstruction.'}

@router.get('/samples')
def samples():
    return [{k:r[k] for k in ('sample_id','source_group','cohort','split','kind','language','anchors')}|
      {'tokens':len(r['prompt_ids']),'captured':paths(r)[0].exists(),'connected':r.get('connected_held',[])} for r in materials()]

@router.get('/sample')
def sample(sample:str=''):
    r=source(sample);v=BASE/('signed_source/connected_visibility.json.gz' if r.get('capture_mode')=='signed' else 'analysis/connected_visibility.json.gz')
    return r|{'connected_visibility':[s for s in compressed(v) if s['sample_id']==sample] if v.exists() else []}

@router.get('/field')
def field(sample:str='',mode:str='all_layers',anchor:int=Query(0,ge=0,le=3),layer:int=Query(12,ge=0,le=36),view:str='raw'):
    r=source(sample);path,sourcepath=paths(r)
    if not path.exists():raise HTTPException(409,'Capture is pending; no demonstration data are substituted')
    if mode=='all_layers':
        if anchor>=len(r['anchors']):raise HTTPException(422,'Anchor absent')
        with np.load(path) as z:a=decode(z['H'][:,anchor])
        labels=[f'H{i} at token{r["anchors"][anchor]}' for i in range(37)]
    elif mode=='H12_sources':
        with np.load(sourcepath or path) as z:a=decode(z['H12_sources'])
        labels=[f'token{i}: ID{r["prompt_ids"][i]}' for i in range(len(a))]
    elif mode=='fixture_all_tokens':
        folder=LAW/'capture/main/full_fields' if r.get('capture_mode')=='main' else BASE/'signed_source/full_fields' if r.get('capture_mode')=='signed' else BASE/'capture/full_fields'
        fixture=folder/f'{sample}.npz'
        if not fixture.exists():raise HTTPException(409,'This is not a predeclared everylayer/everytoken raw fixture; use complete H12 sources or anchor field')
        with np.load(fixture) as z:a=decode(z['H'][layer])
        labels=[f'token{i}: ID{r["prompt_ids"][i]}' for i in range(len(a))]
    else:raise HTTPException(422,'Unknown field mode')
    if view=='RMS':a=a/np.maximum(np.sqrt(np.mean(a*a,-1,keepdims=True)),1e-8)
    elif view!='raw':raise HTTPException(422,'Unknown value normalization')
    return field_response(a,labels,0,2560,'All native2560coordinates, original order; '+view+'. No threshold or Top-K.')

@router.get('/roles')
def roles(sample:str=''):
    r=source(sample);path,sp=paths(r)
    if not path.exists():raise HTTPException(409,'Source capture pending')
    with np.load(sp or path) as z:h=decode(z['H12_sources'])
    with np.load(BASE/'prediction/role_probe.npz') as z:c=z['coefficients']
    h=h/np.sqrt(np.mean(h*h,-1,keepdims=True)).clip(1e-8);v=np.maximum(h@c[:-1]+c[-1],0)+1e-6;v=v/v.sum(-1,keepdims=True)
    return field_response(v,[f'token{i}: ID{r["prompt_ids"][i]}' for i in range(len(h))],0,6,
      'Columns are predicted coarse role scores, NOT hidden coordinates; no gold current/future dependency graph is an input. These normalized ridge scores are not calibrated probabilities.',
      role_names=['subject','object','oblique','modifier','predicate','other'])

@router.get('/scalar')
def scalar(sample:str='',block:int=35,anchor:int=Query(0,ge=0,le=3),unit:int=Query(0,ge=0,le=9727),input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    r=source(sample);path,_=paths(r)
    if block not in (6,16,35) or anchor>=len(r['anchors']):raise HTTPException(422,'No captured block/anchor')
    if not path.exists():raise HTTPException(409,'Native capture pending')
    with np.load(path) as f:z={f'L{block}_{k}':f[f'L{block}_{k}'] for k in ('x','gate','up','activation','mlp')}
    z[f'L{block}_mlp_input']=z[f'L{block}_x']
    return scalar_values(z,block,anchor,unit,input_coordinate,output_coordinate)|{'sample_id':sample,'scope':'Actual scalar terms inside complete native sums. Not a single-concept or necessity claim.'}

@router.get('/gradient')
def gradient(query:int=Query(0,ge=0,le=767),part:str='full',unit:int=Query(0,ge=0,le=9727),input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    if part in ('full','content','format'):path=BASE/'format_content'/f'{part}_gradient_factors.npz'
    else:raise HTTPException(422,'Unknown exact loss-gradient part')
    if not path.exists():raise HTTPException(409,'Decomposition pending')
    with np.load(path) as z:f={k:z[k][query].astype(float) for k in ('x','a','s','bg','bu')}
    g=f['bg'][unit]*f['x'];u=f['bu'][unit]*f['x'];d=f['s'][output_coordinate]*f['a']
    r=compressed(BASE/'program_material.json.gz')[query]
    return {'material':r,'part':part,'scalars':{'gate':float(g[input_coordinate]),'up':float(u[input_coordinate]),'down':float(d[unit])},
      'factor_archive':str(path.relative_to(BASE)),
      'precision':'本次三部分均使用 FP32 MLP/logits、FP64 概率归一化与直接条件 CE；旧 FP32 概率版完整梯度在 gradient_span 档案中单独保留。',
      'input_terms':field_response(np.stack([g,u]),['gate parameter row','up parameter row'],0,2560,'All native input parameter coordinates.'),
      'output_terms':field_response(d[None],['down parameter row'],0,9728,'All native MLP units.'),
      'scope':'Current supervised gradient factors; full=content+format in real arithmetic, nonorthogonal parts. Not original training history.'}

@router.get('/areas')
def areas():return [{'area':k,'files':v} for k,v in sorted(Counter(s.rsplit('/',1)[0] for s in index()).items())]

@router.get('/files')
def files(area:str='atlas/condition_profiles'):
    return [{'file':p.name,'bytes':p.stat().st_size} for k,p in sorted(index().items()) if k.rsplit('/',1)[0]==area]

@router.get('/arrays')
def arrays(area:str='',file:str=''):
    p=registered(area,file);return npz_headers(str(p),p.stat().st_mtime_ns)

@router.get('/array')
def array(area:str='',file:str='',name:str='',row_start:int=Query(0,ge=0),row_count:int=Query(37,ge=1,le=128),start:int=Query(0,ge=0),count:int=Query(8192,ge=1,le=16384)):
    p=registered(area,file)
    with np.load(p) as z:
        if name not in z.files:raise HTTPException(404,'Unknown named tensor')
        a=z[name]
    if a.dtype.kind not in 'buif':raise HTTPException(422,'Not a numeric array')
    original=list(a.shape);a=decode(a) if a.dtype==np.uint16 else a
    if a.ndim<2:a=a.reshape(1,-1)
    v=a.reshape(-1,a.shape[-1]);end=min(len(v),row_start+row_count);ce=min(v.shape[-1],start+count)
    if row_start>=len(v) or start>=v.shape[-1]:raise HTTPException(422,'Page outside original axes')
    selected=v[row_start:end,start:ce]
    if not np.isfinite(selected).all():raise HTTPException(422,'This page has explicitly undefined diagnostic values; choose another page or download the original tensor')
    labels=[str(tuple(map(int,np.unravel_index(i,a.shape[:-1])))) for i in range(row_start,end)]
    synthetic=' SYNTHETIC ambient-state collision certificate, NOT captured native language states.' if area=='verification/source_moment_collision' else ''
    result=field_response(np.nan_to_num(v[row_start:end],nan=0,posinf=0,neginf=0),labels,start,count,
      'Original final-axis index order; leading axes explicitly flattened/paged. Not all tensors represent the same coordinate/MLP/parameter space.'+synthetic,
      tensor_shape=original,total_rows=len(v),row_start=row_start,row_end=end,download=f'/api/rdc-binding/download?area={area}&file={file}')
    result['whole_field_absmax']=float(abs(selected).max());return result

def behavior_paths(mode):
    if mode=='binding':return list((BASE/'binding_live/commits').glob('*/*.json'))
    if mode=='autonomous':return list((BASE/'format_content/autonomous/commits').glob('*/*.json'))
    if mode=='long_native':return list((BASE/'format_content/native_commits').glob('*.json'))
    if mode in ('qwen4','qwen14','glm4'):return list((BASE/'scale'/mode/'commits').glob('*.json'))
    raise HTTPException(422,'Unknown behavior collection')

@router.get('/behavior-index')
def behavior_index(mode:str='qwen4'):
    result=[]
    for p in sorted(behavior_paths(mode)):
        r=read(p)
        if 'generated' not in r:continue
        result.append({'sample_id':r['sample_id'],'branch':r.get('branch','native'),'cohort':r.get('cohort',r.get('representation','')),'split':r['split']})
    return result

@router.get('/behavior')
def behavior(mode:str='qwen4',sample:str='',branch:str='native'):
    for p in behavior_paths(mode):
        if p.stem!=sample:continue
        r=read(p)
        if r.get('branch','native')==branch:return r|{'material':source(sample)}
    raise HTTPException(404,'No committed matching trajectory')

@router.get('/download')
def download(area:str='',file:str=''):return FileResponse(registered(area,file),filename=file)

@router.get('/figure/{name}')
def figure(name:str):
    if name not in {r['path'] for r in read(BASE/'figures/index.json',{}).get('figures',[])}:raise HTTPException(404,'Unknown registered figure')
    return FileResponse(BASE/'figures'/name)
