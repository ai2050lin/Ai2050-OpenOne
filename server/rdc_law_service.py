"""Read-only formation/operation/composition atlas; no CUDA loads or mutation routes."""
from collections import Counter
import numpy as np
from fastapi import APIRouter,HTTPException,Query
from fastapi.responses import FileResponse
from server.rdc_feature_service import ROOT,read,decode
from server.rdc_relation_service import field_response
from server.rdc_joint_service import compressed,npz_headers,scalar_values

BASE=ROOT/'tests/glm5/result/rdc_law_campaign_20260911'
router=APIRouter(prefix='/api/rdc-law',tags=['rdc-formation-operation-composition'])


def materials():return compressed(BASE/'material.json.gz')+compressed(BASE/'confirmation_material.json.gz')


def source(sample):
    row=next((r for r in materials() if r['sample_id']==sample),None)
    if row is None:raise HTTPException(404,'Unknown registered sample')
    visibility=BASE/'confirmation/combination_visibility/anchor_catalog.json.gz'
    return dict(row,held_combination_visibility=[r for r in compressed(visibility) if r['sample_id']==sample]) if visibility.exists() else row


def scope(row):return 'confirmation' if row['split']=='confirmation' else 'main'


def arrays_index():
    return {str(p.relative_to(BASE)).replace('\\','/'):p for p in BASE.rglob('*.npz') if p.is_file()}


def registered(area,file):
    key=area+'/'+file
    if '\\' in key or '..' in key.split('/') or key not in arrays_index():raise HTTPException(404,'Unknown registered result file')
    return arrays_index()[key]


@router.get('/overview')
def overview():
    first=read(BASE/'formation/initial_stable/result.json',{})
    return {'material':read(BASE/'material_audit.json',{}),'review':read(BASE/'review.json',{}).get('corrections',[]),
        'plan':read(BASE/'plan.json',{}),'resources':read(BASE/'resources.json',{}),'compute':read(BASE/'compute_ledger.json',[]),
        'atlas':read(BASE/'atlas/result.json',{}),'prediction':read(BASE/'prediction/frozen.json',{}),
        'initial_training':{k:v for k,v in first.items() if k not in ('actual_updates','gradient_pair_controls','prediction_manifest')},
        'training':read(BASE/'formation/trajectories/result.json',{}),'gradient_controls':read(BASE/'formation/gradient_controls/result.json',{}),
        'comparison':read(BASE/'analysis/result.json',{}),'confirmation':read(BASE/'confirmation/result.json',{}),
        'combination_visibility':read(BASE/'confirmation/combination_visibility/result.json',{}),
        'deployment':read(BASE/'deployment/result.json',{}),'history_followup':read(BASE/'own_history/result.json',{}),
        'deployment_paired':read(BASE/'deployment/paired_analysis.json',{}),
        'scale':{m:read(BASE/'scale'/m/'result.json',{}) for m in ('qwen4','qwen14','glm4')},
        'theory':read(BASE/'theory_snapshot.json',{}),'figures':read(BASE/'figures/index.json',{}).get('figures',[]),
        'integrity':read(BASE/'verification/final.json',{}),'status':'Observation and limited prediction/training evidence; no universal language closure or original pretraining reconstruction.'}


@router.get('/samples')
def samples():
    return [{k:r[k] for k in ('sample_id','cohort','split','language','kind','source_group','anchors')}|
        {'tokens':len(r['prompt_ids']),'full_field':(BASE/'capture'/scope(r)/'full_fields'/f"{r['sample_id']}.npz").exists(),
         'held_relation_combinations':r.get('held_relation_combinations',[])} for r in materials()]


@router.get('/sample')
def sample(sample:str=''):return source(sample)


@router.get('/field')
def field(sample:str='',mode:str='all_layers',anchor:int=Query(0,ge=0,le=2),layer:int=Query(12,ge=0,le=36),view:str='raw'):
    r=source(sample);folder=BASE/'capture'/scope(r)
    if mode=='all_layers':
        if anchor>=len(r['anchors']):raise HTTPException(422,'Anchor absent in this source; QA has one anchor')
        path=folder/'fields'/f'{sample}.npz'
        with np.load(path,allow_pickle=False) as z:v=decode(z['H'][:,anchor])
        labels=[f'H{i}: token {r["anchors"][anchor]}' for i in range(37)];li=None
    elif mode=='all_tokens':
        path=folder/'full_fields'/f'{sample}.npz'
        if not path.exists():raise HTTPException(409,'Only12 predeclared main fixtures retain everylayer/everytoken raw field; this source retains allanchors/allH12sources and full-token statistics.')
        with np.load(path,allow_pickle=False) as z:v=decode(z['H'][layer])
        labels=[f'token {i}: ID {t}' for i,t in enumerate(r['prompt_ids'])];li=layer
    else:raise HTTPException(422,'Unknown field layout')
    note='All native2560coordinates, original order, BF16 values decoded. No threshold or Top-K.'
    if view=='RMS':v=v/np.maximum(np.sqrt(np.mean(v*v,1,keepdims=True)),1e-12);note='Every row divided by its complete-coordinate RMS.'
    elif view=='train_z':
        with np.load(BASE/'atlas/training_coordinate_rulers.npz') as z:mu,sd=z['mean'],np.maximum(z['std'],1e-6)
        v=(v-mu)/sd if li is None else (v-mu[li])/sd[li];note='Training-only per-layer/per-coordinate mean/std, stdfloor1e-6; original coordinate order.'
    elif view!='raw':raise HTTPException(422,'Unknown view')
    return field_response(v,labels,0,2560,note,download=f'/api/rdc-law/download?area={path.parent.relative_to(BASE).as_posix()}&file={path.name}')


@router.get('/areas')
def areas():
    counts=Counter(key.rsplit('/',1)[0] for key in arrays_index())
    return [{'area':a,'files':n} for a,n in sorted(counts.items())]


@router.get('/files')
def files(area:str='atlas/joint_products'):
    valid={r['area'] for r in areas()}
    if area not in valid:raise HTTPException(404,'Unknown result area')
    profiles={r['path'].replace('\\','/'):r for r in read(BASE/'atlas/result.json',{}).get('profiles',[])}
    result=[]
    for k,p in sorted(arrays_index().items()):
        if k.rsplit('/',1)[0]!=area:continue
        descriptor=profiles.get(k)
        label=f"{descriptor['condition']} · {descriptor['anchors']} anchors / {descriptor['source_groups']} sources" if descriptor else p.name
        result.append({'file':p.name,'bytes':p.stat().st_size,'label':label,'profile':descriptor})
    return sorted(result,key=lambda r:r['label']) if area=='atlas/condition_profiles' else result


@router.get('/arrays')
def array_headers(area:str='',file:str=''):
    p=registered(area,file);return npz_headers(str(p),p.stat().st_mtime_ns)


@router.get('/array')
def array(area:str='',file:str='',name:str='',row_start:int=Query(0,ge=0),row_count:int=Query(37,ge=1,le=128),start:int=Query(0,ge=0),count:int=Query(8192,ge=1,le=16384)):
    p=registered(area,file)
    with np.load(p,allow_pickle=False) as z:
        if name not in z.files:raise HTTPException(404,'Unknown named array')
        a=z[name]
    if a.dtype.kind not in 'buif':raise HTTPException(422,'Selected array is not numeric')
    original=list(a.shape);a=decode(a) if a.dtype==np.uint16 else a
    if a.ndim<2:a=a.reshape(1,-1)
    v=a.reshape(-1,a.shape[-1]);end=min(len(v),row_start+row_count);column_end=min(v.shape[-1],start+count)
    if row_start>=len(v) or start>=v.shape[-1]:raise HTTPException(422,'Page is outside original tensor axes')
    selected=v[row_start:end,start:column_end]
    if not np.isfinite(selected).all():raise HTTPException(422,'Selected page includes declared undefined diagnostic values (e.g. finaltoken nextNLL or unavailable permutation); choose finite columns/rows or download exact array.')
    labels=[str(tuple(map(int,np.unravel_index(i,a.shape[:-1])))) for i in range(row_start,end)]
    response=field_response(np.nan_to_num(v[row_start:end],nan=0,posinf=0,neginf=0),labels,start,count,
        'Original last-axis order, explicitly paged leading axes. Hidden coordinates, MLPunits, vocabulary, parameterindices and trainingfactors are distinct; tensor name and shape retain meaning. Nonfinite entries OUTSIDE this requested page are excluded from its color scale.',
        tensor_shape=original,total_rows=len(v),row_start=row_start,row_end=end,
        download=f'/api/rdc-law/download?area={area}&file={file}')
    response['whole_field_absmax']=float(np.max(abs(selected)))
    return response


@router.get('/scalar')
def scalar(sample:str='',block:int=35,anchor:int=Query(0,ge=0,le=2),unit:int=Query(0,ge=0,le=9727),input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    r=source(sample)
    if block not in (6,16,35) or anchor>=len(r['anchors']):raise HTTPException(422,'Requested block/anchor was not captured')
    with np.load(BASE/'capture'/scope(r)/'fields'/f'{sample}.npz') as f:
        z={f'L{block}_{k}':f[f'L{block}_{k}'] for k in ('x','gate','up','activation','mlp')}
    z[f'L{block}_mlp_input']=z[f'L{block}_x']
    return scalar_values(z,block,anchor,unit,input_coordinate,output_coordinate)|{'sample_id':sample,'position':r['anchors'][anchor],
        'scope':'Arbitrary actual scalar path in complete sums, not an independent concept or necessity claim.'}


@router.get('/gradient')
def gradient(panel_index:int=Query(0,ge=0,le=287),unit:int=Query(0,ge=0,le=9727),input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    row=read(BASE/'formation/protocol.json')['panel'][panel_index]
    with np.load(BASE/'formation/initial_stable/complete_gradient_factors.npz') as z:
        f={k:z[k][panel_index].astype(float) for k in ('x','a','s','bg','bu')};gram=z['gram_total'][panel_index]
    g=f['bg'][unit]*f['x'];u=f['bu'][unit]*f['x'];d=f['s'][output_coordinate]*f['a']
    return {'query':row,'scalar_parameter_derivatives':{'gate':float(g[input_coordinate]),'up':float(u[input_coordinate]),'down':float(d[unit])},
        'input_gradient_terms':field_response(np.stack([g,u]),['Entire gate row gradient','Entire up row gradient'],0,2560,'Full native parameter row: b_g[k]*x or b_u[k]*x; all coordinates retained.'),
        'output_gradient_terms':field_response(d[None],['Entire down row gradient'],0,9728,'Full native parameter row: s[j]*activation; all units retained.'),
        'all_query_gradient_inner_products':field_response(gram[None],['Full74711040parameter gradient dot products to288queries'],0,288,'Pair inner products include all native scalar parameters; paired queries are not independent replicates.'),
        'scope':'Initial FP32 smooth full-vocabulary CE gradient, actual next-token target is a training label; not a future-token-free semantic forecast.'}


@router.get('/behavior-index')
def behavior_index(model:str='live'):
    if model in ('live','own_history'):
        folders=list((BASE/('deployment/rollouts' if model=='live' else 'own_history/commits')).glob('*'))
        return [{'sample_id':r['sample_id'],'branch':r['branch'],'cohort':r['cohort'],'language':r['language'],'kind':r['kind']}
            for folder in folders if folder.is_dir() for p in sorted(folder.glob('*.json')) for r in [read(p)]]
    if model not in ('qwen4','qwen14','glm4'):raise HTTPException(422,'Unknown model')
    return [{'sample_id':r['sample_id'],'branch':'native','cohort':r['cohort'],'language':r['language'],'kind':'QA'} for p in sorted((BASE/'scale'/model/'qa/commits').glob('*.json')) for r in [read(p)]]


@router.get('/behavior')
def behavior(model:str='live',sample:str='',branch:str='native'):
    if not any(r['sample_id']==sample and r['branch']==branch for r in behavior_index(model)):raise HTTPException(404,'Unknown committed trajectory')
    if model in ('live','own_history'):path=BASE/('deployment/rollouts' if model=='live' else 'own_history/commits')/branch/f'{sample}.json'
    else:path=BASE/'scale'/model/'qa/commits'/f'{sample}.json'
    return read(path)|{'material':source(sample)}


@router.get('/download')
def download(area:str='',file:str=''):return FileResponse(registered(area,file),filename=file)


@router.get('/figure/{name}')
def figure(name:str):
    if name not in {r['path'] for r in read(BASE/'figures/index.json',{}).get('figures',[])}:raise HTTPException(404,'Unknown registered figure')
    return FileResponse(BASE/'figures'/name)
