"""Read-only ordinary-language operator atlas. No model loading or job-launch endpoints."""
from collections import Counter
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from server.rdc_feature_service import ROOT, read, arrays, decode
from server.rdc_relation_service import field_response
from server.rdc_joint_service import compressed, npz_headers, scalar_values

BASE = ROOT/'tests/glm5/result/rdc_operator_atlas_20260911'
router = APIRouter(prefix='/api/rdc-operator', tags=['rdc-native-conditional-operator'])


def material():
    return compressed(BASE/'material.json.gz')


def sample_row(sample):
    row = next((r for r in material() if r['sample_id']==sample), None)
    if row is None:
        raise HTTPException(404, 'Unknown registered source ID')
    return row


def capture_scope(row):
    return 'confirmation' if row['split']=='confirmation' else 'main'


@router.get('/overview')
def overview():
    return {'plan':read(BASE/'plan.json',{}), 'review':read(BASE/'review.json',{}), 'material':read(BASE/'material_audit.json',{}),
        'QA_material':read(BASE/'qa_extension.json',{}), 'resources':read(BASE/'resources.json',{}),
        'capture':{s:read(BASE/'capture'/s/'result.json',{}) for s in ('main','confirmation')},
        'observation':read(BASE/'observation/result.json',{}), 'operators':read(BASE/'operators/result.json',{}),
        'choices':read(BASE/'operators/frozen.json',{}).get('choices',{}),
        'confirmation':read(BASE/'confirmation/result.json',{}), 'compilation':read(BASE/'compiled/result.json',{}),
        'paired_probability':read(BASE/'compiled/paired_audit.json',{}),
        'QA_atlas':read(BASE/'qa_atlas/result.json',{}), 'calculus':read(BASE/'calculus/result.json',{}),
        'structure':read(BASE/'structure/result.json',{}), 'operations':read(BASE/'operations/result.json',{}),
        'operation_response_agreement':read(BASE/'operations/response_agreement_audit.json',{}),
        'identity_audit':read(BASE/'identity_audit/result.json',{}), 'cue_correction':read(BASE/'identity_audit/correction_result.json',{}),
        'corrected_cue_composition':read(BASE/'identity_audit/corrected_cue_composition.json',{}),
        'metric_followup':read(BASE/'metric_followup/result.json',{}),
        'metric_paired':read(BASE/'metric_followup/paired_audit.json',{}),
        'metric_population':read(BASE/'metric_followup/population_geometry/result.json',{}),
        'behavior':read(BASE/'behavior/result.json',{}),
        'theory':read(BASE/'theory_snapshot.json',{}),
        'scale':{k:read(BASE/'scale'/k/'result.json',{}) for k in ('qwen4','qwen14','glm4')},
        'matched_scale_audit':read(BASE/'scale/paired_audit.json',{}),
        'QA':{k:{s:read(BASE/'qa'/k/s/'result.json',{}) for s in ('main','confirmation')} for k in ('qwen4','qwen14','glm4')},
        'figures':read(BASE/'figures/index.json',{}).get('figures',[]), 'integrity':read(BASE/'verification/final.json',{}),
        'status':'Natural observations and tested local approximations; not a recovered universal semantic gear or AGI closure.'}


@router.get('/samples')
def samples():
    return [{k:r[k] for k in ('sample_id','language','split','source_group','title','anchors')} |
        {'tokens':len(r['prompt_ids']), 'full_field':(BASE/'capture'/capture_scope(r)/'full_fields'/f'{r["sample_id"]}.npz').exists(),
         'committed':(BASE/'capture'/capture_scope(r)/'commits'/f'{r["sample_id"]}.json.gz').exists()} for r in material()]


@router.get('/sample')
def sample(sample:str=''):
    r=sample_row(sample)
    commit=BASE/'capture'/capture_scope(r)/'commits'/f'{sample}.json.gz'
    return {**r, 'capture':compressed(commit) if commit.exists() else None,
        'annotation_scope':'Human question/answer spans are retrospective analysis objects, not online labels. Lexical cues are not gold syntactic/semantic roles.'}


@router.get('/field')
def field(sample:str='',mode:str='all_layers',layer:int=Query(12,ge=0,le=36),anchor:int=Query(0,ge=0,le=1),
          view:str='raw',start:int=Query(0,ge=0),count:int=Query(2560,ge=1,le=2560)):
    r=sample_row(sample)
    folder=BASE/'capture'/capture_scope(r)
    if mode=='all_layers':
        z=arrays(folder/'fields'/f'{sample}.npz')
        v=decode(z['H'][:,anchor])
        labels=[f'H{l} at token {r["anchors"][anchor]}' for l in range(37)]
        li=None
    elif mode=='all_tokens':
        path=folder/'full_fields'/f'{sample}.npz'
        if not path.exists():
            raise HTTPException(409,'This source has full-anchor fields and all-token statistics, but is not a predeclared all-token raw fixture.')
        v=decode(arrays(path)['H'][layer]);labels=[f'token {i}: {t}' for i,t in enumerate(r['tokens'])];li=layer
    else:
        raise HTTPException(422,'Unknown field layout')
    note='Original BF16 full native coordinates; no sorting or Top-K.'
    if view=='RMS':
        v=v/np.maximum(np.sqrt(np.mean(v.astype(float)**2,1,keepdims=True)),1e-12)
        note='Each displayed row divided by its own full-coordinate RMS; all residual coordinates retained.'
    elif view=='train_z':
        scales=arrays(BASE/'observation/training_scales.npz')
        mu,sd=scales['mean'],scales['standard_deviation']
        v=(v-mu)/sd if li is None else (v-mu[li])/sd[li]
        note='Frozen TRAIN ordinary-token coordinate mean/SD at each actual layer; initial positions excluded from scale fit, rare internal events included. No clipping.'
    elif view!='raw':
        raise HTTPException(422,'Unknown normalization')
    return field_response(v,labels,start,count,note,source=sample,tensor_layout=mode,layer=layer,
        download=f'/api/rdc-operator/download?area=capture/{capture_scope(r)}/'+('full_fields' if mode=='all_tokens' else 'fields')+f'&file={sample}.npz')


def allowed_areas():
    areas=['observation','operators','confirmation','calculus','structure','qa_atlas','verification','metric_followup','precision','operations']
    areas += [f'metric_followup/{kind}' for kind in ('fields','full_vocab','autonomous_fields','readout_geometry','population_geometry')]
    areas += ['metric_followup/math_geometry/readout_geometry']
    areas += ['identity_audit/correction_fields','identity_audit/corrected_moments']
    areas += [f'compiled/{s}/full_vocab' for s in ('validation','confirmation')]+['compiled/autonomous_full_vocab','operations/fields']
    areas += [f'capture/{s}/{kind}' for s in ('pilot','main','confirmation') for kind in ('fields','factors','full_fields','energies','moments')]
    areas += [f'qa/{m}/{s}/fields' for m in ('qwen4','qwen14','glm4') for s in ('main','confirmation')]
    areas += [f'scale/{m}/{kind}' for m in ('qwen4','qwen14','glm4') for kind in ('fields','operators','moments')]
    areas += [f'behavior/{m}' for m in ('qwen4','qwen14','glm4')]
    return areas


def registered_file(area,file):
    if area not in allowed_areas() or '/' in file or '\\' in file or file not in {p.name for p in (BASE/area).glob('*.npz')}:
        raise HTTPException(404,'Unknown registered result array')
    return BASE/area/file


@router.get('/areas')
def areas():
    return [{'area':a,'files':sum(1 for _ in (BASE/a).glob('*.npz'))} for a in allowed_areas() if (BASE/a).exists()]


@router.get('/files')
def files(area:str='operators'):
    if area not in allowed_areas():
        raise HTTPException(422,'Unknown result area')
    return [{'file':p.name,'bytes':p.stat().st_size} for p in sorted((BASE/area).glob('*.npz'))]


@router.get('/arrays')
def array_index(area:str='operators',file:str=''):
    p=registered_file(area,file)
    return npz_headers(str(p),p.stat().st_mtime_ns)


@router.get('/array')
def array(area:str='operators',file:str='',name:str='',row_start:int=Query(0,ge=0),row_count:int=Query(37,ge=1,le=128),
          start:int=Query(0,ge=0),count:int=Query(9728,ge=1,le=151936)):
    p=registered_file(area,file)
    z=arrays(p)
    if name not in z:
        raise HTTPException(404,'Unknown array name')
    a=z[name];a=decode(a) if a.dtype==np.uint16 else a
    if a.ndim==0:
        a=a.reshape(1,1)
    if a.ndim==1:
        a=a[None]
    v=a.reshape(-1,a.shape[-1])
    if row_start>=len(v):
        raise HTTPException(422,'Row outside original leading axes')
    end=min(len(v),row_start+row_count)
    # Energy archives have an explicit missing final next-token NLL. JSON must not emit NaN.
    if not np.isfinite(v[row_start:end]).all():
        raise HTTPException(422,'Selected diagnostic contains an explicitly undefined final next-token NLL; download the lossless array, or select defined rows.')
    labels=[str(tuple(map(int,np.unravel_index(i,a.shape[:-1])))) for i in range(row_start,end)]
    note=('Synthetic CPU algebra calibration, NOT native LLM coordinates or language-task evidence. All declared synthetic columns retained.' if 'math_geometry' in area else
        'Unchanged native last-axis order. Original uint16 is decoded BF16; float arrays are stated moments/predictions/diagnostics. Leading axes are explicitly paged; no silent dimensional truncation.')
    return field_response(v[row_start:end],labels,start,count,note,
        tensor_shape=list(a.shape),total_rows=len(v),row_start=row_start,row_end=end,
        download=f'/api/rdc-operator/download?area={area}&file={file}')


@router.get('/scalar')
def scalar(sample:str='',block:int=6,anchor:int=Query(0,ge=0,le=1),unit:int=Query(0,ge=0,le=9727),
           input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    r=sample_row(sample)
    if block not in (6,16,34):
        raise HTTPException(422,'Block not captured in this factor archive')
    z=dict(arrays(BASE/'capture'/capture_scope(r)/'factors'/f'{sample}.npz'))
    z[f'L{block}_mlp_input']=z[f'L{block}_x']
    result=scalar_values(z,block,anchor,unit,input_coordinate,output_coordinate)
    return {**result,'sample_id':sample,'native_token':r['anchors'][anchor],'block':block,
        'scope':'One arbitrary scalar path within complete native sums; parameter identity is not semantic necessity.'}


@router.get('/qa-index')
def qa_index(model:str='qwen4',scope:str='main'):
    if model not in ('qwen4','qwen14','glm4') or scope not in ('main','confirmation'):
        raise HTTPException(422,'Unknown native QA scope')
    return [{k:r[k] for k in ('question_id','sample_id','language','question_type','question','normalized_full_EM','stopped_by_native_EOS')}
            for p in sorted((BASE/'qa'/model/scope/'commits').glob('*.json')) for r in [read(p)]]


@router.get('/qa')
def qa(model:str='qwen4',scope:str='main',id:str=''):
    if id not in {r['question_id'] for r in qa_index(model,scope)}:
        raise HTTPException(404,'Unknown committed question')
    value=read(BASE/'qa'/model/scope/'commits'/f'{id}.json')
    if model=='qwen4' and scope=='main':
        edge=next((r for r in compressed(BASE/'qa_atlas/hyperedges.json.gz') if r['question_id']==id),None)
        value={**value,'typed_hyperedge':edge}
    return {**value,'raw_query_field':f'qa/{model}/{scope}/fields/{id}.npz'}


@router.get('/generation-index')
def generation_index():
    return [{k:r[k] for k in ('sample_id','language','initial_text')} for p in sorted((BASE/'compiled/autonomous').glob('*.json')) for r in [read(p)]]


@router.get('/generation')
def generation(sample:str=''):
    if sample not in {r['sample_id'] for r in generation_index()}:
        raise HTTPException(404,'Unknown committed autonomous source')
    result=read(BASE/'compiled/autonomous'/f'{sample}.json')
    path=BASE/'metric_followup/autonomous'/f'{sample}.json'
    if path.exists():
        hybrid=read(path)
        if hybrid['initial_ids']!=result['initial_ids']:
            raise HTTPException(409,'Autonomous prefix identity differs; no branch merge allowed')
        a=result['branches']['native']['generated_ids'];b=hybrid['generated_ids']
        common=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        result={**result,'branches':{**result['branches'],'output_selected_hybrid':{**hybrid,'token_count':len(b),
            'first_branch_step_1based':common+1 if a!=b else None,'identical_prefix_tokens':common}}}
    return result


@router.get('/operation-index')
def operation_index():
    return [{k:r[k] for k in ('question_id','language','question')} for p in sorted((BASE/'operations/commits').glob('*.json')) for r in [read(p)]]


@router.get('/operation')
def operation(id:str=''):
    if id not in {r['question_id'] for r in operation_index()}:
        raise HTTPException(404,'Unknown committed ordering operation')
    return read(BASE/'operations/commits'/f'{id}.json')


@router.get('/download')
def download(area:str='',file:str=''):
    return FileResponse(registered_file(area,file),filename=file)


@router.get('/figure/{name}')
def figure(name:str):
    if name not in {r['path'] for r in read(BASE/'figures/index.json',{}).get('figures',[])}:
        raise HTTPException(404,'Unknown registered figure')
    return FileResponse(BASE/'figures'/name,media_type='image/png')
