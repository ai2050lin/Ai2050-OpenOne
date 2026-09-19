"""Bounded read-only views of the independent joint atlas. Never load models or launch jobs."""
import gzip,json,zipfile
from collections import Counter,defaultdict
from functools import lru_cache
import numpy as np
from fastapi import APIRouter,HTTPException,Query
from fastapi.responses import FileResponse
from server.rdc_feature_service import ROOT,read,arrays,decode
from server.rdc_relation_service import field_response,parameter_reader

BASE=ROOT/'tests/glm5/result/rdc_joint_atlas_20260911'
router=APIRouter(prefix='/api/rdc-joint',tags=['rdc-joint-atlas'])


@lru_cache(maxsize=5)
def zipped(path,mtime):return json.loads(gzip.decompress(open(path,'rb').read()).decode('utf-8'))


def compressed(path):
    if not path.exists():raise HTTPException(409,'Artifact is not committed yet')
    return zipped(str(path),path.stat().st_mtime_ns)


@lru_cache(maxsize=1024)
def npz_headers(path,mtime):
    """Read tensor metadata without decompressing every full native field on index requests."""
    result=[]
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):continue
            with archive.open(name) as stream:
                version=np.lib.format.read_magic(stream)
                shape,order,dtype=(np.lib.format.read_array_header_1_0(stream) if version==(1,0) else np.lib.format.read_array_header_2_0(stream))
            result.append({'array':name[:-4],'shape':list(shape),'dtype':np.dtype(dtype).str})
    return result


def material(scope):
    if scope not in ('main','fresh'):raise HTTPException(422,'Unknown material scope')
    return compressed(BASE/('material.json.gz' if scope=='main' else 'fresh_material.json.gz'))


def row(scope,sample):
    value=next((r for r in material(scope) if r['sample_id']==sample),None)
    if value is None:raise HTTPException(404,'Unknown sample ID')
    return value


@router.get('/overview')
def overview():
    return {'material':read(BASE/'material_audit.json',{}),'review':read(BASE/'review.json',{}).get('corrections',[]),
        'allocation':read(BASE/'resource_allocation.json',{}),'execution_update':read(BASE/'execution_update.json',{}),
        'choices':read(BASE/'frozen.json',{}).get('choices',{}),'layers':read(BASE/'layer_atlas/result.json',{}),
        'prior_confirmation':read(BASE/'prior_confirmation/result.json',{}),'relations':read(BASE/'relation_atlas/result.json',{}),
        'coverage':read(BASE/'relation_atlas/coverage.json',{}),'probability':read(BASE/'probability_training/result.json',{}).get('probability',[]),
        'confirmation':read(BASE/'confirmation/result.json',{}),'native':read(BASE/'native_factors/result.json',{}),
        'generation':read(BASE/'generation/result.json',{}),'scale':{m:read(BASE/'scale'/m/'result.json',{}) for m in ('qwen4','qwen14','glm4')},
        'figures':read(BASE/'figures/index.json',{}).get('figures',[]),'extension':read(BASE/'extension/result.json',{
            'amplification':read(BASE/'extension/amplification.json',{}),'event_trace':read(BASE/'extension/event_trace/result.json',{}),
            'temperature':read(BASE/'extension/temperature/result.json',{}),'tail_confirmation':read(BASE/'extension/tail_confirmation/result.json',{}),
            'native_regimes':read(BASE/'extension/native_regimes/result.json',{})}),
        'integrity':read(BASE/'verification/final.json',read(BASE/'verification/phase2720_integrity.json',{})),
        'status':'Independent conditional structure and probability adaptation; not autonomous language closure or a recovered universal semantic gear dictionary.'}


@router.get('/samples')
def samples(scope:str='fresh'):
    return [{k:r[k] for k in ('sample_id','language','split','genre','text','positions','source_group','language_mode_families')}|
        {'tokens':len(r['prompt_ids']),'generated':(BASE/'generation/commits'/f'{r["sample_id"]}.json').exists(),
         'native_factors':(BASE/'native_factors/fields'/f'{r["sample_id"]}.npz').exists()} for r in material(scope)]


@router.get('/sample')
def sample(scope:str='fresh',sample:str=''):
    value=row(scope,sample)
    return {**value,'capture':read(BASE/scope/'commits'/f'{sample}.json',{}),
        'archive_status':(BASE/scope/'fields'/f'{sample}.npz').exists(),
        'warning':'All entity/discourse/UD labels here are retrospective source annotations, NOT online forecast inputs or native cognitive correctness labels.'}


@router.get('/field')
def field(scope:str='fresh',sample:str='',layer:str='h12',position_index:int=Query(0,ge=0,le=5),
          view:str='raw',start:int=Query(0,ge=0),count:int=Query(2560,ge=1,le=2560)):
    r=row(scope,sample)
    if layer not in ('h12','h23','h36','postnorm','all_layers'):raise HTTPException(422,'Unknown field')
    z=arrays(BASE/scope/'fields'/f'{sample}.npz')
    if layer=='all_layers':
        v=decode(z['layers'][:,position_index]);labels=[f'H{i} at native token {r["positions"][position_index]}' for i in range(len(v))]
    else:
        v=decode(z[layer]);labels=[f'token {i}: {t}' for i,t in enumerate(r['tokens'])] if layer!='postnorm' else [f'token {i}' for i in r['positions']]
    note='Raw original BF16 values, full native coordinate order'
    if view=='source_RMS':
        v=v/np.maximum(np.sqrt(np.mean(v.astype(float)**2,axis=1,keepdims=True)),1e-8);note='Per-row own full-coordinate RMS normalization; not training z-score'
    elif view=='train_z':
        if layer not in ('h12','h23'):raise HTTPException(422,'Training all-token z scales only for H12/H23')
        sc=arrays(BASE/'relation_atlas/training_scales.npz');li=('h12','h23').index(layer)
        v=(v-sc['mean'][li])/sc['standard_deviation'][li];note='Frozen training all-token coordinate z score, no coordinate reordering'
    elif view!='raw':raise HTTPException(422,'Unknown normalization')
    return field_response(v,labels,start,count,note,download=f'/api/rdc-joint/download?scope={scope}&sample={sample}',
        collection='H12/H23/H36 all tokens; all37 boundaries only at six retained positions; finalnorm separate.')


@router.get('/prediction')
def prediction(scope:str='fresh',sample:str='',choice:str='current_KL',anchor:int=Query(0,ge=0,le=1),start:int=Query(0,ge=0),count:int=Query(2560,ge=1,le=2560)):
    r=row(scope,sample);choices=read(BASE/'frozen.json',{}).get('choices',{})
    if choice not in choices:raise HTTPException(422,'Unknown frozen choice')
    forecast='temporal' if choice.startswith('temporal') else 'current';name=choices[choice]
    meta=read(BASE/'features'/scope/'rows.json',[])
    if scope=='main':
        if r['split'] not in ('validation','test'):raise HTTPException(409,'Frozen forecast display requires held-out main row')
        meta=[m for m in meta if m['split']==r['split']]
        kl=name.startswith('KL_');folder=BASE/('probability_training' if kl else 'rules')/forecast/(name[3:] if kl else name)
        pred=arrays(folder/'predictions.npz')[r['split']]
    else:pred=arrays(BASE/'confirmation/predictions'/f'{forecast}_{name}.npz')['prediction']
    ix=next((i for i,m in enumerate(meta) if m['sample_id']==sample and m['anchor']==anchor),None)
    if ix is None:raise HTTPException(409,'Prediction not committed for sample')
    p=pred[ix,-2560:];z=arrays(BASE/scope/'fields'/f'{sample}.npz');actual=decode(z['h36'][r['anchors'][anchor]+int(forecast=='temporal')])
    return field_response(np.stack([actual,p,p-actual]),['Native raw H36','Frozen predicted H36','Prediction minus native'],start,count,
        'Raw shared-scale coordinates; MSE and output KL are distinct',route=name,MSE=float(np.mean((p.astype(float)-actual)**2)),
        available_inputs='True current H12 at known sources' if forecast=='current' else 'Previous true H36 + known new embedding; history choice also true past H12 at ONE layer + predicted new query, not actual new H12')


@router.get('/matrix')
def matrix(relation:str='ud:nmod',control:str='exact_distance_POS_noninitial',split:str='test',view:str='train_z',
           row_start:int=Query(0,ge=0,le=2559),column_start:int=Query(0,ge=0,le=2559),count:int=Query(48,ge=1,le=256)):
    if split not in ('train','test') or view not in ('raw','train_z','source_RMS'):raise HTTPException(422,'Unknown matrix scope')
    entries=compressed(BASE/'relation_atlas/pair_index.json.gz')
    if relation not in {e['relation'] for e in entries} or control not in ('exact_distance_POS_noninitial','distance_band_POS_noninitial','same_dependent_ID'):raise HTTPException(422,'Unknown registered relation/control')
    selected=[e for e in entries if e['relation']==relation and e['split']==split and e['controls'][control]]
    if not selected:raise HTTPException(409,'No exact matched observations; no fallback to easier control')
    groups=defaultdict(set);by_source=defaultdict(list)
    for e in selected:groups[e['source_group']].add(e['sample_id']);by_source[e['sample_id']].append(e)
    re=min(2560,row_start+count);ce=min(2560,column_start+count);answer=np.zeros((re-row_start,ce-column_start),float)
    sc=arrays(BASE/'relation_atlas/training_scales.npz')
    for sid,ee in by_source.items():
        z=arrays(BASE/'main/fields'/f'{sid}.npz');full=[decode(z[k]) for k in ('h12','h23')]
        if view=='source_RMS':full=[(h/np.maximum(np.sqrt(np.mean(h.astype(float)**2,axis=1,keepdims=True)),1e-8)).astype(np.float32) for h in full]
        elif view=='train_z':full=[((h-sc['mean'][i])/sc['standard_deviation'][i]).astype(np.float32) for i,h in enumerate(full)]
        a,b=full[0][:,row_start:re].astype(float),full[1][:,column_start:ce].astype(float)
        weight=1/len(groups)/len(groups[ee[0]['source_group']])/len(ee)
        for e in ee:
            contrast=np.outer(a[e['dependent']],b[e['head']])
            contrast-=sum(np.outer(a[i],b[j]) for i,j in e['controls'][control])/len(e['controls'][control])
            answer+=weight*contrast
    record=next((r for r in read(BASE/'relation_atlas/result.json',{}).get('entries',[]) if r['relation']==relation and r['control']==control and r.get('view')==view),None)
    return {'values':answer.tolist(),'labels':[f'H12 coordinate {i}' for i in range(row_start,re)],'native_width':2560,
        'start':column_start,'end':ce,'whole_field_absmax':float(np.max(np.abs(answer))), 'row_start':row_start,'groups':len(groups),'windows':len(by_source),'statistics':record,
        'normalization':view+'; exact full-source document→window→edge contrast. FP64 tile accounting can differ slightly from FP32 full BLAS; no rank reduction.',
        'scope':'Coordinates belong to H12/H23. Annotation contrast is not a physical edge or semantic causal effect.'}


@router.get('/factors')
def factors(sample:str='',block:int=6,part:str='activation',start:int=Query(0,ge=0),count:int=Query(9728,ge=1,le=9728)):
    row('fresh',sample);valid=read(BASE/'native_factors/result.json',{}).get('blocks',[])
    if block not in valid or part not in ('input','attention','mlp_input','gate','up','activation','mlp','output','attention_probability'):raise HTTPException(422,'Unknown recorded factor')
    z=arrays(BASE/'native_factors/fields'/f'{sample}.npz');v=decode(z[f'L{block}_{part}'])
    if part=='attention_probability':
        # Stored [heads,4 queries,sources], retain every head/query/source.
        v=v.reshape(-1,v.shape[-1]);labels=[f'head {h}, query slot {q}' for h in range(32) for q in range(4)]
        note='All32 native heads ×4 queries; COLUMNS ARE SOURCE TOKEN POSITIONS, not residual coordinates'
    else:
        labels=[f'native token {p}' for p in z['positions']];note='All9728 MLP units' if part in ('gate','up','activation') else 'All2560 residual coordinates'
    return field_response(v,labels,start,count,note+'; original BF16 factor, not a semantic necessity claim')


@router.get('/scalar')
def scalar(sample:str='',block:int=6,position_index:int=Query(0,ge=0,le=3),unit:int=Query(0,ge=0,le=9727),
           input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    row('fresh',sample)
    if block not in read(BASE/'native_factors/result.json',{}).get('blocks',[]):raise HTTPException(422,'Unknown captured block')
    z=arrays(BASE/'native_factors/fields'/f'{sample}.npz')
    return scalar_values(z,block,position_index,unit,input_coordinate,output_coordinate)


def scalar_values(z,block,position_index,unit,input_coordinate,output_coordinate):
    x=decode(z[f'L{block}_mlp_input'][position_index]).astype(float);a=decode(z[f'L{block}_activation'][position_index]).astype(float)
    reader=parameter_reader();weights=[]
    for name in ('gate_proj','up_proj','down_proj'):
        w=reader.parameter(ROOT,f'model.layers.{block}.mlp.{name}.weight');weights.append(reader.decode(w[output_coordinate if name=='down_proj' else unit]).astype(float))
    gate,up,down=weights;terms=np.stack([x*gate,x*up]);units=a*down
    return {'chain':{'input_coordinate':input_coordinate,'MLP_unit':unit,'output_coordinate':output_coordinate,'normalized_input':float(x[input_coordinate]),
        'gate_scalar':float(gate[input_coordinate]),'up_scalar':float(up[input_coordinate]),'down_scalar':float(down[unit]),
        'native_gate':float(decode(z[f'L{block}_gate'][position_index,unit])),'native_up':float(decode(z[f'L{block}_up'][position_index,unit])),
        'native_activation':float(a[unit]),'one_unit_output_term':float(units[unit]),'all_input_sums':terms.sum(1).tolist(),
        'all_unit_sum':float(units.sum()),'native_MLP_output_coordinate':float(decode(z[f'L{block}_mlp'][position_index,output_coordinate]))},
        'input_terms':field_response(terms,['All gate input-coordinate products','All up input-coordinate products'],0,2560,'Every real input coordinate; FP64 sum versus BF16 native value'),
        'unit_terms':field_response(units[None],['All native MLP unit contributions'],0,9728,'All9728 original activation × down-weight terms; not Top-K or causal necessity')}


@router.get('/generation')
def generation(sample:str='',branch:str='KL_history',start:int=Query(0,ge=0),count:int=Query(2560,ge=1,le=2560)):
    row('fresh',sample)
    if branch not in ('MSE_embedding','KL_embedding','KL_history'):raise HTTPException(422,'Unknown branch')
    path=BASE/'generation/commits'/f'{sample}.json'
    if not path.exists():raise HTTPException(409,'Outside64 preselected generation cases or not yet complete')
    record=read(path);z=arrays(BASE/'generation/fields'/f'{sample}.npz');p=z[branch+'_predicted_h36'];actual=decode(z[branch+'_native_h36_same_own_prefix'])
    v=np.concatenate([p,actual]);labels=[f'{kind} step {i}' for kind in ('predicted','native on SAME own prefix') for i in range(len(p))]
    return {'record':record,'field':field_response(v,labels,start,count,'All coordinates; first half predicted, second half SAME-prefix native diagnostic; no refresh after initialization')}


@router.get('/scale-samples')
def scale_samples(model:str='qwen14'):
    if model not in ('qwen4','qwen14','glm4'):raise HTTPException(422,'Unknown model')
    return [read(p) for p in sorted((BASE/'scale'/model/'rows').glob('*.json'))]


@router.get('/scale-field')
def scale_field(model:str='qwen14',sample:str='',part:str='late',start:int=Query(0,ge=0),count:int=Query(5120,ge=1,le=5120)):
    if model not in ('qwen4','qwen14','glm4') or part not in ('early','mid','late','postnorm','incoming_embedding','all_layer_energy'):raise HTTPException(422,'Unknown native model field')
    valid={p.stem for p in (BASE/'scale'/model/'rows').glob('*.json')}
    if sample not in valid:raise HTTPException(404,'Unknown scale sample')
    z=arrays(BASE/'scale'/model/'fields'/f'{sample}.npz');v=decode(z[part])
    labels=([f'own layer {i}' for i in range(len(v))] if part=='all_layer_energy' else [f'known incoming token at anchor {i}' for i in range(len(v))] if part=='incoming_embedding' else [f'own native token {p}' for p in z['positions']])
    return field_response(v,labels,start,count,'Columns are six position slots, values are full-coordinate energy' if part=='all_layer_energy' else 'Own model native width; identical coordinate numbers across models do NOT imply same function')


@router.get('/analysis-index')
def analysis_index():
    result=[]
    for area in ('layer_atlas','relation_atlas/profiles','native_factors','verification','extension','extension/temperature','extension/tail_confirmation','extension/native_regimes','scale/qwen4','scale/qwen14','scale/glm4'):
        for path in sorted((BASE/area).glob('*.npz')):
            if area=='extension/native_regimes' and path.name=='frozen_training_signatures.npz':continue # preserved superseded L6 amplitude; corrected version separately indexed
            for header in npz_headers(str(path),path.stat().st_mtime_ns):
                key=header['array'];shape=header['shape']
                if shape and np.dtype(header['dtype']).kind in 'fiu' and shape[-1] in (2560,4096,5120,9728,151936):
                    result.append({'id':area+'/'+path.name+':'+key,'file':str(path.relative_to(BASE)).replace('\\','/'),'array':key,'shape':shape})
    return result


@router.get('/analysis-field')
def analysis_field(id:str='',start:int=Query(0,ge=0),count:int=Query(9728,ge=1,le=151936),row_start:int=Query(0,ge=0),row_count:int=Query(37,ge=1,le=128)):
    item=next((r for r in analysis_index() if r['id']==id),None)
    if item is None:raise HTTPException(404,'Unknown registered summary array')
    a=arrays(BASE/item['file'])[item['array']];v=a.reshape(-1,a.shape[-1])
    if row_start>=len(v):raise HTTPException(422,'Summary row outside tensor')
    end=min(len(v),row_start+row_count)
    return field_response(v[row_start:end],[str(tuple(map(int,np.unravel_index(i,a.shape[:-1])))) if a.ndim>1 else item['array'] for i in range(row_start,end)],start,count,
        'Full native final-axis indices; explicit leading-axis row page, no silent truncation. Width151936 means VOCAB ID, not hidden coordinate. Native factor profiles are conditional MEANS even when legacy keys end in _sum; count is separate. Regime L6 amplitude was audited and corrected after capture; frozen originals remain archived.',
        tensor_shape=list(a.shape),total_rows=len(v),row_start=row_start,row_end=end)


@router.get('/extension-index')
def extension_index():
    result=[]
    for kind,area,commit in [('event_layers','extension/event_trace/fields','extension/event_trace/result.json'),('event_factors','extension/event_trace/factors','extension/event_trace/result.json'),
                      ('tail_fixture','extension/tail_confirmation/fields','extension/tail_confirmation/result.json'),('tail_event','extension/tail_confirmation/events','extension/tail_confirmation/result.json'),
                      ('train_factor_fixture','native_factors/fields','native_factors/result.json'),('generation_raw','generation/fields','generation/result.json'),
                      ('regime_layers','extension/native_regimes/fields','extension/native_regimes/result.json'),('regime_factors','extension/native_regimes/factors','extension/native_regimes/result.json')]:
        if not (BASE/commit).exists():continue
        for path in sorted((BASE/area).glob('*.npz')):
            if kind=='train_factor_fixture' and not path.stem.startswith('train-'):continue
            headers=npz_headers(str(path),path.stat().st_mtime_ns)
            with np.load(path,allow_pickle=False) as z:positions=z['positions'].tolist() if 'positions' in z.files else None
            for header in headers:
                key=header['array'];shape=header['shape']
                if key=='positions' or len(shape)<2:continue
                observed_positions=positions if positions is not None else list(range(shape[0]))
                if kind=='generation_raw':
                    initial_length=len(read(BASE/'generation/commits'/f'{path.stem}.json')['initial_prefix_ids'])
                    begin=0 if key=='initial_h12' else initial_length if key.endswith('appended_predicted_h12') else initial_length-1
                    observed_positions=list(range(begin,begin+shape[0]))
                result.append({'id':kind+'/'+path.stem+':'+key,'kind':kind,'sample_id':path.stem,'file':str(path.relative_to(BASE)).replace('\\','/'),
                    'array':key,'shape':shape,'positions':observed_positions})
    return result


@router.get('/extension-scalar')
def extension_scalar(id:str='',block:int=6,position_index:int=Query(0,ge=0),unit:int=Query(0,ge=0,le=9727),
                     input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    item=next((x for x in extension_index() if x['id']==id and x['kind']=='regime_factors'),None)
    if item is None:raise HTTPException(404,'Choose a registered natural regime factor archive')
    z=arrays(BASE/item['file'])
    if block not in (6,16,34) or f'L{block}_activation' not in z:raise HTTPException(422,'Block absent from original archive')
    if position_index>=len(z['positions']):raise HTTPException(422,'Position slot outside original archive')
    result=scalar_values(z,block,position_index,unit,input_coordinate,output_coordinate)
    result.update(sample_id=item['sample_id'],native_position=int(z['positions'][position_index]),block=block)
    return result


@router.get('/extension-field')
def extension_field(id:str='',row_start:int=Query(0,ge=0),row_count:int=Query(37,ge=1,le=128),start:int=Query(0,ge=0),count:int=Query(9728,ge=1,le=151936)):
    item=next((x for x in extension_index() if x['id']==id),None)
    if item is None:raise HTTPException(404,'Unknown registered extension raw field')
    a=arrays(BASE/item['file'])[item['array']];a=decode(a) if a.dtype==np.uint16 else a;v=a.reshape(-1,a.shape[-1])
    if row_start>=len(v):raise HTTPException(422,'Row outside original tensor')
    end=min(len(v),row_start+row_count)
    source=next((r for r in material('main')+material('fresh') if r['sample_id']==item['sample_id']),None)
    if source is None:source=next((r for r in compressed(BASE/'extension/tail_confirmation/material.json.gz') if r['sample_id']==item['sample_id']),None)
    return field_response(v[row_start:end],[str(tuple(map(int,np.unravel_index(i,a.shape[:-1])))) for i in range(row_start,end)],start,count,
        'Complete native last axis; uint16 arrays decode original BF16, float arrays preserve predicted/diagnostic values. Attention columns are source positions; other axes explicitly retained. Numerical events and fitted fields are NOT semantic necessity claims.',
        tensor_shape=list(a.shape),total_rows=len(v),row_start=row_start,row_end=end,positions=item['positions'],
        source={k:source[k] for k in ('sample_id','text','tokens','token_offsets','source_group','language')} if source else None)


@router.get('/download')
def download(scope:str='fresh',sample:str=''):
    row(scope,sample);path=BASE/scope/'fields'/f'{sample}.npz'
    if not path.exists():raise HTTPException(409,'Raw field not archived')
    return FileResponse(path,filename=f'joint_{sample}_full_native_fields.npz')


@router.get('/figure/{name}')
def figure(name:str):
    if name not in {f['path'] for f in read(BASE/'figures/index.json',{}).get('figures',[])}:raise HTTPException(404,'Unknown figure')
    return FileResponse(BASE/'figures'/name,media_type='image/png')
