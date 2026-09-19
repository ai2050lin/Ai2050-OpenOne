"""Read-only, bounded slices of real RDC artifacts. No endpoint loads a model or starts jobs."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tests/glm5/result/rdc_feature_campaign_20260909'
MECHANISM=ROOT/'tests/glm5/result/rdc_mechanism_campaign_20260909'
CONTINUITY=ROOT/'tests/glm5/result/rdc_continuity_campaign_20260909'
CONDITIONAL=ROOT/'tests/glm5/result/rdc_conditional_campaign_20260910'
NEW_RUNS=('i_factorial','k_long','m_order','o_generalization','aligned_qwen4','aligned_qwen14','aligned_glm4')
RUNS=('s0','s1','s2pilot','a_native','b_relations','c_generation','e_confirmation','g_generation','scale_qwen14','scale_glm4')+NEW_RUNS
router=APIRouter(prefix='/api/rdc',tags=['rdc-features'])
FIGURES={
 'i_interactions':('i_factorial','i_factorial/figures/all_coordinate_interactions.png','i_factorial/features/interactions.npz','i_factorial/figures/display_contract.json','八族全部坐标的支持×提问交互（观察对比）'),
 'i_field':('i_factorial','i_factorial/figures/raw_and_normalized_full_coordinate_field.png','i_factorial/features/all_token_moments.npz','i_factorial/figures/display_contract.json','全token原值与标准化视图（聚合方式见说明）'),
 'j_units':('i_factorial','j_predictive_gates/figures/all_unit_conditional_error_difference.png','j_predictive_gates/figures/plotted_full_values.npz','j_predictive_gates/figures/display_contract.json','全部9728单元的条件重建误差差异'),
 'k_errors':('k_long','k_long/figures/full_coordinate_and_vocabulary_error.png','k_long/figures/plotted_full_values.npz','k_long/figures/display_contract.json','全坐标预测误差与全词表KL'),
 'l_readers':('aligned_','l_aligned/figures/prefill_content_all_layer_readers.png','l_aligned/result.json','l_aligned/figures/display_contract.json','三模型全层首内容对齐读取结果'),
 'm_sources':('m_order','m_order/figures/all_coordinate_source_group_vectors.png','m_order/figures/plotted_full_values.npz','m_order/figures/display_contract.json','三类顺序对照的全坐标来源向量'),
 'n_attention':('m_order','n_cached_attention/figures/full_coordinate_attention_forecast_errors.png','n_cached_attention/figures/plotted_values.npz','n_cached_attention/figures/display_contract.json','N/O全部坐标注意力预测与新材料推广'),
 'o_attention':('o_generalization','n_cached_attention/figures/full_coordinate_attention_forecast_errors.png','n_cached_attention/figures/plotted_values.npz','n_cached_attention/figures/display_contract.json','冻结提取器在新材料上的全部坐标误差')}

FIGURES['p_attention']=('o_generalization','p_token_conditioned/figures/all_coordinate_token_conditioned_errors.png','p_token_conditioned/figures/plotted_values.npz','p_token_conditioned/figures/display_contract.json','P · 探索性重拟合：全坐标误差（不是独立确认）')

def figure_list(run):
    return [{'id':key,'title':v[4]} for key,v in FIGURES.items() if (run==v[0] or v[0]=='aligned_' and run.startswith(v[0])) and (CONDITIONAL/v[1]).exists()]

@router.get('/figures/{figure}')
def figure_asset(figure:str,asset:str='image'):
    if figure not in FIGURES or asset not in ('image','values','contract'):raise HTTPException(404,'Unknown scientific figure asset')
    path=CONDITIONAL/FIGURES[figure][{'image':1,'values':2,'contract':3}[asset]]
    if not path.exists():raise HTTPException(409,'Figure asset not completed yet')
    return FileResponse(path,media_type='image/png' if asset=='image' else 'application/json' if path.suffix=='.json' else 'application/octet-stream',filename=None if asset=='image' else path.name)

def folder(run):
    if run not in RUNS:raise HTTPException(404,'Unknown run')
    if run.startswith('aligned_'):return CONDITIONAL/'l_aligned'/run.removeprefix('aligned_')
    if run in ('i_factorial','k_long','m_order','o_generalization'):return CONDITIONAL/run
    if run.startswith('scale_'):return CONTINUITY/'h_scale'/run.replace('scale_','')
    return (BASE if run in RUNS[:3] else CONTINUITY if run in ('e_confirmation','g_generation') else MECHANISM)/run

def source_folder(run,row):
    return folder(row['origin_run']) if run=='a_native' else folder(run)

def read(path,default=None):
    if not path.exists():return default
    return json.loads(path.read_text(encoding='utf-8'))

@lru_cache(maxsize=3)
def cached_material(path,mtime):
    return read(Path(path),[])

def material(run):
    path=folder(run)/('material_scored.json' if run=='m_order' and (folder(run)/'material_scored.json').exists() else 'material.json')
    return cached_material(str(path),path.stat().st_mtime_ns) if path.exists() else []

def row_for(run,sample):
    rows=material(run)
    found=next((r for r in rows if r['sample_id']==sample),None)
    if found is None:raise HTTPException(404,'Unknown stable sample ID')
    return found

def original_input(run,row):
    if run not in ('k_long','m_order','o_generalization'):return None
    path=folder(run)/'prefixes.json'
    if not path.exists():return None
    prefixes=cached_material(str(path),path.stat().st_mtime_ns)
    found=next((r for r in prefixes if r['sample_id']==row.get('prefix_id')),None)
    return {k:found[k] for k in ('sample_id','system','user','prompt','source','record','expected_fields','external_constraints','target') if k in found} if found else None

@lru_cache(maxsize=2)
def load_arrays(path,mtime):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def arrays(path):
    if not path.exists():raise HTTPException(409,'Sample has not committed yet')
    return load_arrays(str(path),path.stat().st_mtime_ns)

def decode(a):
    if a.dtype==np.uint16:return (a.astype(np.uint32)<<16).view(np.float32)
    return a.astype(np.float64)

@router.get('/runs')
def runs():
    return {'runs':[run_status(r) for r in RUNS], 'plan':read(CONDITIONAL/'plan.json',{}),'prior_plan':read(CONTINUITY/'plan.json',{}),
      'automatic_extensions':{'2707':read(CONDITIONAL/'automatic_extension_plan.json'),'2708':read(CONDITIONAL/'n_cached_attention/protocol.json'),'2709':read(CONDITIONAL/'o_generalization/protocol.json'),'2710':read(CONDITIONAL/'p_token_conditioned/protocol.json')},'continuation_decision':read(CONDITIONAL/'continuation_decision.json')}

@router.get('/runs/{run}/status')
def run_status(run:str):
    if run.startswith('aligned_'):
        folder(run)
        return dict(read(CONDITIONAL/('l_'+run)/'status.json',{'state':'not_started'}),run_id=run)
    if run.startswith('scale_'):
        folder(run)
        return dict(read(CONTINUITY/('h_'+run)/'status.json',{'state':'not_started'}),run_id=run)
    return read(folder(run)/'status.json',{'run_id':run,'state':'not_started'})

@router.get('/runs/{run}/material')
def run_material(run:str):
    rows=material(run)
    committed={r['sample_id'] for r in rows if run=='s0' or (source_folder(run,r)/f'commits/{r["sample_id"]}.json').exists()}
    return {'run_id':run,'samples':[dict(r,committed=r['sample_id'] in committed) for r in rows],
            'protocol':read(folder(run)/'protocol.json',{}),
            'scoring_alignment_update':read(folder(run)/'scoring_alignment_protocol_v2.json') if run=='m_order' else None}

@router.get('/runs/{run}/events')
def events(run:str,after:int=Query(0,ge=0)):
    path=folder(run)/'events.jsonl';out=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue # concurrent final incomplete line is not a committed event
            if r['cursor']>after:out.append(r)
    selected=out[:128]
    return {'events':selected,'cursor':selected[-1]['cursor'] if selected else after,'has_more':len(out)>128}

@router.get('/runs/{run}/results')
def results(run:str):
    raw=read(folder(run)/'result.json');extension=read(folder(run)/'extension_result.json')
    if run in NEW_RUNS and raw:
        result=dict(raw);entries=raw.get('results',raw.get('readers',raw.get('forecasts',[])))
        normalized=[]
        for r in entries:
            item=dict(r,algorithm=r.get('model',r.get('algorithm','shared_linear')),split=r.get('scope',r.get('stage','frozen_new_material' if run=='o_generalization' else 'entity_heldout')),
              target=r.get('target','t+y joint' if isinstance(r.get('accuracy'),list) else 'L23 attention output' if run=='o_generalization' else 'H36'),representation=r.get('model','H'+str(r.get('H',''))))
            if isinstance(item.get('accuracy'),list):item['accuracy']=float(np.mean(item['accuracy']))
            normalized.append(item)
        result['reader_and_forecast_details']=entries;result['results']=normalized
        result['display_note']='For t+y joint rows, MSE and accuracy summarize both external labels; separate support/answer counts remain in detailed evidence.' if run!='o_generalization' else 'O MSE measures all2560 native L23 attention-output coordinates under frozen N predictors; no complete-answer correctness score. Exploratory P results are a separate extension.'
        if run=='i_factorial':extension={'predictive_gates':read(CONDITIONAL/'j_predictive_gates/result.json'),
          'capacity_sensitivity':read(folder(run)/'capacity_audit.json'),'summary_audit':read(folder(run)/'summary_audit.json')}
        elif run.startswith('aligned_'):extension={'crossmodel_maps':read(CONDITIONAL/'l_aligned/result.json',{}).get('crossmodel_maps',[]),
          'pairing_controls':read(CONDITIONAL/'l_aligned/summary_audit.json',{}).get('pairing_null_comparisons',[]),
          'stage_calibration':read(CONDITIONAL/'l_aligned/stage_calibration_audit.json'),
          'native_arithmetic':read(CONDITIONAL/'l_aligned/native_factor_arithmetic_audit.json')}
        elif run=='m_order':extension={'scoring_alignment':read(folder(run)/'scoring_alignment_audit.json'),
          'source_accounting_summary':read(folder(run)/'source_result.json',{}).get('summary',[]),
          'cached_attention_prediction':read(CONDITIONAL/'n_cached_attention/result.json')}
        elif run=='o_generalization':extension={'exploratory_token_conditioned':read(CONDITIONAL/'p_token_conditioned/result.json')}
        return {'result':result,'extension':extension,'state':run_status(run),'figures':figure_list(run)}
    return {'result':raw,'extension':extension,'state':run_status(run),'figures':figure_list(run)}

@router.get('/runs/{run}/field')
def field(run:str,sample:str,field:str='h',layer:int=Query(0,ge=0),layers:int=Query(4,ge=1,le=8),
          token:int=Query(0,ge=0),tokens:int=Query(12,ge=1,le=32),coordinate:int=Query(0,ge=0),width:int=Query(128,ge=1,le=256)):
    row=row_for(run,sample);out=source_folder(run,row)
    if run=='s0':
        if field not in ('u','v','c'):field='u'
        arr=arrays(out/row['field'])[field][row['index']][None,None,:]
        layer_ids=[0];token_ids=[0];labels=['synthetic input'];original_shape=list(arr.shape)
        selected=arr[:,:,coordinate:coordinate+width]
    else:
        if not (out/f'commits/{sample}.json').exists():raise HTTPException(409,'No complete sample commit')
        a=arrays(out/f'fields/{sample}.npz')
        if run in NEW_RUNS:
            current=row.get('query_position',len(row['prompt_ids'])-1)
            if field=='h':
                arr=a['h_c'][:,None];layer_ids=list(range(layer,min(layer+layers,len(arr))))
                token_ids=[current];selected=arr[layer:layer+layers,:,coordinate:coordinate+width]
            elif field=='h_full':
                key='h' if run=='i_factorial' else 'h_prefill'
                if key not in a:raise HTTPException(422,'Full-token persistence only in the predeclared panel; choose h for complete current-query coordinates')
                arr=a[key];layer_ids=list(range(layer,min(layer+layers,len(arr))));token_ids=list(range(token,min(token+tokens,arr.shape[1])))
                selected=arr[layer:layer+layers,token:token+tokens,coordinate:coordinate+width]
            elif field in ('u_mean','v_mean'):
                if 'roles' not in a:raise HTTPException(422,'Role means are only captured in factorial I')
                role=0 if field=='u_mean' else 1;arr=a['roles'][:,role:role+1]
                layer_ids=list(range(layer,min(layer+layers,len(arr))));token_ids=[row['spans']['u' if role==0 else 'v']['positions'][0]]
                selected=arr[layer:layer+layers,:,coordinate:coordinate+width]
            elif field=='postnorm':
                key='postnorm_c' if run=='k_long' else 'postnorm'
                if key not in a:raise HTTPException(422,'Postnorm is not persisted for this run')
                arr=a[key][None,None];layer_ids=[len(a['h_c'])-1];token_ids=[current];selected=arr[:,:,coordinate:coordinate+width]
            elif field=='logits':
                if 'logits' not in a:raise HTTPException(422,'Full vocabulary logits are persisted only at declared K analysis / M boundary steps')
                arr=a['logits'][None,None];layer_ids=['vocabulary logits'];token_ids=[current];selected=arr[:,:,coordinate:coordinate+width]
            elif run in ('m_order','o_generalization') and field in ('q','k','v','p','attention_out','head_output'):
                key=f'L{layer}_{field}'
                if key not in a:raise HTTPException(422,'M native source fields: Result-boundary steps; O: declared L23 decode steps1/4/8')
                value=a[key];layer_ids=[layer]
                if field in ('k','v'):arr=value.transpose(1,0,2).reshape(1,value.shape[1],-1)
                elif field=='p':arr=value.T[None]
                else:arr=value.reshape(1,1,-1)
                if field in ('k','v','p'):
                    token_ids=list(range(token,min(token+tokens,arr.shape[1])));selected=arr[:,token:token+tokens,coordinate:coordinate+width]
                else:token_ids=[current];selected=arr[:,:,coordinate:coordinate+width]
            else:
                if field not in ('gate','up','a','down','mlp_x','attention_x'):raise HTTPException(422,'Only declared native fields are captured in this campaign')
                key=f'L{layer}_{field}';value=a.get(key)
                if value is None and field in ('gate','up') and f'L{layer}_gate_up' in a:
                    value=np.split(a[f'L{layer}_gate_up'],2)[0 if field=='gate' else 1]
                if value is None:raise HTTPException(422,'Native field not captured at this layer/step; I:11/23/35, K:L23 every step and11/35 analysis steps, aligned:model-specific checkpoints')
                arr=value.reshape(1,1,-1);layer_ids=[layer];token_ids=[current];selected=arr[:,:,coordinate:coordinate+width]
            original_shape=list(arr.shape)
        elif run.startswith('scale_'):
            if field=='h':
                arr=a['h'];layer_ids=list(range(layer,min(layer+layers,len(arr))));token_ids=list(range(token,min(token+tokens,arr.shape[1])))
                selected=arr[layer:layer+layers,token:token+tokens,coordinate:coordinate+width]
            elif field=='postnorm':
                arr=a['postnorm'][None];layer_ids=[len(a['h'])-1];token_ids=list(range(token,min(token+tokens,arr.shape[1])));selected=arr[:,token:token+tokens,coordinate:coordinate+width]
            else:
                if field not in ('gate','up','a','down'):raise HTTPException(422,'Scale capture: h/postnorm and nativegate/up/a/down')
                native_layers=sorted({int(k.split('_')[0][1:]) for k in a if k.startswith('L')})
                layer_ids=[l for l in native_layers if layer<=l<layer+layers]
                if not layer_ids:raise HTTPException(422,f'Native checkpoints: {native_layers}')
                raw=[]
                for l in layer_ids:
                    v=a.get(f'L{l}_{field}')
                    if v is None and field in ('gate','up'):
                        v=np.split(a[f'L{l}_gate_up'],2,axis=-1)[0 if field=='gate' else 1]
                    raw.append(v)
                arr=np.stack(raw);token_ids=[len(row['prompt_ids'])-1];selected=arr[:,:,coordinate:coordinate+width]
            original_shape=list(arr.shape)
        elif run in ('c_generation','g_generation'):
            if field not in ('h','postnorm','gate','up','a','down','mlp_x','attention_x','q','k','v','p','attention_out','head_output'):
                raise HTTPException(422,'Field not collected in generation run')
            if field=='h':
                arr=a['h'];layer_ids=list(range(layer,min(layer+layers,len(arr))))
                selected=arr[layer:layer+layers,:,coordinate:coordinate+width];token_ids=[row['query_position']]
            elif field=='postnorm':
                arr=a['postnorm'][None];layer_ids=[36];selected=arr[:,:,coordinate:coordinate+width];token_ids=[row['query_position']]
            else:
                if layer not in (11,23,35):raise HTTPException(422,'Generation native layers:11,23,35')
                raw=a[f'L{layer}_{field}'];layer_ids=[layer]
                if field in ('k','v'):
                    arr=raw.transpose(1,0,2).reshape(raw.shape[1],-1)[None]
                    token_ids=list(range(token,min(token+tokens,arr.shape[1])));selected=arr[:,token:token+tokens,coordinate:coordinate+width]
                elif field=='p':
                    arr=raw.T[None];token_ids=list(range(token,min(token+tokens,arr.shape[1])));selected=arr[:,token:token+tokens,coordinate:coordinate+width]
                else:
                    arr=raw.reshape(1,1,-1);selected=arr[:,:,coordinate:coordinate+width];token_ids=[row['query_position']]
            original_shape=list(arr.shape)
        elif field=='h':
            arr=a['h'];original_shape=list(arr.shape)
            layer_ids=list(range(layer,min(layer+layers,len(arr))));token_ids=list(range(token,min(token+tokens,arr.shape[1])))
            selected=arr[layer:layer+layers,token:token+tokens,coordinate:coordinate+width]
        elif field=='postnorm':
            arr=a[field][None];original_shape=list(arr.shape);layer_ids=[36]
            token_ids=list(range(token,min(token+tokens,arr.shape[1])))
            selected=arr[:,token:token+tokens,coordinate:coordinate+width]
        else:
            allowed=('q','k','v','qnorm','knorm','gate','up','a','down','attention_x','mlp_x')
            if field not in allowed:raise HTTPException(422,'Unknown field')
            layer_ids=[l for l in (0,11,23,35) if layer<=l<layer+layers]
            if not layer_ids:raise HTTPException(422,'Native fields exist only at layers0,11,23,35')
            raw=[a[f'L{l}_{field}'] for l in layer_ids]
            arr=np.stack([r.reshape(r.shape[0],-1) for r in raw]);original_shape=list(arr.shape)
            positions=a['native_positions'].tolist();indices=[i for i,t in enumerate(positions) if token<=t<token+tokens]
            token_ids=[positions[i] for i in indices]
            selected=arr[:,indices,coordinate:coordinate+width]
        labels=[row['tokens'][t] for t in token_ids]
    if not selected.size:raise HTTPException(422,'Requested slice is empty; check layer/token/coordinate range')
    values=decode(selected)
    if not np.isfinite(values).all():raise HTTPException(500,'Nonfinite values in committed source')
    behavior_id=row.get('prefix_id',sample) if run in NEW_RUNS else sample
    behavior_path=out/f'behavior/{behavior_id}.json'
    if run in ('k_long','m_order') and (out/f'behavior_scored/{row["prefix_id"]}.json').exists():behavior_path=out/f'behavior_scored/{row["prefix_id"]}.json'
    axes='X=native coordinate, Y=layer/checkpoint, Z=actual token position; layout is not learned semantic geometry'
    if field in ('u_mean','v_mean'):axes+='; role values are explicit span means placed at the first span position only for layout, not a native single-token state'
    if field=='logits':axes='X=every native vocabulary ID, Y=logits, Z=current query position; vocabulary indices are NOT hidden coordinates'
    if run in ('m_order','o_generalization') and field in ('k','v','p'):axes='Z=all actual historical source positions; X=KVhead×128 coordinates for K/V, or all32 query heads for P; P is attention probability, not causal importance'
    if run in NEW_RUNS and field!='h_full' and not (run in ('m_order','o_generalization') and field in ('k','v','p')):axes+='; current-query/declared-role coverage only, not the complete token history'
    return {'run_id':run,'sample_id':sample,'field':field,'layer_ids':layer_ids,'token_ids':token_ids,'token_labels':labels,
            'coordinate_start':coordinate,'coordinate_count':values.shape[-1],'original_shape':original_shape,
            'values':values.tolist(),'shown_values':values.size,'total_stored_values':int(np.prod(original_shape)),
            'dtype':'FP32 span means of native BF16' if field in ('u_mean','v_mean') else 'native_bfloat16' if run!='s0' else 'synthetic_float64','no_topk':True,
            'axes':axes,'behavior':read(behavior_path) if run!='s0' else None,'original_input':original_input(run,row),
            'source_mode':'recorded_model_sample' if run!='s0' else 'synthetic'}

def native_slice_response(row,field,names,values,start,token,axes,**extra):
    return dict(run_id='a_native',sample_id=row['sample_id'],field=field,layer_ids=names,token_ids=[token],
        token_labels=[row['tokens'][token]],coordinate_start=start,coordinate_count=values.shape[-1],
        original_shape=[len(names),1,extra.pop('full_width')],values=values[:,None].tolist(),shown_values=int(values.size),
        total_stored_values=extra.pop('total_values'),dtype='native BF16 values / FP64 explicit arithmetic',
        no_topk=True,axes=axes,source_mode='actual_parameter_composition_NOT_causal_necessity',**extra)

@router.get('/conditional/parameter_path')
def conditional_parameter_path(run:str,sample:str,layer:int=Query(23,ge=0),unit:int=Query(0,ge=0),
        input_coordinate:int=Query(0,ge=0),output_coordinate:int=Query(0,ge=0),
        coordinate:int=Query(0,ge=0),width:int=Query(128,ge=1,le=256)):
    if run not in NEW_RUNS:raise HTTPException(422,'Conditional campaign only')
    row=row_for(run,sample);a=arrays(folder(run)/f'fields/{sample}.npz')
    if f'L{layer}_mlp_x' not in a:raise HTTPException(422,'MLP input not captured at this layer/step')
    model_name={'aligned_qwen14':'Qwen3-14B','aligned_glm4':'glm4-9b-chat-hf'}.get(run,'qwen3-4b')
    model=ROOT/'models/hf'/model_name;config=read(model/'config.json');d=config['hidden_size'];j=config['intermediate_size']
    if unit>=j or max(input_coordinate,output_coordinate,coordinate)>=d:raise HTTPException(422,'Native coordinate/unit out of model range')
    index=read(model/'model.safetensors.index.json')['weight_map'];weights=[];keys=[];physical=[]
    from safetensors import safe_open
    for part in ('gate','up'):
        fused=model_name=='glm4-9b-chat-hf';key=f'model.layers.{layer}.mlp.{"gate_up" if fused else part}_proj.weight'
        pr=unit+(j if fused and part=='up' else 0);keys.append(key);physical.append(pr)
        with safe_open(str(model/index[key]),framework='pt',device='cpu') as f:weights.append(f.get_slice(key)[pr:pr+1,:].float().numpy()[0].astype(np.float64))
    downkey=f'model.layers.{layer}.mlp.down_proj.weight'
    with safe_open(str(model/index[downkey]),framework='pt',device='cpu') as f:wd=float(f.get_slice(downkey)[output_coordinate:output_coordinate+1,unit:unit+1].float().item())
    g,u=weights;x=decode(a[f'L{layer}_mlp_x'])
    observed=[]
    for part in ('gate','up','a'):
        z=a.get(f'L{layer}_{part}')
        if z is None and part in ('gate','up'):z=np.split(a[f'L{layer}_gate_up'],2)[0 if part=='gate' else 1]
        observed.append(float(decode(z)[unit]))
    v=np.stack([x,g,x*g,u,x*u])[:,coordinate:coordinate+width];token=row.get('query_position',len(row['prompt_ids'])-1)
    response=native_slice_response(row,'parameter_path',['native input','gate weight','input × gate','up weight','input × up'],v,coordinate,token,
      f'X=all {d} native input coordinates, Y=distinct quantities, not layers. Full dot includes every coordinate; GLM fused physical rows are explicit.',
      full_width=d,total_values=5*d,unit=unit,weight_keys=keys,physical_rows=physical,
      full_dots=[float(x@g),float(x@u)],observed_gate_up_a=observed,rounding_remainder=[observed[0]-float(x@g),observed[1]-float(x@u)],
      scalar_chain={'model':model_name,'input_coordinate':input_coordinate,'unit':unit,'output_coordinate':output_coordinate,
        'z_i':float(x[input_coordinate]),'Wgate_ji':float(g[input_coordinate]),'Wup_ji':float(u[input_coordinate]),'Wdown_kj':wd,
        'observed_a_j':observed[2],'native_single_unit_write_k':wd*observed[2],
        'gate_input_term_i':float(x[input_coordinate]*g[input_coordinate]),'up_input_term_i':float(x[input_coordinate]*u[input_coordinate]),
        'physical_gate_up_rows':physical,'meaning':'Observed native parameter composition, not a unique semantic unit or causal necessity.'})
    response['run_id']=run;return response

@router.get('/conditional/inspect')
def conditional_inspect(run:str,sample:str,view:str='gate',layer:int=Query(23,ge=0),coordinate:int=Query(0,ge=0),width:int=Query(128,ge=1,le=256)):
    if run not in ('i_factorial','k_long','m_order','o_generalization'):raise HTTPException(422,'Conditional comparison is available in I/K/M/O')
    row=row_for(run,sample);a=arrays(folder(run)/f'fields/{sample}.npz');token=row.get('query_position',len(row['prompt_ids'])-1)
    extra={};root=CONDITIONAL/'j_predictive_gates';total=2560
    if view=='gate':
        if f'L{layer}_gate' not in a:raise HTTPException(422,'Native gate unavailable at this layer/step')
        g,u,y=[decode(a[f'L{layer}_{k}']) for k in ('gate','up','a')];total=len(y)
        language=int(row['language']=='zh');values=[y];names=['actual a']
        for mode in ('ordinary_global','weighted_global','weighted_language'):
            c=arrays(root/f'unit_errors/L{layer}_{mode}.npz')['coefficient'];values.append(g*u*c[language if mode.endswith('_language') else 0]);names.append(mode)
        if run=='i_factorial':
            group=row['family_index']*2+language
            c=arrays(root/f'unit_errors/L{layer}_weighted_family_language.npz')['coefficient'];values.append(g*u*c[group]);names.append('weighted family-language')
        values=np.stack(values);axes='All native MLP units; approximations use observed same-block g/up, NOT earlier-state forecasts. K reuses frozen I coefficients prospectively.'
        source='observed_same_block_reconstruction_NOT_prediction'
    elif view in ('forecast_a','forecast_down') and run=='i_factorial':
        rows=material(run);i=next(j for j,r in enumerate(rows) if r['sample_id']==sample)
        z=arrays(root/'predictions/H12_C_quadratic_g_up.npz')
        if i not in z['test']:raise HTTPException(422,'Choose held-out factorial entity12..15')
        ix=z['test'].tolist().index(i)
        if view=='forecast_a':
            y=decode(a['L23_a']);total=len(y);g,u=np.split(z['prediction'][ix],2)
            derived=(g/(1+np.exp(-np.clip(g,-80,80))))*u
            direct=arrays(root/'predictions/H12_C_quadratic_a.npz')['prediction'][ix]
            equal=arrays(root/'predictions/H12_C_quadratic_a_up.npz')['prediction'][ix,:total]
            values=np.stack([y,derived,direct,equal]);names=['actual a','forecast g/up → a','direct a','equal 2J a+up head']
        else:
            y=decode(a['L23_down']);values=np.stack([y]+[arrays(root/f'predictions/H12_C_quadratic_{name}.npz')['prediction'][ix] for name in ('factor_write','direct_a_write','down')]);names=['actual down','forecast factor write','direct a → Wdown','direct down']
        axes='All native coordinates; predictions use only H12 C and fixed weights. Direct a→Wdown and direct down commute for identical kernel/ridge; not independent mechanistic confirmations.'
        extra['full_coordinate_mse']={name:float(np.mean((p-y)**2)) for name,p in zip(names[1:],values[1:])};source='held_out_earlier_state_prediction'
    elif view=='forecast_attention' and run in ('m_order','o_generalization'):
        root=CONDITIONAL/('n_cached_attention' if run=='m_order' else 'o_generalization')
        selected=read(root/('selected_rows.json' if run=='m_order' else 'features/selected_rows.json'),[])
        index=next((i for i,r in enumerate(selected) if r['sample_id']==sample),None)
        if index is None:raise HTTPException(422,'Choose a completed held-out N Result onset or an O declared analysis state')
        mids=['H12_linear_factors_validation','H12_fullpastKV_linear_factors_validation','H12_fullpastKV_linear_direct_head_equal6144_validation']
        first=arrays(root/f'predictions/{mids[0]}.npz')
        if run=='m_order':
            if index not in first['test']:raise HTTPException(422,'N forecast display uses only held-out entities6/7')
            ix=first['test'].tolist().index(index)
        else:ix=index
        actual=decode(a['L23_attention_out']);values=np.stack([actual]+[arrays(root/f'predictions/{mid}.npz')['prediction'][ix] for mid in mids]);names=['actual L23 attention']+mids
        axes='All2560 native attention-output coordinates. Factor composer always uses already-available pastL23 KV; H12 is the encoder input name, not the whole prediction state. CurrentL23 Q/K/V and nexttoken are targets, not inputs. O uses frozen N predictors without O calibration.'
        extra['full_coordinate_mse']={name:float(np.mean((v-actual)**2)) for name,v in zip(names[1:],values[1:])};source='earlier_query_and_available_past_KV_attention_prediction'
    elif view=='forecast_token_conditioned' and run=='o_generalization':
        root=CONDITIONAL/'p_token_conditioned';selected=read(root/'selected_rows.json',[])
        index=next((i for i,r in enumerate(selected) if r['sample_id']==sample),None)
        mids=['full_quadratic_factors_validation','full_quadratic_hidden23_validation','full_quadratic_direct_head_validation','token_add_factors_validation','token_product_factors_validation']
        first=arrays(root/f'predictions/{mids[0]}.npz')
        if index is None or index not in first['test']:raise HTTPException(422,'P exploratory view requires O entity12..15 at declared analysis steps')
        ix=first['test'].tolist().index(index);actual=decode(a['L23_attention_out'])
        values=np.stack([actual]+[arrays(root/f'predictions/{mid}.npz')['prediction'][ix] for mid in mids]);names=['actual L23 attention']+mids
        axes='EXPLORATORY repartition after inspecting O results, NOT independent confirmation. All2560 coordinates. Inputs: current H0/H12 and available pastKV. Predicted future H23 uses its OWN RMS with real native Q/K/V weights; actual future H23 is target only. O frozen-transfer results remain separate.'
        extra['full_coordinate_mse']={name:float(np.mean((v-actual)**2)) for name,v in zip(names[1:],values[1:])};source='exploratory_token_conditioned_NOT_independent_confirmation'
    elif view=='forecast_h' and run=='k_long':
        rows=read(folder(run)/'features/selected_rows.json');i=next((j for j,r in enumerate(rows) if r['sample_id']==sample),None)
        z=arrays(folder(run)/'predictions/H12_quadratic.npz')
        if i is None or i not in z['test']:raise HTTPException(422,'Choose a declared analysis step in held-out long entity6/7')
        ix=z['test'].tolist().index(i);y=z['target'][ix];p=z['prediction'][ix];q=arrays(folder(run)/'predictions/H12_previousH36_quadratic.npz')['prediction'][ix]
        values=np.stack([y,p,q,q-y]);names=['actual H36','H12 prediction','H12 + past H36 prediction','history prediction minus actual']
        axes='All H36 coordinates. Previous-step H36 is past available state, zero at prefill. Not a complete historical KV state; no future tokens used.'
        extra['full_coordinate_mse']={'H12':float(np.mean((p-y)**2)),'H12_previousH36':float(np.mean((q-y)**2))};source='held_out_earlier_state_prediction'
    else:raise HTTPException(422,'Unknown conditional view for this run')
    values=values[:,coordinate:coordinate+width]
    if not values.size:raise HTTPException(422,'Empty coordinate range')
    response=native_slice_response(row,'conditional_'+view,names,values,coordinate,token,axes,full_width=total,total_values=len(names)*total,**extra)
    response.update(run_id=run,source_mode=source,original_input=original_input(run,row));return response

@router.get('/conditional/source_groups')
def conditional_source_groups(sample:str,layer:int=Query(23,ge=0),coordinate:int=Query(0,ge=0),width:int=Query(128,ge=1,le=256)):
    row=row_for('m_order',sample);a=arrays(folder('m_order')/f'ledgers/{sample}.npz');key=f'L{layer}_source_vectors'
    if key not in a:raise HTTPException(422,'M source ledgers are available at L11/23/35 of selected Result-boundary states')
    names=['record','prompt_other','generated_trace','generated_neutral','generated_result','generated_other','head matmul rounding','O projection rounding','actual attention output']
    values=np.concatenate([a[key],a[f'L{layer}_matmul_round_vector'][None],a[f'L{layer}_o_round_vector'][None],a[f'L{layer}_actual_attention'][None]],0)[:,coordinate:coordinate+width]
    if not values.size:raise HTTPException(422,'Empty native coordinate range')
    response=native_slice_response(row,'conditional_sources',names,values,coordinate,row['query_position'],
      'X=all2560 native attention-output coordinates; source groups use every recorded source/head. Six source vectors + two rounding vectors = actual attention output. Observational parameter identity, NOT causal mediation or unique semantic modules.',
      full_width=2560,total_values=9*2560,account=read(folder('m_order')/f'accounts/{sample}.json'))
    response.update(run_id='m_order',source_mode='native_all_source_group_vector_accounting');return response

@router.get('/mechanism/mlp_units')
def mlp_units(sample:str,layer:int=Query(11),token:int=Query(0,ge=0),block:int=Query(0,ge=0,le=2),class_index:int=Query(0,ge=0,le=7),
              coordinate:int=Query(0,ge=0,le=9727),width:int=Query(128,ge=1,le=256)):
    if layer not in (11,23,35):raise HTTPException(422,'Layers11/23/35')
    row=row_for('a_native',sample);a=arrays(source_folder('a_native',row)/f'fields/{sample}.npz')
    positions=a['native_positions'].tolist()
    if token not in positions:raise HTTPException(422,'Choose a captured U/V/last position')
    x=decode(a[f'L{layer}_a'][positions.index(token)])
    beta=arrays(folder('a_native')/f'ledgers/L{layer}_native_coefficients.npz')['beta'][block,:,class_index]
    v=np.stack([x,beta,x*beta])[:,coordinate:coordinate+width]
    return native_slice_response(row,'mlp_units',['native activation','Wdown × reader','unit contribution'],v,coordinate,token,
        'X=MLP unit j (all9728 addressable); Y=a,beta,a*beta with different units; beta includes a fitted reader',
        full_width=9728,total_values=3*9728,full_sum=float(x@beta),block=block,class_index=class_index)

@router.get('/mechanism/parameter_path')
def parameter_path(sample:str,run:str='a_native',layer:int=Query(11),token:int=Query(0,ge=0),unit:int=Query(0,ge=0,le=9727),
                   output_coordinate:int=Query(0,ge=0,le=2559),input_coordinate:int=Query(0,ge=0,le=2559),
                   coordinate:int=Query(0,ge=0,le=2559),width:int=Query(128,ge=1,le=256)):
    if layer not in (11,23,35):raise HTTPException(422,'Layers11/23/35')
    if run not in ('a_native','e_confirmation'):raise HTTPException(422,'Parameter input paths available in A/E')
    row=row_for(run,sample);a=arrays(source_folder(run,row)/f'fields/{sample}.npz');positions=a['native_positions'].tolist()
    if token not in positions:raise HTTPException(422,'Choose captured U/V/last position')
    ix=positions.index(token);x=decode(a[f'L{layer}_mlp_x'][ix]);weights=[];keys=[]
    from safetensors import safe_open
    model=ROOT/'models/hf/qwen3-4b';index=read(model/'model.safetensors.index.json')['weight_map']
    for part in ('gate','up'):
        key=f'model.layers.{layer}.mlp.{part}_proj.weight';keys.append(key)
        with safe_open(str(model/index[key]),framework='pt',device='cpu') as f:weights.append(f.get_slice(key)[unit:unit+1,:].float().numpy()[0].astype(np.float64))
    downkey=f'model.layers.{layer}.mlp.down_proj.weight'
    with safe_open(str(model/index[downkey]),framework='pt',device='cpu') as f:down=float(f.get_slice(downkey)[output_coordinate:output_coordinate+1,unit:unit+1].float().item())
    g,u=weights;observed=[float(decode(a[f'L{layer}_{k}'][ix])[unit]) for k in ('gate','up','a')]
    v=np.stack([x,g,x*g,u,x*u])[:,coordinate:coordinate+width]
    response=native_slice_response(row,'parameter_path',['native input','gate weight','input × gate','up weight','input × up'],v,coordinate,token,
        'X=input residual coordinate i; rows are distinct quantities. Real scalar Wgate[j,i],Wup[j,i]; full dots include all2560.',
        full_width=2560,total_values=5*2560,unit=unit,weight_keys=keys,full_dots=[float(x@g),float(x@u)],observed_gate_up_a=observed,
        rounding_remainder=[observed[0]-float(x@g),observed[1]-float(x@u)],
        scalar_chain={'input_coordinate':input_coordinate,'unit':unit,'output_coordinate':output_coordinate,
            'z_i':float(x[input_coordinate]),'Wgate_ji':float(g[input_coordinate]),'Wup_ji':float(u[input_coordinate]),'Wdown_kj':down,
            'observed_a_j':observed[2],'native_single_unit_write_k':down*observed[2],
            'gate_input_term_i':float(x[input_coordinate]*g[input_coordinate]),'up_input_term_i':float(x[input_coordinate]*u[input_coordinate]),
            'meaning':'Real scalar weight factors and observed activation; not a unique semantic unit or an intervention effect.'})
    response['run_id']=run;return response

@router.get('/continuity/inspect')
def continuity_inspect(sample:str,view:str='shape',layer:int=Query(24,ge=0,le=36),target:int=Query(0,ge=0,le=1),
                       coordinate:int=Query(0,ge=0),width:int=Query(128,ge=1,le=256)):
    row=row_for('e_confirmation',sample);a=arrays(folder('e_confirmation')/f'fields/{sample}.npz')
    token=len(row['prompt_ids'])-1;extra={};total=2560
    if view=='shape':
        x=decode(a['h'][layer,token]);natural=a['natural_roles'][layer,2]
        values=np.stack([natural,x,x-natural]);names=['natural C','matched C','matched minus natural']
        axes='X=all2560 native coordinates; Y=natural/matched/error at same actual token; not three layers.'
        extra['full_max_abs']=float(np.abs(x-natural).max())
    elif view=='ruler':
        z=arrays(CONTINUITY/'f_continuity/fixed_ruler.npz');x=decode(a['h'][layer,token]);w=z['weight'][:,target]
        values=np.stack([x,w,x*w]);names=['observed C','fixed H24 reader','coordinate contribution']
        axes='One fixed old H24 reader at every layer; external score, not native output. No per-layer refitting.'
        extra.update(full_sum=float(x@w+z['bias'][target]),bias=float(z['bias'][target]))
    elif view=='gate':
        if layer not in (11,23,35):raise HTTPException(422,'Native factors layers11/23/35')
        rows=material('e_confirmation');i=next(j for j,r in enumerate(rows) if r['sample_id']==sample)
        z=arrays(CONTINUITY/f'f_continuity/L{layer}_factors.npz');total=9728
        values=np.stack([z['observed_a'][i],z['global_a'][i],z['family_language_a'][i],z['observed_a'][i]-z['family_language_a'][i]])
        names=['actual a','global gate approximation','family-language gate approximation','actual minus approximation']
        axes='X=all9728 native MLP units; fixed training mean gate versus observed activation. Approximation uses same-block g/up; NOT an earlier-state forecast.'
    elif view=='forecast':
        z=arrays(CONTINUITY/'f_continuity/predictions/H24_C_C_A1_linear.npz');rows=material('e_confirmation');i=next(j for j,r in enumerate(rows) if r['sample_id']==sample)
        if i not in z['test']:raise HTTPException(422,'Choose a held-out unit12..15 sample for this forecast')
        j=z['test'].tolist().index(i);y=z['target'][j];p=z['prediction'][j];values=np.stack([y,p,p-y])
        names=['actual H24 C','from H12 C prediction','prediction minus actual'];extra['full_coordinate_mse']=float(np.mean((p-y)**2))
        axes='H12 C predicts every H24 C coordinate; future H24 used only for scoring. Training/validation/test grouped by entity.'
    else:raise HTTPException(422,'Unknown continuity view')
    values=values[:,coordinate:coordinate+width]
    if not values.size:raise HTTPException(422,'Empty coordinate range')
    response=native_slice_response(row,'continuity_'+view,names,values,coordinate,token,axes,full_width=total,total_values=len(names)*total,**extra)
    response['run_id']='e_confirmation'
    response['source_mode']={'shape':'observed_numerical_protocol_comparison',
        'ruler':'external_fixed_reader_NOT_native_unembedding',
        'gate':'same_block_observed_inputs_conditional_approximation_NOT_forecast',
        'forecast':'held_out_earlier_state_prediction'}[view]
    return response

@router.get('/mechanism/output_ledger')
def output_ledger(sample:str,run:str='c_generation',coordinate:int=Query(0,ge=0,le=2559),width:int=Query(128,ge=1,le=256)):
    if run not in ('c_generation','g_generation'):raise HTTPException(422,'Native output ledgers in C/G')
    row=row_for(run,sample);out=folder(run);a=arrays(out/f'ledgers/{sample}.npz')
    values=np.stack([a['postnorm'],a['delta_unembedding'],a['logit_coordinate_contribution']])[:,coordinate:coordinate+width]
    response=native_slice_response(row,'output_ledger',['actual postnorm','native target weight contrast','coordinate logit contribution'],values,coordinate,row['query_position'],
        'X=all2560 postnorm coordinates; Y=h,deltaW,h*deltaW. G contrast changes by observed stage; exact token IDs in account. Not an external classifier.',
        full_width=2560,total_values=3*2560,account=read(out/f'accounts/{sample}.json'),full_sum=float(a['logit_coordinate_contribution'].sum()))
    response['run_id']=run;return response

@router.get('/mechanism/output_units')
def output_units(sample:str,run:str='c_generation',coordinate:int=Query(0,ge=0,le=9727),width:int=Query(128,ge=1,le=256)):
    if run not in ('c_generation','g_generation'):raise HTTPException(422,'Native output ledgers in C/G')
    row=row_for(run,sample);a=arrays(folder(run)/f'ledgers/{sample}.npz')
    values=np.stack([a['native_a'],a['beta'],a['unit_contribution']])[:,coordinate:coordinate+width]
    response=native_slice_response(row,'output_units',['actual a_j','Wdown × conditional output direction','unit contribution'],values,coordinate,row['query_position'],
        'X=all9728 final-layer MLP units; Y=a,beta,a*beta. Output direction uses observed final normalizer: accounting, not prediction.',
        full_width=9728,total_values=3*9728,full_sum=float(a['unit_contribution'].sum()))
    response['run_id']=run;return response

@router.get('/mechanism/source_ledger')
def source_ledger(sample:str,run:str='c_generation',token:int=Query(0,ge=0),tokens:int=Query(16,ge=1,le=32),coordinate:int=Query(0,ge=0,le=31),width:int=Query(32,ge=1,le=32)):
    if run not in ('c_generation','g_generation'):raise HTTPException(422,'Native output ledgers in C/G')
    row=row_for(run,sample);a=arrays(folder(run)/f'ledgers/{sample}.npz')
    arr=np.stack([a['attention_p'].T,a['attention_source_contribution'].T]);selected=arr[:,token:token+tokens,coordinate:coordinate+width]
    if not selected.size:raise HTTPException(422,'Empty source/head slice')
    tids=list(range(token,min(token+tokens,arr.shape[1])))
    return dict(run_id=run,sample_id=sample,field='source_ledger',layer_ids=['native P','source output contribution'],
        token_ids=tids,token_labels=[row['tokens'][t] for t in tids],coordinate_start=coordinate,coordinate_count=selected.shape[-1],
        original_shape=list(arr.shape),values=selected.tolist(),shown_values=int(selected.size),total_stored_values=int(arr.size),
        dtype='BF16 observed P / FP64 conditional contribution',no_topk=True,
        axes='X=head0..31; Y=P and P*(V dot Wo-output-direction), distinct quantities, NOT model layers; Z=actual historical source token. Final layer35, all sources available.',
        full_sum=float(a['attention_source_contribution'].sum()),source_mode='actual_native_sources_conditional_output_account_NOT_causal_necessity')

@router.get('/prediction')
def prediction(sample:str,split:str='word',algorithm:str='A1_linear',coordinate:int=Query(0,ge=0),width:int=Query(128,ge=1,le=256)):
    if split not in ('word','joint') or algorithm not in ('A0_mean','A0_distance','A1_linear','A2_quadratic','A2_cubic','A3_ordered_pair','A4_conditional'):
        raise HTTPException(422,'Unknown frozen prediction')
    row=row_for('s1',sample);rows=material('s1');idx=next(i for i,r in enumerate(rows) if r['sample_id']==sample)
    model_id=f'{split}__future_H36__{algorithm}'
    a=arrays(folder('s1')/f'predictions/{model_id}.npz')
    matches=np.flatnonzero(a['test_indices']==idx)
    if not len(matches):raise HTTPException(409,'This sample is not in the held-out prediction set; select a test sample')
    i=int(matches[0]);target=a['target'][i];pred=a['prediction'][i]
    values=np.stack([target,pred,pred-target])[:,None,coordinate:coordinate+width]
    if not values.size:raise HTTPException(422,'Empty prediction coordinate slice')
    return {'run_id':'s1','sample_id':sample,'field':'pooled_H36_prediction','layer_ids':['H36 truth','H36 prediction','prediction minus truth'],
            'token_ids':[row['spans']['u']['positions'][0]],'token_labels':['mean over all A span tokens'],
            'coordinate_start':coordinate,'coordinate_count':values.shape[-1],'original_shape':[3,1,len(target)],
            'values':values.tolist(),'shown_values':values.size,'total_stored_values':int(3*len(target)),
            'dtype':'float32 prediction and native-BF16-derived span mean','no_topk':True,
            'axes':'X=native coordinate; Y=truth/prediction/error panels, NOT three model layers; Z=one pooled A span',
            'behavior':read(folder('s1')/f'behavior/{sample}.json'),'source_mode':'held_out_extractor_prediction',
            'model_id':model_id,'full_coordinate_mse':float(np.mean((pred-target)**2))}

@router.get('/parameter')
def parameter(component:str='q',model_name:str='qwen3-4b',layer:int=Query(0,ge=0,le=39),row:int=Query(0,ge=0),start:int=Query(0,ge=0),count:int=Query(16,ge=1,le=256)):
    import torch
    from safetensors import safe_open
    stems={'q':'self_attn.q_proj','k':'self_attn.k_proj','v':'self_attn.v_proj','gate':'mlp.gate_proj',
           'up':'mlp.up_proj','down':'mlp.down_proj'}
    if component not in stems:raise HTTPException(422,'Unknown native parameter component')
    if model_name not in ('qwen3-4b','Qwen3-14B','glm4-9b-chat-hf'):raise HTTPException(422,'Unknown local model')
    model=ROOT/'models/hf'/model_name;config=read(model/'config.json')
    if layer>=config['num_hidden_layers']:raise HTTPException(422,'Layer out of model range')
    physical_row=row
    if model_name=='glm4-9b-chat-hf' and component in ('gate','up'):
        if row>=config['intermediate_size']:raise HTTPException(422,'Native unit out of range')
        stems[component]='mlp.gate_up_proj';physical_row=row+(config['intermediate_size'] if component=='up' else 0)
    key=f'model.layers.{layer}.{stems[component]}.weight'
    index=read(model/'model.safetensors.index.json')['weight_map']
    with safe_open(str(model/index[key]),framework='pt',device='cpu') as f:
        view=f.get_slice(key);shape=view.get_shape()
        if physical_row>=shape[0] or start>=shape[1]:raise HTTPException(422,'Parameter coordinate out of range')
        native=view[physical_row:physical_row+1,start:min(start+count,shape[1])].clone()
    return {'model':model_name,'key':key,'row':row,'physical_row':physical_row,'start':start,'shape':shape,'values':native.float()[0].tolist(),
            'native_bits':native.view(torch.uint16)[0].tolist(),'dtype':str(native.dtype),'source':'actual local safetensors checkpoint; read only'}

@router.get('/linear_contribution')
def linear_contribution(sample:str,layer:int=Query(36,ge=0,le=36),class_index:int=Query(0,ge=0,le=7),
                        coordinate:int=Query(0,ge=0),width:int=Query(128,ge=1,le=256)):
    if layer not in (0,12,24,36):raise HTTPException(422,'Reader layers:0,12,24,36')
    row=row_for('s2pilot',sample);rows=material('s2pilot');i=next(i for i,r in enumerate(rows) if r['sample_id']==sample)
    f=arrays(folder('s2pilot')/'features/all_samples.npz')
    ledger=arrays(folder('s2pilot')/f'coordinate_ledgers/word__family__H{layer}__A1_linear.npz')
    x=np.concatenate([f[f'H{layer}_{b}'][i] for b in ('u','v','c')]).astype(np.float64)
    w=ledger['weights'][:,class_index];bias=float(ledger['bias'][class_index]);product=x*w
    values=np.stack([x,w,product])[:,None,coordinate:coordinate+width]
    if not values.size:raise HTTPException(422,'Empty coordinate slice')
    return {'sample_id':sample,'run_id':'s2pilot','field':'linear_reader_contribution','layer_ids':['raw input','reader weight','input times weight'],
        'token_ids':[row['spans']['u']['positions'][0]],'token_labels':['U/V/C concatenation'],
        'coordinate_start':coordinate,'coordinate_count':values.shape[-1],'original_shape':[3,1,len(x)],'values':values.tolist(),
        'shown_values':values.size,'total_stored_values':int(3*len(x)),'dtype':'float64 raw-unit extractor ledger','no_topk':True,
        'axes':'X=U[0:2560],V[2560:5120],C[5120:7680] native coordinates; Y=input/reader weight/product (different units); not model layers',
        'source_mode':'frozen_extractor_coordinate_contributions_NOT_causal_or_checkpoint_weights',
        'bias':bias,'full_sum':float(product.sum()+bias),'class_index':class_index,
        'behavior':read(folder('s2pilot')/f'behavior/{sample}.json')}

@router.get('/coefficient')
def coefficient(model_id:str,run:str='s1',j:int=Query(0,ge=0),start:int=Query(0,ge=0),count:int=Query(16,ge=1,le=128),target:int=Query(0,ge=0)):
    if run not in ('s1','b_relations'):raise HTTPException(422,'Quadratic models available in S1 and B')
    entries=read(folder('s1')/'model_index.json',[]) if run=='s1' else read(folder(run)/'result.json',{}).get('models',[])
    found=next((r for r in entries if r['model_id']==model_id and r['algorithm']=='A2_quadratic'),None)
    if not found:raise HTTPException(404,'Unknown quadratic extractor')
    a=arrays(folder(run)/found['path']);z=a['z_train'];alpha=a['alpha'];scale=a['raw_scale_vector']
    if j>=z.shape[1] or start>=z.shape[1] or target>=alpha.shape[1]:raise HTTPException(422,'Extractor coordinate out of range')
    w=alpha[:,target];stop=min(start+count,z.shape[1])
    values=(w*z[:,j])@z[:,start:stop]/(scale[j]*scale[start:stop])
    return {'model_id':model_id,'j':j,'start':start,'values':values.tolist(),'bias':float(w.sum()),
            'linear_j':float(2*w@z[:,j]/scale[j]),'dimensions':z.shape[1],
            'source':'fitted extractor coefficients, NOT LLM weights','pair_convention':'x^T M x sums ordered j,k; symmetric off-diagonal is counted twice'}
