"""Read-only natural-prefix atlas. No model loading, jobs, or arbitrary file paths."""
import copy
from functools import lru_cache
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from server.rdc_feature_service import ROOT, read, arrays, decode

BASE=ROOT/'tests/glm5/result/rdc_prefix_atlas_20260910'
RUNS=('qwen4','qwen4_confirmation','qwen14','glm4')
router=APIRouter(prefix='/api/rdc-prefix',tags=['rdc-prefix-atlas'])
MODELS={'qwen4':'qwen3-4b','qwen4_confirmation':'qwen3-4b','qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf'}


def folder(run):
    if run not in RUNS:raise HTTPException(404,'Unknown run')
    return BASE/run


def sample_row(run,sample):
    if not sample or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in sample):raise HTTPException(404,'Invalid sample ID')
    p=folder(run)/f'rows/{sample}.json'
    if not (folder(run)/f'commits/{sample}.json').exists():raise HTTPException(409,'Sample not committed')
    return read(p)


@lru_cache(maxsize=3)
def tokenizer(run):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(ROOT/'models/hf'/MODELS[run]/'tokenizer.json'))


def safe_graph(run,r,k):
    graph=copy.deepcopy(r['actual_anchor_graphs'][k]);p=r['positions'][k]
    text=tokenizer(run).decode(r['prompt_ids'][:p+1],skip_special_tokens=False)
    # The full audit proves cue descriptors unchanged in all repaired Qwen4 positions.
    # Later captures use decoded prefixes directly. Reuse audited exact graph overrides.
    for item in read(BASE/'causal_graph_overrides.json',[]):
        if item['run']==run and item['sample_id']==r['sample_id'] and item['array_index']==k:
            graph=item['causal_graph'];break
    graph['observed_prefix']=text
    graph['decoded_from_actual_ids']=True
    return graph


def metric_table():
    output=[]
    corrections=read(BASE/'causal_hash_control/result.json',{}).get('reports',[])
    for scope,sub in [('test','shared_rules'),('confirmation','confirmation')]:
        result=read(BASE/sub/'result.json',{})
        probs=read(BASE/sub/'full_vocabulary/result.json',{}).get('reports',[])
        probs=[p for p in probs if 'hash' not in p['model']]
        probs+=read(BASE/sub/'full_vocabulary_causal_controls/result.json',{}).get('reports',[])
        probmap={p['model']:p for p in probs}
        reports=[p for p in result.get('reports',[])+result.get('temporal_reports',[]) if 'hash' not in p['model']]
        reports += [p for p in corrections if p.get('evaluation')==scope]
        for r in reports:
            ls=r.get('layers',{})
            score=ls.get('H36',ls.get('h36',ls.get('next_h36',r.get('new_H36',{}))))
            # Corrected controls use the same layer labels as their saved reports.
            if not score:score=r.get('H36',r.get('metrics',r))
            p=probmap.get(r['model'],{})
            output.append({'scope':scope,'model':r['model'],'H36_mse':score.get('mse'),
                'KL':p.get('mean_KL'),'argmax_agreement':p.get('argmax_agreement'),
                'status':'software-corrected control; not fresh confirmation' if 'hash' in r['model'] else 'frozen primary' if scope=='confirmation' else 'held-out test'})
    return output


@router.get('/overview')
def overview():
    return {'runs':[{'run':r,'status':read(folder(r)/'status.json',{}),
      'runtime':{k:v for k,v in read(folder(r)/'runtime.json',{}).items() if k in ('depth','width','dtype','quantized','cache','shape')},
      'complete_panels':len(list((folder(r)/'full_panels').glob('*.npz')))} for r in RUNS],
      'plan':read(BASE/'plan.json'), 'review':read(BASE/'review.json'), 'metrics':metric_table(),
      'atlas':read(BASE/'atlas/result.json'),'scale':read(BASE/'scale_analysis/result.json'),
      'causality_audit':read(BASE/'causal_prefix_audit.json'),'continuation':read(BASE/'continuation_decision.json'),
      'figures':read(BASE/'figures/index.json',[]),
      'coverage':'All observed token/layer/native coordinates entered streaming moments. Individual states retained at six positions per sentence, plus 16 complete Qwen4 panels. No PCA or Top-K coordinate selection.',
      'limits':'Natural sentences, teacher-forced prefixes, incomplete cue graphs. No universal semantic graph, causal gear identification, autonomous generation closure or new theorem.'}


@router.get('/runs/{run}/samples')
def samples(run:str):
    out=[]
    for p in sorted((folder(run)/'commits').glob('*.json')):
        r=read(folder(run)/f'rows/{p.stem}.json')
        out.append({k:r[k] for k in ('sample_id','language','genre','split','text','positions')}|{
          'token_count':len(r['prompt_ids']),'full_panel':(folder(run)/f'full_panels/{p.stem}.npz').exists()})
    return out


@router.get('/runs/{run}/sample/{sample}')
def sample(run:str,sample:str,anchor:int=Query(0,ge=0,le=5)):
    r=sample_row(run,sample)
    return {k:r[k] for k in ('sample_id','language','genre','split','text','tokens','prompt_ids','positions','source_group','token_offsets')}|{
      'graph':safe_graph(run,r,anchor),'retrospective_UD':r.get('retrospective_ud',[]),
      'retrospective_warning':'Full-sentence annotation, not a causal prediction input.',
      'behavior':read(folder(run)/f'behavior/{sample}.json')}


@router.get('/runs/{run}/field/{sample}')
def field(run:str,sample:str,anchor:int=Query(0,ge=0,le=5),view:str='layers',layer:int=Query(12,ge=0,le=64),
          normalized:bool=False,start:int=Query(0,ge=0),count:int=Query(2560,ge=1,le=16384)):
    r=sample_row(run,sample);a=arrays(folder(run)/f'fields/{sample}.npz');h=decode(a['h']);depth,_,width=h.shape
    if view=='layers':v=h[:,anchor];labels=[f'H{i}' for i in range(depth)]
    elif view=='tokens':
        p=folder(run)/f'full_panels/{sample}.npz'
        if not p.exists():raise HTTPException(409,'Only 16 declared Qwen4 samples have all-token panels')
        if layer>=depth:raise HTTPException(422,'Layer out of range')
        v=decode(arrays(p)['h'][layer]);labels=[f'{i}: {t}' for i,t in enumerate(r['tokens'])]
    elif view=='mlp':
        if 'L23_a' not in a:raise HTTPException(409,'Native MLP not captured for this model')
        if normalized:raise HTTPException(422,'MLP normalization is not defined by residual-stream moments')
        v=decode(a['L23_a'][anchor])[None];width=v.shape[1];labels=['L23 native activation a (9728 units)']
    else:raise HTTPException(422,'Unknown view')
    if normalized:
        reference_run='qwen4' if run=='qwen4_confirmation' else run
        m=arrays(folder(reference_run)/'all_token_moments.npz');n=float(m['counts'][:2].sum())
        if n<=0:raise HTTPException(409,'No training moments in this run; normalization unavailable')
        mu=m['sums'][:2].sum(0)/n;std=np.sqrt(np.maximum(m['squares'][:2].sum(0)/n-mu*mu,1e-12))
        v=(v-mu)/std if view=='layers' else (v-mu[layer])/std[layer]
    if start>=width:raise HTTPException(422,'Coordinate outside native width')
    end=min(start+count,width);limit=float(np.max(np.abs(v)))
    return {'run':run,'sample':sample,'view':view,'position':r['positions'][anchor],
      'native_width':width,'start':start,'end':end,'labels':labels,'values':v[:,start:end].tolist(),
      'whole_field_absmax':limit,'normalization':f'{reference_run} frozen training all-token coordinate z-score, each layer separately' if normalized else 'raw native BF16 values decoded without quantization change',
      'axes':'Rows are declared checkpoints / observed token positions; columns retain native coordinate indices. Pixel aggregation is display only; exact values are queryable.',
      'download':f'/api/rdc-prefix/download/{run}/{sample}?kind='+('panel' if view=='tokens' else 'field')}


@router.get('/prediction/{sample}')
def prediction(sample:str,run:str='qwen4',anchor:int=Query(0,ge=0,le=1),model:str='early_linear',start:int=Query(0,ge=0,le=2559),count:int=Query(2560,ge=1,le=2560)):
    if run not in ('qwen4','qwen4_confirmation'):raise HTTPException(409,'Use matched-scale summary for this model')
    sample_row(run,sample)
    if model not in ('early_linear','full_linear','full_quadratic','graph_interaction','train_mean','copy_H12','temporal_full_linear'):raise HTTPException(422,'Unknown frozen predictor')
    sub='shared_rules' if run=='qwen4' else 'confirmation';rr=read(BASE/'shared_rules'/run/'rows.json',[])
    idx=next((i for i,r in enumerate(rr) if r['sample_id']==sample and r['anchor']==anchor),None)
    a=arrays(BASE/sub/f'predictions/{model}.npz');ii=a['test'].tolist()
    if idx not in ii:raise HTTPException(409,'Prediction view requires a held-out test / confirmation sample')
    j=ii.index(idx);temporal=model.startswith('temporal_');pred=a['prediction'][j,-2560:]
    f=arrays(folder(run)/f'fields/{sample}.npz');actual=decode(f['h'][36,anchor*3+int(temporal)])
    end=min(start+count,2560)
    return {'values':np.stack([actual,pred,pred-actual])[:,start:end].tolist(),'labels':['Actual H36','Predicted H36','Prediction minus actual'],
      'native_width':2560,'start':start,'end':end,'whole_field_absmax':float(np.max(np.abs(np.stack([actual,pred,pred-actual])))),
      'normalization':'Raw, common color scale for all three rows','mse':float(np.mean((pred-actual)**2)),
      'available_inputs':'Previous H36, previous meanH12 and newly observed H0; target newH12 and later token unavailable.' if temporal else 'Current H12, prefix meanH12/current H0 and prefix graph as selected by model. No target H36 or actual target norm.',
      'status':'Frozen rule evaluated on official held-out source units' if run.endswith('confirmation') else 'Main held-out test'}


@router.get('/matrix')
def matrix(matrix_id:str='adjacent_H12',row:int=Query(0,ge=0,le=2559),column:int=Query(0,ge=0,le=2559),count:int=Query(64,ge=1,le=256)):
    entry=next((r for r in read(BASE/'atlas/matrix_index.json',[]) if r['id']==matrix_id),None)
    if entry is None:raise HTTPException(404,'Unknown matrix')
    a=arrays(BASE/'atlas'/f'matrices/{matrix_id}.npz')
    rs=slice(row,min(2560,row+count));cs=slice(column,min(2560,column+count))
    v=a['covariance'][rs,cs]/np.maximum(a['source_std'][rs,None]*a['target_std'][None,cs],1e-12)
    return {'id':matrix_id,'n':entry['n'],'kind':entry['kind'],'shape':[2560,2560],'row':row,'column':column,'values':v.tolist(),'warning':'Complete coordinate-pair statistic, not a causal edge or coordinate identity across tokens.'}


@router.get('/matrix-file/{matrix_id}')
def matrix_file(matrix_id:str):
    if matrix_id not in {r['id'] for r in read(BASE/'atlas/matrix_index.json',[])}:raise HTTPException(404,'Unknown matrix')
    p=BASE/'atlas'/f'matrices/{matrix_id}.npz'
    return FileResponse(p,filename=p.name)


@router.get('/native/{sample}')
def native(sample:str,run:str='qwen4',anchor:int=Query(0,ge=0,le=5),unit:int=Query(0,ge=0,le=9727),input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    if run not in ('qwen4','qwen4_confirmation'):raise HTTPException(409,'Native MLP path only captured for Qwen4')
    sample_row(run,sample);a=arrays(folder(run)/f'fields/{sample}.npz');x=decode(a['L23_mlp_x'][anchor]).astype(np.float64)
    from safetensors import safe_open
    model=ROOT/'models/hf/qwen3-4b';index=read(model/'model.safetensors.index.json')['weight_map'];w=[]
    for name in ('gate_proj','up_proj','down_proj'):
        key=f'model.layers.23.mlp.{name}.weight'
        with safe_open(model/index[key],framework='pt',device='cpu') as f:
            part=f.get_slice(key)[output_coordinate:output_coordinate+1,:] if name=='down_proj' else f.get_slice(key)[unit:unit+1,:]
            w.append(part.float().numpy().astype(np.float64)[0])
    g,u,down=w;av=decode(a['L23_a'][anchor]).astype(np.float64);dots=[float(g@x),float(u@x)]
    observed=[float(decode(a['L23_'+k][anchor])[unit]) for k in ('gate','up','a')]
    return {'native_layer':23,'unit':unit,'input_coordinate':input_coordinate,'output_coordinate':output_coordinate,
      'scalar_chain':{'x_i':float(x[input_coordinate]),'Wgate_ji':float(g[input_coordinate]),'Wup_ji':float(u[input_coordinate]),'observed_gate_up_a':observed,'Wdown_kj':float(down[unit]),'unit_write_contribution':float(down[unit]*av[unit])},
      'all_2560_input_dots':dots,'gate_up_rounding_residual':(np.array(dots)-observed[:2]).tolist(),
      'all_9728_unit_write_sum':float(av@down),'native_down_coordinate':float(decode(a['L23_down'][anchor])[output_coordinate]),
      'all_unit_contributions':(av*down).tolist(),'scope':'Real scalar weights and observed input/factors; FP64 accounting differs from native BF16 rounding. This is not early prediction, a single-concept neuron or an identified causal gear.'}


@router.get('/download/{run}/{sample}')
def download(run:str,sample:str,kind:str='field'):
    sample_row(run,sample)
    if kind not in ('field','panel','row'):raise HTTPException(422,'Unknown registered artifact')
    p=folder(run)/{'field':'fields','panel':'full_panels','row':'rows'}[kind]/(sample+('.json' if kind=='row' else '.npz'))
    if not p.exists():raise HTTPException(409,'Artifact not retained for this sample')
    return FileResponse(p,filename=p.name)


@router.get('/figures/{figure}')
def figure(figure:str,contract:bool=False):
    item=next((x for x in read(BASE/'figures/index.json',[]) if x['id']==figure),None)
    if item is None:raise HTTPException(404,'Unknown figure')
    return FileResponse(BASE/'figures'/(item.get('contract','display_contract.json') if contract else item['file']),media_type='application/json' if contract else 'image/png')


HISTORY=BASE/'full_source_history'


def history_rows(scope):
    if scope not in ('main','fresh'):raise HTTPException(422,'Unknown source-history scope')
    return read(BASE/'material_stratified.json',[]) if scope=='main' else read(HISTORY/'fresh_material.json',[])


def history_row(scope,sample):
    r=next((r for r in history_rows(scope) if r['sample_id']==sample),None)
    if r is None:raise HTTPException(404,'Unknown source-history sample')
    return r


def history_path(scope,r):
    return BASE/f'qwen4/full_panels/{r["sample_id"]}.npz' if scope=='main' and r['full_panel'] else HISTORY/scope/f'fields/{r["sample_id"]}.npz'


@router.get('/history/overview')
def history_overview():
    result=read(HISTORY/'result.json',{});fresh=read(HISTORY/'fresh_result.json',{});prob=read(HISTORY/'probability_result.json',{})
    compact=[]
    for scope,data in [('test',result),('fresh',fresh)]:
        for r in data.get('reports',[]):
            p=next((p for p in prob.get('reports',[]) if p['scope']==scope and p['rule']==r['rule']),{})
            compact.append({'scope':scope,'rule':r['rule'],'MSE':r['mse'],'KL':p.get('KL'),'argmax_agreement':p.get('argmax_agreement')})
    return {'protocol':read(HISTORY/'protocol.json'),'status':read(BASE/'full_source_history/status.json'),
      'selected_before_fresh':result.get('selected_before_fresh'),'metrics':compact,'fresh_comparisons':fresh.get('paired_comparisons'),
      'all_samples':{'main':512,'fresh':64},'frozen':read(HISTORY/'frozen.json',{}).get('timestamp'),
      'retrospective_relation_profiles':read(HISTORY/'relations/result.json'),
      'relation_uncertainty_audit':read(HISTORY/'relations/uncertainty_audit.json'),
      'numeric_template_sensitivity':read(HISTORY/'template_sensitivity.json'),
      'limits':'H36-only fitting target; not the same multi-layer objective as2712. Every source coordinate retained, but source alignment is a candidate similarity rule, not identified native attention.'}


@router.get('/history/samples')
def history_samples(scope:str='main'):
    return [{k:r[k] for k in ('sample_id','language','genre','split','text','anchors')}|{'tokens':len(r['prompt_ids'])}
      for r in history_rows(scope) if history_path(scope,r).exists()]


@router.get('/history/field')
def history_field(scope:str='main',sample:str='',start:int=Query(0,ge=0,le=2559),count:int=Query(2560,ge=1,le=2560),normalized:bool=False):
    r=history_row(scope,sample);a=arrays(history_path(scope,r));v=decode(a['h'][12] if scope=='main' and r['full_panel'] else a['h12'])
    if normalized:
        m=arrays(BASE/'qwen4/all_token_moments.npz');n=m['counts'][:2].sum();mu=m['sums'][:2,12].sum(0)/n
        std=np.sqrt(np.maximum(m['squares'][:2,12].sum(0)/n-mu*mu,1e-12));v=(v-mu)/std
    end=min(2560,start+count)
    return {'scope':scope,'sample':sample,'labels':[f'{i}: {t}' for i,t in enumerate(r['tokens'])],
      'values':v[:,start:end].tolist(),'native_width':2560,'start':start,'end':end,'whole_field_absmax':float(np.max(np.abs(v))),
      'normalization':'Frozen main training all-token H12 coordinate z-score' if normalized else 'Raw native H12 (all observed sentence positions)',
      'axes':'Retrospective full-sentence H12 panel; a predictor at anchor t uses ONLY source positions0..t, never later displayed positions.',
      'anchors':r['anchors'],'text':r['text'],'download':f'/api/rdc-prefix/history/download?scope={scope}&sample={sample}'}


@router.get('/history/prediction')
def history_prediction(scope:str='main',sample:str='',anchor:int=Query(0,ge=0,le=1),rule:str='relative_history',start:int=Query(0,ge=0,le=2559),count:int=Query(2560,ge=1,le=2560)):
    r=history_row(scope,sample)
    if rule not in ('current','mean_history','absolute_history','relative_history'):raise HTTPException(422,'Unknown frozen source rule')
    rr=read(HISTORY/('main_rows.json' if scope=='main' else 'fresh_rows.json'),[])
    idx=next((i for i,x in enumerate(rr) if x['sample_id']==sample and x['anchor']==anchor),None)
    a=arrays(HISTORY/f'predictions/{"test" if scope=="main" else "fresh"}_{rule}.npz')
    if scope=='main':
        if idx not in a['test'].tolist():raise HTTPException(409,'Use held-out main test sample for prediction')
        j=a['test'].tolist().index(idx)
        actual=decode(arrays(BASE/f'qwen4/fields/{sample}.npz')['h'][36,anchor*3])
    else:
        if idx is None:raise HTTPException(409,'Fresh prediction not ready')
        j=idx;actual=decode(arrays(HISTORY/f'fresh/fields/{sample}.npz')['h36'][anchor])
    pred=a['prediction'][j];v=np.stack([actual,pred,pred-actual]);end=min(start+count,2560)
    return {'values':v[:,start:end].tolist(),'labels':['Actual H36','Frozen predicted H36','Predicted minus actual'],
      'native_width':2560,'start':start,'end':end,'whole_field_absmax':float(np.max(np.abs(v))),'normalization':'Raw, common color scale',
      'mse':float(np.mean((pred-actual)**2)),'available_inputs':f'Only current/past H12 through position {r["anchors"][anchor]}, according to {rule}; no actual H36/norm.',
      'status':'Fresh64-source frozen confirmation' if scope=='fresh' else 'Main held-out test, H36-only selection objective',
      'prefix':tokenizer('qwen4').decode(r['prompt_ids'][:r['anchors'][anchor]+1],skip_special_tokens=False)}


@router.get('/history/download')
def history_download(scope:str='main',sample:str=''):
    r=history_row(scope,sample);p=history_path(scope,r)
    if not p.exists():raise HTTPException(409,'Source field not committed')
    return FileResponse(p,filename=p.name)


@router.get('/history/relations')
def history_relations(scope:str='main',start:int=Query(0,ge=0,le=2559),count:int=Query(2560,ge=1,le=2560)):
    if scope not in ('main','fresh'):raise HTTPException(422,'Unknown source scope')
    split='test' if scope=='main' else 'fresh';result=read(HISTORY/'relations/result.json',{});a=arrays(HISTORY/'relations/all_coordinate_profiles.npz')
    v=[];labels=[]
    for r in result.get('reports',[]):
        key=r['relation']+'_'+split+'_z_delta'
        if key not in a:continue
        v.append(a[key]);n=r['splits'][split];labels.append(f'{r["relation"]}: {n["pairs"]} pairs / {n["source_units"]} sources')
    if not v:raise HTTPException(409,'Relation profiles not ready')
    v=np.stack(v);end=min(2560,start+count)
    return {'values':v[:,start:end].tolist(),'labels':labels,'native_width':2560,'start':start,'end':end,'whole_field_absmax':float(np.max(np.abs(v))),
      'normalization':'Frozen main all-token H12 z-score products: relation pair minus same-sentence signed-distance-matched pair.',
      'axes':'Rows14 retrospective UD relations; columns all2560 paired native-coordinate products (DIAGONAL statistic, not full covariance). Fresh reuse is exploratory for this new statistic, not another independent confirmation.',
      'status':'Observation with distance control; lexical identity/POS confounding remains; no causal path or semantic gear claim.'}
