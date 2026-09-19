"""Read-only full-coordinate natural relation study. No jobs, model loads, or arbitrary paths."""
from collections import Counter,defaultdict
from functools import lru_cache
import numpy as np
from fastapi import APIRouter,HTTPException,Query
from fastapi.responses import FileResponse
from server.rdc_feature_service import ROOT,read,arrays,decode

BASE=ROOT/'tests/glm5/result/rdc_relation_dynamics_20260910'
router=APIRouter(prefix='/api/rdc-relation',tags=['rdc-relation-study'])
RELATIONS=('nsubj','obj','iobj','obl','nmod','amod','advmod','advcl','acl','conj','compound','case','mark','det')
CONTROLS=('distance','distance_pos','distance_pos_same_dependent_id','distance_pos_noninitial')


def material(scope):
    if scope not in ('main','fresh'):raise HTTPException(422,'Unknown scope')
    return read(BASE/('material.json' if scope=='main' else 'fresh_material.json'),[])


def row(scope,sample):
    r=next((r for r in material(scope) if r['sample_id']==sample),None)
    if r is None:raise HTTPException(404,'Unknown sample')
    if not (BASE/scope/f'commits/{sample}.json').exists():raise HTTPException(409,'Native field not committed')
    return r


def checked_range(start,count,width):
    if start>=width:raise HTTPException(422,'Coordinate outside native width')
    return min(start+count,width)


def field_response(v,labels,start,count,normalization,**extra):
    width=v.shape[-1];end=checked_range(start,count,width)
    return {'values':v[:,start:end].tolist(),'labels':labels,'native_width':width,'start':start,'end':end,'whole_field_absmax':float(np.max(np.abs(v))),
      'normalization':normalization,'axes':'Rows are stated observations, columns are un-reordered native indices. Zoom-out pixel aggregation is display only.',**extra}


@router.get('/overview')
def overview():
    choices=read(BASE/'frozen.json',{}).get('choices',{});review=read(BASE/'review.json',{})
    return {'plan':read(BASE/'plan.json'),'choices':choices,'review':review.get('corrections',[]),'material':read(BASE/'material_audit.json'),
      'capture':{s:read(BASE/s/'capture_result.json',{}) for s in ('main','fresh')},'relations':read(BASE/'relation_atlas/result.json',{}),'coverage':read(BASE/'relation_atlas/coverage.json',{}),
      'parser':read(BASE/'relation_atlas/parser_baseline_result.json',{}),'rules':read(BASE/'rules/result.json',{}),'dynamics':read(BASE/'dynamics/result.json',{}),
      'probability':read(BASE/'probability/result.json',{}).get('reports',[]),'confirmation_probability':read(BASE/'confirmation/result.json',{}).get('probability',[]),
      'native':read(BASE/'native/result.json',{}),'generation':read(BASE/'generation/result.json',{}),'scale':{m:read(BASE/f'scale/{m}/result.json',{}) for m in ('qwen4','qwen14','glm4')},
      'figures':read(BASE/'figures/index.json',[]),'output_geometry':read(BASE/'output_geometry/result.json',{}),
      'scalar_parameters':read(BASE/'scalar_parameters/result.json',{}),'source_normalization':read(BASE/'source_normalization/result.json',{}),'surrogate_stability':read(BASE/'surrogate_stability/result.json',{}),
      'native_source_audit':read(BASE/'native_source_audit/result.json',{}),'boundary_compilation':read(BASE/'boundary_compilation/result.json',{}),'boundary_relations':read(BASE/'boundary_relations/result.json',{}),
      'stable_trajectory':read(BASE/'surrogate_stability/trajectory_result.json',{}),'energy_audit':read(BASE/'energy_audit/result.json',{}),
      'constant_boundary_control':read(BASE/'boundary_compilation/constant_update_result.json',{}),'continuation':read(BASE/'continuation_decision.json',{}),
      'limits':'Selected full-coordinate fields and exact full-matrix recipes; observational relations and conditional predictors, not recovered semantic gears or autonomous language closure.'}


@router.get('/samples')
def samples(scope:str='fresh'):
    return [{k:r[k] for k in ('sample_id','language','genre','split','text','positions')}|{'tokens':len(r['prompt_ids']),'generated':(BASE/f'generation/commits/{r["sample_id"]}.json').exists() if scope=='fresh' else False} for r in material(scope)]


@lru_cache(maxsize=1)
def token_decoder():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(ROOT/'models/hf/qwen3-4b/tokenizer.json'))


def prefix_graph(r,position):
    z=arrays(BASE/'prefix_parser/model.npz');ids=r['prompt_ids'][:position+1];lang=int(r['language']=='zh');lookup={int(v):i for i,v in enumerate(z['ids'])};pos=np.stack([z['lexical_pos'][lookup[t],lang] if t in lookup else z['global_pos'][lang] for t in ids]).astype(float)
    labels=read(BASE/'prefix_parser/protocol.json')['labels'];probs=np.empty((len(ids)-1,len(labels)));bins=np.searchsorted(z['distance_cuts'],len(ids)-1-np.arange(len(ids)-1))
    for b in np.unique(bins):
        mask=bins==b;table=np.einsum('ijr,j->ir',z['relation_tables'][lang,b].astype(float),pos[-1]);probs[mask]=pos[:-1][mask]@table
    if len(probs):probs/=probs.sum(1,keepdims=True)
    return {'query_position':position,'observed_prefix':token_decoder().decode(ids,skip_special_tokens=False),'prefix_ids':ids,'labels':labels,
      'source_positions':list(range(position)),'all_relation_probabilities':probs.tolist(),'all_token_POS_probabilities':pos.tolist(),
      'status':'Only actual token prefix and language supplied; uncertain external candidate, no future gold relation inputs.'}


@router.get('/sample')
def sample(scope:str='fresh',sample:str='',position:int=Query(0,ge=0)):
    r=row(scope,sample)
    if position>=len(r['prompt_ids']):raise HTTPException(422,'Position outside prefix')
    return {**r,'prefix_graph':prefix_graph(r,position),'gold_warning':'Full-sentence retrospective UD annotations below are not prediction inputs.'}


@router.get('/field')
def field(scope:str='fresh',sample:str='',layer:str='h12',start:int=Query(0,ge=0),count:int=Query(2560,ge=1,le=9728),normalized:bool=False):
    r=row(scope,sample)
    if layer not in ('h12','h23','h24','h36','postnorm'):raise HTTPException(422,'Layer not retained')
    v=decode(arrays(BASE/scope/f'fields/{sample}.npz')[layer]);labels=[f'token {i}: {t}' for i,t in enumerate(r['tokens'])] if layer in ('h12','h23') else [f'token {p}' for p in r['positions']]
    norm='Raw native BF16 values; all coordinates'
    if normalized:
        if layer not in ('h12','h23'):raise HTTPException(422,'Training all-token normalization only defined for H12/H23')
        z=arrays(BASE/'relation_atlas/training_scales.npz');j=('h12','h23').index(layer);v=(v-z['mean'][j])/z['standard_deviation'][j];norm='Frozen main-training all-token coordinate z-score'
    return field_response(v,labels,start,count,norm,download=f'/api/rdc-relation/download?scope={scope}&sample={sample}',layer=layer)


@router.get('/prediction')
def prediction(scope:str='fresh',sample:str='',kind:str='current',anchor:int=Query(0,ge=0,le=1),start:int=Query(0,ge=0),count:int=Query(2560,ge=1,le=2560)):
    r=row(scope,sample);z=arrays(BASE/scope/f'fields/{sample}.npz');choices=read(BASE/'frozen.json')['choices']
    if kind not in ('current','temporal','native_all_sources','native_hybrid','boundary_affine'):raise HTTPException(422,'Unknown prediction')
    if kind=='boundary_affine':
        if scope!='fresh':raise HTTPException(409,'Boundary corrective comparison is on reused fresh sources only')
        path=BASE/f'boundary_compilation/fields/{sample}.npz'
        if not path.exists():raise HTTPException(409,'Boundary comparison not completed')
        p=decode(arrays(path)['h24'][anchor]);actual=decode(z['h24'][3*anchor]);name='H24';info='Exploratory: first source predicted by train-only coordinate affine rule; other sources frozen quadratic. No actual past H23 supplied.'
    elif kind.startswith('native'):
        if scope!='fresh':raise HTTPException(409,'Native compilation retained only for fresh source units')
        a=arrays(BASE/f'native/fields/{sample}.npz');p=decode(a['all_predicted_h24' if kind=='native_all_sources' else 'hybrid_h24'][anchor]);actual=decode(z['h24'][3*anchor]);name='H24';info='All source H23 predicted from H12' if kind=='native_all_sources' else 'EXTRA information: actual past H23; only current H23 predicted'
    else:
        route=choices[kind+'_MSE'];name='H36';actual=decode(z['h36'][3*anchor+int(kind=='temporal')])
        if scope=='fresh':
            rr=read(BASE/'confirmation/rows.json');j=next(i for i,m in enumerate(rr) if m['sample_id']==sample and m['anchor']==anchor);p=arrays(BASE/f'confirmation/{kind}_{route}.npz')['prediction'][j,-2560:]
        else:
            if r['split'] not in ('validation','test'):raise HTTPException(409,'Choose validation/test for frozen predictions')
            directory='rules' if kind=='current' else 'dynamics';rr=[m for m in read(BASE/f'{directory}/rows.json') if m['split']==r['split']];j=next(i for i,m in enumerate(rr) if m['sample_id']==sample and m['anchor']==anchor);p=arrays(BASE/f'{directory}/{route}/predictions.npz')[r['split']][j,-2560:]
        info='Actual current H12; frozen full quadratic' if kind=='current' else 'Actual previous H36 plus already known incoming token embedding; not self-fed'
    return field_response(np.stack([actual,p,p-actual]),[f'Actual {name}',f'Predicted {name}','Prediction minus actual'],start,count,'Raw, shared color scale',MSE=float(np.mean((actual-p)**2)),available_inputs=info)


@router.get('/matrix')
def matrix(relation:str='nmod',control:str='distance_pos',pair:str='H12_H23',split:str='test',view:str='train_z',row_start:int=Query(0,ge=0,le=2559),column_start:int=Query(0,ge=0,le=2559),count:int=Query(64,ge=1,le=256)):
    if relation not in RELATIONS or control not in CONTROLS or pair not in ('H12_H12','H12_H23','H23_H23') or split not in ('train','test') or view not in ('raw','train_z'):raise HTTPException(422,'Invalid matrix recipe')
    key=f'{relation}/{control}/{pair}/{view}';records=read(BASE/'relation_atlas/result.json')['entries']+read(BASE/'boundary_relations/result.json',{}).get('entries',[]);record=next((r for r in records if r['key']==key),None)
    entries=[]
    for e in read(BASE/'relation_atlas/pair_index.json'):
        if e['relation']!=relation or e['split']!=split:continue
        if control=='distance_pos_noninitial':
            pairs=[(i,j) for i,j in e['controls']['distance_pos'] if i>0 and j>0]
            if e['dependent']<=0 or e['head']<=0 or not pairs:continue
            entries.append({**e,'controls':{control:pairs}})
        elif e['controls'][control]:entries.append(e)
    if not entries:raise HTTPException(409,'No matched sources for this strict control; no fallback')
    counts=Counter(e['sample_id'] for e in entries);grouped=defaultdict(list)
    for e in entries:grouped[e['sample_id']].append(e)
    endrow=min(row_start+count,2560);endcol=min(column_start+count,2560);answer=np.zeros((endrow-row_start,endcol-column_start),float);scales=arrays(BASE/'relation_atlas/training_scales.npz');la,lb=pair.lower().split('_')
    for sid,ee in grouped.items():
        z=arrays(BASE/f'main/fields/{sid}.npz');a=decode(z[la])[:,row_start:endrow].astype(float);b=decode(z[lb])[:,column_start:endcol].astype(float)
        if view=='train_z':
            i,j=[('h12','h23').index(l) for l in (la,lb)];a=(a-scales['mean'][i,row_start:endrow])/scales['standard_deviation'][i,row_start:endrow];b=(b-scales['mean'][j,column_start:endcol])/scales['standard_deviation'][j,column_start:endcol]
        for e in ee:
            v=np.outer(a[e['dependent']],b[e['head']]);control_mean=sum(np.outer(a[i],b[j]) for i,j in e['controls'][control])/len(e['controls'][control]);answer+=(v-control_mean)/counts[sid]/len(counts)
    return {'key':key,'split':split,'row_start':row_start,'column_start':column_start,'count':count,'sources':len(counts),'statistics':record,
      'values':answer.tolist(),'labels':[f'{pair.split("_")[0]} coordinate {j}' for j in range(row_start,endrow)],'native_width':2560,'start':column_start,'end':endcol,'whole_field_absmax':float(np.max(np.abs(answer))),
      'normalization':view+'; exact equal-source matched outer-product contrast, not covariance or a physical connection','recipe_download':'/api/rdc-relation/matrix-recipe'}


@router.get('/matrix-recipe')
def matrix_recipe():return FileResponse(BASE/'relation_atlas/pair_index.json',filename='complete_relation_pair_control_index.json')


@router.get('/units')
def units(start:int=Query(0,ge=0,le=9727),count:int=Query(9728,ge=1,le=9728)):
    z=arrays(BASE/'native/all_unit_profiles.npz');names=['actual_activation_energy','all_predicted_sources_MSE','actual_past_hybrid_MSE'];v=np.stack([z[k] for k in names]);return field_response(v,names,start,count,'All9728 native block23 MLP units; energy or prediction error, not residual-stream coordinates')


@router.get('/generation')
def generation(sample:str='',field:str='self_predicted_h36',start:int=Query(0,ge=0),count:int=Query(2560,ge=1,le=2560)):
    row('fresh',sample);path=BASE/f'generation/commits/{sample}.json'
    if not path.exists():raise HTTPException(409,'This source is outside the64 preselected generation cases')
    if field not in ('native_h12','native_h36','self_predicted_h36','native_h36_on_self_prefix'):raise HTTPException(422,'Unknown generated field')
    a=arrays(BASE/f'generation/fields/{sample}.npz')[field];v=decode(a) if a.dtype==np.uint16 else a
    return {'record':read(path),'field':field_response(v,[f'generated step {i}' for i in range(len(v))],start,count,'Raw full states; compare self states only against native reference on SAME self-generated prefix')}


@router.get('/download')
def download(scope:str='fresh',sample:str=''):
    row(scope,sample);return FileResponse(BASE/scope/f'fields/{sample}.npz',filename=f'{scope}_{sample}_native_fields.npz')


@router.get('/figure/{figure}')
def figure(figure:str):
    item=next((f for f in read(BASE/'figures/index.json',[]) if f['id']==figure),None)
    if item is None:raise HTTPException(404,'Unknown figure')
    return FileResponse(BASE/'figures'/item['file'],media_type='image/png')


def scale_folder(model):
    if model not in ('qwen4','qwen14','glm4'):raise HTTPException(422,'Unknown model')
    return BASE/'scale'/model


@router.get('/scale-samples')
def scale_samples(model:str='qwen14'):
    folder=scale_folder(model)
    return [{k:r[k] for k in ('sample_id','language','split','text','positions')}|{'fresh':r['fresh'],'generated':(folder/f'generation/{r["sample_id"]}.json').exists()} for p in sorted((folder/'rows').glob('*.json')) if (r:=read(p))]


@router.get('/scale-field')
def scale_field(model:str='qwen14',sample:str='',field:str='late',start:int=Query(0,ge=0),count:int=Query(5120,ge=1,le=9728)):
    folder=scale_folder(model)
    if sample not in {p.stem for p in (folder/'rows').glob('*.json')}:raise HTTPException(404,'Unknown model source')
    r=read(folder/f'rows/{sample}.json');runtime=read(folder/'runtime.json',{})
    if field not in ('early','late','generation_early','generation_late','generation_errors'):raise HTTPException(422,'Unknown scale field')
    generation=read(folder/f'generation/{sample}.json',{})
    if field=='generation_errors':
        path=folder/'generation_all_coordinate_errors.npz'
        if not path.exists():raise HTTPException(409,'Generation not yet committed')
        z=arrays(path);v=np.concatenate([z[k+'_squared_error_sum']/np.maximum(z['counts'],1)[:,None] for k in ('current','temporal')]);labels=[f'{k} step {j}' for k in ('current','temporal') for j in range(8)]
        note='All-coordinate generation mean squared errors across committed sources; temporal step0 is undefined and stored0, not an evaluated update.'
    elif field.startswith('generation_'):
        path=folder/f'generation_fixtures/{sample}.npz'
        if not path.exists():raise HTTPException(409,'Individual generated states retained only for four predeclared fixtures; all-source coordinate errors and input recipes retained')
        v=decode(arrays(path)[field.removeprefix('generation_')]);labels=[f'generated step {i}' for i in range(len(v))];note='Native batch4 continuation, full native width'
    else:
        if model=='qwen4':
            z=arrays(BASE/r['origin']/f'fields/{sample}.npz');v=decode(z['h12'][r['positions']] if field=='early' else z['h36'][[0,1,3,4]])
        else:v=decode(arrays(folder/f'fields/{sample}.npz')[field])
        labels=[f'native token {p}' for p in r['positions']];note='Native batch1 four anchor positions; coordinate indices are not aligned across models'
    return {'record':r,'runtime':runtime,'generation':generation,'field':field_response(v,labels,start,count,note)}


@lru_cache(maxsize=1)
def parameter_reader():
    import importlib.util
    path=ROOT/'tests/glm5/rdc_relation_native_parameters.py';spec=importlib.util.spec_from_file_location('rdc_relation_native_parameters',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@router.get('/scalar-parameters')
def scalar_parameters(sample:str='',anchor:int=Query(0,ge=0,le=1),unit:int=Query(0,ge=0,le=9727),input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    row('fresh',sample)
    if sample not in [r['sample_id'] for r in material('fresh')[:4]]:raise HTTPException(409,'Choose one of the four predeclared complete native factor fixtures')
    path=BASE/f'scalar_parameters/fields/{sample}.npz'
    if not path.exists():raise HTTPException(409,'Parameter witness not yet completed')
    x=decode(arrays(path)['postattention_normalized_input'][anchor]).astype(float);f=arrays(BASE/f'native/fields/{sample}.npz');a=f['fixture_oracle_activation'][anchor].astype(float);reader=parameter_reader();weights=[]
    for name in ('gate_proj','up_proj','down_proj'):
        key=f'model.layers.23.mlp.{name}.weight';matrix=reader.parameter(ROOT,key);weights.append(reader.decode(matrix[output_coordinate if name=='down_proj' else unit]).astype(float))
    gate,up,down=weights;terms=np.stack([x*gate,x*up]);unit_terms=a*down
    return {'sample_id':sample,'layer':23,'anchor':anchor,'unit':unit,'input_coordinate':input_coordinate,'output_coordinate':output_coordinate,
      'scalar_chain':{'normalized_x_i':float(x[input_coordinate]),'Wgate_k_i':float(gate[input_coordinate]),'Wup_k_i':float(up[input_coordinate]),'actual_gate':float(f['fixture_oracle_gate'][anchor,unit]),'actual_up':float(f['fixture_oracle_up'][anchor,unit]),'actual_activation':float(a[unit]),'Wdown_j_k':float(down[unit]),'unit_output_contribution':float(unit_terms[unit])},
      'all_input_coordinate_sums':terms.sum(1).tolist(),'all_unit_output_sum':float(unit_terms.sum()),'actual_output_coordinate':float(f['fixture_oracle_mlp'][anchor,output_coordinate]),
      'input_terms':field_response(terms,['All gate input contributions','All up input contributions'],0,2560,'Actual scalar parameters, full FP64 sums; not a Top-K explanation'),
      'unit_terms':field_response(unit_terms[None],['All MLP units writing to selected output coordinate'],0,9728,'Captured actual BF16 activation times actual BF16 Wdown; FP64 accounting'),
      'boundary':'Postattention input reconstructed with actual native norm. Arithmetic witness, not proof of a one-unit semantic role or causal necessity.'}


@router.get('/analysis-index')
def analysis_index():
    answer=[]
    for area,names in {'output_geometry':['current_all_coordinate_attributions','temporal_all_coordinate_attributions','generation_all_coordinate_attributions','current_coordinate_profiles','temporal_coordinate_profiles','self_generation_coordinate_profiles'],
      'source_normalization':['all_coordinate_source_energy','all_coordinate_prediction_MSE'],'scalar_parameters':['all_coordinate_and_unit_error'],
      'native_source_audit':['all_coordinate_profiles'],'boundary_compilation':['all_coordinate_and_unit_errors','constant_update_control'],
      'energy_audit':['first_boundary_coordinate_errors'],
      'boundary_relations':['nmod_profiles','conj_profiles','compound_profiles'],
      'surrogate_stability':[str(p.relative_to(BASE/'surrogate_stability')).removesuffix('.npz').replace('\\','/') for p in (BASE/'surrogate_stability/cycles').glob('*.npz')]}.items():
        for name in names:
            path=BASE/area/(name+'.npz')
            if path.exists():
                with np.load(path) as z:
                    answer.extend({'area':area,'file':name,'array':key,'shape':list(z[key].shape)} for key in z.files if z[key].ndim in (1,2) and z[key].shape[-1] in (2560,5120,9728))
    return answer


@router.get('/analysis-field')
def analysis_field(area:str='',file:str='',array:str='',start:int=Query(0,ge=0),count:int=Query(9728,ge=1,le=9728)):
    if not any(r['area']==area and r['file']==file and r['array']==array for r in analysis_index()):raise HTTPException(404,'Unknown registered diagnostic array')
    v=arrays(BASE/area/(file+'.npz'))[array];v=v[None] if v.ndim==1 else v
    note='Full native indices. Attribution is a normalized-coordinate path diagnostic, not causal necessity.5120-width prediction arrays concatenate all H23 then all H36 coordinates.'
    if area=='surrogate_stability' and array.startswith('spectrum_'):note='Complete fitted-surrogate spectrum: columns are EIGENVALUE INDICES, NOT native hidden coordinates. Includes algebraic zeros from the exact finite-fit factor identity. Not the original Transformer spectrum.'
    return field_response(v,[f'observation {i}' for i in range(len(v))],start,count,note)


@router.get('/temporal-operator')
def temporal_operator(token_id:int=Query(0,ge=0),input_start:int=Query(0,ge=0,le=2559),output_start:int=Query(0,ge=0,le=2559),count:int=Query(32,ge=1,le=256)):
    result=read(BASE/'surrogate_stability/result.json',{})
    if token_id not in result.get('token_ids',[]):raise HTTPException(409,'This incoming token is not in the registered surrogate operator audit')
    meta=[r for r in read(BASE/'dynamics/rows.json') if r['split']=='train'];ie=min(input_start+count,2560);oe=min(output_start+count,2560)
    h=np.stack([decode(arrays(BASE/f'main/fields/{m["sample_id"]}.npz')['h36'][3*m['anchor']])[input_start:ie] for m in meta]).astype(float)
    model=arrays(BASE/'dynamics/previous_embedding_bilinear/model.npz');b=model['alpha'][:,output_start:oe].astype(float)*model['target_scale'][output_start:oe];w=arrays(BASE/f'surrogate_stability/token_coefficients/{token_id}.npz')
    A=h.T@(w['linear_kernel_weights'][:,None]*b)/result['state_kernel_scale'];constant=w['constant_kernel_weights']@b+model['mean'][output_start:oe]
    return {'values':A.tolist(),'constant':constant.tolist(),'labels':[f'previous H36 coordinate {j}' for j in range(input_start,ie)],'native_width':2560,'start':output_start,'end':oe,
      'whole_field_absmax':float(np.max(np.abs(A))),'normalization':'Exact all-training-factor sum for learned G_e(h)=h A_e+b_e; not native LLM weights or a rank-truncated state','token_id':token_id}
