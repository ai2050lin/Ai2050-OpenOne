"""Read-only query/ordered-event atlas; no inference, training or checkpoint writes."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import gzip,importlib.util,json
import numpy as np
from fastapi import APIRouter,HTTPException,Query
from fastapi.responses import FileResponse
from server.rdc_feature_service import ROOT,read,decode
from server.rdc_relation_service import field_response
from server.rdc_joint_service import npz_headers

BASE=ROOT/'tests/glm5/result/rdc_query_campaign_20260913'
router=APIRouter(prefix='/api/rdc-query',tags=['RDC native query and event evidence'])

@lru_cache(maxsize=4)
def gz(path,mtime):return json.loads(gzip.decompress(Path(path).read_bytes()))

def compressed(path):
    p=Path(path)
    return gz(str(p),p.stat().st_mtime_ns) if p.exists() else []

def material(scope='natural'):
    if scope=='identifiability':
        packet=compressed(BASE/'identifiability/material.json.gz')
        return packet.get('controlled',[]) if packet else []
    files={'natural':'material/natural.json.gz','transfer':'transfer/material.json.gz','scale':'scale/material.json.gz','followup':'followup/material.json.gz'}
    if scope not in files:raise HTTPException(422,'Unknown material scope')
    return compressed(BASE/files[scope])

def source(sid,scope='natural'):
    r=next((r for r in material(scope) if r['sample_id']==sid),None)
    if r is None:raise HTTPException(404,'Unknown registered source ID')
    return r

def folder(scope,model):
    if model not in ['qwen4','qwen14','glm4']:raise HTTPException(422,'Unknown model')
    if scope=='scale':return BASE/'scale'/model
    if model!='qwen4':raise HTTPException(409,'This scope only contains Qwen4, no model substitution')
    if scope not in ['natural','transfer','followup','identifiability']:raise HTTPException(422,'Unknown scope')
    return BASE/{'natural':'capture','transfer':'transfer','followup':'followup/capture','identifiability':'identifiability/relations/native'}[scope]

def registered(path):
    if '\\' in path or '..' in path.split('/') or path.endswith('.tmp.npz'):raise HTTPException(404,'Outside registered archive')
    p=BASE/path
    if not p.resolve().is_relative_to(BASE.resolve()) or not p.is_file() or p.suffix!='.npz':raise HTTPException(404,'Unknown numerical archive')
    return p

def show(a,labels,view='raw',**extra):
    a=np.asarray(a,dtype=float)
    if view=='RMS':a=a/np.sqrt(np.mean(a*a,-1,keepdims=True)).clip(1e-12)
    elif view!='raw':raise HTTPException(422,'Unknown numerical view')
    return field_response(a,labels,0,a.shape[-1],view+'; original native index order; no coordinate pruning. '+extra.get('axes',''),**extra)

@router.get('/overview')
def overview():
    parts={'contract':'contract.json','material':'material/result.json','algebra':'algebra/result.json','atlas':'atlas/result.json',
      'events':'events/result.json','rules':'rules/fit_result.json','vocabulary':'rules/vocabulary_result.json','pairs':'pairs/result.json',
      'transfer':'transfer/fit_result.json','transfer_vocabulary':'transfer/vocabulary_result.json','injection':'transfer/injection/result.json',
      'formation':'formation/result.json','followup':'followup/result.json','identifiability':'identifiability/analysis/result.json','theory':'theory_snapshot.json','final':'verification/final.json','resources':'resources.json',
      'prediction_analysis':'analysis/phase2741.json','formation_history_analysis':'analysis/phase2742.json','next_stage_admission':'next_stage_admission.json','continuation_after_2744':'continuation_after_2744.json'}
    result={k:read(BASE/p,{}) for k,p in parts.items()};captured=sum(1 for _ in (BASE/'capture/commits').glob('*.json'))
    result.update(committed_natural_prefixes=captured,committed_query_endpoints=captured*100,completion={k:(BASE/p).exists() for k,p in parts.items()},
      scale={m:read(BASE/'scale'/m/'result.json',{}) for m in ['qwen4','qwen14','glm4']},
      late={b:read(BASE/'late'/b/'result.json',{}) for b in ['native','fixed128_digit','entropy_digit','entropy_letter','terminal_marker_digit']},
      ledger=read(BASE/'compute_ledger.json',[]),figures=read(BASE/'figures/index.json',{}).get('figures',[]),
      status='Query responses, exact accounting, learned predictions, restricted training and own-history generation are distinct evidence levels.')
    return result

@router.get('/probes')
def probes():return read(BASE/'probes/protocol.json',{}).get('probes',[])

@router.get('/samples')
def samples(scope:str='natural',model:str='qwen4',cohort:str='',split:str='',offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=10000)):
    root=folder(scope,model);rows=[r for r in material(scope) if (not cohort or r['cohort']==cohort) and (not split or r['split']==split)]
    rr=rows[offset:offset+limit];detail=set(read(BASE/'material/protocol.json',{}).get('detailed_prefix_ids',[]))
    native={r['sample_id']:read(root/'commits'/f"{r['sample_id']}.json",{}) for r in rr} if scope=='scale' else {}
    return {'total':len(rows),'offset':offset,'rows':[{k:r[k] for k in ['sample_id','source_group','cohort','split','language']}|
      {'tokens':native.get(r['sample_id'],{}).get('prefix_tokens',len(r['prompt_ids'])),
       'tokenization_model':model if r['sample_id'] in native and native[r['sample_id']] else 'qwen4_material_reference',
       'captured':(root/'commits'/f"{r['sample_id']}.json").exists(),'representation':r.get('representation'),
       'detail':scope=='natural' and r['sample_id'] in detail} for r in rr]}

@router.get('/sample')
def sample(sample:str='',scope:str='natural',model:str='qwen4'):
    r=source(sample,scope);root=folder(scope,model)
    if scope!='scale':return r
    committed=read(root/'commits'/f'{sample}.json',{})
    tokenization=read(root/'tokenization.json',{})
    return r|{'material_reference_tokenization_model':'qwen4','selected_native_model':model,
      'native_prefix_token_ids':committed.get('prefix_ids'),
      'native_query_token_ids':tokenization.get('queries'),
      'native_tokenization_scope':'The original material prompt_ids are a Qwen4 reference. This model uses the separately recorded native prefix/query IDs; no cross-model token or coordinate matching is implied.'}

@router.get('/field')
def field(sample:str='',scope:str='natural',model:str='qwen4',mode:str='queries',layer:int=Query(12,ge=0,le=80),view:str='raw'):
    r=source(sample,scope);root=folder(scope,model);p=root/'fields'/f'{sample}.npz';cp=root/'commits'/f'{sample}.json'
    if not cp.exists():raise HTTPException(409,'Native capture not committed; no substitute data')
    with np.load(p) as z:
        if mode=='queries':a=decode(z['postnorm']);labels=[f"{i}: {q['probe_id']} {q['text']}" for i,q in enumerate(probes())]
        elif mode=='layers':
            if 'prefix_layers' not in z.files:raise HTTPException(409,'All-layer prefix anchors absent in this scope')
            a=decode(z['prefix_layers']);labels=[f'raw H{i}' for i in range(len(a))]
        elif mode=='sources':
            if 'prefix_H12_sources' not in z.files:raise HTTPException(409,'Full H12 sources are retained only for declared detail panel')
            a=decode(z['prefix_H12_sources']);labels=[f'prefix token {i}: {r["prompt_ids"][i]}' for i in range(len(a))]
        elif mode=='fixture':
            if 'full_prefix_layers' not in z.files:raise HTTPException(409,'Not a predeclared full all-token/all-layer fixture')
            if layer>=len(z['full_prefix_layers']):raise HTTPException(422,'Layer outside native depth')
            a=decode(z['full_prefix_layers'][layer]);labels=[f'prefix token {i}: {r["prompt_ids"][i]}' for i in range(len(a))]
        elif mode=='statistics':
            a=z['full_vocabulary_statistics'].T;labels=read(cp).get('statistics_columns',['entropy','KL_native_to_query_only','KL_query_only_to_native','argmax'])
        else:raise HTTPException(422,'Unknown field mode')
    result=show(a,labels,view,sample_id=sample,model=model,scope=scope,mode=mode)
    if mode=='statistics':
        result['axes']='Rows are recorded statistic types; columns are100diagnostic QUERY indices, not hidden coordinates. Argmax token IDs are categorical labels, not a numerical semantic distance.'
        result['normalization']+=' '+result['axes']
    return result

@router.get('/prediction')
def prediction(sample:str='',query:int=Query(0,ge=0,le=99),target:int=Query(2,ge=0,le=2),candidate:int=Query(4,ge=0,le=4)):
    source(sample);file=BASE/'rules/features'/f'{sample}.npz';decoder=BASE/'rules/decoder.npz'
    if not file.exists() or not decoder.exists():raise HTTPException(409,'Prediction detail not committed')
    with np.load(BASE/'capture/fields'/f'{sample}.npz') as z:
        prefix=decode(z['prefix_layers'][12]);actual=decode(z['postnorm'][query]) if target==2 else decode(z['query_H12_H24_rawH36'][query,target+1])
    with np.load(BASE/'prototypes/qwen4.npz') as z:q=decode(z[f'p{query}_H13'][-1])
    with np.load(file) as z:f=decode(z['candidate_H13'][candidate,query])
    with np.load(decoder) as z:b=z['beta'][candidate,target]
    x=np.stack([np.ones_like(q),prefix,q,f],-1);pred=(x*b).sum(-1)
    return show(np.stack([actual,pred,pred-actual]),['Actual native target','Frozen predicted target','Prediction minus target'],
      MSE=float(np.mean((pred-actual)**2)),available_inputs='Prefix H12/KV, known-query standalone prototype, native block12 weights; actual query future state is scoring-only.')

@router.get('/path-index')
def path_index():return read(BASE/'events/paths_index.json',[])

@lru_cache(maxsize=1)
def reader():
    spec=importlib.util.spec_from_file_location('rdc_query_native_scalar_reader',ROOT/'tests/glm5/rdc_relation_native_parameters.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@router.get('/ordered-path')
def ordered_path(path:str='',block:int=16,source_s:int=Query(0,ge=0),source_r:int=Query(0,ge=0),unit:int=Query(0,ge=0,le=9727),input_coordinate:int=Query(0,ge=0,le=2559),output_coordinate:int=Query(0,ge=0,le=2559)):
    item=next((r for r in path_index() if r['path'].replace('\\','/')==path),None)
    if item is None or block not in [16,35]:raise HTTPException(404,'Unknown committed native path')
    p=registered(path)
    with np.load(p) as z:
        g=z[f'L{block}_source_gate_read'].astype(float);u=z[f'L{block}_source_up_read'].astype(float);sig=z[f'L{block}_sigmoid_gate'].astype(float)
        c=z[f'L{block}_source_attention_write'].astype(float);den=float(z[f'L{block}_rms_denominator']);activation=z[f'L{block}_activation'].astype(float)
    if max(source_s,source_r)>=len(g):raise HTTPException(422,'Source index outside all native sources plus explicit other')
    rr=reader();gamma=rr.decode(rr.parameter(ROOT,f'model.layers.{block}.post_attention_layernorm.weight')).astype(float)
    wg=rr.decode(rr.parameter(ROOT,f'model.layers.{block}.mlp.gate_proj.weight')[unit]).astype(float)
    wu=rr.decode(rr.parameter(ROOT,f'model.layers.{block}.mlp.up_proj.weight')[unit]).astype(float)
    wd=rr.decode(rr.parameter(ROOT,f'model.layers.{block}.mlp.down_proj.weight')[output_coordinate]).astype(float)
    pair=sig*g[source_s]*u[source_r];swapped=sig*g[source_r]*u[source_s];fullmatrix=sig[unit]*g[:,unit,None]*u[None,:,unit]
    terms=None
    if source_s<len(c):terms=show(np.stack([c[source_s]*gamma/den*wg,c[source_s]*gamma/den*wu]),['All source-s gate input-coordinate terms','All source-s up input-coordinate terms'])
    return {'source_record':item,'chain':{'block':block,'source_s':source_s,'source_r':source_r,'unit':unit,'input_coordinate':input_coordinate,'output_coordinate':output_coordinate,
      'gate_read_s_k':float(g[source_s,unit]),'up_read_r_k':float(u[source_r,unit]),'sigmoid_g_k':float(sig[unit]),'ordered_pair_k':float(pair[unit]),
      'reverse_pair_k':float(swapped[unit]),'W_gate_k_i':float(wg[input_coordinate]),'W_up_k_i':float(wu[input_coordinate]),'W_down_j_k':float(wd[unit]),
      'selected_pair_write_j_from_unit_k':float(pair[unit]*wd[unit]),'all_units_pair_write_j':float(pair@wd),'other_source_index':len(g)-1},
      'ordered_matrix':show(fullmatrix,[f'gate source {i}' for i in range(len(g))],axes='Rows: gate-source index; columns: up-source index; final source is explicit other.'),
      'all_units':show(np.stack([pair,swapped,pair*wd]),['s-gate/r-up all units','r-gate/s-up all units','s-gate/r-up contributions to selected output'],axes='Columns are all9728MLP unit indices, not residual coordinates.'),
      'input_terms':terms,'native_output_terms':show((activation*wd)[None],['All actual native units writing selected output coordinate']),
      'scope':'Observed gate/up factors at fixed attention and RMS denominator. Source matrix axes are gate-source/up-source, not hidden dimensions. Antisymmetric entries cancel in the complete double sum; neither orientation nor a scalar proves a causal semantic role.'}

@router.get('/archives')
def archives(prefix:str='',offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=1000)):
    if '\\' in prefix or '..' in prefix.split('/'):raise HTTPException(404,'Invalid archive prefix')
    paths=sorted(p for p in BASE.rglob('*.npz') if not p.name.endswith('.tmp.npz') and p.relative_to(BASE).as_posix().startswith(prefix))
    return {'total':len(paths),'rows':[{'path':p.relative_to(BASE).as_posix(),'bytes':p.stat().st_size} for p in paths[offset:offset+limit]]}

@router.get('/arrays')
def arrays(path:str=''):
    p=registered(path);return npz_headers(str(p),p.stat().st_mtime_ns)

@router.get('/array')
def array(path:str='',name:str='',row_start:int=Query(0,ge=0),row_count:int=Query(37,ge=1,le=128),start:int=Query(0,ge=0),count:int=Query(8192,ge=1,le=17408)):
    p=registered(path)
    with np.load(p) as z:
        if name not in z.files:raise HTTPException(404,'Unknown original tensor name')
        a=z[name]
    shape=list(a.shape);a=decode(a) if a.dtype==np.uint16 else a
    if a.dtype.kind not in 'buif':raise HTTPException(422,'Not a numerical tensor')
    if a.ndim<2:a=a.reshape(1,-1)
    v=a.reshape(-1,a.shape[-1]);end=min(len(v),row_start+row_count)
    if row_start>=len(v) or start>=v.shape[-1]:raise HTTPException(422,'Page outside native axes')
    s=v[row_start:end,start:start+count]
    if not np.isfinite(s).all():raise HTTPException(422,'Undefined raw values; inspect archive and validity masks')
    labels=[str(tuple(map(int,np.unravel_index(i,a.shape[:-1])))) for i in range(row_start,end)]
    return field_response(v[row_start:end],labels,start,count,'Original tensor order; page is a display window, not Top-K. Inspect original shape to distinguish samples, queries, coordinates, parameters, heads and source positions.',tensor_shape=shape,row_start=row_start,row_end=end,total_rows=len(v))

@router.get('/event-index')
def event_index():
    packet=compressed(BASE/'events/material.json.gz')
    return [] if not packet else [{'sample_id':r['row']['sample_id'],'representation':r['row']['representation'],'anchors':r['anchors'],
      'captured':(BASE/'events/commits'/f"{r['row']['sample_id']}.json").exists()} for r in packet['trajectories']]

@router.get('/event')
def event(sample:str='',anchor:int=Query(0,ge=0),view:str='raw'):
    if sample not in {r['sample_id'] for r in event_index()}:raise HTTPException(404,'Unknown native event trajectory')
    cp=BASE/'events/commits'/f'{sample}.json'
    if not cp.exists():raise HTTPException(409,'Event replay not committed')
    record=read(cp)
    if anchor>=len(record['anchors']):raise HTTPException(422,'Anchor outside predeclared trajectory')
    old=read(ROOT/next(r['native_record'] for r in compressed(BASE/'events/material.json.gz')['trajectories'] if r['row']['sample_id']==sample))
    with np.load(BASE/'events/fields'/f'{sample}.npz') as z:h=decode(z['H'][anchor]);q=decode(z['dynamic_query_postnorm'][anchor]);stats=z['dynamic_full_vocabulary_statistics'][anchor]
    return {'record':record,'native_generated_text':old['generated_text'],'anchor_step':record['anchors'][anchor],
      'layer_field':show(h,[f'raw H{i}' for i in range(len(h))],view),'query_field':show(q,[p['probe_id'] for p in probes()],view),
      'query_statistics':stats.tolist(),'scope':'State at step j predicts token j, before it is appended. Output-text event annotations are not proof of an internal symbolic executor.'}

@router.get('/behavior-index')
def behavior_index(mode:str='late',branch:str='native'):
    if mode=='late' and branch in ['native','fixed128_digit','entropy_digit','entropy_letter','terminal_marker_digit']:p=BASE/'late'/branch/'commits'
    elif mode=='injection' and branch in ['native','code_identity','mapped_code']:p=BASE/'transfer/injection/commits'/branch
    else:raise HTTPException(422,'Unknown exact behavior branch')
    return [{'sample_id':r['sample_id'],'source_group':r['source_group'],'representation':r['representation']} for f in sorted(p.glob('*.json')) if (r:=read(f))]

@router.get('/behavior')
def behavior(mode:str='late',branch:str='native',sample:str=''):
    if sample not in {r['sample_id'] for r in behavior_index(mode,branch)}:raise HTTPException(404,'No exact committed trajectory')
    p=BASE/'late'/branch/'commits'/f'{sample}.json' if mode=='late' else BASE/'transfer/injection/commits'/branch/f'{sample}.json'
    return read(p)

@router.get('/download')
def download(path:str=''):
    p=registered(path);return FileResponse(p,filename=p.name)

@router.get('/identity-index')
def identity_index():
    return [{k:r[k] for k in ['sample_id','source_group','pair_id','family','language','world','target']} for r in material('identifiability')]

@router.get('/identity-pair')
def identity_pair(sample:str='',variant:str='native',mode:str='queries',view:str='raw'):
    variants=['native','natural_target_2742','within_cohort_permuted_target_2742','natural_target_2743','within_cohort_permuted_target_2743']
    if variant not in variants:raise HTTPException(422,'Unknown actual parameter variant')
    row=source(sample,'identifiability');pair=sorted([r for r in material('identifiability') if r['pair_id']==row['pair_id']],key=lambda r:r['world'])
    if len(pair)!=2:raise HTTPException(409,'Token-matched pair not frozen')
    arrays=[];labels=[];commits=[];behavior=[]
    for r in pair:
        root=BASE/'identifiability/relations'/variant;cp=root/'commits'/f"{r['sample_id']}.json"
        if not cp.exists():raise HTTPException(409,'Requested variant is not captured; no substitute model')
        commit=read(cp);commits.append(commit)
        with np.load(root/'fields'/f"{r['sample_id']}.npz") as z:
            if mode=='queries':
                a=decode(z['matched_subset_postnorm']);ll=[f'world {r["world"]}, query {q}' for q in z['matched_query_indices']]
            elif mode=='units':
                names=[f'L{b}_{f}' for b in [16,35] for f in ['gate_proj','up_proj','activation']]
                a=np.stack([decode(z[n]) for n in names]);ll=[f'world {r["world"]}, {n}' for n in names]
            elif mode=='layers':
                a=decode(z['prefix_layers' if variant=='native' else 'prefix_selected_layers_H16_H17_H36']);ll=[f'world {r["world"]}, H{i}' for i in commit['layer_indices']]
            else:raise HTTPException(422,'Unknown identity field mode')
        arrays.append(a);labels+=ll
        behavior.append(read(BASE/'identifiability/behavior'/variant/'commits'/f"{r['sample_id']}.json",{}))
    scope='Columns are all9728native MLP units; rows distinguish block,gate/up/activation and world.' if mode=='units' else 'Columns are all2560native residual coordinates, never PCA or selected coordinates.'
    field=show(np.concatenate(arrays),labels,view);field['normalization']+=' '+scope
    return {'material':pair,'variant':variant,'field':field,'capture':commits,'own_history':behavior,
      'scope':'Exact full-token multiset and fixed question; ordered relations differ. This excludes a bag-only explanation, not every positional or shallow sequence explanation. Trained variants are actual2742BF16deltas; six query strings use identical native-vs-trained batch grouping. '+scope}

@router.get('/figure/{name}')
def figure(name:str):
    if name not in {r['path'] for r in read(BASE/'figures/index.json',{}).get('figures',[])}:raise HTTPException(404,'Unknown registered figure')
    return FileResponse(BASE/'figures'/name)
