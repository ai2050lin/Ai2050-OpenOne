"""Read-only native trace queries and bounded CPU parameter contribution checks."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from fastapi import APIRouter,HTTPException,Query
from shared.dist_protocol.mechanism_trace import TraceStore,unbits

router=APIRouter(prefix='/api/traces',tags=['native-mechanism-traces'])
from server.research_space_service import router as research_space_router,binding
router.include_router(research_space_router)
from server.mlp_bank_service import router as mlp_bank_router
router.include_router(mlp_bank_router)
from server.mlp_dictionary_service import router as mlp_dictionary_router
router.include_router(mlp_dictionary_router)
from server.mlp_parameter_audit_service import router as mlp_parameter_audit_router
router.include_router(mlp_parameter_audit_router)
from server.mlp_fidelity_service import router as mlp_fidelity_router
router.include_router(mlp_fidelity_router)
from server.mlp_functional_service import router as mlp_functional_router
router.include_router(mlp_functional_router)
_store=None


def store():
    global _store
    if _store is None:_store=TraceStore()
    return _store


def checked(fn,*args,**kwargs):
    try:return fn(*args,**kwargs)
    except KeyError as e:raise HTTPException(409,str(e)) from e
    except (ValueError,OSError) as e:raise HTTPException(422,str(e)) from e


@router.get('/catalog')
def catalog(run_id:str='',parent:str=''):
    from server.trace_catalog import catalog as query
    if parent:checked(store().manifest,parent)
    return query(store(),run_id,parent)


@router.get('/history')
def history():
    p=Path(__file__).resolve().parents[1]/'research/phase2793/history-index.research.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {'entries':[],'state':'not_migrated'}


@router.get('/context')
def context(run_id:str,model_snapshot:str=''):
    try:linked=binding(run_id,model_snapshot or None)
    except ValueError as e:raise HTTPException(409,str(e)) from e
    if linked:return {**linked.get('context',{}),'run_id':run_id,'model_snapshot':linked['model_snapshot'],'report_sha256':linked['report_sha256'],'graph_sha256':linked['graph_sha256']}
    if run_id=='NMLP-LONG-20261104-v1':
        p=Path(__file__).resolve().parents[1]/'research/native_mlp_long/v1/unified-evidence.research.json'
        if not p.is_file():raise HTTPException(404,'原生MLP长文本证据尚未归档。')
        return json.loads(p.read_text(encoding='utf-8'))
    folder={'P2793-ENTITY-ARBITRATION-v1':'phase2793','P2794-ATOMIC-ARBITRATION-v4':'phase2794'}.get(run_id)
    if folder is None:raise HTTPException(404,'该运行暂无统一研究上下文。')
    p=Path(__file__).resolve().parents[1]/'research'/folder/'unified-evidence.research.json'
    if not p.is_file():raise HTTPException(404,'研究上下文尚未归档。')
    return json.loads(p.read_text(encoding='utf-8'))


@router.get('/profile')
def profile(run_id:str,block:int=Query(0,ge=0,le=35),space:str='mlp'):
    ctx=context(run_id)
    if ctx.get('study_type'):raise HTTPException(404,'该研究不使用文档−现实剖面；请查询对应类型的单元说明书或已归档题族响应。')
    if space not in ['mlp','hidden']:raise HTTPException(422,'只支持原生mlp/hidden空间。')
    folder='phase2793_arbitration' if run_id=='P2793-ENTITY-ARBITRATION-v1' else 'phase2794_arbitration_v4'
    p=Path(__file__).resolve().parents[1]/'tests/glm5/result'/folder/'native-profiles.npz'
    if not p.is_file():raise HTTPException(404,'本节点没有原生响应剖面。')
    with np.load(p) as z:
        if 'profile_block' in z and int(z['profile_block'])!=block:raise HTTPException(404,f'本运行响应剖面仅分析发现集选定的块{int(z["profile_block"])}，其他块的原生阵列仍可查询。')
        disc=z[space+'_discovery_delta'][block];held=z[space+'_heldout_delta'][block]
    return {'schema':'mechanism_response_profile.v1','kind':'real','run_id':run_id,'block':block,'space':space,
        'discovery_delta':disc.tolist(),'heldout_delta':held.tolist(),'n_discovery_entities':ctx.get('n_discovery_entities',16),'n_heldout_entities':ctx.get('n_heldout_entities',32),
        'boundary':'mlp.down_proj.input' if space=='mlp' else f'hidden_states[{block if run_id=="P2793-ENTITY-ARBITRATION-v1" else block+1}]',
        'interpretation':'Document-minus-reality scope-prompt response, including errors; descriptive only, no FWER-controlled semantic or causal claim.'}


@router.get('/{trace_id}/manifest')
def manifest(trace_id:str):return checked(store().manifest,trace_id)[0]


@router.get('/{trace_id}/field')
def field(trace_id:str,block:int=Query(0,ge=0),position:int=Query(...,ge=0),boundary:str='mlp_activation'):
    if boundary in ['hidden_state','embedding']:return checked(hidden_at,trace_id,0 if boundary=='embedding' else block,position)
    return checked(store().field,trace_id,block,position,boundary)


def hidden_at(trace_id,layer,position):
    d,_=store().manifest(trace_id)
    if not d.get('parent_trace_id'):return store().hidden(trace_id,layer,position)
    # Cached generation forwards contain only the newly evaluated token, while
    # manifest positions refer to the complete absolute token history.
    if not 0<=layer<=d['n_layers']:raise ValueError('hidden layer outside trace')
    if position not in d['readout_positions']:raise KeyError('cached forward did not recapture this historical token')
    a=np.load(store().native_path(trace_id,'hidden_readouts.npy'),mmap_mode='r')
    v=unbits(a[layer,d['readout_positions'].index(position)])
    return {'schema':'mechanism_field.v1','trace_id':trace_id,'layer':layer,'position':position,
        'boundary':'embedding' if layer==0 else 'hidden_state','shape':[len(v)],'values':v.tolist(),'source':'native_BF16_bits_cached_current_token'}


@router.get('/{trace_id}/attention-row')
def attention_row(trace_id:str,block:int=Query(0,ge=0),head:int=Query(0,ge=0),query:int=Query(...,ge=0)):
    d,_=checked(store().manifest,trace_id)
    if d.get('parent_trace_id') and query not in d['readout_positions']:raise HTTPException(409,'缓存生成步仅重新采集当前query；历史query请查前缀轨迹。')
    if d.get('reconstruction',{}).get(str(block),{}).get('attention_relative_l2',0)>.05:
        raise HTTPException(409,'该历史注意力行未通过与真实SDPA的重建核对；不展示为真实权重。请查看同一案例的新采集版本。MLP/Head实际输出另行保留。')
    return checked(store().attention_row,trace_id,block,head,query)


def trace_model(d):
    from deploy.parameter_service import catalog
    from deploy.parameter_reader import model_snapshot
    preferred=d.get('provenance',{}).get('model_path','')
    models=sorted(catalog(),key=lambda m:str(m['_path']).replace('\\','/').lower()!=preferred.replace('\\','/').lower())
    for m in models:
        if m['config']['hidden_size']!=d['d_model'] or m['config']['num_hidden_layers']!=d['n_layers']:continue
        if model_snapshot(m['_path'])['sha256']==d['model_snapshot']:return m
    raise ValueError('no configured local weights match the complete trace snapshot')


@lru_cache(maxsize=8)
def tensor_file(folder,name):
    from deploy.parameter_reader import header
    for p in sorted(Path(folder).glob('*.safetensors')):
        h,_,_=header(p)
        if name in h:return str(p)
    raise ValueError('parameter tensor missing')


def parameter(folder,name,index):
    from safetensors import safe_open
    # Reads only selected rows/columns on CPU; no loaded research model/GPU.
    with safe_open(tensor_file(str(folder),name),framework='pt',device='cpu') as f:
        x=f.get_slice(name)[index]
    return x.float().numpy()


def contributions(trace_id,block,position,head,unit,coordinate,*,native=None):
    d,_=store().manifest(trace_id) if native is None else (native[0],None)
    if not 0<=block<d['n_layers'] or not 0<=head<d['n_heads'] or not 0<=unit<d['d_ff'] or not 0<=coordinate<d['d_model']:raise ValueError('index outside native dimensions')
    if position not in d['readout_positions']:raise KeyError('module source decomposition not captured at selected query')
    m=trace_model(d);folder=m['_path'];r=d['readout_positions'].index(position)
    if native is None:
        z=store().block(trace_id,block)
        vals={k:unbits(z[k][r]) for k in ['gate','value','activation','mlp_input','head_z','attention_output','mlp_output','resid_in']}
    else:vals=native[1]
    prefix=f'model.layers.{block}.';gate=parameter(folder,prefix+'mlp.gate_proj.weight',(unit,slice(None)))
    up=parameter(folder,prefix+'mlp.up_proj.weight',(unit,slice(None)))
    down=parameter(folder,prefix+'mlp.down_proj.weight',(coordinate,slice(None)))
    orow=parameter(folder,prefix+'self_attn.o_proj.weight',(coordinate,slice(None)))
    gt=gate*vals['mlp_input'];ut=up*vals['mlp_input'];dt=down*vals['activation']
    heads=(orow.reshape(d['n_heads'],d['head_dim'])*vals['head_z']).sum(-1)
    # Decompose this unit's gate input with the actual sample RMS held fixed.
    import torch
    mid=(torch.from_numpy(vals['resid_in']).to(torch.bfloat16)+torch.from_numpy(vals['attention_output']).to(torch.bfloat16)).float().numpy()
    gamma=parameter(folder,prefix+'post_attention_layernorm.weight',slice(None))
    read=gate*gamma/np.sqrt(np.mean(mid**2)+m['config']['rms_norm_eps'])
    omatrix=parameter(folder,prefix+'self_attn.o_proj.weight',(slice(None),slice(None)))
    readheads=(omatrix.T@read).reshape(d['n_heads'],d['head_dim'])
    hg=(readheads*vals['head_z']).sum(-1);residual=float(read@vals['resid_in'])
    source=None
    if d['deep'] and native is None:
        weights=np.asarray(store().attention_row(trace_id,block,head,position)['weights'])
        group=head//(d['n_heads']//d['n_kv_heads']);v=unbits(z['v'][group])
        source=(weights*(v@orow.reshape(d['n_heads'],d['head_dim'])[head])).tolist()
    return {'schema':'mechanism_connections.v1','kind':'real','trace_id':trace_id,'model_snapshot':d['model_snapshot'],
        'local_model_id':m['id'],'verified_full_snapshot':True,'block':block,'position':position,'head':head,'unit':unit,'coordinate':coordinate,
        'gate_terms':gt.tolist(),'value_terms':ut.tolist(),'mlp_output_terms':dt.tolist(),'head_output_terms':heads.tolist(),
        'head_gate_terms':hg.tolist(),'residual_gate_term':residual,'source_token_terms':source,
        'observed':{'gate':float(vals['gate'][unit]),'value':float(vals['value'][unit]),'activation':float(vals['activation'][unit]),
            'mlp_output_coordinate':float(vals['mlp_output'][coordinate]),'attention_output_coordinate':float(vals['attention_output'][coordinate])},
        'reconstructed':{'gate':float(gt.sum(dtype=np.float64)),'value':float(ut.sum(dtype=np.float64)),
            'mlp_output_coordinate':float(dt.sum(dtype=np.float64)),'attention_output_coordinate':float(heads.sum(dtype=np.float64)),
            'gate_via_heads_plus_residual':float(hg.sum(dtype=np.float64)+residual)},
        'source':'exact original parameter slices and native sample responses; FP32 sums vs native BF16 can differ by rounding',
        'edge_types':{'head_to_mlp':'local direct gate-input attribution with actual RMS held fixed; approximate, not causal',
            'mlp_to_output':'sum of all native unit products for selected output coordinate','embedding_to_layers':'residual computational continuity; no direct semantic-causality claim'},
        'causal_evidence':'not established by this arithmetic decomposition'}


@router.get('/{trace_id}/connections')
def connections(trace_id:str,block:int=Query(0,ge=0),position:int=Query(...,ge=0),head:int=Query(0,ge=0),unit:int=Query(0,ge=0),coordinate:int=Query(0,ge=0)):
    return checked(contributions,trace_id,block,position,head,unit,coordinate)
