"""Read-only native BF16 checkpoint/source addresses, no torch or model load.

QKV scalar edits are measured in Phase2689; baseline native fields in2687.
Never synthesize per-case changed-P arrays from aggregate response maps.
"""
import json,math,struct
from functools import lru_cache
from pathlib import Path
import numpy as np
from fastapi import HTTPException
from .native_path_parameter_query import RESULT

MODEL=RESULT.parents[2]/'models/hf/qwen3-4b'
FIELD=RESULT/'phase2687_role_qkv_field'
SCALAR=RESULT/'phase2689_native_qkv_scalar'
LAYERS=(0,5,17,23,26,27,28,35)

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def decode(a):return (a.astype(np.uint32)<<16).view(np.float32) if a.dtype==np.uint16 else a

@lru_cache(maxsize=1)
def metadata():
    if not (SCALAR/'analysis/final.json').exists():raise HTTPException(404,'QKV scalar measurements not completed')
    rows=read(RESULT/'phase2686_independent_role_contract/material/initial.json')
    return {r['case_index']:r for r in rows if r['parameter_published']}

@lru_cache(maxsize=1)
def controls():return read(SCALAR/'protocol/frozen.json')['controls']

@lru_cache(maxsize=1)
def checkpoint_index():return read(MODEL/'model.safetensors.index.json')['weight_map']

@lru_cache(maxsize=8)
def header(shard):
    path=(MODEL/shard).resolve()
    if not path.is_relative_to(MODEL.resolve()):raise HTTPException(400,'Invalid checkpoint shard')
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        assert 0<n<32*1024**2
        obj=json.loads(f.read(n))
    return path,n+8,obj

def native_weight(key,index):
    """Memory-map validated learned tensor, copy only requested row/scalar."""
    shard=checkpoint_index()[key];path,start,meta=header(shard);m=meta[key]
    assert m['dtype']=='BF16'
    a,b=m['data_offsets'];shape=tuple(m['shape'])
    assert b-a==math.prod(shape)*2 and start+b<=path.stat().st_size
    mm=np.memmap(path,mode='r',dtype='<u2',offset=start+a,shape=shape)
    value=decode(np.asarray(mm[index])).astype(np.float64,copy=True)
    del mm
    assert np.isfinite(value).all()
    return value

@lru_cache(maxsize=2)
def source_pack(case,layer):
    with np.load(FIELD/f'source/case_{case:04d}.npz',allow_pickle=False) as z:
        return {k.split('__',1)[1]:decode(z[k]).astype(np.float64) for k in z.files if k.startswith(f'L{layer}__')}

@lru_cache(maxsize=2)
def hidden(case):
    with np.load(FIELD/f'field/case_{case:04d}.npz',allow_pickle=False) as z:return z['full__h']

@lru_cache(maxsize=1)
def natural_records():return {r['case_index']:r for r in read(FIELD/'analysis/records.json')}

@lru_cache(maxsize=2)
def scalar_record(case):
    cid=metadata()[case]['case_id'];ids=read(SCALAR/'protocol/frozen.json')['case_ids'];i=ids.index(cid)
    return read(SCALAR/f'analysis/case_{i:03d}.json')

def options():
    return {'source_phase':2687,'scalar_phase':2689,'layers':LAYERS,'coordinates':2560,'MLP_units':9728,
        'cases':[{'case':i,'label':r['case_id'],'tokens':len(r['prompt_ids']),'body_token':r['body_end_token']} for i,r in metadata().items()],
        'controls':controls(),'boundary':'16 preregistered truth/v0 examples. Full coordinate/native parameter addressing; not all128 intervention prefixes. Published changed-weight natural outputs all unchanged. No model load.'}

def query(case=0,layer=0,kind='q',output_row=0,input_coordinate=947,token=0,query_position=1,
          source_token=0,head=0,head_coordinate=0,checkpoint=0,unit=0,output_coordinate=0):
    row=metadata().get(case)
    if row is None:raise HTTPException(404,'Unknown published native QKV case')
    n=len(row['prompt_ids']);k=input_coordinate;j=unit;d=head_coordinate;out=output_coordinate
    if not(layer in LAYERS and kind in ('q','k','v') and 0<=output_row<(4096 if kind=='q' else 1024)
           and 0<=k<2560 and 0<=token<n and query_position in (0,1) and 0<=source_token<n
           and 0<=head<32 and 0<=d<128 and 0<=checkpoint<37 and 0<=j<9728 and 0<=out<2560):
        raise HTTPException(400,'Physical coordinate out of range')
    s=source_pack(case,layer);pre=f'model.layers.{layer}.';q=query_position;qt=(row['body_end_token'],row['task_end_token'])[q];kv=head//4
    w=native_weight(pre+f'self_attn.{kind}_proj.weight',(output_row,slice(None)))
    x=s['upstream_attention_x'][token];terms=w*x;linear=s['upstream_linear_'+kind][token,output_row]
    norm_kind=kind if kind!='v' else 'k';nh=output_row//128;nd=output_row%128
    z=s['upstream_linear_'+norm_kind][token].reshape(-1,128)[nh]
    gamma=native_weight(pre+f'self_attn.{norm_kind}_norm.weight',slice(None))
    eps=read(MODEL/'config.json')['rms_norm_eps'];den=float(np.sqrt(np.mean(z*z)+eps))
    norm=s['upstream_normalized_'+norm_kind][token,nh]
    rotated=s['upstream_query_post_rope_full'][token,nh] if norm_kind=='q' else s['actual_key_post_rope'][token,nh]
    partner=(nd+64)%128;rotate_sign=-1 if nd<64 else 1
    cos=s['upstream_rope_cos'][token,nd];sin=s['upstream_rope_sin'][token,nd]
    expected_rope=norm[nd]*cos+rotate_sign*norm[partner]*sin
    p=s['actual_probability'][q,head];Q=s['actual_query_post_rope'][q,head];K=s['actual_key_post_rope'][:,kv];V=s['actual_value'][:,kv]
    wo=native_weight(pre+'self_attn.o_proj.weight',(out,slice(head*128,(head+1)*128)))
    source_projection=(V@wo)*p
    gate=native_weight(pre+'mlp.gate_proj.weight',(j,slice(None)));up=native_weight(pre+'mlp.up_proj.weight',(j,slice(None)))
    down=native_weight(pre+'mlp.down_proj.weight',(out,j));mx=s['mlp_x'][q]
    hb=hidden(case);E=decode(hb[0,token,k]);H=decode(hb[checkpoint,token,k])
    checkpoint_E=native_weight('model.embed_tokens.weight',(row['prompt_ids'][token],k))
    assert float(E)==float(checkpoint_E)
    selected_controls=[i for i,c in enumerate(controls()) if (c['layer'],c['kind'],c['output_row'],c['input_coordinate'])==(layer,kind,output_row,k)]
    scalar=scalar_record(case);effects=[e for e in scalar['conditions'] if e['control_index'] in selected_controls]
    changed=[e for e in scalar['natural_changed_generations'] if e['control_index'] in selected_controls]
    values={'actual_checkpoint_E_token_k':float(checkpoint_E),'H_checkpoint_token_k':float(H),
        'actual_W_kind_r_k':float(w[k]),'actual_normalized_attention_x_token_k':float(x[k]),'single_input_product':float(terms[k]),
        'full2560_input_product_sum':float(terms.sum()),'native_linear_output_r':float(linear),'native_linear_rounding_residual':float(linear-terms.sum()),
        'norm_observed_FP64_denominator':den,'actual_headnorm_gamma_d':float(gamma[nd]),'native_headnorm_d':float(norm[nd]),
        'headnorm_analysis_error_d':float(norm[nd]-z[nd]*gamma[nd]/den),
        'native_RoPE_d':float(rotated[nd]),'RoPE_partner_d':partner,'RoPE_partner_sign':rotate_sign,
        'actual_cos_d':float(cos),'actual_sin_d':float(sin),'RoPE_rounding_residual_d':float(rotated[nd]-expected_rope),
        'actual_P_query_head_source':float(p[source_token]),'actual_Wo_out_hd':float(wo[d]),
        'single_source_head_dimension_output_term':float(p[source_token]*V[source_token,d]*wo[d]),
        'single_source_head_all128_output_term':float(source_projection[source_token]),'all_sources_selected_head_output_term':float(source_projection.sum()),
        'actual_gate_j_k':float(gate[k]),'actual_up_j_k':float(up[k]),'actual_down_out_j':float(down),
        'native_mlp_x_query_k':float(mx[k]),'gate_input_term_k':float(gate[k]*mx[k]),'up_input_term_k':float(up[k]*mx[k]),
        'native_gate_j':float(s['gate'][q,j]),'native_up_j':float(s['up'][q,j]),'native_activation_j':float(s['mlp_a'][q,j]),
        'single_neuron_down_out_term':float(down*s['mlp_a'][q,j]),'native_all_neurons_down_out':float(s['mlp_down'][q,out])}
    assert all(np.isfinite(v) for v in values.values())
    traces={'projection_input':{'W':w.tolist(),'x':x.tolist(),'Wx':terms.tolist()},
        'headnorm':{'kind':norm_kind,'head':nh,'linear':z.tolist(),'gamma':gamma.tolist(),'native_norm':norm.tolist(),'native_RoPE':rotated.tolist()},
        'all_source_tokens':[{'position':t,'token_id':row['prompt_ids'][t],'token':row['token_strings'][t],
            'P':float(p[t]),'causal_allowed':t<=qt,'QK_all128_before_mask':float(Q@K[t]*s['scaling']),
            'V_d':float(V[t,d]),'selected_head_output_term':float(source_projection[t])} for t in range(n)]}
    return {'source_phase':2687,'scalar_phase':2689,'case_id':row['case_id'],'prompt':row['prompt'],'natural':natural_records()[case],
        'indices':{'case':case,'layer':layer,'kind':kind,'output_row':output_row,'input_coordinate':k,'token':token,'token_string':row['token_strings'][token],
            'query_position':q,'query_token':qt,'source_token':source_token,'head':head,'kv_head':kv,'head_coordinate':d,'checkpoint':checkpoint,'unit':j,'output_coordinate':out},
        'values':values,'traces':traces,'scalar_effects':effects,'changed_weight_natural':changed,'percase_changed_P_available':False,
        'boundary':'Actual learned BF16 checkpoint bits and native fields, FP64 products/denominators only analysis. E/H at selected token; MLP and P at body/task query. V has no headnorm: selecting V shows the corresponding K headnorm only, explicitly labelled. Headnorm uses output_row head; P uses separately selected query head. All2560input,128head andactualsources retained. Source terms do not include separately measured AV/Wo rounding. Scalar effects only for exact measured addresses; no invented percase changed P or MLP intervention. Fixed256 scoring and naturalcache are distinct protocols. No semantic closure.'}
