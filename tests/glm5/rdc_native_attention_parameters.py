"""Exact original Q/K/V/O factor addresses and explicitly conditioned products."""
import importlib
from functools import lru_cache
import numpy as np
from rdc_construction_common import ROOT,MODELS
from rdc_construction_parameters import catalog
from rdc_relation_native_parameters import parameter,decode


def factors(key,block,head):
    meta=catalog(key);c=meta['config']
    assert 0<=block<c['num_hidden_layers'] and 0<=head<c['num_attention_heads']
    d=c.get('head_dim',c['hidden_size']//c['num_attention_heads'])
    group=c['num_attention_heads']//c['num_key_value_heads'];kv_head=head//group
    prefix=f'model.layers.{block}.self_attn.'
    names={r['name'] for r in meta['parameters']}
    def raw(name):
        return parameter(ROOT,prefix+name,MODELS[key])
    qslice=slice(head*d,(head+1)*d);kslice=slice(kv_head*d,(kv_head+1)*d)
    result={'Wq':decode(raw('q_proj.weight')[qslice]).astype(float),
        'Wk':decode(raw('k_proj.weight')[kslice]).astype(float),
        'Wv':decode(raw('v_proj.weight')[kslice]).astype(float),
        'Wo':decode(raw('o_proj.weight')[:,qslice]).astype(float)}
    for role,sl in [('q',qslice),('k',kslice),('v',kslice)]:
        name=role+'_proj.bias'
        result['b'+role]=decode(raw(name)[sl]).astype(float) if prefix+name in names else np.zeros(d)
    for role in ['q','k']:
        name=role+'_norm.weight'
        result['gamma_'+role]=decode(raw(name)).astype(float) if prefix+name in names else np.ones(d)
    result['metadata']={'model':key,'block':block,'query_head':head,'kv_head':kv_head,
        'GQA_group_size':group,'head_dim':d,'hidden_size':c['hidden_size'],
        'has_qk_norm':prefix+'q_norm.weight' in names,'rms_norm_eps':c['rms_norm_eps'],
        'parameter_prefix':prefix,'original_config_sha256':meta['config_sha256'],
        'scope':'Complete original head-factor vectors/matrices, in original coordinate order. Input means the actual input-layer-normalized residual; residual RMSNorm is not silently folded into fixed weights.'}
    return result


@lru_cache(maxsize=3)
def rotary_module(key):
    from transformers import AutoConfig
    config=AutoConfig.from_pretrained(ROOT/'models/hf'/MODELS[key],local_files_only=True)
    module=importlib.import_module('transformers.models.'+config.model_type+'.modeling_'+config.model_type)
    name='Qwen3RotaryEmbedding' if config.model_type=='qwen3' else 'GlmRotaryEmbedding'
    return config,getattr(module,name)(config,device='cpu'),module.apply_rotary_pos_emb


def rotary_matrix(key,position):
    """Linear map with the native CPU-generated BF16 cosine/sine values.

    Products are evaluated in FP64 without intermediate BF16 rounding. This is
    an algebraic coefficient, not a bit-exact replacement for the native kernel.
    """
    import torch
    assert position>=0
    config,rotary,apply=rotary_module(key);d=config.head_dim
    dummy=torch.zeros((1,1,d),dtype=torch.bfloat16)
    cos,sin=rotary(dummy,torch.tensor([[position]],dtype=torch.int64))
    basis=torch.eye(d,dtype=torch.float64)[None,None]
    result,_=apply(basis,basis,cos.double(),sin.double())
    return result[0,0].numpy().T


def entries(key,block,head,input_i,input_r,output_j,query_position=0,key_position=0):
    f=factors(key,block,head);m=f['metadata'];d=m['head_dim'];width=m['hidden_size']
    assert min(input_i,input_r,output_j)>=0 and max(input_i,input_r,output_j)<width
    rq=rotary_matrix(key,query_position);rk=rotary_matrix(key,key_position)
    q=f['gamma_q']*f['Wq'][:,input_i];k=f['gamma_k']*f['Wk'][:,input_r]
    qr,kr=rq@q,rk@k
    qk_terms=qr*kr/np.sqrt(d)
    ov_terms=f['Wo'][output_j]*f['Wv'][:,input_i]
    return {'metadata':m,'input_i':input_i,'input_r':input_r,'output_j':output_j,
        'query_position':query_position,'key_position':key_position,
        'qk_numerator_coefficient':float(qk_terms.sum()),'ov_coefficient':float(ov_terms.sum()),
        'qk_all_head_component_terms':qk_terms,'ov_all_head_component_terms':ov_terms,
        'read_write_factors':np.stack([f['Wq'][:,input_i],f['Wk'][:,input_r],f['Wv'][:,input_i],f['Wo'][output_j],
            f['gamma_q'],f['gamma_k'],f['bq'],f['bk'],f['bv']]),
        'scope':'QK numerator includes native RoPE layout and norm gains. Qwen requires division by both state-dependent head RMS denominators. GLM also has affine bias terms. Neither scalar is an attention probability or a semantic edge. Native BF16 intermediate rounding is not included in this FP64 factor contraction.'}


def conditioned_score(f,xq,xk,rq,rk):
    """Same-valued-weight smooth algebra, including bias and dynamic QK norms."""
    q=f['Wq']@xq+f['bq'];k=f['Wk']@xk+f['bk'];m=f['metadata']
    sq=sk=1.0
    if m['has_qk_norm']:
        sq=np.sqrt(np.mean(q*q)+m['rms_norm_eps']);sk=np.sqrt(np.mean(k*k)+m['rms_norm_eps'])
    q=f['gamma_q']*q/sq;k=f['gamma_k']*k/sk
    return float((rq@q)@(rk@k)/np.sqrt(m['head_dim'])),float(sq),float(sk)
