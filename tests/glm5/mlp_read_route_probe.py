"""Measure the native answer-query reads and freeze candidate writers on discovery only."""
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

import mlp_read_route_common as C
from mlp_read_route_materials import ROLES, WORDS
from mlp_kv_behavior import word_indices
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease

LAYERS=[34,35]
WRITERS=[33,34]


def inputs():
    path=C.OUT/'inputs.json.gz';plan=C.read(C.OUT/'prereg.json')
    assert C.file_sha(path)==plan['inputs_sha256']
    return C.read_gz(path)['rows'],plan


def batch_ids(rows):return torch.tensor([r['tokens'] for r in rows],device='cuda')


def positions(rows,roles):
    return [sorted({a['position'] for role in roles for a in r['anchors'][role]}) for r in rows]


def token_indices(pos,device):
    return (torch.tensor([j for j,v in enumerate(pos) for _ in v],device=device),
            torch.tensor([p for v in pos for p in v],device=device))


@contextmanager
def cache_patch(cache, edits):
    """Overwrites use [batch, GQA, token, channel] slices and restore even after decoding."""
    assert cache.get_seq_length()==C.CUT
    copies=[]
    try:
        for b,kind,index,value in edits:
            dst=getattr(cache.layers[b],kind)
            copies.append((b,kind,index,dst[index].clone()))
            dst[index]=value
        yield
    finally:
        cache.crop(C.CUT)
        for b,kind,index,value in reversed(copies):getattr(cache.layers[b],kind)[index]=value


def answer_read(model,x,cache,edits=(),layers=LAYERS):
    """Read native Q and SDPA output at the actual assistant boundary."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    fields={b:{} for b in layers};hooks=[]
    for b in layers:
        attn=model.model.layers[b].self_attn
        assert attn.sliding_window is None
        def pre(_,args,kwargs,b=b):fields[b]['rope']=kwargs['position_embeddings']
        def query(_,args,value,b=b):fields[b]['q']=value[:,-1:].transpose(1,2).detach().clone()
        def head(_,args,b=b):fields[b]['z']=args[0][:,-1].reshape(len(x),32,128).detach().clone()
        hooks.extend([attn.register_forward_pre_hook(pre,with_kwargs=True),attn.q_norm.register_forward_hook(query),
                      attn.o_proj.register_forward_pre_hook(head)])
    try:
        with cache_patch(cache,edits):
            out=model(input_ids=x[:,C.CUT:],past_key_values=cache,use_cache=True,logits_to_keep=1)
            for b in layers:
                field=fields[b];cos,sin=field.pop('rope');q=field['q']
                q,_=apply_rotary_pos_emb(q,q,cos[:,-1:],sin[:,-1:]);q=q[:,:,0].float()
                layer=out.past_key_values.layers[b]
                k=layer.keys.float().repeat_interleave(4,dim=1);v=layer.values.float().repeat_interleave(4,dim=1)
                weights=torch.einsum('bhd,bhtd->bht',q,k).mul_(128**-.5).softmax(-1)
                prediction=torch.einsum('bht,bhtd->bhd',weights,v)
                field.update(q=q,weights=weights,v=v,error=(prediction-field['z'].float()).flatten(1).norm(dim=-1)/field['z'].float().flatten(1).norm(dim=-1).clamp_min(1e-30))
            return out.logits[:,-1].float(),fields
    finally:
        for h in hooks:h.remove()


def gram(model):
    result={}
    for b in LAYERS:
        w=model.model.layers[b].self_attn.o_proj.weight.detach().float().reshape(2560,32,128).permute(1,2,0).contiguous()
        result[b]=w@w.transpose(-1,-2)
    return result


def read_statistics(rows,fields,metric):
    stats=[]
    for j,r in enumerate(rows):
        by_role={}
        for role in ROLES:
            role_pos=[a['position'] for a in r['anchors'][role]]
            layers={}
            for b,f in fields.items():
                index=torch.tensor(role_pos,device=f['weights'].device)
                weights=f['weights'][j].index_select(-1,index);v=f['v'][j].index_select(-2,index)
                norms=torch.einsum('hpd,hde,hpe->hp',v,metric[b],v).clamp_min(0).sqrt()
                reads=weights*norms
                layers[str(b)]={'mass_by_head':weights.sum(-1).cpu().tolist(),
                                'projected_norm_by_head':reads.sum(-1).cpu().tolist(),
                                'per_token_mass':weights.cpu().tolist(),
                                'per_token_projected_norm':reads.cpu().tolist()}
            by_role[role]=layers
        stats.append({'id':r['id'],'anchors':r['anchors'],'roles':by_role,
                      'native_reconstruction_relative_error':{str(b):float(f['error'][j]) for b,f in fields.items()}})
    return stats


def gradient_batch(model,rows,roles,words):
    """Zero-valued leaves preserve native forward and upstream gradients."""
    x=batch_ids(rows);i,j=token_indices(positions(rows,roles),x.device);saved={};hooks=[]
    def attach(value,b,kind):
        true=value[i,j];leaf=torch.zeros_like(true,requires_grad=True)
        modified=value.clone();modified[i,j]=true+leaf
        saved[b,kind]=(leaf,true.detach());return modified
    for b in WRITERS:
        l=model.model.layers[b]
        hooks.append(l.self_attn.o_proj.register_forward_pre_hook(lambda _,args,b=b:(attach(args[0],b,'head'),*args[1:])))
        hooks.append(l.mlp.register_forward_hook(lambda _,args,out,b=b:attach(out,b,'mlp')))
    try:
        logits=model(input_ids=x,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        gaps=torch.stack([logits[k,words[r['expected']]].max()-logits[k,words[r['contrast']]].max() for k,r in enumerate(rows)])
        grads=torch.autograd.grad(gaps.sum(),[v[0] for v in saved.values()])
        assert all(torch.isfinite(v).all() for v in grads)
        result={}
        for ((b,kind),(_,true)),grad in zip(saved.items(),grads):
            product=grad.float()*true.float()
            if kind=='head':product=product.reshape(-1,32,128).sum(-1)
            else:product=product.sum(-1,keepdim=True)
            per=torch.stack([product[i==k].sum(0) for k in range(len(rows))]).cpu().tolist()
            result[b,kind]=per
        return result
    finally:
        for h in hooks:h.remove()


def main():
    output=C.OUT/'route-freeze.json'
    if output.exists():return
    all_rows,plan=inputs();rows=[r for r in all_rows if r['split']=='discovery'];assert len(rows)==C.N_DISCOVERY
    prereg={**C.identity(),'schema':'mlp_early_token_discovery_execution.v1','inputs_sha256':plan['inputs_sha256'],
            'source_sha256':C.file_sha(__file__),'ids':[r['id'] for r in rows],'reader_layers':LAYERS,'candidate_writer_layers':WRITERS}
    C.save(C.OUT/'discovery-execution.json',prereg)
    torch.set_num_threads(4);readings=[];weights=defaultdict(list);derivative=defaultdict(list)
    with GpuLease():
        model,tok,eos=load_model();words=word_indices(tok);metric=gram(model)
        with torch.inference_mode():
            for lo in range(0,len(rows),4):
                subset=rows[lo:lo+4];x=batch_ids(subset)
                cache=model(input_ids=x[:,:C.CUT],use_cache=True,logits_to_keep=1).past_key_values
                _,fields=answer_read(model,x,cache)
                readings.extend(read_statistics(subset,fields,metric))
                for b,f in fields.items():weights[b].append(f['weights'].cpu().numpy())
                del fields,cache
        scores={r:float(np.mean([sum(sum(layer['projected_norm_by_head']) for layer in row['roles'][r].values()) for row in readings])) for r in ROLES}
        roles=sorted(ROLES,key=lambda r:(-scores[r],ROLES.index(r)))[:2]
        for lo in range(0,len(rows),2):
            for key,values in gradient_batch(model,rows[lo:lo+2],roles,words).items():derivative[key].extend(values)
            if (lo+2)%16==0:print('discovery derivative',lo+2,'/',len(rows),flush=True)
    raw=C.OUT/'discovery-reads.npz';np.savez_compressed(raw,**{f'B{b}_weights':np.concatenate(v) for b,v in weights.items()})
    head_ranking=sorted([(float(s),b,h) for (b,kind),v in derivative.items() if kind=='head' for h,s in enumerate(np.mean(v,axis=0))],key=lambda x:(-x[0],x[1],x[2]))
    mlp_ranking=sorted([(float(np.mean(v)),b) for (b,kind),v in derivative.items() if kind=='mlp'],key=lambda x:(-x[0],x[1]))
    heads=[list(v[1:]) for v in head_ranking[:2]];mlps=[mlp_ranking[0][1]]
    offsets=sorted({int(np.median([1033-p['position'] for r in rows for p in r['anchors'][role]])) for role in roles})
    selected={**C.identity(),'schema':'mlp_early_token_route_freeze.v1','prereg_sha256':C.file_sha(C.OUT/'prereg.json'),
              'discovery_execution_sha256':C.file_sha(C.OUT/'discovery-execution.json'),'roles':roles,'role_scores':scores,
              'heads':heads,'mlps':mlps,'capability_offsets':offsets,'ranked_heads':head_ranking,'ranked_mlps':mlp_ranking,
              'reader_layers':LAYERS,'discovery_n':len(rows),'raw_weight_sha256':C.file_sha(raw),
              'discovery_max_native_error':max(v for r in readings for v in r['native_reconstruction_relative_error'].values())}
    C.write_gz(C.OUT/'discovery-readings.json.gz',{'rows':readings,'derivative':{f'{b}:{k}':v for (b,k),v in derivative.items()}})
    C.save(output,selected)
    print('route frozen',roles,heads,mlps,'attention reconstruction max relative error',selected['discovery_max_native_error'],flush=True)


if __name__=='__main__':main()
