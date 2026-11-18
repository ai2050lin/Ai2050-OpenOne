"""Reuse the serial Phase 423 replay for several CPU-resident chunks per weight stream."""
import gc
import time

import torch
from torch.nn import functional as F

from shared.dist_protocol.mechanism_trace import GpuLease
from shared.dist_protocol.natural_register_runner import snapshot
import source_binding_design as D
import source_binding_sites as S

MODEL = 'qwen14'


def prepare(rows, arms, length, device):
    offset = [length-len(r['tokens']) for r in rows]
    start = min(offset[i]+min(S.GROUPS['names'](r)) for i,r in enumerate(rows))
    cache = {}

    def indices(site, donors):
        key = site, tuple(donors)
        if key not in cache:
            pairs = [(i,offset[i]+p,j,offset[j]+q) for i,j in enumerate(donors)
                     for p,q in zip(S.GROUPS[site](rows[i]),S.GROUPS[site](rows[j]),strict=True)]
            assert all(rows[i]['family']==rows[j]['family'] for i,_,j,_ in pairs)
            cache[key] = tuple(torch.tensor(a,device=device) for a in zip(*pairs))
        return cache[key]

    result = {'noop':dict(first=0,start=length-1,swap=None,edit={})}
    for name,arm in arms.items():
        if 'swap' in arm:
            site,block = arm['swap']
            result[name] = dict(first=block,start=start,swap=indices(site,arm['donor']),edit={})
        else:
            edit = {block:[(indices(site,arm['donor']),None if heads is None else torch.tensor(heads,device=device))
                           for sites,heads in actions for site in sites] for block,actions in arm['don'].items()}
            result[name] = dict(first=min(edit),start=length-1,swap=None,edit=edit)
    return result


def replay(rt, layer, block, h, pe, mask, plans, states, cap):
    n,length = h.shape[:2]
    hd,kv,heads = rt.hd,rt.kv,rt.heads
    repeat = rt.mod.repeat_kv
    grp = heads//kv
    a = layer.self_attn
    cos,sin = pe
    kb,vb = cap['k'],cap['v']
    kd = vd = None

    def rotate(x,c,s):
        return x*c + rt.mod.rotate_half(x)*s

    for name,x in states.items():
        plan = plans[name]
        start = plan['start']
        span = length-start
        xn = layer.input_layernorm(x)
        q = a.q_norm(a.q_proj(xn).view(n,span,heads,hd)).transpose(1,2)
        k = a.k_norm(torch.cat([kb[:,:start],a.k_proj(xn)],1).view(n,length,kv,hd)).transpose(1,2)
        v = torch.cat([vb[:,:start],a.v_proj(xn)],1).view(n,length,kv,hd).transpose(1,2)
        k,v = repeat(k,grp),repeat(v,grp)
        q = rotate(q,cos[:,None,start:],sin[:,None,start:])
        if block in plan['edit']:
            if kd is None:
                kd = repeat(a.k_norm(kb.view(n,length,kv,hd)).transpose(1,2),grp)
                vd = repeat(vb.view(n,length,kv,hd).transpose(1,2),grp)
            for (bi,bp,di,dp),hs in plan['edit'][block]:
                if hs is None:
                    k[bi,:,bp] = kd[di,:,dp]
                    v[bi,:,bp] = vd[di,:,dp]
                else:
                    k[bi[:,None],hs[None],bp[:,None]] = kd[di[:,None],hs[None],dp[:,None]]
                    v[bi[:,None],hs[None],bp[:,None]] = vd[di[:,None],hs[None],dp[:,None]]
        o = F.scaled_dot_product_attention(q,rotate(k,cos[:,None],sin[:,None]),v,attn_mask=mask[:,:,start:,:])
        x = x + a.o_proj(o.transpose(1,2).reshape(n,span,-1))
        states[name] = x + layer.mlp(layer.post_attention_layernorm(x))


def select_scores(logits, rows):
    ids = torch.tensor([r['choice_ids'] for r in rows],device=logits.device)
    values = logits.gather(1,ids).cpu()
    best = values.argmax(1).tolist()
    top = logits.argmax(1).tolist()
    return [dict(id=r['id'],pred_record=choice,top1=first,scores=score,forced_correct=choice==r['target_record'],
                 top1_correct=first==r['gold_id']) for r,choice,first,score in zip(rows,best,top,values.tolist())]


def run(chunks, runtime, make_arms, stage, output, after):
    identity = snapshot(D.MODELS[MODEL])
    assert identity == D.read(D.OUT/f'{MODEL}-identity.json')['model_snapshot']
    S.OUT = output
    started = time.monotonic()
    with GpuLease(), torch.inference_mode():
        torch.cuda.reset_peak_memory_stats()
        rt = runtime(MODEL)
        batches = []
        total = 40*sum(map(len,chunks))
        S.log(stage,0,total,started,rows=total//40)
        for rows in chunks:
            ids,pos,pe,mask = rt.context(rows)
            plan = prepare(rows,make_arms(rows),ids.shape[1],ids.device) if make_arms else {}
            batches.append(dict(h=rt.embed(ids).cpu(),pos=pos,pe=pe,mask=mask,states={},plan=plan))
        for block in range(40):
            layer = rt.block(block)
            for batch in batches:
                h = batch['h'].cuda()
                states = {name:value.cuda() for name,value in batch['states'].items()}
                for name,plan in batch['plan'].items():
                    if block==plan['first']:
                        start = plan['start']
                        state = h[:,start:].clone()
                        if plan['swap'] is not None:
                            bi,bp,di,dp = plan['swap']
                            state[bi,bp-start] = h[di,dp]
                        states[name] = state
                cap = {}
                handles = []
                if states:
                    a = layer.self_attn
                    handles = [a.k_proj.register_forward_hook(lambda m,i,o:cap.__setitem__('k',o)),
                               a.v_proj.register_forward_hook(lambda m,i,o:cap.__setitem__('v',o))]
                try:
                    y = layer(h,attention_mask=batch['mask'],position_ids=batch['pos'],position_embeddings=batch['pe'],use_cache=False)
                finally:
                    for handle in handles:
                        handle.remove()
                if states:
                    replay(rt,layer,block,h,batch['pe'],batch['mask'],batch['plan'],states,cap)
                batch['h'] = (y if block<39 else y[:,-1]).cpu()
                batch['states'] = {name:(value if block<39 else value[:,-1]).cpu() for name,value in states.items()}
                del h,y,states,cap
            del layer
            gc.collect()
            torch.cuda.empty_cache()
            if (block+1)%2==0:
                after()
                S.log(stage,(block+1)*total//40,total,started,block=block,arms=len(batch['states']))
        norm = rt.Norm(rt.dim,eps=rt.cfg.rms_norm_eps).to(device='cuda',dtype=torch.bfloat16)
        norm.weight = torch.nn.Parameter(rt.weight('model.norm.weight'),requires_grad=False)
        weight = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
        output = []
        for rows,batch in zip(chunks,batches):
            output.append({name:select_scores((norm(hidden.cuda())@weight.T).float(),rows)
                           for name,hidden in {'base':batch['h'],**batch['states']}.items()})
        meta = dict(peak_MiB=round(torch.cuda.max_memory_allocated()/2**20),seconds=round(time.monotonic()-started,1),
                    chunk_count=len(chunks))
    return output,meta
