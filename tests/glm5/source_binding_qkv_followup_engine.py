"""Replay the original BF16 suffix computation with independently replaceable Q, K and V."""
from collections import defaultdict
import gc
import itertools
import time

import torch
from torch.nn import functional as F

from shared.dist_protocol.mechanism_trace import GpuLease
from deploy.storage_policy import check_write
from source_binding_readheads import H16
from source_binding_runtime import Runtime
from source_binding_transfer_factor_engine import select_scores
import source_binding_sites as sites

WINDOW = (23, 24, 25)
NAMES = ('qsrc','qent','names')
KINDS = ('W','H','K','V','QH','D','Dself','ALL','Hpos','Hback')


def make_batch(rt, rows):
    ids,pos,rope,mask = rt.context(rows)
    length = ids.shape[1]
    left = [length-len(r['tokens']) for r in rows]
    index = {(r['original_id'],r['cue']):i for i,r in enumerate(rows)}
    cross = [index[r['original_id'],1-r['cue']] for r in rows]
    donors = [index[r['donor'],r['cue']] for r in rows]
    assert all(rows[i]['family']==rows[j]['family'] for i,j in enumerate(donors))
    assert all(cross[j]==donors[cross[i]] for i,j in enumerate(donors))
    device = ids.device
    tensor = lambda a: torch.tensor(a,device=device,dtype=torch.long)

    def positions(r, site):
        if site in NAMES:
            return sites.GROUPS[site](r)
        names = set(sites.GROUPS['names'](r))
        stop = r['tokens'].index(rt.tok.convert_tokens_to_ids('<|im_end|>'))
        return [p for p in range(min(names),stop) if p not in names]

    maps = {}
    for site in (*NAMES,'holders'):
        pairs = [(i,left[i]+p,j,left[j]+q) for i,j in enumerate(donors)
                 for p,q in zip(positions(rows[i],site),positions(rows[j],site),strict=True)]
        maps[site] = tuple(tensor(x) for x in zip(*pairs))
    start = min(left[i]+min(sites.GROUPS['names'](r)) for i,r in enumerate(rows))
    h = rt.embed(ids).cpu()
    all_name_rows = [(i,left[i]+p) for i,r in enumerate(rows) for p in sites.GROUPS['names'](r)]
    name_map = tuple(tensor(x) for x in zip(*all_name_rows))
    pairs = [(i,left[i]+p,left[cross[i]]+p) for i,r in enumerate(rows) if r['cue']==0 for p in sites.GROUPS['names'](r)]
    audit = tuple(tensor(x) for x in zip(*pairs))
    answer_cue = tensor([r['cue'] for r in rows])==0
    # Remap just the edited key tokens to an endpoint distance of 22; keep the
    # separation between sub-tokens by applying one positional shift per name.
    hpos = pos.clone()
    for i,r in enumerate(rows):
        for site in ('qsrc','qent'):
            target = [left[i]+p for p in r['up'][site]]
            end = pos[i,-1] - 22
            hpos[i,target] += end-pos[i,target[-1]]
    rotary = rt.Rotary(rt.cfg,device=device)
    rope_pos = rotary(torch.zeros((*ids.shape,1),device=device,dtype=torch.bfloat16),hpos)
    return dict(rows=rows,h=h,pos=pos,rope=rope,rope_pos=rope_pos,mask=mask,cross=tensor(cross),
                maps=maps,name_map=name_map,audit=audit,start=start,answers=answer_cue,
                states={},diagnostics={},name_invariance=[],dself_exact=True)


def query(rt, layer, x):
    a = layer.self_attn
    norm = layer.input_layernorm(x)
    return a.q_norm(a.q_proj(norm).view(len(x),x.shape[1],rt.heads,rt.hd)).transpose(1,2)


def change(k, v, dk, dv, mapping, heads, use_k, use_v):
    bi,bp,di,dp = mapping
    index = (bi,slice(None),bp) if heads is None else (bi[:,None],heads[None],bp[:,None])
    source = (di,slice(None),dp) if heads is None else (di[:,None],heads[None],dp[:,None])
    if use_k:
        k[index] = dk[source]
    if use_v:
        v[index] = dv[source]


def layer_step(rt, layer, block, batch):
    h = batch['h'].cuda()
    states = {k:v.cuda() for k,v in batch['states'].items()}
    a = layer.self_attn
    n,length = h.shape[:2]
    cos,sin = batch['rope']
    cross = batch['cross']
    start = batch['start']
    last = length-1
    kv, hd, heads = rt.kv,rt.hd,rt.heads
    grp = heads//kv
    repeat = rt.mod.repeat_kv
    at = torch.as_tensor(H16.get(block,[]),device=h.device,dtype=torch.long)
    selected = len(at)>0

    def rot(x,c,s):
        return x*c + rt.mod.rotate_half(x)*s

    if block==0:
        states['noop'] = h[:,last:].clone()
    if block==23:
        for label in ('Q','self'):
            states[label] = h[:,last:].clone()
    if block in WINDOW:
        for site in NAMES:
            cell = f'{site}@{block}'
            for kind in KINDS:
                if kind in ('D','Dself'):
                    continue
                begin = start if kind=='W' else last
                x = h[:,begin:].clone()
                if kind=='W':
                    bi,bp,di,dp = batch['maps'][site]
                    x[bi,bp-start] = h[di,dp]
                states[f'{kind}|{cell}'] = x
    if block==28:
        for site,b in itertools.product(NAMES,WINDOW):
            cell = f'{site}@{b}'
            source = states['H|'+cell]
            delta = source.float()-h[:,last:].float()
            states['D|'+cell] = (h[:,last:].float()+delta[cross]).to(h.dtype)
            same = (h[:,last:].float()+delta).to(h.dtype)
            batch['dself_exact'] &= bool(torch.equal(source,same))
            states['Dself|'+cell] = same
    cap = {}
    hooks = [module.register_forward_hook(lambda _,args,out,key=key: cap.__setitem__(key,out))
             for key,module in (('k',a.k_proj),('v',a.v_proj))]
    try:
        y = layer(h,attention_mask=batch['mask'],position_ids=batch['pos'],position_embeddings=(cos,sin),use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    kb,vb = cap['k'],cap['v']
    kd = vd = qb = None
    if selected:
        ar,ap,cp = batch['audit']
        invariant = [(kb[ar,ap]-kb[cross[ar],cp]).abs().max().item(), (vb[ar,ap]-vb[cross[ar],cp]).abs().max().item()]
        batch['name_invariance'].append([block,*invariant])
        kd = repeat(a.k_norm(kb.view(n,length,kv,hd)).transpose(1,2),grp)
        vd = repeat(vb.view(n,length,kv,hd).transpose(1,2),grp)
        qb = query(rt,layer,h[:,last:])
    q_cache = {}
    if selected:
        q_cache = {k[2:]:query(rt,layer,x) for k,x in states.items() if k.startswith('H|')}
        ni,np_ = batch['name_map']
        name_mask = torch.zeros((n,length),device=h.device,dtype=torch.bool)
        name_mask[ni,np_] = True
    for key,x in states.items():
        label,_,cell = key.partition('|')
        site = cell.split('@')[0] if cell else 'names'
        begin = start if label=='W' else last
        span = length-begin
        xn = layer.input_layernorm(x)
        q = q_cache[cell].clone() if selected and label=='H' else a.q_norm(a.q_proj(xn).view(n,span,heads,hd)).transpose(1,2)
        k = a.k_norm(torch.cat([kb[:,:begin],a.k_proj(xn)],1).view(n,length,kv,hd)).transpose(1,2)
        v = torch.cat([vb[:,:begin],a.v_proj(xn)],1).view(n,length,kv,hd).transpose(1,2)
        k,v = repeat(k,grp),repeat(v,grp)
        if selected and label in ('Q','QH'):
            q[:,at] = (qb if label=='Q' else q_cache[cell])[cross][:,at]
        if selected and label in ('H','K','V','QH','ALL','Hpos','Hback','self'):
            mapping = batch['maps'][site]
            if label=='self':
                bi,bp,_,_ = mapping
                mapping = bi,bp,bi,bp
            change(k,v,kd,vd,mapping,None if label=='ALL' else at,label!='V',label!='K')
            if label=='Hback':
                holder_map = batch['maps']['holders']
                holder_map = tuple(col[batch['answers'][holder_map[0]]] for col in holder_map)
                change(k,v,kd,vd,holder_map,at,True,True)
        q = rot(q,cos[:,None,begin:],sin[:,None,begin:])
        if selected and label=='Hpos':
            bi,bp,_,_ = batch['maps'][site]
            cp,sp = batch['rope_pos']
            i,j,p = bi[:,None],at[None],bp[:,None]
            new = rot(k[i,j,p],cp[bi,bp][:,None],sp[bi,bp][:,None])
            k = rot(k,cos[:,None],sin[:,None])
            k[i,j,p] = new
        else:
            k = rot(k,cos[:,None],sin[:,None])
        allowed = batch['mask'][:,:,begin:,:]
        o = F.scaled_dot_product_attention(q,k,v,attn_mask=allowed)
        if selected and label in ('H','K','V','QH','Hpos','Q','noop'):
            score = (q[:,at].float()@k[:,at].float().transpose(-2,-1)).squeeze(-2)/(hd**.5)
            score = score.masked_fill(~allowed[:,0,-1,None,:],float('-inf'))
            weights = score.softmax(-1)
            mass = (weights*name_mask[:,None]).sum(-1)
            choice = weights.argmax(-1)
            on_name = name_mask.gather(1,choice)
            norm = o[:,at,0].float().norm(dim=-1)
            batch['diagnostics'][f'{block}|{key}'] = torch.stack([mass.mean(-1),on_name.float().mean(-1),norm.mean(-1)],-1).cpu().tolist()
        x = x + a.o_proj(o.transpose(1,2).reshape(n,span,-1))
        states[key] = x + layer.mlp(layer.post_attention_layernorm(x))
    batch['h'] = (y if block<39 else y[:,-1]).cpu()
    batch['states'] = {name:(v if block<39 else v[:,-1]).cpu() for name,v in states.items()}


def checked_outputs(rt, batches):
    norm = rt.Norm(rt.dim,eps=rt.cfg.rms_norm_eps).to(device='cuda',dtype=torch.bfloat16)
    norm.weight = torch.nn.Parameter(rt.weight('model.norm.weight'),requires_grad=False)
    weight = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
    groups = []
    behavior = defaultdict(list)
    errors = defaultdict(float)
    choices = defaultdict(int)
    inv = 0.0
    dself_exact = True
    for batch in batches:
        rows = batch['rows']
        results = {k:select_scores((norm(h.cuda())@weight.T).float(),rows) for k,h in {'base':batch['h'],**batch['states']}.items()}
        for row,out in zip(rows,results['base']):
            behavior[str(row['cue'])].append(out['forced_correct'])
        tests = [('noop','base'),('self','base')] + [(f'Dself|{s}@{b}',f'H|{s}@{b}') for s,b in itertools.product(NAMES,WINDOW)]
        for a,b in tests:
            label = a.split('|')[0]
            for x,y in zip(results[a],results[b]):
                errors[label] = max(errors[label],max(abs(p-q) for p,q in zip(x['scores'],y['scores'])))
                choices[label] += x['pred_record'] != y['pred_record']
        inv = max(inv,max(max(k,v) for _,k,v in batch['name_invariance']))
        dself_exact &= batch['dself_exact']
        groups.append(dict(ids=[r['id'] for r in rows],items={k:[[x['pred_record'],*x['scores']] for x in v] for k,v in results.items()},
                           diagnostics=batch['diagnostics'],name_invariance=batch['name_invariance']))
    accuracy = {k:sum(v)/len(v) for k,v in behavior.items()}
    ok = all(x>=.95 for x in accuracy.values()) and all(v<=1 for v in errors.values()) and not any(choices.values()) and inv==0 and dself_exact
    return groups,dict(passed=ok,behavior=accuracy,max_score_difference=errors,choice_difference=choices,
                       name_KV_max_cue_difference=inv,dself_exact=dself_exact)


def run(chunks, output, stage, model_snapshot):
    from shared.dist_protocol.natural_register_runner import snapshot
    import source_binding_design as d

    assert snapshot(d.MODELS['qwen14'])==model_snapshot
    sites.OUT = output
    started = time.monotonic()
    total = 40*sum(map(len,chunks))
    with GpuLease(), torch.inference_mode():
        torch.cuda.reset_peak_memory_stats()
        rt = Runtime('qwen14')
        sites.log(stage,0,total,started)
        batches = [make_batch(rt,rs) for rs in chunks]
        for block in range(40):
            layer = rt.block(block)
            for batch in batches:
                layer_step(rt,layer,block,batch)
            del layer
            gc.collect()
            torch.cuda.empty_cache()
            if (block+1)%2==0:
                check_write(output,2**30,operation=stage)
                sites.log(stage,(block+1)*total//40,total,started,block=block,arms=len(batches[0]['states']))
        groups,controls = checked_outputs(rt,batches)
        meta = dict(peak_MiB=round(torch.cuda.max_memory_allocated()/2**20),seconds=round(time.monotonic()-started,1))
    return dict(groups=groups,controls=controls,meta=meta)
