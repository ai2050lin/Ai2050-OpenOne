"""Use the Phase 423 BF16 suffix replay with sparse attention-edge interventions."""
import torch
from torch.nn import functional as F

import source_binding_transfer_factor_engine as serial


def prepare(rows, arms, length, device):
    offset = [length-len(r['tokens']) for r in rows]
    cache = {}

    def mapping(group, donors):
        key = group, tuple(donors)
        if key not in cache:
            pairs = []
            for i, j in enumerate(donors):
                assert rows[i]['family'] == rows[j]['family']
                pairs.extend((i,offset[i]+a,j,offset[j]+b) for a,b in
                             zip(rows[i]['groups'][group],rows[j]['groups'][group],strict=True))
            cols = list(zip(*pairs)) if pairs else [()] * 4
            cache[key] = tuple(torch.tensor(col,device=device,dtype=torch.long) for col in cols)
        return cache[key]

    plans = {'noop':dict(first=0,start=length-1,swap=None,edits={})}
    for name,arm in arms.items():
        edits = {block: [(action,mapping(group,arm['donor']),torch.tensor(heads,device=device,dtype=torch.long))
                         for action,group in arm['actions']] for block,heads in arm['heads'].items()}
        plans[name] = dict(first=min(edits),start=length-1,swap=None,edits=edits)
    return plans


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
        start = length-1
        xn = layer.input_layernorm(x)
        q = a.q_norm(a.q_proj(xn).view(n,1,heads,hd)).transpose(1,2)
        k = a.k_norm(torch.cat([kb[:,:start],a.k_proj(xn)],1).view(n,length,kv,hd)).transpose(1,2)
        v = torch.cat([vb[:,:start],a.v_proj(xn)],1).view(n,length,kv,hd).transpose(1,2)
        k,v = repeat(k,grp),repeat(v,grp)
        q = rotate(q,cos[:,None,start:],sin[:,None,start:])
        allowed = mask[:,:,start:,:]
        altered_mask = False
        for action,(bi,bp,di,dp),hs in plans[name]['edits'].get(block,()):
            if not len(bi):
                continue
            dst = bi[:,None],hs[None],bp[:,None]
            src = di[:,None],hs[None],dp[:,None]
            if action == 'mask':
                if not altered_mask:
                    allowed = allowed.expand(-1,heads,-1,-1).clone()
                    altered_mask = True
                allowed[bi[:,None],hs[None],0,bp[:,None]] = False
                continue
            if action == 'zero':
                v[dst] = 0
                continue
            if vd is None:
                vd = repeat(vb.view(n,length,kv,hd).transpose(1,2),grp)
            if action == 'kv':
                if kd is None:
                    kd = repeat(a.k_norm(kb.view(n,length,kv,hd)).transpose(1,2),grp)
                k[dst] = kd[src]
            v[dst] = vd[src if action in ('kv','donor_v') else dst]
        o = F.scaled_dot_product_attention(q,rotate(k,cos[:,None],sin[:,None]),v,attn_mask=allowed)
        x = x + a.o_proj(o.transpose(1,2).reshape(n,1,-1))
        states[name] = x + layer.mlp(layer.post_attention_layernorm(x))


def run(chunks, arms, stage, output, check):
    prepare_original,replay_original = serial.prepare,serial.replay
    serial.prepare,serial.replay = prepare,replay
    try:
        return serial.run(chunks,serial.S.Runtime,arms,stage,output,check)
    finally:
        serial.prepare,serial.replay = prepare_original,replay_original
