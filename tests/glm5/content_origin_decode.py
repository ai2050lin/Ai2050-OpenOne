"""Greedy, unrestricted next-token decoding on the original streamed BF16 model."""
from itertools import batched
import time
import kv_bf_q14 as native

PANEL = 16
SLAB = 256
WIDTH = 88
CAP = 64


def prefill(model, rows):
    t = model.t
    ids = [r["ids"] for r in rows]
    assert max(map(len, ids)) <= WIDTH
    pad = t.tensor([WIDTH-len(x) for x in ids], device="cuda")
    tokens = t.tensor([[model.tok.pad_token_id]*(WIDTH-len(x))+x for x in ids], device="cuda")
    valid = t.arange(WIDTH, device="cuda")[None] >= pad[:,None]
    pos = (valid.cumsum(1)-1).masked_fill(~valid, 0)
    mask = valid[:,None,None,:] & t.ones((WIDTH,WIDTH),device="cuda",dtype=t.bool).tril()
    hidden = model.decoder.embed_tokens(tokens)
    return dict(rows=rows, hidden=hidden, rope=model.rope(hidden,pos), mask=mask, valid=valid,
                length=pos[:,-1]+1, cache={}, emitted=[], alive=t.ones(len(rows),device="cuda",dtype=t.bool))


def greedy(model, rows, progress, shadow=False):
    t = model.t
    panels = [prefill(model, p) for p in batched(rows,PANEL)]
    eos = model.model.generation_config.eos_token_id
    ends = t.tensor(sorted(set(([eos] if isinstance(eos,int) else eos)+[model.tok.eos_token_id])),device="cuda")
    for step in range(CAP):
        active = [p for p in panels if bool(p["alive"].any())]
        if not active:
            break
        if step:
            for p in active:
                x = model.decoder.embed_tokens(p["emitted"][-1][:,None])
                p["hidden"],p["rope"] = x,model.rope(x,(p["length"]+step-1)[:,None])
                valid = t.cat((p["valid"],t.ones((len(x),step),device="cuda",dtype=t.bool)),dim=1)
                p["mask"] = valid[:,None,None,:]
        for layer in range(40):
            with model.block(layer) as block:
                for p in active:
                    x = p["hidden"]
                    q,k,v = model.qkv(block,x,p["rope"])
                    if step:
                        prev_k,prev_v = p["cache"].pop(layer)
                        k,v = t.cat((prev_k,k),2),t.cat((prev_v,v),2)
                    p["cache"][layer] = k,v
                    p["hidden"] = model.residual(block,x,model.attention(block,q,k,v,p["mask"]))
            if layer in (9,19,29):
                progress(step,layer+1)
        for p in active:
            top = model.logits(p.pop("hidden")[:,-1]).argmax(-1)
            p["emitted"].append(t.where(p["alive"],top,ends[0]))
            p["alive"] &= ~t.isin(top,ends)
            if not bool(p["alive"].any()):
                p["cache"].clear()
        progress(step+1,0)
    output = []
    stops = set(ends.tolist())
    for p in panels:
        for row,seq in zip(p["rows"],t.stack(p["emitted"],1).tolist(),strict=True):
            stop = next((i for i,k in enumerate(seq) if k in stops),None)
            actual = seq if stop is None else seq[:stop+1]
            output.append(dict(id=row["id"], tokens=actual, stopped=stop is not None,
                               text=model.tok.decode(actual,skip_special_tokens=True)))
    if shadow:
        # Recompute the first two greedy decisions by full-prefix, uncached native forward.
        checks = []
        for step in (0,1):
            subset = [dict(r,ids=r["ids"]+seq["tokens"][:step]) for r,seq in zip(rows[:4],output[:4],strict=True)]
            p = prefill(model, subset)
            for i in range(40):
                with model.block(i) as b:
                    q,k,v = model.qkv(b,p["hidden"],p["rope"])
                    p["hidden"] = model.residual(b,p["hidden"],model.attention(b,q,k,v,p["mask"]))
            top = model.logits(p["hidden"][:,-1]).argmax(-1).tolist()
            checks.append(top==[x["tokens"][step] for x in output[:4]])
        assert all(checks), ("cached/full-prefix top1 mismatch",checks)
    return output
