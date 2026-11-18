"""Collect and verify genuine State and one-boundary receipt decisions."""

import sys
import time

import numpy as np

import kv_length_run as C
from kv_prefix_native import F


A,B=C.A,C.B


def donors(m,rows):
    options=[]
    for r in rows[:32]:
        if r["fact"]!=r["state"]:
            continue
        u=dict(r,prompt=C.prompt_for(r,r["distance"]))
        u["token"]=C.Material.token_plan(m.tok,u)
        options.append(u)
    found={};t=m.t
    for _,packets in m.batches(options,"length-donor-capture"):
        for p in packets:
            x=p["x"].cuda();n=len(p["rows"])
            b=t.tensor([C.receipt_anchor(r)+pad for r,pad in zip(p["rows"],p["padding"],strict=True)],device="cuda")
            u=x[t.arange(n,device="cuda"),b]
            observed=m.logits(u).argmax(-1).cpu().tolist()
            for j,r in enumerate(p["rows"]):
                k=r["distance"];ix=C.receipt_anchor(r);target=r["token"]["ids"][ix+1]
                if k not in found and observed[j]==target:
                    found[k]=dict(id=r["id"],prompt=r["prompt"],input_to_anchor=r["token"]["ids"][:ix+1],
                                  x=F(u[j]).tolist(),next=target,text=m.tok.decode([target]),boundary=ix)
    assert set(found)=={0,1},{k:v["text"] for k,v in found.items()}
    return dict(sources=[found[0],found[1]],extension=m.tok.encode(C.LONG[len(C.SHORT):],add_special_tokens=False))


def materialize(m,bank):
    t=m.t
    new=t.as_tensor([v["x"] for v in bank["sources"]],device="cuda",dtype=t.bfloat16)
    actual=m.logits(new).argmax(-1).cpu().tolist()
    assert actual==[x["next"] for x in bank["sources"]]
    return actual


def observe(m,rows,bank,label):
    before={};records={};previous={}
    t=m.t;boundary=materialize(m,bank)
    queue=rows
    for attempt in range(6):
        nxt=[]
        for _,packets in m.batches(queue,label):
            for p in packets:
                r0=p["rows"];n=len(r0);start=min(p["starts"])-1
                x=p["x"].cuda();logits=m.logits(x[:,start:]);pred=logits.argmax(-1).cpu().numpy();pq=F(logits[:,-1,m.letters])
                if not attempt:
                    E=p["slots"][:,0].numpy()-1
                    native=x[t.arange(n,device="cuda"),t.as_tensor(E,device="cuda")]
                    ix={(r["family"],r["distance"],r["map"],r["state"],r["fact"]):j for j,r in enumerate(r0)}
                    donors=[ix[r["family"],r["distance"],r["map"],r["state"],r["state"]] for r in r0]
                    picked=m.logits(native[donors]).argmax(-1).cpu().tolist()
                    for j,r in enumerate(r0):
                        before[r["id"]]=dict(untreated=int(pred[j,E[j]-start]),state=int(picked[j]),donor=r0[donors[j]]["id"])
                for j,r in enumerate(r0):
                    rid=r["id"]; k=r["token"]; first=k["start"];e=k["slots"][0]-1;g=e+1
                    d=r["distance"];pivot=C.receipt_anchor(r);pads=p["padding"][j];expect=k["ids"]
                    raw_b=int(pred[j,pivot+pads-start])
                    pred[j,e+pads-start]=before[rid]["state"]
                    pred[j,pivot+pads-start]=boundary[d]
                    v=pred[j,p["starts"][j]-1-start:]
                    mismatch=np.flatnonzero(v[:-1]!=expect[first:])
                    accepted=int(mismatch[0]) if len(mismatch) else len(expect)-first
                    hit=int(v[accepted]);matched=not len(mismatch)
                    if not matched and attempt<5 and first+accepted>g:
                        text=m.tok.decode([hit])
                        w=m.whitespace_proposal(r,first+accepted,hit,text) if text.endswith("\n") else None
                        if w is not None:
                            line,plan=w
                            try: C.receipt_anchor(r,plan)
                            except AssertionError: w=None
                        if w is not None:
                            previous.setdefault(rid,[]).append(dict(token=hit,line=line,index=first+accepted))
                            nxt.append(dict(r,token=plan));continue
                    emitted=expect[first:first+accepted]+[hit]
                    records[rid]=dict(id=rid,matched=matched,expected_state=r["word_ids"][r["state"]],
                                      state_emitted=first+accepted>=g and before[rid]["state"]==r["word_ids"][r["state"]],
                                      boundary_reached=first+accepted>pivot,pivot=pivot,untreated_boundary=raw_b,
                                      intended_boundary=boundary[d],first_divergence=None if matched else first+accepted,
                                      observation=hit,emitted=emitted,rendered=m.tok.decode(emitted),
                                      pq=pq[j].tolist() if matched else None,top=hit if matched else None,
                                      adapted=k if rid in previous else None,whitespace=previous.get(rid,[]),**before[rid])
        queue=nxt
        if not queue:break
    return [records[r["id"]] for r in rows]


def gate(rows,seen,bank):
    by={};eligible=[];lacks=[];pairs=0;hole=[]
    for j in range(0,len(rows),16):
        rs=rows[j:j+16]; y=seen[j:j+16]; good=[]
        for r,o in zip(rs,y,strict=True):
            assert r["id"]==o["id"]
            k=r["id"].rsplit("-",1)[-1];n=by.setdefault(k,dict(n=0,state=0,boundary=0,pq=0,congruent_correct=0))
            n["n"]+=1;n["state"]+=int(o["state_emitted"]);n["boundary"]+=int(o["boundary_reached"])
            valid=o["matched"] and o["top"] in r["token"]["letters"]
            n["pq"]+=int(valid)
            correct=valid and o["top"]==r["token"]["letters"][r["code_s"]=="Q"]
            n["congruent_correct"]+=int(correct)
            good.append(correct if r["fact"]==r["state"] else valid)
            if not good[-1]:lacks.append(r["id"])
        exact=[]
        for i in range(8):
            l,h=rs[i],rs[i+8];st,lt=y[i],y[i+8]
            assert (l["prompt"],l["fact"],l["state"],l["map"])==(h["prompt"],h["fact"],h["state"],h["map"])
            short=(st["adapted"] or l["token"])["ids"];long=(lt["adapted"] or h["token"])["ids"]
            b=st["pivot"]+1
            ok=st["matched"] and lt["matched"] and st["pivot"]==lt["pivot"] and long==short[:b]+bank["extension"]+short[b:]
            exact.append(ok)
            if ok:pairs+=1
            else:hole.append([l["id"],h["id"]])
        if all(good) and all(exact):eligible.append(rs[0]["family"])
    each=all((v["congruent_correct"] if k[-1]==k[-2] else v["pq"])/v["n"]>=.9 for k,v in by.items())
    return dict(passed=each,eligible=eligible,exact_pairs=pairs,cells=by,individual_failures=lacks,pair_failures=hole)


def pilot(m):
    if not (C.OUT/"material-pilot.json").exists():
        A.save("material-pilot.json",C.material("pilot",m.tok,C.pools(m.tok)["pilot"]))
    rows=A.load("material-pilot.json")
    if not (C.OUT/"neutral-donors.json").exists():
        A.save("neutral-donors.json",donors(m,rows))
    bank=A.load("neutral-donors.json")
    if not (C.OUT/"pilot-natural.json").exists():
        A.save("pilot-natural.json",dict(binding=B.sha("material-pilot.json"),rows=observe(m,rows,bank,"length-pilot-natural")))
    stat=gate(rows,A.load("pilot-natural.json")["rows"],bank)
    A.save("pilot-natural-gate.json",stat)
    print("Pilot length results",stat,flush=True)


def main():
    import torch
    C.setup();C.identity()
    torch.set_num_threads(4);torch.set_float32_matmul_precision("highest")
    with C.P.GpuLease(),torch.no_grad():
        t=time.monotonic();C.N.progress("length-pilot-load",0,1,t);m=C.N.Engine();C.N.progress("length-pilot-load",1,1,t)
        try:pilot(m)
        finally:m.close()


if __name__=="__main__":main()
