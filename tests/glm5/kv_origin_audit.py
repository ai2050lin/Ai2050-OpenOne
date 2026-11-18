"""CPU re-read of frozen origin/path effects, telescoping, and counterexamples."""

import numpy as np

import kv_origin_run as C
from kv_origin_study import final_config, signed
from kv_origin_trace import POSITIONS


A,B=C.A,C.B


def summarize(x,confidence=.95):
    v=np.asarray(x,dtype=np.float64)
    ix=np.random.default_rng(C.SEED).integers(0,len(v),size=(20000,len(v)))
    alpha=(1-confidence)/2
    return dict(mean=float(v.mean()),ci=np.quantile(v[ix].mean(1),[alpha,1-alpha]).tolist(),n=len(v))


def fam(x):
    return x.reshape(-1,8,*x.shape[1:]).mean(1)


def leave_lexeme(values,rows):
    group=[set(r["values"].values())|{r["rejected"]} for r in rows[::8]]
    v=[float(values[[w not in x for x in group]].mean()) for w in C.Prev.words(rows) if not all(w in x for x in group)]
    return [min(v),max(v)]


def main():
    p=C.verify(); f=A.load("freeze.json"); ledger=A.load("confirm.json"); b=A.load("confirm-behavior.json")
    assert f["plan"]==B.sha("plan.json") and ledger["freeze"]==b["freeze"]==B.sha("freeze.json")
    assert all(B.sha(k)==v for k,v in f["discovery"].items())
    assert (C.OUT/"freeze.json").stat().st_mtime_ns<(C.OUT/"confirm-behavior.json").stat().st_mtime_ns
    assert ledger["behavior"]==B.sha("confirm-behavior.json")
    mat=A.load(p["materials"]["confirm"]["name"])
    assert C.Prev.behavior(mat,b["rows"])["cells"]==b["cells"]
    fails=[dict(row=r,observed=o) for r,o in zip(mat,b["rows"],strict=True) if not o["code_correct"]]
    if ledger["state"]!="finished":
        A.save("audit.json",dict(state=ledger["state"],cells=b["cells"],natural_failures=fails))
        return
    rows=C.resolved(mat,b); d=B.load_data(ledger)
    assert list(d["ids"])==[q["id"] for q in rows] and len(rows)==8*ledger["families"]
    assert all(len({r["family"] for r in rows[i:i+8]})==1 for i in range(0,len(rows),8))
    arms={a:j for j,a in enumerate(f["arms"])}
    assert list(arms)==list(final_config(f)["arms"])
    m=signed(d,rows)
    arm=lambda a:m[:,arms[a]]
    drop=lambda a:arm("sham")-arm(a)
    mid=lambda a:(arm(a+"-rand1")+arm(a+"-rand2"))/2
    gold=np.array([r["token"]["letters"][r["code"]=="Q"] for r in rows])
    errors=(d["top"]!=gold[:,None]); assert not errors[:,0].any()
    ep={k:mid(k)-arm(k) for k in ("fact","book","gen")}
    ep["prompt-gen"]=drop("prompt")-drop("gen")
    for key in ("fact","gen"):
        ep["rest-"+key]=arm("restore-"+key)-arm("all")
    for key in ("head","unit"):
        k="res-"+key
        ep[k+"-specific"]=arm(k)-mid(k)
        ep[k+"-absolute"]=arm(k)-arm("u")
    assert len(ep)==len(p["primary"]["list"])
    primary={}
    for k,v in ep.items():
        fv=fam(v)
        z=summarize(fv,p["primary"]["confidence"])
        sense=-1 if k=="prompt-gen" and z["mean"]<0 else 1
        q=z["ci"][0] if sense==1 else -z["ci"][1]
        primary[k]=dict(**z,passed=bool(q>0 and sense*z["mean"]>=p["primary"]["floor"] and len(fv)>=p["primary"]["coverage"]),
                        leave_lexeme=leave_lexeme(fv,rows))
    n0=float(drop("u").mean())
    behavior={k:dict(drop=summarize(fam(drop(k))),errors=int(errors[:,j].sum()),
                     complete=int((~errors[:,j].reshape(-1,8).any(1)).sum()),
                     mean_rescue_from_u=float((arm(k)-arm("u")).mean())) for k,j in arms.items()}
    path={}
    for k in ("att","mlp","head","unit"):
        res="res-"+k
        score=summarize(fam(arm(res)-arm("u")))
        abs_block=summarize(fam(drop("block-"+k)))
        fraction=score["mean"]/n0 if n0>0 else None
        major=(k in ("head","unit") and primary[res+"-specific"]["passed"] and primary[res+"-absolute"]["passed"]
               and fraction is not None and fraction>=p["primary"]["fraction"] and abs_block["ci"][0]>0)
        path[k]=dict(rescue=score,block=abs_block,fraction=fraction,major=bool(major))
    for a,h in [("res-head","receiver_head"),("res-unit","unit_activations")]:
        assert np.array_equal(d[h][:,arms[a]],d[h][:,0]),(a,h)
    for a,h in [("block-head","receiver_head"),("block-unit","unit_activations")]:
        assert not d[h][:,arms[a]].any()
    for key in ("fact","book","gen","prompt","all","u"):
        assert np.array_equal(d["source_head"][:,arms["self-"+key]],d["source_head"][:,0])
    assert not d["source_head"][:,1].any()
    assert np.allclose(d["source_alpha"][:,[arms[x] for x in ("fact","book","gen","all","prompt")]],d["source_alpha"][:,[0]])
    assert np.max(abs(d["readout_closure"]))<1e-6
    physical={k:dict(source_head_change=float(np.median(np.linalg.norm(d["source_head"][:,arms[k]]-d["source_head"][:,0],axis=-1))),
                    block25_projected_change=float(np.median(d["actual_output_norm"][:,arms[k],0,0]))) for k in arms}
    med=np.median(d["source_alpha"][:,0],axis=0)
    layer_site=[]
    for j,l in enumerate(C.LAYERS):
        layer_site.append(dict(layer=l,attention=summarize(fam(d["readout_sites"][:,j*2])),mlp=summarize(fam(d["readout_sites"][:,j*2+1]))))
    no_evidence=np.argwhere(errors)
    ranked=lambda x,mask:sorted(np.flatnonzero(mask),key=lambda i:x[i])[:6]
    cases={"changed_codes":list(dict.fromkeys(no_evidence[:,0].tolist()))[:10],
           "factual_block_helps":ranked(drop("fact"),drop("fact")<0),
           "generated_block_larger":ranked(drop("prompt")-drop("gen"),drop("gen")>drop("prompt")),
           "strong_source_block_weak_factual_and_generated":ranked(-drop("u"),(abs(drop("gen"))<.1)&(abs(drop("fact"))<.1)),
           "attention_rescue_harms":ranked(arm("res-head")-arm("u"),arm("res-head")<arm("u")),
           "mlp_unit_rescue_harms":ranked(arm("res-unit")-arm("u"),arm("res-unit")<arm("u"))}
    examples={k:[dict(row=rows[i],pq={x:d["pq"][i,j].tolist() for x,j in arms.items()},top={x:int(d["top"][i,j]) for x,j in arms.items()},
                      attention=d["source_alpha"][i,0].tolist(),drop_u=float(drop("u")[i]),drop_fact=float(drop("fact")[i]),
                      drop_book=float(drop("book")[i]),drop_gen=float(drop("gen")[i]),readout=d["readout_sites"][i].tolist()) for i in ix] for k,ix in cases.items()}
    cells={}
    for k in (0,1):
        for l in (0,1):
            mask=np.array([r["compete"]==k and r["distance"]==l for r in rows])
            cells[f"{k}{l}"]={x:summarize(v[mask].reshape(-1,2).mean(1)) for x,v in {**{w:drop(w) for w in ("u","fact","book","gen","prompt","all")},**ep}.items()}
    rounding=d["readout_total"]-drop("u")
    unit_norm=np.median(d["unit_write_norm"][:,[arms[k] for k in ("sham","u","res-unit")]],axis=0)
    A.save("audit.json",dict(state="finished",ledger=B.sha("confirm.json"),freeze=B.sha("freeze.json"),
           eligible=len(rows)//8,n=len(rows),baseline=b["cells"],natural_failures=fails,primary=primary,behavior=behavior,path=path,
           cells=cells,examples=examples,output_geometry=physical,alpha=dict(zip(POSITIONS,med.tolist(),strict=True)),
           receiver_qdiff_u=float(np.median(d["receiver_qdiff"][:,1])),unit_write_norm=unit_norm.tolist(),
           continuous=dict(layers=layer_site,mean=float(d["readout_total"].mean()),close=float(abs(d["readout_closure"]).max()),
                           bf16_mean_error=float(rounding.mean()),bf16_abs_error_median=float(np.median(abs(rounding)))),
           interaction=summarize(fam(drop("all")-drop("fact")-drop("book")-drop("gen"))),
           numeric=dict(sham_max=float(abs(d["pq"][:,0]-d["untreated"]).max()),v_kernel_error=float(d["source_kernel_error"].max()),
                        same_shape_restoration_max=max(float(abs(d["pq"][:,j]-d["pq"][:,0]).max()) for k,j in arms.items() if k.startswith("self-")))))
    print("Primary",{k:(v["mean"],v["passed"]) for k,v in primary.items()},"Path",path,flush=True)


if __name__=="__main__":
    C.setup(); main()
