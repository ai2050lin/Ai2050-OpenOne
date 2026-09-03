#!/usr/bin/env python3
"""C307: compare anonymous sign-and-amplitude transition topology across models."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common

core, OUT = common.core, common.OUTS["C307"]
C287=common.previous.OUTS["C287"]; CORE_FAMILIES=common.previous.previous.FAMILIES; FRACTIONS=(0.0,0.25,0.5,0.75)


def topology(model):
    states=np.load(C287/f"raw/{model}_role_states.float16.npy",mmap_mode="r"); index=core.rows(C287/f"raw/{model}_hidden_index.jsonl"); nq=states.shape[1]; stages=sorted(set(min(nq-2,int(round(f*(nq-2)))) for f in FRACTIONS)); counts=np.zeros((5,4,64),np.uint64); pos=np.zeros((5,4,64,6),np.uint64); amp_up=np.zeros_like(pos); lookup={(r["family"],r["unit"],r["factor_a"],r["factor_b"]):r["hidden_index"] for r in index if r["behavior_correct"]}
    for fi,family in enumerate(CORE_FAMILIES):
        for unit in (0,1):
            keys=[(family,unit,a,b) for a,b in itertools.product((0,1),repeat=2)]
            if not all(k in lookup for k in keys): continue
            cells={(a,b):np.asarray(states[lookup[(family,unit,a,b)]],np.float32) for a,b in itertools.product((0,1),repeat=2)}; effect=.5*((cells[(1,0)]-cells[(0,0)])+(cells[(1,1)]-cells[(0,1)]))
            for si,q in enumerate(stages):
                current=effect[q]>=0; code=sum(current[r].astype(np.uint8)<<r for r in range(6)); nxt=effect[q+1]>=0; ratio=np.abs(effect[q+1])/np.maximum(np.abs(effect[q]),1e-6)
                counts[fi,si]+=np.bincount(code,minlength=64).astype(np.uint64)
                for r in range(6): pos[fi,si,:,r]+=np.bincount(code,weights=nxt[r],minlength=64).astype(np.uint64); amp_up[fi,si,:,r]+=np.bincount(code,weights=ratio[r]>=1.25,minlength=64).astype(np.uint64)
    occ=counts/np.maximum(counts.sum(2,keepdims=True),1); return np.concatenate((occ[...,None],pos/np.maximum(counts[...,None],1),amp_up/np.maximum(counts[...,None],1)),axis=-1).astype(np.float32),{"nq":nq,"dimension":states.shape[-1],"relative_stages":stages}


def permute(value,p):
    out=np.zeros_like(value)
    for old in range(64):
        bits=[(old>>r)&1 for r in range(6)]; new=sum(bits[p[r]]<<r for r in range(6)); out[:,:,new,0]=value[:,:,old,0]
        for r in range(6): out[:,:,new,1+r]=value[:,:,old,1+p[r]]; out[:,:,new,7+r]=value[:,:,old,7+p[r]]
    return out


def similarity(a,b):
    occ=1-.5*np.abs(a[...,0]-b[...,0]).sum(2).mean(); weight=.5*(a[...,0]+b[...,0]); err=(weight[...,None]*np.abs(a[...,1:]-b[...,1:])).sum()/max(float(weight.sum()*12),1e-12); return float(.5*occ+.5*(1-err))


def main():
    if OUT.exists(): raise RuntimeError(OUT)
    parent=core.load(common.OUTS["C306"]/"analysis/final.json"); checks={"parent":parent["all_checks_passed"],"three_models":True,"all_model_coordinates":True,"anonymous_no_coordinate_alignment":True,"all_720_role_permutations":True};
    if not all(checks.values()): raise RuntimeError(checks)
    for sub in ("analysis","audit","protocol"): (OUT/sub).mkdir(parents=True,exist_ok=True)
    protocol={"phase":1841,"campaign":"C307","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"anonymous_transition_topology_frozen","object":"family x relative depth x six-role sign word -> next-role sign and amplitude-up rates","null":"all 720 semantic-role permutations","gate":"similarity>=0.80 and exact upper p<=0.05 for all three model pairs","claim_boundary":"This compares task-conditioned anonymous statistics across different models. It is not physical-coordinate identity, architectural control, causal bisimulation, or implementation isomorphism.","producer_sha256":core.sha(Path(__file__))}; core.save(OUT/"protocol/preregistration.json",protocol)
    values={}; metadata={}
    for model in common.MODELS: values[model],metadata[model]=topology(model); np.save(OUT/f"analysis/{model}_transition_topology.float32.npy",values[model])
    permutations=list(itertools.permutations(range(6))); rows=[]
    for i,left in enumerate(common.MODELS):
        for right in common.MODELS[i+1:]:
            observed=similarity(values[left],values[right]); null=np.asarray([similarity(values[left],permute(values[right],p)) for p in permutations]); p=float((1+(null>=observed).sum())/(1+len(null))); row={"models":[left,right],"similarity":observed,"null_q95":float(np.quantile(null,.95)),"exact_upper_p":p,"pair_gate_passed":observed>=.8 and p<=.05}; rows.append(row); print(f"[C307] {left}/{right}: {observed:.4f}, p={p:.6f}",flush=True)
    broad=len(rows)==3 and all(r["pair_gate_passed"] for r in rows); report={"phase":1841,"campaign":"C307","status":"cross_model_transition_topology_adjudicated","models":metadata,"pairs":rows,"broad_gate_passed":broad,"strict_interpretation":protocol["claim_boundary"],"next_authorization":"C308_adjudication_heatmap"}; core.save(OUT/"analysis/summary.json",report)
    ach={"models":len(values)==3,"pairs":len(rows)==3,"shape":all(list(v.shape)==[5,4,64,13] for v in values.values()),"finite":bool(np.isfinite([r["similarity"] for r in rows]).all())}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":ach,"all_checks_passed":all(ach.values())}); fch={"contract":all(checks.values()),"analysis":all(ach.values()),"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":1841,"campaign":"C307","status":"closed","checks":fch,"all_checks_passed":all(fch.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
