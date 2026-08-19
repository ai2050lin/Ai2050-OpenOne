#!/usr/bin/env python3
"""Independent pre/post audit for Phase1306."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";OUT=T/"result/phase1306_c033_frozen_answer_boundary_hidden";PARENT=T/"result/phase1305_c033_qwen3_behavior";CONTRACT=T/"result/phase1304_c033_role_typed_causal_graph_contract";P=OUT/"protocol/preregistration.json";M=OUT/"protocol/frozen_hidden_manifest.jsonl";PRE=OUT/"audit/independent_preaudit.json";POST=OUT/"audit/independent_final_audit.json";A=OUT/"raw/hidden_arrays.npz";META=OUT/"raw/run_metadata.json";S=OUT/"analysis/hidden_summary.json";F=OUT/"analysis/final.json";C=OUT/"protocol/formal_run_complete.json";MAIN=T/"phase1306_c033_frozen_answer_boundary_hidden.py";SCRIPT=Path(__file__).resolve();PARTS=("discovery","confirmation","holdout");PANELS=("active","matched_null","surface_only","semantic_neighbor");SURF=("catalog_prose","inventory_ledger");ATTRS=("color","material","location","size","shape","status");DEPTHS=(25,26);EPS=1e-12
def canonical(v:Any)->str:return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  while c:=f.read(1024*1024):h.update(c)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def add(c,n,p,d):c.append({"name":n,"passed":bool(p),"detail":d})
def base(p):
 c=[];timeless={k:v for k,v in p.items() if k not in {"created_at_utc","protocol_digest"}};add(c,"digest",digest(timeless)==p["protocol_digest"],p["protocol_digest"]);add(c,"sources",p["source_hashes"]=={"main":sha(MAIN),"auditor":sha(SCRIPT)},p["source_hashes"]);add(c,"parent",load(PARENT/"analysis/final.json")["authorization"]=="phase1306_frozen_hidden_only" and load(PARENT/"audit/independent_final_audit.json")["all_checks_passed"],"parent");mm=rows(M);add(c,"manifest",len(mm)==1152 and p["material"]["manifest_sha256"]==sha(M),len(mm));add(c,"order",all([x["panel"] for x in mm[i:i+4]]==list(PANELS) for i in range(0,1152,4)),"order");add(c,"fixed",p["event"]=="assistant_answer_boundary" and p["depths"]==[25,26],{"event":p["event"],"depths":p["depths"]});add(c,"hard_stops",p["hard_stops"]==["No new event or depth","No discovery selection","No component scan","No causal patch","No threshold change"],p["hard_stops"]);return c
def write(path,c,stage,auth):
 ok=all(x["passed"] for x in c);d={"phase":1306,"campaign":"C033","audit_stage":stage,"created_at_utc":datetime.now(timezone.utc).isoformat(),"auditor_imports_main":False,"checks":c,"passed_count":sum(x["passed"] for x in c),"total_count":len(c),"all_checks_passed":ok,"authorization":auth if ok else "none","protocol_digest":load(P)["protocol_digest"]};path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(canonical({"stage":stage,"passed":d["passed_count"],"total":d["total_count"],"authorization":d["authorization"]}));
 if not ok:raise SystemExit(1)
def pre():
 p=load(P);c=base(p);add(c,"clear",not any(x.exists() for x in (A,META,S,F,C)),"clear");write(PRE,c,"pre_model","run_phase1306_once")
def cell(norm,identity,meta,partition,di):
 lookup={(x["profile_index"],x["attribute"],x["surface"],x["panel"]):i for i,x in enumerate(meta) if x["partition"]==partition};active=[];aid=[];controls={p:[] for p in PANELS if p!="active"};cid={p:[] for p in controls};wins=[]
 for profile in range(8):
  for attr in ATTRS:
   for surf in SURF:
    ai=lookup[(profile,attr,surf,"active")];n=float(norm[ai,di]);iv=float(identity[ai,di]);cv={p:float(norm[lookup[(profile,attr,surf,p)],di]) for p in controls};active.append(n);aid.append(iv);wins.append(n>max(cv.values()))
    for p in controls:controls[p].append(cv[p]);cid[p].append(abs(float(identity[lookup[(profile,attr,surf,p)],di])))
 nm=float(np.median(active));cn=max(float(np.median(v)) for v in controls.values());im=float(np.median(aid));ci=max(float(np.median(v)) for v in cid.values());return {"active_norm_median":nm,"max_control_norm_median":cn,"norm_ratio":nm/(cn+EPS),"norm_win_fraction":float(np.mean(wins)),"active_identity_median":im,"max_control_abs_identity_median":ci,"identity_ratio":im/(ci+EPS),"identity_positive_fraction":float(np.mean(np.asarray(aid)>0))}
def passes(x,th):return x["active_norm_median"]>=th["active_norm_median_min"] and x["norm_ratio"]>=th["active_to_max_control_ratio_min"] and x["norm_win_fraction"]>=th["active_over_controls_fraction_min"] and x["identity_positive_fraction"]>=th["identity_positive_fraction_min"] and x["identity_ratio"]>=th["identity_to_max_control_ratio_min"]
def post():
 p=load(P);c=base(p);z=np.load(A,allow_pickle=False);norm=z["relative_norm"];identity=z["identity_delta"];behavior=z["behavior_correct"];add(c,"shapes",norm.shape==identity.shape==(1152,2) and behavior.shape==(1152,2),[norm.shape,identity.shape,behavior.shape]);add(c,"finite",np.isfinite(norm).all() and np.isfinite(identity).all(),"finite");meta=load(META)["pair_metadata"];cells={part:{str(d):cell(norm,identity,meta,part,di) for di,d in enumerate(DEPTHS)} for part in PARTS};th=p["thresholds"];gates={"finite":True,"behavior_replay":float(np.mean(behavior))>=th["behavior_replay_accuracy_min"]}
 for part in PARTS:
  for d in DEPTHS:gates[f"{part}_depth{d}"]=passes(cells[part][str(d)],th)
 summary=load(S);add(c,"analysis",summary["cells"]==cells and summary["gates"]==gates,{"gates":gates});passed=all(gates.values());auth="phase1307_bidirectional_swap_only" if passed else "close_c033_without_causal";add(c,"authorization",load(F)["authorization"]==auth and load(F)["all_gates_passed"]==passed,load(F));qa=load(META)["model_audit"];add(c,"fp16",qa["has_fp16_parameters"] and not qa["has_quantized_modules"],qa);add(c,"complete",load(C)["formal_runs_consumed"]==1,load(C));write(POST,c,"post_model",auth)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=("preaudit","postaudit"));a=ap.parse_args();pre() if a.stage=="preaudit" else post()
