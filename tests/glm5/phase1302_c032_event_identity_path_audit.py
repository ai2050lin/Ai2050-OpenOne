#!/usr/bin/env python3
"""Independent pre/post audit for Phase1302."""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";OUT=T/"result/phase1302_c032_event_identity_path";PARENT=T/"result/phase1301_c032_qwen3_behavior";P=OUT/"protocol/preregistration.json";M=OUT/"protocol/frozen_event_manifest.jsonl";PRE=OUT/"audit/independent_preaudit.json";POST=OUT/"audit/independent_final_audit.json";A=OUT/"raw/event_identity_arrays.npz";META=OUT/"raw/run_metadata.json";S=OUT/"analysis/path_summary.json";F=OUT/"analysis/final.json";C=OUT/"protocol/formal_run_complete.json";MAIN=T/"phase1302_c032_event_identity_path.py";SCRIPT=Path(__file__).resolve();PARTS=("discovery","confirmation","holdout");PANELS=("active","matched_null","surface_only","semantic_neighbor");SURF=("catalog_prose","inventory_ledger");ATTRS=("color","material","location","size","shape","status");EVENTS=("record_slot0_entity","record_slot0_value","query_clause_end","user_answer_cue_end","assistant_answer_boundary");PRIMARY=EVENTS[-2:];TH={"finite_fraction_min":1.0,"behavior_replay_accuracy_min":0.99,"discovery_norm_median_min":0.001,"discovery_norm_ratio_min":1.20,"discovery_norm_win_fraction_min":0.75,"discovery_identity_positive_fraction_min":0.75,"discovery_identity_ratio_min":1.15,"transfer_norm_median_min":0.001,"transfer_norm_ratio_min":1.10,"transfer_norm_win_fraction_min":0.70,"transfer_identity_positive_fraction_min":0.70,"transfer_identity_ratio_min":1.05,"adjacent_depths_min":2};EPS=1e-12
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
 c=[];timeless={k:v for k,v in p.items() if k not in {"created_at_utc","protocol_digest"}};add(c,"digest",digest(timeless)==p["protocol_digest"],p["protocol_digest"]);add(c,"sources",p["source_hashes"]=={"main":sha(MAIN),"auditor":sha(SCRIPT)},p["source_hashes"]);add(c,"parent",load(PARENT/"analysis/final.json")["authorization"]=="phase1302_event_identity_hidden_only" and load(PARENT/"audit/independent_final_audit.json")["all_checks_passed"],"parent");mm=rows(M);add(c,"manifest",len(mm)==1152 and p["material"]["manifest_sha256"]==sha(M),len(mm));add(c,"order",all([x["panel"] for x in mm[i:i+4]]==list(PANELS) for i in range(0,1152,4)),"four panels");add(c,"events",p["events"]==list(EVENTS) and p["primary_events"]==list(PRIMARY),[p["events"],p["primary_events"]]);add(c,"thresholds",p["thresholds"]==TH,p["thresholds"]);add(c,"runtime",p["runtime"]=={"compiler":"right_padding","global_fixed_length":True,"tau":1e-7,"pair_batch":4},p["runtime"]);add(c,"hard_stops",p["hard_stops"]==["No head or MLP scan","No causal patch","No event/depth reselection","No threshold change"],p["hard_stops"]);return c
def write(path,c,stage,auth):
 ok=all(x["passed"] for x in c);d={"phase":1302,"campaign":"C032","audit_stage":stage,"created_at_utc":datetime.now(timezone.utc).isoformat(),"auditor_imports_main":False,"checks":c,"passed_count":sum(x["passed"] for x in c),"total_count":len(c),"all_checks_passed":ok,"authorization":auth if ok else "none","protocol_digest":load(P)["protocol_digest"]};path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(canonical({"stage":stage,"passed":d["passed_count"],"total":d["total_count"],"authorization":d["authorization"]}));
 if not ok:raise SystemExit(1)
def pre():
 p=load(P);c=base(p);add(c,"clear",not any(x.exists() for x in (A,META,S,F,C)),"clear");write(PRE,c,"pre_model","run_phase1302_once")
def cell(norm,identity,meta,partition,ei,d):
 lookup={(x["profile_index"],x["attribute"],x["surface"],x["panel"]):i for i,x in enumerate(meta) if x["partition"]==partition};av=[];aid=[];control_abs={p:[] for p in PANELS if p!="active"};wins=[];control_norm={p:[] for p in control_abs}
 for profile in range(8):
  for attr in ATTRS:
   for surf in SURF:
    ai=lookup[(profile,attr,surf,"active")];n=float(norm[ai,d,ei]);iv=float(identity[ai,d,ei]);cv={p:float(norm[lookup[(profile,attr,surf,p)],d,ei]) for p in control_abs};av.append(n);aid.append(iv);wins.append(n>max(cv.values()))
    for p in control_abs:control_abs[p].append(abs(float(identity[lookup[(profile,attr,surf,p)],d,ei])));control_norm[p].append(cv[p])
 nm=float(np.median(av));cn=max(float(np.median(v)) for v in control_norm.values());im=float(np.median(aid));ci=max(float(np.median(v)) for v in control_abs.values());return {"active_norm_median":nm,"max_control_norm_median":cn,"norm_ratio":nm/(cn+EPS),"norm_win_fraction":float(np.mean(wins)),"active_identity_median":im,"max_control_abs_identity_median":ci,"identity_ratio":im/(ci+EPS),"identity_positive_fraction":float(np.mean(np.asarray(aid)>0))}
def passes(x,disc):
 q="discovery" if disc else "transfer";return x["active_norm_median"]>=TH[f"{q}_norm_median_min"] and x["norm_ratio"]>=TH[f"{q}_norm_ratio_min"] and x["norm_win_fraction"]>=TH[f"{q}_norm_win_fraction_min"] and x["identity_positive_fraction"]>=TH[f"{q}_identity_positive_fraction_min"] and x["identity_ratio"]>=TH[f"{q}_identity_ratio_min"]
def post():
 p=load(P);c=base(p);z=np.load(A,allow_pickle=False);norm=z["relative_norm"];identity=z["identity_delta"];behavior=z["behavior_correct"];prefix=z["prefix_identity"];add(c,"shapes",norm.shape==identity.shape==(1152,37,5) and behavior.shape==(1152,2) and prefix.shape==(288,37,2,2),[norm.shape,identity.shape,behavior.shape,prefix.shape]);add(c,"finite",all(np.isfinite(x).all() for x in (norm,identity,prefix)),"finite");meta=load(META)["pair_metadata"];tables={part:{e:[cell(norm,identity,meta,part,ei,d) for d in range(37)] for ei,e in enumerate(EVENTS)} for part in PARTS};selected={}
 for e in PRIMARY:
  ok=[passes(tables["discovery"][e][d],True) for d in range(37)];start=next((d for d in range(36) if ok[d] and ok[d+1]),None);selected[e]=[] if start is None else [start,start+1]
 transfer={part:{e:{"depths":selected[e],"cells":[tables[part][e][d] for d in selected[e]],"passed":bool(selected[e]) and all(passes(tables[part][e][d],False) for d in selected[e])} for e in PRIMARY} for part in ("confirmation","holdout")};gates={"finite":True,"behavior_replay":float(np.mean(behavior))>=TH["behavior_replay_accuracy_min"],"prefix_identity":float(prefix.max())<=1e-7,"discovery_user_answer_cue":bool(selected["user_answer_cue_end"]),"discovery_assistant_boundary":bool(selected["assistant_answer_boundary"]),"confirmation_user_answer_cue":transfer["confirmation"]["user_answer_cue_end"]["passed"],"confirmation_assistant_boundary":transfer["confirmation"]["assistant_answer_boundary"]["passed"],"holdout_user_answer_cue":transfer["holdout"]["user_answer_cue_end"]["passed"],"holdout_assistant_boundary":transfer["holdout"]["assistant_answer_boundary"]["passed"]};summary=load(S);add(c,"analysis",summary["selected_discovery_bands"]==selected and summary["transfer"]==transfer and summary["gates"]==gates,{"selected":selected,"gates":gates});passed=all(gates.values());auth="phase1303_frozen_causal_rescue" if passed else "close_c032_without_causal_claim";add(c,"authorization",load(F)["authorization"]==auth and load(F)["all_gates_passed"]==passed and load(F)["causal_intervention_performed"] is False,load(F));qa=load(META)["model_audit"];add(c,"fp16",qa["has_fp16_parameters"] and not qa["has_quantized_modules"],qa);add(c,"complete",load(C)["formal_runs_consumed"]==1,load(C));write(POST,c,"post_model",auth)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=("preaudit","postaudit"));a=ap.parse_args();pre() if a.stage=="preaudit" else post()
