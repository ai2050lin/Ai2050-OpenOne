#!/usr/bin/env python3
"""Independent pre/post audit for Phase1300."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";OUT=T/"result/phase1300_c032_execution_compiler_competition";PARENT=T/"result/phase1299_c032_execution_compiler_contract";P=OUT/"protocol/preregistration.json";M=OUT/"protocol/frozen_compiler_manifest.jsonl";PRE=OUT/"audit/independent_preaudit.json";POST=OUT/"audit/independent_final_audit.json";A=OUT/"raw/compiler_errors.npz";META=OUT/"raw/run_metadata.json";S=OUT/"analysis/compiler_summary.json";R=OUT/"protocol/frozen_runtime.json";F=OUT/"analysis/final.json";C=OUT/"protocol/formal_run_complete.json";MAIN=T/"phase1300_c032_execution_compiler_competition.py";SCRIPT=Path(__file__).resolve();ARMS=["left_global_baseline","right_padding","record_event_aligned","equalized_suffix"];CANDS=ARMS[1:];TH={"case_count_min":96,"finite_fraction_min":1.0,"exact_duplicate_relative_max":1e-6,"same_prefix_relative_max":1e-6,"cross_composition_relative_max":1e-6,"candidate_compilers_passing_min":2,"tau_multiplier":4.0,"tau_floor":1e-7,"tau_cap":1e-4}
def canonical(v:Any)->str:return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(v:Any)->str:return hashlib.sha256(canonical(v).encode()).hexdigest()
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  while c:=f.read(1024*1024):h.update(c)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def add(c,n,p,d):c.append({"name":n,"passed":bool(p),"detail":d})
def base(p):
 c=[];timeless={k:v for k,v in p.items() if k not in {"created_at_utc","protocol_digest"}};add(c,"digest",digest(timeless)==p["protocol_digest"],p["protocol_digest"]);add(c,"sources",p["source_hashes"]=={"main":sha(MAIN),"auditor":sha(SCRIPT)},p["source_hashes"]);add(c,"parent",load(PARENT/"analysis/final.json")["authorization"]=="phase1300_compiler_competition_only" and load(PARENT/"audit/independent_final_audit.json")["all_checks_passed"],"authorized");mm=rows(M);add(c,"manifest",len(mm)==96 and len({x["calibration_id"] for x in mm})==96 and p["dependencies"]["manifest"]==sha(M),len(mm));add(c,"prefix",all(x["prefix_identity"] for x in mm),"all");add(c,"arms",p["arms"]==ARMS and p["candidate_arms"]==CANDS and p["priority"]==CANDS,[p["arms"],p["priority"]]);add(c,"thresholds",p["thresholds"]==TH,p["thresholds"]);add(c,"roles_depths",p["roles"]==["record_slot0_entity","record_slot0_value"] and p["depths"]==list(range(37)),[p["roles"],len(p["depths"])]);add(c,"run",p["formal_run_budget"]==1 and p["fixed_batch_size"]==12,[p["formal_run_budget"],p["fixed_batch_size"]]);return c
def write(path,c,stage,auth):
 ok=all(x["passed"] for x in c);d={"phase":1300,"campaign":"C032","audit_stage":stage,"created_at_utc":datetime.now(timezone.utc).isoformat(),"auditor_imports_main":False,"checks":c,"passed_count":sum(x["passed"] for x in c),"total_count":len(c),"all_checks_passed":ok,"authorization":auth if ok else "none","protocol_digest":load(P)["protocol_digest"]};path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(canonical({"stage":stage,"passed":d["passed_count"],"total":d["total_count"],"authorization":d["authorization"]}));
 if not ok:raise SystemExit(1)
def pre():
 p=load(P);c=base(p);add(c,"clear",not any(x.exists() for x in (A,META,S,R,F,C)),"clear");write(PRE,c,"pre_model","run_phase1300_once")
def post():
 p=load(P);c=base(p);z=np.load(A,allow_pickle=False);exact=z["exact_duplicate"];prefix=z["same_prefix"];cross=z["cross_composition"];add(c,"shape",exact.shape==prefix.shape==cross.shape==(4,96,37,2),[exact.shape,prefix.shape,cross.shape]);add(c,"finite",all(np.isfinite(x).all() for x in (exact,prefix,cross)),"finite");summary=load(S);arms={}
 for ai,arm in enumerate(ARMS):
  maxima={"exact_duplicate_relative_max":float(exact[ai].max()),"same_prefix_relative_max":float(prefix[ai].max()),"cross_composition_relative_max":float(cross[ai].max())};passed=maxima["exact_duplicate_relative_max"]<=TH["exact_duplicate_relative_max"] and maxima["same_prefix_relative_max"]<=TH["same_prefix_relative_max"] and maxima["cross_composition_relative_max"]<=TH["cross_composition_relative_max"];arms[arm]={"maxima":maxima,"median":{"exact_duplicate":float(np.median(exact[ai])),"same_prefix":float(np.median(prefix[ai])),"cross_composition":float(np.median(cross[ai]))},"q99_same_prefix":float(np.quantile(prefix[ai],0.99)),"passed":passed}
 add(c,"arms_recompute",canonical(summary["arms"])==canonical(arms),arms);passing=[x for x in CANDS if arms[x]["passed"]];selected=next((x for x in CANDS if x in passing),None);noise=0.0 if selected is None else max(arms[selected]["maxima"].values());tau=max(TH["tau_floor"],min(TH["tau_cap"],TH["tau_multiplier"]*noise));gates={"candidate_count":len(passing)>=2,"selected_exists":selected is not None,"selected_tau_below_cap":selected is not None and tau<TH["tau_cap"],"finite":True};add(c,"selection",summary["passing_candidate_arms"]==passing and summary["selected_runtime"]==selected and abs(summary["frozen_tau"]-tau)<1e-15 and summary["gates"]==gates,{"passing":passing,"selected":selected,"tau":tau,"gates":gates});passed=all(gates.values());auth="phase1301_qwen3_behavior_only" if passed else "close_c032_without_semantic_run";runtime=load(R);final=load(F);add(c,"runtime_final",runtime["selected_runtime"]==selected and runtime["array_sha256"]==sha(A) and final["authorization"]==auth and final["all_gates_passed"]==passed,[runtime,final]);qa=load(META)["model_audit"];add(c,"fp16",qa["has_fp16_parameters"] and not qa["has_quantized_modules"],qa);add(c,"complete",load(C)["formal_runs_consumed"]==1,load(C));write(POST,c,"post_model",auth)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=("preaudit","postaudit"));a=ap.parse_args();pre() if a.stage=="preaudit" else post()
