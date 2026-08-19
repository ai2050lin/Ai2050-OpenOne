#!/usr/bin/env python3
"""Final independent Phase1300 audit incorporating the exact source erratum."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";OUT=T/"result/phase1300_c032_execution_compiler_competition";P=OUT/"protocol/preregistration.json";OLDPOST=OUT/"audit/independent_final_audit.json";REPAIR=OUT/"audit/independent_repair_audit.json";ERR=OUT/"protocol/execution_erratum.json";A=OUT/"raw/compiler_errors.npz";S=OUT/"analysis/compiler_summary.json";R=OUT/"protocol/frozen_runtime.json";F=OUT/"analysis/final.json";C=OUT/"protocol/formal_run_complete.json";META=OUT/"raw/run_metadata.json";DEST=OUT/"audit/independent_final_audit_after_erratum.json";MAIN=T/"phase1300_c032_execution_compiler_competition.py";ARMS=["left_global_baseline","right_padding","record_event_aligned","equalized_suffix"];CANDS=ARMS[1:];TH={"exact":1e-6,"prefix":1e-6,"cross":1e-6,"min":2,"mult":4.0,"floor":1e-7,"cap":1e-4}
def canonical(v:Any)->str:return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  while c:=f.read(1024*1024):h.update(c)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def add(c,n,p,d):c.append({"name":n,"passed":bool(p),"detail":d})
def main():
 c=[];protocol=load(P);old=load(OLDPOST);repair=load(REPAIR);err=load(ERR);failed=[x["name"] for x in old["checks"] if not x["passed"]];add(c,"old_audit_only_source_hash_failed",old["passed_count"]==15 and old["total_count"]==16 and failed==["sources"],failed);add(c,"repair_audit",repair["all_checks_passed"] and repair["authorization"]=="retry_phase1300_once_after_exact_engineering_repair",repair);add(c,"repaired_source",sha(MAIN)==err["repaired_main_sha256"] and err["scientific_constants_changed"] is False,err)
 z=np.load(A,allow_pickle=False);exact=z["exact_duplicate"];prefix=z["same_prefix"];cross=z["cross_composition"];add(c,"shape_finite",exact.shape==prefix.shape==cross.shape==(4,96,37,2) and all(np.isfinite(x).all() for x in (exact,prefix,cross)),[exact.shape,prefix.shape,cross.shape]);arms={}
 for i,a in enumerate(ARMS):
  mx={"exact_duplicate_relative_max":float(exact[i].max()),"same_prefix_relative_max":float(prefix[i].max()),"cross_composition_relative_max":float(cross[i].max())};ok=mx["exact_duplicate_relative_max"]<=TH["exact"] and mx["same_prefix_relative_max"]<=TH["prefix"] and mx["cross_composition_relative_max"]<=TH["cross"];arms[a]={"maxima":mx,"median":{"exact_duplicate":float(np.median(exact[i])),"same_prefix":float(np.median(prefix[i])),"cross_composition":float(np.median(cross[i]))},"q99_same_prefix":float(np.quantile(prefix[i],.99)),"passed":ok}
 summary=load(S);add(c,"arms_recompute",canonical(arms)==canonical(summary["arms"]),arms);passing=[a for a in CANDS if arms[a]["passed"]];selected=next((a for a in CANDS if a in passing),None);noise=max(arms[selected]["maxima"].values()) if selected else 0.0;tau=max(TH["floor"],min(TH["cap"],TH["mult"]*noise));gates={"candidate_count":len(passing)>=TH["min"],"selected_exists":selected is not None,"selected_tau_below_cap":selected is not None and tau<TH["cap"],"finite":True};add(c,"selection_recompute",summary["passing_candidate_arms"]==passing and summary["selected_runtime"]==selected and summary["gates"]==gates and abs(summary["frozen_tau"]-tau)<1e-15,{"passing":passing,"selected":selected,"tau":tau,"gates":gates});runtime=load(R);final=load(F);add(c,"runtime_final",runtime["selected_runtime"]==selected=="right_padding" and runtime["array_sha256"]==sha(A) and final["authorization"]=="phase1301_qwen3_behavior_only" and final["all_gates_passed"], [runtime,final]);qa=load(META)["model_audit"];add(c,"fp16",qa["has_fp16_parameters"] and not qa["has_quantized_modules"],qa);add(c,"completion",load(C)["formal_runs_consumed"]==1,load(C));ok=all(x["passed"] for x in c);doc={"phase":1300,"campaign":"C032","audit_stage":"post_model_after_exact_engineering_erratum","created_at_utc":datetime.now(timezone.utc).isoformat(),"auditor_imports_main":False,"checks":c,"passed_count":sum(x["passed"] for x in c),"total_count":len(c),"all_checks_passed":ok,"scientific_authorization":"phase1301_qwen3_behavior_only" if ok else "none","protocol_digest":protocol["protocol_digest"],"supersedes_for_authorization":"independent_final_audit.json source-hash-only failure"};DEST.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(canonical({"passed":doc["passed_count"],"total":doc["total_count"],"authorization":doc["scientific_authorization"]}));
 if not ok:raise SystemExit(1)
if __name__=="__main__":main()
