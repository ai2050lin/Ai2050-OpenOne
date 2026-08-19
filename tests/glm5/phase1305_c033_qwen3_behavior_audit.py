#!/usr/bin/env python3
"""Independent pre/post audit for Phase1305."""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";OUT=T/"result/phase1305_c033_qwen3_behavior";PARENT=T/"result/phase1304_c033_role_typed_causal_graph_contract";P=OUT/"protocol/preregistration.json";PRE=OUT/"audit/independent_preaudit.json";POST=OUT/"audit/independent_final_audit.json";RAW=OUT/"raw/candidate_scores.jsonl";GEN=OUT/"raw/list_free_generations.jsonl";S=OUT/"analysis/behavior_summary.json";F=OUT/"analysis/final.json";C=OUT/"protocol/formal_run_complete.json";MAIN=T/"phase1305_c033_qwen3_behavior.py";SCRIPT=Path(__file__).resolve();MATERIAL=PARENT/"material/frozen_role_typed_lookup_cases.jsonl"
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
 c=[];timeless={k:v for k,v in p.items() if k not in {"created_at_utc","protocol_digest"}};add(c,"digest",digest(timeless)==p["protocol_digest"],p["protocol_digest"]);add(c,"sources",p["source_hashes"]=={"main":sha(MAIN),"auditor":sha(SCRIPT)},p["source_hashes"]);add(c,"parent",load(PARENT/"analysis/final.json")["authorization"]=="phase1305_qwen3_behavior_only" and load(PARENT/"audit/independent_final_audit.json")["all_checks_passed"],"parent");add(c,"material",p["material"]["sha256"]==sha(MATERIAL) and p["material"]["case_count"]==6912,p["material"]);add(c,"hidden_forbidden",p["hidden_states_read"] is False,p["hidden_states_read"]);add(c,"single_run",p["formal_run_budget"]==1,p["formal_run_budget"]);return c
def write(path,c,stage,auth):
 ok=all(x["passed"] for x in c);d={"phase":1305,"campaign":"C033","audit_stage":stage,"created_at_utc":datetime.now(timezone.utc).isoformat(),"auditor_imports_main":False,"checks":c,"passed_count":sum(x["passed"] for x in c),"total_count":len(c),"all_checks_passed":ok,"authorization":auth if ok else "none","protocol_digest":load(P)["protocol_digest"]};path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(canonical({"stage":stage,"passed":d["passed_count"],"total":d["total_count"],"authorization":d["authorization"]}));
 if not ok:raise SystemExit(1)
def pre():
 p=load(P);c=base(p);add(c,"clear",not any(x.exists() for x in (RAW,GEN,S,F,C)),"clear");write(PRE,c,"pre_model","run_phase1305_once")
def rate(x):
 x=list(x);return float(np.mean(x)) if x else 0.0
def post():
 p=load(P);c=base(p);raw=rows(RAW);gen=rows(GEN);summary=load(S);th=p["thresholds"];add(c,"counts",len(raw)==6912 and len(gen)==1536,[len(raw),len(gen)]);add(c,"hashes",summary["raw_hashes"]=={"candidate_scores":sha(RAW),"list_free_generations":sha(GEN)},summary["raw_hashes"]);overall=rate(x["correct"] for x in raw);parts={k:rate(x["correct"] for x in raw if x["partition"]==k) for k in ("discovery","confirmation","holdout")};panels={k:rate(x["correct"] for x in raw if x["panel"]==k) for k in ("active","matched_null","surface_only","semantic_neighbor")};surfaces={k:rate(x["correct"] for x in raw if x["surface"]==k) for k in ("catalog_prose","inventory_ledger")};states={str(k):rate(x["correct"] for x in raw if x["binding_state"]==k) for k in (0,1)};pairs=defaultdict(list);orders=defaultdict(list);cross=defaultdict(list)
 for x in raw:pairs[x["group_id"]].append(x);orders[(x["partition"],x["profile_index"],x["attribute"],x["panel"],x["surface"],x["binding_state"])].append(x);cross[(x["partition"],x["profile_index"],x["attribute"],x["panel"],x["candidate_order"],x["binding_state"])].append(x)
 pairrate={k:rate(len(v)==2 and all(x["correct"] for x in v) for v in pairs.values() if v[0]["panel"]==k) for k in panels};order=rate(len(v)==3 and all(x["correct"] for x in v) for v in orders.values());crossrate=rate(len(v)==2 and all(x["correct"] for x in v) for v in cross.values());finite=rate(x["finite"] for x in raw);candidate_gates={"finite":finite>=th["finite_fraction_min"],"overall_candidate":overall>=th["overall_candidate_accuracy_min"],"partition_candidate":min(parts.values())>=th["partition_candidate_accuracy_min"],"panel_candidate":min(panels.values())>=th["panel_candidate_accuracy_min"],"surface_candidate":min(surfaces.values())>=th["surface_candidate_accuracy_min"],"binding_state":min(states.values())>=th["base_side_accuracy_min"],"active_pair":pairrate["active"]>=th["active_pair_success_min"],"matched_null_pair":pairrate["matched_null"]>=th["matched_null_pair_success_min"],"surface_only_pair":pairrate["surface_only"]>=th["surface_only_pair_success_min"],"semantic_neighbor_pair":pairrate["semantic_neighbor"]>=th["semantic_neighbor_pair_success_min"],"candidate_order_triple":order>=th["candidate_order_triple_success_min"],"cross_surface_pair":crossrate>=th["cross_surface_pair_success_min"],"shortcut":p["zero_models"]["shortcut_ceiling"]<=th["shortcut_program_accuracy_max"]};add(c,"candidate_gates",summary["candidate"]["gates"]==candidate_gates and summary["candidate"]["passed"]==all(candidate_gates.values()),candidate_gates)
 coverage=rate(x["covered"] for x in gen);exact=rate(x["exact_correct"] for x in gen);groups=defaultdict(list)
 for x in gen:groups[(x["partition"],x["profile_index"],x["attribute"],x["panel"],x["surface"])].append(x)
 pair=rate(len(v)==2 and all(x["exact_correct"] for x in v) for v in groups.values());gen_gates={"coverage":coverage>=th["generation_coverage_min"],"accuracy":exact>=th["generation_accuracy_min"],"pair_success":pair>=th["generation_pair_success_min"]};add(c,"generation_gates",summary["generation"]["gates"]==gen_gates and summary["generation"]["passed"]==all(gen_gates.values()),gen_gates);passed=all(candidate_gates.values()) and all(gen_gates.values());auth="phase1306_frozen_hidden_only" if passed else "close_c033_without_hidden";add(c,"authorization",load(F)["authorization"]==auth and load(F)["hidden_states_read"] is False,load(F));qa=summary["model_audit"];add(c,"fp16",qa["has_fp16_parameters"] and not qa["has_quantized_modules"],qa);add(c,"complete",load(C)["formal_runs_consumed"]==1,load(C));write(POST,c,"post_model",auth)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=("preaudit","postaudit"));a=ap.parse_args();pre() if a.stage=="preaudit" else post()
