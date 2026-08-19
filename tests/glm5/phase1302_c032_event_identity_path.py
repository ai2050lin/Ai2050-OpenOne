#!/usr/bin/env python3
"""Phase1302: C032 five-event residual magnitude and identity-response path."""
from __future__ import annotations
import argparse,hashlib,inspect,json,shutil,sys,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";sys.path.insert(0,str(T))
from model_utils import MODEL_CONFIGS  # noqa:E402
from phase1023_fp16_utils import load_fp16,quantization_audit,release_fp16  # noqa:E402
PHASE=1302;CAMPAIGN="C032";SCRIPT=Path(__file__).resolve();AUDITOR=T/"phase1302_c032_event_identity_path_audit.py";PARENT=T/"result/phase1301_c032_qwen3_behavior";PARENT_PROTOCOL=PARENT/"protocol/preregistration.json";PARENT_FINAL=PARENT/"analysis/final.json";PARENT_AUDIT=PARENT/"audit/independent_final_audit.json";RUNTIME=T/"result/phase1300_c032_execution_compiler_competition/protocol/frozen_runtime.json";CONTRACT=T/"result/phase1299_c032_execution_compiler_contract";CONTRACT_PROTOCOL=CONTRACT/"protocol/preregistration.json";MATERIAL=CONTRACT/"material/frozen_inverse_lookup_cases.jsonl";OUT=T/"result/phase1302_c032_event_identity_path";PROTOCOL=OUT/"protocol/preregistration.json";MANIFEST=OUT/"protocol/frozen_event_manifest.jsonl";PRE=OUT/"audit/independent_preaudit.json";POST=OUT/"audit/independent_final_audit.json";ARRAYS=OUT/"raw/event_identity_arrays.npz";META=OUT/"raw/run_metadata.json";SUMMARY=OUT/"analysis/path_summary.json";FINAL=OUT/"analysis/final.json";COMPLETE=OUT/"protocol/formal_run_complete.json"
SYSTEM="Use only the supplied catalog. Reply exactly as requested and do not explain.";PARTITIONS=("discovery","confirmation","holdout");PANELS=("active","matched_null","surface_only","semantic_neighbor");SURFACES=("catalog_prose","inventory_ledger");ATTRS=("color","material","location","size","shape","status");EVENTS=("record_slot0_entity","record_slot0_value","query_clause_end","user_answer_cue_end","assistant_answer_boundary");PRIMARY=("user_answer_cue_end","assistant_answer_boundary");DEPTHS=tuple(range(37));PAIR_BATCH=4;EPS=1e-12;TAU=1e-7
TH={"finite_fraction_min":1.0,"behavior_replay_accuracy_min":0.99,"discovery_norm_median_min":0.001,"discovery_norm_ratio_min":1.20,"discovery_norm_win_fraction_min":0.75,"discovery_identity_positive_fraction_min":0.75,"discovery_identity_ratio_min":1.15,"transfer_norm_median_min":0.001,"transfer_norm_ratio_min":1.10,"transfer_norm_win_fraction_min":0.70,"transfer_identity_positive_fraction_min":0.70,"transfer_identity_ratio_min":1.05,"adjacent_depths_min":2}
def canonical(v:Any)->str:return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(v:Any)->str:return hashlib.sha256(canonical(v).encode()).hexdigest()
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  while c:=f.read(1024*1024):h.update(c)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def write(p,rr):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",encoding="utf-8",newline="\n") as f:
  for r in rr:f.write(canonical(r)+"\n")
def render(tok,prompt):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
def overlap(offsets,left,right):
 x=[i for i,(a,b) in enumerate(offsets) if b>left and a<right and b>a]
 if not x:raise RuntimeError((left,right))
 return x
def state(tok,row):
 text=render(tok,row["candidate_prompt"]);enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=[int(x) for x in enc["input_ids"]];offs=[(int(a),int(b)) for a,b in enc["offset_mapping"]];base=text.find(row["candidate_prompt"]);rec=row["typed_spans"]["records"][0];query=row["typed_spans"]["query"][0];cue=row["typed_spans"]["answer_boundary"][0];spans={"record_slot0_entity":rec["entity_spans"][0],"record_slot0_value":rec["queried_attribute_value_spans"][0],"query_clause_end":query,"user_answer_cue_end":cue};pos={k:overlap(offs,base+a,base+b)[-1] for k,(a,b) in spans.items()};pos["assistant_answer_boundary"]=len(ids)-1;cids=[]
 for n in row["candidates"]:
  full=tok.encode(text+" "+n,add_special_tokens=False)
  if full[:len(ids)]!=ids or len(full)!=len(ids)+1:raise RuntimeError("candidate drift")
  cids.append(int(full[-1]))
 return {"case_id":row["case_id"],"ids":ids,"positions":pos,"candidate_ids":cids,"gold_position":row["gold_position"],"input_digest":digest(ids)}
def build_manifest(tok,rr):
 chosen=[r for r in rr if r["candidate_order"]==0];groups=defaultdict(list)
 for r in chosen:groups[r["group_id"]].append(r)
 bybase=defaultdict(dict)
 for gid,pair in groups.items():
  pair=sorted(pair,key=lambda x:x["binding_state"]);r=pair[0];base=(r["partition"],r["profile_index"],r["attribute"],r["surface"]);bybase[base][r["panel"]]=pair
 out=[]
 for base in sorted(bybase):
  active=bybase[base]["active"];e0=active[0]["gold_candidate"];e1=active[1]["gold_candidate"]
  for panel in PANELS:
   pair=bybase[base][panel];entities=pair[0]["candidates"]
   identity_positions=[entities.index(e0),entities.index(e1)];states=[state(tok,pair[0]),state(tok,pair[1])]
   out.append({"group_id":pair[0]["group_id"],"partition":base[0],"profile_index":base[1],"attribute":base[2],"surface":base[3],"panel":panel,"identity_entity0":e0,"identity_entity1":e1,"identity_positions":identity_positions,"identity_token_ids":[states[0]["candidate_ids"][i] for i in identity_positions],"states":states})
 return out
def preregister(force):
 if load(PARENT_FINAL).get("authorization")!="phase1302_event_identity_hidden_only" or not load(PARENT_AUDIT).get("all_checks_passed"):raise RuntimeError("parent auth")
 rt=load(RUNTIME);cp=load(CONTRACT_PROTOCOL)
 if rt["selected_runtime"]!="right_padding" or rt["tau"]!=TAU or cp["hidden"]["thresholds"]!=TH:raise RuntimeError("frozen contract drift")
 if OUT.exists() and not force:raise RuntimeError(f"{OUT} exists")
 if OUT.exists():shutil.rmtree(OUT)
 tok=AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"],trust_remote_code=True,local_files_only=True,use_fast=True);mm=build_manifest(tok,rows(MATERIAL));write(MANIFEST,mm)
 timeless={"phase":PHASE,"campaign":CAMPAIGN,"schema_version":"phase1302.c032.event_identity.v1","model":"qwen3-4b-fp16-cuda-no-quantization","formal_run_budget":1,"runtime":{"compiler":"right_padding","global_fixed_length":True,"tau":TAU,"pair_batch":PAIR_BATCH},"material":{"sha256":sha(MATERIAL),"manifest_sha256":sha(MANIFEST),"pair_count":len(mm),"state_count":2*len(mm),"partition_counts":{p:sum(x["partition"]==p for x in mm) for p in PARTITIONS}},"events":list(EVENTS),"primary_events":list(PRIMARY),"depths":list(DEPTHS),"measurements":{"norm":"L2 state delta normalized by mean state norm","identity":"change across binding states in logit-lens contrast identity_entity1 minus identity_entity0","controls":"matched_null, surface_only, semantic_neighbor","instrument_prefix_identity":"active and matched-null raw states through record events must be within frozen tau"},"selection":"earliest adjacent discovery depth pair jointly passing norm and identity thresholds separately for each primary event","transfer":"same event-specific depths must pass confirmation and holdout without reselection","thresholds":TH,"success":"all finite, behavior replay, prefix identities, both discovery bands, and both transfer partitions pass","success_authorization":"phase1303_frozen_causal_rescue","failure":"close_c032_without_causal_claim","hard_stops":["No head or MLP scan","No causal patch","No event/depth reselection","No threshold change"],"dependencies":{"parent_protocol":sha(PARENT_PROTOCOL),"parent_final":sha(PARENT_FINAL),"parent_audit":sha(PARENT_AUDIT),"runtime":sha(RUNTIME),"contract":sha(CONTRACT_PROTOCOL),"material":sha(MATERIAL),"manifest":sha(MANIFEST)},"source_hashes":{"main":sha(SCRIPT),"auditor":sha(AUDITOR)},"model_weights_loaded":False};p={**timeless,"created_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_digest":digest(timeless)};save(PROTOCOL,p);print(canonical({"pairs":len(mm),"digest":p["protocol_digest"]}))
def cell(norm,identity,meta,partition,event_i,depth):
 lookup={(x["profile_index"],x["attribute"],x["surface"],x["panel"]):i for i,x in enumerate(meta) if x["partition"]==partition};av=[];aid=[];controls={p:[] for p in PANELS if p!="active"};wins=[]
 for profile in range(8):
  for attr in ATTRS:
   for surf in SURFACES:
    ai=lookup[(profile,attr,surf,"active")];n=float(norm[ai,depth,event_i]);iv=float(identity[ai,depth,event_i]);cv={p:float(norm[lookup[(profile,attr,surf,p)],depth,event_i]) for p in controls};av.append(n);aid.append(iv);wins.append(n>max(cv.values()))
    for p in controls:controls[p].append(float(abs(identity[lookup[(profile,attr,surf,p)],depth,event_i])))
 norm_m=float(np.median(av));control_norm=[]
 for p in controls:
  vals=[float(norm[lookup[(profile,a,s,p)],depth,event_i]) for profile in range(8) for a in ATTRS for s in SURFACES];control_norm.append(float(np.median(vals)))
 max_norm=max(control_norm);id_m=float(np.median(aid));max_id=max(float(np.median(v)) for v in controls.values())
 return {"active_norm_median":norm_m,"max_control_norm_median":max_norm,"norm_ratio":norm_m/(max_norm+EPS),"norm_win_fraction":float(np.mean(wins)),"active_identity_median":id_m,"max_control_abs_identity_median":max_id,"identity_ratio":id_m/(max_id+EPS),"identity_positive_fraction":float(np.mean(np.asarray(aid)>0))}
def passcell(x,discovery):
 q="discovery" if discovery else "transfer";return x["active_norm_median"]>=TH[f"{q}_norm_median_min"] and x["norm_ratio"]>=TH[f"{q}_norm_ratio_min"] and x["norm_win_fraction"]>=TH[f"{q}_norm_win_fraction_min"] and x["identity_positive_fraction"]>=TH[f"{q}_identity_positive_fraction_min"] and x["identity_ratio"]>=TH[f"{q}_identity_ratio_min"]
def analyze(norm,identity,behavior,prefix_errors,meta):
 tables={p:{e:[cell(norm,identity,meta,p,ei,d) for d in DEPTHS] for ei,e in enumerate(EVENTS)} for p in PARTITIONS};selected={}
 for e in PRIMARY:
  ok=[passcell(tables["discovery"][e][d],True) for d in DEPTHS];start=next((d for d in range(36) if ok[d] and ok[d+1]),None);selected[e]=[] if start is None else [start,start+1]
 transfer={p:{e:{"depths":selected[e],"cells":[tables[p][e][d] for d in selected[e]],"passed":bool(selected[e]) and all(passcell(tables[p][e][d],False) for d in selected[e])} for e in PRIMARY} for p in ("confirmation","holdout")};gates={"finite":all(np.isfinite(x).all() for x in (norm,identity,prefix_errors)),"behavior_replay":float(np.mean(behavior))>=TH["behavior_replay_accuracy_min"],"prefix_identity":float(prefix_errors.max())<=TAU,"discovery_user_answer_cue":bool(selected["user_answer_cue_end"]),"discovery_assistant_boundary":bool(selected["assistant_answer_boundary"]),"confirmation_user_answer_cue":transfer["confirmation"]["user_answer_cue_end"]["passed"],"confirmation_assistant_boundary":transfer["confirmation"]["assistant_answer_boundary"]["passed"],"holdout_user_answer_cue":transfer["holdout"]["user_answer_cue_end"]["passed"],"holdout_assistant_boundary":transfer["holdout"]["assistant_answer_boundary"]["passed"]};return {"finite_fraction":float(np.mean(np.isfinite(norm))),"behavior_replay_accuracy":float(np.mean(behavior)),"prefix_identity_relative_max":float(prefix_errors.max()),"selected_discovery_bands":selected,"transfer":transfer,"tables":tables,"gates":gates,"all_gates_passed":all(gates.values())}
@torch.inference_mode()
def run():
 p=load(PROTOCOL);pre=load(PRE)
 if pre.get("authorization")!="run_phase1302_once" or not pre.get("all_checks_passed"):raise RuntimeError("preaudit")
 if any(x.exists() for x in (ARRAYS,META,SUMMARY,FINAL,COMPLETE)):raise RuntimeError("consumed")
 mm=rows(MANIFEST);model=tok=None;started=time.time()
 try:
  model,tok,device,placement=load_fp16("qwen3");qa=quantization_audit(model)
  if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:raise RuntimeError(qa)
  norm=np.empty((1152,37,5),np.float32);identity=np.empty_like(norm);behavior=np.empty((1152,2),np.bool_);prefix_errors=np.empty((288,37,2,2),np.float32);raw_max=max(len(s["ids"]) for x in mm for s in x["states"]);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id);supports="logits_to_keep" in inspect.signature(model.forward).parameters;prefix_cursor=0
  for start in range(0,1152,PAIR_BATCH):
   group=mm[start:start+PAIR_BATCH]
   if len({(x["partition"],x["profile_index"],x["attribute"],x["surface"]) for x in group})!=1 or [x["panel"] for x in group]!=list(PANELS):raise RuntimeError("group ordering")
   specs=[s for x in group for s in x["states"]];ids=torch.full((8,raw_max),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
   for i,s in enumerate(specs):ids[i,:len(s["ids"])]=torch.tensor(s["ids"],device=device);mask[i,:len(s["ids"])]=1
   pos=mask.cumsum(-1)-1;kw={"input_ids":ids,"attention_mask":mask,"position_ids":pos,"use_cache":False,"output_hidden_states":True,"return_dict":True};
   if supports:kw["logits_to_keep"]=1
   out=model(**kw);event_pos=torch.tensor([[s["positions"][e] for e in EVENTS] for s in specs],device=device);batch_index=torch.arange(8,device=device)[:,None]
   for d,h in enumerate(out.hidden_states):
    ev=h[batch_index,event_pos];normed_ev=model.model.norm(ev)
    for local,x in enumerate(group):
     i0,i1=2*local,2*local+1;v0=ev[i0].float();v1=ev[i1].float();dn=torch.linalg.vector_norm(v1-v0,dim=-1);bn=0.5*(torch.linalg.vector_norm(v0,dim=-1)+torch.linalg.vector_norm(v1,dim=-1));norm[start+local,d]=((dn/(bn+EPS)).cpu().numpy());token_ids=torch.tensor(x["identity_token_ids"],dtype=torch.long,device=device);weights=model.lm_head.weight[token_ids];score0=torch.einsum("ed,kd->ek",normed_ev[i0],weights);score1=torch.einsum("ed,kd->ek",normed_ev[i1],weights);identity[start+local,d]=(score1[:,1]-score1[:,0]-score0[:,1]+score0[:,0]).float().cpu().numpy()
    active0,active1=0,1;null0,null1=2,3
    for ri,role in enumerate(EVENTS[:2]):
     for si,(a,n) in enumerate(((active0,null0),(active1,null1))):
      va=ev[a,ri].float();vn=ev[n,ri].float();prefix_errors[prefix_cursor,d,ri,si]=float((torch.linalg.vector_norm(va-vn)/(0.5*(torch.linalg.vector_norm(va)+torch.linalg.vector_norm(vn))+EPS)).item())
   final=out.hidden_states[-1]
   for local,x in enumerate(group):
    for si in (0,1):
     ii=2*local+si;h=final[ii,specs[ii]["positions"]["assistant_answer_boundary"]];candidate_ids=torch.tensor(specs[ii]["candidate_ids"],dtype=torch.long,device=device);scores=model.lm_head.weight[candidate_ids]@model.model.norm(h);behavior[start+local,si]=int(torch.argmax(scores).item())==int(specs[ii]["gold_position"])
   prefix_cursor+=1;del out
   if prefix_cursor%50==0:print(canonical({"base_groups":prefix_cursor,"total":288}),flush=True)
  ARRAYS.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(ARRAYS,relative_norm=norm,identity_delta=identity,behavior_correct=behavior,prefix_identity=prefix_errors,events=np.asarray(EVENTS),depths=np.asarray(DEPTHS));meta=[{k:x[k] for k in ("group_id","partition","profile_index","attribute","surface","panel","identity_entity0","identity_entity1")} for x in mm];analysis=analyze(norm,identity,behavior,prefix_errors,meta);auth="phase1303_frozen_causal_rescue" if analysis["all_gates_passed"] else "close_c032_without_causal_claim";save(META,{"phase":PHASE,"campaign":CAMPAIGN,"protocol_digest":p["protocol_digest"],"array_sha256":sha(ARRAYS),"manifest_sha256":sha(MANIFEST),"model_audit":qa,"placement":placement,"runtime_seconds":time.time()-started,"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,"pair_metadata":meta});save(SUMMARY,{**analysis,"phase":PHASE,"campaign":CAMPAIGN,"protocol_digest":p["protocol_digest"],"authorization":auth});save(FINAL,{"phase":PHASE,"campaign":CAMPAIGN,"verdict":"event_identity_path_qualified" if analysis["all_gates_passed"] else "event_identity_path_gate_failed","all_gates_passed":analysis["all_gates_passed"],"selected_discovery_bands":analysis["selected_discovery_bands"],"authorization":auth,"protocol_digest":p["protocol_digest"],"array_sha256":sha(ARRAYS),"causal_intervention_performed":False});save(COMPLETE,{"completed_at_utc":datetime.now(timezone.utc).isoformat(),"formal_runs_consumed":1,"protocol_digest":p["protocol_digest"]});print(canonical({"selected":analysis["selected_discovery_bands"],"gates":analysis["gates"],"authorization":auth}))
 finally:
  if model is not None:release_fp16(model)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("command",choices=("preregister","run"));ap.add_argument("--force",action="store_true");a=ap.parse_args();preregister(a.force) if a.command=="preregister" else run()
