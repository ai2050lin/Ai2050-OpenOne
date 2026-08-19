#!/usr/bin/env python3
"""Phase1306: fixed answer-boundary depth-25/26 response replication for C033."""
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
PHASE=1306;CAMPAIGN="C033";SCRIPT=Path(__file__).resolve();AUDITOR=T/"phase1306_c033_frozen_answer_boundary_hidden_audit.py";PARENT=T/"result/phase1305_c033_qwen3_behavior";CONTRACT=T/"result/phase1304_c033_role_typed_causal_graph_contract";MATERIAL=CONTRACT/"material/frozen_role_typed_lookup_cases.jsonl";OUT=T/"result/phase1306_c033_frozen_answer_boundary_hidden";PROTOCOL=OUT/"protocol/preregistration.json";MANIFEST=OUT/"protocol/frozen_hidden_manifest.jsonl";PRE=OUT/"audit/independent_preaudit.json";POST=OUT/"audit/independent_final_audit.json";ARRAYS=OUT/"raw/hidden_arrays.npz";META=OUT/"raw/run_metadata.json";SUMMARY=OUT/"analysis/hidden_summary.json";FINAL=OUT/"analysis/final.json";COMPLETE=OUT/"protocol/formal_run_complete.json";SYSTEM="Use only the supplied catalog. Reply exactly as requested and do not explain.";PARTS=("discovery","confirmation","holdout");PANELS=("active","matched_null","surface_only","semantic_neighbor");SURF=("catalog_prose","inventory_ledger");ATTRS=("color","material","location","size","shape","status");DEPTHS=(25,26);EVENT="assistant_answer_boundary";BATCH=4;EPS=1e-12
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
def state(tok,row):
 text=render(tok,row["candidate_prompt"]);enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=[int(x) for x in enc["input_ids"]];offsets=[(int(a),int(b)) for a,b in enc["offset_mapping"]];base=text.find(row["candidate_prompt"]);span=row["typed_spans"]["answer_boundary"][0];left=base+span[0];right=base+span[1];positions=[i for i,(a,b) in enumerate(offsets) if b>left and a<right and b>a]
 if not positions:raise RuntimeError("answer boundary span missing")
 candidate_ids=[]
 for name in row["candidates"]:
  full=tok.encode(text+" "+name,add_special_tokens=False)
  if full[:len(ids)]!=ids or len(full)!=len(ids)+1:raise RuntimeError("candidate token drift")
  candidate_ids.append(int(full[-1]))
 return {"case_id":row["case_id"],"ids":ids,"position":positions[-1],"candidate_ids":candidate_ids,"gold_position":row["gold_position"]}
def build_manifest(tok,rr):
 groups=defaultdict(list)
 for r in rr:
  if r["candidate_order"]==0:groups[r["group_id"]].append(r)
 bybase=defaultdict(dict)
 for pair in groups.values():
  pair=sorted(pair,key=lambda x:x["binding_state"]);r=pair[0];bybase[(r["partition"],r["profile_index"],r["attribute"],r["surface"])][r["panel"]]=pair
 out=[]
 for base in sorted(bybase):
  active=bybase[base]["active"];e0=active[0]["gold_candidate"];e1=active[1]["gold_candidate"]
  for panel in PANELS:
   pair=bybase[base][panel];entities=pair[0]["candidates"];ipos=[entities.index(e0),entities.index(e1)];states=[state(tok,pair[0]),state(tok,pair[1])];out.append({"group_id":pair[0]["group_id"],"partition":base[0],"profile_index":base[1],"attribute":base[2],"surface":base[3],"panel":panel,"identity_positions":ipos,"identity_token_ids":[states[0]["candidate_ids"][i] for i in ipos],"states":states})
 return out
def preregister(force):
 if load(PARENT/"analysis/final.json").get("authorization")!="phase1306_frozen_hidden_only" or not load(PARENT/"audit/independent_final_audit.json").get("all_checks_passed"):raise RuntimeError("parent auth")
 cp=load(CONTRACT/"protocol/preregistration.json");th=cp["hidden"]["thresholds"]
 if cp["hidden"]["depths"]!=[25,26] or cp["hidden"]["event"]!=EVENT:raise RuntimeError("contract drift")
 if OUT.exists() and not force:raise RuntimeError(f"{OUT} exists")
 if OUT.exists():shutil.rmtree(OUT)
 tok=AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"],trust_remote_code=True,local_files_only=True,use_fast=True);mm=build_manifest(tok,rows(MATERIAL));write(MANIFEST,mm);timeless={"phase":PHASE,"campaign":CAMPAIGN,"schema_version":"phase1306.c033.fixed_hidden.v1","model":"qwen3-4b-fp16-cuda-no-quantization","formal_run_budget":1,"runtime":{"compiler":"right_padding","global_fixed_length":True,"batch":BATCH},"material":{"sha256":sha(MATERIAL),"manifest_sha256":sha(MANIFEST),"pair_count":len(mm),"state_count":2*len(mm)},"event":EVENT,"depths":list(DEPTHS),"panels":list(PANELS),"measurements":{"norm":"paired L2 delta divided by mean state norm","identity":"paired change in identity1-minus-identity0 logit-lens contrast","behavior_replay":"candidate argmax at final answer boundary"},"thresholds":th,"gate":"every partition and both frozen depths must pass; no selection","success_authorization":"phase1307_bidirectional_swap_only","failure_authorization":"close_c033_without_causal","hard_stops":["No new event or depth","No discovery selection","No component scan","No causal patch","No threshold change"],"dependencies":{"parent_protocol":sha(PARENT/"protocol/preregistration.json"),"parent_final":sha(PARENT/"analysis/final.json"),"parent_audit":sha(PARENT/"audit/independent_final_audit.json"),"contract":sha(CONTRACT/"protocol/preregistration.json"),"material":sha(MATERIAL),"manifest":sha(MANIFEST)},"source_hashes":{"main":sha(SCRIPT),"auditor":sha(AUDITOR)},"model_weights_loaded":False};p={**timeless,"created_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_digest":digest(timeless)};save(PROTOCOL,p);print(canonical({"pairs":len(mm),"digest":p["protocol_digest"]}))
def cell(norm,identity,meta,partition,di):
 lookup={(x["profile_index"],x["attribute"],x["surface"],x["panel"]):i for i,x in enumerate(meta) if x["partition"]==partition};active=[];active_id=[];controls={p:[] for p in PANELS if p!="active"};control_id={p:[] for p in controls};wins=[]
 for profile in range(8):
  for attr in ATTRS:
   for surf in SURF:
    ai=lookup[(profile,attr,surf,"active")];n=float(norm[ai,di]);iv=float(identity[ai,di]);cv={p:float(norm[lookup[(profile,attr,surf,p)],di]) for p in controls};active.append(n);active_id.append(iv);wins.append(n>max(cv.values()))
    for p in controls:controls[p].append(cv[p]);control_id[p].append(abs(float(identity[lookup[(profile,attr,surf,p)],di])))
 nm=float(np.median(active));cn=max(float(np.median(v)) for v in controls.values());im=float(np.median(active_id));ci=max(float(np.median(v)) for v in control_id.values());return {"active_norm_median":nm,"max_control_norm_median":cn,"norm_ratio":nm/(cn+EPS),"norm_win_fraction":float(np.mean(wins)),"active_identity_median":im,"max_control_abs_identity_median":ci,"identity_ratio":im/(ci+EPS),"identity_positive_fraction":float(np.mean(np.asarray(active_id)>0))}
def passes(x,th):return x["active_norm_median"]>=th["active_norm_median_min"] and x["norm_ratio"]>=th["active_to_max_control_ratio_min"] and x["norm_win_fraction"]>=th["active_over_controls_fraction_min"] and x["identity_positive_fraction"]>=th["identity_positive_fraction_min"] and x["identity_ratio"]>=th["identity_to_max_control_ratio_min"]
def analyze(norm,identity,behavior,meta,th):
 cells={p:{str(d):cell(norm,identity,meta,p,di) for di,d in enumerate(DEPTHS)} for p in PARTS};gates={"finite":bool(np.isfinite(norm).all() and np.isfinite(identity).all()),"behavior_replay":float(np.mean(behavior))>=th["behavior_replay_accuracy_min"]}
 for p in PARTS:
  for d in DEPTHS:gates[f"{p}_depth{d}"]=passes(cells[p][str(d)],th)
 return {"finite_fraction":float(np.mean(np.isfinite(norm))),"behavior_replay_accuracy":float(np.mean(behavior)),"cells":cells,"gates":gates,"all_gates_passed":all(gates.values())}
@torch.inference_mode()
def run():
 p=load(PROTOCOL);pre=load(PRE)
 if pre.get("authorization")!="run_phase1306_once" or not pre.get("all_checks_passed"):raise RuntimeError("preaudit")
 if any(x.exists() for x in (ARRAYS,META,SUMMARY,FINAL,COMPLETE)):raise RuntimeError("consumed")
 mm=rows(MANIFEST);model=tok=None;started=time.time()
 try:
  model,tok,device,placement=load_fp16("qwen3");qa=quantization_audit(model)
  if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:raise RuntimeError(qa)
  norm=np.empty((1152,2),np.float32);identity=np.empty_like(norm);behavior=np.empty((1152,2),np.bool_);raw_max=max(len(s["ids"]) for x in mm for s in x["states"]);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id);supports="logits_to_keep" in inspect.signature(model.forward).parameters
  for start in range(0,1152,BATCH):
   group=mm[start:start+BATCH]
   if [x["panel"] for x in group]!=list(PANELS):raise RuntimeError("manifest order")
   specs=[s for x in group for s in x["states"]];ids=torch.full((8,raw_max),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
   for i,s in enumerate(specs):ids[i,:len(s["ids"])]=torch.tensor(s["ids"],device=device);mask[i,:len(s["ids"])]=1
   pos=mask.cumsum(-1)-1;kw={"input_ids":ids,"attention_mask":mask,"position_ids":pos,"use_cache":False,"output_hidden_states":True,"return_dict":True}
   if supports:kw["logits_to_keep"]=1
   out=model(**kw);event_pos=torch.tensor([s["position"] for s in specs],device=device)
   for di,d in enumerate(DEPTHS):
    ev=out.hidden_states[d][torch.arange(8,device=device),event_pos];nev=model.model.norm(ev)
    for local,x in enumerate(group):
     i0,i1=2*local,2*local+1;v0=ev[i0].float();v1=ev[i1].float();norm[start+local,di]=float((torch.linalg.vector_norm(v1-v0)/(0.5*(torch.linalg.vector_norm(v0)+torch.linalg.vector_norm(v1))+EPS)).item());token_ids=torch.tensor(x["identity_token_ids"],device=device);w=model.lm_head.weight[token_ids];s0=w@nev[i0];s1=w@nev[i1];identity[start+local,di]=float((s1[1]-s1[0]-s0[1]+s0[0]).float().item())
   final=out.hidden_states[-1]
   for local,x in enumerate(group):
    for si in (0,1):
     ii=2*local+si;h=final[ii,specs[ii]["position"]];cids=torch.tensor(specs[ii]["candidate_ids"],device=device);scores=model.lm_head.weight[cids]@model.model.norm(h);behavior[start+local,si]=int(torch.argmax(scores).item())==int(specs[ii]["gold_position"])
   del out
  meta=[{k:x[k] for k in ("group_id","partition","profile_index","attribute","surface","panel")} for x in mm];analysis=analyze(norm,identity,behavior,meta,p["thresholds"]);auth="phase1307_bidirectional_swap_only" if analysis["all_gates_passed"] else "close_c033_without_causal";ARRAYS.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(ARRAYS,relative_norm=norm,identity_delta=identity,behavior_correct=behavior,depths=np.asarray(DEPTHS));save(META,{"phase":PHASE,"campaign":CAMPAIGN,"protocol_digest":p["protocol_digest"],"array_sha256":sha(ARRAYS),"manifest_sha256":sha(MANIFEST),"model_audit":qa,"placement":placement,"runtime_seconds":time.time()-started,"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,"pair_metadata":meta});save(SUMMARY,{**analysis,"phase":PHASE,"campaign":CAMPAIGN,"authorization":auth});save(FINAL,{"phase":PHASE,"campaign":CAMPAIGN,"verdict":"fixed_hidden_qualified" if analysis["all_gates_passed"] else "fixed_hidden_gate_failed","all_gates_passed":analysis["all_gates_passed"],"authorization":auth,"protocol_digest":p["protocol_digest"]});save(COMPLETE,{"completed_at_utc":datetime.now(timezone.utc).isoformat(),"formal_runs_consumed":1,"protocol_digest":p["protocol_digest"]});print(canonical({"gates":analysis["gates"],"authorization":auth}))
 finally:
  if model is not None:release_fp16(model)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("command",choices=("preregister","run"));ap.add_argument("--force",action="store_true");a=ap.parse_args();preregister(a.force) if a.command=="preregister" else run()
