#!/usr/bin/env python3
"""Phase1300: one-shot four-arm execution compiler competition for C032."""

from __future__ import annotations
import argparse, hashlib, inspect, json, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]; T=ROOT/"tests/glm5"; sys.path.insert(0,str(T))
from model_utils import MODEL_CONFIGS  # noqa: E402
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402

PHASE=1300; CAMPAIGN="C032"; SCRIPT=Path(__file__).resolve(); AUDITOR=T/"phase1300_c032_execution_compiler_competition_audit.py"
PARENT=T/"result/phase1299_c032_execution_compiler_contract"; PARENT_PROTOCOL=PARENT/"protocol/preregistration.json"; PARENT_FINAL=PARENT/"analysis/final.json"; PARENT_AUDIT=PARENT/"audit/independent_final_audit.json"; MATERIAL=PARENT/"material/frozen_inverse_lookup_cases.jsonl"
OUT=T/"result/phase1300_c032_execution_compiler_competition"; PROTOCOL=OUT/"protocol/preregistration.json"; MANIFEST=OUT/"protocol/frozen_compiler_manifest.jsonl"; PRE=OUT/"audit/independent_preaudit.json"; POST=OUT/"audit/independent_final_audit.json"; ARRAYS=OUT/"raw/compiler_errors.npz"; META=OUT/"raw/run_metadata.json"; SUMMARY=OUT/"analysis/compiler_summary.json"; RUNTIME=OUT/"protocol/frozen_runtime.json"; FINAL=OUT/"analysis/final.json"; COMPLETE=OUT/"protocol/formal_run_complete.json"
SYSTEM="Use only the supplied catalog. Reply exactly as requested and do not explain."; DEPTHS=tuple(range(37)); ROLES=("record_slot0_entity","record_slot0_value"); ARMS=("left_global_baseline","right_padding","record_event_aligned","equalized_suffix"); CANDIDATES=ARMS[1:]; PRIORITY=CANDIDATES; BATCH_GROUPS=4; EPS=1e-12
THRESHOLDS={"case_count_min":96,"finite_fraction_min":1.0,"exact_duplicate_relative_max":1e-6,"same_prefix_relative_max":1e-6,"cross_composition_relative_max":1e-6,"candidate_compilers_passing_min":2,"tau_multiplier":4.0,"tau_floor":1e-7,"tau_cap":1e-4}

def canonical(v:Any)->str:return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(v:Any)->str:return hashlib.sha256(canonical(v).encode()).hexdigest()
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  while c:=f.read(1024*1024):h.update(c)
 return h.hexdigest()
def load(p:Path)->Any:return json.loads(p.read_text(encoding="utf-8"))
def read_jsonl(p:Path)->list[dict[str,Any]]:return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def save(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def write_jsonl(p:Path,rows:list[dict[str,Any]])->None:
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",encoding="utf-8",newline="\n") as f:
  for r in rows:f.write(canonical(r)+"\n")
def render(tok:Any,prompt:str)->str:return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
def overlap(offsets,left,right):
 x=[i for i,(a,b) in enumerate(offsets) if b>left and a<right and b>a]
 if not x:raise RuntimeError((left,right))
 return x
def spec(tok:Any,row:dict[str,Any])->dict[str,Any]:
 text=render(tok,row["candidate_prompt"]); enc=tok(text,add_special_tokens=False,return_offsets_mapping=True); ids=[int(x) for x in enc["input_ids"]]; offsets=[(int(a),int(b)) for a,b in enc["offset_mapping"]]; base=text.find(row["candidate_prompt"]); rec=row["typed_spans"]["records"][0]; spans={"record_slot0_entity":rec["entity_spans"][0],"record_slot0_value":rec["queried_attribute_value_spans"][0]}; pos={k:overlap(offsets,base+a,base+b)[-1] for k,(a,b) in spans.items()}; return {"case_id":row["case_id"],"ids":ids,"positions":pos,"input_digest":digest(ids)}
def manifest(tok:Any,rows:list[dict[str,Any]])->list[dict[str,Any]]:
 idx={(r["profile_index"],r["attribute"],r["surface"],r["panel"],r["binding_state"],r["candidate_order"]):r for r in rows if r["partition"]=="discovery"}; out=[]
 for p in range(8):
  for a in ("color","material","location","size","shape","status"):
   for s in ("catalog_prose","inventory_ledger"):
    ar=idx[(p,a,s,"active",0,0)]; nr=idx[(p,a,s,"matched_null",0,0)]; aa=spec(tok,ar); nn=spec(tok,nr); end_a=ar["typed_spans"]["records"][0]["queried_attribute_value_spans"][0][1]; end_n=nr["typed_spans"]["records"][0]["queried_attribute_value_spans"][0][1]
    if ar["candidate_prompt"][:end_a]!=nr["candidate_prompt"][:end_n] or aa["ids"][:aa["positions"]["record_slot0_value"]+1]!=nn["ids"][:nn["positions"]["record_slot0_value"]+1]:raise RuntimeError("prefix mismatch")
    out.append({"calibration_id":f"p{p:02d}|{a}|{s}","profile_index":p,"attribute":a,"surface":s,"active":aa,"matched_null":nn,"prefix_identity":True})
 return out

def preregister(force:bool)->None:
 if load(PARENT_FINAL).get("authorization")!="phase1300_compiler_competition_only" or not load(PARENT_AUDIT).get("all_checks_passed"):raise RuntimeError("parent auth")
 if OUT.exists() and not force:raise RuntimeError(f"{OUT} exists")
 if OUT.exists():shutil.rmtree(OUT)
 tok=AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"],trust_remote_code=True,local_files_only=True,use_fast=True); mm=manifest(tok,read_jsonl(MATERIAL));write_jsonl(MANIFEST,mm)
 timeless={"phase":PHASE,"campaign":CAMPAIGN,"schema_version":"phase1300.c032.compiler.v1","model":"qwen3-4b-fp16-cuda-no-quantization","formal_run_budget":1,"arms":list(ARMS),"candidate_arms":list(CANDIDATES),"priority":list(PRIORITY),"roles":list(ROLES),"depths":list(DEPTHS),"cases":len(mm),"fixed_batch_size":3*BATCH_GROUPS,"compiler_definitions":{"left_global_baseline":"left pad every raw sequence to global raw maximum","right_padding":"place every raw sequence at physical column zero and right pad","record_event_aligned":"left/right pad so record_slot0_value is at one global physical anchor","equalized_suffix":"append attended pad-token fillers after each shorter full prompt to equalize each active-null pair, then global left pad"},"comparisons":["exact duplicate same batch","causal same prefix same batch","same full input changed batch composition"],"thresholds":THRESHOLDS,"selection":"at least two candidate arms pass all maximum gates; choose first passing priority arm","tau_rule":"max(floor,min(cap,multiplier*selected_max_noise)); must be below cap","success_authorization":"phase1301_qwen3_behavior_only","failure":"close_c032_without_semantic_run","dependencies":{"parent_protocol":sha(PARENT_PROTOCOL),"parent_final":sha(PARENT_FINAL),"parent_audit":sha(PARENT_AUDIT),"material":sha(MATERIAL),"manifest":sha(MANIFEST)},"source_hashes":{"main":sha(SCRIPT),"auditor":sha(AUDITOR)},"model_weights_loaded":False};p={**timeless,"created_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_digest":digest(timeless)};save(PROTOCOL,p);print(canonical({"cases":len(mm),"digest":p["protocol_digest"]}))

def arm_layout(specs:list[dict[str,Any]],arm:str,pad:int,raw_max:int,anchor:int,aligned_max:int)->tuple[list[list[int]],list[list[int]],list[int]]:
 values=[];masks=[];starts=[]
 for sp in specs:
  raw=list(sp["ids"]); effective=raw
  if arm=="equalized_suffix":effective=raw+[pad]*(int(sp["pair_target_len"])-len(raw))
  if arm=="right_padding":start=0; total=raw_max
  elif arm=="record_event_aligned":start=anchor-int(sp["positions"]["record_slot0_value"]);total=aligned_max
  else:start=raw_max-len(effective);total=raw_max
  if start<0 or start+len(effective)>total:raise RuntimeError((arm,start,len(effective),total))
  row=[pad]*total;mask=[0]*total;row[start:start+len(effective)]=effective;mask[start:start+len(effective)]=[1]*len(effective);values.append(row);masks.append(mask);starts.append(start)
 return values,masks,starts
def rel(a:torch.Tensor,b:torch.Tensor)->float:
 a=a.float();b=b.float();return float((torch.linalg.vector_norm(a-b)/(0.5*(torch.linalg.vector_norm(a)+torch.linalg.vector_norm(b))+EPS)).item())

@torch.inference_mode()
def run()->None:
 p=load(PROTOCOL);pre=load(PRE)
 if pre.get("authorization")!="run_phase1300_once" or not pre.get("all_checks_passed"):raise RuntimeError("preaudit")
 if any(x.exists() for x in (ARRAYS,META,SUMMARY,RUNTIME,FINAL,COMPLETE)):raise RuntimeError("run consumed")
 mm=read_jsonl(MANIFEST);model=tok=None;started=time.time()
 try:
  model,tok,device,placement=load_fp16("qwen3");qa=quantization_audit(model)
  if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:raise RuntimeError(qa)
  pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id);raw_max=max(len(side["ids"]) for x in mm for side in (x["active"],x["matched_null"]));anchor=max(side["positions"]["record_slot0_value"] for x in mm for side in (x["active"],x["matched_null"]));aligned_max=max(anchor-side["positions"]["record_slot0_value"]+len(side["ids"]) for x in mm for side in (x["active"],x["matched_null"]));supports="logits_to_keep" in inspect.signature(model.forward).parameters
  exact=np.empty((len(ARMS),96,37,2),np.float32);prefix=np.empty_like(exact);cross=np.empty_like(exact)
  def forward(specs,arm):
   vals,masks,starts=arm_layout(specs,arm,pad,raw_max,anchor,aligned_max);ids=torch.tensor(vals,dtype=torch.long,device=device);mask=torch.tensor(masks,dtype=torch.long,device=device);pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0);kw={"input_ids":ids,"attention_mask":mask,"position_ids":pos,"use_cache":False,"output_hidden_states":True,"return_dict":True};
   if supports:kw["logits_to_keep"]=1
   return model(**kw),starts
  for ai,arm in enumerate(ARMS):
   baseline={}
   for start in range(0,96,BATCH_GROUPS):
    group=mm[start:start+BATCH_GROUPS];specs=[]
    for x in group:
     target=max(len(x["active"]["ids"]),len(x["matched_null"]["ids"]));a={**x["active"],"pair_target_len":target};n={**x["matched_null"],"pair_target_len":target};specs.extend((a,a,n))
    out,starts=forward(specs,arm)
    for local,x in enumerate(group):
     a,dup,n=3*local,3*local+1,3*local+2
     for d,h in enumerate(out.hidden_states):
      for ri,role in enumerate(ROLES):
       pa=starts[a]+specs[a]["positions"][role];pd=starts[dup]+specs[dup]["positions"][role];pn=starts[n]+specs[n]["positions"][role];exact[ai,start+local,d,ri]=rel(h[a,pa],h[dup,pd]);prefix[ai,start+local,d,ri]=rel(h[a,pa],h[n,pn])
     baseline[x["calibration_id"]]=np.stack([torch.stack([h[a,starts[a]+specs[a]["positions"][r]].float().cpu() for r in ROLES]).numpy() for h in out.hidden_states])
    del out
   rev=list(reversed(mm))
   for start in range(0,96,BATCH_GROUPS):
    group=rev[start:start+BATCH_GROUPS];specs=[]
    for x in group:
     target=max(len(x["active"]["ids"]),len(x["matched_null"]["ids"]));a={**x["active"],"pair_target_len":target};n={**x["matched_null"],"pair_target_len":target};specs.extend((n,a,n))
    out,starts=forward(specs,arm)
    for local,x in enumerate(group):
     a=3*local+1;oi=next(i for i,z in enumerate(mm) if z["calibration_id"]==x["calibration_id"])
     for d,h in enumerate(out.hidden_states):
      for ri,role in enumerate(ROLES):cross[ai,oi,d,ri]=rel(h[a,starts[a]+specs[a]["positions"][role]],torch.from_numpy(baseline[x["calibration_id"]][d,ri]).to(device))
    del out
   print(canonical({"arm":arm,"exact":float(exact[ai].max()),"prefix":float(prefix[ai].max()),"cross":float(cross[ai].max())}),flush=True)
  ARRAYS.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(ARRAYS,exact_duplicate=exact,same_prefix=prefix,cross_composition=cross,arms=np.asarray(ARMS),depths=np.asarray(DEPTHS),roles=np.asarray(ROLES))
  arms={}
  for ai,arm in enumerate(ARMS):
   maxima={"exact_duplicate_relative_max":float(exact[ai].max()),"same_prefix_relative_max":float(prefix[ai].max()),"cross_composition_relative_max":float(cross[ai].max())};passed=all((maxima["exact_duplicate_relative_max"]<=THRESHOLDS["exact_duplicate_relative_max"],maxima["same_prefix_relative_max"]<=THRESHOLDS["same_prefix_relative_max"],maxima["cross_composition_relative_max"]<=THRESHOLDS["cross_composition_relative_max"]));arms[arm]={"maxima":maxima,"median":{"exact_duplicate":float(np.median(exact[ai])),"same_prefix":float(np.median(prefix[ai])),"cross_composition":float(np.median(cross[ai]))},"q99_same_prefix":float(np.quantile(prefix[ai],0.99)),"passed":passed}
  passing=[a for a in CANDIDATES if arms[a]["passed"]];selected=next((a for a in PRIORITY if a in passing),None);max_noise=0.0 if selected is None else max(arms[selected]["maxima"].values());tau=max(THRESHOLDS["tau_floor"],min(THRESHOLDS["tau_cap"],THRESHOLDS["tau_multiplier"]*max_noise));gates={"candidate_count":len(passing)>=THRESHOLDS["candidate_compilers_passing_min"],"selected_exists":selected is not None,"selected_tau_below_cap":selected is not None and tau<THRESHOLDS["tau_cap"],"finite":all(np.isfinite(x).all() for x in (exact,prefix,cross))};passed=all(gates.values());auth="phase1301_qwen3_behavior_only" if passed else "close_c032_without_semantic_run";summary={"arms":arms,"passing_candidate_arms":passing,"selected_runtime":selected,"selected_max_noise":max_noise,"frozen_tau":tau,"gates":gates,"all_gates_passed":passed};save(SUMMARY,summary);save(RUNTIME,{"phase":PHASE,"campaign":CAMPAIGN,"protocol_digest":p["protocol_digest"],"selected_runtime":selected,"tau":tau,"passing_candidate_arms":passing,"array_sha256":sha(ARRAYS),"frozen_before_behavior":True});save(META,{"phase":PHASE,"campaign":CAMPAIGN,"protocol_digest":p["protocol_digest"],"array_sha256":sha(ARRAYS),"model_audit":qa,"placement":placement,"runtime_seconds":time.time()-started,"raw_max_length":raw_max,"event_anchor":anchor,"event_aligned_length":aligned_max,"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0});save(FINAL,{"phase":PHASE,"campaign":CAMPAIGN,"verdict":"execution_compiler_qualified" if passed else "execution_compiler_gate_failed","all_gates_passed":passed,"selected_runtime":selected,"frozen_tau":tau,"authorization":auth,"protocol_digest":p["protocol_digest"],"array_sha256":sha(ARRAYS)});save(COMPLETE,{"completed_at_utc":datetime.now(timezone.utc).isoformat(),"formal_runs_consumed":1,"protocol_digest":p["protocol_digest"]});print(canonical({"passing":passing,"selected":selected,"tau":tau,"gates":gates,"authorization":auth}))
 finally:
  if model is not None:release_fp16(model)

if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("command",choices=("preregister","run"));ap.add_argument("--force",action="store_true");args=ap.parse_args();preregister(args.force) if args.command=="preregister" else run()
