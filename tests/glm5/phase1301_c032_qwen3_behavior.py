#!/usr/bin/env python3
"""Phase1301: C032 Qwen3 behavior gate under frozen right-padding runtime."""
from __future__ import annotations
import argparse,hashlib,inspect,json,os,re,shutil,sys,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";sys.path.insert(0,str(T))
from phase1023_fp16_utils import load_fp16,quantization_audit,release_fp16  # noqa:E402
import phase1295_c030_qwen3_grounded_lookup_behavior as helpers  # noqa:E402
PHASE=1301;CAMPAIGN="C032";SCRIPT=Path(__file__).resolve();AUDITOR=T/"phase1301_c032_qwen3_behavior_audit.py";PARENT=T/"result/phase1300_c032_execution_compiler_competition";PARENT_PROTOCOL=PARENT/"protocol/preregistration.json";PARENT_FINAL=PARENT/"analysis/final.json";PARENT_AUDIT=PARENT/"audit/independent_final_audit_after_erratum.json";RUNTIME=PARENT/"protocol/frozen_runtime.json";CONTRACT=T/"result/phase1299_c032_execution_compiler_contract";MATERIAL=CONTRACT/"material/frozen_inverse_lookup_cases.jsonl";CONTRACT_PROTOCOL=CONTRACT/"protocol/preregistration.json";OUT=T/"result/phase1301_c032_qwen3_behavior";PROTOCOL=OUT/"protocol/preregistration.json";PRE=OUT/"audit/independent_preaudit.json";POST=OUT/"audit/independent_final_audit.json";RAW=OUT/"raw/candidate_scores.jsonl";GEN=OUT/"raw/list_free_generations.jsonl";SUMMARY=OUT/"analysis/behavior_summary.json";FINAL=OUT/"analysis/final.json";COMPLETE=OUT/"protocol/formal_run_complete.json";SYSTEM="Use only the supplied catalog. Reply exactly as requested and do not explain.";SCORE_BATCH=32;GEN_BATCH=8;MAX_NEW=8
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
def chat(tok,prompt):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
def preregister(force):
 if load(PARENT_FINAL).get("authorization")!="phase1301_qwen3_behavior_only" or not load(PARENT_AUDIT).get("all_checks_passed"):raise RuntimeError("parent auth")
 rt=load(RUNTIME)
 if rt["selected_runtime"]!="right_padding" or rt["tau"]!=1e-7:raise RuntimeError("runtime drift")
 if OUT.exists() and not force:raise RuntimeError(f"{OUT} exists")
 if OUT.exists():shutil.rmtree(OUT)
 cp=load(CONTRACT_PROTOCOL);timeless={"phase":PHASE,"campaign":CAMPAIGN,"schema_version":"phase1301.c032.behavior.v1","model":"qwen3-4b-fp16-cuda-no-quantization","formal_run_budget":1,"runtime":{"compiler":"right_padding","tau":rt["tau"],"batching":"exact token-length buckets; hence physical start column zero and no intra-bucket pad","score_batch_max":SCORE_BATCH,"generation_batch_max":GEN_BATCH},"material":{"sha256":sha(MATERIAL),"case_count":6912,"generation_filter":"confirmation or holdout and candidate_order=0","generation_count":1536},"thresholds":cp["behavior"]["thresholds"],"zero_models":cp["zero_models"],"scoring":"three frozen one-token name continuations; highest log probability","generation":"greedy, max 8 new tokens, exact first-line entity required","hidden_states_read":False,"success":"all candidate and list-free ledgers pass","success_authorization":"phase1302_event_identity_hidden_only","failure":"close_c032_without_hidden","dependencies":{"parent_protocol":sha(PARENT_PROTOCOL),"parent_final":sha(PARENT_FINAL),"parent_audit":sha(PARENT_AUDIT),"runtime":sha(RUNTIME),"contract":sha(CONTRACT_PROTOCOL),"material":sha(MATERIAL),"helper":sha(T/"phase1295_c030_qwen3_grounded_lookup_behavior.py")},"source_hashes":{"main":sha(SCRIPT),"auditor":sha(AUDITOR)},"model_weights_loaded":False};p={**timeless,"created_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_digest":digest(timeless)};save(PROTOCOL,p);print(canonical({"digest":p["protocol_digest"],"runtime":p["runtime"]}))
def prepare(tok,rr,prompt_key):
 out=[]
 for i,r in enumerate(rr):
  rendered=chat(tok,r[prompt_key]);ids=tok.encode(rendered,add_special_tokens=False);out.append({"index":i,"row":r,"rendered":rendered,"ids":ids})
 return out
def buckets(items):
 d=defaultdict(list)
 for x in items:d[len(x["ids"])].append(x)
 return [d[k] for k in sorted(d)]
@torch.inference_mode()
def score(model,tok,device,rr):
 prepared=prepare(tok,rr,"candidate_prompt");out=[];supports="logits_to_keep" in inspect.signature(model.forward).parameters
 for bucket in buckets(prepared):
  for start in range(0,len(bucket),SCORE_BATCH):
   b=bucket[start:start+SCORE_BATCH];ids=torch.tensor([x["ids"] for x in b],dtype=torch.long,device=device);mask=torch.ones_like(ids);pos=mask.cumsum(-1)-1;kw={"input_ids":ids,"attention_mask":mask,"position_ids":pos,"use_cache":False,"return_dict":True};
   if supports:kw["logits_to_keep"]=1
   logits=model(**kw).logits[:,-1,:].float();lp=torch.log_softmax(logits,dim=-1)
   for j,x in enumerate(b):
    r=x["row"];cids=[]
    for name in r["candidates"]:
     full=tok.encode(x["rendered"]+" "+name,add_special_tokens=False)
     if full[:len(x["ids"])]!=x["ids"] or len(full)!=len(x["ids"])+1:raise RuntimeError("candidate drift")
     cids.append(full[-1])
    scores={n:float(lp[j,t].item()) for n,t in zip(r["candidates"],cids)};order=sorted(scores,key=lambda n:(-scores[n],n));pred=order[0] if scores[order[0]]>scores[order[1]] else None;gold=r["gold_candidate"];other=max(v for n,v in scores.items() if n!=gold);out.append({"_index":x["index"],"case_id":r["case_id"],"group_id":r["group_id"],"partition":r["partition"],"profile_index":r["profile_index"],"attribute":r["attribute"],"panel":r["panel"],"surface":r["surface"],"candidate_order":r["candidate_order"],"binding_state":r["binding_state"],"entities":r["entities"],"record_order":r["record_order"],"candidates":r["candidates"],"gold_candidate":gold,"candidate_token_ids":cids,"candidate_log_prob":scores,"prediction":pred,"correct":pred==gold,"gold_margin":float(scores[gold]-other),"finite":bool(all(np.isfinite(list(scores.values()))))})
 return [{k:v for k,v in x.items() if k!="_index"} for x in sorted(out,key=lambda z:z["_index"])]
@torch.inference_mode()
def generate(model,tok,device,rr):
 selected=[r for r in rr if r["partition"] in {"confirmation","holdout"} and r["candidate_order"]==0];prepared=prepare(tok,selected,"generation_prompt");out=[];pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
 for bucket in buckets(prepared):
  for start in range(0,len(bucket),GEN_BATCH):
   b=bucket[start:start+GEN_BATCH];ids=torch.tensor([x["ids"] for x in b],dtype=torch.long,device=device);mask=torch.ones_like(ids);generated=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=pad,eos_token_id=tok.eos_token_id)[:,ids.shape[1]:];texts=tok.batch_decode(generated,skip_special_tokens=True)
   for x,text in zip(b,texts):
    r=x["row"];hits=[n for n in r["candidates"] if re.search(rf"\b{re.escape(n)}\b",text,re.I)];pred=hits[0] if len(hits)==1 else None;norm=helpers.normalize_first_line(text);out.append({"_index":x["index"],"case_id":r["case_id"],"group_id":r["group_id"],"partition":r["partition"],"profile_index":r["profile_index"],"attribute":r["attribute"],"panel":r["panel"],"surface":r["surface"],"candidate_order":r["candidate_order"],"binding_state":r["binding_state"],"candidates":r["candidates"],"gold_candidate":r["gold_candidate"],"generation":text,"normalized_first_line":norm,"candidate_hits":hits,"covered":len(hits)==1,"prediction":pred,"label_correct":pred==r["gold_candidate"],"exact_correct":norm==r["gold_candidate"].lower()})
 return [{k:v for k,v in x.items() if k!="_index"} for x in sorted(out,key=lambda z:z["_index"])]
def run():
 p=load(PROTOCOL);pre=load(PRE)
 if pre.get("authorization")!="run_phase1301_once" or not pre.get("all_checks_passed"):raise RuntimeError("preaudit")
 if any(x.exists() for x in (RAW,GEN,SUMMARY,FINAL,COMPLETE)):raise RuntimeError("consumed")
 rr=rows(MATERIAL);model=tok=None;started=time.time()
 try:
  model,tok,device,placement=load_fp16("qwen3");qa=quantization_audit(model)
  if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:raise RuntimeError(qa)
  raw=score(model,tok,device,rr);write(RAW,raw);gen=generate(model,tok,device,rr);write(GEN,gen);candidate=helpers.candidate_summary(raw,p["thresholds"],float(p["zero_models"]["shortcut_ceiling"]));generation=helpers.generation_summary(gen,p["thresholds"]);passed=candidate["passed"] and generation["passed"];auth="phase1302_event_identity_hidden_only" if passed else "close_c032_without_hidden";summary={"phase":PHASE,"campaign":CAMPAIGN,"protocol_digest":p["protocol_digest"],"runtime":"right_padding_exact_length_buckets","candidate":candidate,"generation":generation,"all_behavior_gates_passed":passed,"authorization":auth,"raw_hashes":{"candidate_scores":sha(RAW),"list_free_generations":sha(GEN)},"counts":{"candidate":len(raw),"generation":len(gen)},"model_audit":qa,"placement":placement,"runtime_seconds":time.time()-started,"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,"hidden_states_read":False};save(SUMMARY,summary);save(FINAL,{"phase":PHASE,"campaign":CAMPAIGN,"verdict":"behavior_qualified" if passed else "behavior_gate_failed","protocol_digest":p["protocol_digest"],"raw_hashes":summary["raw_hashes"],"all_behavior_gates_passed":passed,"authorization":auth,"hidden_states_read":False});save(COMPLETE,{"completed_at_utc":datetime.now(timezone.utc).isoformat(),"formal_runs_consumed":1,"protocol_digest":p["protocol_digest"]});print(canonical({"candidate":candidate["metrics"]["overall_candidate_accuracy"],"generation":generation["metrics"]["exact_accuracy"],"authorization":auth}))
 finally:
  if model is not None:release_fp16(model)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("command",choices=("preregister","run"));ap.add_argument("--force",action="store_true");a=ap.parse_args();preregister(a.force) if a.command=="preregister" else run()
