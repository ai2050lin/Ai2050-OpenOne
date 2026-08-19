#!/usr/bin/env python3
"""Phase1343: frozen C048 factorial behavior qualification."""
from __future__ import annotations
import argparse,json,math,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import median
import torch
R=Path(__file__).resolve().parents[2];T=R/"tests/glm5";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
PHASE,CAMPAIGN=1343,"C048";P=T/"result/phase1342_c048_factorial_causal_contract";O=T/"result/phase1343_c048_factorial_behavior";MODELS=("qwen3","glm4","deepseek7b")
def parent():
 f=core.load(P/"analysis/final.json");a=core.load(P/"audit/independent_final_audit.json")
 if f.get("authorization")!="run_phase1343_c048_factorial_behavior" or not a.get("all_checks_passed"):raise RuntimeError("parent")
 return core.load(P/"protocol/preregistration.json")
def tensors(batch,width,pad,dev):
 ids=torch.full((len(batch),width),int(pad),dtype=torch.long,device=dev);mask=torch.zeros_like(ids);lens=[]
 for i,r in enumerate(batch):x=torch.tensor(r["prompt_ids"],device=dev);ids[i,:len(x)]=x;mask[i,:len(x)]=1;lens.append(len(x))
 pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0);return ids,mask,pos,lens
@torch.inference_mode()
def score(model,dev,batch,width,pad):
 ids,mask,pos,lens=tensors(batch,width,pad,dev);o=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True);ans=[]
 for i,r in enumerate(batch):lp=torch.log_softmax(o.logits[i,lens[i]-1].float(),-1);ans.append([float(lp[c[0]]) for c in r["candidate_ids"]])
 del ids,mask,pos,o;return ans
def prepare():
 pr=parent()
 if (O/"protocol/execution_manifest.json").exists():raise RuntimeError("exists")
 groups={};widths={};sent=pr["executor_gate"]["sentinel_case_ids"]
 for m in MODELS:
  rows=core.rows(P/f"compiled/{m}_factorial.jsonl");widths[m]=max(len(x["prompt_ids"]) for x in rows);order=list(sent);perm=order[::2]+order[1::2];groups[m]={"a":[order[i:i+4] for i in range(0,len(order),4)],"p":[perm[i:i+4] for i in range(0,len(perm),4)]}
 x={"phase":PHASE,"campaign":CAMPAIGN,"contract_sha256":pr["contract_sha256"],"model_order":list(MODELS),"precision":"bfloat16-no-quantization","batch_size":4,"widths":widths,"executor_groups":groups,"gate":pr["behavior_gate"],"created_at_utc":datetime.now(timezone.utc).isoformat()};core.save(O/"protocol/execution_manifest.json",x);print(json.dumps(x,indent=2))
def by(rows,key,vals):return {str(v):sum(x["correct"] for x in rows if x[key]==v)/sum(x[key]==v for x in rows) for v in vals}
def run(m):
 pr=parent();man=core.load(O/"protocol/execution_manifest.json");src=core.rows(P/"material/frozen_factorial_cases.jsonl");comp=core.rows(P/f"compiled/{m}_factorial.jsonl");byid={x["case_id"]:x for x in comp};model=None
 try:
  model,tok,dev,place=load_bf16(m);qa=quantization_audit(model);pad=tok.pad_token_id or tok.eos_token_id;width=man["widths"][m]
  def grouped(gs):
   z={}
   for g in gs:z.update(zip(g,score(model,dev,[byid[i] for i in g],width,pad)))
   return z
  a=grouped(man["executor_groups"][m]["a"]);p=grouped(man["executor_groups"][m]["p"]);r=grouped(man["executor_groups"][m]["a"]);ex=[{"case_id":i,"a":a[i],"p":p[i],"r":r[i]} for i in pr["executor_gate"]["sentinel_case_ids"]];finite=all(math.isfinite(v) for x in ex for k in ("a","p","r") for v in x[k]);rank=sum((x["a"][0]>x["a"][1])==(x["p"][0]>x["p"][1]) for x in ex)/len(ex);diff=max(abs(u-v) for x in ex for k in ("p","r") for u,v in zip(x["a"],x[k]));executor=finite and rank>=1 and diff<=1e-6
  rec=[]
  if executor:
   for st in range(0,len(comp),4):
    vals=score(model,dev,comp[st:st+4],width,pad)
    for s,z in zip(src[st:st+4],vals):
     margin=z[0]-z[1];rec.append({**{k:s[k] for k in ("case_id","partition","family_pair","pair_index","surface","quartet_key","cell","interaction_sign","target","target_family","tested_family","truth")},"scores":z,"semantic_margin":margin,"correct":((margin>0)==s["truth"])})
  qs=defaultdict(list)
  for x in rec:qs[x["quartet_key"]].append(x)
  ints=[];pairwins=[];allcorrect=[]
  for q in qs.values():
   z={x["cell"]:x for x in q};ints.append(z["aa"]["semantic_margin"]-z["ab"]["semantic_margin"]-z["ba"]["semantic_margin"]+z["bb"]["semantic_margin"]);pairwins.extend([z["aa"]["semantic_margin"]>z["ab"]["semantic_margin"],z["bb"]["semantic_margin"]>z["ba"]["semantic_margin"]]);allcorrect.append(all(x["correct"] for x in q))
  met={"accuracy":sum(x["correct"] for x in rec)/len(rec),"partition":by(rec,"partition",("discovery","confirmation","holdout")),"surface":by(rec,"surface",("ordinary","dictionary","claim")),"family":by(rec,"target_family",("dance","spice","bread","beverage")),"truth":by(rec,"truth",(True,False)),"pairwise_true_over_false":sum(pairwins)/len(pairwins),"quartet_all_correct":sum(allcorrect)/len(allcorrect),"positive_interaction_fraction":sum(x>0 for x in ints)/len(ints),"median_interaction":median(ints),"case_count":len(rec),"quartet_count":len(qs)};g=pr["behavior_gate"];gates={"accuracy":met["accuracy"]>=g["accuracy_min"],"partition":min(met["partition"].values())>=g["partition_min"],"surface":min(met["surface"].values())>=g["surface_min"],"family":min(met["family"].values())>=g["family_min"],"truth":min(met["truth"].values())>=g["truth_min"],"pair":met["pairwise_true_over_false"]>=g["pairwise_true_over_false_min"],"quartet":met["quartet_all_correct"]>=g["quartet_all_correct_min"],"fraction":met["positive_interaction_fraction"]>=g["positive_interaction_fraction_min"],"magnitude":met["median_interaction"]>=g["median_interaction_min"]};qualified=executor and all(gates.values());core.write_rows(O/f"raw/{m}_executor.jsonl",ex);core.write_rows(O/f"raw/{m}_behavior.jsonl",rec);core.save(O/f"runtime/{m}.json",{"placement":place,"quantization_audit":qa,"finished_at_utc":datetime.now(timezone.utc).isoformat()});core.save(O/f"analysis/{m}_summary.json",{"model":m,"executor":{"finite":finite,"rank":rank,"max_abs_diff":diff,"qualified":executor},"behavior_metrics":met,"behavior_gates":gates,"qualified":qualified});print(json.dumps({"model":m,"metrics":met,"gates":gates,"qualified":qualified},indent=2))
 finally:
  if model is not None:release_bf16(model)
def final():
 pr=parent();s={m:core.load(O/f"analysis/{m}_summary.json") for m in MODELS};q=[m for m in MODELS if s[m]["qualified"]];passed=len(q)>=pr["behavior_gate"]["minimum_authorized_models"];auth="run_phase1344_c048_interaction_field" if passed else "close_c048_behavior";core.save(O/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"qualified_models":q,"all_gates_passed":passed,"authorization":auth,"finished_at_utc":datetime.now(timezone.utc).isoformat()});print(json.dumps({"qualified":q,"authorization":auth},indent=2))
if __name__=="__main__":
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument("--prepare",action="store_true");g.add_argument("--model",choices=MODELS);g.add_argument("--finalize",action="store_true");x=a.parse_args();prepare() if x.prepare else run(x.model) if x.model else final()
