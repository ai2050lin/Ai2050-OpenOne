#!/usr/bin/env python3
"""Phase1340: frozen C047 standard yes/no paired-differential behavior."""
from __future__ import annotations
import argparse,json,math,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import median
import torch
R=Path(__file__).resolve().parents[2]; T=R/"tests/glm5"; sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
PHASE,CAMPAIGN=1340,"C047"; P=T/"result/phase1339_c047_paired_relation_contract"; O=T/"result/phase1340_c047_paired_behavior"; MODELS=("qwen3","glm4","deepseek7b")
def parent():
 f=core.load(P/"analysis/final.json");a=core.load(P/"audit/independent_final_audit.json")
 if f.get("authorization")!="run_phase1340_c047_paired_behavior" or not a.get("all_checks_passed"):raise RuntimeError("parent")
 return core.load(P/"protocol/preregistration.json")
def tensors(batch,width,pad,device):
 ids=torch.full((len(batch),width),int(pad),dtype=torch.long,device=device);mask=torch.zeros_like(ids);lens=[]
 for i,r in enumerate(batch):
  x=torch.tensor(r["prompt_ids"],device=device);ids[i,:len(x)]=x;mask[i,:len(x)]=1;lens.append(len(x))
 pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0);return ids,mask,pos,lens
@torch.inference_mode()
def score(model,device,batch,width,pad):
 ids,mask,pos,lens=tensors(batch,width,pad,device);o=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True);ans=[]
 for i,r in enumerate(batch):
  lp=torch.log_softmax(o.logits[i,lens[i]-1].float(),-1);ans.append([float(lp[c[0]]) for c in r["candidate_ids"]])
 del ids,mask,pos,o;return ans
def prepare():
 pr=parent()
 if (O/"protocol/execution_manifest.json").exists():raise RuntimeError("manifest exists")
 groups={};widths={};sent=set(pr["executor_gate"]["case_ids"])
 for m in MODELS:
  rows=core.rows(P/f"compiled/{m}_behavior.jsonl");widths[m]=max(len(x["prompt_ids"]) for x in rows);order=[x["case_id"] for x in rows if x["case_id"] in sent];perm=order[::2]+order[1::2]
  groups[m]={"a":[order[i:i+8] for i in range(0,len(order),8)],"p":[perm[i:i+8] for i in range(0,len(perm),8)]}
 x={"phase":PHASE,"campaign":CAMPAIGN,"parent_contract_sha256":pr["contract_sha256"],"model_order":list(MODELS),"precision":"bfloat16-no-quantization","batch_size":8,"widths":widths,"groups":groups,"behavior_gate":pr["behavior_gate"],"created_at_utc":datetime.now(timezone.utc).isoformat()};core.save(O/"protocol/execution_manifest.json",x);print(json.dumps(x,indent=2))
def by(rows,key,vals):return {str(v):sum(r["correct"] for r in rows if r[key]==v)/sum(r[key]==v for r in rows) for v in vals}
def run(m):
 pr=parent();man=core.load(O/"protocol/execution_manifest.json");src=core.rows(P/"material/frozen_behavior_cases.jsonl");comp=core.rows(P/f"compiled/{m}_behavior.jsonl");byid={x["case_id"]:x for x in comp};model=None
 try:
  model,tok,dev,placement=load_bf16(m);qa=quantization_audit(model);pad=tok.pad_token_id or tok.eos_token_id;width=man["widths"][m]
  def grouped(gs):
   out={}
   for g in gs:
    out.update(zip(g,score(model,dev,[byid[i] for i in g],width,pad)))
   return out
  a=grouped(man["groups"][m]["a"]);p=grouped(man["groups"][m]["p"]);r=grouped(man["groups"][m]["a"])
  ex=[]
  for cid in pr["executor_gate"]["case_ids"]:ex.append({"case_id":cid,"a":a[cid],"p":p[cid],"r":r[cid]})
  finite=all(math.isfinite(v) for x in ex for k in ("a","p","r") for v in x[k]);rank=sum((x["a"][0]>x["a"][1])==(x["p"][0]>x["p"][1]) for x in ex)/len(ex);diff=max(abs(u-v) for x in ex for k in ("p","r") for u,v in zip(x["a"],x[k]));executor=finite and rank==1 and diff<=1e-6
  rec=[]
  if executor:
   for st in range(0,len(comp),8):
    vals=score(model,dev,comp[st:st+8],width,pad)
    for s,z in zip(src[st:st+8],vals):
     margin=z[0]-z[1];pred=0 if margin>0 else 1
     rec.append({**{k:s[k] for k in ("case_id","partition","surface","target","target_family","tested_family","truth","quartet_key")},"scores":z,"semantic_margin":margin,"correct":pred==s["gold_position"]})
  qs=defaultdict(list)
  for x in rec:qs[x["quartet_key"]].append(x)
  gaps=[];quart=[]
  for q in qs.values():
   t=next(x for x in q if x["truth"]);wrong=[x for x in q if not x["truth"]];g=[t["semantic_margin"]-x["semantic_margin"] for x in wrong];gaps+=g;quart.append(all(v>0 for v in g))
  met={"accuracy":sum(x["correct"] for x in rec)/len(rec),"partition":by(rec,"partition",("discovery","confirmation","holdout")),"surface":by(rec,"surface",("noun_class","common_sense","category_claim")),"family":by(rec,"target_family",("insect","dessert","fabric","tree")),"truth":by(rec,"truth",(True,False)),"pairwise_gap_win":sum(v>0 for v in gaps)/len(gaps),"quartet_rank":sum(quart)/len(quart),"median_relation_gap":median(gaps),"case_count":len(rec)}
  g=pr["behavior_gate"];gates={"accuracy":met["accuracy"]>=g["accuracy_min"],"partition":min(met["partition"].values())>=g["partition_min"],"surface":min(met["surface"].values())>=g["surface_min"],"family":min(met["family"].values())>=g["target_family_min"],"truth":min(met["truth"].values())>=g["truth_cell_min"],"pair":met["pairwise_gap_win"]>=g["pairwise_gap_win_min"],"quartet":met["quartet_rank"]>=g["quartet_rank_min"],"gap":met["median_relation_gap"]>=g["median_relation_gap_min"]}
  core.write_rows(O/f"raw/{m}_executor.jsonl",ex);core.write_rows(O/f"raw/{m}_behavior.jsonl",rec);core.save(O/f"runtime/{m}.json",{"placement":placement,"quantization_audit":qa,"finished_at_utc":datetime.now(timezone.utc).isoformat()});core.save(O/f"analysis/{m}_summary.json",{"model":m,"executor":{"finite":finite,"rank":rank,"max_abs_diff":diff,"qualified":executor},"behavior_metrics":met,"behavior_gates":gates,"qualified":executor and all(gates.values())});print(json.dumps({"model":m,"metrics":met,"gates":gates,"qualified":executor and all(gates.values())},indent=2))
 finally:
  if model is not None:release_bf16(model)
def finalize():
 pr=parent();ss={m:core.load(O/f"analysis/{m}_summary.json") for m in MODELS};q=[m for m in MODELS if ss[m]["qualified"]];auth="run_phase1341_c047_full_relation_field" if len(q)>=pr["behavior_gate"]["minimum_authorized_models"] else "close_c047_behavior";core.save(O/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"qualified_models":q,"qualified_model_count":len(q),"all_gates_passed":len(q)>=2,"authorization":auth,"finished_at_utc":datetime.now(timezone.utc).isoformat()});print(json.dumps({"qualified":q,"authorization":auth},indent=2))
if __name__=="__main__":
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument("--prepare",action="store_true");g.add_argument("--model",choices=MODELS);g.add_argument("--finalize",action="store_true");x=a.parse_args();prepare() if x.prepare else run(x.model) if x.model else finalize()
