#!/usr/bin/env python3
"""Phase1341: C047 full-dimensional role-aligned paired response field."""
from __future__ import annotations
import argparse,json,math,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import median
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[2];T=R/"tests/glm5";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
PHASE,CAMPAIGN=1341,"C047";P=T/"result/phase1339_c047_paired_relation_contract";B=T/"result/phase1340_c047_paired_behavior";O=T/"result/phase1341_c047_full_relation_field";MODELS=("qwen3","glm4")
ROLES=("target","tested_family","boundary")
def parents():
 bf=core.load(B/"analysis/final.json");ba=core.load(B/"audit/independent_final_audit.json");pr=core.load(P/"protocol/preregistration.json")
 if bf.get("authorization")!="run_phase1341_c047_full_relation_field" or not ba.get("all_checks_passed") or bf.get("qualified_models")!=list(MODELS):raise RuntimeError("parents")
 return pr
def prep():
 pr=parents()
 if (O/"protocol/execution_manifest.json").exists():raise RuntimeError("exists")
 x={"phase":PHASE,"campaign":CAMPAIGN,"model_order":list(MODELS),"qualified_behavior_parent":core.sha(B/"analysis/final.json"),"contract_sha256":pr["contract_sha256"],"depths":pr["hidden_gate"]["normalized_depths"],"roles":list(ROLES),"batch_size":4,"precision":"bfloat16-no-quantization","primary_storage":"float32 complete selected vectors","sentinel_case_ids":pr["executor_gate"]["case_ids"][:16],"gate":pr["hidden_gate"],"created_at_utc":datetime.now(timezone.utc).isoformat()};core.save(O/"protocol/execution_manifest.json",x);print(json.dumps(x,indent=2))
def tensors(batch,width,pad,dev):
 ids=torch.full((len(batch),width),int(pad),dtype=torch.long,device=dev);mask=torch.zeros_like(ids);lens=[]
 for i,r in enumerate(batch):x=torch.tensor(r["prompt_ids"],device=dev);ids[i,:len(x)]=x;mask[i,:len(x)]=1;lens.append(len(x))
 pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0);return ids,mask,pos,lens
def depth_ids(n):return [round(x*(n-1)) for x in (0,.25,.5,.75,1)]
@torch.inference_mode()
def capture(model,dev,batch,width,pad):
 ids,mask,pos,lens=tensors(batch,width,pad,dev);o=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,output_hidden_states=True,return_dict=True);di=depth_ids(len(o.hidden_states));out=[]
 for i,r in enumerate(batch):
  vals=[]
  spans=(r["target_span"],r["tested_family_span"],[lens[i]-1])
  for d in di:
   h=o.hidden_states[d][i].float();vals.append(torch.stack([h[s].mean(0) for s in spans]).cpu())
  out.append(torch.stack(vals))
 del ids,mask,pos,o;return out
def cosine(a,b):return float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0))
def run(m):
 parents();man=core.load(O/"protocol/execution_manifest.json");src=core.rows(P/"material/frozen_behavior_cases.jsonl");comp=core.rows(P/f"compiled/{m}_behavior.jsonl");width=max(len(x["prompt_ids"]) for x in comp);model=None
 try:
  model,tok,dev,place=load_bf16(m);qa=quantization_audit(model);pad=tok.pad_token_id or tok.eos_token_id;vec={}
  for st in range(0,len(comp),4):
   for r,v in zip(comp[st:st+4],capture(model,dev,comp[st:st+4],width,pad)):vec[r["case_id"]]=v
  sent=man["sentinel_case_ids"];byid={x["case_id"]:x for x in comp};a={i:v.clone() for i,v in vec.items() if i in sent};perm=sent[::2]+sent[1::2];p={}
  for st in range(0,len(perm),4):
   batch=[byid[i] for i in perm[st:st+4]]
   for r,v in zip(batch,capture(model,dev,batch,width,pad)):p[r["case_id"]]=v
  rel=[float((a[i]-p[i]).norm()/(a[i].norm()+1e-12)) for i in sent];numeric={"relative_l2_p95":sorted(rel)[math.ceil(.95*len(rel))-1],"relative_l2_max":max(rel)}
  groups=defaultdict(list)
  meta={x["case_id"]:x for x in src}
  for x in src:groups[x["quartet_key"]].append(x)
  signatures={}
  for key,z in groups.items():
   t=next(x for x in z if x["truth"]);w=[x for x in z if not x["truth"]];signatures[key]=vec[t["case_id"]]-torch.stack([vec[x["case_id"]] for x in w]).mean(0)
  targets=defaultdict(dict)
  for k,v in signatures.items():
   part,target,surface=k.split(":");targets[(part,target)][surface]=v
  ordered=sorted(targets);null={k:ordered[(i+1)%len(ordered)] for i,k in enumerate(ordered)};metrics={}
  for d in range(5):
   for ri,role in enumerate(ROLES):
    same=[];wrong=[]
    for k,smap in targets.items():
     nk=null[k];ns=targets[nk]
     for sa,sb in (("noun_class","common_sense"),("noun_class","category_claim"),("common_sense","category_claim")):
      same.append(cosine(smap[sa][d,ri],smap[sb][d,ri]));wrong.append(cosine(smap[sa][d,ri],ns[sb][d,ri]))
    metrics[f"d{d}:{role}"]={"identity_win":sum(a>b for a,b in zip(same,wrong))/len(same),"median_same":median(same),"median_null":median(wrong),"median_gap":median([a-b for a,b in zip(same,wrong)])}
  g=man["gate"];eligible=[metrics[f"d{d}:tested_family"] for d in (1,2,3,4)];passing=sum(x["identity_win"]>=g["cross_surface_identity_win_min"] and x["median_gap"]>=g["permutation_gap_min"] for x in eligible);num=numeric["relative_l2_p95"]<=g["numeric_relative_l2_p95_max"] and numeric["relative_l2_max"]<=g["numeric_relative_l2_max"];qualified=num and passing>=2
  (O/"raw").mkdir(parents=True,exist_ok=True)
  torch.save({"case_ids":[x["case_id"] for x in src],"vectors":torch.stack([vec[x["case_id"]] for x in src]),"depths":man["depths"],"roles":ROLES},O/f"raw/{m}_field.pt")
  summary={"model":m,"numeric":numeric,"numeric_qualified":num,"metrics":metrics,"passing_internal_depths":passing,"qualified":qualified,"claim":"descriptive paired-response field only; no semantic or causal identification","runtime":{"placement":place,"quantization_audit":qa,"finished_at_utc":datetime.now(timezone.utc).isoformat()}};core.save(O/f"analysis/{m}_summary.json",summary);print(json.dumps({"model":m,"numeric":numeric,"passing_internal_depths":passing,"qualified":qualified},indent=2))
 finally:
  if model is not None:release_bf16(model)
def final():
 pr=parents();s={m:core.load(O/f"analysis/{m}_summary.json") for m in MODELS};q=[m for m in MODELS if s[m]["qualified"]];passed=len(q)>=pr["hidden_gate"]["minimum_authorized_models"];auth="close_c047_descriptive_field_and_authorize_separate_causal_preregistration" if passed else "close_c047_descriptive_field";core.save(O/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"qualified_models":q,"all_gates_passed":passed,"authorization":auth,"causal_claim":False,"finished_at_utc":datetime.now(timezone.utc).isoformat()});print(json.dumps({"qualified":q,"authorization":auth},indent=2))
if __name__=="__main__":
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument("--prepare",action="store_true");g.add_argument("--model",choices=MODELS);g.add_argument("--finalize",action="store_true");x=a.parse_args();prep() if x.prepare else run(x.model) if x.model else final()
