#!/usr/bin/env python3
"""Separated Q, K, V, KV and whole-head causal interventions on token-atomic lockbox data."""
from __future__ import annotations

import gc, hashlib, json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result"
P2538=RESULT/"phase2538_c117505_c121600_token_atomic_hypergraph_behavior";P2539=RESULT/"phase2539_c121601_c125696_full_token_qkv_edge_ledger"
OUT=RESULT/"phase2540_c125697_c129792_qkv_separated_causal_lockbox";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE,CAMPAIGN=2540,"C125697-C129792";LATE=tuple(range(20,36));SOURCE_NAMES=("facts_entity","facts_relation","facts_value")
sys.path.insert(0,str(TESTS));import model_utils  # noqa:E402

def load(p:Path)->Any:return json.loads(p.read_text(encoding="utf-8-sig"))
def read(p:Path)->list[dict]:return[json.loads(x)for x in p.read_text(encoding="utf-8-sig").splitlines()if x.strip()]
def save(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def write(p:Path,rows:list[dict])->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n"for r in rows),encoding="utf-8")
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb")as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def pad_width(seqs:list[list[int]],pad:int,device,width:int):
 ids=torch.full((len(seqs),width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
 for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s,device=device);mask[i,:len(s)]=1
 return ids,mask
def route_map(items:list[dict])->dict[int,list[int]]:
 out={}
 for x in items:out.setdefault(int(x["layer"]),[]).append(int(x["head"]))
 return out

def build_jobs(tokenizer,rows:list[dict])->tuple[list[dict],dict]:
 idx={(r["family_id"],r["language"],r["surface"],r["meaning_swap"],r["query_property"]):r for r in rows if r["unit"]==35}
 jobs=[];drop=defaultdict(int)
 for r in rows:
  if r["unit"]!=35 or r["surface"]!=0 or r["meaning_swap"]!=0:continue
  dm=idx[(r["family_id"],r["language"],0,1,r["query_property"])]
  dq=idx[(r["family_id"],r["language"],0,0,1-r["query_property"])]
  src=lambda z:sorted({p for name in SOURCE_NAMES for p in z["regions"][name]})
  if len(r["prompt_ids"])!=len(dm["prompt_ids"]):drop["meaning_shape"]+=1;continue
  if len(src(r))!=len(src(dm)):drop["source_span"]+=1;continue
  for ci,entity in enumerate(r["entities"]):
   prefix=" "if r["language"]=="en"else"";cont=[int(x)for x in tokenizer.encode(prefix+entity,add_special_tokens=False)]
   jobs.append({"case_id":r["case_id"],"candidate_index":ci,"candidate":entity,"target":r["target"],"donor_target":dm["target"],
                "family_id":r["family_id"],"family":r["family"],"language":r["language"],"query_property":r["query_property"],
                "base":r["prompt_ids"]+cont,"meaning":dm["prompt_ids"]+cont,"query":dq["prompt_ids"]+cont,
                "prompt_length":len(r["prompt_ids"]),"query_prompt_length":len(dq["prompt_ids"]),
                "source_base":src(r),"source_meaning":src(dm)})
 return jobs,dict(drop)

class Controller:
 def __init__(self,model):
  self.layers=model_utils.get_layers(model);self.nh=int(model.config.num_attention_heads);self.nkv=int(model.config.num_key_value_heads);self.hd=int(model.config.head_dim);self.groups=self.nh//self.nkv
  self.mode="none";self.condition="";self.routes={};self.jobs=[];self.store={"base":{},"meaning":{},"query":{}};self.handles=[]
  for li in LATE:
   for kind,name in (("q","q_proj"),("k","k_proj"),("v","v_proj")):
    mod=getattr(self.layers[li].self_attn,name)
    def hook(_m,_a,out,li=li,kind=kind):return self.projection_hook(out,li,kind)
    self.handles.append(mod.register_forward_hook(hook))
   def opre(_m,args,li=li):
    x=args[0]
    if self.mode in self.store:self.store[self.mode][("o",li)]=x.detach().clone();return None
    if self.mode!="patch"or not self.condition:return None
    heads=self.routes.get(li,[])
    if not heads:return None
    if self.condition=="o_zero_top":
     y=x.clone().view(x.shape[0],x.shape[1],self.nh,self.hd)
     for bi,j in enumerate(self.jobs):y[bi,j["prompt_length"]-1:len(j["base"])-1,heads,:]=0
     return(y.reshape_as(x),*args[1:])
    if self.condition=="whole_head_meaning_top":
     donor=self.store["meaning"][("o",li)].to(x.device);y=x.clone().view(x.shape[0],x.shape[1],self.nh,self.hd);d=donor.view_as(y)
     for bi,j in enumerate(self.jobs):y[bi,j["prompt_length"]-1:len(j["base"])-1,heads,:]=d[bi,j["prompt_length"]-1:len(j["base"])-1,heads,:]
     return(y.reshape_as(x),*args[1:])
    return None
   self.handles.append(self.layers[li].self_attn.o_proj.register_forward_pre_hook(opre))
 def close(self):
  for h in self.handles:h.remove()
 def projection_hook(self,out,li,kind):
  if self.mode in self.store:self.store[self.mode][(kind,li)]=out.detach().clone();return None
  if self.mode!="patch"or not self.condition:return None
  c=self.condition;heads=self.routes.get(li,[])
  if not heads:return None
  patch_kind=(kind=="q"and(c.startswith("q_")or c.startswith("qkv_")))or(kind=="k"and(c.startswith("k_")or c.startswith("kv_")or c.startswith("qkv_")))or(kind=="v"and(c.startswith("v_")or c.startswith("kv_")or c.startswith("qkv_")))
  zero_kind=(kind=="q"and c=="q_zero_top")or(kind=="k"and c in("k_zero_top","kv_zero_top"))or(kind=="v"and c in("v_zero_top","kv_zero_top"))
  if not patch_kind and not zero_kind:return None
  y=out.clone();width=self.nh if kind=="q"else self.nkv;view=y.view(y.shape[0],y.shape[1],width,self.hd)
  selected=heads if kind=="q"else sorted({h//self.groups for h in heads})
  if zero_kind:
   for bi,j in enumerate(self.jobs):
    pos=range(j["prompt_length"]-1,len(j["base"])-1)if kind=="q"else j["source_base"]
    for p in pos:view[bi,p,selected,:]=0
   return view.reshape_as(out)
  source="query"if("query"in c)else("base"if"base"in c else"meaning")
  donor=self.store[source][(kind,li)].to(out.device).view_as(view)
  shuffled="shuffled"in c
  for bi,j in enumerate(self.jobs):
   di=(bi-1)%len(self.jobs)if shuffled else bi
   if kind=="q":
    donor_start=j["query_prompt_length"]-1 if source=="query" else j["prompt_length"]-1
    for oi,p in enumerate(range(j["prompt_length"]-1,len(j["base"])-1)):view[bi,p,selected,:]=donor[di,donor_start+oi,selected,:]
   else:
    dpos=j["source_meaning"]if source=="meaning"and not shuffled else j["source_base"]
    for bp,dp in zip(j["source_base"],dpos):view[bi,bp,selected,:]=donor[di,dp,selected,:]
  return view.reshape_as(out)

def score_logits(logits:torch.Tensor,jobs:list[dict])->list[float]:
 lp=torch.log_softmax(logits.float(),-1);vals=[]
 for bi,j in enumerate(jobs):
  cont=j["base"][j["prompt_length"]:]
  vals.append(float(sum(lp[bi,j["prompt_length"]-1+oi,t]for oi,t in enumerate(cont))))
 return vals

def run(model,tokenizer,jobs:list[dict],routes:dict)->list[dict]:
 device=model.get_input_embeddings().weight.device;control=Controller(model);rows=[]
 top=route_map(routes["top"]);matched=[route_map(x)for x in routes["matched_random_sets"]];allroutes={l:list(range(control.nh))for l in LATE}
 conditions=["no_patch","base_qkv_top","q_query_top","q_meaning_top","k_meaning_top","v_meaning_top","kv_meaning_top","qkv_meaning_top","whole_head_meaning_top","kv_query_top","kv_shuffled_top","q_query_all","kv_meaning_all","q_zero_top","k_zero_top","v_zero_top","kv_zero_top","o_zero_top"]+[f"kv_meaning_matched{i}"for i in range(5)]
 try:
  for start in range(0,len(jobs),8):
   batch=jobs[start:start+8];control.jobs=batch;width=max(max(len(j[s])for s in("base","meaning","query"))for j in batch)
   tensors={s:pad_width([j[s]for j in batch],tokenizer.pad_token_id,device,width)for s in("base","meaning","query")}
   base_scores=None
   for source in ("base","meaning","query"):
    control.mode=source;ids,mask=tensors[source]
    with torch.inference_mode():logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
    if source=="base":base_scores=score_logits(logits,batch)
   for condition in conditions:
    if condition=="no_patch":scores=base_scores
    else:
     if condition.endswith("_all"):control.routes=allroutes
     elif"matched"in condition:control.routes=matched[int(condition[-1])]
     else:control.routes=top
     control.condition=condition;control.mode="patch";ids,mask=tensors["base"]
     with torch.inference_mode():logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
     scores=score_logits(logits,batch)
    for j,s in zip(batch,scores):rows.append({"case_id":j["case_id"],"candidate":j["candidate"],"candidate_index":j["candidate_index"],"condition":condition,"score":s,
                                              "target":j["target"],"donor_target":j["donor_target"],"family":j["family"],"language":j["language"]})
   if(start+len(batch))%64==0:print(f"[phase2540] {start+len(batch)}/{len(jobs)}",flush=True)
 finally:control.close()
 return rows

def panels(rows:list[dict])->dict:
 grouped=defaultdict(list)
 for r in rows:grouped[(r["condition"],r["case_id"])].append(r)
 by=defaultdict(list)
 for (condition,_case),xs in grouped.items():
  scores={x["candidate"]:x["score"]for x in xs};meta=xs[0];pred=max(scores,key=scores.get);target=meta["target"];donor=meta["donor_target"];wrong=donor
  by[condition].append({"correct":pred==target,"donor_flip":pred==donor,"base_margin":scores[target]-scores[wrong],"donor_margin":scores[donor]-scores[target]})
 out={}
 for c,xs in by.items():out[c]={"n":len(xs),"accuracy":float(np.mean([x["correct"]for x in xs])),"donor_flip":float(np.mean([x["donor_flip"]for x in xs])),"mean_target_margin":float(np.mean([x["base_margin"]for x in xs])),"mean_donor_margin":float(np.mean([x["donor_margin"]for x in xs]))}
 return out

def diversify_matched(routes:dict)->dict:
 table=read(P2539/"analysis/all_route_features.jsonl");topkeys={(int(x["layer"]),int(x["head"]))for x in routes["top"]}
 feature_names=("head_rms","external_mass","entropy","wo_norm")
 values=np.asarray([[float(x[k])for k in feature_names]for x in table]);mu=values.mean(0);sd=values.std(0)+1e-9
 bykey={(int(x["layer"]),int(x["head"])):x for x in table};sets=[]
 for rep in range(5):
  used=set();chosen=[]
  for ti,t in enumerate(routes["top"]):
   key=(int(t["layer"]),int(t["head"]));f=bykey[key];target=(np.asarray([f[k]for k in feature_names])-mu)/sd
   candidates=[x for x in table if int(x["layer"])==key[0]and(int(x["layer"]),int(x["head"]))not in topkeys|used]
   if not candidates:candidates=[x for x in table if(int(x["layer"]),int(x["head"]))not in topkeys|used]
   candidates.sort(key=lambda x:float(np.sum((((np.asarray([x[k]for k in feature_names])-mu)/sd)-target)**2)))
   pick=candidates[min((rep+ti)%min(5,len(candidates)),len(candidates)-1)];used.add((int(pick["layer"]),int(pick["head"])))
   chosen.append({"layer":int(pick["layer"]),"head":int(pick["head"]),"features":{k:float(pick[k])for k in feature_names}})
  sets.append(chosen)
 out=dict(routes);out["matched_random_sets"]=sets;return out

def append_memo(r:dict)->None:
 if f"## Phase {PHASE}:"in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
 text=rf"""


## Phase {PHASE}: destination-Q、source-K/V与$W_O$分离因果锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Qwen3-4B BF16非量化、unit35、surface0、32族英中双语的exact-shape candidate锁箱上，unit34全场冻结top32及五组同层位/物理特征匹配路线。对每个base分别构造“只换query”和“只换事实绑定”的donor，在layer20–35按真实投影接口分别替换答案阶段Q、事实实体/关系/值token的K、V、K+V、Q+K+V与o_proj前whole-head；另做全route、base no-op、来自未来query的source-KV因果顺序对照、batch错source、Q/K/V/KV/whole-head zero和五组matched路线。GQA的K/V干预按共享KV-head组执行，不伪装为单query-head独立操作。

$$\operatorname{{do}}(Q_a\leftarrow Q_a'),\quad \operatorname{{do}}(K_S\leftarrow K_S'),\quad \operatorname{{do}}(V_S\leftarrow V_S'),\quad \operatorname{{do}}((K,V)_S\leftarrow(K,V)_S').$$

**结果汇总。** 设计 `{json.dumps(r['design'],ensure_ascii=False)}`；各条件 `{json.dumps(r['panels'],ensure_ascii=False)}`；职责裁决 `{json.dumps(r['adjudication'],ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2540_c125697_c129792_qkv_separated_causal_lockbox.py`；逐候选logprob、pair覆盖、路线合同和final位于`{OUT}`。

**分析与理论进展。** 该Phase第一次把架构公式中的Q/K/V角色变成彼此独立的干预对象。只有query-donor Q改变答案，才支持Q携带当前读取请求；只有meaning-donor K改变答案，才支持地址变化具有因果作用；V或K+V改变答案，才支持source内容经共享KV组搬运。任何分量阴性只关闭该剂量和route集合下的充分性，不关闭其自然参与；whole-head和全route是上界，不用来替单分量背书。来自未来query的source-KV应因因果mask近似no-op，是实现阳性/阴性校验。

**问题硬伤与结论。** projection patch仍可能分布外；K/V在GQA内无法按单query head隔离；多层连续替换会累积兼容性误差；候选logprob仍含输出身份编译；matched路线只平衡可测协变量。Q/K/V的语义职责按实测差异分级，不因公式直觉预先宣布闭合。
"""
 with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)

def main()->None:
 prior=load(P2539/"analysis/final.json");routes=diversify_matched(prior["routes"]);material=read(P2538/"material/token_atomic_rows.jsonl");model,tokenizer,_=model_utils.load_model("qwen3",dtype=torch.bfloat16,use_8bit=False)
 try:jobs,dropped=build_jobs(tokenizer,material);rows=run(model,tokenizer,jobs,routes)
 finally:model_utils.release_model(model);gc.collect()
 path=OUT/"causal/qkv_candidate_scores.jsonl";write(path,rows);panel=panels(rows)
 matched=[panel[f"kv_meaning_matched{i}"]for i in range(5)]
 adjudication={"q_query_selective_sufficiency":bool(panel["q_query_top"]["donor_flip"]>np.mean([m["donor_flip"]for m in matched])),
               "k_address_sufficiency":panel["k_meaning_top"]["donor_flip"]>.05,"v_content_sufficiency":panel["v_meaning_top"]["donor_flip"]>.05,
               "kv_joint_sufficiency":panel["kv_meaning_top"]["donor_flip"]>.05,"qkv_joint_sufficiency":panel["qkv_meaning_top"]["donor_flip"]>.05,
               "whole_head_upper_bound":panel["whole_head_meaning_top"]["donor_flip"],"language_mechanism_closed":False}
 design={"candidate_sequences":len(jobs),"paired_cases":len(jobs)//2,"families":len({j["family"]for j in jobs}),"languages":sorted({j["language"]for j in jobs}),"conditions":len(panel),"dropped":dropped,"top_routes":32,"matched_sets":5}
 unique_sets=len({tuple(sorted((x["layer"],x["head"])for x in s))for s in routes["matched_random_sets"]})
 route_path=OUT/"analysis/intervention_routes.json";save(route_path,routes)
 checks={"source_passed":prior["all_checks_passed"],"at_least_100_pairs":len(jobs)//2>=100,"baseline_gate":panel["no_patch"]["accuracy"]>=.9,
         "base_qkv_noop":abs(panel["base_qkv_top"]["mean_target_margin"]-panel["no_patch"]["mean_target_margin"])<.05,
         "future_query_source_noop":abs(panel["kv_query_top"]["mean_target_margin"]-panel["no_patch"]["mean_target_margin"])<.05,
         "five_distinct_matched_controls":len(matched)==5 and unique_sets==5,"all_conditions_complete":all(x["n"]==len(jobs)//2 for x in panel.values()),"claim_boundary":True}
 result={"phase":PHASE,"campaign":CAMPAIGN,"model":"Qwen3-4B BF16 CUDA nonquantized","design":design,"panels":panel,"adjudication":adjudication,
         "files":{"scores":{"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path)},"routes":{"path":str(route_path),"bytes":route_path.stat().st_size,"sha256":sha(route_path)}},"checks":checks,"all_checks_passed":all(checks.values())}
 save(OUT/"analysis/final.json",result)
 if result["all_checks_passed"]:append_memo(result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if not result["all_checks_passed"]:raise RuntimeError(checks)

if __name__=="__main__":main()
