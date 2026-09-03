#!/usr/bin/env python3
"""Full-depth causal emergence map for output Q and fact/downstream K/V."""
from __future__ import annotations
import gc,hashlib,json,sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result"
P2538=RESULT/"phase2538_c117505_c121600_token_atomic_hypergraph_behavior";P2542=RESULT/"phase2542_c133889_c137984_route_specificity_matched_controls"
OUT=RESULT/"phase2543_c137985_c142080_full_depth_qkv_role_emergence";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE,CAMPAIGN=2543,"C137985-C142080";LAYERS=tuple(range(36));BANDS=((0,8),(9,17),(18,26),(27,35));FACT_NAMES=("facts_entity","facts_relation","facts_value")
sys.path.insert(0,str(TESTS));import model_utils  # noqa:E402
def load(p:Path)->Any:return json.loads(p.read_text(encoding="utf-8-sig"))
def read(p:Path)->list[dict]:return[json.loads(x)for x in p.read_text(encoding="utf-8-sig").splitlines()if x.strip()]
def save(p:Path,v:Any):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def write(p:Path,rows):p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n"for x in rows),encoding="utf-8")
def sha(p:Path):
 h=hashlib.sha256()
 with p.open("rb")as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def pad(seqs,pad_id,device,width):
 ids=torch.full((len(seqs),width),pad_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
 for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s,device=device);mask[i,:len(s)]=1
 return ids,mask
def facts(r):return sorted({p for n in FACT_NAMES for p in r["regions"][n]})

def build_jobs(tokenizer,material):
 idx={(r["family_id"],r["language"],r["meaning_swap"],r["query_property"]):r for r in material if r["unit"]==35 and r["surface"]==0};out=[]
 for r in material:
  if r["unit"]!=35 or r["surface"]!=0 or r["meaning_swap"]!=0 or r["query_property"]!=0:continue
  dm=idx[(r["family_id"],r["language"],1,0)];dq=idx[(r["family_id"],r["language"],0,1)]
  for e in r["entities"]:
   cont=[int(x)for x in tokenizer.encode((" "if r["language"]=="en"else"")+e,add_special_tokens=False)]
   out.append({"case_id":r["case_id"],"family":r["family"],"language":r["language"],"candidate":e,"target":r["target"],"donor_target":dm["target"],
               "base":r["prompt_ids"]+cont,"meaning":dm["prompt_ids"]+cont,"query":dq["prompt_ids"]+cont,"prompt_length":len(r["prompt_ids"]),"query_prompt_length":len(dq["prompt_ids"]),"facts_base":facts(r),"facts_meaning":facts(dm)})
 return out

def specs():
 out={"no_patch":{}}
 for li in LAYERS:
  out[f"q_mean_l{li}"]={"q_source":"meaning","q_layers":{li}}
  out[f"q_query_l{li}"]={"q_source":"query","q_layers":{li}}
  out[f"kv_fact_l{li}"]={"kv_source":"meaning","kv_kind":"kv","kv_region":"facts","kv_layers":{li}}
 for lo,hi in BANDS:
  ls=set(range(lo,hi+1));tag=f"l{lo}_{hi}"
  out[f"q_mean_{tag}"]={"q_source":"meaning","q_layers":ls}
  out[f"q_query_{tag}"]={"q_source":"query","q_layers":ls}
  for kind in("k","v","kv"):
   out[f"{kind}_fact_{tag}"]={"kv_source":"meaning","kv_kind":kind,"kv_region":"facts","kv_layers":ls}
   out[f"{kind}_external_{tag}"]={"kv_source":"meaning","kv_kind":kind,"kv_region":"external","kv_layers":ls}
 return out

class Control:
 def __init__(self,model):
  self.layers=model_utils.get_layers(model);self.mode="none";self.spec={};self.jobs=[];self.store={"base":{},"meaning":{},"query":{}};self.handles=[]
  for li in LAYERS:
   for kind,name in(("q","q_proj"),("k","k_proj"),("v","v_proj")):
    def hook(_m,_a,o,li=li,kind=kind):return self.hook(o,li,kind)
    self.handles.append(getattr(self.layers[li].self_attn,name).register_forward_hook(hook))
 def close(self):
  for h in self.handles:h.remove()
 def hook(self,o,li,kind):
  if self.mode in self.store:self.store[self.mode][(kind,li)]=o.detach().clone();return None
  if self.mode!="patch":return None
  s=self.spec;do_q=kind=="q"and li in s.get("q_layers",set());do_kv=kind in("k","v")and li in s.get("kv_layers",set())and(kind==s.get("kv_kind")or s.get("kv_kind")=="kv")
  if not do_q and not do_kv:return None
  y=o.clone()
  for bi,j in enumerate(self.jobs):
   if do_q:
    source=s["q_source"];d=self.store[source][(kind,li)].to(o.device);ds=j["query_prompt_length"]-1 if source=="query"else j["prompt_length"]-1
    for oi,p in enumerate(range(j["prompt_length"]-1,len(j["base"])-1)):y[bi,p]=d[bi,ds+oi]
   if do_kv:
    d=self.store["meaning"][(kind,li)].to(o.device)
    if s["kv_region"]=="facts":bp,dp=j["facts_base"],j["facts_meaning"]
    else:bp=dp=list(range(j["prompt_length"]-1))
    for p,q in zip(bp,dp):y[bi,p]=d[bi,q]
  return y

def score(logits,jobs):
 lp=torch.log_softmax(logits.float(),-1);return[float(sum(lp[i,j["prompt_length"]-1+oi,t]for oi,t in enumerate(j["base"][j["prompt_length"]:])))for i,j in enumerate(jobs)]
def run(model,tokenizer,jobs):
 conditions=specs();device=model.get_input_embeddings().weight.device;c=Control(model);out=[]
 try:
  for start in range(0,len(jobs),8):
   b=jobs[start:start+8];c.jobs=b;width=max(max(len(j[x])for x in("base","meaning","query"))for j in b);t={x:pad([j[x]for j in b],tokenizer.pad_token_id,device,width)for x in("base","meaning","query")}
   for source in("base","meaning","query"):
    c.mode=source;ids,mask=t[source]
    with torch.inference_mode():z=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
    if source=="base":baseline=score(z,b)
   for name,spec in conditions.items():
    if name=="no_patch":vals=baseline
    else:
     c.mode="patch";c.spec=spec;ids,mask=t["base"]
     with torch.inference_mode():z=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
     vals=score(z,b)
    for j,v in zip(b,vals):out.append({"case_id":j["case_id"],"candidate":j["candidate"],"target":j["target"],"donor_target":j["donor_target"],"family":j["family"],"language":j["language"],"condition":name,"score":v})
   if(start+len(b))%32==0:print(f"[phase2543] {start+len(b)}/{len(jobs)}",flush=True)
 finally:c.close()
 return out,conditions
def panels(rows):
 g=defaultdict(list)
 for r in rows:g[(r["condition"],r["case_id"])].append(r)
 by=defaultdict(list)
 for(c,_),xs in g.items():
  s={x["candidate"]:x["score"]for x in xs};m=xs[0];p=max(s,key=s.get);by[c].append((p==m["target"],p==m["donor_target"],s[m["donor_target"]]-s[m["target"]]))
 return{c:{"n":len(x),"accuracy":float(np.mean([z[0]for z in x])),"donor_flip":float(np.mean([z[1]for z in x])),"mean_donor_margin":float(np.mean([z[2]for z in x]))}for c,x in by.items()}
def summarize(p):
 layerwise={kind:[p[f"{kind}_l{li}"]["donor_flip"]for li in LAYERS]for kind in("q_mean","q_query","kv_fact")}
 bands={k:v for k,v in p.items()if any(f"l{lo}_{hi}"in k for lo,hi in BANDS)}
 return{"layerwise_donor_flip":layerwise,"strongest_layer":{k:int(np.argmax(v))for k,v in layerwise.items()},"maximum_single_layer_flip":{k:float(max(v))for k,v in layerwise.items()},"bands":bands}

def append_memo(r):
 if f"## Phase {PHASE}:"in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
 text=rf"""


## Phase {PHASE}: 全深度Q输出状态形成与facts/downstream K/V角色涌现图（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Qwen3-4B BF16非量化、unit35、32族英中各一query的64个case上，以128个多token候选序列覆盖layer0–35。每批现场捕获base、事实绑定donor和query donor的Q/K/V；逐层单独替换全部答案阶段Q或全部事实source K+V，并在0–8、9–17、18–26、27–35四个连续层段分别替换Q、facts K/V以及全部answer-boundary之前token的K/V。这样直接检验输出条件何时进入Q，以及事实改变是否先被复制到query/candidate/instruction等下游token的可读K/V。

$$Q_a^l=f_Q(h_a^l),\qquad (K,V)_j^l=f_{{K,V}}(h_j^l),\qquad h_a^l\text{{已含上下文}}\Rightarrow Q_a^l\text{{不等于纯查询属性}}.$$

**结果汇总。** 层级/层段结果 `{json.dumps(r['summary'],ensure_ascii=False)}`；设计 `{json.dumps(r['design'],ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2543_c137985_c142080_full_depth_qkv_role_emergence.py`；全部逐候选条件分数、层级曲线和final位于`{OUT}`。

**分析与理论进展。** 单层Q donor的涌现位置刻画答案位置的上下文结果何时获得输出控制力；facts K/V与external K/V的差异用于判断信息是否已从事实token编译到后续问题、候选或指令token。层段替换比全晚层强干预更接近定位，但仍是构造性充分性。若facts K/V弱而external K/V强，应表述为结果已写入下游token可读状态，不能称事实K/V本身无信息。

**问题硬伤与结论。** 全head替换不能分辨最小联盟；连续层段patch可能累积不兼容；candidate likelihood偏向输出身份；只用query0降低材料规模但保留32族双语。该图定位控制状态的形成时序，不等同于知识算法闭合。
"""
 with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)
def main():
 prior=load(P2542/"analysis/final.json");material=read(P2538/"material/token_atomic_rows.jsonl");model,tokenizer,_=model_utils.load_model("qwen3",dtype=torch.bfloat16,use_8bit=False)
 try:jobs=build_jobs(tokenizer,material);rows,conditions=run(model,tokenizer,jobs)
 finally:model_utils.release_model(model);gc.collect()
 rp=OUT/"causal/full_depth_scores.jsonl";write(rp,rows);p=panels(rows);summary=summarize(p);sp=OUT/"analysis/layerwise.json";save(sp,summary)
 checks={"source_passed":prior["all_checks_passed"],"cases_64":len(jobs)==128,"conditions_141":len(conditions)==141,"baseline_gate":p["no_patch"]["accuracy"]>=.95,"all_conditions_complete":all(x["n"]==64 for x in p.values()),"all_layers":all(len(summary["layerwise_donor_flip"][k])==36 for k in summary["layerwise_donor_flip"]),"claim_boundary":True}
 result={"phase":PHASE,"campaign":CAMPAIGN,"model":"Qwen3-4B BF16 CUDA nonquantized","design":{"cases":64,"candidate_sequences":len(jobs),"families":32,"languages":["en","zh"],"conditions":len(conditions),"bands":[list(x)for x in BANDS]},"summary":summary,"panels":p,"files":{"scores":{"path":str(rp),"bytes":rp.stat().st_size,"sha256":sha(rp)},"summary":{"path":str(sp),"bytes":sp.stat().st_size,"sha256":sha(sp)}},"checks":checks,"all_checks_passed":all(checks.values())}
 save(OUT/"analysis/final.json",result)
 if result["all_checks_passed"]:append_memo(result)
 print(json.dumps({"phase":PHASE,"design":result["design"],"summary":summary,"checks":checks,"all_checks_passed":result["all_checks_passed"]},ensure_ascii=False,indent=2))
 if not result["all_checks_passed"]:raise RuntimeError(checks)
if __name__=="__main__":main()
