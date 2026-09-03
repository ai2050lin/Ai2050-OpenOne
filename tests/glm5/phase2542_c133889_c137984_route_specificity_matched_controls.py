#!/usr/bin/env python3
"""Matched direct-output controls and dose tests for route specificity."""
from __future__ import annotations
import gc,hashlib,json,re,sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result"
P2538=RESULT/"phase2538_c117505_c121600_token_atomic_hypergraph_behavior";P2540=RESULT/"phase2540_c125697_c129792_qkv_separated_causal_lockbox";P2541=RESULT/"phase2541_c129793_c133888_source_mlp_kv_write_dynamics"
OUT=RESULT/"phase2542_c133889_c137984_route_specificity_matched_controls";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE,CAMPAIGN=2542,"C133889-C137984";LATE=tuple(range(20,36))
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
def route_map(items):
 out={}
 for x in items:out.setdefault(int(x["layer"]),[]).append(int(x["head"]))
 return out
def pad(seqs,pad,device):
 w=max(map(len,seqs));ids=torch.full((len(seqs),w),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
 for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s,device=device);mask[i,:len(s)]=1
 return ids,mask

def compile_cases(tokenizer,material):
 relations=[];controls=[]
 for r in material:
  if r["unit"]!=35 or r["surface"]!=0 or r["meaning_swap"]!=0:continue
  base={"case_id":"rel_"+r["case_id"],"task":"relation","family":r["family"],"language":r["language"],"entities":r["entities"],"target":r["target"],"prompt_ids":r["prompt_ids"]}
  relations.append(base)
  e0,e1=r["entities"]
  if r["language"]=="en":text=f"Reference names: [{e0}] and [{e1}]. The explicitly requested answer is [{r['target']}]. Candidates: [{e0}] or [{e1}]. Return only the complete requested name. Answer:"
  else:text=f"参考名称：[{e0}]和[{e1}]。明确指定的答案是[{r['target']}]。候选：[{e0}]或[{e1}]。只返回指定的完整名称。答案："
  controls.append({"case_id":"ctrl_"+r["case_id"],"task":"direct_output_control","family":r["family"],"language":r["language"],"entities":r["entities"],"target":r["target"],"prompt_ids":[int(x)for x in tokenizer.encode(text,add_special_tokens=False)]})
 return relations,controls

class Zero:
 def __init__(self,model):
  self.layers=model_utils.get_layers(model);self.nh=int(model.config.num_attention_heads);self.hd=int(model.config.head_dim);self.active={};self.jobs=[];self.handles=[]
  for li in LATE:
   def hook(_m,args,li=li):
    heads=self.active.get(li,[])
    if not heads:return None
    x=args[0];y=x.clone().view(x.shape[0],x.shape[1],self.nh,self.hd)
    for bi,j in enumerate(self.jobs):
     for p in range(j["prompt_length"]-1,len(j["sequence"])-1):y[bi,p,heads,:]=0
    return(y.reshape_as(x),*args[1:])
   self.handles.append(self.layers[li].self_attn.o_proj.register_forward_pre_hook(hook))
 def close(self):
  for h in self.handles:h.remove()
def jobs(tokenizer,cases):
 out=[]
 for r in cases:
  for e in r["entities"]:
   cont=[int(x)for x in tokenizer.encode((" "if r["language"]=="en"else"")+e,add_special_tokens=False)]
   out.append({**r,"candidate":e,"prompt_length":len(r["prompt_ids"]),"sequence":r["prompt_ids"]+cont})
 return out
def run(model,tokenizer,job_rows,routes):
 top=routes["top"];matched=routes["matched_random_sets"];conditions={"no_patch":{}}
 for d in(8,16,24,32):conditions[f"zero_top{d}"]=route_map(top[:d])
 for i,s in enumerate(matched):conditions[f"zero_matched32_{i}"]=route_map(s)
 conditions["zero_all_late"]={l:list(range(int(model.config.num_attention_heads)))for l in LATE}
 device=model.get_input_embeddings().weight.device;z=Zero(model);out=[]
 try:
  for start in range(0,len(job_rows),8):
   b=job_rows[start:start+8];z.jobs=b;ids,mask=pad([j["sequence"]for j in b],tokenizer.pad_token_id,device)
   for c,route in conditions.items():
    z.active=route
    with torch.inference_mode():logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float()
    lp=torch.log_softmax(logits,-1)
    for bi,j in enumerate(b):
     cont=j["sequence"][j["prompt_length"]:];score=float(sum(lp[bi,j["prompt_length"]-1+o,t]for o,t in enumerate(cont)))
     out.append({"case_id":j["case_id"],"task":j["task"],"family":j["family"],"language":j["language"],"candidate":j["candidate"],"target":j["target"],"condition":c,"score":score})
   if(start+len(b))%128==0:print(f"[phase2542] {start+len(b)}/{len(job_rows)}",flush=True)
 finally:z.close()
 return out
def panels(rows):
 g=defaultdict(list)
 for r in rows:g[(r["task"],r["condition"],r["case_id"])].append(r)
 by=defaultdict(list)
 for(task,c,_),xs in g.items():
  s={x["candidate"]:x["score"]for x in xs};m=xs[0];wrong=next(x for x in s if x!=m["target"]);by[(task,c)].append((max(s,key=s.get)==m["target"],s[m["target"]]-s[wrong]))
 out={}
 for(task,c),xs in by.items():out.setdefault(task,{})[c]={"n":len(xs),"accuracy":float(np.mean([x[0]for x in xs])),"mean_margin":float(np.mean([x[1]for x in xs]))}
 return out

def append_memo(r):
 if f"## Phase {PHASE}:"in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
 text=rf"""


## Phase {PHASE}: 晚层route联盟的通用输出混杂与五重匹配对照（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 使用Qwen3-4B BF16非量化，对unit35的32族×英中×双query共128个关系case，并为每个case构造相同实体、相同候选、相同目标输出的“答案已在问题中直接给出”控制。两类任务都用多token候选总logprob，避免原自由生成控制仅0.375的行为门。持续zero unit34冻结top8/16/24/32、五组互不相同且同层位并按head RMS/source mass/entropy/$W_O$范数匹配的32-route集合，以及全部晚层heads；比较关系损伤与一般答案输出损伤。

$$S_{{\rm rel}}(d)=\Delta\operatorname{{Acc}}_{{\rm relation}}(d)-\Delta\operatorname{{Acc}}_{{\rm direct}}(d).$$

**结果汇总。** 两类条件 `{json.dumps(r['panels'],ensure_ascii=False)}`；选择性 `{json.dumps(r['specificity'],ensure_ascii=False)}`；设计 `{json.dumps(r['design'],ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2542_c133889_c137984_route_specificity_matched_controls.py`；控制材料、逐候选分数和final位于`{OUT}`。

**分析与理论进展。** 若top32同时摧毁direct-output控制，它主要包含一般答案编译/多token输出功能；若关系损伤显著更大，才支持关系条件附加作用。五组matched集合用于估计非top路线在相似层位和物理强度下的损伤分布，不能当随机化实验。剂量曲线揭示联盟累计效应，但嵌套集合不支持逐head独立排序或最小联盟。

**问题硬伤与结论。** direct控制显式暴露答案，计算需求低于关系绑定；两者prompt长度和局部词法不完全相同；head zero是强干预；所有case共享少量实体。结果只回答路线损伤是否超出一般输出需求，不把差值直接称“纯语义效应”。
"""
 with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)
def main():
 p40=load(P2540/"analysis/final.json");p41=load(P2541/"analysis/final.json");routes=load(Path(p40["files"]["routes"]["path"]));material=read(P2538/"material/token_atomic_rows.jsonl")
 model,tokenizer,_=model_utils.load_model("qwen3",dtype=torch.bfloat16,use_8bit=False)
 try:relation,control=compile_cases(tokenizer,material);jr=jobs(tokenizer,relation+control);rows=run(model,tokenizer,jr,routes)
 finally:model_utils.release_model(model);gc.collect()
 mp=OUT/"material/direct_controls.jsonl";write(mp,control);rp=OUT/"causal/route_specificity_scores.jsonl";write(rp,rows);pan=panels(rows)
 rel=pan["relation"];ctrl=pan["direct_output_control"];specificity={}
 for c in rel:
  specificity[c]={"relation_accuracy_loss":rel["no_patch"]["accuracy"]-rel[c]["accuracy"],"control_accuracy_loss":ctrl["no_patch"]["accuracy"]-ctrl[c]["accuracy"],"excess_relation_loss":(rel["no_patch"]["accuracy"]-rel[c]["accuracy"])-(ctrl["no_patch"]["accuracy"]-ctrl[c]["accuracy"])}
 design={"relation_cases":len(relation),"direct_control_cases":len(control),"conditions":len(rel),"matched_sets":5,"all_target_multitoken":True}
 checks={"sources_passed":p40["all_checks_passed"]and p41["all_checks_passed"],"case_counts":len(relation)==128 and len(control)==128,"relation_baseline":rel["no_patch"]["accuracy"]>=.95,"control_baseline":ctrl["no_patch"]["accuracy"]>=.95,
         "five_matched_complete":all(rel[f"zero_matched32_{i}"]["n"]==128 for i in range(5)),"top_dose_complete":all(rel[f"zero_top{d}"]["n"]==128 for d in(8,16,24,32)),"claim_boundary":True}
 files={"material":{"path":str(mp),"bytes":mp.stat().st_size,"sha256":sha(mp)},"scores":{"path":str(rp),"bytes":rp.stat().st_size,"sha256":sha(rp)}}
 result={"phase":PHASE,"campaign":CAMPAIGN,"model":"Qwen3-4B BF16 CUDA nonquantized","design":design,"panels":pan,"specificity":specificity,"files":files,"checks":checks,"all_checks_passed":all(checks.values())}
 save(OUT/"analysis/final.json",result)
 if result["all_checks_passed"]:append_memo(result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if not result["all_checks_passed"]:raise RuntimeError(checks)
if __name__=="__main__":main()
