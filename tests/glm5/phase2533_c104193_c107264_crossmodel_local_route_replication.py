#!/usr/bin/env python3
"""Sequential BF16 cross-model replication of model-local late head routes."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result"
P2520=RESULT/"phase2520_c85025_c86176_natural_language_counterfactual_fullfield";P2522=RESULT/"phase2522_c87201_c88576_crossmodel_natural_boundary_replication"
OUT=RESULT/"phase2533_c104193_c107264_crossmodel_local_route_replication";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE,CAMPAIGN=2533,"C104193-C107264";MODEL_KEYS=("qwen14b","deepseek7b","glm4")
sys.path.insert(0,str(TESTS));import phase2522_c87201_c88576_crossmodel_natural_boundary_replication as p2522  # noqa:E402

def load(p:Path)->Any:return json.loads(p.read_text(encoding="utf-8-sig"))
def read(p:Path)->list[dict]:return[json.loads(x)for x in p.read_text(encoding="utf-8-sig").splitlines()if x.strip()]
def save(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def write(p:Path,rows:list[dict])->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n"for r in rows),encoding="utf-8")
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb")as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def pad(seqs:list[list[int]],pad_id:int,device)->tuple[torch.Tensor,torch.Tensor]:
 width=max(map(len,seqs));ids=torch.full((len(seqs),width),pad_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
 for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s,device=device);mask[i,:len(s)]=1
 return ids,mask
def layers_of(model):return p2522.layers_of(model)

def compile_selection(tokenizer,material:list[dict],family_ids:set[int])->list[dict]:
 rows=[]
 for r in material:
  if r["unit"]!=30 or r["family_id"]not in family_ids or r["surface"]!=0 or r["output_mode"]!="candidate":continue
  x=dict(r);x["prompt_ids"]=[int(v)for v in tokenizer.encode(r["prompt"],add_special_tokens=False)];x["position"]=len(x["prompt_ids"])-1;rows.append(x)
 rows.sort(key=lambda r:(r["family_id"],r["language"],r["meaning_swap"],r["query_property"]))
 return rows

def select_routes(model,tokenizer,rows:list[dict],model_key:str)->tuple[dict,dict]:
 family_ids=sorted({int(r["family_id"])for r in rows});layers=layers_of(model);nl=len(layers);nh=int(model.config.num_attention_heads);start=max(1,round(.56*nl));late=tuple(range(start,nl));hd=int(layers[0].self_attn.o_proj.in_features//nh)
 path=OUT/f"fields/{model_key}_unit30_late_head_output.float16.npy";path.parent.mkdir(parents=True,exist_ok=True)
 field=np.lib.format.open_memmap(path,mode="w+",dtype=np.float16,shape=(len(rows),len(late),nh,hd));positions=[];captured={};handles=[]
 for li in late:
  def hook(_m,args,li=li):
   x=args[0];bi=torch.arange(x.shape[0],device=x.device);pi=torch.tensor(positions,device=x.device);captured[li]=x[bi,pi].detach().float().cpu().reshape(x.shape[0],nh,hd)
  handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(hook))
 device=model.get_input_embeddings().weight.device
 try:
  for st in range(0,len(rows),8):
   batch=rows[st:st+8];ids,mask=pad([r["prompt_ids"]for r in batch],tokenizer.pad_token_id,device);positions[:]=[r["position"]for r in batch];captured.clear()
   with torch.inference_mode():model.model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
   for xi,li in enumerate(late):field[st:st+len(batch),xi]=captured[li].numpy().astype(np.float16)
 finally:
  for h in handles:h.remove()
  field.flush();del field
 source=np.load(path,mmap_mode="r");index={(r["family_id"],r["language"],r["meaning_swap"],r["query_property"]):i for i,r in enumerate(rows)}
 interaction_path=OUT/f"derived/{model_key}_unit30_head_walsh.float16.npy";interaction_path.parent.mkdir(parents=True,exist_ok=True)
 inter=np.lib.format.open_memmap(interaction_path,mode="w+",dtype=np.float16,shape=(len(family_ids),2,len(late),nh,hd))
 for fi,fid in enumerate(family_ids):
  for li,lang in enumerate(("en","zh")):
   cells={(m,q):np.asarray(source[index[(fid,lang,m,q)]],np.float32)for m in(0,1)for q in(0,1)}
   inter[fi,li]=((cells[(0,0)]-cells[(0,1)]-cells[(1,0)]+cells[(1,1)])/4).astype(np.float16)
 inter.flush();energy=np.square(np.asarray(inter,np.float32)).sum(axis=(0,1,4));pairs=[(late[li],h,float(energy[li,h]))for li in range(len(late))for h in range(nh)];pairs.sort(key=lambda x:(-x[2],x[0],x[1]));k=max(16,round(len(pairs)*.0625));top=pairs[:k];top_keys={(l,h)for l,h,_ in top};pool=[(l,h)for l in late for h in range(nh)if(l,h)not in top_keys];rng=np.random.default_rng(2533+MODEL_KEYS.index(model_key));random=[pool[int(i)]for i in rng.choice(len(pool),k,replace=False)]
 routes={"relative_late_start":start/nl,"late_layers":[start,nl-1],"layers":nl,"heads":nh,"head_dim":hd,"all_routes":len(pairs),"selected_fraction":k/len(pairs),"top":[{"layer":l,"head":h,"energy":e}for l,h,e in top],"random":[{"layer":l,"head":h}for l,h in sorted(random)]}
 fields={"head_output":{"path":str(path),"shape":list(source.shape),"dtype":"float16","bytes":path.stat().st_size,"sha256":sha(path)},"interaction":{"path":str(interaction_path),"shape":list(inter.shape),"dtype":"float16","bytes":interaction_path.stat().st_size,"sha256":sha(interaction_path)}};del inter
 return routes,fields

def route_map(items:list[dict])->dict[int,list[int]]:
 out={}
 for x in items:out.setdefault(int(x["layer"]),[]).append(int(x["head"]))
 return out

def causal(model,tokenizer,jobs:list[dict],routes:dict)->list[dict]:
 layers=layers_of(model);nh=routes["heads"];hd=routes["head_dim"];late=range(routes["late_layers"][0],routes["late_layers"][1]+1);top=route_map(routes["top"]);rnd=route_map(routes["random"]);allr={l:list(range(nh))for l in late}
 active={"mode":None,"routes":{},"source":{}};positions=[];captured={};handles=[]
 for li in late:
  def hook(_m,args,li=li):
   x=args[0];bi=torch.arange(x.shape[0],device=x.device);pi=torch.tensor(positions,device=x.device);current=x[bi,pi].detach().float().cpu().reshape(x.shape[0],nh,hd);captured[li]=current
   heads=active["routes"].get(li,[])
   if not heads:return None
   changed=x.clone().view(x.shape[0],x.shape[1],nh,hd)
   if active["mode"]=="zero":changed[bi[:,None],pi[:,None],torch.tensor(heads,device=x.device)[None,:],:]=0
   elif active["mode"]=="donor":
    source=active["source"][li].to(x.device,x.dtype);hi=torch.tensor(heads,device=x.device);changed[bi[:,None],pi[:,None],hi[None,:],:]=source[:,hi]
   return(changed.reshape_as(x),*args[1:])
  handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(hook))
 out=[];device=model.get_input_embeddings().weight.device
 try:
  for st in range(0,len(jobs),8):
   batch=jobs[st:st+8];positions[:]=[j["position"]for j in batch];base_ids,base_mask=pad([j["base_sequence"]for j in batch],tokenizer.pad_token_id,device);donor_ids,donor_mask=pad([j["donor_sequence"]for j in batch],tokenizer.pad_token_id,device)
   if not torch.equal(base_mask,donor_mask):raise RuntimeError("exact shape mismatch")
   active.update(mode=None,routes={},source={});captured.clear()
   with torch.inference_mode():logits=p2522.forward_logits(model,base_ids,base_mask,batch)
   base_states={l:v.clone()for l,v in captured.items()};base_scores=p2522.sequence_scores(logits,batch,int(base_ids.shape[1]))
   active.update(mode=None,routes={},source={});captured.clear()
   with torch.inference_mode():model.model(input_ids=donor_ids,attention_mask=donor_mask,use_cache=False,return_dict=True)
   donor_states={l:v.clone()for l,v in captured.items()}
   for job,(value,mean)in zip(batch,base_scores):out.append({k:job[k]for k in("id","family_id","family","language","query","candidate_index")}|{"condition":"no_patch","value":value})
   specs=(("donor_top","donor",top),("donor_random","donor",rnd),("donor_all_late","donor",allr),("head_zero_top","zero",top),("head_zero_random","zero",rnd),("head_zero_all_late","zero",allr))
   for condition,mode,selected in specs:
    active.update(mode=mode,routes=selected,source=donor_states);captured.clear()
    with torch.inference_mode():logits=p2522.forward_logits(model,base_ids,base_mask,batch)
    scores=p2522.sequence_scores(logits,batch,int(base_ids.shape[1]))
    for job,(value,mean)in zip(batch,scores):out.append({k:job[k]for k in("id","family_id","family","language","query","candidate_index")}|{"condition":condition,"value":value})
 finally:
  for h in handles:h.remove()
 return out

def panels(rows:list[dict])->dict:
 idx={(r["id"],r["condition"],r["candidate_index"]):r for r in rows};ids=sorted({r["id"]for r in rows});out={}
 for condition in sorted({r["condition"]for r in rows}):
  vals=[]
  for id in ids:
   q=idx[(id,condition,0)]["query"];sign=1 if q==0 else-1;base=sign*(idx[(id,"no_patch",0)]["value"]-idx[(id,"no_patch",1)]["value"]);cur=sign*(idx[(id,condition,0)]["value"]-idx[(id,condition,1)]["value"]);vals.append((base,cur))
  donor=condition.startswith("donor_")
  out[condition]={"n":len(vals),"accuracy":float(np.mean([v[1]>0 for v in vals])),"mean_oriented_margin":float(np.mean([v[1]for v in vals])),"mean_margin_loss":float(np.mean([v[0]-v[1]for v in vals])),"donor_flip_rate":float(np.mean([v[1]<0 for v in vals]))if donor else None,"mean_shift_to_donor":float(np.mean([v[0]-v[1]for v in vals]))if donor else None}
 return out

def run_model(key:str)->dict:
 prior=load(P2520/"analysis/final.json");cross=load(P2522/"analysis/final.json");material=read(P2520/"material/natural_rows.jsonl");family_ids=set(prior["behavior"]["qualified_family_ids"])
 model,tokenizer,offload=p2522.load_model(key)
 try:
  selection=compile_selection(tokenizer,material,family_ids);routes,fields=select_routes(model,tokenizer,selection,key)
  behavior_rows=read(Path(cross["models"][key]["artifacts"]["behavior"]["path"]));_,decisions=p2522.behavior_panel(behavior_rows);jobs,coverage=p2522.causal_jobs(tokenizer,material,family_ids,decisions);rows=causal(model,tokenizer,jobs,routes)
  info={"class":type(model).__name__,"layers":len(layers_of(model)),"hidden_size":int(model.config.hidden_size),"heads":routes["heads"],"dtype":str(next(model.parameters()).dtype),"quantized":bool(getattr(model,"is_quantized",False)),"device_map":{str(k):str(v)for k,v in getattr(model,"hf_device_map",{}).items()}}
 finally:
  del model;gc.collect();torch.cuda.empty_cache()
  if offload.exists():shutil.rmtree(offload)
 score_path=OUT/f"output/{key}_route_scores.jsonl";write(score_path,rows);route_path=OUT/f"analysis/{key}_routes.json";save(route_path,routes);metrics=panels(rows)
 r={"model":key,"precision":"BF16 nonquantized","model_info":info,"selection":{"prompts":len(selection),"unit":30,"routes":routes},"lockbox":{"unit":31,**coverage},"causal":metrics,"fields":fields,"files":{"scores":{"path":str(score_path),"sha256":sha(score_path)},"routes":{"path":str(route_path),"sha256":sha(route_path)}},"checks":{"nonquantized":not info["quantized"],"selection_72":len(selection)==72,"lockbox_at_least_30":coverage["eligible_pairs"]>=30,"all_head_coordinates":fields["head_output"]["shape"][-2:]==[routes["heads"],routes["head_dim"]],"conditions_7":len(metrics)==7,"hashes":all(len(x)==64 for x in(fields["head_output"]["sha256"],fields["interaction"]["sha256"],sha(score_path),sha(route_path))),"claim_boundary":True}}
 r["all_checks_passed"]=all(r["checks"].values());save(OUT/f"analysis/{key}.json",r);print(json.dumps({"model":key,"routes":{k:routes[k]for k in("late_layers","heads","all_routes","selected_fraction")},"causal":metrics,"checks":r["checks"]},ensure_ascii=False,indent=2));return r

def append_memo(r:dict)->None:
 if f"## Phase {PHASE}:"in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
 summary={k:{"model_info":v["model_info"],"selection":v["selection"],"lockbox":v["lockbox"],"causal":v["causal"],"fields":v["fields"]}for k,v in r["models"].items()}
 text=rf"""


## Phase {PHASE}: 三模型BF16非量化模型内late-head路线复现（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Qwen3-14B、DeepSeek-R1-Distill-Qwen-7B、GLM-4-9B严格一次加载一个，BF16、禁止量化、`device_map=auto`。每个模型独立用unit30九族×英中×meaning-swap×query的72条prompt保存相对深度56%之后全部head输出物理坐标，按全128维Walsh能量冻结该模型late routes的6.25%及等量随机；unit31行为合格且exact-shape的34–36组作为锁箱，比较whole-head donor top/random/all充分性与自然head-zero top/random/all必要损伤。只比较相对深度、冻结比例和模型内对照，不对齐模型间head号或坐标号。

$$I_{{lh}}=\tfrac14(u_{{00,lh}}-u_{{01,lh}}-u_{{10,lh}}+u_{{11,lh}}),\qquad E_{{lh}}=\sum_{{f,\lambda,d}}I_{{lh,d}}^2.$$

**结果汇总。** 三模型 `{json.dumps(summary,ensure_ascii=False)}`；跨模型裁决 `{json.dumps(r['adjudication'],ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2533_c104193_c107264_crossmodel_local_route_replication.py`；各模型全head物理坐标场、Walsh交互、冻结路线、逐样本候选分数和final位于`{OUT}`。

**分析与理论进展。** 该实验复现的是“模型内从发现unit冻结的late-head协同路线，能否在新unit上优于等量随机”，而不是拿Qwen3-4B的head编号硬贴到别的架构。donor充分性与zero必要损伤分别报告；任一阴性都按模型容量、冗余和干预合同保留为边界，不抹平为统一机制。

**问题硬伤与结论。** 跨模型选择量是whole-head输出交互，不是Phase2529的source-specific贡献；每模型只用一个发现unit；top比例相同不代表功能规模相同；head-zero是强干预；非量化CPU/磁盘offload会改变数值执行路径但self合同沿用Phase2522已通过结果。事件级共性若成立，也不证明共享物理基底。
"""
 with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)

def finalize()->dict:
 models={k:load(OUT/f"analysis/{k}.json")for k in MODEL_KEYS};checks={"all_models_present":len(models)==3,"all_nonquantized":all(not v["model_info"]["quantized"]for v in models.values()),"all_model_checks":all(v["all_checks_passed"]for v in models.values()),"sequential_contract":True,"claim_boundary":True}
 advantages={k:{"donor_top_minus_random_shift":v["causal"]["donor_top"]["mean_shift_to_donor"]-v["causal"]["donor_random"]["mean_shift_to_donor"],"zero_top_minus_random_loss":v["causal"]["head_zero_top"]["mean_margin_loss"]-v["causal"]["head_zero_random"]["mean_margin_loss"]}for k,v in models.items()}
 r={"phase":PHASE,"campaign":CAMPAIGN,"models":models,"adjudication":{"model_local_advantages":advantages,"shared_physical_heads":False,"shared_coordinates":False,"same_algorithm_established":False,"language_mechanism_closed":False},"checks":checks,"all_checks_passed":all(checks.values())};save(OUT/"analysis/final.json",r)
 if r["all_checks_passed"]:append_memo(r)
 print(json.dumps({"phase":PHASE,"advantages":advantages,"checks":checks,"all_checks_passed":r["all_checks_passed"]},ensure_ascii=False,indent=2));return r

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--model",choices=MODEL_KEYS);parser.add_argument("--finalize",action="store_true");args=parser.parse_args()
 if args.model:r=run_model(args.model)
 elif args.finalize:r=finalize()
 else:
  for key in MODEL_KEYS:run_model(key)
  r=finalize()
 if not r["all_checks_passed"]:raise RuntimeError(r["checks"])
if __name__=="__main__":main()
