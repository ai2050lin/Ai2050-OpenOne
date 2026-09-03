#!/usr/bin/env python3
"""Natural-language counterfactual relation families with candidate and open answer modes."""
from __future__ import annotations

import gc, hashlib, json, re, sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result"
OUT=RESULT/"phase2520_c85025_c86176_natural_language_counterfactual_fullfield";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE,CAMPAIGN,DIM=2520,"C85025-C86176",2560;UNITS=(30,31);EVENTS=("facts_end","query_property","answer_boundary")
sys.path.insert(0,str(TESTS));import model_utils  # noqa:E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa:E402

FAMILIES=(
 ("taxonomy",("fruit","tool"),("水果","工具")),("part_whole",("engine","house"),("发动机","房屋")),
 ("role",("doctor","teacher"),("医生","教师")),("preference",("tea","coffee"),("茶","咖啡")),
 ("membership",("club","team"),("社团","团队")),("translation",("river","lake"),("河流","湖泊")),
 ("temporal",("morning","evening"),("早晨","夜晚")),("spatial",("east","west"),("东方","西方")),
 ("causal",("fire","rain"),("火焰","降雨")),("negation",("allowed","blocked"),("允许","禁止")),
 ("multihop",("bird","fish"),("鸟类","鱼类")),("long_reorder",("front","back"),("前面","后面")),)
EN_NAMES={30:("Arin","Bela","Ciro","Dena"),31:("Evan","Faye","Gino","Hana")};ZH_NAMES={30:("安宁","白川","苍禾","岱青"),31:("恩泽","枫林","观海","寒松")}

def save(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def write_jsonl(p:Path,rows:list[dict])->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
def read_jsonl(p:Path)->list[dict]:return[json.loads(x)for x in p.read_text(encoding="utf-8-sig").splitlines()if x.strip()]
def digest(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb")as f:
  for b in iter(lambda:f.read(16*1024*1024),b""):h.update(b)
 return h.hexdigest()
def spans(tok,prompt,text):
 out=[];start=0
 while True:
  i=prompt.find(text,start)
  if i<0:break
  out.append([len(tok.encode(prompt[:i],add_special_tokens=False)),len(tok.encode(prompt[:i+len(text)],add_special_tokens=False))]);start=i+len(text)
 return out

def fact_sentence(family:str,e:str,p:str,lang:str)->str:
 if lang=="en":
  templates={"taxonomy":"[ {e} ] is a [ {p} ]","part_whole":"[ {e} ] is contained in the [ {p} ]","role":"[ {e} ] works as a [ {p} ]",
   "preference":"[ {e} ] prefers [ {p} ]","membership":"[ {e} ] belongs to the [ {p} ]","translation":"[ {e} ] translates to [ {p} ]",
   "temporal":"[ {e} ] occurs in the [ {p} ]","spatial":"[ {e} ] is placed to the [ {p} ]","causal":"[ {e} ] causes [ {p} ]",
   "negation":"[ {e} ] is marked [ {p} ]","long_reorder":"[ {e} ] must be placed at the [ {p} ]"}
 else:
  templates={"taxonomy":"[ {e} ]属于[ {p} ]","part_whole":"[ {e} ]包含在[ {p} ]中","role":"[ {e} ]的职业是[ {p} ]",
   "preference":"[ {e} ]偏好[ {p} ]","membership":"[ {e} ]属于[ {p} ]","translation":"[ {e} ]翻译为[ {p} ]",
   "temporal":"[ {e} ]发生在[ {p} ]","spatial":"[ {e} ]位于[ {p} ]","causal":"[ {e} ]导致[ {p} ]",
   "negation":"[ {e} ]被标记为[ {p} ]","long_reorder":"[ {e} ]重排后应位于[ {p} ]"}
 return templates[family].format(e=e,p=p)

def compile_rows(tok)->list[dict]:
 rows=[];case=85025
 for unit in UNITS:
  for fi,(family,enprops,zhprops) in enumerate(FAMILIES):
   for lang in("en","zh"):
    props=enprops if lang=="en" else zhprops;names=EN_NAMES[unit]if lang=="en"else ZH_NAMES[unit];shift=(unit+fi)%4;names=names[shift:]+names[:shift];a,b,c,d=names
    assert len(tok.encode(props[0],add_special_tokens=False))==len(tok.encode(props[1],add_special_tokens=False))
    for surface in(0,1):
     for output_mode in("candidate","open"):
      for swap in(0,1):
       mapping=(0,1)if swap==0 else(1,0)
       if family=="multihop":
        facts=[f"[ {a} ] links to [ {c} ]. [ {c} ] is a [ {props[mapping[0]]} ].",f"[ {b} ] links to [ {d} ]. [ {d} ] is a [ {props[mapping[1]]} ]."]if lang=="en"else[f"[ {a} ]连接到[ {c} ]。[ {c} ]属于[ {props[mapping[0]]} ]。",f"[ {b} ]连接到[ {d} ]。[ {d} ]属于[ {props[mapping[1]]} ]。"]
       else:facts=[fact_sentence(family,a,props[mapping[0]],lang)+("."if lang=="en"else"。"),fact_sentence(family,b,props[mapping[1]],lang)+("."if lang=="en"else"。")]
       if surface:facts=facts[::-1]
       prefix=("Natural facts: "+" ".join(facts)+" End facts. ")if lang=="en"else("自然事实："+"".join(facts)+"事实结束。")
       for query in(0,1):
        prop=props[query];target=a if mapping[0]==query else b;candidates=[a,b]if(fi+unit+surface)%2==0 else[b,a]
        if lang=="en":
         qtext=(f"Question: which entity has property [ {prop} ] according to the facts?" if surface==0 else f"Using the facts, identify the entity associated with [ {prop} ].")
         choices=(f" Candidates: [ {candidates[0]} ] [ {candidates[1]} ]." if output_mode=="candidate" else "")
         prompt=prefix+qtext+choices+" Return only the entity name. Answer:"
        else:
         qtext=(f"问题：根据事实，哪个实体具有属性[ {prop} ]？"if surface==0 else f"请依据事实找出与[ {prop} ]对应的实体。")
         choices=(f"候选：[ {candidates[0]} ] [ {candidates[1]} ]。"if output_mode=="candidate"else"")
         prompt=prefix+qtext+choices+"只返回实体名称。答案："
        ids=[int(v)for v in tok.encode(prompt,add_special_tokens=False)];anchors={"facts_end":"End facts"if lang=="en"else"事实结束","query_property":prop}
        sp={k:spans(tok,prompt,v)for k,v in anchors.items()}
        rows.append({"case_id":f"c{case:05d}-u{unit}-f{fi}-{lang}-s{surface}-{output_mode}-m{swap}-q{query}","unit":unit,"family_id":fi,"family":family,"language":lang,"surface":surface,"output_mode":output_mode,"meaning_swap":swap,"query_property":query,"property":prop,"properties":list(props),"entities":[a,b],"target":target,"candidates":candidates,"prompt":prompt,"prompt_ids":ids,"spans":sp,"answer_boundary_token":len(ids)-1});case+=1
 # Equalize complete matrix shape across the two queried properties. Chinese BPE
 # can tokenize the same one-token property differently inside a sentence, so a
 # neutral full stop is inserted before the answer marker on the shorter side.
 groups={}
 for row in rows:groups.setdefault((row["unit"],row["family_id"],row["language"],row["surface"],row["output_mode"]),[]).append(row)
 for group in groups.values():
  example={r["query_property"]:r for r in group if r["meaning_swap"]==0}
  chosen=None
  for k0 in range(7):
   for k1 in range(7):
    lengths=[]
    for q,k in ((0,k0),(1,k1)):
     marker=" Answer:"if example[q]["language"]=="en"else"答案：";note=("."if example[q]["language"]=="en"else"。")*k
     candidate=example[q]["prompt"].replace(marker,note+marker)
     lengths.append(len(tok.encode(candidate,add_special_tokens=False)))
    if lengths[0]==lengths[1]:chosen=(k0,k1);break
   if chosen is not None:break
  assert chosen is not None
  for row in group:
   k=chosen[row["query_property"]];marker=" Answer:"if row["language"]=="en"else"答案：";note=("."if row["language"]=="en"else"。")*k
   row["prompt"]=row["prompt"].replace(marker,note+marker);row["prompt_ids"]=[int(v)for v in tok.encode(row["prompt"],add_special_tokens=False)]
   row["length_balance_punctuation"]=k;row["spans"]={"facts_end":spans(tok,row["prompt"],"End facts"if row["language"]=="en"else"事实结束"),"query_property":spans(tok,row["prompt"],row["property"])};row["answer_boundary_token"]=len(row["prompt_ids"])-1
 return rows

def normalize(x:str)->str:return re.sub(r"[^0-9a-z\u4e00-\u9fff]+","",x.casefold())
def behavior(model,tok,rows:list[dict])->list[dict]:
 tok.padding_side="left";device=model.get_input_embeddings().weight.device;out=[]
 for start in range(0,len(rows),8):
  batch=rows[start:start+8];enc=tok([r["prompt"]for r in batch],return_tensors="pt",padding=True,add_special_tokens=False);enc={k:v.to(device)for k,v in enc.items()}
  with torch.inference_mode():seq=model.generate(**enc,max_new_tokens=8,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
  width=enc["input_ids"].shape[1]
  for r,s in zip(batch,seq):
   text=tok.decode(s[width:].cpu().tolist(),skip_special_tokens=True);hits=[e for e in r["entities"]if normalize(e)in normalize(text)];parsed=hits[0]if len(set(hits))==1 else None
   out.append({"case_id":r["case_id"],"unit":r["unit"],"family_id":r["family_id"],"family":r["family"],"language":r["language"],"surface":r["surface"],"output_mode":r["output_mode"],"meaning_swap":r["meaning_swap"],"query_property":r["query_property"],"target":r["target"],"generated_text":text,"parsed":parsed,"correct":parsed==r["target"]})
  if(start+len(batch))%128==0:print(f"[phase2520 behavior] {start+len(batch)}/{len(rows)}",flush=True)
 return out

def audit(rows:list[dict])->dict:
 idx={(r["unit"],r["family_id"],r["language"],r["surface"],r["output_mode"],r["meaning_swap"],r["query_property"]):r for r in rows};ql=[];prefix=[];bag=[];flip=[]
 for unit in UNITS:
  for f in range(12):
   for lang in("en","zh"):
    for s in(0,1):
     for mode in("candidate","open"):
      for m in(0,1):
       a,b=(idx[(unit,f,lang,s,mode,m,q)]for q in(0,1));ql.append(len(a["prompt_ids"])==len(b["prompt_ids"]));end=a["spans"]["facts_end"][0][1];prefix.append(a["prompt_ids"][:end]==b["prompt_ids"][:end])
      for q in(0,1):
       a,b=(idx[(unit,f,lang,s,mode,m,q)]for m in(0,1));bag.append(Counter(a["prompt_ids"])==Counter(b["prompt_ids"]));flip.append(a["target"]!=b["target"])
 return{"rows":len(rows),"query_length_equal_rate":float(np.mean(ql)),"query_prefix_equal_rate":float(np.mean(prefix)),"swap_token_multiset_equal_rate":float(np.mean(bag)),"answer_flip_rate":float(np.mean(flip))}

def summarize(rows:list[dict],gen:list[dict])->dict:
 by={r["case_id"]:r for r in gen};detail={};qualified=[]
 for fi,(family,*_)in enumerate(FAMILIES):
  detail[str(fi)]={};gates=[]
  for unit in UNITS:
   vals=[by[r["case_id"]]for r in rows if r["unit"]==unit and r["family_id"]==fi];lang={l:float(np.mean([v["correct"]for v in vals if v["language"]==l]))for l in("en","zh")};modes={m:float(np.mean([v["correct"]for v in vals if v["output_mode"]==m]))for m in("candidate","open")};swaps={str(m):float(np.mean([v["correct"]for v in vals if v["meaning_swap"]==m]))for m in(0,1)}
   p={"rows":len(vals),"accuracy":float(np.mean([v["correct"]for v in vals])),"language":lang,"output_mode":modes,"swap":swaps};p["gate"]=p["accuracy"]>=.70 and min(lang.values())>=.5 and min(modes.values())>=.5 and min(swaps.values())>=.5;detail[str(fi)][str(unit)]=p;gates.append(p["gate"])
  if all(gates):qualified.append(fi)
 return{"aggregate":{str(u):float(np.mean([v["correct"]for v in gen if v["unit"]==u]))for u in UNITS},"qualified_family_ids":qualified,"qualified_families":[FAMILIES[i][0]for i in qualified],"detail":detail}

def capture(model,rows:list[dict],qualified:set[int])->dict:
 selected=[r for r in rows if r["family_id"]in qualified]
 mods=field_utils.modules(model);raw=OUT/"raw";raw.mkdir(parents=True,exist_ok=True)
 path=raw/"natural_threeevent_allqpoint.float16.npy"
 arr=np.lib.format.open_memmap(path,mode="w+",dtype=np.float16,shape=(len(selected),3,len(mods),DIM));caps={};handles=[]
 for q,m in enumerate(mods):
  def hook(_m,_i,o,q=q):caps[q]=(o[0]if isinstance(o,tuple)else o).detach()
  handles.append(m.register_forward_hook(hook))
 device=model.get_input_embeddings().weight.device;index=[]
 try:
  with torch.inference_mode():
   for i,r in enumerate(selected):
    ids=torch.tensor([r["prompt_ids"]],device=device);caps.clear();model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    pos=[r["spans"]["facts_end"][0][1]-1,r["spans"]["query_property"][-1][1]-1,r["answer_boundary_token"]]
    for q in range(len(mods)):arr[i,:,q]=caps[q][0,pos].float().cpu().numpy().astype(np.float16)
    index.append({**{k:r[k]for k in("case_id","unit","family_id","family","language","surface","output_mode","meaning_swap","query_property","target","prompt_ids")},"model_row":i,"event_positions":pos})
    if(i+1)%128==0:arr.flush();print(f"[phase2520 field] {i+1}/{len(selected)}",flush=True)
 finally:
  for h in handles:h.remove()
  arr.flush();del arr
 ip=OUT/"index/field_rows.jsonl";write_jsonl(ip,index)
 return{"field":str(path),"shape":[len(selected),3,len(mods),DIM],"index":str(ip),"events":list(EVENTS),"sha256":digest(path)}

def append_memo(r:dict)->None:
 if f"## Phase {PHASE}:"in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M");text=rf"""


## Phase {PHASE}: 十二自然语言模式族的反事实行为门、开放回答与全坐标场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 从nonce二候选扩展到taxonomy、part-whole、role、preference、membership、translation、temporal、spatial、causal、negation、multihop和long-reorder十二族。每族使用两个实体与两个等token长度属性；只交换属性在事实中的绑定，问题分别查询两属性，正确实体随swap翻转。unit30/31、两语言、两事实顺序/问法、candidate与无候选open两输出模式、两swap、两query全交叉，共768条。多跳族需经中间实体，long-reorder询问重排后的前/后实体。双unit行为门后保存facts-end、query-property、answer-boundary的38×2560全场。

$$N=2\times12\times2^5=768,\qquad I=\tfrac14(H_{{00}}-H_{{01}}-H_{{10}}+H_{{11}}).$$

**结果汇总。** 设计 `{json.dumps(r['design_audit'],ensure_ascii=False)}`；行为 `{json.dumps(r['behavior'],ensure_ascii=False)}`；采集 `{json.dumps(r['collection'],ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2520_c85025_c86176_natural_language_counterfactual_fullfield.py`；768条自然材料、生成、合格族三事件全场、索引、哈希与final位于`{OUT}`。

**分析与理论进展。** 该合同检验自然谓词和开放实体输出中是否仍有“事实绑定×查询属性”的行为必要组合。通过只说明模型根据事实回答；三事件全场用于比较合成nonce结果是否只是模板伪影。不同family物理属性词不同，不能直接把跨family余弦叫共同语义轴。

**问题硬伤与结论。** open仍要求短实体名，不是真正自由写作；事实是人工微世界，不调用预训练世界知识；long-reorder只输出一个位置实体，不等于整段长句重排；未通过行为门的族不能用于内部机制负裁决。跨模型只能先复核行为和事件层级，不能比较物理坐标编号。
"""
 with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)

def main()->None:
 model,tok,_=model_utils.load_model("qwen3",dtype=torch.bfloat16,use_8bit=False)
 try:
  rows=compile_rows(tok);write_jsonl(OUT/"material/natural_rows.jsonl",rows);gen=behavior(model,tok,rows);write_jsonl(OUT/"behavior/generation.jsonl",gen);beh=summarize(rows,gen);collection=capture(model,rows,set(beh["qualified_family_ids"]))
 finally:model_utils.release_model(model);gc.collect()
 da=audit(rows);checks={"rows_768":len(rows)==768,"query_equal":da["query_length_equal_rate"]==1,"prefix_equal":da["query_prefix_equal_rate"]==1,"swap_multiset":da["swap_token_multiset_equal_rate"]==1,"answer_flip":da["answer_flip_rate"]==1,"at_least_six_families":len(beh["qualified_family_ids"])>=6,"field_shape":collection["shape"]==[len(beh["qualified_family_ids"])*64,3,38,DIM],"hash":len(collection["sha256"])==64,"claim_boundary":True}
 r={"phase":PHASE,"campaign":CAMPAIGN,"model":"Qwen3-4B nonquantized BF16 CUDA","design_audit":da,"behavior":beh,"collection":collection,"adjudication":{"natural_counterfactual_behavior_established_for_qualified_families":True,"open_generation_is_short_constrained":True,"language_encoding_mechanism_closed":False},"checks":checks,"all_checks_passed":all(checks.values())};save(OUT/"analysis/final.json",r)
 if r["all_checks_passed"]:append_memo(r)
 print(json.dumps({"phase":PHASE,"design":da,"behavior":beh,"collection":collection,"checks":checks,"all_checks_passed":r["all_checks_passed"]},ensure_ascii=False,indent=2))
 if not r["all_checks_passed"]:raise RuntimeError(checks)
if __name__=="__main__":main()
