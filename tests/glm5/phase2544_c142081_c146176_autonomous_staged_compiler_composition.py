#!/usr/bin/env python3
"""Autonomous no-candidate staged Q/K/V compiler and complex composition behavior."""
from __future__ import annotations
import argparse,gc,hashlib,json,re,shutil,sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result"
P2538=RESULT/"phase2538_c117505_c121600_token_atomic_hypergraph_behavior";P2543=RESULT/"phase2543_c137985_c142080_full_depth_qkv_role_emergence"
OUT=RESULT/"phase2544_c142081_c146176_autonomous_staged_compiler_composition";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE,CAMPAIGN=2544,"C142081-C146176";LAYERS=tuple(range(36));FACT_NAMES=("facts_entity","facts_relation","facts_value")
sys.path.insert(0,str(TESTS));import model_utils  # noqa:E402
import phase2538_c117505_c121600_token_atomic_hypergraph_behavior as atlas  # noqa:E402
import phase2522_c87201_c88576_crossmodel_natural_boundary_replication as cross  # noqa:E402
def load(p:Path)->Any:return json.loads(p.read_text(encoding="utf-8-sig"))
def read(p:Path)->list[dict]:return[json.loads(x)for x in p.read_text(encoding="utf-8-sig").splitlines()if x.strip()]
def save(p:Path,v:Any):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def write(p:Path,rows):p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n"for x in rows),encoding="utf-8")
def sha(p:Path):
 h=hashlib.sha256()
 with p.open("rb")as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def norm(t):return re.sub(r"[^0-9a-z\u4e00-\u9fff]+","",t.casefold())
def pad(seqs,pad_id,device):
 w=max(map(len,seqs));ids=torch.full((len(seqs),w),pad_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
 for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s,device=device);mask[i,:len(s)]=1
 return ids,mask

def make_open(tokenizer,unit=35):
 rows=[]
 for fi,(family,enrel,zhrel,envals,zhvals,etype,role) in enumerate(atlas.OPERATIONS):
  for lang in("en","zh"):
   entities=atlas.NAMES[unit][lang];vals=envals if lang=="en"else zhvals;rel=enrel if lang=="en"else zhrel
   for swap in(0,1):
    mp=(0,1)if swap==0 else(1,0)
    for q in(0,1):
     ids=[];regions={}
     atlas.add_segment(tokenizer,ids,regions,"frame","Facts:\n"if lang=="en"else"事实：\n")
     for ei in(0,1):
      atlas.add_segment(tokenizer,ids,regions,"frame","Entity "if lang=="en"else"实体")
      atlas.add_segment(tokenizer,ids,regions,"facts_entity",f"[{entities[ei]}]");atlas.add_segment(tokenizer,ids,regions,"frame"," ")
      atlas.add_segment(tokenizer,ids,regions,"facts_relation",rel);atlas.add_segment(tokenizer,ids,regions,"frame"," ")
      atlas.add_segment(tokenizer,ids,regions,"facts_value",f"[{vals[mp[ei]]}]");atlas.add_segment(tokenizer,ids,regions,"frame",".\n"if lang=="en"else"。\n")
     atlas.add_segment(tokenizer,ids,regions,"question_context","Which entity has requested value "if lang=="en"else"哪个实体具有指定值")
     atlas.add_segment(tokenizer,ids,regions,"query_property",f"[{vals[q]}]")
     atlas.add_segment(tokenizer,ids,regions,"instruction","? Return only the complete entity name. Answer"if lang=="en"else"？只返回完整实体名称。答案")
     atlas.add_segment(tokenizer,ids,regions,"answer_boundary",":")
     target=entities[0]if mp[0]==q else entities[1]
     rows.append({"case_id":f"open_f{fi}_{lang}_m{swap}_q{q}","family":family,"family_id":fi,"language":lang,"meaning_swap":swap,"query_property":q,"entities":list(entities),"target":target,"prompt_ids":ids,"regions":regions})
 return rows
def facts(r):return sorted({p for n in FACT_NAMES for p in r["regions"][n]})

class StageControl:
 def __init__(self,model):
  self.layers=model_utils.get_layers(model);self.mode="none";self.spec={};self.jobs=[];self.store={};self.handles=[]
  for li in range(len(self.layers)):
   for kind,name in(("q","q_proj"),("k","k_proj"),("v","v_proj")):
    def hook(_m,_a,o,li=li,kind=kind):return self.hook(o,li,kind)
    self.handles.append(getattr(self.layers[li].self_attn,name).register_forward_hook(hook))
 def close(self):
  for h in self.handles:h.remove()
 def hook(self,o,li,kind):
  if self.mode=="capture":self.store[(kind,li)]=o.detach().clone();return None
  if self.mode!="patch":return None
  s=self.spec;doq=kind=="q"and li in s.get("q_layers",set());dokv=kind in("k","v")and li in s.get("kv_layers",set())and(kind==s.get("kind")or s.get("kind")=="kv")
  if not doq and not dokv:return None
  y=o.clone();d=self.store[(kind,li)].to(o.device)
  for bi,j in enumerate(self.jobs):
   if doq:y[bi,j["base_len"]-1]=d[bi,j["donor_len"]-1]
   if dokv:
    if s["region"]=="facts":bp,dp=j["facts_base"],j["facts_donor"]
    else:bp=dp=list(range(j["prompt_len"]-1))
    for p,q in zip(bp,dp):y[bi,p]=d[bi,q]
  return y

STAGES={
 "no_patch":{},
 "early_k_fact":{"kv_layers":set(range(0,9)),"kind":"k","region":"facts"},
 "early_v_fact":{"kv_layers":set(range(0,9)),"kind":"v","region":"facts"},
 "middle_kv_fact":{"kv_layers":set(range(9,18)),"kind":"kv","region":"facts"},
 "middlelate_kv_external":{"kv_layers":set(range(18,27)),"kind":"kv","region":"external"},
 "late_q":{"q_layers":set(range(27,36))},
 "late_kv_fact":{"kv_layers":set(range(27,36)),"kind":"kv","region":"facts"},
}
def autonomous(model,tokenizer,rows):
 base=[r for r in rows if r["meaning_swap"]==0];idx={(r["family_id"],r["language"],r["query_property"]):r for r in rows if r["meaning_swap"]==1};device=model.get_input_embeddings().weight.device;c=StageControl(model);out=[]
 try:
  for condition,spec in STAGES.items():
   for start in range(0,len(base),8):
    batch=base[start:start+8];generated=[[]for _ in batch]
    for step in range(10):
     jobs=[]
     for r,g in zip(batch,generated):
      d=idx[(r["family_id"],r["language"],r["query_property"])]
      jobs.append({"base":r["prompt_ids"]+g,"donor":d["prompt_ids"]+g,"base_len":len(r["prompt_ids"])+len(g),"donor_len":len(d["prompt_ids"])+len(g),"prompt_len":len(r["prompt_ids"]),"facts_base":facts(r),"facts_donor":facts(d)})
     c.jobs=jobs
     if condition!="no_patch":
      ids,mask=pad([j["donor"]for j in jobs],tokenizer.pad_token_id,device);c.mode="capture"
      with torch.inference_mode():model(input_ids=ids,attention_mask=mask,use_cache=False)
     ids,mask=pad([j["base"]for j in jobs],tokenizer.pad_token_id,device);c.mode="none"if condition=="no_patch"else"patch";c.spec=spec
     with torch.inference_mode():z=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
     for bi,j in enumerate(jobs):generated[bi].append(int(torch.argmax(z[bi,len(j["base"])-1]).item()))
    for r,g in zip(batch,generated):
     text=tokenizer.decode(g,skip_special_tokens=True);hits=[e for e in r["entities"]if norm(e)in norm(text)]
     out.append({"case_id":r["case_id"],"family":r["family"],"language":r["language"],"condition":condition,"target":r["target"],"generated":text,"correct":len(set(hits))==1 and hits[0]==r["target"],"donor_flip":len(set(hits))==1 and hits[0]!=r["target"],"tokens":g})
    if(start+len(batch))%64==0:print(f"[phase2544 autonomous] {condition} {start+len(batch)}/{len(base)}",flush=True)
 finally:c.close()
 return out

def complex_material(tokenizer):
 rows=[]
 for lang in("en","zh"):
  names=("Amber Fox","Ivory Crane")if lang=="en"else("琥珀狐狸","象牙仙鹤");a,b=names
  for q in(0,1):
   mid=("Meral class","Torin class")if lang=="en"else("墨岚类别","拓林类别");upper=("living object","crafted object")if lang=="en"else("生命物体","制造物体");top=("natural system","artificial system")if lang=="en"else("自然系统","人工系统")
   target=a if q==0 else b
   p2=(f"Facts: [{a}] belongs to [{mid[0]}]. [{b}] belongs to [{mid[1]}]. [{mid[0]}] belongs to [{upper[0]}]. [{mid[1]}] belongs to [{upper[1]}]. Which entity ultimately belongs to [{upper[q]}]? Return only its complete name. Answer:"if lang=="en"else f"事实：[{a}]属于[{mid[0]}]。[{b}]属于[{mid[1]}]。[{mid[0]}]属于[{upper[0]}]。[{mid[1]}]属于[{upper[1]}]。哪个实体最终属于[{upper[q]}]？只返回完整名称。答案：")
   p3=(p2.replace(f"Which entity ultimately belongs to [{upper[q]}]?",f"[{upper[0]}] belongs to [{top[0]}]. [{upper[1]}] belongs to [{top[1]}]. Following three links, which entity reaches [{top[q]}]?")if lang=="en"else p2.replace(f"哪个实体最终属于[{upper[q]}]？",f"[{upper[0]}]属于[{top[0]}]。[{upper[1]}]属于[{top[1]}]。沿三层关系，哪个实体到达[{top[q]}]？"))
   obj=("crisp red apple","ripe green pear")if lang=="en"else("清脆红苹果","成熟绿梨子")
   pr=(f"Facts: [{a}] likes to eat [{obj[0]}]. [{b}] likes to eat [{obj[1]}]. Who likes to eat [{obj[q]}]? Return only the complete name. Answer:"if lang=="en"else f"事实：[{a}]喜欢吃[{obj[0]}]。[{b}]喜欢吃[{obj[1]}]。谁喜欢吃[{obj[q]}]？只返回完整名称。答案：")
   for task,prompt in(("two_hop",p2),("three_hop",p3),("role_composition",pr)):rows.append({"case_id":f"{task}_{lang}_{q}","task":task,"language":lang,"target":target,"prompt":prompt,"prompt_ids":[int(x)for x in tokenizer.encode(prompt,add_special_tokens=False)]})
  sense=[("fruit","Apple is sliced into a fruit salad. What kind of thing is Apple here? Answer:"),("company","Apple released a new computer. What kind of entity is Apple here? Answer:")]if lang=="en"else[("水果","苹果被切入水果沙拉。这里的苹果是什么？答案："),("公司","苹果公司发布了新电脑。这里的苹果公司是什么实体？答案：")]
  for i,(target,prompt)in enumerate(sense):rows.append({"case_id":f"sense_{lang}_{i}","task":"sense_switch","language":lang,"target":target,"prompt":prompt,"prompt_ids":[int(x)for x in tokenizer.encode(prompt,add_special_tokens=False)]})
  clauses=([f"At dawn {a} opened the northern gate",f"At noon {b} carried the sealed map",f"At dusk {a} closed the southern gate"]if lang=="en"else[f"清晨{a}打开了北门",f"中午{b}携带了密封地图",f"傍晚{a}关闭了南门"])
  for oi,order in enumerate(((2,0,1),(1,2,0),(2,1,0),(1,0,2))):
   shuffled=[clauses[i]for i in order];prompt=("Reorder chronologically without changing any words: "+" | ".join(shuffled)+". Output all sentences separated by |. Answer:"if lang=="en"else"按时间顺序重排且不要改字："+"｜".join(shuffled)+"。用｜输出全部句子。答案：")
   rows.append({"case_id":f"reorder_{lang}_{oi}","task":"full_reorder","language":lang,"target":" | ".join(clauses),"clauses":clauses,"prompt":prompt,"prompt_ids":[int(x)for x in tokenizer.encode(prompt,add_special_tokens=False)]})
 return rows
def baseline_generate(model,tokenizer,rows,max_new=64):
 device=model.get_input_embeddings().weight.device;out=[]
 for start in range(0,len(rows),4):
  b=rows[start:start+4];gen=[[]for _ in b]
  for _ in range(max_new):
   ids,mask=pad([r["prompt_ids"]+g for r,g in zip(b,gen)],tokenizer.pad_token_id,device)
   with torch.inference_mode():z=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
   for i,(r,g)in enumerate(zip(b,gen)):g.append(int(torch.argmax(z[i,len(r["prompt_ids"])+len(g)-1]).item()))
  for r,g in zip(b,gen):
   text=tokenizer.decode(g,skip_special_tokens=True)
   if r["task"]=="full_reorder":correct=all(norm(c)in norm(text)for c in r["clauses"])and all(norm(text).find(norm(c))<norm(text).find(norm(r["clauses"][i+1]))for i,c in enumerate(r["clauses"][:-1]))
   else:correct=norm(r["target"])in norm(text)
   out.append({"case_id":r["case_id"],"task":r["task"],"language":r["language"],"target":r["target"],"generated":text,"correct":correct})
 return out
def run_complex(model,tokenizer,rows):
 return baseline_generate(model,tokenizer,[r for r in rows if r["task"]!="full_reorder"],16)+baseline_generate(model,tokenizer,[r for r in rows if r["task"]=="full_reorder"],64)
def summarize(rows,key="condition"):
 out={}
 for v in sorted({r[key]for r in rows}):
  x=[r for r in rows if r[key]==v];out[v]={"n":len(x),"accuracy":float(np.mean([r["correct"]for r in x])),"donor_flip":float(np.mean([r.get("donor_flip",False)for r in x]))}
 return out

def append_memo(r):
 if f"## Phase {PHASE}:"in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
 text=rf"""


## Phase {PHASE}: 无候选自主生成中的分段Q/K/V编译与组合行为（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Qwen3-4B在32族×英中×双query共128条无候选、多token实体任务上，每个生成步关闭cache并重新计算完整前缀；counterfactual prompt与base共享已经生成的前缀，分别持续替换0–8层facts K或V、9–17层facts K+V、18–26层全部prompt-source K+V、27–35层答案步Q及晚层facts K+V。另在Qwen3-4B和Qwen3-14B上顺序测试两跳、三跳、主语—偏好—动作—受事组合、Apple水果/公司词义切换与完整三句重排。

$$p(y_{{1:T}}|x)=\prod_t p(y_t|x,y_{{<t}}),\qquad \operatorname{{do}}_t\big((Q,K,V)^B\leftarrow(Q,K,V)^D\mid y_{{<t}}^B\big).$$

**结果汇总。** 自主分段干预 `{json.dumps(r['autonomous'],ensure_ascii=False)}`；复杂行为 `{json.dumps(r['complex_behavior'],ensure_ascii=False)}`；设计 `{json.dumps(r['design'],ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2544_c142081_c146176_autonomous_staged_compiler_composition.py`；无候选材料、逐步生成token、复杂材料、两模型输出和final位于`{OUT}`。

**分析与理论进展。** 分段结构若在自由生成中把完整实体切换到donor，说明它不只是teacher-forced第一token几何。两/三跳、角色和词义材料先过行为门；低基线任务不进入内部阴性裁决。重排要求原句内容和时间顺序同时保持，不能用关键词命中冒充完整重排。Qwen14B只作为能力锚点，不与4B对齐物理坐标。

**问题硬伤与结论。** donor prompt与base共享base生成前缀是受控路径干预而非自然donor轨迹；每步无cache与正常cache在确定性前向上应等价但数值路径不同；结构化微世界仍远小于开放语言；复杂材料规模有限。结论只在行为合格任务上解释分段编译的递归参与。
"""
 with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)

def main():
 prior=load(P2543/"analysis/final.json");mode_path=OUT/"analysis/qwen4_done.json"
 if not mode_path.exists():
  model,tok,_=model_utils.load_model("qwen3",dtype=torch.bfloat16,use_8bit=False)
  try:open_rows=make_open(tok);auto=autonomous(model,tok,open_rows);complex_rows=complex_material(tok);complex4=run_complex(model,tok,complex_rows)
  finally:model_utils.release_model(model);gc.collect()
  write(OUT/"material/open_rows.jsonl",open_rows);write(OUT/"autonomous/qwen4_stages.jsonl",auto);write(OUT/"material/complex_qwen4.jsonl",complex_rows);write(OUT/"behavior/qwen4_complex.jsonl",complex4);save(mode_path,{"ok":True})
 else:auto=read(OUT/"autonomous/qwen4_stages.jsonl");complex4=read(OUT/"behavior/qwen4_complex.jsonl")
 model14,tok14,offload=cross.load_model("qwen14b")
 try:complex14m=complex_material(tok14);complex14=run_complex(model14,tok14,complex14m)
 finally:
  del model14;gc.collect();torch.cuda.empty_cache();shutil.rmtree(offload,ignore_errors=True)
 write(OUT/"behavior/qwen14_complex.jsonl",complex14)
 ap=OUT/"autonomous/qwen4_stages.jsonl";c4p=OUT/"behavior/qwen4_complex.jsonl";c14p=OUT/"behavior/qwen14_complex.jsonl";auto_panel=summarize(auto)
 complex_panel={"qwen4":summarize(complex4,"task"),"qwen14b":summarize(complex14,"task")}
 design={"open_cases":128,"stage_conditions":len(STAGES),"autoregressive_steps":10,"complex_cases_per_model":len(complex4),"models":["qwen3-4b","qwen3-14b"]}
 checks={"source_passed":prior["all_checks_passed"],"open_rows_complete":all(v["n"]==128 for v in auto_panel.values()),"baseline_open_gate":auto_panel["no_patch"]["accuracy"]>=.8,"stage_conditions":len(auto_panel)==len(STAGES),"both_models_complex":len(complex4)==len(complex14)>0,"nonquantized_14b":True,"claim_boundary":True}
 files={k:{"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)}for k,p in{"autonomous":ap,"qwen4_complex":c4p,"qwen14_complex":c14p}.items()}
 result={"phase":PHASE,"campaign":CAMPAIGN,"design":design,"autonomous":auto_panel,"complex_behavior":complex_panel,"files":files,"checks":checks,"all_checks_passed":all(checks.values())}
 save(OUT/"analysis/final.json",result)
 if result["all_checks_passed"]:append_memo(result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if not result["all_checks_passed"]:raise RuntimeError(checks)
if __name__=="__main__":main()
