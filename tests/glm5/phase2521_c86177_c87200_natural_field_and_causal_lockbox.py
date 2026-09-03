#!/usr/bin/env python3
"""Analyze natural-family full fields and replicate q36 answer-boundary causality."""
from __future__ import annotations
import gc,hashlib,json,sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result";P2520=RESULT/"phase2520_c85025_c86176_natural_language_counterfactual_fullfield";P2519=RESULT/"phase2519_c84001_c85024_answer_boundary_causal_layer_emergence";OUT=RESULT/"phase2521_c86177_c87200_natural_field_and_causal_lockbox";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md";PHASE,CAMPAIGN,DIM=2521,"C86177-C87200",2560
sys.path.insert(0,str(TESTS));import model_utils  # noqa:E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as fu  # noqa:E402
import phase2502_c66177_c67200_semantic_selection_walsh_fullcoordinate_lockbox as walsh  # noqa:E402
def load(p:Path):return json.loads(p.read_text(encoding="utf-8-sig"))
def read(p:Path):return[json.loads(x)for x in p.read_text(encoding="utf-8-sig").splitlines()if x.strip()]
def save(p:Path,v:Any):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def write(p:Path,rows:list[dict]):p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n"for r in rows),encoding="utf-8")
def sha(p:Path):
 h=hashlib.sha256()
 with p.open("rb")as f:
  for b in iter(lambda:f.read(16*1024*1024),b""):h.update(b)
 return h.hexdigest()
def build(f:dict)->tuple[np.ndarray,list[dict]]:
 rows=read(Path(f["collection"]["index"]));field=np.load(f["collection"]["field"],mmap_mode="r");families=f["behavior"]["qualified_family_ids"];out=np.zeros((2,len(families),2,2,2,3,38,DIM),np.float32);meta=[];idx={(r["unit"],r["family_id"],r["language"],r["surface"],r["output_mode"],r["meaning_swap"],r["query_property"]):r for r in rows}
 for ui,u in enumerate((30,31)):
  for fi,fid in enumerate(families):
   for li,lang in enumerate(("en","zh")):
    for s in(0,1):
     for oi,mode in enumerate(("candidate","open")):
      for e in range(3):
       for qpoint in range(38):
        c={(m,q):np.asarray(field[idx[(u,fid,lang,s,mode,m,q)]["model_row"],e,qpoint],np.float32)for m in(0,1)for q in(0,1)};out[ui,fi,li,s,oi,e,qpoint]=(c[0,0]-c[0,1]-c[1,0]+c[1,1])/4
      meta.append({"unit":u,"unit_index":ui,"family_id":fid,"family":idx[(u,fid,lang,s,mode,0,0)]["family"],"family_index":fi,"language":lang,"language_index":li,"surface":s,"output_mode":mode,"output_index":oi})
 return out,meta
def identity(v1:np.ndarray,v2:np.ndarray)->dict:
 views={"a":v1,"b":v2};return walsh.identity_metric(views)
def field_analysis(a:np.ndarray,families:list[str])->dict:
 report={}
 for ui,u in enumerate((30,31)):
  report[str(u)]={}
  for e,event in enumerate(("facts_end","query_property","answer_boundary")):
   layers=[]
   for q in range(38):
    x=a[ui,...,e,q,:]
    # x axes are family, language, surface, output-mode, coordinate.
    # Average only nuisance axes; never accidentally index the coordinate axis.
    mode=identity(x[:,:,:,0,:].mean((1,2)),x[:,:,:,1,:].mean((1,2)))
    language=identity(x[:,0,:,:,:].mean((1,2)),x[:,1,:,:,:].mean((1,2)))
    layers.append({"qpoint":q,"rms":float(np.sqrt(np.mean(x*x))),"cross_output_mode":mode,"cross_language":language})
   report[str(u)][event]=layers
 select=max(report["30"]["answer_boundary"][1:],key=lambda p:min(p["cross_output_mode"]["identity_advantage_over_q95"],p["cross_language"]["identity_advantage_over_q95"]))["qpoint"]
 return{"selection":{"rule":"unit30 answer-boundary maximin cross-output-mode/cross-language identity advantage","qpoint":select},"by_unit":report,"lockbox_selected":report["31"]["answer_boundary"][select],"families":families}
def pad(seqs,pad_id,device):
 w=max(map(len,seqs));ids=torch.full((len(seqs),w),pad_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
 for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s,device=device);mask[i,:len(s)]=1
 return ids,mask
def scores(logits,jobs):
 out=[]
 for i,j in enumerate(jobs):
  v=[]
  for k,t in enumerate(j["cont"]):z=logits[i,j["plen"]-1+k].float();v.append(float(z[t]-torch.logsumexp(z,-1)))
  out.append(sum(v))
 return out
def causal(model,tok,rows:list[dict],fids:list[int])->list[dict]:
 idx={(r["unit"],r["family_id"],r["language"],r["surface"],r["output_mode"],r["meaning_swap"],r["query_property"]):r for r in rows};items=[]
 for fid in fids:
  for lang in("en","zh"):
   for s in(0,1):
    for mode in("candidate","open"):
     for q in(0,1):items.append({"id":f"f{fid}-{lang}-s{s}-{mode}-q{q}","query":q,"base":idx[(31,fid,lang,s,mode,0,q)],"donor":idx[(31,fid,lang,s,mode,1,q)]})
 mods=fu.modules(model);active={"q":None,"source":None};captured={};pos=[];handles=[]
 for q in(28,36):
  def hook(_m,_i,o,q=q):
   h=o[0]if isinstance(o,tuple)else o;captured[q]=h.detach().clone()
   if active["q"]!=q:return None
   c=h.clone();c[torch.arange(h.shape[0],device=h.device),torch.tensor(pos,device=h.device)]=active["source"].to(c.dtype);return(c,*o[1:])if isinstance(o,tuple)else c
  handles.append(mods[q].register_forward_hook(hook))
 jobs=[]
 for item in items:
  for ri,e in enumerate(item["base"]["entities"]):
   cont=[int(v)for v in tok.encode((" "if item["base"]["language"]=="en"else"")+e,add_special_tokens=False)];jobs.append({"id":item["id"],"query":item["query"],"ri":ri,"cont":cont,"plen":len(item["base"]["prompt_ids"]),"position":len(item["base"]["prompt_ids"])-1,"base":item["base"]["prompt_ids"]+cont,"donor":item["donor"]["prompt_ids"]+cont})
 out=[];device=model.get_input_embeddings().weight.device
 try:
  for start in range(0,len(jobs),8):
   b=jobs[start:start+8];pos[:]=[j["position"]for j in b];bs,ds=[j["base"]for j in b],[j["donor"]for j in b];assert[len(x)for x in bs]==[len(x)for x in ds];bi,mask=pad(bs,tok.pad_token_id,device);di,dm=pad(ds,tok.pad_token_id,device);assert torch.equal(mask,dm)
   active.update(q=None,source=None);captured.clear()
   with torch.inference_mode():log=model(input_ids=bi,attention_mask=mask,use_cache=False).logits
   batch_index=torch.arange(len(b),device=bi.device);position_index=torch.tensor(pos,device=bi.device)
   base_states={q:captured[q][batch_index,position_index].clone()for q in(28,36)}
   for j,v in zip(b,scores(log,b)):out.append({"id":j["id"],"condition":"no_patch","ri":j["ri"],"value":v,"query":j["query"]})
   active.update(q=None,source=None);captured.clear()
   with torch.inference_mode():model(input_ids=di,attention_mask=dm,use_cache=False)
   donors={q:captured[q][batch_index,position_index].clone()for q in(28,36)}
   for name,q,src in(("self_q36",36,base_states[36]),("donor_q28",28,donors[28]),("donor_q36",36,donors[36]),("shuffled_q36",36,donors[36].roll(2,0))):
    active.update(q=q,source=src);captured.clear()
    with torch.inference_mode():log=model(input_ids=bi,attention_mask=mask,use_cache=False).logits
    for j,v in zip(b,scores(log,b)):out.append({"id":j["id"],"condition":name,"ri":j["ri"],"value":v,"query":j["query"]})
 finally:
  for h in handles:h.remove()
 return out
def causal_panels(rows:list[dict])->dict:
 x={(r["id"],r["condition"],r["ri"]):r for r in rows};ids=sorted({r["id"]for r in rows});conds=sorted({r["condition"]for r in rows});out={}
 for c in conds:
  v=[]
  for k in ids:
   q=x[k,c,0]["query"];sgn=1 if q==0 else-1;base=x[k,"no_patch",0]["value"]-x[k,"no_patch",1]["value"];p=x[k,c,0]["value"]-x[k,c,1]["value"];v.append((-sgn*(p-base),-sgn*p,abs(p-base)))
  out[c]={"n":len(v),"mean_shift":float(np.mean([z[0]for z in v])),"positive_shift_rate":float(np.mean([z[0]>0 for z in v])),"donor_flip_rate":float(np.mean([z[1]>0 for z in v])),"max_self_error":float(max(z[2]for z in v))}
 return out
def append_memo(r):
 if f"## Phase {PHASE}:"in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M");text=rf"""


## Phase {PHASE}: 自然九模式族三事件全坐标图谱与q36因果复核（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2520双unit合格九族，在facts-end、query-property、answer-boundary、q0–q37构造事实绑定swap×查询属性的全2560坐标Walsh交互；unit30按跨candidate/open和跨语言身份优势maximin选择描述层，unit31锁箱。因果部分不重选层：直接将合成任务冻结q36与q28带入unit31九族×两语言×两surface×两output-mode×两query=144个exact-shape answer-boundary自然patch，比较self、matched和batch错配。

$$I^{{natural}}=\tfrac14(H_{{00}}-H_{{01}}-H_{{10}}+H_{{11}}).$$

**结果汇总。** 全场分析 `{json.dumps(r['field_analysis'],ensure_ascii=False)}`；因果 `{json.dumps(r['causal'],ensure_ascii=False)}`；字段 `{json.dumps(r['fields'],ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2521_c86177_c87200_natural_field_and_causal_lockbox.py`；九族全层交互场、unit31因果分数、逐层指标、哈希和final位于`{OUT}`。

**分析与理论进展。** 若q36 matched在自然candidate/open两模式仍强于q28和shuffled，说明“晚层answer-boundary汇聚输出身份”跨模式族复现；这不是同一物理语义向量复用，而是事件角色规律。自然四格跨输出模式相似仅说明同一事实选择在两输出接口留下对应纹理。

**问题硬伤与结论。** 九族使用人工事实与短实体；属性词、句法和family同变；q36靠近unembedding，强因果效应可能是输出身份而非关系算法。多跳未过行为门，尚不能证明组合知识链。下一步跨模型只比较行为、因果涌现的相对深度与事件位置，不比较坐标号。
"""
 with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)
def main():
 f=load(P2520/"analysis/final.json");p19=load(P2519/"analysis/final.json");a,meta=build(f);derived=OUT/"derived";derived.mkdir(parents=True,exist_ok=True);fp=derived/"natural_walsh_allqpoint.float32.npy";np.save(fp,a);fa=field_analysis(a,f["behavior"]["qualified_families"])
 rows=read(P2520/"material/natural_rows.jsonl");model,tok,_=model_utils.load_model("qwen3",dtype=torch.bfloat16,use_8bit=False)
 try:cs=causal(model,tok,rows,f["behavior"]["qualified_family_ids"])
 finally:model_utils.release_model(model);gc.collect()
 cp=OUT/"output/natural_causal_scores.jsonl";write(cp,cs);ca=causal_panels(cs);checks={"sources_passed":f["all_checks_passed"]and p19["all_checks_passed"],"nine_families":len(f["behavior"]["qualified_family_ids"])==9,"field_shape":a.shape==(2,9,2,2,2,3,38,DIM),"facts_prefix_zero":bool(np.max(np.abs(a[...,0,:,:]))==0),"causal_144":ca["donor_q36"]["n"]==144,"self_exact":ca["self_q36"]["max_self_error"]==0,"hashes":len(sha(fp))==64 and len(sha(cp))==64,"claim_boundary":True}
 r={"phase":PHASE,"campaign":CAMPAIGN,"model":"Qwen3-4B nonquantized BF16 CUDA","field_analysis":fa,"causal":ca,"fields":{"interactions":{"path":str(fp),"shape":list(a.shape),"sha256":sha(fp)},"causal_scores":{"path":str(cp),"sha256":sha(cp)}},"adjudication":{"natural_answer_boundary_q36_replication":ca["donor_q36"]["donor_flip_rate"]>.75 and ca["donor_q36"]["mean_shift"]>ca["shuffled_q36"]["mean_shift"],"event_role_rule_candidate":True,"semantic_code_identified":False,"language_encoding_mechanism_closed":False},"checks":checks,"all_checks_passed":all(checks.values())};save(OUT/"analysis/final.json",r)
 if r["all_checks_passed"]:append_memo(r)
 print(json.dumps({"phase":PHASE,"field_selection":fa["selection"],"lockbox_field":fa["lockbox_selected"],"causal":ca,"checks":checks,"all_checks_passed":r["all_checks_passed"]},ensure_ascii=False,indent=2))
 if not r["all_checks_passed"]:raise RuntimeError(checks)
if __name__=="__main__":main()
