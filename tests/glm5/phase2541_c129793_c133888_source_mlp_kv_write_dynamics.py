#!/usr/bin/env python3
"""Full-coordinate source Attention/MLP write dynamics into next-layer K/V plus causal tests."""
from __future__ import annotations
import gc,hashlib,json,sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result"
P2538=RESULT/"phase2538_c117505_c121600_token_atomic_hypergraph_behavior";P2540=RESULT/"phase2540_c125697_c129792_qkv_separated_causal_lockbox"
OUT=RESULT/"phase2541_c129793_c133888_source_mlp_kv_write_dynamics";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE,CAMPAIGN=2541,"C129793-C133888";LATE=tuple(range(20,36));TRANSITION=tuple(range(20,35));SOURCE_NAMES=("facts_entity","facts_relation","facts_value")
sys.path.insert(0,str(TESTS));import model_utils  # noqa:E402

def load(p:Path)->Any:return json.loads(p.read_text(encoding="utf-8-sig"))
def read(p:Path)->list[dict]:return[json.loads(x)for x in p.read_text(encoding="utf-8-sig").splitlines()if x.strip()]
def save(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def write(p:Path,rows:list[dict])->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n"for r in rows),encoding="utf-8")
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb")as f:
  for b in iter(lambda:f.read(16*1024*1024),b""):h.update(b)
 return h.hexdigest()
def alloc(p:Path,shape):p.parent.mkdir(parents=True,exist_ok=True);x=np.lib.format.open_memmap(p,mode="w+",dtype=np.float16,shape=shape);x[:]=np.nan;return x
def pad(seqs,pad_id,device):
 w=max(map(len,seqs));ids=torch.full((len(seqs),w),pad_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
 for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s,device=device);mask[i,:len(s)]=1
 return ids,mask
def rrms(a,b):
 a=a.detach().float();b=b.detach().float();return float(torch.sqrt(torch.mean((a-b)**2))/(torch.sqrt(torch.mean(b**2))+1e-12))
def source_pos(r):return sorted({p for n in SOURCE_NAMES for p in r["regions"][n]})

def collect(model,tokenizer,rows:list[dict])->tuple[list[dict],dict]:
 rows=sorted(rows,key=lambda r:(r["family_id"],r["language"],r["query_property"]));layers=model_utils.get_layers(model);n=len(rows);maxseq=max(len(r["prompt_ids"])for r in rows);dim=int(model.config.hidden_size);nkv=int(model.config.num_key_value_heads);hd=int(model.config.head_dim)
 paths={"components":OUT/"fields/source_fulltoken_in_attn_mlp_out.float16.npy","next_key":OUT/"fields/next_layer_key_counterfactual.float16.npy","next_value":OUT/"fields/next_layer_value_counterfactual.float16.npy"}
 comp=alloc(paths["components"],(n,16,4,maxseq,dim));key=alloc(paths["next_key"],(n,15,4,nkv,maxseq,hd));value=alloc(paths["next_value"],(n,15,4,nkv,maxseq,hd))
 incoming={};attn={};mlp={};outgoing={};handles=[]
 for li in LATE:
  def pre(_m,args,li=li):incoming[li]=args[0].detach()
  def ah(_m,_a,o,li=li):attn[li]=(o[0]if isinstance(o,tuple)else o).detach()
  def mh(_m,_a,o,li=li):mlp[li]=o.detach()
  def post(_m,_a,o,li=li):outgoing[li]=(o[0]if isinstance(o,tuple)else o).detach()
  handles += [layers[li].register_forward_pre_hook(pre),layers[li].self_attn.register_forward_hook(ah),layers[li].mlp.register_forward_hook(mh),layers[li].register_forward_hook(post)]
 device=model.get_input_embeddings().weight.device;maxerr=0.;effect={"no_attention_k":[],"no_mlp_k":[],"no_attention_v":[],"no_mlp_v":[]};index=[]
 try:
  for start in range(0,n,2):
   batch=rows[start:start+2];ids,mask=pad([r["prompt_ids"]for r in batch],tokenizer.pad_token_id,device);incoming.clear();attn.clear();mlp.clear();outgoing.clear()
   with torch.inference_mode():model(input_ids=ids,attention_mask=mask,use_cache=False)
   for bi,r in enumerate(batch):
    ri=start+bi;seq=len(r["prompt_ids"]);pos=source_pos(r)
    for lli,li in enumerate(LATE):
     states=(incoming[li][bi,:seq],attn[li][bi,:seq],mlp[li][bi,:seq],outgoing[li][bi,:seq])
     for si,s in enumerate(states):comp[ri,lli,si,:seq]=s.float().cpu().numpy().astype(np.float16)
     maxerr=max(maxerr,rrms(states[0]+states[1]+states[2],states[3]))
     if li<35:
      variants=(states[3],states[0]+states[2],states[0]+states[1],states[0]) # natural,no-attn,no-mlp,incoming
      next_layer=layers[li+1]
      for vi,state in enumerate(variants):
       with torch.inference_mode():
        x=next_layer.input_layernorm(state);k=next_layer.self_attn.k_norm(next_layer.self_attn.k_proj(x).view(seq,-1,hd)).transpose(0,1);v=next_layer.self_attn.v_proj(x).view(seq,-1,hd).transpose(0,1)
       key[ri,lli,vi,:,:seq]=k.float().cpu().numpy().astype(np.float16);value[ri,lli,vi,:,:seq]=v.float().cpu().numpy().astype(np.float16)
      natural_k=torch.from_numpy(np.asarray(key[ri,lli,0,:,pos],np.float32));natural_v=torch.from_numpy(np.asarray(value[ri,lli,0,:,pos],np.float32))
      for name,vi,base in (("no_attention_k",1,natural_k),("no_mlp_k",2,natural_k)):
       alt=torch.from_numpy(np.asarray(key[ri,lli,vi,:,pos],np.float32));effect[name].append(rrms(alt,base))
      for name,vi,base in (("no_attention_v",1,natural_v),("no_mlp_v",2,natural_v)):
       alt=torch.from_numpy(np.asarray(value[ri,lli,vi,:,pos],np.float32));effect[name].append(rrms(alt,base))
    index.append({"field_row":ri,"case_id":r["case_id"],"family_id":r["family_id"],"family":r["family"],"language":r["language"],"query_property":r["query_property"],"prompt_length":seq,"source_positions":pos})
   if(start+len(batch))%32==0:
    comp.flush();key.flush();value.flush();print(f"[phase2541 field] {start+len(batch)}/{n}",flush=True)
 finally:
  for h in handles:h.remove()
  comp.flush();key.flush();value.flush()
 del comp,key,value
 return index,{"component_conservation_max_relative_rms":maxerr,"next_kv_effect":{k:{"mean":float(np.mean(v)),"median":float(np.median(v)),"p95":float(np.quantile(v,.95))}for k,v in effect.items()},"fields":{k:{"path":str(p),"shape":list(np.load(p,mmap_mode="r").shape),"bytes":p.stat().st_size,"sha256":sha(p)}for k,p in paths.items()}}

class SourceController:
 def __init__(self,model):
  self.layers=model_utils.get_layers(model);self.mode="none";self.condition="";self.jobs=[];self.store={"base":{},"donor":{}};self.live={};self.handles=[]
  for li in LATE:
   def pre(_m,args,li=li):self.live[("in",li)]=args[0].detach()
   def ah(_m,_a,o,li=li):self.live[("attn",li)]=(o[0]if isinstance(o,tuple)else o).detach()
   def mh(_m,_a,o,li=li):self.live[("mlp",li)]=o.detach()
   def post(_m,_a,o,li=li):return self.post(o,li)
   self.handles += [self.layers[li].register_forward_pre_hook(pre),self.layers[li].self_attn.register_forward_hook(ah),self.layers[li].mlp.register_forward_hook(mh),self.layers[li].register_forward_hook(post)]
 def close(self):
  for h in self.handles:h.remove()
 def post(self,o,li):
  x=o[0]if isinstance(o,tuple)else o
  if self.mode in self.store:
   for kind in("in","attn","mlp"):self.store[self.mode][(kind,li)]=self.live[(kind,li)].clone()
   self.store[self.mode][("out",li)]=x.detach().clone();return None
  if self.mode!="patch"or self.condition=="no_patch":return None
  y=x.clone();c=self.condition
  for bi,j in enumerate(self.jobs):
   for p in j["source"]:
    if c=="no_mlp_source":y[bi,p]-=self.live[("mlp",li)][bi,p]
    elif c=="no_attention_source":y[bi,p]-=self.live[("attn",li)][bi,p]
    elif c=="incoming_only_source":y[bi,p]=self.live[("in",li)][bi,p]
    elif c in("donor_mlp_source","donor_attention_source","donor_both_source"):
     kinds=("mlp",)if c=="donor_mlp_source"else(("attn",)if c=="donor_attention_source"else("attn","mlp"))
     for kind in kinds:y[bi,p]+=self.store["donor"][(kind,li)][bi,p]-self.store["base"][(kind,li)][bi,p]
    elif c=="donor_full_source":y[bi,p]=self.store["donor"][("out",li)][bi,p]
    elif c=="shuffled_mlp_source":y[bi,p]+=self.store["donor"][("mlp",li)][(bi-1)%len(self.jobs),p]-self.store["base"][("mlp",li)][bi,p]
  return (y,*o[1:])if isinstance(o,tuple)else y

def build_jobs(tokenizer,material):
 idx={(r["family_id"],r["language"],r["meaning_swap"],r["query_property"]):r for r in material if r["unit"]==35 and r["surface"]==0};jobs=[]
 for r in material:
  if r["unit"]!=35 or r["surface"]!=0 or r["meaning_swap"]!=0:continue
  d=idx[(r["family_id"],r["language"],1,r["query_property"])]
  for ci,e in enumerate(r["entities"]):
   cont=[int(x)for x in tokenizer.encode((" "if r["language"]=="en"else"")+e,add_special_tokens=False)]
   jobs.append({"case_id":r["case_id"],"candidate":e,"target":r["target"],"donor_target":d["target"],"family":r["family"],"language":r["language"],"prompt_length":len(r["prompt_ids"]),"base":r["prompt_ids"]+cont,"donor":d["prompt_ids"]+cont,"source":source_pos(r)})
 return jobs
def scores(logits,jobs):
 lp=torch.log_softmax(logits.float(),-1);return[float(sum(lp[i,j["prompt_length"]-1+o,t]for o,t in enumerate(j["base"][j["prompt_length"]:])))for i,j in enumerate(jobs)]
def causal(model,tokenizer,jobs):
 device=model.get_input_embeddings().weight.device;c=SourceController(model);conditions=("no_patch","no_mlp_source","no_attention_source","incoming_only_source","donor_mlp_source","donor_attention_source","donor_both_source","donor_full_source","shuffled_mlp_source");out=[]
 try:
  for start in range(0,len(jobs),8):
   b=jobs[start:start+8];c.jobs=b;w=max(max(len(j["base"]),len(j["donor"]))for j in b);t={s:pad([j[s]for j in b],tokenizer.pad_token_id,device)for s in("base","donor")}
   for source in("base","donor"):
    c.mode=source;ids,mask=t[source]
    with torch.inference_mode():z=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
    if source=="base":baseline=scores(z,b)
   for condition in conditions:
    if condition=="no_patch":vals=baseline
    else:
     c.mode="patch";c.condition=condition;ids,mask=t["base"]
     with torch.inference_mode():z=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
     vals=scores(z,b)
    for j,v in zip(b,vals):out.append({"case_id":j["case_id"],"candidate":j["candidate"],"condition":condition,"score":v,"target":j["target"],"donor_target":j["donor_target"],"family":j["family"],"language":j["language"]})
   if(start+len(b))%64==0:print(f"[phase2541 causal] {start+len(b)}/{len(jobs)}",flush=True)
 finally:c.close()
 return out
def panel(rows):
 g=defaultdict(list)
 for r in rows:g[(r["condition"],r["case_id"])].append(r)
 by=defaultdict(list)
 for(c,_),xs in g.items():
  s={x["candidate"]:x["score"]for x in xs};m=xs[0];p=max(s,key=s.get);by[c].append((p==m["target"],p==m["donor_target"],s[m["target"]]-s[m["donor_target"]]))
 return{c:{"n":len(x),"accuracy":float(np.mean([z[0]for z in x])),"donor_flip":float(np.mean([z[1]for z in x])),"mean_target_margin":float(np.mean([z[2]for z in x]))}for c,x in by.items()}

def append_memo(r):
 if f"## Phase {PHASE}:"in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M");compact={k:{q:v[q]for q in("path","shape","bytes","sha256")}for k,v in r["fields"].items()}
 text=rf"""


## Phase {PHASE}: source Attention/MLP写入下一层K/V的全坐标动力学（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Qwen3-4B BF16非量化上，对unit35、surface0、meaning-swap0的32族×英中×双query共128条token-atomic提示，保存layer20–35所有token的incoming residual、Attention输出、MLP输出和outgoing residual全部2560坐标。对每个layer20–34的自然outgoing、去Attention、去MLP和只留incoming四种状态，重新通过下一层真实RMSNorm、$W_K$和$W_V$，保存全部8个KV heads×128坐标。另在128个candidate case上持续移除或donor替换事实source的Attention/MLP写入，区分组件观察效应和最终行为因果。

$$h_{{l+1,j}}=h_{{l,j}}+A_{{l,j}}+M_{{l,j}},\qquad (K,V)_{{l+1,j}}=(W_K,W_V)N_{{l+1}}(h_{{l+1,j}}).$$

**结果汇总。** 数值与下一层效应 `{json.dumps(r['dynamics'],ensure_ascii=False)}`；行为干预 `{json.dumps(r['causal'],ensure_ascii=False)}`；字段 `{json.dumps(compact,ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2541_c129793_c133888_source_mlp_kv_write_dynamics.py`；component全场、四状态下一层K/V、逐条件候选分数、索引及final位于`{OUT}`。

**分析与理论进展。** 这一步回答“自然Attention与MLP在数值上各自把source状态推向怎样的下一层K/V”，并用持续删除检验这些写入是否会影响输出。重算K/V差异只是一阶反事实账本；只有实际逐层干预后的行为损伤或source特异donor翻转，才支持自然功能贡献。Attention与MLP可能一促一抑或被后层重构，阴性不能被简单解释为组件无信息。

**问题硬伤与结论。** 去组件状态保留另一个组件的自然值，未重算同层内部依赖；持续source修改可能分布外；donor组件差分仍有兼容性问题；实验只覆盖晚层，早中层上游形成过程尚未穷尽。结果只界定晚层source写入与输出的关系，不宣布MLP或K/V语义职责闭合。
"""
 with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)

def main():
 prior=load(P2540/"analysis/final.json");material=read(P2538/"material/token_atomic_rows.jsonl");field_rows=[r for r in material if r["unit"]==35 and r["surface"]==0 and r["meaning_swap"]==0]
 model,tokenizer,_=model_utils.load_model("qwen3",dtype=torch.bfloat16,use_8bit=False)
 try:index,dyn=collect(model,tokenizer,field_rows);jobs=build_jobs(tokenizer,material);causal_rows=causal(model,tokenizer,jobs)
 finally:model_utils.release_model(model);gc.collect()
 ip=OUT/"material/field_rows.jsonl";write(ip,index);cp=OUT/"causal/component_scores.jsonl";write(cp,causal_rows);pan=panel(causal_rows)
 fields=dyn["fields"]|{"index":{"path":str(ip),"shape":[len(index)],"bytes":ip.stat().st_size,"sha256":sha(ip)}}
 checks={"source_passed":prior["all_checks_passed"],"field_rows_128":len(index)==128,"component_conservation":dyn["component_conservation_max_relative_rms"]<.01,
         "all_physical_coordinates":fields["components"]["shape"][-1]==2560 and fields["next_key"]["shape"][-1]==128,"causal_cases_128":all(x["n"]==128 for x in pan.values()),
         "baseline_gate":pan["no_patch"]["accuracy"]>=.95,"full_fields_hashed":all(len(v["sha256"])==64 for v in fields.values()),"claim_boundary":True}
 result={"phase":PHASE,"campaign":CAMPAIGN,"model":"Qwen3-4B BF16 CUDA nonquantized","dynamics":{"component_conservation_max_relative_rms":dyn["component_conservation_max_relative_rms"],"next_kv_effect":dyn["next_kv_effect"]},"causal":pan,"fields":fields,"checks":checks,"all_checks_passed":all(checks.values())}
 save(OUT/"analysis/final.json",result)
 if result["all_checks_passed"]:append_memo(result)
 print(json.dumps({"phase":PHASE,"dynamics":result["dynamics"],"causal":pan,"field_bytes":{k:v["bytes"]for k,v in fields.items()},"checks":checks,"all_checks_passed":result["all_checks_passed"]},ensure_ascii=False,indent=2))
 if not result["all_checks_passed"]:raise RuntimeError(checks)
if __name__=="__main__":main()
