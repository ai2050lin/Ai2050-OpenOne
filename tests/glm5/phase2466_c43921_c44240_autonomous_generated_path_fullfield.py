#!/usr/bin/env python3
"""Capture full-coordinate states along the model's own greedy output prefix."""
from __future__ import annotations
import gc,json,math,sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result";P2464=RESULT/"phase2464_c43281_c43600_balanced_code_interface_calibration";P2465=RESULT/"phase2465_c43601_c43920_behavior_gated_output_identity_vjp"
OUT=RESULT/"phase2466_c43921_c44240_autonomous_generated_path_fullfield";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md";PHASE,CAMPAIGN,DIM,SHIFT=2466,"C43921-C44240",2560,791;QPOINTS=(0,16,18,37);EVENTS=("prompt_query_end","prompt_answer_boundary","generated_token1");VARIANTS=("valid","broken_a","broken_b")
sys.path.insert(0,str(TESTS));import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability;import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils;import phase2465_c43601_c43920_behavior_gated_output_identity_vjp as p2465  # noqa:E402
def save(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def close(v):
 m=getattr(v,"_mmap",None)
 if m is not None:m.close()
def cosine(a,b):
 a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1);d=float(np.linalg.norm(a)*np.linalg.norm(b));return float(np.dot(a,b)/d) if d>1e-30 else 0.0
def derangements(n,size,seed):
 rng=np.random.default_rng(seed);o=[]
 while len(o)<n:
  p=rng.permutation(size)
  if np.all(p!=np.arange(size)):o.append(p)
 return np.stack(o)
def capture(model,rows):
 raw=OUT/"raw";raw.mkdir(parents=True,exist_ok=True);fp=raw/"autonomous_path_states.float16.npy";fields=np.lib.format.open_memmap(fp,mode="w+",dtype=np.float16,shape=(len(rows),3,4,DIM));modules=field_utils.modules(model);device=model.get_input_embeddings().weight.device;caps={};handles=[]
 for slot,q in enumerate(QPOINTS):
  def hook(_m,_i,res,slot=slot):caps[slot]=res[0] if isinstance(res,tuple) else res
  handles.append(modules[q].register_forward_hook(hook))
 generated=[]
 try:
  with torch.inference_mode():
   for i,r in enumerate(rows):
    prompt=r["prompt_ids"];ids=torch.tensor([prompt],dtype=torch.long,device=device);mask=torch.ones_like(ids);pos=torch.arange(ids.shape[1],device=device)[None];caps.clear();out=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True);first=int(torch.argmax(out.logits[0,-1]).item())
    qi=int(r["query_end_token"])
    for s in range(4):fields[i,0,s]=caps[s][0,qi].float().cpu().numpy();fields[i,1,s]=caps[s][0,-1].float().cpu().numpy()
    ids1=torch.tensor([prompt+[first]],dtype=torch.long,device=device);mask1=torch.ones_like(ids1);pos1=torch.arange(ids1.shape[1],device=device)[None];caps.clear();out1=model(input_ids=ids1,attention_mask=mask1,position_ids=pos1,use_cache=False,return_dict=True);second=int(torch.argmax(out1.logits[0,-1]).item())
    for s in range(4):fields[i,2,s]=caps[s][0,-1].float().cpu().numpy()
    expected=[int(x) for x in r["target_ids"]];actual=[first]
    prefix=prompt+[first]
    while len(actual)<len(expected):
     if len(actual)==1:nxt=second
     else:
      ii=torch.tensor([prefix],dtype=torch.long,device=device);oo=model(input_ids=ii,attention_mask=torch.ones_like(ii),use_cache=False,return_dict=True);nxt=int(torch.argmax(oo.logits[0,-1]).item())
     actual.append(nxt);prefix.append(nxt)
    generated.append({"case_id":r["case_id"],"interface":r["interface"],"family":r["family"],"language":r["language"],"variant":r["variant"],"query_role":r["query_role"],"target_ids":expected,"generated_ids":actual,"exact":actual==expected,"target_first":first==expected[0]})
    fields.flush()
    if(i+1)%16==0:print(f"[phase2466 autonomous] {i+1}/{len(rows)}",flush=True)
 finally:
  for h in handles:h.remove()
  fields.flush();close(fields)
 idx=OUT/"index/autonomous_rows.jsonl";idx.parent.mkdir(parents=True,exist_ok=True);idx.write_text("".join(json.dumps({k:r[k] for k in ("case_id","interface","family","unit","language","variant","query_role")},ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
 gp=OUT/"behavior/greedy_generation.jsonl";gp.parent.mkdir(parents=True,exist_ok=True);gp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in generated),encoding="utf-8")
 return{"field":str(fp),"index":str(idx),"generation":str(gp),"shape":[len(rows),3,4,DIM],"rows":len(rows),"events":list(EVENTS),"qpoints":list(QPOINTS),"dtype":"float16 full physical coordinates","forward_passes_minimum":len(rows)*2},generated
def passports(rows,values,families):
 lookup={(r["interface"],r["family"],r["language"],r["variant"],r["query_role"]):i for i,r in enumerate(rows)};o=np.zeros((2,2,3,4,2,8,DIM),dtype=np.float32)
 for ii,interface in enumerate(("candidate_entity","letter_code")):
  for li,lang in enumerate(("en","zh")):
   for fi,fam in enumerate(families):
    role={}
    for v in VARIANTS:
     s=lookup[(interface,fam,lang,v,"source")];t=lookup[(interface,fam,lang,v,"target")];role[v]=np.asarray(values[t],dtype=np.float32)-np.asarray(values[s],dtype=np.float32)
    o[0,ii,:,:,li,fi]=role["valid"]-role["broken_a"];o[1,ii,:,:,li,fi]=role["broken_a"]-role["broken_b"]
 return o
def analyze(rows,collection,generated,qualified):
 families=sorted({r["family"] for r in rows});v=np.load(collection["field"],mmap_mode="r");p=passports(rows,v,families);close(v);pp=OUT/"derived/autonomous_path_passports.float32.npy";pp.parent.mkdir(parents=True,exist_ok=True);np.save(pp,p);perms=derangements(64,8,2466);summary={}
 for inter,name in enumerate(("semantic_validity","lexical_control")):
  summary[name]={}
  for event in range(3):
   summary[name][EVENTS[event]]={}
   for qs,q in enumerate(QPOINTS):
    a,b=p[inter,0,event,qs],p[inter,1,event,qs];obs=float(np.mean([cosine(a[l,f],b[l,f]) for l in range(2) for f in range(8)]));shift=float(np.mean([cosine(a[l,f],np.roll(b[l,f],SHIFT)) for l in range(2) for f in range(8)]));null=np.asarray([np.mean([cosine(a[l,f],b[l,x[f]]) for l in range(2) for f in range(8)]) for x in perms]);q95=float(np.quantile(null,.95));qi=[families.index(f) for f in qualified];qobs=float(np.mean([cosine(a[l,f],b[l,f]) for l in range(2) for f in qi])) if qi else float("nan")
    summary[name][EVENTS[event]][f"q{q}"]={"coordinate":obs,"shift791":shift,"family_null_q95":q95,"physical_advantage":obs-shift,"family_identity_advantage":obs-q95,"behavior_qualified_coordinate":qobs,"behavior_qualified_family_count":len(qi)}
 behavior={}
 for interface in ("candidate_entity","letter_code"):
  behavior[interface]={}
  for fam in families:
   selected=[r for r in generated if r["interface"]==interface and r["family"]==fam and r["variant"]=="valid"]
   behavior[interface][fam]={"rows":len(selected),"exact_rate":float(np.mean([r["exact"] for r in selected])),"target_first_rate":float(np.mean([r["target_first"] for r in selected]))}
  valid=[r for r in generated if r["interface"]==interface and r["variant"]=="valid"];behavior[interface]["aggregate"]={"rows":len(valid),"exact_rate":float(np.mean([r["exact"] for r in valid])),"target_first_rate":float(np.mean([r["target_first"] for r in valid]))}
 return{"families":families,"behavior_qualified_families":qualified,"behavior":behavior,"crossinterface_path":summary,"passports":str(pp)}
def append_memo(result):
 if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M");text=rf"""


## Phase {PHASE}: 模型自主贪心输出前缀的Embedding—HiddenState全坐标路径（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对冻结unit7八族中英、三variant双角色，同时运行candidate-entity与Phase2464选定letter-code接口，共192条。模型不接受教师强制答案：先从prompt实际argmax首token，再用该token作为前缀计算下一token。保存prompt-query-end、prompt-answer-boundary、generated-token1在q0 embedding、q16、q18、q37的全部2560坐标float16场；按目标实际token长度报告完整贪心前缀exact。随后构造语义/词项interaction，比较实体与代码路径的同坐标、+791及64 family错配。

$$\hat y_1=\arg\max_v\ell_v(x),\qquad \hat y_2=\arg\max_v\ell_v(x,\hat y_1).$$

**结果汇总。** 采集 `{json.dumps(result['collection'],ensure_ascii=False)}`；自主行为 `{json.dumps(result['analysis']['behavior'],ensure_ascii=False)}`；三事件四检查点跨接口 `{json.dumps(result['analysis']['crossinterface_path'],ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'],ensure_ascii=False)}`；检查 `{json.dumps(result['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2466_c43921_c44240_autonomous_generated_path_fullfield.py`；192×3事件×4qpoint×2560全坐标场、逐行实际生成、语义/词项路径护照和final位于同名结果目录。

**分析与理论进展。** 这是本阶段第一次不把正确首token前缀塞回模型。prompt端纹理若跨接口复用而generated-token1崩解，说明相对编码在输出身份具体化时分化；若行为合格族在实际生成token后仍胜family错配，才支持自主路径复用候选。

**问题硬伤与结论。** 贪心不等于采样分布，生成长度仅为实体名目标长度；错误输出路径仍会被测量，但不能叫成功语义执行。特别更正Phase2465：合格四族的语义跨接口坐标0.428低于词项0.440，因此只确认部分几何复用，不确认语义专属输出身份齿轮。
""";MEMO.open("a",encoding="utf-8",newline="\n").write(text)
def main():
 f=json.loads((P2464/"analysis/final.json").read_text(encoding="utf-8"));qualified=f["selection"]["unit7_qualified_families"];model,tokenizer,_=capability.load_model("qwen4b")
 try:allrows,_=p2465.compile_all(tokenizer);rows=[r for r in allrows if int(r["unit"])==7];collection,generated=capture(model,rows)
 finally:del model,tokenizer;gc.collect();torch.cuda.empty_cache()
 analysis=analyze(rows,collection,generated,qualified);s=analysis["crossinterface_path"]["semantic_validity"]["generated_token1"]["q18"];l=analysis["crossinterface_path"]["lexical_control"]["generated_token1"]["q18"];entity=analysis["behavior"]["candidate_entity"]["aggregate"];code=analysis["behavior"]["letter_code"]["aggregate"]
 adj={"entity_autonomous_exact_rate":entity["exact_rate"],"code_autonomous_exact_rate":code["exact_rate"],"autonomous_generated_token_geometry_lockbox":s["physical_advantage"]>0 and s["family_identity_advantage"]>0,"behavior_gated_semantic_specific_autonomous_path":s["behavior_qualified_coordinate"]>l["behavior_qualified_coordinate"] and min(entity["target_first_rate"],code["target_first_rate"])>=.75,"language_encoding_mechanism_closed":False}
 checks={"rows_192":collection["rows"]==192,"full_coordinates":collection["shape"]==[192,3,4,2560],"actual_prefix":all(len(r["generated_ids"])==len(r["target_ids"]) for r in generated),"files":all(Path(collection[k]).exists() for k in ("field","index","generation")) and Path(analysis["passports"]).exists(),"finite":all(math.isfinite(v) for x in analysis["crossinterface_path"].values() for y in x.values() for z in y.values() for v in z.values()),"claim_boundary":not adj["language_encoding_mechanism_closed"]}
 result={"phase":PHASE,"campaign":CAMPAIGN,"collection":collection,"analysis":analysis,"adjudication":adj,"checks":checks,"all_checks_passed":all(checks.values())};save(OUT/"analysis/final.json",result);append_memo(result);print(json.dumps(result,ensure_ascii=False,indent=2))
 if not result["all_checks_passed"]:raise RuntimeError(checks)
if __name__=="__main__":main()
