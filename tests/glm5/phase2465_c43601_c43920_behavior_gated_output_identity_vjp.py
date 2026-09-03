#!/usr/bin/env python3
"""All-coordinate entity-vs-code VJP with behavior-gated family adjudication."""
from __future__ import annotations
import gc,json,math,sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result";P2463=RESULT/"phase2463_c42961_c43280_autonomous_output_identity_contract";P2464=RESULT/"phase2464_c43281_c43600_balanced_code_interface_calibration"
OUT=RESULT/"phase2465_c43601_c43920_behavior_gated_output_identity_vjp";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md";PHASE,CAMPAIGN,DIM,SHIFT=2465,"C43601-C43920",2560,791;QPOINTS=(16,18);VARIANTS=("valid","broken_a","broken_b")
sys.path.insert(0,str(TESTS));import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability;import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils;import phase2397_c21681_c22000_operation_behavior_token_calibration as behavior  # noqa:E402
def save(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def read_rows(path:Path)->list[dict]:return[json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
def close(v):
    m=getattr(v,"_mmap",None)
    if m is not None:m.close()
def cosine(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1);d=float(np.linalg.norm(a)*np.linalg.norm(b));return float(np.dot(a,b)/d) if d>1e-30 else 0.0
def derangements(n,size,seed):
    rng=np.random.default_rng(seed);out=[]
    while len(out)<n:
        p=rng.permutation(size)
        if np.all(p!=np.arange(size)):out.append(p)
    return np.stack(out)
def find_subsequence(a,b):
    for i in range(len(a)-len(b)+1):
        if a[i:i+len(b)]==b:return i
    return-1
def common_prefix(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y:break
        n+=1
    return n
def compile_all(tokenizer)->tuple[list[dict],str]:
    bases=read_rows(P2463/"material/unseen_unit_rows.jsonl");candidate,audit=behavior.compile_rows(tokenizer,bases)
    for r in candidate:r["interface"]="candidate_entity";r["query_end_token"]=next(e["token_index"] for e in r["event_tokens"] if e["event"]=="query_end")
    final=json.loads((P2464/"analysis/final.json").read_text(encoding="utf-8"));selected=final["selection"]["selected_protocol"]
    source=[r for r in read_rows(P2464/"index/code_interface_rows.jsonl") if r["protocol"]==selected]
    code=[]
    base_lookup={r["case_id"]:r for r in bases}
    for r in source:
        base=base_lookup[r["base_case_id"]];raw=[int(x) for x in tokenizer.encode(r["prompt"],add_special_tokens=False)];start=find_subsequence(r["prompt_ids"],raw);end=r["prompt"].index(base["query"])+len(base["query"]);prefix=[int(x) for x in tokenizer.encode(r["prompt"][:end],add_special_tokens=False)];qi=start+max(0,common_prefix(raw,prefix)-1)
        code.append({**r,"interface":"letter_code","query_end_token":qi})
    rows=candidate+code;rows.sort(key=lambda r:(r["interface"],r["case_id"]));return rows,selected
def capture(model,rows):
    raw=OUT/"raw";raw.mkdir(parents=True,exist_ok=True);fp=raw/"entity_code_fields.float32.npy";mp=raw/"margin.float32.npy";fields=np.lib.format.open_memmap(fp,mode="r+" if fp.exists() else"w+",dtype=np.float32,shape=(len(rows),2,2,DIM));margins=np.lib.format.open_memmap(mp,mode="r+" if mp.exists() else"w+",dtype=np.float32,shape=(len(rows),));progress=raw/"progress.json";done=int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    for p in model.parameters():p.requires_grad_(False)
    modules=field_utils.modules(model);device=model.get_input_embeddings().weight.device;captures={};handles=[]
    def leaf(_m,_i,res):
        t=res[0] if isinstance(res,tuple) else res
        if not t.requires_grad:t.requires_grad_(True)
    handles.append(modules[0].register_forward_hook(leaf))
    for slot,q in enumerate(QPOINTS):
        def hook(_m,_i,res,slot=slot):
            t=res[0] if isinstance(res,tuple) else res;t.retain_grad();captures[slot]=t
        handles.append(modules[q].register_forward_hook(hook))
    try:
        for i in range(done,len(rows)):
            r=rows[i];ids=torch.tensor([r["prompt_ids"]],dtype=torch.long,device=device);mask=torch.ones_like(ids);pos=torch.arange(ids.shape[1],device=device)[None];captures.clear()
            out=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True);margin=out.logits[0,-1,int(r["target_ids"][0])]-out.logits[0,-1,int(r["foil_ids"][0])];margin.backward();ti=int(r["query_end_token"])
            for s in range(2):
                h=captures[s][0,ti].detach().float().cpu().numpy();g=captures[s].grad[0,ti].detach().float().cpu().numpy();fields[i,s,0]=g;fields[i,s,1]=h*g
            margins[i]=float(margin.detach().float().cpu());fields.flush();margins.flush();save(progress,{"completed":i+1,"rows":len(rows)})
            if(i+1)%16==0:print(f"[phase2465 VJP] {i+1}/{len(rows)}",flush=True)
    finally:
        for h in handles:h.remove()
        fields.flush();margins.flush();close(fields);close(margins)
    idx=OUT/"index/entity_code_rows.jsonl";idx.parent.mkdir(parents=True,exist_ok=True);idx.write_text("".join(json.dumps({k:r[k] for k in ("case_id","interface","family","unit","language","variant","query_role")},ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    return{"fields":str(fp),"margins":str(mp),"rows":len(rows),"shape":[len(rows),2,2,DIM],"qpoints":list(QPOINTS),"index":str(idx)}
def build_passports(rows,values,families):
    lookup={(r["interface"],int(r["unit"]),r["family"],r["language"],r["variant"],r["query_role"]):i for i,r in enumerate(rows)};out=np.zeros((2,2,2,2,2,2,8,DIM),dtype=np.float32)
    for ii,interface in enumerate(("candidate_entity","letter_code")):
      for ui,unit in enumerate((6,7)):
       for li,lang in enumerate(("en","zh")):
        for fi,fam in enumerate(families):
         role={}
         for v in VARIANTS:
          s=lookup[(interface,unit,fam,lang,v,"source")];t=lookup[(interface,unit,fam,lang,v,"target")];role[v]=values[t]-values[s]
         out[0,ii,ui,li,:,:,fi]=role["valid"]-role["broken_a"];out[1,ii,ui,li,:,:,fi]=role["broken_a"]-role["broken_b"]
    return out
def analyze(rows,collection,qualified):
    families=sorted({r["family"] for r in rows});v=np.load(collection["fields"],mmap_mode="r");p=build_passports(rows,v,families);close(v);pp=OUT/"derived/entity_code_passports.float32.npy";pp.parent.mkdir(parents=True,exist_ok=True);np.save(pp,p);perms=derangements(64,8,2465);summary={}
    for inter,name in enumerate(("semantic_validity","lexical_control")):
     summary[name]={}
     for field,fname in enumerate(("gradient","state_times_gradient")):
      slot=1 if field==0 else 0;summary[name][fname]={}
      for ui,unit in enumerate((6,7)):
       a,b=p[inter,0,ui,: ,slot,field],p[inter,1,ui,:,slot,field];obs=float(np.mean([cosine(a[l,f],b[l,f]) for l in range(2) for f in range(8)]));shift=float(np.mean([cosine(a[l,f],np.roll(b[l,f],SHIFT)) for l in range(2) for f in range(8)]));null=np.asarray([np.mean([cosine(a[l,f],b[l,x[f]]) for l in range(2) for f in range(8)]) for x in perms]);q95=float(np.quantile(null,.95));qi=[families.index(f) for f in qualified];qobs=float(np.mean([cosine(a[l,f],b[l,f]) for l in range(2) for f in qi])) if qi else float("nan")
       summary[name][fname][f"unit{unit}"]={"coordinate":obs,"shift791":shift,"family_null_mean":float(np.mean(null)),"family_null_q95":q95,"physical_advantage":obs-shift,"family_identity_advantage":obs-q95,"behavior_qualified_coordinate":qobs,"behavior_qualified_family_count":len(qi)}
    return{"families":families,"behavior_qualified_families":qualified,"summary":summary,"passports":str(pp)}
def append_memo(result):
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M");text=rf"""


## Phase {PHASE}: 行为门控实体输出—代码输出的双unit全坐标VJP锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 冻结Phase2464由unit6选择的代码协议，在unit6/7的八族中英三variant双角色上并列原candidate-entity与letter-code接口，共384条；采q16/q18 query-end全部2560坐标gradient与$H\odot g$。构造valid−brokenA语义interaction和brokenA−brokenB词项control，比较实体↔代码同坐标、+791错位、64个family置乱；八族输入响应与unit7行为合格族分别报告。

$$T_f=\cos(I^{{entity}}_f,I^{{code}}_f).$$

**结果汇总。** 采集 `{json.dumps(result['collection'],ensure_ascii=False)}`；全坐标裁决 `{json.dumps(result['analysis'],ensure_ascii=False)}`；总裁决 `{json.dumps(result['adjudication'],ensure_ascii=False)}`；检查 `{json.dumps(result['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2465_c43601_c43920_behavior_gated_output_identity_vjp.py`；384行原场、margin、实体/代码interaction护照和final位于同名结果目录。

**分析与理论进展。** 该Phase不再用行为近随机的八族整体为语义命名：只有Phase2464锁箱通过的family可以升级为“输出身份部分解耦候选”，其余四族仅是共享任务输入响应。词项control若同样强，仍不能声称语义专属。

**问题硬伤与结论。** 代码接口只有部分family合格，且仍共享候选映射和问题模板；首token margin不是完整自主回答。行为门控子集只有四族，family置乱q95仍来自八族，因此小子集优势只作补充描述。
""";MEMO.open("a",encoding="utf-8",newline="\n").write(text)
def main():
    f2464=json.loads((P2464/"analysis/final.json").read_text(encoding="utf-8"));qualified=f2464["selection"]["unit7_qualified_families"];model,tokenizer,_=capability.load_model("qwen4b")
    try:rows,protocol=compile_all(tokenizer);collection=capture(model,rows)
    finally:del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    analysis=analyze(rows,collection,qualified);s=analysis["summary"]["semantic_validity"]["state_times_gradient"]["unit7"];l=analysis["summary"]["lexical_control"]["state_times_gradient"]["unit7"]
    geometry=len(qualified)>=4 and s["behavior_qualified_coordinate"]>0 and s["physical_advantage"]>0 and s["family_identity_advantage"]>0
    semantic_specific=s["behavior_qualified_coordinate"]>l["behavior_qualified_coordinate"]
    adj={"selected_protocol":protocol,"all8_output_identity_geometry_lockbox":s["physical_advantage"]>0 and s["family_identity_advantage"]>0,
         "partial_behavior_gated_output_identity_geometry":geometry,
         "semantic_specific_output_identity_candidate":geometry and semantic_specific,
         "universal_output_identity_decoupling":False,"language_encoding_mechanism_closed":False}
    checks={"rows_384":collection["rows"]==384,"full_coordinates":collection["shape"]==[384,2,2,2560],"qualified_families_frozen":len(qualified)==4,"files":all(Path(collection[k]).exists() for k in ("fields","margins","index")) and Path(analysis["passports"]).exists(),"finite":all(math.isfinite(v) for x in analysis["summary"].values() for y in x.values() for z in y.values() for v in z.values()),"claim_boundary":not adj["language_encoding_mechanism_closed"]}
    result={"phase":PHASE,"campaign":CAMPAIGN,"collection":collection,"analysis":analysis,"adjudication":adj,"checks":checks,"all_checks_passed":all(checks.values())};save(OUT/"analysis/final.json",result);append_memo(result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result["all_checks_passed"]:raise RuntimeError(checks)
if __name__=="__main__":main()
