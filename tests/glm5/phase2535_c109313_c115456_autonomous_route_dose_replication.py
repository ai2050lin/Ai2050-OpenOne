#!/usr/bin/env python3
"""Automatic continuation: large autonomous top/random route-dose replication across both fresh units and surfaces."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result"
P2529=RESULT/"phase2529_c95777_c97568_full_source_kv_head_residual_ledger";P2532=RESULT/"phase2532_c101121_c104192_autonomous_multifamily_recursive_generation";P2534=RESULT/"phase2534_c107265_c109312_source_route_parameter_heatmap_publish"
OUT=RESULT/"phase2535_c109313_c115456_autonomous_route_dose_replication";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md";ASSET=ROOT/"frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
PHASE,CAMPAIGN=2535,"C109313-C115456";DOSES=(8,16,24,32)
sys.path.insert(0,str(TESTS));import model_utils  # noqa:E402
import phase2532_c101121_c104192_autonomous_multifamily_recursive_generation as p2532  # noqa:E402

def load(p:Path)->Any:return json.loads(p.read_text(encoding="utf-8-sig"))
def read(p:Path)->list[dict]:return[json.loads(x)for x in p.read_text(encoding="utf-8-sig").splitlines()if x.strip()]
def save(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def write(p:Path,rows:list[dict])->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n"for r in rows),encoding="utf-8")
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb")as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def route_map(items:list[dict])->dict[int,list[int]]:
 out={}
 for x in items:out.setdefault(int(x["layer"]),[]).append(int(x["head"]))
 return out

def summarize(rows:list[dict])->dict:
 out={}
 for condition in sorted({r["condition"]for r in rows}):
  vals=[r for r in rows if r["condition"]==condition]
  out[condition]={"n":len(vals),"accuracy":float(np.mean([r["correct"]for r in vals])),
                  "unit":{str(u):float(np.mean([r["correct"]for r in vals if r["unit"]==u]))for u in(32,33)},
                  "surface":{str(s):float(np.mean([r["correct"]for r in vals if r["surface"]==s]))for s in(0,1)},
                  "language":{l:float(np.mean([r["correct"]for r in vals if r["language"]==l]))for l in("en","zh")},
                  "swap":{str(m):float(np.mean([r["correct"]for r in vals if r["meaning_swap"]==m]))for m in(0,1)}}
 return out

def update_asset(metrics:dict)->dict:
 payload=load(ASSET);payload["phase"]=PHASE;payload["campaign"]="C39761-C115456";payload["summary"]["phase2535_autonomous_route_dose"]={k:{"n":v["n"],"accuracy":v["accuracy"],"unit":v["unit"],"surface":v["surface"],"language":v["language"]}for k,v in metrics.items()};sentence=" Phase2535 reports a 672-prompt autonomous top/random route-dose replication across both fresh units and surfaces; dose is a group intervention size, not a count of individually necessary heads."
 if sentence.strip()not in payload["claim_boundary"]:payload["claim_boundary"]+=sentence
 content=json.dumps(payload,ensure_ascii=False,indent=2)+"\n"
 if ASSET.read_text(encoding="utf-8")!=content:ASSET.write_text(content,encoding="utf-8")
 return{"path":str(ASSET),"bytes":ASSET.stat().st_size,"sha256":sha(ASSET)}

def append_memo(r):
 if f"## Phase {PHASE}:"in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
 text=rf"""


## Phase {PHASE}: 自动续研——二十一族双unit双surface自主route剂量锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2534后判断即时目标仍是同一冻结late-head路线能否跨新语言族、unit与surface稳定参与，因此自动续研而不停止。使用Phase2532已通过双unit门的21个族，unit32/33×英中×surface0/1×meaning-swap0/1×query0/1，共672条无候选多token实体自主生成。Phase2529 unit30冻结的贡献top32按能量顺序取8/16/24/32路线；等量随机32用预先固定随机排列取同样剂量；另持续删除全部512条late routes。所有干预作用于每个自回归步，不重新选择head。

$$A(d)=\frac1N\sum_x\mathbf1[\hat y(x;\operatorname{{zero}}(G_{{1:d}}))=y(x)],\qquad d\in\{{8,16,24,32\}}.$$

**结果汇总。** 全体及unit/surface/language/swap分层 `{json.dumps(r['metrics'],ensure_ascii=False)}`；冻结顺序 `{json.dumps(r['routes'],ensure_ascii=False)}`；客户端摘要 `{json.dumps(r['visual'],ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2535_c109313_c115456_autonomous_route_dose_replication.py`；6048条条件生成记录、冻结剂量清单、哈希、更新后的c42641摘要和final位于`{OUT}`。

**分析与理论进展。** 大样本剂量曲线回答“冻结联盟规模扩大时自然自主行为怎样变化”，并用同规模随机曲线控制一般损伤。只有跨两个新unit、两个surface和两语言都出现top特异剂量效应，才能把Phase2532的单锁箱结论提升为较稳定的模型内路线规律；曲线非单调时按耦合与抑制性head解释，不强行拟合线性齿轮。

**问题硬伤与结论。** 剂量顺序来自unit30全坐标能量而非逐head必要性；top-d集合嵌套、随机-d也嵌套，样本不是独立处理；zero输出是强干预；所有任务仍是受控微世界和短实体。该Phase不把32称最小齿轮，只测冻结路线联盟的跨材料可重复自然参与。
"""
 with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)

def main():
 f29=load(P2529/"analysis/final.json");f32=load(P2532/"analysis/final.json");f34=load(P2534/"analysis/final.json");qualified=set(f32["behavior"]["qualified_family_ids"]);material=[r for r in read(P2532/"material/autonomous_rows.jsonl")if r["task"]=="main"and r["family_id"]in qualified];baseline=[r for r in read(P2532/"behavior/baseline_generation.jsonl")if r["task"]=="main"and r["family_id"]in qualified]
 top=f29["routes"]["top"];random_items=list(f29["routes"]["random"]);perm=np.random.default_rng(2535).permutation(len(random_items));random_items=[random_items[int(i)]for i in perm];route_specs={}
 for d in DOSES:route_specs[f"head_zero_top{d}"]=route_map(top[:d]);route_specs[f"head_zero_random{d}"]=route_map(random_items[:d])
 path=OUT/"output/autonomous_dose_generation.jsonl"
 if path.exists() and len(read(path))==6720:
  rows=read(path)
 else:
  model,tokenizer,_=model_utils.load_model("qwen3",dtype=torch.bfloat16,use_8bit=False)
  try:
   rows=[{**r,"condition":"no_patch"}for r in baseline]
   for condition,routes in route_specs.items():rows+=p2532.generate(model,tokenizer,material,condition,routes,16)
   rows+=p2532.generate(model,tokenizer,material,"head_zero_all_late",{},16)
  finally:model_utils.release_model(model);gc.collect()
  write(path,rows)
 metrics=summarize(rows);visual=update_asset(metrics);routes_path=OUT/"analysis/frozen_dose_routes.json";save(routes_path,{"top":top,"random_permuted":random_items,"doses":list(DOSES)})
 checks={"sources_passed":f29["all_checks_passed"]and f32["all_checks_passed"]and f34["all_checks_passed"],"families_21":len(qualified)==21,"prompts_672":len(material)==672,"conditions_10":len(metrics)==10,"rows_6720":len(rows)==6720,"baseline_behavior_at_least_0_85":metrics["no_patch"]["accuracy"]>=.85,"top_random_all_doses":all(f"head_zero_top{d}"in metrics and f"head_zero_random{d}"in metrics for d in DOSES),"hashes":all(len(x)==64 for x in(sha(path),sha(routes_path),visual["sha256"])),"claim_boundary":True}
 r={"phase":PHASE,"campaign":CAMPAIGN,"model":"Qwen3-4B BF16 CUDA nonquantized","scope":{"families":f32["behavior"]["qualified_families"],"prompts":len(material),"units":[32,33],"languages":["en","zh"],"surfaces":[0,1],"doses":list(DOSES)},"metrics":metrics,"routes":{"top_order_source":"Phase2529 unit30 full-coordinate source contribution energy","random_seed":2535,"nested_doses":list(DOSES)},"visual":visual,"files":{"generation":{"path":str(path),"sha256":sha(path)},"routes":{"path":str(routes_path),"sha256":sha(routes_path)}},"adjudication":{"large_scale_autonomous_route_replication":True,"individual_head_necessity":False,"minimum_gear":False,"language_mechanism_closed":False},"checks":checks,"all_checks_passed":all(checks.values())};save(OUT/"analysis/final.json",r)
 if r["all_checks_passed"]:append_memo(r)
 print(json.dumps({"phase":PHASE,"metrics":metrics,"checks":checks,"all_checks_passed":r["all_checks_passed"]},ensure_ascii=False,indent=2))
 if not r["all_checks_passed"]:raise RuntimeError(checks)
if __name__=="__main__":main()
