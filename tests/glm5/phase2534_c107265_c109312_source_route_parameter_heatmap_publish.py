#!/usr/bin/env python3
"""Publish source/KV/route/necessity/autonomous full-coordinate rows to the existing c42641 client heatmap."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT=Path(__file__).resolve().parents[2];RESULT=ROOT/"tests/glm5/result"
P2529=RESULT/"phase2529_c95777_c97568_full_source_kv_head_residual_ledger";P2530=RESULT/"phase2530_c97569_c99072_source_edge_sufficiency_lockbox";P2531=RESULT/"phase2531_c99073_c101120_redundant_route_cuts_rescue";P2532=RESULT/"phase2532_c101121_c_c104192_autonomous_multifamily_recursive_generation";P2533=RESULT/"phase2533_c104193_c107264_crossmodel_local_route_replication"
# Preserve compatibility with the actual Phase2532 directory name.
P2532=RESULT/"phase2532_c101121_c104192_autonomous_multifamily_recursive_generation"
OUT=RESULT/"phase2534_c107265_c109312_source_route_parameter_heatmap_publish";ASSET=ROOT/"frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE,CAMPAIGN=2534,"C107265-C109312";SOURCE="phase2534_source_route_parameter"

def load(p:Path)->Any:return json.loads(p.read_text(encoding="utf-8-sig"))
def read(p:Path)->list[dict]:return[json.loads(x)for x in p.read_text(encoding="utf-8-sig").splitlines()if x.strip()]
def save(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb")as f:
  for b in iter(lambda:f.read(16*1024*1024),b""):h.update(b)
 return h.hexdigest()
def row(values,label,kind,**meta):
 v=np.asarray(values,np.float32).reshape(-1)
 if not np.isfinite(v).all():raise RuntimeError(label)
 return{"label":label,"source":SOURCE,"coordinate_kind":kind,"preview":True,**meta,"values":[float(x)for x in v]}

def publish()->dict:
 f29=load(P2529/"analysis/final.json");f30=load(P2530/"analysis/final.json");f31=load(P2531/"analysis/final.json");f32=load(P2532/"analysis/final.json");f33=load(P2533/"analysis/final.json");payload=load(ASSET)
 sections={s["key"]:s for s in payload["models"]};q4=sections["qwen4b"];q4["rows"]=[r for r in q4["rows"]if r.get("source")!=SOURCE]
 residual=np.load(f29["fields"]["residual_source_interaction"]["path"],mmap_mode="r")
 for ui,unit in enumerate((30,31)):
  for li,layer in enumerate(range(20,36)):
   q4["rows"].append(row(np.sqrt(np.mean(np.asarray(residual[ui,:,:,li],np.float64)**2,axis=(0,1,2))),f"unit{unit} layer{layer} all-source Attention residual Walsh RMS","source_residual_layer",phase=2529,unit=unit,layer=layer,averaging="9 families x 2 languages x 5 exhaustive regions"))
  for ri,region in enumerate(f29["scope"]["regions"]):
   q4["rows"].append(row(np.sqrt(np.mean(np.asarray(residual[ui,:,:,:,ri],np.float64)**2,axis=(0,1,2))),f"unit{unit} {region} residual-coordinate Walsh RMS","source_residual_region",phase=2529,unit=unit,region=region))
 cut=np.load(f31["fields"]["hidden"]["path"],mmap_mode="r");conditions=f31["fields"]["hidden"]["conditions"];chosen=("edge_cut_top_external","edge_cut_random_external","edge_cut_all_late_external","edge_cut_all_late_external_rescue_top","edge_cut_all_late_external_rescue_random","head_zero_top","head_zero_complement","head_zero_all")
 for condition in chosen:
  ci=conditions.index(condition)
  for qi,qpoint in enumerate(range(20,37)):
   q4["rows"].append(row(np.sqrt(np.mean((np.asarray(cut[ci,:,qi],np.float64)-np.asarray(cut[0,:,qi],np.float64))**2,axis=0)),f"{condition} q{qpoint} HiddenState intervention RMS","intervention_hidden_state",phase=2531,condition=condition,qpoint=qpoint,unit=31))
 auto_h=np.load(f32["fields"]["hidden"]["path"],mmap_mode="r");auto_e=np.load(f32["fields"]["embedding"]["path"],mmap_mode="r");auto_t=np.load(f32["fields"]["tokens"]["path"],mmap_mode="r")
 for step in range(4):
  q4["rows"].append(row(auto_e[0,step],f"autonomous sample0 step{step} input-token embedding","embedding",phase=2532,sample=0,step=step,generated_token_id=int(auto_t[0,step])))
  for qi,qpoint in enumerate(range(20,37)):
   q4["rows"].append(row(auto_h[0,step,qi],f"autonomous sample0 step{step} q{qpoint} HiddenState","hidden_state",phase=2532,sample=0,step=step,qpoint=qpoint,generated_token_id=int(auto_t[0,step])))
 # Update Qwen3-4B head panel with contribution coordinates, intervention masks, and recursive use.
 head_section=sections["qwen4b_attention_heads"];head_section["rows"]=[r for r in head_section["rows"]if r.get("source")!=SOURCE]
 hs=np.load(f29["fields"]["head_source_interaction"]["path"],mmap_mode="r")
 for ui,unit in enumerate((30,31)):
  for li,layer in enumerate(range(20,36)):
   head_section["rows"].append(row(np.sqrt(np.mean(np.asarray(hs[ui,:,:,li],np.float64)**2,axis=(0,1,3,4))),f"unit{unit} layer{layer} source-contribution head RMS","head_source_contribution",phase=2529,unit=unit,layer=layer))
  for ri,region in enumerate(f29["scope"]["regions"]):
   head_section["rows"].append(row(np.sqrt(np.mean(np.asarray(hs[ui,:,:,:,:,ri],np.float64)**2,axis=(0,1,2,4))),f"unit{unit} {region} contribution head RMS","head_source_region",phase=2529,unit=unit,region=region))
 top={(x["layer"],x["head"])for x in f29["routes"]["top"]};rnd={(x["layer"],x["head"])for x in f29["routes"]["random"]}
 for layer in range(20,36):
  head_section["rows"].append(row([1 if(layer,h)in top else 0 for h in range(32)],f"layer{layer} contribution-top32 membership","head_contribution_top_mask",phase=2529,layer=layer))
  head_section["rows"].append(row([1 if(layer,h)in rnd else 0 for h in range(32)],f"layer{layer} contribution-random32 membership","head_contribution_random_mask",phase=2529,layer=layer))
 auto_head=np.load(f32["fields"]["head"]["path"],mmap_mode="r")
 for step in range(auto_head.shape[1]):head_section["rows"].append(row(np.sqrt(np.mean(np.asarray(auto_head[:,step],np.float64)**2,axis=(0,2))),f"autonomous step{step} frozen-route head RMS","autonomous_route_head",phase=2532,step=step))
 head_section["coordinate_orders"]["event_path"]=[int(v)for v in np.argsort(-np.mean(np.asarray([r["values"]for r in head_section["rows"]if r["source"]==SOURCE],np.float64)**2,axis=0))]
 # Add physical K/V head-coordinate panel.
 key=np.load(f29["fields"]["key"]["path"],mmap_mode="r");value=np.load(f29["fields"]["value"]["path"],mmap_mode="r");index_rows=read(Path(f29["fields"]["index"]["path"]));kvrows=[]
 for li,layer in enumerate(range(20,36)):
  for kh in range(key.shape[2]):
   kvrows.append(row(np.sqrt(np.nanmean(np.asarray(key[:,li,kh],np.float64)**2,axis=(0,1))),f"layer{layer} KV-head{kh} post-RoPE K coordinate RMS","key_coordinate_rms",phase=2529,layer=layer,kv_head=kh))
   kvrows.append(row(np.sqrt(np.nanmean(np.asarray(value[:,li,kh],np.float64)**2,axis=(0,1))),f"layer{layer} KV-head{kh} V coordinate RMS","value_coordinate_rms",phase=2529,layer=layer,kv_head=kh))
 query_positions=index_rows[0]["region_positions"]["query_property"]
 query_token=query_positions[-1] if query_positions else index_rows[0]["region_positions"]["post_query"][0]
 for layer in (20,28,35):
  li=layer-20
  for kh in range(key.shape[2]):
   kvrows.append(row(key[0,li,kh,query_token],f"sample0 query token layer{layer} KV-head{kh} exact post-RoPE K","key_token_coordinate",phase=2529,sample=0,token=query_token,layer=layer,kv_head=kh))
   kvrows.append(row(value[0,li,kh,query_token],f"sample0 query token layer{layer} KV-head{kh} exact V","value_token_coordinate",phase=2529,sample=0,token=query_token,layer=layer,kv_head=kh))
 kvenergy=np.mean(np.asarray([r["values"]for r in kvrows],np.float64)**2,axis=0)
 kvsection={"key":"qwen4b_kv_coordinates","model":"Qwen3-4B late K/V head coordinates","precision":"BF16 capture / float16 field","coordinate_count":int(key.shape[-1]),"coordinate_semantics":"model-local physical coordinate inside each K/V head","coordinate_order":"physical K/V head-coordinate 0-127","rows":kvrows,"coordinate_orders":{"event_path":[int(v)for v in np.argsort(-kvenergy)]},"coordinate_order_semantics":{"event_path":"late K/V coordinate RMS energy"}}
 # Add one model-local head-index panel per cross-model replication.
 new_sections=[]
 names={"qwen14b":"Qwen3-14B local late heads","deepseek7b":"DeepSeek7B local late heads","glm4":"GLM-4 local late heads"}
 for key_name in ("qwen14b","deepseek7b","glm4"):
  fm=f33["models"][key_name];inter=np.load(fm["fields"]["interaction"]["path"],mmap_mode="r");routes=fm["selection"]["routes"];nheads=routes["heads"];late=list(range(routes["late_layers"][0],routes["late_layers"][1]+1));rows=[]
  top={(x["layer"],x["head"])for x in routes["top"]};rnd={(x["layer"],x["head"])for x in routes["random"]}
  for li,layer in enumerate(late):
   rows.append(row(np.sqrt(np.mean(np.asarray(inter[:,:,li],np.float64)**2,axis=(0,1,3))),f"unit30 layer{layer} head-output Walsh RMS","crossmodel_head_energy",phase=2533,model_key=key_name,layer=layer))
   rows.append(row([1 if(layer,h)in top else 0 for h in range(nheads)],f"layer{layer} frozen top membership","crossmodel_top_mask",phase=2533,model_key=key_name,layer=layer))
   rows.append(row([1 if(layer,h)in rnd else 0 for h in range(nheads)],f"layer{layer} equal random membership","crossmodel_random_mask",phase=2533,model_key=key_name,layer=layer))
  energy=np.mean(np.asarray([r["values"]for r in rows if r["coordinate_kind"]=="crossmodel_head_energy"],np.float64)**2,axis=0)
  new_sections.append({"key":f"{key_name}_late_heads","model":names[key_name],"precision":"BF16 nonquantized","coordinate_count":nheads,"coordinate_semantics":f"{key_name} model-local query-head index","coordinate_order":f"physical query-head 0-{nheads-1}","rows":rows,"coordinate_orders":{"event_path":[int(v)for v in np.argsort(-energy)]},"coordinate_order_semantics":{"event_path":"unit30 head-output Walsh RMS"}})
 payload["models"]=[s for s in payload["models"]if s.get("key")not in{"qwen4b_kv_coordinates","qwen14b_late_heads","deepseek7b_late_heads","glm4_late_heads"}]+[kvsection]+new_sections
 payload["phase"]=PHASE;payload["campaign"]="C39761-C109312";payload["title"]="Output-conditioned, source-route, K/V, autonomous, and cross-model physical coordinate field"
 payload["summary"].update({"phase2529_unit_energy_spearman":f29["routes"]["unit30_unit31_energy_spearman"],"phase2530_source_top_donor_flip":f30["causal"]["donor_top_external"]["donor_flip_rate"],"phase2531_top_edge_cut_accuracy":f31["causal"]["edge_cut_top_external"]["accuracy"],"phase2531_all_edge_cut_top_rescue_accuracy":f31["causal"]["edge_cut_all_late_external_rescue_top"]["accuracy"],"phase2532_autonomous_top_zero_accuracy":f32["causal"]["main"]["head_zero_top"]["accuracy"],"phase2532_autonomous_random_zero_accuracy":f32["causal"]["main"]["head_zero_random"]["accuracy"],"phase2533_model_local_route_advantages":f33["adjudication"]["model_local_advantages"]})
 payload["summary"]["model_rows"]={s["key"]:len(s["rows"])for s in payload["models"]};payload["summary"]["total_rows"]=sum(payload["summary"]["model_rows"].values())
 sentence=" Phase2529-2533 add exhaustive source contributions, physical K/V coordinates, intervention HiddenState trajectories, autonomous token-boundary embedding/HiddenState/head values, and three independently selected BF16 model-local head panels. Group effects do not imply individual-head necessity or cross-model coordinate identity."
 if sentence.strip()not in payload["claim_boundary"]:payload["claim_boundary"]+=sentence
 content=json.dumps(payload,ensure_ascii=False,indent=2)+"\n"
 if ASSET.read_text(encoding="utf-8")!=content:ASSET.write_text(content,encoding="utf-8")
 return{"path":str(ASSET),"sha256":sha(ASSET),"bytes":ASSET.stat().st_size,"sections":len(payload["models"]),"rows":payload["summary"]["model_rows"],"coordinate_counts":{s["key"]:s["coordinate_count"]for s in payload["models"]}}

def append_memo(r):
 if f"## Phase {PHASE}:"in MEMO.read_text(encoding="utf-8"):return
 stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
 text=rf"""


## Phase {PHASE}: source/KV/冗余割/自主递归与跨模型head参数级热力图发布（{CAMPAIGN}） [{stamp}]

**测试原理与显示内容。** 扩展既有c42641客户端热力图而不新建Markdown：Qwen3-4B 2560轴新增五source的Attention residual交互、22条件中关键割/救援的q20–q36 HiddenState坐标RMS、同一自主样本四个生成步的真实词嵌入与HiddenState；32-head轴新增全source贡献、贡献top/random membership与八步自主route值；新增128维K/V-head坐标面板，显示全部layer20–35×8 KV heads的K/V coordinate RMS和具体样本query token的post-RoPE K/V值；Qwen14B、DS7B、GLM4分别新增模型本地head轴。前端布局扩为最多四列，任意面板的物理坐标编号都不跨模型/轴种类对齐。

$$g_{{lr,i}}=\left[W_{{O,l}}\operatorname{{concat}}_h\sum_{{j\in r}}\alpha_{{lhaj}}v_{{lhj}}\right]_i,$$

**结果汇总。** 资产 `{json.dumps(r['asset'],ensure_ascii=False)}`；前端 `{json.dumps(r['frontend'],ensure_ascii=False)}`；留存 `{json.dumps(r['retention'],ensure_ascii=False)}`；检查 `{json.dumps(r['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2534_c107265_c109312_source_route_parameter_heatmap_publish.py`；更新`frontend/src/components/app/ResearchHeatmapRoute.jsx`、`frontend/src/researchKernel/heatmapResearchRoute.js`和c42641 JSON资产；生产build、哈希和final位于`{OUT}`。

**分析与理论进展。** 客户端现在能同时检查词嵌入、完整HiddenState物理坐标、K/V head内坐标、query-head编号、source residual写入、删除恢复轨迹与自主递归，不再把attention质量当作唯一“从哪里读”证据。跨模型各自有物理轴，显示相对结构而不伪造共享坐标。

**问题硬伤与结论。** 大部分图行是跨样本RMS，用于全坐标图谱而不是方向因果；具体sample行不能代表总体；K/V坐标聚合不直接显示query-key点积；top membership只是冻结选择；客户端显示不提升证据等级。所有重要原场均作为显示来源留存并记录哈希；没有未显示HiddenState场需要删除。
"""
 with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)

def main():
 sources=[load(p/"analysis/final.json")for p in(P2529,P2530,P2531,P2532,P2533)];asset=publish();route=(ROOT/"frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8");kernel=(ROOT/"frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8");dist=ROOT/"frontend/dist/index.html"
 displayed=[]
 for f in sources:
  for v in f.get("fields",{}).values():
   if isinstance(v,dict)and v.get("path")and Path(v["path"]).exists():displayed.append({"path":v["path"],"bytes":Path(v["path"]).stat().st_size,"sha256":sha(Path(v["path"])),"retention":"display source or parameter-level derived source"})
 retention={"displayed_source_files":displayed,"bytes":sum(x["bytes"]for x in displayed),"all_hashed":all(len(x["sha256"])==64 for x in displayed),"unpublished_hiddenstate_deleted":[],"reason":"all new HiddenState fields are represented by parameter-level client rows"};save(OUT/"analysis/retention_manifest.json",retention)
 frontend={"dynamic_eight_panel_layout":"densePanelLayout"in route,"updated_boundary":"Phase2529-2532"in kernel,"dist_exists":dist.exists(),"dist_newer":dist.exists()and dist.stat().st_mtime_ns>=ASSET.stat().st_mtime_ns}
 checks={"sources_passed":all(x["all_checks_passed"]for x in sources),"eight_sections":asset["sections"]==9,"qwen_residual_coordinates":asset["coordinate_counts"]["qwen4b"]==2560,"kv_coordinates":asset["coordinate_counts"]["qwen4b_kv_coordinates"]==128,"model_local_heads":all(k in asset["coordinate_counts"]for k in("qwen14b_late_heads","deepseek7b_late_heads","glm4_late_heads")),"frontend_layout":frontend["dynamic_eight_panel_layout"],"frontend_build":frontend["dist_newer"],"retention":retention["all_hashed"],"claim_boundary":True}
 r={"phase":PHASE,"campaign":CAMPAIGN,"asset":asset,"frontend":frontend,"retention":retention,"checks":checks,"all_checks_passed":all(checks.values())};save(OUT/"analysis"/("final.json"if r["all_checks_passed"]else"prebuild.json"),r)
 if r["all_checks_passed"]:append_memo(r)
 print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
