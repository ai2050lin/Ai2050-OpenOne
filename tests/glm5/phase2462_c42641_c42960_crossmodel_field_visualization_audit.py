#!/usr/bin/env python3
"""Publish model-local full-coordinate output-conditioned fields to the client."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT=Path(__file__).resolve().parents[2]; RESULT=ROOT/"tests/glm5/result"; PUBLIC=ROOT/"frontend/public/vis_data/research_kernel"
OUT=RESULT/"phase2462_c42641_c42960_crossmodel_field_visualization_audit"; MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
ASSET=PUBLIC/"c42641_output_conditioned_crossmodel_field.json"; PHASE,CAMPAIGN=2462,"C42641-C42960"
MODELS={
    "qwen4b":{"label":"Qwen3-4B-BF16","dimension":2560,"precision":"BF16"},
    "qwen14b":{"label":"Qwen3-14B-NF4-BF16","dimension":5120,"precision":"NF4 storage / BF16 compute"},
    "glm4":{"label":"GLM4-9B-INT8","dimension":4096,"precision":"INT8 weights"},
    "deepseek7b":{"label":"DeepSeek-R1-Distill-Qwen-7B-INT8","dimension":3584,"precision":"INT8 weights"},
}


def save(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")


def save_if_changed(path:Path,value:Any)->None:
    content=json.dumps(value,ensure_ascii=False,indent=2,default=str)+"\n"
    if not path.exists() or path.read_text(encoding="utf-8")!=content:
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content,encoding="utf-8")


def close(value:Any)->None:
    mmap=getattr(value,"_mmap",None)
    if mmap is not None:mmap.close()


def add(rows:dict[str,list[dict]],model:str,vector:np.ndarray,label:str,source:str,kind:str,preview:bool=False,**meta)->None:
    value=np.asarray(vector,dtype=np.float32).reshape(-1); expected=MODELS[model]["dimension"]
    if value.shape!=(expected,) or not np.isfinite(value).all():raise RuntimeError((model,label,value.shape))
    rows[model].append({"label":label,"source":source,"coordinate_kind":kind,"preview":preview,**meta,"values":[float(x) for x in value]})


def add_reference_states(rows:dict[str,list[dict]])->None:
    q4=json.loads((PUBLIC/"c32561_semantic_encoding_output_field.json").read_text(encoding="utf-8"))
    for kind,layer,label in (("embedding_activation",0,"embedding q0 query-end"),("hidden_state",24,"HiddenState q24 query-end")):
        row=next(r for r in q4["rows"] if r.get("coordinate_kind")==kind and int(r.get("layer",-1))==layer)
        add(rows,"qwen4b",row["values"],label,"phase2436/c32561",kind,True,layer=layer,event="query_end",family="taxonomy",language="en")
    for model in ("qwen14b","glm4","deepseek7b"):
        meta=json.loads((PUBLIC/f"c9721_{model}_natural_multifuture_key_hiddenstate.json").read_text(encoding="utf-8"))
        matrix=np.load(PUBLIC/f"c9721_{model}_natural_multifuture_key_hiddenstate.float16.npy",mmap_mode="r")
        candidates=[(i,r) for i,r in enumerate(meta["rows"]) if r["family"]=="taxonomy" and r["language"]=="en" and int(r["unit"])==5 and r["query"]=="source"]
        embedding=next((i,r) for i,r in candidates if int(r["qpoint"])==0)
        hidden=next((i,r) for i,r in candidates if int(r["qpoint"]) not in (0,max(int(x[1]["qpoint"]) for x in candidates)))
        add(rows,model,matrix[embedding[0]],"embedding q0 source token","c9721_natural_multifuture_key_hiddenstate","embedding_activation",True,layer=0,event="source",family="taxonomy",language="en")
        add(rows,model,matrix[hidden[0]],f"HiddenState q{hidden[1]['qpoint']} source token","c9721_natural_multifuture_key_hiddenstate","hidden_state",True,layer=int(hidden[1]["qpoint"]),event="source",family="taxonomy",language="en")
        close(matrix)


def add_crossmodel_output_fields(rows:dict[str,list[dict]])->None:
    phases={"qwen14b":"phase2454_*","glm4":"phase2455_*","deepseek7b":"phase2456_*"}
    for model,pattern in phases.items():
        directory=next(RESULT.glob(pattern)); final=json.loads((directory/"analysis/final.json").read_text(encoding="utf-8")); families=final["analysis"]["families"]
        passports=np.load(directory/"derived/semantic_lexical_passports.float32.npy",mmap_mode="r")
        for ui,unit in enumerate((4,5),start=1):
            for li,language in enumerate(("en","zh")):
                for fi,family in enumerate(families):
                    add(rows,model,passports[0,1,ui,li,0,fi],f"semantic HxVJP unit{unit} {language} {family}",f"phase{final['phase']}","state_times_gradient",
                        preview=(unit==5 and language=="en" and family=="taxonomy"),layer=final["collection"]["qpoints"][0],event="query_end",family=family,language=language,unit=unit,interaction="semantic_validity")
        close(passports)


def add_q4_autoregressive_and_interface(rows:dict[str,list[dict]])->None:
    p2460=next(RESULT.glob("phase2460_*")); j2460=json.loads((p2460/"analysis/final.json").read_text(encoding="utf-8")); families=j2460["analysis"]["families"]
    passports=np.load(p2460/"derived/two_token_semantic_lexical_passports.float32.npy",mmap_mode="r")
    for step,step_name in enumerate(("first_token","second_token_path_conditioned","two_token_total")):
        for li,language in enumerate(("en","zh")):
            for fi,family in enumerate(families):
                add(rows,"qwen4b",passports[0,step,0,1,li,fi],f"semantic HxVJP {step_name} {language} {family}","phase2460_two_token","state_times_gradient",
                    preview=(language=="en" and family=="taxonomy"),layer=16,event="query_end",family=family,language=language,step=step_name,interaction="semantic_validity")
    for li,language in enumerate(("en","zh")):
        for fi,family in enumerate(families):
            add(rows,"qwen4b",passports[0,2,1,0,li,fi],f"semantic gradient two_token_total {language} {family}","phase2460_two_token","gradient",
                preview=(language=="en" and family=="taxonomy"),layer=18,event="query_end",family=family,language=language,step="two_token_total",interaction="semantic_validity")
    close(passports)
    p2461=next(RESULT.glob("phase2461_*")); j2461=json.loads((p2461/"analysis/final.json").read_text(encoding="utf-8")); combined=np.load(p2461/"derived/three_interface_passports.float32.npy",mmap_mode="r")
    for ii,interface in enumerate(("candidate_entity","bare_entity","letter_code")):
        for li,language in enumerate(("en","zh")):
            for fi,family in enumerate(families):
                add(rows,"qwen4b",combined[0,ii,1,li,0,1,fi],f"semantic HxVJP {interface} unit5 {language} {family}","phase2461_output_interface","state_times_gradient",
                    preview=(language=="en" and family=="taxonomy"),layer=16,event="query_end",family=family,language=language,unit=5,interface=interface,interaction="semantic_validity")
    close(combined)


def build_asset()->dict:
    rows={key:[] for key in MODELS}; add_reference_states(rows); add_crossmodel_output_fields(rows); add_q4_autoregressive_and_interface(rows)
    sections=[]; binary_info={}
    for key,definition in MODELS.items():
        matrix=np.stack([np.asarray(row["values"],dtype=np.float32) for row in rows[key]])
        binary=PUBLIC/f"c42641_{key}_output_conditioned_field.float32.npy"; np.save(binary,matrix)
        digest=hashlib.sha256(binary.read_bytes()).hexdigest(); binary_info[key]={"path":str(binary),"shape":list(matrix.shape),"bytes":binary.stat().st_size,"sha256":digest}
        sections.append({"key":key,"model":definition["label"],"precision":definition["precision"],"coordinate_count":definition["dimension"],
            "coordinate_semantics":"model-local embedding/HiddenState activation, output-conditioned gradient, or HxVJP; see each row kind", "coordinate_order":"original model-local physical coordinate order; no Top-K/PCA/compression/cross-model coordinate alignment",
            "binary_url":f"/vis_data/research_kernel/{binary.name}","binary_shape":list(matrix.shape),"binary_dtype":"float32","binary_sha256":digest,"rows":rows[key]})
    payload={"schema":"c42641.output_conditioned_crossmodel_field.v1","phase":PHASE,"campaign":"C39761-C42640","result_type":"output_conditioned_crossmodel_field_heatmap",
        "title":"Cross-model embedding, HiddenState, output-conditioned VJP, two-token and interface field","models":sections,
        "summary":{"model_rows":{key:len(rows[key]) for key in MODELS},"total_rows":sum(map(len,rows.values())),"phase2457_crossmodel_relation_universal":False,
            "phase2458_even_curvature_resolved":False,"phase2460_two_token_finite_lockbox":True,"phase2461_output_identity_decoupled":False},
        "claim_boundary":"Every row retains its model's complete native physical coordinates. Coordinate IDs and amplitudes are not compared across architectures or precisions. Qwen14B is NF4/BF16; GLM4 and DS7B are INT8. DS passes only two behavior families, stable BF16 curvature was not resolved, and the A/B interface has one behavior-qualified family. These fields are observations/analyst VJPs, not universal physical gears or a closed language mechanism."}
    save_if_changed(ASSET,payload)
    return {"asset":str(ASSET),"json_bytes":ASSET.stat().st_size,"models":binary_info,"rows":payload["summary"]["model_rows"],"total_rows":payload["summary"]["total_rows"]}


def retention_manifest()->dict:
    targets=[]
    for phase in (2454,2455,2456):
        directory=next(RESULT.glob(f"phase{phase}_*")); targets.extend((directory/"raw/output_conditioned_fields.float32.npy",directory/"derived/semantic_lexical_passports.float32.npy"))
    targets.extend((next(RESULT.glob("phase2460_*"))/"raw/two_token_query_fields.float32.npy",next(RESULT.glob("phase2460_*"))/"derived/two_token_semantic_lexical_passports.float32.npy",
                    next(RESULT.glob("phase2461_*"))/"raw/new_interface_fields.float32.npy",next(RESULT.glob("phase2461_*"))/"derived/three_interface_passports.float32.npy"))
    records=[]
    for path in targets:
        digest=hashlib.sha256(path.read_bytes()).hexdigest(); records.append({"path":str(path),"bytes":path.stat().st_size,"sha256":digest,"retention":"retained; unique full-coordinate evidence represented by the client asset"})
    save(OUT/"analysis/retention_manifest.json",records)
    return {"files":len(records),"bytes":sum(r["bytes"] for r in records),"all_hashes":all(len(r["sha256"])==64 for r in records),"cleanup":"No HiddenState field deleted because all are unique evidence and represented in the client; Phase2458/2459 contain only compact margins/masks."}


def frontend_contract()->dict:
    route=(ROOT/"frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8-sig"); hook=(ROOT/"frontend/src/researchKernel/useResearchKernel.js").read_text(encoding="utf-8-sig"); component=(ROOT/"frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8-sig"); app=(ROOT/"frontend/src/App.jsx").read_text(encoding="utf-8-sig"); dist=ROOT/"frontend/dist/index.html"
    return {"route": "C42641_OUTPUT_CONDITIONED_CROSSMODEL_FIELD_ROUTE" in route,"hook":"setC42641OutputConditionedCrossmodelField" in hook,
            "four_panel_component":"buildC42641CrossmodelFieldData" in component and "panel.coordinateCount" in component,"app_prop":"c42641OutputConditionedCrossmodelField=" in app,
            "dist_exists":dist.exists(),"dist_newer_than_asset":dist.exists() and dist.stat().st_mtime_ns>=ASSET.stat().st_mtime_ns}


def append_memo(result:dict)->None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text=rf"""


## Phase {PHASE}: 四模型原生坐标输出条件场、两token与三接口热力图发布审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 将Phase2454–2456的Qwen14B/GLM4/DS八族中英、unit4/5语义$H\odot g$，Phase2460的第一token/路径条件第二token/两token总和$H\odot g$与总gradient，以及Phase2461三输出接口fresh $H\odot g$发布为新热力图类型。每个模型保留自己的原生物理坐标轴（2560/5120/4096/3584），不补零对齐、不Top-K替代原场。每模型还从已留存C32561/C9721资产加入真实词嵌入q0与HiddenState代表行，使激活、gradient和归因可在同一模型面板逐坐标查看。

$$\mathcal V_M=\{{E^M,H^M,g^M,H^M\odot g^M\}}\subset\mathbb R^{{d_M}},\qquad(d_M)_M=(2560,5120,4096,3584).$$

**结果汇总。** 资产 `{json.dumps(result['asset'],ensure_ascii=False)}`；客户端合同/构建 `{json.dumps(result['frontend'],ensure_ascii=False)}`；原场留存与清理裁决 `{json.dumps(result['retention'],ensure_ascii=False)}`；检查 `{json.dumps(result['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2462_c42641_c42960_crossmodel_field_visualization_audit.py`；主资产`frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json`，四个float32原生维度二进制同目录；路由、加载hook、四面板3D预览更新于`heatmapResearchRoute.js`、`useResearchKernel.js`、`ResearchHeatmapRoute.jsx`和`App.jsx`；final与SHA256清单位于同名结果目录。

**分析与理论进展。** 客户端现在不会把跨架构坐标号伪装成同一基底：每个面板独立取Top坐标或展示全部原生坐标。它可直接观察真实embedding/HiddenState与输出条件归因如何不同，也把第二token路径和输出接口改变后的纹理并列，服务于继续积累条件坐标齿轮图谱。

**问题硬伤与结论。** 可视化代表行不是统计门本身，完整裁决仍以Phase2454–2461结果为准。量化模型幅值不可横比；VJP是分析者求导量。补充更正Phase2461自动标签：三接口几何无条件通过，但A/B接口仅coreference_binding在unit4/5均达0.75有效胜率，因此行为门控三接口语义锁箱失败；输出token身份解耦仍未成立。所有独特全坐标场已进入客户端或由其代表并保留，因此不删除；小型margin/掩码数据不属于HiddenState大场。
"""
    with MEMO.open("a",encoding="utf-8",newline="\n") as handle:handle.write(text)


def main()->None:
    asset=build_asset(); retention=retention_manifest(); frontend=frontend_contract()
    checks={"four_native_dimensions":sorted(v["shape"][1] for v in asset["models"].values())==[2560,3584,4096,5120],"all_binary_hashes":all(len(v["sha256"])==64 for v in asset["models"].values()),
            "rows":asset["total_rows"]>=150,"frontend_source":frontend["route"] and frontend["hook"] and frontend["four_panel_component"] and frontend["app_prop"],
            "frontend_built":frontend["dist_newer_than_asset"],"retained":retention["files"]==10 and retention["all_hashes"]}
    result={"phase":PHASE,"campaign":CAMPAIGN,"asset":asset,"frontend":frontend,"retention":retention,"checks":checks,"all_checks_passed":all(checks.values())}
    if result["all_checks_passed"]:save(OUT/"analysis/final.json",result);append_memo(result)
    else:save(OUT/"analysis/prebuild.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result["all_checks_passed"]:raise RuntimeError(checks)


if __name__=="__main__":main()
