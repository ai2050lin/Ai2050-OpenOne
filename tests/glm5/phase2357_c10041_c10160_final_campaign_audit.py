#!/usr/bin/env python3
"""Final evidence, visualization provenance, cleanup, and next-stage audit for Phase2351-2356."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result"
OUT=RESULT/"phase2357_c10041_c10160_final_campaign_audit";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
VIS=ROOT/"frontend/public/vis_data/research_kernel";CATALOG=ROOT/"frontend/public/research_data/current/language_encoding_catalog.json"
PHASE=2357;CAMPAIGN="C10041-C10160"
PHASE_DIRS={
2351:"phase2351_c9121_c9240_evidence_generation_audit",2352:"phase2352_c9241_c9400_natural_multifuture_transient_field",
2353:"phase2353_c9401_c9560_conditional_equivalence_route_competition",2354:"phase2354_c9561_c9720_norm_preserving_conditional_causality",
2355:"phase2355_c9721_c9880_crossmodel_natural_equivalence",2356:"phase2356_c9881_c10040_generation_success_coordinate_diagnostic"}
DATASETS={
"c9241_qwen4b_natural_multifuture_prompt_field":(2352,"C9241-C9400"),"c9242_qwen4b_natural_generation_token_trajectory":(2352,"C9241-C9400"),
"c9401_qwen4b_conditional_equivalence_prompt_passport":(2353,"C9401-C9560"),"c9402_qwen4b_generation_conditioned_equivalence_passport":(2353,"C9401-C9560"),
"c9561_qwen4b_norm_preserving_family_directions":(2354,"C9561-C9720"),
"c9721_qwen14b_natural_multifuture_key_hiddenstate":(2355,"C9721-C9880"),"c9721_glm4_natural_multifuture_key_hiddenstate":(2355,"C9721-C9880"),
"c9721_deepseek7b_natural_multifuture_key_hiddenstate":(2355,"C9721-C9880"),"c9881_qwen4b_generation_success_coordinate_passport":(2356,"C9881-C10040")}

sys.path.insert(0,str(TESTS))
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa:E402

if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8",errors="replace")

def save(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        while chunk:=handle.read(8*1024*1024):digest.update(chunk)
    return digest.hexdigest()

def phase_audit()->dict:
    memo=MEMO.read_text(encoding="utf-8");records={}
    for phase,directory in PHASE_DIRS.items():
        path=RESULT/directory/"analysis/final.json";value=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        records[str(phase)]={"final_exists":path.exists(),"memo_heading_count":len(re.findall(rf"^## Phase {phase}:",memo,flags=re.MULTILINE)),
                             "engineering_checks":value.get("all_checks_passed")}
    return {"continuous":sorted(PHASE_DIRS)==list(range(2351,2357)),"records":records,
            "all_final":all(v["final_exists"] for v in records.values()),"all_memo_once":all(v["memo_heading_count"]==1 for v in records.values()),
            "all_engineering_checks":all(v["engineering_checks"] for v in records.values())}

def repair_and_verify()->tuple[dict,list[dict]]:
    catalog=json.loads(CATALOG.read_text(encoding="utf-8"));catalog_changed=[];metadata_changed=[]
    for row in catalog.get("datasets",[]):
        if row.get("id") in DATASETS:
            phase,campaign=DATASETS[row["id"]]
            if row.get("phase")!=phase or row.get("campaign")!=campaign:
                row["phase"],row["campaign"]=phase,campaign;catalog_changed.append(row["id"])
    CATALOG.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    verification=[]
    for dataset_id,(phase,campaign) in DATASETS.items():
        path=VIS/f"{dataset_id}.json";meta=json.loads(path.read_text(encoding="utf-8"))
        if meta.get("phase")!=phase or meta.get("campaign")!=campaign:
            meta["phase"],meta["campaign"]=phase,campaign;path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");metadata_changed.append(dataset_id)
        binary=ROOT/"frontend/public"/meta["binary_url"].lstrip("/")
        verification.append({"id":dataset_id,"phase":phase,"campaign":campaign,"coordinate_count":meta["coordinate_count"],
                             "binary_exists":binary.exists(),"sha256":binary.exists() and sha256(binary)==meta["binary_sha256"]})
    return {"catalog_changed":catalog_changed,"metadata_changed":metadata_changed,"all_present":all(any(r.get("id")==i for r in catalog["datasets"]) for i in DATASETS)},verification

def cleanup_audit()->dict:
    retained=[]
    for phase,directory in PHASE_DIRS.items():
        base=RESULT/directory
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in (".npy",".npz") and re.search(r"hidden|state|trajectory",path.name,re.I):
                retained.append({"phase":phase,"path":str(path),"bytes":path.stat().st_size})
    return {"unvisualized_raw_hiddenstate_files":retained,"count":len(retained),
            "reclaimed_bytes_documented":21516779776+796917888+896532608+1195376768+330301568+264241280+165150848,
            "note":"Published visualization binaries are intentionally retained; decisions, intervention outcomes and compact directions are not raw HiddenState fields."}

def evidence()->dict:
    p51=json.loads((RESULT/PHASE_DIRS[2351]/"analysis/final.json").read_text(encoding="utf-8"));p52=json.loads((RESULT/PHASE_DIRS[2352]/"analysis/final.json").read_text(encoding="utf-8"));
    p53=json.loads((RESULT/PHASE_DIRS[2353]/"analysis/final.json").read_text(encoding="utf-8"));p54=json.loads((RESULT/PHASE_DIRS[2354]/"analysis/final.json").read_text(encoding="utf-8"));
    p55=json.loads((RESULT/PHASE_DIRS[2355]/"analysis/final.json").read_text(encoding="utf-8"));p56=json.loads((RESULT/PHASE_DIRS[2356]/"analysis/final.json").read_text(encoding="utf-8"))
    return {"retained_results":[
        f"Phase2350 raw exact=0 decomposes into {p51['generation']['metrics']['categories']}; first-line semantic identifier accuracy={p51['generation']['metrics']['first_line_exact']:.6f}.",
        f"Phase2352 6144-row teacher-forced preference={p52['behavior']['teacher_forced_overall']:.6f}, but autonomous first-line exact={p52['behavior']['generation']['first_line_exact']:.6f}.",
        f"Phase2353 prompt q{p53['prompt']['selected_qpoint']} signed residual lockbox min={p53['prompt']['lockbox']['minimum_accuracy']:.6f}, sorted={p53['prompt']['sorted_lockbox']['minimum_accuracy']:.6f}.",
        "Phase2353 generation optimum is step0; all actual post-first-token routes fail the minimum transfer gate.",
        f"Phase2354 norm-preserving causal candidate={p54['analysis']['gate']['causal_candidate_passed']}; family selectivity={p54['analysis']['gate']['family_selectivity_count']}/12.",
        f"Phase2355 cross-model summary={json.dumps(p55['summary'],ensure_ascii=False)}.",
        f"Phase2356 coordinate BA={p56['analysis']['lockbox']['balanced_accuracy']:.6f} but query-only BA={p56['analysis']['nuisance_baselines']['query_only_balanced_accuracy']:.6f}; universal marker=false."],
        "corrected_overclaims":["natural_generation_pass was only a prefix gate, not raw exact closure.",
        "The step0 generation atlas is an output-preparation boundary, not a self-generated-history transient.",
        "Phase2356 apparent generation-success marker is weaker than the query-only nuisance baseline.",
        "Qwen14B/GLM strict descriptive replication and DeepSeek partial replication do not imply common coordinates, common depth, a manifold or an algebra.",
        "Norm preservation removes one RMSNorm scale confound but its failed selectivity/rescue gates provide no causal gear."],
        "current_theory":"A robust family-conditioned, coordinate-address-dependent prompt preparation field exists across several controlled domains and two cross-model strict replications. It does not persist as a demonstrated post-token invariant, does not selectively control output under current interventions, and does not yet explain whether a known answer will be realized during free generation.",
        "next_stage":{"same_goal":True,"title":"Query-balanced natural generation realization atlas",
        "required_design":["Calibrate per query and language so every cell has comparable success/failure prevalence; do not reuse deterministic easy source/hard first cells.",
        "Use natural entity names and independent prompt authors/templates, plus explicit stop-token handling and semantic scoring separated from exact formatting.",
        "Freeze train/selection/lockbox across both units and whole families; require a coordinate model to beat query, language, length, target-token-frequency and teacher-margin baselines.",
        "Capture prompt boundary plus first divergence token, but call post-token analyses retrospective unless predicted before generation.",
        "Only a prospectively superior marker should receive matched norm-preserving intervention; otherwise continue atlas accumulation rather than forcing closure."]}}

def append_memo(result:dict)->None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text=rf"""

## Phase {PHASE}: 多未来自然生成大阶段总证据、可视化来源与清理审计（{CAMPAIGN}） [{stamp}]

**测试原理、测试用例与公式。** 本Phase不新增模型主张，而对Phase2351–2356的`final.json`、MEMO连续性、客户端二进制SHA256、phase/campaign来源和未发布HiddenState残留逐项审计。再次严格区分teacher forcing、第一行语义正确、raw exact、prompt准备态、生成后瞬态与因果门；并以最简单的query-only基线重审Phase2356。

$$
\mathcal E_{{prompt-atlas}}\not\Rightarrow\mathcal E_{{generation-trajectory}}\not\Rightarrow\mathcal E_{{causal}},
\qquad BA_{{coord}}=0.7255<BA_{{query-only}}=0.9406.
$$

**结果汇总。** Phase审计 `{json.dumps(result['phase_audit'],ensure_ascii=False)}`；证据裁决 `{json.dumps(result['evidence'],ensure_ascii=False)}`；可视化来源与哈希 `{json.dumps(result['provenance'],ensure_ascii=False)}`、`{json.dumps(result['verification'],ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'],ensure_ascii=False)}`；前端 `{json.dumps(result['frontend_build'],ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2357_c10041_c10160_final_campaign_audit.py`；结果 `tests/glm5/result/phase2357_c10041_c10160_final_campaign_audit`；MEMO为唯一新增Markdown记录。

**理论进展、问题硬伤与结论。** 可保留的是：行为合格域中存在依赖具体坐标地址的prompt准备图谱，q24有符号残差在12族变深度材料上锁箱复现，Qwen14B与GLM4功能门复现、DeepSeek部分复现。必须否定的是：完整自然生成闭包、step>0瞬态等价、选择性因果齿轮、普遍生成成败标志、共同跨模型层拓扑或新数学闭合。核心硬伤已从“能否识别族”转为“受控概率偏好为何不能稳定兑现为自主输出”，且现有成败被查询难度严重混淆。

**下一阶段判断。** 总目标相同，且本轮已按要求自动追加并完成Phase2356；它反而发现query-only基线击败坐标模型。下一完整大阶段必须先建立query×language每格成败近似配平的自然生成合同，加入长度、词频、teacher margin和模板基线，只有输出前坐标在whole-family锁箱上前瞻击败全部基线，才重启因果实验。该结论是新的材料设计边界，不是停止研究，也不授权流形/范畴等叙事。
"""
    with MEMO.open("a",encoding="utf-8",newline="\n") as h:h.write(text)

def main()->None:
    final=OUT/"analysis/final.json"
    if final.exists():
        result=json.loads(final.read_text(encoding="utf-8"));append_memo(result);print(json.dumps(result,ensure_ascii=False,indent=2));return
    audit=phase_audit();provenance,verification=repair_and_verify();cleanup=cleanup_audit();evid=evidence();build=atlas.frontend_build()
    checks={"continuous":audit["continuous"] and audit["all_final"] and audit["all_memo_once"],"engineering":audit["all_engineering_checks"],
            "assets":provenance["all_present"] and all(v["sha256"] for v in verification),"no_unpublished_raw_hiddenstate":cleanup["count"]==0,"frontend_build":build["passed"]}
    result={"phase":PHASE,"campaign":CAMPAIGN,"phase_audit":audit,"evidence":evid,"provenance":provenance,"verification":verification,
            "cleanup":cleanup,"frontend_build":build,"checks":checks,"all_checks_passed":all(checks.values())}
    save(final,result)
    if not result["all_checks_passed"]:raise RuntimeError(("phase2357_failed",checks))
    append_memo(result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
