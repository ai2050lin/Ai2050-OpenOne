#!/usr/bin/env python3
"""Freeze unseen-unit material and gates for output identity and autonomous paths."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"
OUT=RESULT/"phase2463_c42961_c43280_autonomous_output_identity_contract"; MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE,CAMPAIGN=2463,"C42961-C43280"
FAMILIES=("taxonomy","causal","temporal","negation_scope","preposition_role","coreference_binding","punctuation_attachment","sentence_reordering")
TRIPLES={6:("Sela","Tovin","Umera"),7:("Varek","Wiona","Xeran")}; VARIANTS=("valid","broken_a","broken_b")
sys.path.insert(0,str(TESTS)); import phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior as material  # noqa:E402


def save(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")


def build_rows()->list[dict]:
    rows=[]
    for fi,family in enumerate(FAMILIES):
        for unit,triple in TRIPLES.items():
            for language in ("en","zh"):
                source,middle,target=triple
                for variant in VARIANTS:
                    context,spans,queries=material.context_for(family,language,"natural",variant,source,middle,target)
                    for role,query in enumerate(queries):
                        answer,foil=(source,target) if role==0 else (target,source)
                        order=(fi+unit+(language=="zh")+role)%2; candidates=[answer,foil] if order==0 else [foil,answer]
                        prompt,events=material.prompt_with_events(language,context,spans,query,candidates)
                        rows.append({"case_id":f"c42961-{family}-u{unit}-{language}-{variant}-r{role}","family":family,"unit":unit,
                            "partition":"code_interface_calibration" if unit==6 else "frozen_autonomous_lockbox","language":language,"surface":"natural","direction":0,
                            "variant":variant,"query_role":"source" if role==0 else "target","candidate_order":order,"target_candidate_slot":candidates.index(answer),
                            "source":source,"middle":middle,"target":target,"context":context,"query":query,"candidates":candidates,"answer":answer,"foil":foil,"prompt":prompt,"events":events})
    return rows


def append_memo(result:dict)->None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text=rf"""


## Phase {PHASE}: 自动续研——行为合格输出身份与自主路径冻结合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2453–2462完成后目标仍相同，故自动续研。冻结未参与上一阶段的unit6 `(Sela,Tovin,Umera)` 与unit7 `(Varek,Wiona,Xeran)`，覆盖八族、中英、valid/brokenA/brokenB、source/target共192条natural材料；unit6只允许选择代码接口，unit7作为冻结锁箱。下一步并列A/B无示例、A/B平衡双示例、1/2平衡双示例、X/Y平衡双示例；选择只看unit6，unit7不得重选协议或层。随后采q16/q18全坐标VJP，并对模型实际贪心输出的prompt-query、answer-boundary和generated-token1全坐标轨迹作锁箱。

$$o^*=\arg\max_o\left[\operatorname{{Win}}^{{u6}}_{{valid}}(o)+\Delta^{{u6}}_{{valid-brokenA}}(o)-\tfrac12|b_0(o)-b_1(o)|\right].$$

**结果汇总。** 冻结合同 `{json.dumps(result,ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2463_c42961_c43280_autonomous_output_identity_contract.py`；192条源材料与final位于同名结果目录；后续脚本仅引用该索引。

**分析与理论进展。** 这一步修复Phase2461的核心硬伤：不能在A/B行为近随机时把跨接口几何升级为语义证据。先取得行为合格代码编译，再问实体输出和抽象代码是否复用坐标纹理；同时从教师强制转向模型自己选出的首token路径。

**问题硬伤与结论。** 平衡示例会引入新的few-shot脚手架，因此代码接口通过也只能说明条件协议下的输出身份变换。unit6/7仍是合成伪词，不能代表开放自然语言；所有层、接口和门必须冻结，失败不否定模型内部其他编码，只否定本协议的强结论。
"""
    with MEMO.open("a",encoding="utf-8",newline="\n") as handle:handle.write(text)


def main()->None:
    rows=build_rows(); path=OUT/"material/unseen_unit_rows.jsonl";path.parent.mkdir(parents=True,exist_ok=True);path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    contracts={"phase":PHASE,"campaign":CAMPAIGN,"rows":len(rows),"families":list(FAMILIES),"units":{"selection":6,"lockbox":7},"interfaces":["ab_plain","ab_balanced_demo","numeric_balanced_demo","xy_balanced_demo"],
        "frozen_qpoints":[16,18],"autonomous_events":["prompt_query_end","prompt_answer_boundary","generated_token1"],
        "gates":{"code_behavior":"valid win >=.75 and positive valid-brokenA delta on lockbox family","vjp":"same coordinate > shift791 and 64 family q95","autonomous":"actual greedy prefix only; exact generation reported separately"},
        "checks":{"rows_192":len(rows)==192,"eight_families":len({r['family'] for r in rows})==8,"balanced_units":sum(r['unit']==6 for r in rows)==sum(r['unit']==7 for r in rows)==96,
            "unique":len({r['case_id'] for r in rows})==len(rows),"event_bounds":all(all(0<=e['char_start']<=e['char_end']<=len(r['prompt']) for e in r['events']) for r in rows)}}
    contracts["all_checks_passed"]=all(contracts["checks"].values());save(OUT/"analysis/final.json",contracts);append_memo(contracts);print(json.dumps(contracts,ensure_ascii=False,indent=2))
    if not contracts["all_checks_passed"]:raise RuntimeError(contracts["checks"])


if __name__=="__main__":main()
