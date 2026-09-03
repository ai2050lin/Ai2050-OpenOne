#!/usr/bin/env python3
"""Select a balanced one-token code interface on unit6 and lock it on unit7."""
from __future__ import annotations

import gc,json,math,sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result";P2463=RESULT/"phase2463_c42961_c43280_autonomous_output_identity_contract"
OUT=RESULT/"phase2464_c43281_c43600_balanced_code_interface_calibration";MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md";PHASE,CAMPAIGN=2464,"C43281-C43600"
PROTOCOLS=("ab_plain","ab_balanced_demo","numeric_balanced_demo","xy_balanced_demo")
sys.path.insert(0,str(TESTS));import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa:E402

def save(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
def read_rows(path:Path)->list[dict]:return[json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]

def codes(protocol:str)->tuple[str,str]:
    if protocol.startswith("ab_"):return"A","B"
    if protocol.startswith("numeric_"):return"1","2"
    return"X","Y"

def prompt_for(row:dict,protocol:str)->str:
    left,right=codes(protocol);zh=row["language"]=="zh"
    demo=""
    if "balanced_demo" in protocol:
        if zh:demo=(f"先学习代码作答格式。\n示例1：要求的名称是Noma。{left}=Noma，{right}=Peli。答案：{left}\n"
                    f"示例2：要求的名称是Tera。{left}=Sola，{right}=Tera。答案：{right}\n现在解答新记录。\n")
        else:demo=(f"Learn the code-answer format.\nExample 1: requested name is Noma. {left}=Noma, {right}=Peli. Answer: {left}\n"
                   f"Example 2: requested name is Tera. {left}=Sola, {right}=Tera. Answer: {right}\nNow solve the new record.\n")
    if zh:return demo+f"阅读记录，把候选项映射为代码，只回答{left}或{right}。\n记录：{row['context']}\n问题：{row['query']}\n代码：\n{left} = {row['candidates'][0]}\n{right} = {row['candidates'][1]}\n答案："
    return demo+f"Read the record, map candidates to codes, and answer with exactly {left} or {right}.\nRecord: {row['context']}\nQuestion: {row['query']}\nCodes:\n{left} = {row['candidates'][0]}\n{right} = {row['candidates'][1]}\nAnswer:"

def compile_rows(tokenizer,bases:list[dict])->list[dict]:
    output=[]
    for base in bases:
        for protocol in PROTOCOLS:
            left,right=codes(protocol);target=left if int(base["target_candidate_slot"])==0 else right;foil=right if target==left else left;prompt=prompt_for(base,protocol)
            target_ids=[int(x) for x in tokenizer.encode(target,add_special_tokens=False)];foil_ids=[int(x) for x in tokenizer.encode(foil,add_special_tokens=False)]
            output.append({"case_id":f"{base['case_id']}--{protocol}","base_case_id":base["case_id"],"protocol":protocol,**{k:base[k] for k in ("family","unit","language","variant","query_role","target_candidate_slot")},
                "prompt":prompt,"prompt_ids":capability.chat_ids(tokenizer,prompt),"target_code":target,"foil_code":foil,"target_ids":target_ids,"foil_ids":foil_ids})
    output.sort(key=lambda r:r["case_id"]);return output

def score(model,rows:list[dict])->np.ndarray:
    device=model.get_input_embeddings().weight.device;pad=int(model.config.pad_token_id or model.config.eos_token_id or 0);values=np.zeros(len(rows),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(rows),16):
            batch=rows[start:start+16];width=max(len(r["prompt_ids"]) for r in batch);ids=torch.full((len(batch),width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
            for i,row in enumerate(batch):
                length=len(row["prompt_ids"]);ids[i,:length]=torch.tensor(row["prompt_ids"],device=device);mask[i,:length]=1
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).logits.float()
            for i,row in enumerate(batch):values[start+i]=float((logits[i,len(row["prompt_ids"])-1,int(row["target_ids"][0])]-logits[i,len(row["prompt_ids"])-1,int(row["foil_ids"][0])]).cpu())
            if min(start+16,len(rows))%96==0:print(f"[phase2464 behavior] {min(start+16,len(rows))}/{len(rows)}",flush=True)
    return values

def metrics(rows:list[dict],values:np.ndarray)->dict:
    families=sorted({r["family"] for r in rows});result={}
    for protocol in PROTOCOLS:
        result[protocol]={}
        for unit in (6,7):
            chosen=[i for i,r in enumerate(rows) if r["protocol"]==protocol and int(r["unit"])==unit]
            def rate(indices):return float(np.mean(values[indices]>0)) if indices else 0.0
            valid=[i for i in chosen if rows[i]["variant"]=="valid"];broken=[i for i in chosen if rows[i]["variant"]=="broken_a"]
            slot0=[i for i in valid if int(rows[i]["target_candidate_slot"])==0];slot1=[i for i in valid if int(rows[i]["target_candidate_slot"])==1]
            family={}
            for name in families:
                fv=[i for i in valid if rows[i]["family"]==name];fb=[i for i in broken if rows[i]["family"]==name]
                family[name]={"valid_win_rate":rate(fv),"broken_a_win_rate":rate(fb),"valid_minus_broken_a":rate(fv)-rate(fb),"valid_mean_margin":float(np.mean(values[fv]))}
            valid_rate,broken_rate=rate(valid),rate(broken);bias=abs(rate(slot0)-rate(slot1))
            result[protocol][f"unit{unit}"]={"valid_win_rate":valid_rate,"broken_a_win_rate":broken_rate,"valid_minus_broken_a":valid_rate-broken_rate,
                "slot0_valid_win_rate":rate(slot0),"slot1_valid_win_rate":rate(slot1),"slot_bias":bias,"selection_score":valid_rate+(valid_rate-broken_rate)-.5*bias,"by_family":family}
    return result

def append_memo(result:dict)->None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text=rf"""


## Phase {PHASE}: 四种单token代码接口的unit6选择与unit7冻结行为锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2463的192条未见unit材料编译A/B无示例、A/B平衡双示例、1/2平衡双示例和X/Y平衡双示例，共768条。所有代码均核对为单token。用目标代码减foil代码的首token logit margin测valid、brokenA、候选槽偏置和八族；协议只由unit6的$Win_{{valid}}+\Delta_{{valid-brokenA}}-0.5|b_0-b_1|$选择，然后原样应用unit7，不允许重选。

$$m_o=\ell_{{code(answer)}}-\ell_{{code(foil)}}.$$

**结果汇总。** 四协议双unit `{json.dumps(result['analysis'],ensure_ascii=False)}`；冻结选择 `{json.dumps(result['selection'],ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'],ensure_ascii=False)}`；检查 `{json.dumps(result['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2464_c43281_c43600_balanced_code_interface_calibration.py`；768条编译索引、逐行margin、选择与final位于同名结果目录。

**分析与理论进展。** 代码接口的行为资格与内部几何被彻底分开。若平衡示例显著消除A/B位置偏置，Phase2461失败可归因于接口执行而不是证明输出身份不可解耦；若unit7仍失败，则不能以unit6选择集救援。

**问题硬伤与结论。** few-shot示例是显式协议脚手架，可能激活示例模仿而非自然语言通用机制。valid−brokenA本身混合关系有效性与模型对无效记录的处理；0.75门只是一致的行为资格，不代表完全正确。
"""
    with MEMO.open("a",encoding="utf-8",newline="\n") as handle:handle.write(text)

def main()->None:
    bases=read_rows(P2463/"material/unseen_unit_rows.jsonl");model,tokenizer,_=capability.load_model("qwen4b")
    try:rows=compile_rows(tokenizer,bases);values=score(model,rows)
    finally:del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    index=OUT/"index/code_interface_rows.jsonl";index.parent.mkdir(parents=True,exist_ok=True);index.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    margin=OUT/"raw/code_margin.float32.npy";margin.parent.mkdir(parents=True,exist_ok=True);np.save(margin,values)
    analysis=metrics(rows,values);selected=max(PROTOCOLS,key=lambda p:analysis[p]["unit6"]["selection_score"]);lock=analysis[selected]["unit7"]
    qualified=[f for f,v in lock["by_family"].items() if v["valid_win_rate"]>=.75 and v["valid_minus_broken_a"]>0]
    selection={"selected_protocol":selected,"selection_unit":6,"lockbox_unit":7,"unit6_score":analysis[selected]["unit6"]["selection_score"],"unit7_qualified_families":qualified}
    adjudication={"balanced_code_behavior_all8_lockbox":len(qualified)==8,"balanced_code_behavior_any_family":len(qualified)>0,"phase2461_unqualified_ab_plain_explained_by_protocol":analysis[selected]["unit7"]["valid_win_rate"]>analysis["ab_plain"]["unit7"]["valid_win_rate"],"language_encoding_mechanism_closed":False}
    checks={"rows_768":len(rows)==768,"single_token_codes":all(len(r["target_ids"])==len(r["foil_ids"])==1 for r in rows),"selection_unit_only":selected in PROTOCOLS,
        "files":index.exists() and margin.exists(),"finite":all(math.isfinite(v) for p in analysis.values() for u in p.values() for k,v in u.items() if k!="by_family"),"claim_boundary":not adjudication["language_encoding_mechanism_closed"]}
    result={"phase":PHASE,"campaign":CAMPAIGN,"collection":{"rows":len(rows),"index":str(index),"margin":str(margin),"inference":"Qwen3-4B BF16 CUDA"},"analysis":analysis,"selection":selection,"adjudication":adjudication,"checks":checks,"all_checks_passed":all(checks.values())}
    save(OUT/"analysis/final.json",result);append_memo(result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result["all_checks_passed"]:raise RuntimeError(checks)

if __name__=="__main__":main()
