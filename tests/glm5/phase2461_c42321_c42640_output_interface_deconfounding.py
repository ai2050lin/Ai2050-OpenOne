#!/usr/bin/env python3
"""Output-identity and prompt-interface deconfounding with all-coordinate VJPs."""
from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
P2435 = RESULT / "phase2435_c33841_c34160_hypergraph_material_fourmodel_behavior/qwen4b"
P2448 = RESULT / "phase2448_c38001_c38480_vjp_semantic_multiunit_replication"
OUT = RESULT / "phase2461_c42321_c42640_output_interface_deconfounding"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM, SHIFT = 2461, "C42321-C42640", 2560, 791
QPOINTS = (16, 18); FIELDS = ("gradient", "state_times_gradient")
NEW_INTERFACES = ("bare_entity", "letter_code"); ALL_INTERFACES = ("candidate_entity",) + NEW_INTERFACES
VARIANTS = ("valid", "broken_a", "broken_b")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a,b=np.asarray(left,dtype=np.float64).reshape(-1),np.asarray(right,dtype=np.float64).reshape(-1); den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/den) if den>1e-30 else 0.0


def derangements(count: int, size: int, seed: int) -> np.ndarray:
    rng=np.random.default_rng(seed); values=[]
    while len(values)<count:
        p=rng.permutation(size)
        if np.all(p!=np.arange(size)): values.append(p)
    return np.stack(values)


def find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    for start in range(len(sequence)-len(subsequence)+1):
        if sequence[start:start+len(subsequence)]==subsequence: return start
    return -1


def common_prefix(left: list[int], right: list[int]) -> int:
    count=0
    for a,b in zip(left,right):
        if a!=b: break
        count+=1
    return count


def base_rows() -> list[dict]:
    rows=read_rows(P2435/"index/trajectory_rows.jsonl")
    rows=[r for r in rows if int(r["unit"]) in (4,5) and r["surface"]=="natural" and int(r["direction"])==0]
    rows.sort(key=lambda r:r["case_id"]); return rows


def compile_rows(tokenizer, bases: list[dict]) -> tuple[list[dict], dict]:
    compiled=[]; exact=0; code_single=0
    for base in bases:
        for interface in NEW_INTERFACES:
            if interface=="bare_entity":
                if base["language"]=="en":
                    prompt=f"Read the record and answer with exactly the requested name.\nRecord: {base['context']}\nQuestion: {base['query']}\nAnswer:"
                else:
                    prompt=f"阅读记录，只回答问题要求的名称。\n记录：{base['context']}\n问题：{base['query']}\n答案："
                answer,foil=base["answer"],base["foil"]
            else:
                if base["language"]=="en":
                    prompt=(f"Read the record, map the two candidates to codes, and answer with exactly A or B.\nRecord: {base['context']}\n"
                            f"Question: {base['query']}\nCodes:\nA = {base['candidates'][0]}\nB = {base['candidates'][1]}\nAnswer:")
                else:
                    prompt=(f"阅读记录，把两个候选项映射到代码，只能回答A或B。\n记录：{base['context']}\n问题：{base['query']}\n"
                            f"代码：\nA = {base['candidates'][0]}\nB = {base['candidates'][1]}\n答案：")
                answer="A" if int(base["target_candidate_slot"])==0 else "B"; foil="B" if answer=="A" else "A"
            prompt_ids=capability.chat_ids(tokenizer,prompt); raw_ids=[int(x) for x in tokenizer.encode(prompt,add_special_tokens=False)]
            start=find_subsequence(prompt_ids,raw_ids); exact+=int(start>=0)
            query_end_char=prompt.index(base["query"])+len(base["query"])
            prefix=[int(x) for x in tokenizer.encode(prompt[:query_end_char],add_special_tokens=False)]
            token_index=start+max(0,common_prefix(raw_ids,prefix)-1) if start>=0 else len(capability.chat_ids(tokenizer,prompt[:query_end_char]))-1
            target_ids=[int(x) for x in tokenizer.encode(answer,add_special_tokens=False)]; foil_ids=[int(x) for x in tokenizer.encode(foil,add_special_tokens=False)]
            if interface=="letter_code": code_single+=int(len(target_ids)==len(foil_ids)==1)
            compiled.append({"case_id":f"{base['case_id']}--{interface}","base_case_id":base["case_id"],"interface":interface,
                **{k:base[k] for k in ("family","unit","language","variant","query_role")},"prompt":prompt,"prompt_ids":prompt_ids,
                "query_end_token":int(token_index),"answer":answer,"foil":foil,"target_ids":target_ids,"foil_ids":foil_ids})
    compiled.sort(key=lambda r:r["case_id"])
    return compiled,{"rows":len(compiled),"raw_prompt_exact_rate":exact/len(compiled),"letter_single_token_rate":code_single/len(bases)}


def capture(rows: list[dict], compile_audit: dict) -> dict:
    raw=OUT/"raw"; raw.mkdir(parents=True,exist_ok=True)
    field_path,margin_path=raw/"new_interface_fields.float32.npy",raw/"new_interface_margin.float32.npy"
    fields=np.lib.format.open_memmap(field_path,mode="r+" if field_path.exists() else "w+",dtype=np.float32,shape=(len(rows),2,2,DIM))
    margins=np.lib.format.open_memmap(margin_path,mode="r+" if margin_path.exists() else "w+",dtype=np.float32,shape=(len(rows),))
    progress=raw/"progress.json"; completed=int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    model=tokenizer=None; captures={}; handles=[]
    if completed<len(rows):
        model,tokenizer,_=capability.load_model("qwen4b"); model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        modules=field_utils.modules(model); device=model.get_input_embeddings().weight.device
        def leaf_hook(_module,_inputs,result):
            tensor=result[0] if isinstance(result,tuple) else result
            if not tensor.requires_grad: tensor.requires_grad_(True)
        handles.append(modules[0].register_forward_hook(leaf_hook))
        for slot,qpoint in enumerate(QPOINTS):
            def field_hook(_module,_inputs,result,slot=slot):
                tensor=result[0] if isinstance(result,tuple) else result; tensor.retain_grad(); captures[slot]=tensor
            handles.append(modules[qpoint].register_forward_hook(field_hook))
    else: device=None
    try:
        for index in range(completed,len(rows)):
            row=rows[index]; ids=torch.tensor([row["prompt_ids"]],dtype=torch.long,device=device); attention=torch.ones_like(ids); positions=torch.arange(ids.shape[1],device=device)[None]
            captures.clear()
            with torch.enable_grad():
                output=model(input_ids=ids,attention_mask=attention,position_ids=positions,use_cache=False,return_dict=True)
                target,foil=int(row["target_ids"][0]),int(row["foil_ids"][0]); margin=output.logits[0,-1,target]-output.logits[0,-1,foil]; margin.backward()
            for slot in range(2):
                state=captures[slot][0,row["query_end_token"]].detach().float().cpu().numpy(); gradient=captures[slot].grad[0,row["query_end_token"]].detach().float().cpu().numpy()
                fields[index,slot,0]=gradient; fields[index,slot,1]=state*gradient
            margins[index]=float(margin.detach().float().cpu()); fields.flush(); margins.flush(); save(progress,{"completed":index+1,"rows":len(rows)})
            if (index+1)%16==0 or index+1==len(rows): print(f"[phase2461 interfaces] {index+1}/{len(rows)}",flush=True)
            del ids,attention,positions,output,margin
    finally:
        for handle in handles: handle.remove()
        del model,tokenizer
        for value in (fields,margins): value.flush(); close(value)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    index_path=OUT/"index/interface_rows.jsonl"; index_path.parent.mkdir(parents=True,exist_ok=True)
    index_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    return {"fields":str(field_path),"margins":str(margin_path),"shape":[len(rows),2,2,DIM],"rows":len(rows),"qpoints":list(QPOINTS),
            "interfaces":list(NEW_INTERFACES),"compile_audit":compile_audit,"all_physical_coordinates":True,"inference":"Qwen3-4B BF16 CUDA first-token target-minus-foil VJP"}


def build_new_passports(rows: list[dict], values: np.ndarray, families: list[str]) -> np.ndarray:
    lookup={(r["interface"],int(r["unit"]),r["family"],r["language"],r["variant"],r["query_role"]):i for i,r in enumerate(rows)}
    output=np.zeros((2,2,2,2,2,2,8,DIM),dtype=np.float32)
    for ii,interface in enumerate(NEW_INTERFACES):
        for ui,unit in enumerate((4,5)):
            for li,language in enumerate(("en","zh")):
                for fi,family in enumerate(families):
                    role={}
                    for variant in VARIANTS:
                        source,target=lookup[(interface,unit,family,language,variant,"source")],lookup[(interface,unit,family,language,variant,"target")]
                        role[variant]=values[target]-values[source]
                    output[0,ii,ui,li,:,:,fi]=role["valid"]-role["broken_a"]
                    output[1,ii,ui,li,:,:,fi]=role["broken_a"]-role["broken_b"]
    return output


def analyze(rows: list[dict], collection: dict, families: list[str]) -> dict:
    values=np.load(collection["fields"],mmap_mode="r"); new=build_new_passports(rows,values,families); close(values)
    combined=np.zeros((2,3,2,2,2,2,8,DIM),dtype=np.float32)
    combined[:,1:]=new
    p2448=json.loads((P2448/"analysis/final.json").read_text(encoding="utf-8")); natural=np.load(p2448["analysis"]["passports"],mmap_mode="r")
    for interaction in range(2):
        for ui in range(2):
            for li in range(2):
                for slot,qpoint in enumerate(QPOINTS):
                    for field in range(2): combined[interaction,0,ui,li,slot,field]=natural[interaction,field,ui+1,li,qpoint]
    close(natural)
    combined_path=OUT/"derived/three_interface_passports.float32.npy"; combined_path.parent.mkdir(parents=True,exist_ok=True); np.save(combined_path,combined)
    permutations=derangements(64,8,2461); pairs=((0,1),(0,2),(1,2)); cross={}
    for interaction,iname in enumerate(("semantic_validity","lexical_control")):
        cross[iname]={}
        for field,fname in enumerate(FIELDS):
            slot=1 if field==0 else 0; cross[iname][fname]={}
            for ui,unit in enumerate((4,5)):
                cross[iname][fname][f"unit{unit}"]={}
                for left,right in pairs:
                    key=f"{ALL_INTERFACES[left]}__{ALL_INTERFACES[right]}"; a,b=combined[interaction,left,ui,: ,slot,field],combined[interaction,right,ui,:,slot,field]
                    observed=float(np.mean([cosine(a[l,f],b[l,f]) for l in range(2) for f in range(8)])); shifted=float(np.mean([cosine(a[l,f],np.roll(b[l,f],SHIFT)) for l in range(2) for f in range(8)]))
                    null=np.asarray([np.mean([cosine(a[l,f],b[l,p[f]]) for l in range(2) for f in range(8)]) for p in permutations]); q95=float(np.quantile(null,.95))
                    cross[iname][fname][f"unit{unit}"][key]={"coordinate":observed,"shift791":shifted,"family_null_mean":float(np.mean(null)),"family_null_q95":q95,
                        "physical_advantage":observed-shifted,"family_identity_advantage":observed-q95}
    crosslanguage={}
    for interaction,iname in enumerate(("semantic_validity","lexical_control")):
        crosslanguage[iname]={}
        for field,fname in enumerate(FIELDS):
            slot=1 if field==0 else 0; crosslanguage[iname][fname]={}
            for ii,interface in enumerate(ALL_INTERFACES):
                crosslanguage[iname][fname][interface]={}
                for ui,unit in enumerate((4,5)):
                    en,zh=combined[interaction,ii,ui,0,slot,field],combined[interaction,ii,ui,1,slot,field]
                    obs=float(np.mean([cosine(en[f],zh[f]) for f in range(8)])); null=np.asarray([np.mean([cosine(en[f],zh[p[f]]) for f in range(8)]) for p in permutations]); q95=float(np.quantile(null,.95))
                    crosslanguage[iname][fname][interface][f"unit{unit}"]={"coordinate":obs,"family_null_q95":q95,"family_identity_advantage":obs-q95}
    margins=np.asarray(np.load(collection["margins"],mmap_mode="r"),dtype=np.float64)
    behavior={}
    for interface in NEW_INTERFACES:
        behavior[interface]={}
        for unit in (4,5):
            behavior[interface][f"unit{unit}"]={}
            for family in families:
                indices=[i for i,r in enumerate(rows) if r["interface"]==interface and int(r["unit"])==unit and r["family"]==family]
                valid=[i for i in indices if rows[i]["variant"]=="valid"]
                behavior[interface][f"unit{unit}"][family]={"valid_win_rate":float(np.mean(margins[valid]>0)),"valid_mean_margin":float(np.mean(margins[valid])),"all_win_rate":float(np.mean(margins[indices]>0))}
    return {"families":families,"combined_passports":str(combined_path),"combined_shape":list(combined.shape),"behavior":behavior,
            "crossinterface":cross,"crosslanguage":crosslanguage}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text=rf"""


## Phase {PHASE}: 实体候选、裸实体与字母代码三输出接口的全坐标去混淆（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在unit4确认与unit5 fresh的natural-direction0八族中英材料上，保留原`candidate_entity`接口，并新增两种输出：`bare_entity`删除候选列表但仍输出实体名；`letter_code`保留实体—候选映射却只输出单token A/B。新采384条（三variant×双角色）q16/q18 query-end全部2560坐标gradient与$H\odot g$，与Phase2448原接口护照组成三接口张量。逐unit比较同family同物理坐标、+791错位与64个family置乱，语义interaction和词项control并列。

$$I^o_{{sem}}=[(F^o_t-F^o_s)_{{valid}}-(F^o_t-F^o_s)_{{brokenA}}],\qquad
T(o_1,o_2)=\mathbb E_{{l,f}}\cos(I^{{o_1}}_{{l,f}},I^{{o_2}}_{{l,f}}).$$

**结果汇总。** 新接口采集 `{json.dumps(result['collection'],ensure_ascii=False)}`；逐接口行为 `{json.dumps(result['analysis']['behavior'],ensure_ascii=False)}`；跨接口全坐标裁决 `{json.dumps(result['analysis']['crossinterface'],ensure_ascii=False)}`；接口内跨语言 `{json.dumps(result['analysis']['crosslanguage'],ensure_ascii=False)}`；总裁决 `{json.dumps(result['adjudication'],ensure_ascii=False)}`；检查 `{json.dumps(result['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2461_c42321_c42640_output_interface_deconfounding.py`；384×2 qpoint×2 field×2560新接口原场、逐行margin、三接口完整护照和final位于同名结果目录。原candidate接口仍引用Phase2448全坐标护照。

**分析与理论进展。** `candidate→bare`主要改变任务框架而保持输出实体；`candidate/bare→letter`改变输出token身份但保留语义选择。只有语义纹理在两个held unit跨三对接口都胜物理错位与family错配、并优于词项control，才可称输出身份解耦候选。单对或单unit通过只说明部分坐标复用。

**问题硬伤与结论。** A/B仍通过候选表绑定实体，裸实体接口仍共享记录与问题模板；输出接口没有覆盖开放词表自由回答。第一token VJP忽略代码后续解释，且接口行为失败的family不能升级为语义机制。VJP仍是输出条件分析量，不是模型内显式齿轮对象。
"""
    with MEMO.open("a",encoding="utf-8",newline="\n") as handle: handle.write(text)


def main() -> None:
    final=OUT/"analysis/final.json"
    if final.exists():
        result=json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result,ensure_ascii=False,indent=2)); return
    model,tokenizer,_=capability.load_model("qwen4b"); bases=base_rows(); rows,audit=compile_rows(tokenizer,bases); del model,tokenizer; gc.collect(); torch.cuda.empty_cache()
    families=sorted({r["family"] for r in bases}); collection=capture(rows,audit); analysis=analyze(rows,collection,families)
    semantic=analysis["crossinterface"]["semantic_validity"]["state_times_gradient"]; lexical=analysis["crossinterface"]["lexical_control"]["state_times_gradient"]
    gates=[]; specificity=[]
    for unit in (4,5):
        for pair,value in semantic[f"unit{unit}"].items():
            gates.append(value["physical_advantage"]>0 and value["family_identity_advantage"]>0)
            specificity.append(value["coordinate"]>lexical[f"unit{unit}"][pair]["coordinate"])
    qualified_letter=[family for family in families if all(analysis["behavior"]["letter_code"][f"unit{unit}"][family]["valid_win_rate"]>=.75 for unit in (4,5))]
    candidate_bare=all(semantic[f"unit{unit}"]["candidate_entity__bare_entity"][key]>0 for unit in (4,5) for key in ("physical_advantage","family_identity_advantage"))
    adjudication={"three_interface_hxg_geometry_lockbox_unconditional":all(gates),
                  "letter_code_behavior_qualified_families":qualified_letter,
                  "letter_code_behavior_qualified_family_count":len(qualified_letter),
                  "behavior_gated_semantic_hxg_all_three_interfaces":all(gates) and len(qualified_letter)==8,
                  "candidate_to_bare_entity_taskframe_lockbox":candidate_bare,
                  "semantic_exceeds_lexical_all_interface_pairs":all(specificity),
                  "output_identity_decoupling_candidate":all(gates) and all(specificity) and len(qualified_letter)==8,
                  "language_encoding_mechanism_closed":False}
    checks={"new_rows_384":collection["rows"]==384,"full_coordinates":collection["shape"]==[384,2,2,2560],"three_interfaces":analysis["combined_shape"]==[2,3,2,2,2,2,8,2560],
            "letter_single_token":collection["compile_audit"]["letter_single_token_rate"]==1.0,"files":Path(collection["fields"]).exists() and Path(collection["margins"]).exists() and Path(analysis["combined_passports"]).exists(),
            "finite":all(math.isfinite(v) for interaction in analysis["crossinterface"].values() for field in interaction.values() for unit in field.values() for pair in unit.values() for v in pair.values()),
            "claim_boundary":not adjudication["language_encoding_mechanism_closed"]}
    result={"phase":PHASE,"campaign":CAMPAIGN,"collection":collection,"analysis":analysis,"adjudication":adjudication,"checks":checks,"all_checks_passed":all(checks.values())}
    save(final,result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
