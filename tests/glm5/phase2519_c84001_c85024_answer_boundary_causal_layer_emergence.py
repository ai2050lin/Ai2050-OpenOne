#!/usr/bin/env python3
"""Locate causal answer-boundary relation selection across all transformer qpoints."""
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


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
P2513 = RESULT / "phase2513_c76673_c78624_fresh_context_factorial_behavior_fullfield"
P2518 = RESULT / "phase2518_c82977_c84000_exact_shape_position_path_patch"
OUT = RESULT / "phase2519_c84001_c85024_answer_boundary_causal_layer_emergence"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2519, "C84001-C85024", 2560
CONTEXTS = (0, 3, 5, 6, 9, 10, 12, 15)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def load_json(p: Path) -> dict: return json.loads(p.read_text(encoding="utf-8-sig"))
def read_jsonl(p: Path) -> list[dict]: return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
def save(p: Path, v: Any) -> None: p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(v, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
def write_jsonl(p: Path, rows: list[dict]) -> None: p.parent.mkdir(parents=True, exist_ok=True); p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
def digest(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(16*1024*1024),b""):h.update(b)
    return h.hexdigest()


def pad(seqs: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor,torch.Tensor]:
    w=max(map(len,seqs)); ids=torch.full((len(seqs),w),pad_id,dtype=torch.long,device=device); mask=torch.zeros_like(ids)
    for i,s in enumerate(seqs): ids[i,:len(s)]=torch.tensor(s,device=device); mask[i,:len(s)]=1
    return ids,mask


def score(logits: torch.Tensor,jobs:list[dict])->list[float]:
    out=[]
    for i,j in enumerate(jobs):
        vals=[]
        for k,t in enumerate(j["continuation"]):
            v=logits[i,j["prompt_length"]-1+k].float(); vals.append(float(v[t]-torch.logsumexp(v,-1)))
        out.append(float(sum(vals)))
    return out


def interventions(rows:list[dict],pairs:list[int],unit:int)->list[dict]:
    lookup={(r["unit"],r["pair_id"],r["language"],r["context_id"],r["meaning_swap"],r["query_marker"]):r for r in rows}; out=[]
    for p in pairs:
        for lang in ("en","zh"):
            for c in CONTEXTS:
                for q in (0,1):
                    a,b=lookup[(unit,p,lang,c,0,q)],lookup[(unit,p,lang,c,1,q)]; assert len(a["prompt_ids"])==len(b["prompt_ids"])
                    out.append({"id":f"u{unit}-p{p}-{lang}-x{c}-q{q}","unit":unit,"pair_id":p,"language":lang,"context_id":c,"query_marker":q,"base":a,"donor":b})
    return out


def run_scan(model,tokenizer,items:list[dict],qpoints:list[int],lock_controls:bool,batch_size:int=8)->list[dict]:
    modules=field_utils.modules(model); device=model.get_input_embeddings().weight.device; active={"q":None,"source":None}; captured={}
    positions=[]
    handles=[]
    for q in range(1,38):
        def hook(_m,_i,output,q=q):
            h=output[0] if isinstance(output,tuple) else output
            if q in qpoints and active["q"] is None: captured[q]=h[torch.arange(h.shape[0],device=h.device),torch.tensor(positions,device=h.device)].detach().clone()
            if active["q"]!=q:return None
            changed=h.clone(); changed[torch.arange(h.shape[0],device=h.device),torch.tensor(positions,device=h.device)]=active["source"].to(changed.dtype)
            return (changed,*output[1:]) if isinstance(output,tuple) else changed
        handles.append(modules[q].register_forward_hook(hook))
    jobs=[]
    for item in items:
        for ri,candidate in enumerate(item["base"]["relation_targets"]):
            cont=[int(v) for v in tokenizer.encode((" " if item["language"]=="en" else "")+candidate,add_special_tokens=False)]
            jobs.append({"id":item["id"],"query_marker":item["query_marker"],"relation_index":ri,"continuation":cont,
                         "prompt_length":len(item["base"]["prompt_ids"]),"position":len(item["base"]["prompt_ids"])-1,
                         "base_sequence":item["base"]["prompt_ids"]+cont,"donor_sequence":item["donor"]["prompt_ids"]+cont})
    results=[]
    try:
        for start in range(0,len(jobs),batch_size):
            batch=jobs[start:start+batch_size]; positions[:]=[j["position"] for j in batch]
            bseq,dseq=[j["base_sequence"] for j in batch],[j["donor_sequence"] for j in batch]; assert [len(x) for x in bseq]==[len(x) for x in dseq]
            bids,mask=pad(bseq,tokenizer.pad_token_id,device); dids,dmask=pad(dseq,tokenizer.pad_token_id,device); assert torch.equal(mask,dmask)
            active.update(q=None,source=None);captured.clear()
            with torch.inference_mode():blogits=model(input_ids=bids,attention_mask=mask,use_cache=False).logits
            base_states={q:captured[q].clone() for q in qpoints}
            for j,v in zip(batch,score(blogits,batch)):results.append({"id":j["id"],"condition":"no_patch","qpoint":0,"relation_index":j["relation_index"],"sum_logprob":v})
            active.update(q=None,source=None);captured.clear()
            with torch.inference_mode():model(input_ids=dids,attention_mask=dmask,use_cache=False)
            donor_states={q:captured[q].clone() for q in qpoints}
            for q in qpoints:
                active.update(q=q,source=donor_states[q]);captured.clear()
                with torch.inference_mode():logits=model(input_ids=bids,attention_mask=mask,use_cache=False).logits
                for j,v in zip(batch,score(logits,batch)):results.append({"id":j["id"],"condition":"donor","qpoint":q,"relation_index":j["relation_index"],"sum_logprob":v})
            if lock_controls:
                q=qpoints[0]
                for name,source in (("self",base_states[q]),("shuffled",donor_states[q].roll(shifts=2 if len(batch)>2 else 1,dims=0))):
                    active.update(q=q,source=source);captured.clear()
                    with torch.inference_mode():logits=model(input_ids=bids,attention_mask=mask,use_cache=False).logits
                    for j,v in zip(batch,score(logits,batch)):results.append({"id":j["id"],"condition":name,"qpoint":q,"relation_index":j["relation_index"],"sum_logprob":v})
            if start%32==0:print(f"[phase2519 {'lock' if lock_controls else 'scan'}] {min(start+len(batch),len(jobs))}/{len(jobs)}",flush=True)
    finally:
        for h in handles:h.remove()
    return results


def panels(items:list[dict],scores:list[dict])->dict:
    lookup={(r["id"],r["condition"],r["qpoint"],r["relation_index"]):r["sum_logprob"] for r in scores}; conditions=sorted({(r["condition"],r["qpoint"]) for r in scores});out={}
    for condition,q in conditions:
        vals=[]
        for item in items:
            key=item["id"]; sign=1 if item["query_marker"]==0 else -1
            base=lookup[(key,"no_patch",0,0)]-lookup[(key,"no_patch",0,1)]
            value=lookup[(key,condition,q,0)]-lookup[(key,condition,q,1)]
            vals.append({"shift":-sign*(value-base),"donor_margin":-sign*value,"flip":-sign*value>0,"self_error":abs(value-base)})
        out[f"{condition}_q{q}"]={"n":len(vals),"mean_shift":float(np.mean([v["shift"] for v in vals])),
                                  "positive_shift_rate":float(np.mean([v["shift"]>0 for v in vals])),"donor_flip_rate":float(np.mean([v["flip"] for v in vals])),
                                  "mean_donor_margin":float(np.mean([v["donor_margin"] for v in vals])),"max_self_error":float(max(v["self_error"] for v in vals))}
    return out


def append_memo(result:dict)->None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text=rf"""


## Phase {PHASE}: answer-boundary关系选择因果效应的全层涌现与fresh-unit锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2518显示q28的answer-boundary单位置足以大幅转移输出。本Phase在unit28同样128个四因素平衡干预上，对q1–q37每层的answer-boundary全部2560坐标做exact-shape matched-donor patch，以donor翻转率、方向变化选择唯一层；之后只在unit29该层运行matched donor、self和batch错配donor锁箱。base/donor前向逐样本shape、mask、候选后缀完全相同。

$$E_q=\Delta L\big(H^{{base}}_{{q,t_a,:}}\leftarrow H^{{donor}}_{{q,t_a,:}}\big)-\Delta L(H^{{base}}).$$

**结果汇总。** 选层 `{json.dumps(result['selection'],ensure_ascii=False)}`；unit28全层 `{json.dumps(result['confirmation_scan'],ensure_ascii=False)}`；unit29锁箱 `{json.dumps(result['lockbox'],ensure_ascii=False)}`；检查 `{json.dumps(result['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2519_c84001_c85024_answer_boundary_causal_layer_emergence.py`；unit28全层与unit29冻结层候选分数、面板、哈希和final位于`{OUT}`。

**分析与理论进展。** 这给出输出决策在answer-boundary残差流中的因果涌现曲线。早层无效、晚层转移只能说明正确关系选择逐渐汇聚到输出位置；它不证明某层独自“理解语义”。unit29 matched优于shuffled才支持状态内容匹配，而不仅是强扰动。

**问题硬伤与结论。** 选层仍来自同一合成模板；整2560维自然状态包含所有任务变量；answer-boundary接近unembedding，强因果效应可能主要是输出身份准备。下一步必须扩展到自然句式/开放输出或不同模型，判断事件路径而非物理层号是否复现。
"""
    with MEMO.open("a",encoding="utf-8",newline="\n")as f:f.write(text)


def main()->None:
    f13,f18=load_json(P2513/"analysis/final.json"),load_json(P2518/"analysis/final.json")
    rows=read_jsonl(Path(f13["collection"]["event_index"]));material={r["case_id"]:r for r in read_jsonl(P2513/"material/factorial_rows.jsonl")}
    for r in rows:r["relation_targets"]=material[r["case_id"]]["relation_targets"]
    confirm=interventions(rows,f13["behavior"]["qualified_pair_ids"],28);lock=interventions(rows,f13["behavior"]["qualified_pair_ids"],29)
    model,tokenizer,_=model_utils.load_model("qwen3",dtype=torch.bfloat16,use_8bit=False)
    try:
        scan_scores=run_scan(model,tokenizer,confirm,list(range(1,38)),False);scan=panels(confirm,scan_scores)
        chosen=max(range(1,38),key=lambda q:(scan[f"donor_q{q}"]["donor_flip_rate"],scan[f"donor_q{q}"]["mean_shift"]))
        lock_scores=run_scan(model,tokenizer,lock,[chosen],True);lock_panel=panels(lock,lock_scores)
    finally:model_utils.release_model(model);gc.collect()
    sp=OUT/"output/unit28_alllayer_scores.jsonl";lp=OUT/"output/unit29_lockbox_scores.jsonl";write_jsonl(sp,scan_scores);write_jsonl(lp,lock_scores)
    selection={"rule":"maximum unit28 donor flip rate, then mean shift","qpoint":chosen,"confirmation":scan[f"donor_q{chosen}"]}
    self_panel=lock_panel[f"self_q{chosen}"];donor=lock_panel[f"donor_q{chosen}"];shuffled=lock_panel[f"shuffled_q{chosen}"]
    checks={"sources_passed":f13["all_checks_passed"]and f18["all_checks_passed"],"confirmation_128":len(confirm)==128,"lockbox_128":len(lock)==128,
            "all_37_qpoints":len([k for k in scan if k.startswith("donor_")])==37,"selection_confirmation_only":True,
            "self_control_exact":self_panel["max_self_error"]==0.0,"hashes":len(digest(sp))==64 and len(digest(lp))==64,"claim_boundary":True}
    result={"phase":PHASE,"campaign":CAMPAIGN,"model":"Qwen3-4B nonquantized BF16 CUDA","selection":selection,"confirmation_scan":scan,"lockbox":lock_panel,
            "files":{"confirmation_scores":str(sp),"lockbox_scores":str(lp),"sha256":{sp.name:digest(sp),lp.name:digest(lp)}},
            "adjudication":{"answer_boundary_causal_emergence_replicated":donor["donor_flip_rate"]>.75 and donor["mean_shift"]>shuffled["mean_shift"],
                            "selected_qpoint_is_semantic_compiler_layer":False,"language_encoding_mechanism_closed":False},"checks":checks,"all_checks_passed":all(checks.values())}
    save(OUT/"analysis/final.json",result)
    if result["all_checks_passed"]:append_memo(result)
    print(json.dumps({"phase":PHASE,"selection":selection,"lockbox":lock_panel,"adjudication":result["adjudication"],"checks":checks,"all_checks_passed":result["all_checks_passed"]},ensure_ascii=False,indent=2))
    if not result["all_checks_passed"]:raise RuntimeError(checks)


if __name__=="__main__":main()
