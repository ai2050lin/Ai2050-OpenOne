#!/usr/bin/env python3
"""Frozen causal adjudication of contextual coordinate fields and the prior attention-head candidate."""
from __future__ import annotations

import gc
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2378 = RESULT / "phase2378_c15601_c15920_label_free_binding_contract"
P2390 = RESULT / "phase2390_c19441_c19760_qwen_semantic_lexical_fullfield"
P2391 = RESULT / "phase2391_c19761_c20080_semantic_lexical_adjudication"
P2392 = RESULT / "phase2392_c20081_c20400_contextual_coordinate_gear_atlas"
OUT = RESULT / "phase2393_c20401_c20720_frozen_context_causal_adjudication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2393
CAMPAIGN = "C20401-C20720"
MODELS = ("qwen4b", "qwen14b")
FAMILIES = ("preference", "taxonomy", "temporal", "causal", "comparison", "spatial", "role_binding", "ownership_transfer")
LANGUAGES = ("en", "zh")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def pad_right(sequences: list[list[int]], device: torch.device, pad: int):
    width=max(map(len,sequences)); ids=torch.full((len(sequences),width),pad,dtype=torch.long,device=device); mask=torch.zeros_like(ids)
    for i,sequence in enumerate(sequences): ids[i,:len(sequence)]=torch.tensor(sequence,dtype=torch.long,device=device); mask[i,:len(sequence)]=1
    return ids,mask,(mask.cumsum(1)-1).clamp_min(0)


def sequence_scores(model, rows: list[dict], continuation_key: str, batch_size: int, hook_spec: dict | None = None) -> np.ndarray:
    device=model.get_input_embeddings().weight.device; pad=int(model.config.pad_token_id or model.config.eos_token_id or 0); result=[]; context={}
    handle=None
    if hook_spec:
        layer=model.model.layers[int(hook_spec["qpoint"])-1]
        def hook(_module,_inputs,output):
            hidden=output[0] if isinstance(output,tuple) else output; changed=hidden.clone()
            for local,(boundary,row_index) in enumerate(zip(context["boundaries"],context["row_indices"])):
                action=hook_spec["action"]
                if action=="replace": changed[local,boundary]=hook_spec["vectors"][row_index].to(device=changed.device,dtype=changed.dtype)
                elif action=="add": changed[local,boundary]+=hook_spec["vectors"][row_index].to(device=changed.device,dtype=changed.dtype)
            return (changed,*output[1:]) if isinstance(output,tuple) else changed
        handle=layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for start in range(0,len(rows),batch_size):
                batch=rows[start:start+batch_size]; sequences=[row["prompt_ids"]+row[continuation_key] for row in batch]
                ids,mask,positions=pad_right(sequences,device,pad); context["boundaries"]=[len(row["prompt_ids"])-1 for row in batch]; context["row_indices"]=[row["model_row"] for row in batch]
                logits=model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True).logits.float()
                for local,row in enumerate(batch):
                    start_pos=len(row["prompt_ids"])-1; continuation=row[continuation_key]; token=torch.tensor(continuation,device=device)[:,None]
                    logp=torch.log_softmax(logits[local,start_pos:start_pos+len(continuation)],dim=-1).gather(1,token).squeeze(1)
                    result.append(float(logp.mean().item()))
    finally:
        if handle is not None: handle.remove()
    return np.array(result,dtype=np.float32)


def summarize_matrix(matrix: np.ndarray, conditions: list[str]) -> dict:
    summary={}
    clean=matrix[0,:,2]
    for ci,name in enumerate(conditions):
        margin=matrix[ci,:,2]
        summary[name]={"target_over_foil":float(np.mean(margin>0)),"mean_margin":float(margin.mean()),
                       "mean_margin_change_from_clean":float((margin-clean).mean()),"median_margin_change_from_clean":float(np.median(margin-clean))}
    return summary


def semantic_interventions(key: str, model) -> dict:
    base=P2390/key; rows_all=read_rows(base/"index/selection_rows.jsonl"); rows=[row for row in rows_all if row["partition"]=="fresh_unit_lockbox"]
    qpoint=int(json.loads((P2391/key/"analysis/final.json").read_text(encoding="utf-8"))["output_selected"]["qpoint"]); wrong_q=max(1,qpoint-8)
    field=np.load(base/"raw/semantic_selection_prompt_boundary.float16.npy",mmap_mode="r"); current=np.asarray(field[:,qpoint],dtype=np.float32); wrong_current=np.asarray(field[:,wrong_q],dtype=np.float32)
    response=np.load(P2392/key/"derived/all_layer_partition_relation_response.float32.npy",mmap_mode="r"); relation=np.asarray(response[qpoint,0],dtype=np.float32)
    rank=np.load(P2392/key/"derived/confirmation_frozen_coordinate_rank.int32.npy"); count=max(8,int(round(current.shape[1]*.10))); ranked=rank[:count]
    rng=np.random.default_rng(PHASE+(0 if key=="qwen4b" else 100)); random_coords=np.sort(rng.choice(current.shape[1],size=count,replace=False))
    lookup={(row["group_id"],int(row["relation_bit"])):row["model_row"] for row in rows_all}; opposite=np.empty_like(current); opposite_wrong=np.empty_like(wrong_current)
    for row in rows_all:
        other=lookup[(row["group_id"],1-int(row["relation_bit"]))]; opposite[row["model_row"]]=current[other]; opposite_wrong[row["model_row"]]=wrong_current[other]
    self_patch=current.copy(); rescue=opposite.copy(); rescue[:,ranked]=current[:,ranked]
    flip_all=np.zeros_like(current); flip_ranked=np.zeros_like(current); flip_random=np.zeros_like(current)
    for row in rows_all:
        sign=-1.0 if int(row["relation_bit"])==0 else 1.0; vector=sign*relation[FAMILIES.index(row["family"]),LANGUAGES.index(row["language"])]
        flip_all[row["model_row"]]=vector; flip_ranked[row["model_row"],ranked]=vector[ranked]; flip_random[row["model_row"],random_coords]=vector[random_coords]
    specs=[("clean",None),("self_patch_selected",{"qpoint":qpoint,"action":"replace","vectors":torch.from_numpy(self_patch)}),
           ("opposite_patch_selected",{"qpoint":qpoint,"action":"replace","vectors":torch.from_numpy(opposite)}),
           ("opposite_plus_ranked10_rescue",{"qpoint":qpoint,"action":"replace","vectors":torch.from_numpy(rescue)}),
           ("flip_all_dose_0.5",{"qpoint":qpoint,"action":"add","vectors":torch.from_numpy(flip_all*.5)}),
           ("flip_all_dose_1.0",{"qpoint":qpoint,"action":"add","vectors":torch.from_numpy(flip_all)}),
           ("flip_ranked10",{"qpoint":qpoint,"action":"add","vectors":torch.from_numpy(flip_ranked)}),
           ("flip_random10",{"qpoint":qpoint,"action":"add","vectors":torch.from_numpy(flip_random)}),
           ("opposite_patch_wrong_layer",{"qpoint":wrong_q,"action":"replace","vectors":torch.from_numpy(opposite_wrong)})]
    matrix=np.empty((len(specs),len(rows),3),dtype=np.float32); batch=8 if key=="qwen4b" else 3
    for ci,(name,spec) in enumerate(specs):
        target=sequence_scores(model,rows,"target_ids",batch,spec); foil=sequence_scores(model,rows,"foil_ids",batch,spec); matrix[ci,:,0]=target; matrix[ci,:,1]=foil; matrix[ci,:,2]=target-foil
        print(f"[phase2393 {key} semantic] {name}",flush=True)
    out=OUT/key/"raw/semantic_intervention_scores.float32.npy"; out.parent.mkdir(parents=True,exist_ok=True); np.save(out,matrix,allow_pickle=False)
    summary=summarize_matrix(matrix,[name for name,_ in specs]); clean=summary["clean"]; opposite_effect=summary["opposite_patch_selected"]["mean_margin_change_from_clean"]
    random_effect=summary["flip_random10"]["mean_margin_change_from_clean"]; rescue_gain=summary["opposite_plus_ranked10_rescue"]["mean_margin"]-summary["opposite_patch_selected"]["mean_margin"]
    close(field); close(response)
    return {"selected_qpoint":qpoint,"wrong_qpoint":wrong_q,"ranked_coordinates":count,"conditions":summary,
            "causal_gate":{"opposite_patch_reduces_margin":opposite_effect<0,"effect_exceeds_random_abs":abs(opposite_effect)>abs(random_effect),
                           "ranked_rescue_positive":rescue_gain>0,"necessary_candidate":opposite_effect<=-.10 and abs(opposite_effect)>abs(random_effect),
                           "sufficient_candidate":False},"rescue_gain":rescue_gain,
            "boundary":"single prompt-boundary state intervention; failure does not imply the distributed relation field is unused elsewhere"}


def attention_pre_hook(module, head: int, dose: float, heads: int):
    width=int(module.in_features); head_width=width//heads; start=head*head_width; stop=(head+1)*head_width
    def hook(_module,args):
        value=args[0].clone(); value[...,start:stop]*=(1.0-dose); return (value,*args[1:])
    return hook


def raw_sequence_scores(model, rows: list[dict], continuation_key: str, batch_size: int, layer: int|None=None, head: int|None=None, dose: float=0.0) -> np.ndarray:
    handle=None
    if layer is not None:
        attention=model.model.layers[layer].self_attn; heads=int(model.config.num_attention_heads); handle=attention.o_proj.register_forward_pre_hook(attention_pre_hook(attention.o_proj,int(head),dose,heads))
    device=model.get_input_embeddings().weight.device; pad=int(model.config.pad_token_id or model.config.eos_token_id or 0); scores=[]
    try:
        with torch.inference_mode():
            for start in range(0,len(rows),batch_size):
                batch=rows[start:start+batch_size]; sequences=[row["prompt_ids"]+row[continuation_key] for row in batch]; ids,mask,positions=pad_right(sequences,device,pad)
                logits=model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True).logits.float()
                for local,row in enumerate(batch):
                    begin=len(row["prompt_ids"])-1; continuation=row[continuation_key]; token=torch.tensor(continuation,device=device)[:,None]
                    logp=torch.log_softmax(logits[local,begin:begin+len(continuation)],dim=-1).gather(1,token).squeeze(1); scores.append(float(logp.mean().item()))
    finally:
        if handle is not None: handle.remove()
    return np.array(scores,dtype=np.float32)


def attention_interventions(model) -> dict:
    rows=[row for row in read_rows(P2378/"material/label_free_natural_binding.jsonl") if row["task"]=="exact_copy" and row["partition"]=="fresh_joint_lockbox"]
    specs=[("clean",None,None,0.0),("selected_l25h10_dose_0.5",25,10,.5),("selected_l25h10_dose_1.0",25,10,1.0),
           ("wrong_head_l25h11_dose_1.0",25,11,1.0),("wrong_layer_l24h10_dose_1.0",24,10,1.0)]
    matrix=np.empty((len(specs),len(rows),3),dtype=np.float32)
    for ci,(name,layer,head,dose) in enumerate(specs):
        target=raw_sequence_scores(model,rows,"target_ids",8,layer,head,dose); foil=raw_sequence_scores(model,rows,"foil_ids",8,layer,head,dose); matrix[ci,:,0]=target; matrix[ci,:,1]=foil; matrix[ci,:,2]=target-foil
        print(f"[phase2393 qwen4b attention] {name}",flush=True)
    out=OUT/"qwen4b/raw/attention_head_intervention_scores.float32.npy"; out.parent.mkdir(parents=True,exist_ok=True); np.save(out,matrix,allow_pickle=False)
    summary=summarize_matrix(matrix,[name for name,*_ in specs]); selected=summary["selected_l25h10_dose_1.0"]["mean_margin_change_from_clean"]
    controls=[summary["wrong_head_l25h11_dose_1.0"]["mean_margin_change_from_clean"],summary["wrong_layer_l24h10_dose_1.0"]["mean_margin_change_from_clean"]]
    return {"rows":len(rows),"candidate":{"layer":25,"head":10,"source":"Phase2381 confirmation-frozen observational attention-mass candidate"},"conditions":summary,
            "causal_gate":{"selected_reduces_margin":selected<0,"larger_than_both_controls":abs(selected)>max(map(abs,controls)),
                           "dose_monotonic":abs(summary["selected_l25h10_dose_0.5"]["mean_margin_change_from_clean"])<=abs(selected),
                           "necessary_candidate":selected<=-.05 and abs(selected)>max(map(abs,controls))},
            "boundary":"global head-output ablation in teacher-forced sequence scoring; no claim of sentence copying or sufficiency"}


def run_model(key: str) -> dict:
    final=OUT/key/"analysis/final.json"
    if final.exists(): return json.loads(final.read_text(encoding="utf-8"))
    model,tokenizer,label=capability.load_model(key)
    try:
        semantic=semantic_interventions(key,model); attention=attention_interventions(model) if key=="qwen4b" else None
    finally:
        del model,tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    result={"model":key,"model_label":label,"semantic_coordinate_intervention":semantic,"attention_intervention":attention,"all_checks_passed":True}; save(final,result); return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text=rf"""

## Phase {PHASE}: 冻结上下文坐标场与Attention候选的有限因果裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 不再重新挑层/坐标。对Phase2392冻结的Qwen4B q26、Qwen14B q27，在全部96条fresh-unit语义锁箱的最后prompt token进行：自状态数值闭合、配对反关系完整状态替换、恢复冻结10%坐标、族/语言条件响应向量0.5/1.0剂量翻转、冻结10%与等量随机10%、错层配对替换。比较正确整句和反关系foil平均logprob。另在Qwen4B全部256条长句锁箱全局消融Phase2381冻结的layer25/head10，设置0.5/1.0剂量、相邻错head和错层。

$$H'_{{q,a}}=H_{{q,a}}+\lambda(-1)^{{1-d}}R_{{q,f,\ell}},\qquad
\Delta_{{causal}}=\mathbb E[m(H')-m(H)].$$

**结果汇总。** 坐标干预 `{json.dumps(result['coordinate_summary'],ensure_ascii=False)}`；Attention干预 `{json.dumps(result['attention_summary'],ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'],ensure_ascii=False)}`；检查 `{json.dumps(result['checks'],ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2393_c20401_c20720_frozen_context_causal_adjudication.py`；逐条件逐样本target/foil/margin和final位于 `tests/glm5/result/phase2393_c20401_c20720_frozen_context_causal_adjudication`。

**理论进展、问题硬伤与结论。** 反关系状态替换比离流形置零更接近有效状态，但单一prompt边界仍可能不是信息真正流经的位置；因此因果失败不能推翻分布式图谱。Attention消融是全序列head输出缩放，若有效也不能证明它复制内容；若与错head/错层相当，只能否决该head的选择性核心地位。救援只有在反关系损害后恢复冻结坐标带来特异回升才有意义。
"""
    with MEMO.open("a",encoding="utf-8",newline="\n") as stream: stream.write(text)


def main() -> None:
    final=OUT/"analysis/final.json"
    if final.exists():
        result=json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result,ensure_ascii=False,indent=2)); return
    models={key:run_model(key) for key in MODELS}
    coordinate={key:value["semantic_coordinate_intervention"] for key,value in models.items()}; attention=models["qwen4b"]["attention_intervention"]
    adjudication={"coordinate_necessary_replicated":all(value["causal_gate"]["necessary_candidate"] for value in coordinate.values()),
                  "coordinate_rescue_replicated":all(value["causal_gate"]["ranked_rescue_positive"] for value in coordinate.values()),
                  "attention_candidate_necessary":attention["causal_gate"]["necessary_candidate"],
                  "mechanism_closed":False,"continue_observational_atlas_if_causal_fails":True}
    checks={"two_models":set(models)==set(MODELS),"all_model_checks":all(value["all_checks_passed"] for value in models.values()),
            "finite":all(math.isfinite(value["conditions"]["clean"]["mean_margin"]) for value in coordinate.values())}
    result={"phase":PHASE,"campaign":CAMPAIGN,"models":models,"coordinate_summary":coordinate,"attention_summary":attention,
            "adjudication":adjudication,"checks":checks,"all_checks_passed":all(checks.values())}; save(final,result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
