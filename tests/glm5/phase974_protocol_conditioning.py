#!/usr/bin/env python3
"""Phase 974: native-chat protocol conditioning of boundary dynamics.

Repeats the 160-item semantic x punctuation grid and raw greedy trajectories
inside each model's native chat template. Plain-text Phase 973 results remain
separate and are never overwritten.
"""
from __future__ import annotations
import gc,json,re,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tests"/"glm5"))
from model_utils import load_model,release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log
from phase973_conditional_trajectory import build_dataset,batch_iter,get_eos_ids,summarize_logits

OUT=Path("tests/glm5/result/phase974_protocol_conditioning")
CONDS=["incomplete_none","incomplete_period","incomplete_comma",
       "complete_none","complete_period","complete_comma"]


def chat_prefix(tok,model_name,prompt,teacher_forced=False):
    messages=[{"role":"user","content":prompt}]
    kwargs={"enable_thinking":False} if model_name=="qwen3" else {}
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kwargs)
    # DeepSeek-R1's template always opens a think block. For teacher-forced
    # final-answer states, close an empty block explicitly; natural generation
    # keeps the unmodified native template.
    if model_name=="deepseek7b" and teacher_forced:
        text += "</think>\n\n"
    return text


def encode_protocol(tok,texts,device):
    old=tok.padding_side;tok.padding_side="right"
    enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False,
            return_attention_mask=True)
    tok.padding_side=old
    return {k:v.to(device) for k,v in enc.items()}


def assistant_contents(answer):
    words=answer.split();partial=" ".join(words[:max(1,len(words)//2)]) if len(words)>1 else ""
    u=re.sub(r"[\s.!?,;:]+$","",partial)
    c=re.sub(r"[\s.!?,;:]+$","",answer)
    return {"incomplete_none":u,"incomplete_period":u+".","incomplete_comma":u+",",
            "complete_none":c,"complete_period":c+".","complete_comma":c+","}


def relevant_mask_check(model,tok,device,eos_ids,texts):
    e1=encode_protocol(tok,[texts[0]],device);e2=encode_protocol(tok,texts[:2],device)
    with torch.no_grad():o1=model(**e1,use_cache=False).logits;o2=model(**e2,use_cache=False).logits
    p1=e1["attention_mask"].sum(-1)-1;p2=e2["attention_mask"].sum(-1)-1
    s1=summarize_logits(o1,eos_ids,p1);s2=summarize_logits(o2[:1],eos_ids,p2[:1])
    return {"gap_abs_diff":abs(s1["gap"][0]-s2["gap"][0]),
            "eos_abs_diff":abs(s1["eos_logit"][0]-s2["eos_logit"][0]),
            "competitor_same":s1["top_id"][0]==s2["top_id"][0]}


def run(model_name):
    ensure_dir(OUT);items=build_dataset();t0=time.time()
    model,tok,device=load_model(model_name);eos_ids=get_eos_ids(model,tok)
    tf_rows=[]
    for x in items:
        prefix=chat_prefix(tok,model_name,x["prompt"],teacher_forced=True)
        for cond,content in assistant_contents(x["answer"]).items():
            tf_rows.append({"id":x["id"],"task":x["task"],"prompt_template":x["prompt_template"],
                            "condition":cond,"text":prefix+content})
    mask_check=relevant_mask_check(model,tok,device,eos_ids,[r["text"] for r in tf_rows])
    # Native-chat batches showed architecture-dependent BF16 kernel drift.
    # Use batch=1 for the decisive protocol comparison; mask_check is retained
    # as a diagnostic rather than relaxed after observation.
    bs=1;raw=[]
    for batch in batch_iter(tf_rows,bs):
        enc=encode_protocol(tok,[r["text"] for r in batch],device);pos=enc["attention_mask"].sum(-1)-1
        with torch.no_grad():logits=model(**enc,use_cache=False).logits
        s=summarize_logits(logits,eos_ids,pos)
        for i,r in enumerate(batch):
            raw.append({**{k:r[k] for k in ["id","task","prompt_template","condition"]},
                        "gap":s["gap"][i],"eos_logit":s["eos_logit"][i],"eos_rank":s["eos_rank"][i],
                        "eos_id":s["eos_id"][i],"competitor_id":s["top_id"][i],"eos_won":s["gap"][i]<0})
    by=defaultdict(dict)
    for r in raw:by[r["id"]][r["condition"]]=r
    pairs=[]
    for x in items:
        v=by[x["id"]]
        pairs.append({"id":x["id"],"task":x["task"],"prompt_template":x["prompt_template"],
          "semantic_effect_none":v["complete_none"]["gap"]-v["incomplete_none"]["gap"],
          "semantic_effect_period":v["complete_period"]["gap"]-v["incomplete_period"]["gap"],
          "punctuation_effect_incomplete_period":v["incomplete_period"]["gap"]-v["incomplete_none"]["gap"],
          "punctuation_effect_complete_period":v["complete_period"]["gap"]-v["complete_none"]["gap"],
          "punctuation_effect_incomplete_comma":v["incomplete_comma"]["gap"]-v["incomplete_none"]["gap"],
          "punctuation_effect_complete_comma":v["complete_comma"]["gap"]-v["complete_none"]["gap"],
          "period_semantic_interaction":(v["complete_period"]["gap"]-v["complete_none"]["gap"])
            -(v["incomplete_period"]["gap"]-v["incomplete_none"]["gap"])})
    fields=[k for k in pairs[0] if k not in ["id","task","prompt_template"]]
    factor_summary={f:{"mean":float(np.mean([p[f] for p in pairs])),
                       "negative_rate":float(np.mean([p[f]<0 for p in pairs]))} for f in fields}

    # Native chat natural trajectories, no intervention and no EOS bias.
    natural=[];max_new=64 if model_name=="deepseek7b" else 32
    for idx,x in enumerate(items):
        prefix=chat_prefix(tok,model_name,x["prompt"],teacher_forced=False)
        enc=encode_protocol(tok,[prefix],device)
        with torch.no_grad():out=model.generate(**enc,max_new_tokens=max_new,do_sample=False,
            pad_token_id=tok.pad_token_id,eos_token_id=eos_ids,
            return_dict_in_generate=True,output_scores=True)
        ids=out.sequences[0,enc["input_ids"].shape[1]:];plain=tok.decode(ids,skip_special_tokens=True)
        expected=x["answer"].lower() in plain.lower();eos_pos=[i for i,t in enumerate(ids.tolist()) if t in eos_ids]
        gaps=[summarize_logits(sc[:,None,:],eos_ids)["gap"][0] for sc in out.scores]
        natural.append({"id":x["id"],"task":x["task"],"prompt_template":x["prompt_template"],
                        "answer":x["answer"],"generated":tok.decode(ids,skip_special_tokens=False),
                        "plain":plain,"has_expected":expected,"has_eos":bool(eos_pos),
                        "first_eos_step":eos_pos[0] if eos_pos else None,"n_tokens":len(ids),"gap_series":gaps})
        if (idx+1)%40==0:log(f"  {model_name} chat natural {idx+1}/160")
    natural_summary={"n":160,"expected_rate":float(np.mean([r["has_expected"] for r in natural])),
                     "eos_rate":float(np.mean([r["has_eos"] for r in natural])),
                     "expected_and_eos_rate":float(np.mean([r["has_expected"] and r["has_eos"] for r in natural])),
                     "mean_tokens":float(np.mean([r["n_tokens"] for r in natural]))}
    result={"phase":974,"model":model_name,"protocol":"native_chat_template",
            "qwen_thinking_disabled":model_name=="qwen3",
            "deepseek_teacher_forced_empty_think_closed":model_name=="deepseek7b",
            "n_items":160,"eos_token_ids":eos_ids,"attention_mask_explicit":True,
            "teacher_forced_batch_size":1,"mask_check":mask_check,
            "factorial_summary":factor_summary,"factorial_pairs":pairs,
            "factorial_raw":raw,"natural_max_new_tokens":max_new,"natural_summary":natural_summary,
            "natural_rows":natural,"elapsed_seconds":time.time()-t0}
    path=OUT/f"{model_name}_result.json";path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    release_model(model);gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    log(f"Saved {path}; elapsed={result['elapsed_seconds']/60:.1f} min")


if __name__=="__main__":run(sys.argv[1] if len(sys.argv)>1 else "glm4")
