#!/usr/bin/env python3
"""Phase 973 natural greedy trajectories on all 160 prompts (no intervention)."""
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
from phase973_conditional_trajectory import build_dataset,encode_batch,get_eos_ids,summarize_logits,regression_checks

OUT=Path("tests/glm5/result/phase973_conditional_trajectory")


def run(model_name):
    ensure_dir(OUT);items=build_dataset();t0=time.time()
    model,tok,device=load_model(model_name);eos_ids=get_eos_ids(model,tok)
    checks=regression_checks(model,tok,device,[],eos_ids,items)
    if not checks["mask_consistency_pass_le_0.25"]:raise RuntimeError(checks)
    rows=[]
    for idx,item in enumerate(items):
        enc=encode_batch(tok,[item["prompt"]],device)
        with torch.no_grad():
            out=model.generate(**enc,max_new_tokens=24,do_sample=False,
                pad_token_id=tok.pad_token_id,eos_token_id=eos_ids,
                return_dict_in_generate=True,output_scores=True)
        gen_ids=out.sequences[0,enc["input_ids"].shape[1]:]
        plain=tok.decode(gen_ids,skip_special_tokens=True)
        raw=tok.decode(gen_ids,skip_special_tokens=False)
        expected=item["answer"].lower() in plain.lower()
        eos_positions=[i for i,t in enumerate(gen_ids.tolist()) if t in eos_ids]
        completion_step=None;punct_after_completion=None;prefix=[]
        punctuation_steps=[];gap_series=[]
        for step,(tid,score) in enumerate(zip(gen_ids.tolist(),out.scores)):
            sm=summarize_logits(score[:,None,:],eos_ids)
            gap_series.append(sm["gap"][0]);prefix.append(tid)
            text=tok.decode(prefix,skip_special_tokens=True)
            token_text=tok.decode([tid])
            if re.search(r"[.!?]\s*$",token_text):punctuation_steps.append(step)
            if completion_step is None and item["answer"].lower() in text.lower():completion_step=step
            if completion_step is not None and punct_after_completion is None and step>=completion_step and re.search(r"[.!?]\s*$",token_text):
                punct_after_completion=step
        rows.append({"id":item["id"],"task":item["task"],"prompt_template":item["prompt_template"],
                     "answer":item["answer"],"generated":raw,"plain":plain,
                     "has_expected":expected,"has_eos":bool(eos_positions),
                     "first_eos_step":eos_positions[0] if eos_positions else None,
                     "completion_step":completion_step,"punctuation_steps":punctuation_steps,
                     "first_punctuation_after_completion":punct_after_completion,
                     "n_tokens":len(gen_ids),"gap_series":gap_series})
        if (idx+1)%40==0:log(f"  natural {idx+1}/160")
    summary={"n":160,"expected_rate":float(np.mean([r["has_expected"] for r in rows])),
             "eos_rate":float(np.mean([r["has_eos"] for r in rows])),
             "expected_and_eos_rate":float(np.mean([r["has_expected"] and r["has_eos"] for r in rows])),
             "completion_reached_rate":float(np.mean([r["completion_step"] is not None for r in rows])),
             "punctuation_after_completion_rate":float(np.mean([r["first_punctuation_after_completion"] is not None for r in rows])),
             "mean_tokens":float(np.mean([r["n_tokens"] for r in rows]))}
    task_summary={}
    for task in sorted(set(r["task"] for r in rows)):
        rs=[r for r in rows if r["task"]==task]
        task_summary[task]={"n":len(rs),"expected":sum(r["has_expected"] for r in rs),
          "eos":sum(r["has_eos"] for r in rs),"expected_and_eos":sum(r["has_expected"] and r["has_eos"] for r in rs),
          "punctuation_after_completion":sum(r["first_punctuation_after_completion"] is not None for r in rs)}
    result={"phase":973,"model":model_name,"kind":"natural_raw_greedy","n_items":160,
            "max_new_tokens":24,"intervention":"none","external_eos_bias":"none",
            "eos_token_ids":eos_ids,"attention_mask_explicit":True,"regression_checks":checks,
            "summary":summary,"task_summary":task_summary,"rows":rows,"elapsed_seconds":time.time()-t0}
    path=OUT/f"{model_name}_natural_trace.json";path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    release_model(model);gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    log(f"Saved {path}; elapsed={result['elapsed_seconds']/60:.1f} min")


if __name__=="__main__":run(sys.argv[1] if len(sys.argv)>1 else "glm4")
