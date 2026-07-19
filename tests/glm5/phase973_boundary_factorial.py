#!/usr/bin/env python3
"""Phase 973 extension: separate semantic completion from final punctuation.

For each of the same 160 independent items, construct a 2 x 3 elementary grid:
semantic status {incomplete, complete} x suffix {none, period, comma}.
No intervention is used. All valid EOS ids and explicit attention masks are used.
"""
from __future__ import annotations
import gc, json, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"tests"/"glm5"))
from model_utils import load_model, release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log
from phase973_conditional_trajectory import (
    build_dataset, batch_iter, encode_batch, get_eos_ids, summarize_logits,
    regression_checks,
)

RESULT_DIR=Path("tests/glm5/result/phase973_conditional_trajectory")
CONDITIONS=["incomplete_none","incomplete_period","incomplete_comma",
            "complete_none","complete_period","complete_comma"]


def strip_suffix(text):
    return re.sub(r"[\s.!?,;:]+$", "", text)


def build_rows(items):
    rows=[]
    for x in items:
        u=strip_suffix(x["states"]["unfinished"])
        c=strip_suffix(x["states"]["just_complete"])
        texts={"incomplete_none":u, "incomplete_period":u+".", "incomplete_comma":u+",",
               "complete_none":c, "complete_period":c+".", "complete_comma":c+","}
        for cond,text in texts.items():
            rows.append({"id":x["id"],"task":x["task"],"prompt_template":x["prompt_template"],
                         "condition":cond,"text":text})
    return rows


def run(model_name):
    ensure_dir(RESULT_DIR); items=build_dataset(); rows=build_rows(items); t0=time.time()
    model,tok,device=load_model(model_name); eos_ids=get_eos_ids(model,tok)
    # Reuse the same pre-flight checks before the new expensive grid.
    checks=regression_checks(model,tok,device,[],eos_ids,items)
    if not checks["mask_consistency_pass_le_0.25"]: raise RuntimeError(checks)
    bs=8 if model_name=="glm4" else 16; raw=[]
    for batch in batch_iter(rows,bs):
        enc=encode_batch(tok,[r["text"] for r in batch],device)
        pos=enc["attention_mask"].sum(-1)-1
        with torch.no_grad(): logits=model(**enc,use_cache=False).logits
        s=summarize_logits(logits,eos_ids,pos)
        for i,r in enumerate(batch):
            raw.append({**{k:r[k] for k in ["id","task","prompt_template","condition"]},
                        "gap":s["gap"][i],"eos_logit":s["eos_logit"][i],
                        "eos_rank":s["eos_rank"][i],"eos_id":s["eos_id"][i],
                        "competitor_id":s["top_id"][i],"eos_won":s["gap"][i]<0})
    by=defaultdict(dict)
    for r in raw: by[r["id"]][r["condition"]]=r
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
    summary={}
    for f in fields:
        vals=[p[f] for p in pairs]
        summary[f]={"mean":float(np.mean(vals)),"negative_rate":float(np.mean([v<0 for v in vals])),
                    "le_minus_0_5_rate":float(np.mean([v<=-0.5 for v in vals]))}
    task_summary={}
    for task in sorted(set(x["task"] for x in items)):
        ps=[p for p in pairs if p["task"]==task]
        task_summary[task]={f:float(np.mean([p[f] for p in ps])) for f in fields}
    result={"phase":973,"extension":"semantic_x_punctuation_factorial","model":model_name,
            "n_items":160,"conditions":CONDITIONS,"eos_token_ids":eos_ids,
            "attention_mask_explicit":True,"regression_checks":checks,
            "summary":summary,"task_summary":task_summary,"pairs":pairs,"raw":raw,
            "elapsed_seconds":time.time()-t0}
    out=RESULT_DIR/f"{model_name}_boundary_factorial.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    release_model(model);gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    log(f"Saved {out}; elapsed={result['elapsed_seconds']/60:.1f} min")


if __name__=="__main__": run(sys.argv[1] if len(sys.argv)>1 else "glm4")
