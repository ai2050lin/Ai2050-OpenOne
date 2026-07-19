#!/usr/bin/env python3
"""Phase 975 DS7B-specific thinking/final-answer boundary audit.

DeepSeek-R1-Distill-Qwen opens a <think> block in its native generation
template.  This script therefore refuses to treat a 64-token open-thinking
trajectory as evidence about final-answer termination.  It measures synthetic,
teacher-forced stage boundaries separately and then runs a larger natural
rollout audit to observe whether/when the model actually closes thinking,
reaches a final answer, and emits EOS.

The teacher-forced reasoning sentence is an explicit experimental scaffold,
not a claim about a natural latent state.  Natural trajectories are reported
independently.
"""
from __future__ import annotations

import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log
from phase973_conditional_trajectory import build_dataset, get_eos_ids, summarize_logits


OUT = Path("tests/glm5/result/phase975_ds_two_stage")
MODEL_NAME = "deepseek7b"


def native_prefix(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)


def final_contents(item):
    answer = re.sub(r"[\s.!?;:]+$", "", item["answer"])
    words = answer.split()
    partial = " ".join(words[:max(1, len(words)//2)]) if len(words) > 1 else ""

    def remainder(name):
        text = item["states"][name]
        return text[len(item["prompt"]):].lstrip()

    return {"final_U": partial, "final_C": answer, "final_P": answer + ".",
            "final_X": remainder("continuation_incomplete"),
            "final_XC": remainder("continuation_complete")}


def teacher_texts(tok, item):
    prefix = native_prefix(tok, item["prompt"])
    reasoning_u = "We need to identify the requested"
    reasoning_p = "We need to identify the requested short answer."
    transition = reasoning_p + "</think>\n\n"
    rows = {
        "thinking_open_U": prefix + reasoning_u,
        "thinking_sentence_P": prefix + reasoning_p,
        # After </think>, the next required event is a final answer, not EOS.
        "after_think_close": prefix + transition,
    }
    for state, content in final_contents(item).items():
        rows[state] = prefix + transition + content
    return rows


def encode(tok, text, device):
    enc = tok(text, add_special_tokens=False, return_tensors="pt",
              return_attention_mask=True)
    return {k: v.to(device) for k, v in enc.items()}


def teacher_forced(model, tok, device, eos_ids, items):
    raw = []
    for idx, item in enumerate(items):
        for stage, text in teacher_texts(tok, item).items():
            enc = encode(tok, text, device)
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            sm = summarize_logits(logits, eos_ids)
            raw.append({"id": item["id"], "task": item["task"],
                        "prompt_template": item["prompt_template"], "stage": stage,
                        "gap": float(sm["gap"][0]), "eos_logit": float(sm["eos_logit"][0]),
                        "eos_rank": int(sm["eos_rank"][0]), "eos_won": bool(sm["gap"][0] < 0),
                        "competitor_id": int(sm["top_id"][0])})
        if (idx + 1) % 40 == 0:
            log(f"  DS two-stage teacher forced {idx+1}/{len(items)}")
    summary = {}
    for stage in teacher_texts(tok, items[0]):
        vals = [r for r in raw if r["stage"] == stage]
        summary[stage] = {"n": len(vals), "mean_gap": float(np.mean([r["gap"] for r in vals])),
                          "eos_win_rate": float(np.mean([r["eos_won"] for r in vals])),
                          "mean_eos_rank": float(np.mean([r["eos_rank"] for r in vals]))}
    return raw, summary


def generate_one(model, tok, device, eos_ids, item, max_new):
    prefix = native_prefix(tok, item["prompt"])
    enc = encode(tok, prefix, device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos_ids,
                             return_dict_in_generate=True)
    ids = out.sequences[0, enc["input_ids"].shape[1]:]
    decoded = tok.decode(ids, skip_special_tokens=False)
    plain = tok.decode(ids, skip_special_tokens=True)
    closed = "</think>" in decoded
    final_text = decoded.split("</think>", 1)[1] if closed else ""
    id_list = ids.tolist()
    return {"id": item["id"], "task": item["task"],
            "prompt_template": item["prompt_template"], "answer": item["answer"],
            "max_new_tokens": max_new, "generated": decoded, "plain": plain,
            "thinking_closed": closed, "has_eos": any(int(t) in eos_ids for t in id_list),
            "final_has_expected": closed and item["answer"].lower() in final_text.lower(),
            "anywhere_has_expected": item["answer"].lower() in plain.lower(),
            "n_tokens": len(id_list), "hit_budget": len(id_list) >= max_new}


def natural_summary(rows):
    return {"n": len(rows),
            "thinking_close_rate": float(np.mean([r["thinking_closed"] for r in rows])),
            "eos_rate": float(np.mean([r["has_eos"] for r in rows])),
            "final_expected_rate": float(np.mean([r["final_has_expected"] for r in rows])),
            "final_expected_and_eos_rate": float(np.mean([
                r["final_has_expected"] and r["has_eos"] for r in rows])),
            "budget_hit_rate": float(np.mean([r["hit_budget"] for r in rows])),
            "mean_tokens": float(np.mean([r["n_tokens"] for r in rows]))}


def natural_audit(model, tok, device, eos_ids, items):
    rows256 = []
    for idx, item in enumerate(items):
        rows256.append(generate_one(model, tok, device, eos_ids, item, 256))
        if (idx + 1) % 24 == 0:
            log(f"  DS natural 256 {idx+1}/{len(items)}")
    s256 = natural_summary(rows256)

    # Pre-registered extension: if >25% remain open and budget-limited, rerun
    # exactly those rows to 512; do not select by answer correctness.
    unresolved = [r for r in rows256 if not r["thinking_closed"] and r["hit_budget"]]
    rows512 = []
    if len(unresolved) / max(len(rows256), 1) > 0.25:
        lookup = {x["id"]: x for x in items}
        for idx, old in enumerate(unresolved):
            rows512.append(generate_one(model, tok, device, eos_ids, lookup[old["id"]], 512))
            if (idx + 1) % 24 == 0:
                log(f"  DS natural extension 512 {idx+1}/{len(unresolved)}")
    replacement = {r["id"]: r for r in rows512}
    final_rows = [replacement.get(r["id"], r) for r in rows256]
    return {"initial_256": {"summary": s256, "rows": rows256},
            "extension_rule_triggered": bool(rows512), "extended_n": len(rows512),
            "final_mixed_budget": {"summary": natural_summary(final_rows), "rows": final_rows}}


def run():
    ensure_dir(OUT)
    t0 = time.time()
    items = build_dataset()
    # Natural audit uses the 96 intervention-holdout rows (indices 08..19/task).
    natural_items = []
    for task in sorted({x["task"] for x in items}):
        natural_items.extend([x for x in items if x["task"] == task][8:])
    model, tok, device = load_model(MODEL_NAME)
    eos_ids = get_eos_ids(model, tok)
    raw, summary = teacher_forced(model, tok, device, eos_ids, items)
    result = {"phase": 975, "model": MODEL_NAME,
              "experiment": "thinking_vs_final_answer_two_stage",
              "eos_token_ids": eos_ids, "teacher_forced_n": len(items),
              "teacher_scaffold_warning": "generic synthetic reasoning and explicit </think>; not natural R1 state",
              "teacher_summary": summary, "teacher_rows": raw,
              "natural_n": len(natural_items),
              "natural_split_warning": "texts were seen in Phase974; rollout budget/stage audit is new"}
    path = OUT / "deepseek7b_result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["natural"] = natural_audit(model, tok, device, eos_ids, natural_items)
    result["elapsed_seconds"] = time.time() - t0
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Saved {path}; elapsed={result['elapsed_seconds']/60:.1f} min")


if __name__ == "__main__":
    run()
