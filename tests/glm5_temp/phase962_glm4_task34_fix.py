#!/usr/bin/env python3
"""Phase 962 GLM4 task3+4 re-run with fixed reverse lock hook."""
import sys, json, time, gc
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U
from phase951_protocol_atlas import ensure_dir
from phase962_eos_promoter_search import (
    log, get_head_dims, evaluate_strict_clean, make_head_hook, make_channel_hook,
    MODEL_HEADS, EN_PROMPTS_50, MAX_TOKENS, RESULT_DIR
)

model_name = "glm4"

def run():
    log(f"\n{'='*60}")
    log(f"Phase 962 GLM4 RE-RUN (fixed hook): task3+task4")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L  (load: {time.time()-t0:.0f}s)")

    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    heads = MODEL_HEADS[model_name]
    L0, H0 = heads[0]["layer"], heads[0]["head"]
    sc, ec = H0 * d_head, (H0 + 1) * d_head

    prompts = EN_PROMPTS_50[:3]
    results = {"model": model_name}
    t_start = time.time()

    # ---- Task 3: Reverse lock (FIXED: scale=1-λ via o_proj hook) ----
    log("  Task 3 (FIXED): Reverse lock intervention...")
    lambdas = [0.0, 0.5, 1.0, 2.0]
    all_r3 = []
    for pi, prompt in enumerate(prompts):
        for lam in lambdas:
            handles = []
            if lam != 0.0:
                handles.append(layers[L0].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc, ec, 1.0 - lam)))
            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=MAX_TOKENS,
                                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                ng = len(gt)
            except Exception as e:
                gen = f"ERROR: {e}"; he = False; ng = 0
            for h in handles: h.remove()
            is_ascii = all(ord(c) < 256 for c in gen)
            all_r3.append({"prompt": prompt, "lambda": lam, "generated": gen[:200],
                           "has_eos": he, "n_tokens": ng, "lang_switched": not is_ascii})
        log(f"    {pi+1}/3 prompts")

    s3 = {}
    for lam in lambdas:
        lr = [r for r in all_r3 if r["lambda"] == lam]
        s3[f"lam_{lam}"] = {"eos_rate": np.mean([r["has_eos"] for r in lr]) if lr else 0,
            "lang_switch_rate": np.mean([r["lang_switched"] for r in lr]) if lr else 0,
            "mean_tokens": np.mean([r["n_tokens"] for r in lr]) if lr else 0}
    results["task3"] = {"task": "task3_reverse_lock_fixed", "model": model_name,
        "primary_head": f"L{L0}_H{H0}", "lambdas": lambdas, "summary": s3, "raw_results": all_r3}
    (model_dir / "task3_reverse_lock.json").write_text(
        json.dumps(results["task3"], ensure_ascii=False, indent=2), encoding="utf-8")
    for lam in lambdas:
        sv = s3[f"lam_{lam}"]
        log(f"    λ={lam:.1f}: eos={sv['eos_rate']:.2f}  switch={sv['lang_switch_rate']:.2f}  toks={sv['mean_tokens']:.1f}")
    log(f"    Sample (p0):")
    for r in all_r3:
        if r["prompt"] == prompts[0]:
            log(f"      λ={r['lambda']:.1f}: eos={r['has_eos']}  switch={r['lang_switched']}  text={r['generated'][:80]}")
    log(f"    Task 3 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 4: Combined (FIXED) ----
    log("  Task 4 (FIXED): Combined intervention...")
    # Use L39_C1226 (best EOS channel from task2)
    eos_ch_layer, eos_ch_idx = 39, 1226
    conditions = [
        ("normal", 0.0, 1.0, None, None),
        ("ablate_lock", 1.0, 1.0, None, None),
        ("reverse_lock_2.0", 2.0, 1.0, None, None),
        ("boost_eos_ch", 0.0, 3.0, eos_ch_layer, eos_ch_idx),
        ("rev2.0+boost_eos", 2.0, 3.0, eos_ch_layer, eos_ch_idx),
        ("ablate+boost_eos", 1.0, 3.0, eos_ch_layer, eos_ch_idx),
    ]
    all_r4 = []
    for pi, prompt in enumerate(prompts):
        for cn, lam, cs, cl, ci in conditions:
            handles = []
            if lam != 0.0:
                handles.append(layers[L0].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc, ec, 1.0 - lam)))
            if cs != 1.0 and cl is not None:
                handles.append(layers[cl].mlp.down_proj.register_forward_pre_hook(
                    make_channel_hook([ci], cs)))
            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=MAX_TOKENS,
                                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                ng = len(gt)
            except Exception as e:
                gen = f"ERROR: {e}"; he = False; ng = 0
            for h in handles: h.remove()
            ce = evaluate_strict_clean(prompt, gen, he, ng)
            all_r4.append({"prompt": prompt, "condition": cn, "generated": gen[:200],
                           "has_eos": he, "n_tokens": ng, "strict_clean": ce["strict_clean"],
                           "lang_switched": not ce["is_ascii"]})
        log(f"    {pi+1}/3 prompts")

    cagg = defaultdict(lambda: {"eos": 0, "clean": 0, "switch": 0, "n": 0, "toks": []})
    for r in all_r4:
        c = r["condition"]; cagg[c]["eos"] += int(r["has_eos"])
        cagg[c]["clean"] += int(r["strict_clean"])
        cagg[c]["switch"] += int(r["lang_switched"])
        cagg[c]["n"] += 1; cagg[c]["toks"].append(r["n_tokens"])
    s4 = {c: {"eos_rate": d["eos"]/max(d["n"],1), "strict_clean_rate": d["clean"]/max(d["n"],1),
              "lang_switch_rate": d["switch"]/max(d["n"],1),
              "mean_tokens": float(np.mean(d["toks"])) if d["toks"] else 0}
          for c, d in cagg.items()}
    results["task4"] = {"task": "task4_combined_fixed", "model": model_name,
        "primary_head": f"L{L0}_H{H0}", "eos_channel": f"L{eos_ch_layer}_C{eos_ch_idx}",
        "summary": s4, "raw_results": all_r4}
    (model_dir / "task4_combined.json").write_text(
        json.dumps(results["task4"], ensure_ascii=False, indent=2), encoding="utf-8")
    for c in [c[0] for c in conditions]:
        sv = s4.get(c, {})
        log(f"    {c:25s}: eos={sv.get('eos_rate',0):.2f}  clean={sv.get('strict_clean_rate',0):.2f}  "
            f"switch={sv.get('lang_switch_rate',0):.2f}  toks={sv.get('mean_tokens',0):.1f}")
    log(f"    Sample (p0):")
    for r in all_r4:
        if r["prompt"] == prompts[0]:
            log(f"      {r['condition']:25s}: eos={r['has_eos']}  clean={r['strict_clean']}  text={r['generated'][:80]}")
    log(f"    Task 4 done ({time.time()-t_start:.0f}s)")

    elapsed = time.time() - t_start
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)

    # Merge with existing results
    existing_path = RESULT_DIR / f"{model_name}_result.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        existing["task3"] = results["task3"]
        existing["task4"] = results["task4"]
        existing["task3_4_fixed"] = True
        existing_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        log(f"Updated: {existing_path}")


if __name__ == "__main__":
    run()
