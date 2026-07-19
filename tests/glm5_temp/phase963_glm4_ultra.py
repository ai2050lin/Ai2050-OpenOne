#!/usr/bin/env python3
"""Phase 963 GLM4 ultra-minimal: Task1(boost EOS head) + Task5(margin) only."""
import sys, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, get_layers, get_model_info, release_model
from phase951_protocol_atlas import ensure_dir
from phase963_eos_boost import (
    log, get_head_dims, evaluate_strict_clean, make_head_hook, make_channel_hook,
    EN_PROMPTS_50, MAX_TOKENS, RESULT_DIR963
)

model_name = "glm4"

def run():
    log(f"Phase 963 GLM4 ultra-minimal")
    model_dir = RESULT_DIR963 / model_name
    ensure_dir(model_dir)
    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    log(f"  {info.model_class}, {info.n_layers}L  (load: {time.time()-t0:.0f}s)")

    # GLM4: EOS head L38_H6, lock head L39_H21, EOS channel L39_C1226
    L_eos, H_eos = 38, 6
    L_lock, H_lock = 39, 21
    L_ch, C_ch = 39, 1226
    sc_eos = H_eos * d_head; ec_eos = sc_eos + d_head
    sc_lock = H_lock * d_head; ec_lock = sc_lock + d_head

    prompts = EN_PROMPTS_50[:2]
    results = {"model": model_name}
    t_start = time.time()

    # ---- Task 1: Boost EOS head L38_H6 ----
    log(f"  Task 1: Boost EOS head L{L_eos}_H{H_eos}...")
    boost_scales = [1.0, 3.0, 5.0]
    all_r1 = []
    for pi, prompt in enumerate(prompts):
        for bs in boost_scales:
            handles = []
            if bs != 1.0:
                handles.append(layers[L_eos].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc_eos, ec_eos, bs)))
            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=30,
                                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                ng = len(gt)
            except Exception as e:
                gen = f"ERROR: {e}"; he = False; ng = 0
            for h in handles: h.remove()
            ce = evaluate_strict_clean(prompt, gen, he, ng)
            all_r1.append({"prompt": prompt, "boost_scale": bs, "generated": gen[:200],
                           "has_eos": he, "n_tokens": ng, "strict_clean": ce["strict_clean"],
                           "lang_switched": not ce["is_ascii"]})
            log(f"    p{pi} bs={bs}: eos={he} clean={ce['strict_clean']} text={gen[:60]}")

    s1 = {}
    for bs in boost_scales:
        br = [r for r in all_r1 if r["boost_scale"] == bs]
        s1[f"scale_{bs}"] = {"eos_rate": np.mean([r["has_eos"] for r in br]) if br else 0,
            "strict_clean_rate": np.mean([r["strict_clean"] for r in br]) if br else 0,
            "lang_switch_rate": np.mean([r["lang_switched"] for r in br]) if br else 0,
            "mean_tokens": np.mean([r["n_tokens"] for r in br]) if br else 0}
    results["task1"] = {"task": "task1_boost_eos_head", "eos_head": f"L{L_eos}_H{H_eos}",
        "scales": boost_scales, "summary": s1, "raw_results": all_r1}
    log(f"    Task 1 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 2: Boost EOS head + Ablate lock ----
    log(f"  Task 2: Boost EOS head + Ablate lock L{L_lock}_H{H_lock}...")
    conditions2 = [("normal", 1.0, 1.0), ("boost_eos_3x", 3.0, 1.0),
                   ("ablate_lock", 1.0, 0.0), ("boost_eos+ablate_lock", 3.0, 0.0),
                   ("boost_eos5x+ablate_lock", 5.0, 0.0)]
    all_r2 = []
    for pi, prompt in enumerate(prompts):
        for cn, es, ls in conditions2:
            handles = []
            if es != 1.0:
                handles.append(layers[L_eos].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc_eos, ec_eos, es)))
            if ls != 1.0:
                handles.append(layers[L_lock].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc_lock, ec_lock, ls)))
            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=30,
                                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                ng = len(gt)
            except Exception as e:
                gen = f"ERROR: {e}"; he = False; ng = 0
            for h in handles: h.remove()
            ce = evaluate_strict_clean(prompt, gen, he, ng)
            all_r2.append({"prompt": prompt, "condition": cn, "generated": gen[:200],
                           "has_eos": he, "n_tokens": ng, "strict_clean": ce["strict_clean"],
                           "lang_switched": not ce["is_ascii"]})
            log(f"    p{pi} {cn}: eos={he} clean={ce['strict_clean']} text={gen[:50]}")

    cagg = defaultdict(lambda: {"eos": 0, "clean": 0, "switch": 0, "n": 0, "toks": []})
    for r in all_r2:
        c = r["condition"]; cagg[c]["eos"] += int(r["has_eos"])
        cagg[c]["clean"] += int(r["strict_clean"])
        cagg[c]["switch"] += int(r["lang_switched"])
        cagg[c]["n"] += 1; cagg[c]["toks"].append(r["n_tokens"])
    s2 = {c: {"eos_rate": d["eos"]/max(d["n"],1), "strict_clean_rate": d["clean"]/max(d["n"],1),
              "lang_switch_rate": d["switch"]/max(d["n"],1),
              "mean_tokens": float(np.mean(d["toks"])) if d["toks"] else 0}
          for c, d in cagg.items()}
    results["task2"] = {"summary": s2, "raw_results": all_r2}
    log(f"    Task 2 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 5: Margin analysis ----
    log(f"  Task 5: Margin analysis...")
    margin_data = []
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            base_logits = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
        base_argmax = int(np.argmax(base_logits))
        base_top1 = float(np.sort(base_logits)[-1])
        base_top2 = float(np.sort(base_logits)[-2])
        base_margin = base_top1 - base_top2
        base_eos_logit = float(base_logits[tokenizer.eos_token_id]) if tokenizer.eos_token_id else 0

        for bs in [1.0, 3.0, 5.0, 10.0]:
            if bs == 1.0:
                pl = base_logits
            else:
                handle = layers[L_eos].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc_eos, ec_eos, bs))
                try:
                    with torch.no_grad():
                        pl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
                except: pl = base_logits
                handle.remove()

            patched_argmax = int(np.argmax(pl))
            patched_eos = float(pl[tokenizer.eos_token_id]) if tokenizer.eos_token_id else 0
            patched_top1 = float(np.sort(pl)[-1])
            patched_top2 = float(np.sort(pl)[-2])
            margin_data.append({
                "prompt": prompt, "boost_scale": bs,
                "base_eos_logit": base_eos_logit, "patched_eos_logit": patched_eos,
                "delta_eos": patched_eos - base_eos_logit,
                "base_margin": base_margin, "patched_margin": patched_top1 - patched_top2,
                "base_argmax": base_argmax, "patched_argmax": patched_argmax,
                "argmax_changed": int(patched_argmax != base_argmax),
                "eos_is_argmax": int(patched_argmax == tokenizer.eos_token_id) if tokenizer.eos_token_id else 0,
            })
        log(f"    {pi+1}/2 prompts")

    results["task5"] = {"data": margin_data}
    log(f"    Margin analysis:")
    for md in margin_data:
        if md["boost_scale"] != 1.0:
            log(f"      p='{md['prompt'][:25]}' bs={md['boost_scale']}: ΔEOS={md['delta_eos']:.4f}  "
                f"margin={md['base_margin']:.3f}→{md['patched_margin']:.3f}  argmax_chg={md['argmax_changed']}  eos_argmax={md['eos_is_argmax']}")

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)

    save_path = RESULT_DIR963 / f"{model_name}_result.json"
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"Saved: {save_path}")

if __name__ == "__main__":
    run()
