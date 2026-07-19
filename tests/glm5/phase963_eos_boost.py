#!/usr/bin/env python3
"""
Phase 963: EOS促进head boost与三联联合干预
=============================================
Phase 962发现L38_H6(cos_EOS=0.255)是EOS促进head，L39_H21是模式锁定head。
本阶段测试：boost EOS促进head能否提高EOS率？三联干预能否实现strict-clean？

Task 1: Boost L38_H6 alone (scale=1.5/2.0/3.0/5.0)
Task 2: Boost L38_H6 + Ablate L39_H21
Task 3: Boost L38_H6 + Boost EOS channel (L39_C1226)
Task 4: Triple: Boost(L38_H6) + Ablate(L39_H21) + Boost(L39_C1226)
Task 5: Logit margin分析 (为什么boost是否翻转argmax)
Task 6: 跨模型验证 (qwen3 L34_H1, DS7B L27_H13)
"""

from __future__ import annotations
import gc, json, sys, time, math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U
from phase951_protocol_atlas import ensure_dir
from phase962_eos_promoter_search import (
    log, get_head_dims, cosine_similarity, evaluate_strict_clean,
    make_head_hook, make_channel_hook, EN_PROMPTS_50, MAX_TOKENS, RESULT_DIR
)

PHASE = 963
RESULT_DIR963 = Path("results/phase963_eos_boost")

# Per-model EOS-promoting heads and channels (from Phase 962)
EOS_PROMOTER_HEADS = {
    "qwen3":       {"layer": 34, "head": 1,  "cos_eos": 0.082},
    "glm4":        {"layer": 38, "head": 6,  "cos_eos": 0.255},
    "deepseek7b":  {"layer": 27, "head": 13, "cos_eos": 0.170},
}

LOCK_HEADS = {
    "qwen3":       {"layer": 35, "head": 0},
    "glm4":        {"layer": 39, "head": 21},
    "deepseek7b":  {"layer": 26, "head": 19},
}

EOS_CHANNELS = {
    "qwen3":       {"layer": 34, "channel": 149},
    "glm4":        {"layer": 39, "channel": 1226},
    "deepseek7b":  {"layer": 27, "channel": 14975},
}

EXPECTED = {
    "The capital of France is": "Paris", "The largest planet is": "Jupiter",
    "Water boils at": "100", "The speed of light is": "299",
    "The sun is a": "star", "Dogs are": "animal", "The sky is": "blue",
    "Grass is": "green", "Fire needs": "oxygen", "Ice is": "frozen",
}


def run_model(model_name):
    log(f"\n{'='*60}")
    log(f"Phase 963: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR963 / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    eos_head = EOS_PROMOTER_HEADS[model_name]
    lock_head = LOCK_HEADS[model_name]
    eos_ch = EOS_CHANNELS[model_name]

    L_eos, H_eos = eos_head["layer"], eos_head["head"]
    L_lock, H_lock = lock_head["layer"], lock_head["head"]
    L_ch, C_ch = eos_ch["layer"], eos_ch["channel"]

    sc_eos = H_eos * d_head; ec_eos = sc_eos + d_head
    sc_lock = H_lock * d_head; ec_lock = sc_lock + d_head

    # Prompts — keep small to fit timeout
    if model_name == "glm4":
        prompts = EN_PROMPTS_50[:2]
    else:
        prompts = EN_PROMPTS_50[:5]

    results = {"model": model_name}
    t_start = time.time()

    # ---- Task 1: Boost EOS head alone ----
    log(f"  Task 1: Boost EOS head L{L_eos}_H{H_eos} (cos_eos={eos_head['cos_eos']:.3f})...")
    boost_scales = [1.0, 2.0, 3.0, 5.0]
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
            all_r1.append({"prompt": prompt, "boost_scale": bs, "generated": gen[:200],
                           "has_eos": he, "n_tokens": ng, "strict_clean": ce["strict_clean"],
                           "lang_switched": not ce["is_ascii"]})
        if (pi+1) % 5 == 0: log(f"    {pi+1}/{len(prompts)} prompts")

    s1 = {}
    for bs in boost_scales:
        br = [r for r in all_r1 if r["boost_scale"] == bs]
        s1[f"scale_{bs}"] = {"eos_rate": np.mean([r["has_eos"] for r in br]) if br else 0,
            "strict_clean_rate": np.mean([r["strict_clean"] for r in br]) if br else 0,
            "lang_switch_rate": np.mean([r["lang_switched"] for r in br]) if br else 0,
            "mean_tokens": np.mean([r["n_tokens"] for r in br]) if br else 0}
    results["task1"] = {"task": "task1_boost_eos_head", "model": model_name,
        "eos_head": f"L{L_eos}_H{H_eos}", "scales": boost_scales, "summary": s1, "raw_results": all_r1}
    (model_dir / "task1_boost_eos_head.json").write_text(json.dumps(results["task1"], ensure_ascii=False, indent=2), encoding="utf-8")
    for bs in boost_scales:
        sv = s1[f"scale_{bs}"]
        log(f"    scale={bs:.1f}: eos={sv['eos_rate']:.2f}  clean={sv['strict_clean_rate']:.2f}  switch={sv['lang_switch_rate']:.2f}  toks={sv['mean_tokens']:.1f}")
    log(f"    Sample (p0):")
    for r in all_r1:
        if r["prompt"] == prompts[0]:
            log(f"      scale={r['boost_scale']:.1f}: eos={r['has_eos']}  clean={r['strict_clean']}  text={r['generated'][:80]}")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 2: Boost EOS head + Ablate lock head ----
    log(f"  Task 2: Boost EOS head + Ablate lock head L{L_lock}_H{H_lock}...")
    conditions2 = [
        ("normal", 1.0, 1.0),
        ("boost_eos_3x", 3.0, 1.0),
        ("ablate_lock", 1.0, 0.0),
        ("boost_eos+ablate_lock", 3.0, 0.0),
    ]
    all_r2 = []
    for pi, prompt in enumerate(prompts):
        for cn, eos_scale, lock_scale in conditions2:
            handles = []
            if eos_scale != 1.0:
                handles.append(layers[L_eos].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc_eos, ec_eos, eos_scale)))
            if lock_scale != 1.0:
                handles.append(layers[L_lock].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc_lock, ec_lock, lock_scale)))
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
            all_r2.append({"prompt": prompt, "condition": cn, "generated": gen[:200],
                           "has_eos": he, "n_tokens": ng, "strict_clean": ce["strict_clean"],
                           "lang_switched": not ce["is_ascii"]})
        if (pi+1) % 5 == 0: log(f"    {pi+1}/{len(prompts)} prompts")

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
    results["task2"] = {"task": "task2_boost_eos_ablate_lock", "model": model_name, "summary": s2, "raw_results": all_r2}
    (model_dir / "task2_boost_eos_ablate_lock.json").write_text(json.dumps(results["task2"], ensure_ascii=False, indent=2), encoding="utf-8")
    for c in [c[0] for c in conditions2]:
        sv = s2.get(c, {})
        log(f"    {c:30s}: eos={sv.get('eos_rate',0):.2f}  clean={sv.get('strict_clean_rate',0):.2f}  switch={sv.get('lang_switch_rate',0):.2f}  toks={sv.get('mean_tokens',0):.1f}")
    log(f"    Sample (p0):")
    for r in all_r2:
        if r["prompt"] == prompts[0]:
            log(f"      {r['condition']:30s}: eos={r['has_eos']}  clean={r['strict_clean']}  text={r['generated'][:80]}")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 3: Boost EOS head + Boost EOS channel ----
    log(f"  Task 3: Boost EOS head + Boost EOS channel L{L_ch}_C{C_ch}...")
    conditions3 = [
        ("normal", 1.0, 1.0),
        ("boost_eos_head_3x", 3.0, 1.0),
        ("boost_eos_ch_3x", 1.0, 3.0),
        ("boost_both_3x", 3.0, 3.0),
        ("boost_both_5x", 5.0, 5.0),
    ]
    all_r3 = []
    for pi, prompt in enumerate(prompts):
        for cn, hs, cs in conditions3:
            handles = []
            if hs != 1.0:
                handles.append(layers[L_eos].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc_eos, ec_eos, hs)))
            if cs != 1.0:
                handles.append(layers[L_ch].mlp.down_proj.register_forward_pre_hook(
                    make_channel_hook([C_ch], cs)))
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
            all_r3.append({"prompt": prompt, "condition": cn, "generated": gen[:200],
                           "has_eos": he, "n_tokens": ng, "strict_clean": ce["strict_clean"],
                           "lang_switched": not ce["is_ascii"]})
        if (pi+1) % 5 == 0: log(f"    {pi+1}/{len(prompts)} prompts")

    cagg = defaultdict(lambda: {"eos": 0, "clean": 0, "switch": 0, "n": 0, "toks": []})
    for r in all_r3:
        c = r["condition"]; cagg[c]["eos"] += int(r["has_eos"])
        cagg[c]["clean"] += int(r["strict_clean"])
        cagg[c]["switch"] += int(r["lang_switched"])
        cagg[c]["n"] += 1; cagg[c]["toks"].append(r["n_tokens"])
    s3 = {c: {"eos_rate": d["eos"]/max(d["n"],1), "strict_clean_rate": d["clean"]/max(d["n"],1),
              "lang_switch_rate": d["switch"]/max(d["n"],1),
              "mean_tokens": float(np.mean(d["toks"])) if d["toks"] else 0}
          for c, d in cagg.items()}
    results["task3"] = {"task": "task3_boost_eos_head_ch", "model": model_name, "summary": s3, "raw_results": all_r3}
    (model_dir / "task3_boost_eos_head_ch.json").write_text(json.dumps(results["task3"], ensure_ascii=False, indent=2), encoding="utf-8")
    for c in [c[0] for c in conditions3]:
        sv = s3.get(c, {})
        log(f"    {c:30s}: eos={sv.get('eos_rate',0):.2f}  clean={sv.get('strict_clean_rate',0):.2f}  switch={sv.get('lang_switch_rate',0):.2f}  toks={sv.get('mean_tokens',0):.1f}")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 4: Triple intervention ----
    log(f"  Task 4: Triple: Boost(EOS head) + Ablate(lock) + Boost(EOS ch)...")
    conditions4 = [
        ("normal", 1.0, 1.0, 1.0),
        ("triple_3x", 3.0, 0.0, 3.0),
        ("triple_5x", 5.0, 0.0, 5.0),
        ("boost_eos+ablate_lock", 3.0, 0.0, 1.0),
        ("boost_eos+boost_ch", 3.0, 1.0, 3.0),
    ]
    all_r4 = []
    for pi, prompt in enumerate(prompts):
        for cn, hs, ls, cs in conditions4:
            handles = []
            if hs != 1.0:
                handles.append(layers[L_eos].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc_eos, ec_eos, hs)))
            if ls != 1.0:
                handles.append(layers[L_lock].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc_lock, ec_lock, ls)))
            if cs != 1.0:
                handles.append(layers[L_ch].mlp.down_proj.register_forward_pre_hook(
                    make_channel_hook([C_ch], cs)))
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
        if (pi+1) % 5 == 0: log(f"    {pi+1}/{len(prompts)} prompts")

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
    results["task4"] = {"task": "task4_triple", "model": model_name, "summary": s4, "raw_results": all_r4}
    (model_dir / "task4_triple.json").write_text(json.dumps(results["task4"], ensure_ascii=False, indent=2), encoding="utf-8")
    for c in [c[0] for c in conditions4]:
        sv = s4.get(c, {})
        log(f"    {c:30s}: eos={sv.get('eos_rate',0):.2f}  clean={sv.get('strict_clean_rate',0):.2f}  switch={sv.get('lang_switch_rate',0):.2f}  toks={sv.get('mean_tokens',0):.1f}")
    log(f"    Sample (p0):")
    for r in all_r4:
        if r["prompt"] == prompts[0]:
            log(f"      {r['condition']:30s}: eos={r['has_eos']}  clean={r['strict_clean']}  text={r['generated'][:80]}")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 5: Logit margin analysis ----
    log(f"  Task 5: Logit margin analysis (boost EOS head effect on logits)...")
    proto_ids = {}
    if tokenizer.eos_token_id is not None:
        proto_ids["EOS"] = tokenizer.eos_token_id
    for ts in [".", " "]:
        toks = tokenizer.encode(ts, add_special_tokens=False)
        if toks: proto_ids[ts] = toks[0]

    margin_data = []
    test_prompts = prompts[:3] if model_name != "glm4" else prompts[:2]
    for pi, prompt in enumerate(test_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            base_logits = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
        base_argmax = int(np.argmax(base_logits))
        base_top1 = float(np.sort(base_logits)[-1])
        base_top2 = float(np.sort(base_logits)[-2])
        base_margin = base_top1 - base_top2
        base_eos_logit = float(base_logits[tokenizer.eos_token_id]) if tokenizer.eos_token_id else 0

        for bs in [1.0, 3.0, 5.0]:
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
                "base_eos_logit": base_eos_logit,
                "patched_eos_logit": patched_eos,
                "delta_eos": patched_eos - base_eos_logit,
                "base_margin": base_margin,
                "patched_margin": patched_top1 - patched_top2,
                "base_argmax": base_argmax,
                "patched_argmax": patched_argmax,
                "argmax_changed": int(patched_argmax != base_argmax),
                "eos_is_argmax": int(patched_argmax == tokenizer.eos_token_id) if tokenizer.eos_token_id else 0,
            })
        log(f"    {pi+1}/5 prompts")

    results["task5"] = {"task": "task5_margin", "model": model_name, "data": margin_data}
    (model_dir / "task5_margin.json").write_text(json.dumps(results["task5"], ensure_ascii=False, indent=2), encoding="utf-8")
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
    log(f"  {model_name} complete. Saved: {save_path}")
    return results


def main():
    ensure_dir(RESULT_DIR963)
    log(f"Phase {PHASE} started")
    log(f"Tasks: 1=boost_eos_head, 2=boost+ablate_lock, 3=boost_head+ch, 4=triple, 5=margin")

    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if model_name:
        run_model(model_name)
    else:
        for m in ["qwen3", "glm4", "deepseek7b"]:
            try:
                run_model(m)
            except Exception as e:
                log(f"  {m} FAILED: {e}"); import traceback; traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
