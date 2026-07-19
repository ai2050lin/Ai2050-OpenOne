#!/usr/bin/env python3
"""
Phase 964: 前向差分EOS促进搜索与直接EOS注入基线
==================================================
Phase 963证明权重cos无法预测实际ΔEOS。本阶段转向：
Task 1: Head差分搜索——对每个head做ablate, 测实际ΔEOS (last 5 layers)
Task 2: 直接EOS logit注入——z'_EOS = z_EOS + b, b∈{2,5,10}, delayed版
Task 3: DS7B triple scan——scale∈{3.5,4,4.5,5,5.5,6,7}
Task 4: Top差分正head的boost验证
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
from phase963_eos_boost import (
    log, get_head_dims, evaluate_strict_clean, make_head_hook, make_channel_hook,
    EN_PROMPTS_50, MAX_TOKENS, RESULT_DIR963
)

PHASE = 964
RESULT_DIR = Path("results/phase964_forward_diff")

LOCK_HEADS = {
    "qwen3":      {"layer": 35, "head": 0},
    "glm4":       {"layer": 39, "head": 21},
    "deepseek7b": {"layer": 26, "head": 19},
}
EOS_CHANNELS = {
    "qwen3":      {"layer": 34, "channel": 149},
    "glm4":       {"layer": 39, "channel": 1226},
    "deepseek7b": {"layer": 27, "channel": 14975},
}
EOS_PROMOTER_HEADS = {
    "qwen3":      {"layer": 34, "head": 1},
    "glm4":       {"layer": 38, "head": 6},
    "deepseek7b": {"layer": 27, "head": 13},
}

EXPECTED = {
    "The capital of France is": "Paris", "The largest planet is": "Jupiter",
    "Water boils at": "100", "The speed of light is": "299",
    "The sun is a": "star", "Dogs are": "animal", "The sky is": "blue",
    "Grass is": "green", "Fire needs": "oxygen", "Ice is": "frozen",
}


def make_eos_inject_hook(eos_id, bias, delay=0):
    """Add bias to EOS logit at each step (after delay steps)."""
    step = [0]
    def hook(module, input, output):
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        if step[0] >= delay:
            logits = logits.clone()
            if logits.ndim == 3:
                logits[:, -1, eos_id] += bias
            elif logits.ndim == 2:
                logits[:, eos_id] += bias
        step[0] += 1
        if isinstance(output, tuple):
            return (logits,) + output[1:]
        return logits
    return hook


def run_model(model_name):
    log(f"\n{'='*60}")
    log(f"Phase 964: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    eos_id = tokenizer.eos_token_id
    proto_ids = {"EOS": eos_id}
    for ts in [".", " "]:
        toks = tokenizer.encode(ts, add_special_tokens=False)
        if toks: proto_ids[ts] = toks[0]

    if model_name == "glm4":
        n_prompts_gen = 2
        n_prompts_diff = 2
    else:
        n_prompts_gen = 5
        n_prompts_diff = 3

    prompts_gen = EN_PROMPTS_50[:n_prompts_gen]
    prompts_diff = EN_PROMPTS_50[:n_prompts_diff]

    results = {"model": model_name}
    t_start = time.time()

    # ============================================================
    # Task 2: Direct EOS logit injection (MOST CRITICAL)
    # ============================================================
    log("  Task 2: Direct EOS logit injection...")
    inject_conditions = [
        ("normal", 0, 0),
        ("b=2", 2, 0),
        ("b=5", 5, 0),
        ("b=10", 10, 0),
        ("delayed2_b=5", 5, 2),
        ("delayed2_b=10", 10, 2),
    ]

    all_r2 = []
    for pi, prompt in enumerate(prompts_gen):
        for cond_name, bias, delay in inject_conditions:
            handle = None
            if bias > 0:
                handle = model.lm_head.register_forward_hook(
                    make_eos_inject_hook(eos_id, bias, delay))
            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=MAX_TOKENS,
                                         do_sample=False, pad_token_id=eos_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item() == eos_id if eos_id else False
                ng = len(gt)
            except Exception as e:
                gen = f"ERROR: {e}"; he = False; ng = 0
            if handle: handle.remove()
            ce = evaluate_strict_clean(prompt, gen, he, ng)
            all_r2.append({"prompt": prompt, "condition": cond_name,
                           "generated": gen[:200], "has_eos": he, "n_tokens": ng,
                           "strict_clean": ce["strict_clean"],
                           "lang_switched": not ce["is_ascii"]})
        log(f"    {pi+1}/{len(prompts_gen)} prompts")

    cagg = defaultdict(lambda: {"eos": 0, "clean": 0, "switch": 0, "n": 0, "toks": []})
    for r in all_r2:
        c = r["condition"]; cagg[c]["eos"] += int(r["has_eos"])
        cagg[c]["clean"] += int(r["strict_clean"])
        cagg[c]["switch"] += int(r["lang_switched"])
        cagg[c]["n"] += 1; cagg[c]["toks"].append(r["n_tokens"])
    s2 = {c: {"eos_rate": d["eos"]/max(d["n"],1),
              "strict_clean_rate": d["clean"]/max(d["n"],1),
              "lang_switch_rate": d["switch"]/max(d["n"],1),
              "mean_tokens": float(np.mean(d["toks"])) if d["toks"] else 0}
          for c, d in cagg.items()}
    results["task2"] = {"task": "task2_direct_eos_injection", "summary": s2, "raw_results": all_r2}
    (model_dir / "task2_direct_eos_injection.json").write_text(json.dumps(results["task2"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Direct EOS injection results:")
    for c in [c[0] for c in inject_conditions]:
        sv = s2.get(c, {})
        log(f"      {c:20s}: eos={sv.get('eos_rate',0):.2f}  clean={sv.get('strict_clean_rate',0):.2f}  "
            f"switch={sv.get('lang_switch_rate',0):.2f}  toks={sv.get('mean_tokens',0):.1f}")
    log(f"    Sample (p0):")
    for r in all_r2:
        if r["prompt"] == prompts_gen[0]:
            log(f"      {r['condition']:20s}: eos={r['has_eos']}  clean={r['strict_clean']}  text={r['generated'][:80]}")
    log(f"    Task 2 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ============================================================
    # Task 1: Head difference search (last 5 layers, ablate)
    # ============================================================
    log("  Task 1: Head difference search (last 5 layers, ablate)...")
    search_layers = list(range(max(0, n_layers - 5), n_layers))
    log(f"    Searching layers {search_layers[0]}-{search_layers[-1]} ({len(search_layers)} layers × {n_heads} heads)")

    # Normal forward for each prompt
    diff_results = defaultdict(lambda: defaultdict(list))  # head_key -> {token -> [ΔEOS values]}

    for pi, prompt in enumerate(prompts_diff):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            base_out = model(input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()

        for L in search_layers:
            for H in range(n_heads):
                sc = H * d_head; ec = sc + d_head
                handle = layers[L].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc, ec, 0.0))
                try:
                    with torch.no_grad():
                        out = model(input_ids, use_cache=False)
                        patched_logits = out.logits[0, -1].detach().float().cpu().numpy()
                except:
                    patched_logits = base_logits.copy()
                handle.remove()

                key = f"L{L}_H{H}"
                for tn, tid in proto_ids.items():
                    if tid < len(patched_logits):
                        delta = float(patched_logits[tid] - base_logits[tid])
                        diff_results[key][tn].append(delta)

        log(f"    {pi+1}/{len(prompts_diff)} prompts ({time.time()-t_start:.0f}s)")

    # Aggregate
    head_diff = {}
    for key, tokens in diff_results.items():
        means = {tn: float(np.mean(v)) for tn, v in tokens.items()}
        head_diff[key] = means

    # Find heads where ablation DECREASES EOS (meaning head promotes EOS)
    eos_promoters = sorted(head_diff.items(), key=lambda x: x[1].get("EOS", 0))[:20]
    eos_suppressors = sorted(head_diff.items(), key=lambda x: -x[1].get("EOS", 0))[:10]

    results["task1"] = {"task": "task1_head_diff_search",
        "search_layers": search_layers, "n_heads": n_heads,
        "n_prompts": len(prompts_diff),
        "top_eos_promoters": [(k, v["EOS"]) for k, v in eos_promoters if v.get("EOS", 0) < -0.01],
        "top_eos_suppressors": [(k, v["EOS"]) for k, v in eos_suppressors if v.get("EOS", 0) > 0.01],
        "all_heads": head_diff}
    (model_dir / "task1_head_diff.json").write_text(json.dumps(results["task1"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Top 10 EOS promoters (ablate decreases EOS → head promotes EOS):")
    for key, eos_d in [(k, v["EOS"]) for k, v in eos_promoters if v.get("EOS", 0) < -0.01][:10]:
        v = head_diff[key]
        log(f"      {key}: ΔEOS_ablate={eos_d:.4f}  Δperiod={v.get('.', 0):.4f}  Δspace={v.get(' ', 0):.4f}")
    log(f"    Top 5 EOS suppressors (ablate increases EOS → head suppresses EOS):")
    for key, eos_d in [(k, v["EOS"]) for k, v in eos_suppressors if v.get("EOS", 0) > 0.01][:5]:
        v = head_diff[key]
        log(f"      {key}: ΔEOS_ablate={eos_d:.4f}")
    log(f"    Task 1 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ============================================================
    # Task 4: Boost top diff-positive EOS head
    # ============================================================
    log("  Task 4: Boost top diff-positive EOS head...")
    # Find the head with most negative ΔEOS_ablate (strongest EOS promoter)
    top_promoter = None
    for key, eos_d in [(k, v["EOS"]) for k, v in eos_promoters if v.get("EOS", 0) < -0.01]:
        parts = key.split("_")
        L = int(parts[0][1:]); H = int(parts[1][1:])
        top_promoter = {"layer": L, "head": H, "delta_ablate": eos_d, "key": key}
        break

    if top_promoter:
        L_p = top_promoter["layer"]; H_p = top_promoter["head"]
        sc_p = H_p * d_head; ec_p = sc_p + d_head
        log(f"    Top promoter: L{L_p}_H{H_p} (ΔEOS_ablate={top_promoter['delta_ablate']:.4f})")

        boost_scales = [1.0, 3.0, 5.0]
        all_r4 = []
        for pi, prompt in enumerate(prompts_gen):
            for bs in boost_scales:
                handles = []
                if bs != 1.0:
                    handles.append(layers[L_p].self_attn.o_proj.register_forward_pre_hook(
                        make_head_hook(sc_p, ec_p, bs)))
                try:
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        oid = model.generate(input_ids, max_new_tokens=MAX_TOKENS,
                                             do_sample=False, pad_token_id=eos_id)
                    gt = oid[0][input_ids.shape[1]:]
                    gen = tokenizer.decode(gt, skip_special_tokens=False)
                    he = gt[-1].item() == eos_id if eos_id else False
                    ng = len(gt)
                except Exception as e:
                    gen = f"ERROR: {e}"; he = False; ng = 0
                for h in handles: h.remove()
                ce = evaluate_strict_clean(prompt, gen, he, ng)
                all_r4.append({"prompt": prompt, "boost_scale": bs,
                               "generated": gen[:200], "has_eos": he, "n_tokens": ng,
                               "strict_clean": ce["strict_clean"], "lang_switched": not ce["is_ascii"]})
            log(f"    {pi+1}/{len(prompts_gen)} prompts")

        s4 = {}
        for bs in boost_scales:
            br = [r for r in all_r4 if r["boost_scale"] == bs]
            s4[f"scale_{bs}"] = {"eos_rate": np.mean([r["has_eos"] for r in br]) if br else 0,
                "strict_clean_rate": np.mean([r["strict_clean"] for r in br]) if br else 0,
                "mean_tokens": np.mean([r["n_tokens"] for r in br]) if br else 0}
        results["task4"] = {"task": "task4_boost_diff_promoter", "promoter": top_promoter,
            "summary": s4, "raw_results": all_r4}
        (model_dir / "task4_boost_diff_promoter.json").write_text(json.dumps(results["task4"], ensure_ascii=False, indent=2), encoding="utf-8")
        for bs in boost_scales:
            sv = s4[f"scale_{bs}"]
            log(f"    scale={bs}: eos={sv['eos_rate']:.2f}  clean={sv['strict_clean_rate']:.2f}  toks={sv['mean_tokens']:.1f}")
        log(f"    Sample (p0):")
        for r in all_r4:
            if r["prompt"] == prompts_gen[0]:
                log(f"      scale={r['boost_scale']}: eos={r['has_eos']}  text={r['generated'][:80]}")
    else:
        log("    No positive diff promoter found")
        results["task4"] = {"error": "no promoter found"}

    log(f"    Task 4 done ({time.time()-t_start:.0f}s)")

    # ============================================================
    # Task 3: DS7B triple scan (only DS7B)
    # ============================================================
    if model_name == "deepseek7b":
        log("  Task 3: DS7B triple scan (scale sweep)...")
        lock = LOCK_HEADS[model_name]
        eos_ch = EOS_CHANNELS[model_name]
        eos_head = EOS_PROMOTER_HEADS[model_name]
        L_lock, H_lock = lock["layer"], lock["head"]
        L_ch, C_ch = eos_ch["layer"], eos_ch["channel"]
        L_eos, H_eos = eos_head["layer"], eos_head["head"]
        sc_lock = H_lock * d_head; ec_lock = sc_lock + d_head
        sc_eos = H_eos * d_head; ec_eos = sc_eos + d_head

        scales = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
        all_r3 = []
        for pi, prompt in enumerate(prompts_gen[:2]):
            for s in scales:
                handles = []
                handles.append(layers[L_eos].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc_eos, ec_eos, s)))
                handles.append(layers[L_lock].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc_lock, ec_lock, 0.0)))
                handles.append(layers[L_ch].mlp.down_proj.register_forward_pre_hook(
                    make_channel_hook([C_ch], s)))
                try:
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        oid = model.generate(input_ids, max_new_tokens=MAX_TOKENS,
                                             do_sample=False, pad_token_id=eos_id)
                    gt = oid[0][input_ids.shape[1]:]
                    gen = tokenizer.decode(gt, skip_special_tokens=False)
                    he = gt[-1].item() == eos_id if eos_id else False
                    ng = len(gt)
                except Exception as e:
                    gen = f"ERROR: {e}"; he = False; ng = 0
                for h in handles: h.remove()
                ce = evaluate_strict_clean(prompt, gen, he, ng)
                all_r3.append({"prompt": prompt, "scale": s, "generated": gen[:200],
                               "has_eos": he, "n_tokens": ng, "strict_clean": ce["strict_clean"],
                               "lang_switched": not ce["is_ascii"]})
                log(f"    p{pi} s={s}: eos={he} clean={ce['strict_clean']} toks={ng} text={gen[:50]}")

        results["task3"] = {"task": "task3_ds7b_scan", "scales": scales, "raw_results": all_r3}
        (model_dir / "task3_ds7b_scan.json").write_text(json.dumps(results["task3"], ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"    Task 3 done ({time.time()-t_start:.0f}s)")

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)

    save_path = RESULT_DIR / f"{model_name}_result.json"
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"  {model_name} complete. Saved: {save_path}")
    return results


def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started")
    log(f"Tasks: 1=head_diff_search, 2=direct_eos_inject, 3=ds7b_scan, 4=boost_diff_promoter")

    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if model_name:
        run_model(model_name)
    else:
        for m in ["qwen3", "glm4", "deepseek7b"]:
            try: run_model(m)
            except Exception as e:
                log(f"  {m} FAILED: {e}"); import traceback; traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
