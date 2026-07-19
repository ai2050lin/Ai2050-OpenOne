#!/usr/bin/env python3
"""Phase 962 minimal runner for GLM4 — reduced prompts to fit timeout."""
import sys, json, time, math, gc
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
    log, get_head_dims, cosine_similarity, evaluate_strict_clean,
    make_head_hook, make_channel_hook, make_capture_hook,
    MODEL_HEADS, EN_PROMPTS_50, CN_PROMPTS_10, MAX_TOKENS, RESULT_DIR
)

model_name = "glm4"
EN_5 = EN_PROMPTS_50[:5]
EN_3 = EN_PROMPTS_50[:3]
CN_3 = CN_PROMPTS_10[:3]


def run_glm4():
    log(f"\n{'='*60}")
    log(f"Phase 962 MINIMAL: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}  (load: {time.time()-t0:.0f}s)")

    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers
    heads = MODEL_HEADS[model_name]
    primary = heads[0]
    L0, H0 = primary["layer"], primary["head"]
    sc = H0 * d_head; ec = sc + d_head
    W_O_slice = layers[L0].self_attn.o_proj.weight[:, sc:ec].detach()

    results = {"model": model_name}
    t_start = time.time()

    # ---- Task 1: EOS head search (5 prompts) ----
    log("  Task 1: EOS-promoting head search (5 prompts)...")
    W_U = get_W_U(model, model_name)
    eos_id = tokenizer.eos_token_id
    W_U_eos = W_U[eos_id].astype(np.float64)
    W_U_eos_norm = np.linalg.norm(W_U_eos)
    proto_dirs = {"EOS": W_U_eos}
    for tok_str, tok_key in [(".", "period"), (" ", "space")]:
        toks = tokenizer.encode(tok_str, add_special_tokens=False)
        if toks: proto_dirs[tok_key] = W_U[toks[0]].astype(np.float64)

    cos_scores = defaultdict(lambda: defaultdict(list))
    for pi, prompt in enumerate(EN_5):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        captured = {}
        hooks = []
        for L in range(n_layers):
            def make_pre(Li):
                def h(module, args):
                    inp = args[0] if isinstance(args, tuple) else args
                    captured[Li] = inp[0, -1, :].detach().float().cpu().numpy()
                return h
            hooks.append(layers[L].self_attn.o_proj.register_forward_pre_hook(make_pre(L)))
        with torch.no_grad(): model(input_ids, use_cache=False)
        for h in hooks: h.remove()
        for L in range(n_layers):
            if L not in captured: continue
            o_proj_input = captured[L]
            W_O = layers[L].self_attn.o_proj.weight.detach().float().cpu().numpy()
            for H in range(n_heads):
                s2, e2 = H * d_head, (H + 1) * d_head
                O_h = W_O[:, s2:e2] @ o_proj_input[s2:e2]
                key = f"L{L}_H{H}"
                for dn, dv in proto_dirs.items():
                    cos_scores[key][dn].append(cosine_similarity(O_h, dv))
        log(f"    {pi+1}/5 prompts")

    head_results = {}
    for key, dirs in cos_scores.items():
        head_results[key] = {d: float(np.mean(v)) for d, v in dirs.items()}
    eos_sorted = sorted(head_results.items(), key=lambda x: -x[1].get("EOS", 0))
    top_promoters = [(k, v["EOS"]) for k, v in eos_sorted if v.get("EOS", 0) > 0.02][:20]
    top_suppressors = [(k, v["EOS"]) for k, v in eos_sorted if v.get("EOS", 0) < -0.02][:10]

    results["task1"] = {"task": "task1_eos_head_search", "model": model_name,
        "n_prompts": 5, "top_eos_promoters": top_promoters,
        "top_eos_suppressors": top_suppressors, "all_heads": head_results}
    (model_dir / "task1_eos_head_search.json").write_text(
        json.dumps(results["task1"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Top 10 EOS-promoting heads:")
    for key, cos in top_promoters[:10]:
        v = head_results[key]
        log(f"      {key}: cos_EOS={cos:.4f}  cos_period={v.get('period',0):.4f}")
    log(f"    Top 5 EOS-suppressing heads:")
    for key, cos in top_suppressors[:5]:
        log(f"      {key}: cos_EOS={cos:.4f}")
    log(f"    Task 1 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 2: EOS channel search ----
    log("  Task 2: EOS-promoting channel search...")
    channel_results = {}
    for L in range(n_layers):
        W_down = layers[L].mlp.down_proj.weight.detach().float().cpu().numpy()
        dot = W_U_eos @ W_down
        col_norms = np.linalg.norm(W_down, axis=0)
        cos_vals = dot / (W_U_eos_norm * col_norms + 1e-10)
        for idx in np.argsort(cos_vals)[-5:][::-1]:
            key = f"L{L}_C{idx}"
            channel_results[key] = {"layer": L, "channel": int(idx), "cos_eos": float(cos_vals[idx])}

    # Natural activation
    input_ids = tokenizer.encode(EN_5[0], return_tensors="pt").to(device)
    captured_down = {}
    hooks = []
    for L in range(n_layers):
        def make_hook(Li):
            def h(module, args):
                inp = args[0] if isinstance(args, tuple) else args
                captured_down[Li] = inp[0, -1, :].detach().float().cpu().numpy()
            return h
        hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(L)))
    with torch.no_grad(): model(input_ids, use_cache=False)
    for h in hooks: h.remove()

    for key, cr in channel_results.items():
        L, c = cr["layer"], cr["channel"]
        if L in captured_down and c < len(captured_down[L]):
            act = float(captured_down[L][c])
            cr["activation"] = act
            cr["contribution_eos"] = act * cr["cos_eos"] * W_U_eos_norm

    sorted_ch = sorted(channel_results.items(), key=lambda x: -abs(x[1].get("contribution_eos", 0)))
    top_ch = [(k, v) for k, v in sorted_ch if v.get("contribution_eos", 0) > 0][:10]

    results["task2"] = {"task": "task2_eos_channel_search", "model": model_name,
        "top_eos_promoting_channels": [(k, v) for k, v in top_ch]}
    (model_dir / "task2_eos_channel_search.json").write_text(
        json.dumps(results["task2"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"    Top 5 EOS-promoting channels:")
    for key, v in top_ch[:5]:
        log(f"      {key}: cos_eos={v['cos_eos']:.4f}  act={v.get('activation',0):.3f}  contrib={v.get('contribution_eos',0):.3f}")

    eos_ch_layer = top_ch[0][1]["layer"] if top_ch else n_layers - 1
    eos_ch_idx = top_ch[0][1]["channel"] if top_ch else 0
    log(f"    Best EOS channel: L{eos_ch_layer}_C{eos_ch_idx}")
    log(f"    Task 2 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 3: Reverse lock (3 prompts × 4 lambdas) ----
    log("  Task 3: Reverse lock intervention (3 prompts)...")
    lambdas = [0.0, 0.5, 1.0, 2.0]
    all_r3 = []
    for pi, prompt in enumerate(EN_3):
        for lam in lambdas:
            handles = []; captured_head = {}
            if lam != 0.0:
                def o_pre(module, args):
                    inp = args[0] if isinstance(args, tuple) else args
                    captured_head['input'] = inp[:, -1, sc:ec].detach().clone()
                def l_out(module, input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    hi = captured_head.get('input')
                    if hi is not None:
                        O_lock = torch.matmul(hi, W_O_slice.T)
                        out[:, -1, :] = out[:, -1, :] - lam * O_lock.to(out.dtype).to(out.device)
                    if isinstance(output, tuple): return (out,) + output[1:]
                    return out
                handles.append(layers[L0].self_attn.o_proj.register_forward_pre_hook(o_pre))
                handles.append(layers[L0].register_forward_hook(l_out))
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
    results["task3"] = {"task": "task3_reverse_lock", "model": model_name,
        "primary_head": f"L{L0}_H{H0}", "lambdas": lambdas, "summary": s3, "raw_results": all_r3}
    (model_dir / "task3_reverse_lock.json").write_text(
        json.dumps(results["task3"], ensure_ascii=False, indent=2), encoding="utf-8")
    for lam in lambdas:
        sv = s3[f"lam_{lam}"]
        log(f"    λ={lam:.1f}: eos={sv['eos_rate']:.2f}  switch={sv['lang_switch_rate']:.2f}  toks={sv['mean_tokens']:.1f}")
    log(f"    Sample (p0):")
    for r in all_r3:
        if r["prompt"] == EN_3[0]:
            log(f"      λ={r['lambda']:.1f}: eos={r['has_eos']}  text={r['generated'][:80]}")
    log(f"    Task 3 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 4: Combined (3 prompts × 5 conditions) ----
    log("  Task 4: Combined intervention (3 prompts)...")
    conditions = [
        ("normal", 0.0, 1.0, None, None),
        ("ablate_lock", 1.0, 1.0, None, None),
        ("reverse_lock_2.0", 2.0, 1.0, None, None),
        ("boost_eos_ch", 0.0, 3.0, eos_ch_layer, eos_ch_idx),
        ("rev2.0+boost_eos", 2.0, 3.0, eos_ch_layer, eos_ch_idx),
    ]
    all_r4 = []
    for pi, prompt in enumerate(EN_3):
        for cn, lam, cs, cl, ci in conditions:
            handles = []; captured_head = {}
            if lam != 0.0:
                def o_pre(module, args):
                    inp = args[0] if isinstance(args, tuple) else args
                    captured_head['input'] = inp[:, -1, sc:ec].detach().clone()
                def l_out(module, input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    hi = captured_head.get('input')
                    if hi is not None:
                        O_lock = torch.matmul(hi, W_O_slice.T)
                        out[:, -1, :] = out[:, -1, :] - lam * O_lock.to(out.dtype).to(out.device)
                    if isinstance(output, tuple): return (out,) + output[1:]
                    return out
                handles.append(layers[L0].self_attn.o_proj.register_forward_pre_hook(o_pre))
                handles.append(layers[L0].register_forward_hook(l_out))
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
    results["task4"] = {"task": "task4_combined", "model": model_name,
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
        if r["prompt"] == EN_3[0]:
            log(f"      {r['condition']:25s}: eos={r['has_eos']}  clean={r['strict_clean']}  text={r['generated'][:80]}")
    log(f"    Task 4 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 5: De-headed mode direction (3 EN + 3 CN) ----
    log("  Task 5: De-headed mode direction (3 EN + 3 CN)...")
    en_n, en_a, cn_n, cn_a = [], [], [], []
    head_Ohs = []
    all_p = [(p, "en") for p in EN_3] + [(p, "cn") for p in CN_3]
    for pi, (prompt, lang) in enumerate(all_p):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        cap_n = {}; hk_n = layers[L0].register_forward_hook(make_capture_hook(cap_n, "n"))
        with torch.no_grad(): model(input_ids, use_cache=False)
        hk_n.remove()
        cap_a = {}
        hk_ab = layers[L0].self_attn.o_proj.register_forward_pre_hook(make_head_hook(sc, ec, 0.0))
        hk_ca = layers[L0].register_forward_hook(make_capture_hook(cap_a, "a"))
        with torch.no_grad(): model(input_ids, use_cache=False)
        hk_ab.remove(); hk_ca.remove()
        nv = cap_n.get("n"); av = cap_a.get("a")
        if nv is not None and av is not None:
            if lang == "en": en_n.append(nv); en_a.append(av); head_Ohs.append(nv - av)
            else: cn_n.append(nv); cn_a.append(av)
        log(f"    {pi+1}/{len(all_p)} prompts")

    if en_n and cn_n:
        dm_n = np.mean(en_n, axis=0) - np.mean(cn_n, axis=0)
        dm_a = np.mean(en_a, axis=0) - np.mean(cn_a, axis=0)
        mOh = np.mean(head_Ohs, axis=0)
        cos_n = cosine_similarity(mOh, dm_n)
        cos_a = cosine_similarity(mOh, dm_a)
        pp_n = [cosine_similarity(o, dm_n) for o in head_Ohs]
        pp_a = [cosine_similarity(o, dm_a) for o in head_Ohs]
        results["task5"] = {"task": "task5_deheaded_mode", "model": model_name,
            "head": f"L{L0}_H{H0}",
            "d_mode_normal_norm": float(np.linalg.norm(dm_n)),
            "d_mode_deheaded_norm": float(np.linalg.norm(dm_a)),
            "cos_Oh_vs_d_mode_normal": cos_n,
            "cos_Oh_vs_d_mode_deheaded": cos_a,
            "mean_per_prompt_cos_normal": float(np.mean(pp_n)),
            "mean_per_prompt_cos_deheaded": float(np.mean(pp_a)),
            "norm_Oh": float(np.linalg.norm(mOh))}
        log(f"    d_mode_normal norm: {np.linalg.norm(dm_n):.3f}")
        log(f"    d_mode_deheaded norm: {np.linalg.norm(dm_a):.3f}")
        log(f"    cos(O_h, d_mode_normal)   = {cos_n:.4f}")
        log(f"    cos(O_h, d_mode_deheaded) = {cos_a:.4f}")
        log(f"    Per-prompt: normal={np.mean(pp_n):.4f}  deheaded={np.mean(pp_a):.4f}")
    else:
        results["task5"] = {"error": "insufficient data"}
    (model_dir / "task5_deheaded_mode.json").write_text(
        json.dumps(results["task5"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Task 5 done ({time.time()-t_start:.0f}s)")

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)

    save_path = RESULT_DIR / f"{model_name}_result.json"
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"Saved: {save_path}")


if __name__ == "__main__":
    run_glm4()
