#!/usr/bin/env python3
"""Phase 961 ultra-minimal runner for GLM4 — only Task 1 + Task 2/3."""
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
from phase961_mode_head_mechanism import (
    log, get_head_dims, classify_token, get_proto_token_ids,
    cosine_similarity, make_head_hook, make_capture_hook,
    evaluate_strict_clean, make_channel_hook,
    MODEL_HEADS, SUPER_CHANNELS, MAX_TOKENS_ROLLOUT, RESULT_DIR
)

model_name = "glm4"

EN_10 = [
    "The capital of France is", "The largest planet is", "Water boils at",
    "The speed of light is", "The sun is a", "Dogs are", "The sky is",
    "Grass is", "Fire needs", "Ice is",
]
CN_5 = ["法国的首都是", "最大的行星是", "水的沸点是", "光速是", "太阳是一颗"]
EN_5 = EN_10[:5]

def run():
    log(f"\n{'='*60}")
    log(f"Phase 961 MINIMAL: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}  (load: {time.time()-t0:.0f}s)")

    n_heads, d_head = get_head_dims(model, info)
    log(f"  n_heads={n_heads}, d_head={d_head}")
    layers = get_layers(model)
    heads = MODEL_HEADS[model_name]
    channels = SUPER_CHANNELS[model_name]

    results = {"model": model_name}
    t_start = time.time()

    # ---- Task 1: Attention pattern (10 prompts) ----
    log("  Task 1: Attention pattern (10 prompts)...")
    attn_mass = defaultdict(lambda: defaultdict(float))
    attn_entropy = defaultdict(list)
    detail_attn = {}
    prompt_count = 0

    for pi, prompt in enumerate(EN_10):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 3:
            continue
        try:
            with torch.no_grad():
                outputs = model(input_ids, output_attentions=True, use_cache=False)
        except Exception as e:
            log(f"    output_attentions failed: {e}")
            break

        attentions = outputs.attentions
        token_strs = [tokenizer.decode([input_ids[0, i].item()]) for i in range(seq_len)]

        for h_info in heads:
            L, H = h_info["layer"], h_info["head"]
            if L >= len(attentions) or H >= attentions[L].shape[1]:
                continue
            head_attn = attentions[L][0, H, -1, :].float().cpu().numpy()
            key = f"L{L}_H{H}"
            for i in range(seq_len):
                attn_mass[key][classify_token(token_strs[i])] += float(head_attn[i])
            p = head_attn[head_attn > 1e-10]
            if len(p) > 0:
                ent = -np.sum(p * np.log(p))
                attn_entropy[key].append(float(ent / max(math.log(seq_len), 1e-10)))
            if pi == 0:
                detail_attn[key] = {"prompt": prompt, "tokens": token_strs,
                    "attention": head_attn.tolist(),
                    "categories": [classify_token(t) for t in token_strs]}

        prompt_count += 1
        del outputs, attentions

    t1_summary = {}
    for key, cats in attn_mass.items():
        total = sum(cats.values())
        t1_summary[key] = {cat: (m / total if total > 0 else 0.0) for cat, m in cats.items()}
        t1_summary[key]["n_prompts"] = prompt_count
        t1_summary[key]["mean_entropy"] = float(np.mean(attn_entropy[key])) if attn_entropy[key] else 0.0

    results["task1"] = {"task": "task1_attention_pattern", "model": model_name,
        "n_prompts": prompt_count, "attention_mass_by_category": t1_summary,
        "detail_first_prompt": detail_attn}
    (model_dir / "task1_attention_pattern.json").write_text(
        json.dumps(results["task1"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved task1 ({time.time()-t_start:.0f}s)")
    for key, s in t1_summary.items():
        cats_str = "  ".join(f"{c}={v:.3f}" for c, v in sorted(s.items())
                             if c not in ["n_prompts", "mean_entropy"])
        log(f"    {key}: entropy={s.get('mean_entropy', 0):.3f}  {cats_str}")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 2+3: Head output + mode direction (5 EN + 5 CN) ----
    log("  Task 2+3: Head output + mode direction (5 EN + 5 CN)...")
    try:
        W_U = get_W_U(model, model_name)
        log(f"    W_U: {W_U.shape}")
    except Exception as e:
        log(f"    W_U failed: {e}"); W_U = None

    proto_ids = get_proto_token_ids(tokenizer)
    en_layer_out = defaultdict(list)
    cn_layer_out = defaultdict(list)
    head_outputs = defaultdict(lambda: {"en": [], "cn": []})

    all_prompts = [(p, "en") for p in EN_5] + [(p, "cn") for p in CN_5]

    for pi, (prompt, lang) in enumerate(all_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Normal forward
        captured_n = {}
        nhooks = []
        seen = set()
        for h_info in heads:
            L = h_info["layer"]
            if L not in seen:
                seen.add(L)
                nhooks.append(layers[L].register_forward_hook(make_capture_hook(captured_n, f"n_L{L}")))
        with torch.no_grad():
            model(input_ids, use_cache=False)
        for h in nhooks: h.remove()

        # Ablated forward for each head
        for h_info in heads:
            L, H = h_info["layer"], h_info["head"]
            sc, ec = H * d_head, (H + 1) * d_head
            captured_a = {}
            hab = layers[L].self_attn.o_proj.register_forward_pre_hook(make_head_hook(sc, ec, 0.0))
            hca = layers[L].register_forward_hook(make_capture_hook(captured_a, f"a_L{L}"))
            with torch.no_grad():
                model(input_ids, use_cache=False)
            hab.remove(); hca.remove()

            nv = captured_n.get(f"n_L{L}")
            av = captured_a.get(f"a_L{L}")
            if nv is not None and av is not None:
                head_outputs[f"L{L}_H{H}"][lang].append(nv - av)

        for h_info in heads:
            L = h_info["layer"]
            k = f"n_L{L}"
            if k in captured_n:
                (en_layer_out if lang == "en" else cn_layer_out)[L].append(captured_n[k])

        log(f"    {pi+1}/{len(all_prompts)} prompts ({time.time()-t_start:.0f}s)")

    # Mode direction
    mode_dirs = {}
    for L in en_layer_out:
        em = np.mean(en_layer_out[L], axis=0)
        cm = np.mean(cn_layer_out[L], axis=0)
        mode_dirs[L] = {"d_mode": (em - cm).tolist(), "d_mode_norm": float(np.linalg.norm(em - cm))}

    head_analysis = {}
    for h_info in heads:
        L, H = h_info["layer"], h_info["head"]
        hk = f"L{L}_H{H}"
        eOhs = head_outputs[hk]["en"]
        cOhs = head_outputs[hk]["cn"]
        if not eOhs: continue

        mOe = np.mean(eOhs, axis=0)
        mOc = np.mean(cOhs, axis=0) if cOhs else np.zeros_like(mOe)
        ne = float(np.linalg.norm(mOe))
        nc = float(np.linalg.norm(mOc))
        dm = np.array(mode_dirs.get(L, {}).get("d_mode", [0] * len(mOe)))
        ce = cosine_similarity(mOe, dm)
        cc = cosine_similarity(mOc, dm)

        wu_cos = {}
        top_prom = []
        if W_U is not None:
            for tn, tid in proto_ids.items():
                if tid < W_U.shape[0]:
                    wu_cos[tn] = cosine_similarity(mOe, W_U[tid])
            dl = W_U @ mOe
            for tid in np.argsort(dl)[-15:][::-1]:
                top_prom.append({"token": tokenizer.decode([int(tid)]), "token_id": int(tid),
                                 "delta_logit": float(dl[tid])})

        pp_cos = [cosine_similarity(o, dm) for o in eOhs]
        head_analysis[hk] = {"layer": L, "head": H, "role": h_info["role"],
            "norm_Oh_en": ne, "norm_Oh_cn": nc,
            "cos_Oh_en_vs_dmode": ce, "cos_Oh_cn_vs_dmode": cc,
            "wu_cosines": wu_cos, "top_promoted_tokens_en": top_prom,
            "mean_per_prompt_cos_mode_en": float(np.mean(pp_cos)) if pp_cos else 0.0}

        log(f"    {hk} ({h_info['role']}): ||O_h||={ne:.3f}  cos_en={ce:.4f}  cos_cn={cc:.4f}")
        if wu_cos:
            tw = sorted(wu_cos.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            log(f"      W_U cos: {'  '.join(f'{n}={v:.4f}' for n, v in tw)}")
        if top_prom:
            ts = [f"'{t['token']}'({t['delta_logit']:.3f})" for t in top_prom[:3]]
            log(f"      Top promoted: {'  '.join(ts)}")

    results["task2_3"] = {"task": "task2_3_head_output_mode", "model": model_name,
        "mode_directions": {str(k): v for k, v in mode_dirs.items()},
        "head_analysis": head_analysis}
    (model_dir / "task2_3_head_output_mode.json").write_text(
        json.dumps(results["task2_3"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved task2_3 ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 4: Boost analysis (5 prompts logit only, no rollout) ----
    log("  Task 4: Boost logit analysis (5 prompts)...")
    alphas = [1.0, 1.5, 2.0, 3.0]
    logit_results = defaultdict(lambda: defaultdict(list))

    for pi, prompt in enumerate(EN_5):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            base_logits = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
        base_argmax = int(np.argmax(base_logits))

        for h_info in heads:
            L, H = h_info["layer"], h_info["head"]
            sc, ec = H * d_head, (H + 1) * d_head
            hk = f"L{L}_H{H}"
            for alpha in alphas:
                if alpha == 1.0:
                    pl = base_logits.copy()
                else:
                    handle = layers[L].self_attn.o_proj.register_forward_pre_hook(make_head_hook(sc, ec, alpha))
                    try:
                        with torch.no_grad():
                            pl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
                    except:
                        pl = base_logits.copy()
                    handle.remove()
                for tn, tid in proto_ids.items():
                    if tid < len(pl):
                        logit_results[hk][f"delta_{tn}_a{alpha}"].append(float(pl[tid] - base_logits[tid]))
                logit_results[hk][f"argmax_chg_a{alpha}"].append(int(int(np.argmax(pl)) != base_argmax))
        log(f"    {pi+1}/{len(EN_5)} prompts ({time.time()-t_start:.0f}s)")

    logit_summary = {hk: {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in d.items()}
                     for hk, d in logit_results.items()}

    results["task4"] = {"task": "task4_boost_analysis", "model": model_name,
        "alphas": alphas, "logit_summary": logit_summary}
    (model_dir / "task4_boost_analysis.json").write_text(
        json.dumps(results["task4"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved task4")
    for hk, s in logit_summary.items():
        log(f"    {hk}:")
        for a in alphas:
            ek = f"delta_EOS_a{a}"
            if ek in s:
                log(f"      a={a}: dEOS={s[ek]['mean']:.4f}  argmax_chg={s.get(f'argmax_chg_a{a}', {}).get('mean', 0):.2f}")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 5: Joint intervention (3 prompts) ----
    log("  Task 5: Joint intervention (3 prompts)...")
    last_layer = info.n_layers - 1
    target_mlp = layers[last_layer].mlp.down_proj
    top_ch = channels[:5]
    L0, H0 = heads[0]["layer"], heads[0]["head"]
    sc0, ec0 = H0 * d_head, (H0 + 1) * d_head

    conditions = [
        ("normal", 1.0, 1.0), ("ablate_head", 0.0, 1.0),
        ("boost_channel", 1.0, 2.0), ("ablate_head+boost_ch", 0.0, 2.0),
        ("boost_head+boost_ch", 2.0, 2.0),
    ]
    all_r5 = []
    for pi, prompt in enumerate(EN_5[:3]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        for cn, hs, cs in conditions:
            handles = []
            if hs != 1.0:
                handles.append(layers[L0].self_attn.o_proj.register_forward_pre_hook(make_head_hook(sc0, ec0, hs)))
            if cs != 1.0:
                handles.append(target_mlp.register_forward_pre_hook(make_channel_hook(top_ch, cs)))
            try:
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=MAX_TOKENS_ROLLOUT,
                                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                ng = len(gt)
            except Exception as e:
                gen = f"ERROR: {e}"; he = False; ng = 0
            for h in handles: h.remove()
            ce = evaluate_strict_clean(prompt, gen, he, ng)
            all_r5.append({"prompt": prompt, "condition": cn, "generated": gen[:200],
                           "has_eos": he, "n_tokens": ng, "strict_clean": ce["strict_clean"],
                           "is_ascii": ce["is_ascii"], "lang_switched": not ce["is_ascii"]})
        log(f"    {pi+1}/3 prompts ({time.time()-t_start:.0f}s)")

    cagg = defaultdict(lambda: {"eos": 0, "clean": 0, "switch": 0, "n": 0, "toks": []})
    for r in all_r5:
        c = r["condition"]; cagg[c]["eos"] += int(r["has_eos"])
        cagg[c]["clean"] += int(r["strict_clean"]); cagg[c]["switch"] += int(r["lang_switched"])
        cagg[c]["n"] += 1; cagg[c]["toks"].append(r["n_tokens"])
    s5 = {c: {"eos_rate": d["eos"]/max(d["n"],1), "strict_clean_rate": d["clean"]/max(d["n"],1),
              "lang_switch_rate": d["switch"]/max(d["n"],1),
              "mean_tokens": float(np.mean(d["toks"])) if d["toks"] else 0} for c, d in cagg.items()}

    results["task5"] = {"task": "task5_joint_intervention", "model": model_name,
        "primary_head": f"L{L0}_H{H0}", "summary": s5, "raw_results": all_r5}
    (model_dir / "task5_joint_intervention.json").write_text(
        json.dumps(results["task5"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved task5")
    for c in [c[0] for c in conditions]:
        s = s5.get(c, {})
        log(f"    {c:25s}: eos={s.get('eos_rate',0):.2f}  clean={s.get('strict_clean_rate',0):.2f}  "
            f"switch={s.get('lang_switch_rate',0):.2f}  toks={s.get('mean_tokens',0):.1f}")

    log(f"    Sample (prompt 0):")
    for r in all_r5:
        if r["prompt"] == EN_5[0]:
            log(f"      {r['condition']:25s}: eos={r['has_eos']}  clean={r['strict_clean']}  "
                f"text={r['generated'][:80]}")

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)

    save_path = RESULT_DIR / f"{model_name}_result.json"
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"Saved: {save_path}")

if __name__ == "__main__":
    run()
