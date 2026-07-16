#!/usr/bin/env python3
"""Phase 961 fast runner for GLM4 — reduced prompt counts to fit timeout."""
import sys, gc, json, time, math
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
from phase961_mode_head_mechanism import (
    log, get_input_device, get_head_dims, classify_token, get_proto_token_ids,
    cosine_similarity, make_head_hook, make_channel_hook, make_capture_hook,
    evaluate_strict_clean, MODEL_HEADS, SUPER_CHANNELS, MAX_TOKENS_ROLLOUT,
    RESULT_DIR
)

model_name = "glm4"
PHASE = 961

# Reduced prompts for GLM4 (8bit, slow)
EN_PROMPTS_FAST = [
    "The capital of France is", "The largest planet is", "Water boils at",
    "The speed of light is", "The sun is a", "Dogs are", "The sky is",
    "Grass is", "Fire needs", "Ice is", "The Earth is", "A triangle has",
    "Shakespeare was", "Tokyo is the capital of", "The Pacific Ocean is",
    "Gold is a", "Plants need", "Humans breathe", "The moon is", "Birds can",
]

CN_PROMPTS_FAST = [
    "法国的首都是", "最大的行星是", "水的沸点是", "光速是", "太阳是一颗",
    "狗是", "天空是", "草是", "火需要", "冰是",
    "地球是", "三角形有", "莎士比亚是", "东京是", "太平洋是",
    "金是一种", "植物需要", "人类呼吸", "月亮是", "鸟能",
]

ROLLOUT_PROMPTS_FAST = EN_PROMPTS_FAST[:8]
JOINT_PROMPTS_FAST = EN_PROMPTS_FAST[:5]

def run_glm4():
    log(f"\n{'='*60}")
    log(f"Phase 961 FAST: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    n_heads, d_head = get_head_dims(model, info)
    log(f"  n_heads={n_heads}, d_head={d_head}")

    layers = get_layers(model)
    heads = MODEL_HEADS[model_name]
    channels = SUPER_CHANNELS[model_name]

    results = {"model": model_name}
    t_start = time.time()

    # ---- Task 1: Attention pattern (20 prompts) ----
    log("  Task 1: Attention pattern (20 prompts)...")
    attn_mass = defaultdict(lambda: defaultdict(float))
    attn_entropy = defaultdict(list)
    detail_attn = {}
    prompt_count = 0

    for pi, prompt in enumerate(EN_PROMPTS_FAST):
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
            if L >= len(attentions):
                continue
            attn_tensor = attentions[L]
            if H >= attn_tensor.shape[1]:
                continue
            head_attn = attn_tensor[0, H, -1, :].float().cpu().numpy()
            key = f"L{L}_H{H}"
            for i in range(seq_len):
                cat = classify_token(token_strs[i])
                attn_mass[key][cat] += float(head_attn[i])
            p = head_attn[head_attn > 1e-10]
            if len(p) > 0:
                ent = -np.sum(p * np.log(p))
                norm_ent = ent / max(math.log(seq_len), 1e-10)
                attn_entropy[key].append(float(norm_ent))
            if pi == 0:
                detail_attn[key] = {
                    "prompt": prompt, "tokens": token_strs,
                    "attention": head_attn.tolist(),
                    "categories": [classify_token(t) for t in token_strs],
                }

        prompt_count += 1
        del outputs, attentions
        if (pi + 1) % 10 == 0:
            log(f"    {pi+1}/{len(EN_PROMPTS_FAST)} prompts")

    t1_summary = {}
    for key, cats in attn_mass.items():
        total = sum(cats.values())
        t1_summary[key] = {cat: (mass / total if total > 0 else 0.0) for cat, mass in cats.items()}
        t1_summary[key]["n_prompts"] = prompt_count
        t1_summary[key]["mean_entropy"] = float(np.mean(attn_entropy[key])) if attn_entropy[key] else 0.0

    results["task1"] = {
        "task": "task1_attention_pattern", "model": model_name,
        "n_prompts": prompt_count,
        "attention_mass_by_category": t1_summary,
        "detail_first_prompt": detail_attn,
    }
    (model_dir / "task1_attention_pattern.json").write_text(
        json.dumps(results["task1"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved task1")
    for key, s in t1_summary.items():
        cats_str = "  ".join(f"{c}={v:.3f}" for c, v in sorted(s.items())
                             if c not in ["n_prompts", "mean_entropy"])
        log(f"    {key}: entropy={s.get('mean_entropy', 0):.3f}  {cats_str}")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 2+3: Head output + mode direction (10 EN + 10 CN) ----
    log("  Task 2+3: Head output + mode direction (10 EN + 10 CN)...")
    try:
        W_U = get_W_U(model, model_name)
        log(f"    W_U loaded: {W_U.shape}")
    except Exception as e:
        log(f"    W_U load failed: {e}")
        W_U = None

    proto_ids = get_proto_token_ids(tokenizer)
    en_layer_outputs = defaultdict(list)
    cn_layer_outputs = defaultdict(list)
    head_outputs = defaultdict(lambda: {"en": [], "cn": []})

    all_prompts = [(p, "en") for p in EN_PROMPTS_FAST[:10]] + [(p, "cn") for p in CN_PROMPTS_FAST[:10]]

    for pi, (prompt, lang) in enumerate(all_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        captured_normal = {}
        normal_hooks = []
        seen_layers = set()
        for h_info in heads:
            L = h_info["layer"]
            if L not in seen_layers:
                seen_layers.add(L)
                hk = layers[L].register_forward_hook(make_capture_hook(captured_normal, f"normal_L{L}"))
                normal_hooks.append(hk)

        with torch.no_grad():
            model(input_ids, use_cache=False)
        for hk in normal_hooks:
            hk.remove()

        for h_info in heads:
            L, H = h_info["layer"], h_info["head"]
            sc, ec = H * d_head, (H + 1) * d_head

            captured_ablated = {}
            hk_ab = layers[L].self_attn.o_proj.register_forward_pre_hook(make_head_hook(sc, ec, 0.0))
            hk_cap = layers[L].register_forward_hook(make_capture_hook(captured_ablated, f"ablated_L{L}"))

            with torch.no_grad():
                model(input_ids, use_cache=False)

            hk_ab.remove()
            hk_cap.remove()

            normal_vec = captured_normal.get(f"normal_L{L}")
            ablated_vec = captured_ablated.get(f"ablated_L{L}")
            if normal_vec is not None and ablated_vec is not None:
                O_h = normal_vec - ablated_vec
                head_outputs[f"L{L}_H{H}"][lang].append(O_h)

        for h_info in heads:
            L = h_info["layer"]
            key = f"normal_L{L}"
            if key in captured_normal:
                if lang == "en":
                    en_layer_outputs[L].append(captured_normal[key])
                else:
                    cn_layer_outputs[L].append(captured_normal[key])

        if (pi + 1) % 10 == 0:
            log(f"    {pi+1}/{len(all_prompts)} prompts")

    mode_directions = {}
    for L in en_layer_outputs:
        en_mean = np.mean(en_layer_outputs[L], axis=0)
        cn_mean = np.mean(cn_layer_outputs[L], axis=0)
        d_mode = en_mean - cn_mean
        mode_directions[L] = {
            "d_mode": d_mode.tolist(),
            "d_mode_norm": float(np.linalg.norm(d_mode)),
        }

    head_analysis = {}
    for h_info in heads:
        L, H = h_info["layer"], h_info["head"]
        head_key = f"L{L}_H{H}"
        en_Ohs = head_outputs[head_key]["en"]
        cn_Ohs = head_outputs[head_key]["cn"]
        if not en_Ohs:
            continue

        mean_Oh_en = np.mean(en_Ohs, axis=0)
        mean_Oh_cn = np.mean(cn_Ohs, axis=0) if cn_Ohs else np.zeros_like(mean_Oh_en)
        norm_en = float(np.linalg.norm(mean_Oh_en))
        norm_cn = float(np.linalg.norm(mean_Oh_cn))

        d_mode = np.array(mode_directions.get(L, {}).get("d_mode", [0] * len(mean_Oh_en)))
        cos_mode_en = cosine_similarity(mean_Oh_en, d_mode)
        cos_mode_cn = cosine_similarity(mean_Oh_cn, d_mode)

        wu_cosines = {}
        top_promoted = []
        if W_U is not None:
            for tok_name, tid in proto_ids.items():
                if tid < W_U.shape[0]:
                    wu_cosines[tok_name] = cosine_similarity(mean_Oh_en, W_U[tid])
            delta_logits = W_U @ mean_Oh_en
            top_idx = np.argsort(delta_logits)[-15:][::-1]
            for tid in top_idx:
                top_promoted.append({
                    "token": tokenizer.decode([int(tid)]),
                    "token_id": int(tid),
                    "delta_logit": float(delta_logits[tid]),
                })

        per_prompt_cos = [cosine_similarity(Oh, d_mode) for Oh in en_Ohs]

        head_analysis[head_key] = {
            "layer": L, "head": H, "role": h_info["role"],
            "norm_Oh_en": norm_en, "norm_Oh_cn": norm_cn,
            "cos_Oh_en_vs_dmode": cos_mode_en,
            "cos_Oh_cn_vs_dmode": cos_mode_cn,
            "wu_cosines": wu_cosines,
            "top_promoted_tokens_en": top_promoted,
            "mean_per_prompt_cos_mode_en": float(np.mean(per_prompt_cos)) if per_prompt_cos else 0.0,
        }

        log(f"    {head_key} ({h_info['role']}): ||O_h||={norm_en:.3f}  "
            f"cos_en={cos_mode_en:.4f}  cos_cn={cos_mode_cn:.4f}")
        if wu_cosines:
            top_wu = sorted(wu_cosines.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            log(f"      W_U cos: {'  '.join(f'{n}={v:.4f}' for n, v in top_wu)}")
        if top_promoted:
            top3 = top_promoted[:3]
            tok_strs = [f"'{t['token']}'({t['delta_logit']:.3f})" for t in top3]
            log(f"      Top promoted: {'  '.join(tok_strs)}")

    results["task2_3"] = {
        "task": "task2_3_head_output_mode", "model": model_name,
        "mode_directions": {str(k): v for k, v in mode_directions.items()},
        "head_analysis": head_analysis,
    }
    (model_dir / "task2_3_head_output_mode.json").write_text(
        json.dumps(results["task2_3"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved task2_3")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 4: Boost analysis (10 prompts logit, 4 rollout) ----
    log("  Task 4: Boost analysis (10 prompts logit, 4 rollout)...")
    alphas = [1.0, 1.5, 2.0, 3.0]
    prompts_10 = EN_PROMPTS_FAST[:10]
    logit_results = defaultdict(lambda: defaultdict(list))

    for pi, prompt in enumerate(prompts_10):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            base_out = model(input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()
        base_argmax = int(np.argmax(base_logits))
        base_top1 = float(np.sort(base_logits)[-1])
        base_top2 = float(np.sort(base_logits)[-2])

        for h_info in heads:
            L, H = h_info["layer"], h_info["head"]
            sc, ec = H * d_head, (H + 1) * d_head
            head_key = f"L{L}_H{H}"

            for alpha in alphas:
                if alpha == 1.0:
                    patched_logits = base_logits.copy()
                else:
                    handle = layers[L].self_attn.o_proj.register_forward_pre_hook(
                        make_head_hook(sc, ec, alpha))
                    try:
                        with torch.no_grad():
                            out = model(input_ids, use_cache=False)
                            patched_logits = out.logits[0, -1].detach().float().cpu().numpy()
                    except:
                        patched_logits = base_logits.copy()
                    handle.remove()

                patched_argmax = int(np.argmax(patched_logits))
                for tok_name, tid in proto_ids.items():
                    if tid < len(patched_logits):
                        delta = float(patched_logits[tid] - base_logits[tid])
                        logit_results[head_key][f"delta_{tok_name}_a{alpha}"].append(delta)
                logit_results[head_key][f"argmax_changed_a{alpha}"].append(int(patched_argmax != base_argmax))

        if (pi + 1) % 5 == 0:
            log(f"    Logit: {pi+1}/{len(prompts_10)} prompts")

    logit_summary = {}
    for head_key, data in logit_results.items():
        logit_summary[head_key] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                    for m, v in data.items()}

    # Rollout (4 prompts × 3 alphas)
    rollout_alphas = [1.0, 2.0, 3.0]
    rollout_prompts = prompts_10[:4]
    rollout_results = []
    primary_head = heads[0]
    L0, H0 = primary_head["layer"], primary_head["head"]
    sc0, ec0 = H0 * d_head, (H0 + 1) * d_head

    for pi, prompt in enumerate(rollout_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        for alpha in rollout_alphas:
            handles = []
            if alpha != 1.0:
                handles.append(layers[L0].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc0, ec0, alpha)))
            try:
                with torch.no_grad():
                    output_ids = model.generate(input_ids, max_new_tokens=MAX_TOKENS_ROLLOUT,
                                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gen_tokens = output_ids[0][input_ids.shape[1]:]
                generated = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                has_eos = gen_tokens[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                n_gen = len(gen_tokens)
            except Exception as e:
                generated = f"ERROR: {e}"; has_eos = False; n_gen = 0
            for h in handles: h.remove()
            is_ascii = all(ord(c) < 256 for c in generated)
            rollout_results.append({
                "prompt": prompt, "alpha": alpha,
                "generated": generated[:200], "has_eos": has_eos,
                "n_tokens": n_gen, "lang_switched": not is_ascii,
            })
        log(f"    Rollout: {pi+1}/{len(rollout_prompts)} prompts")

    rollout_summary = {}
    for alpha in rollout_alphas:
        ar = [r for r in rollout_results if r["alpha"] == alpha]
        rollout_summary[f"alpha_{alpha}"] = {
            "eos_rate": np.mean([r["has_eos"] for r in ar]) if ar else 0,
            "lang_switch_rate": np.mean([r["lang_switched"] for r in ar]) if ar else 0,
            "mean_tokens": np.mean([r["n_tokens"] for r in ar]) if ar else 0,
        }

    results["task4"] = {
        "task": "task4_boost_analysis", "model": model_name,
        "alphas": alphas, "primary_head": f"L{L0}_H{H0}",
        "logit_summary": logit_summary,
        "rollout_summary": rollout_summary,
        "rollout_results": rollout_results,
    }
    (model_dir / "task4_boost_analysis.json").write_text(
        json.dumps(results["task4"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved task4")

    for head_key, s in logit_summary.items():
        log(f"    {head_key}:")
        for alpha in alphas:
            ek = f"delta_EOS_a{alpha}"
            if ek in s:
                log(f"      a={alpha}: ΔEOS={s[ek]['mean']:.4f}  "
                    f"argmax_chg={s.get(f'argmax_changed_a{alpha}', {}).get('mean', 0):.2f}")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 5: Joint intervention (5 prompts) ----
    log("  Task 5: Joint intervention (5 prompts)...")
    last_layer = info.n_layers - 1
    target_mlp = layers[last_layer].mlp.down_proj
    top_channels = channels[:5]

    conditions = [
        ("normal", 1.0, 1.0),
        ("ablate_head", 0.0, 1.0),
        ("boost_channel", 1.0, 2.0),
        ("ablate_head+boost_ch", 0.0, 2.0),
        ("boost_head+boost_ch", 2.0, 2.0),
    ]

    all_results_5 = []
    for pi, prompt in enumerate(JOINT_PROMPTS_FAST):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        for cond_name, hs, cs in conditions:
            handles = []
            if hs != 1.0:
                handles.append(layers[L0].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc0, ec0, hs)))
            if cs != 1.0:
                handles.append(target_mlp.register_forward_pre_hook(
                    make_channel_hook(top_channels, cs)))
            try:
                with torch.no_grad():
                    output_ids = model.generate(input_ids, max_new_tokens=MAX_TOKENS_ROLLOUT,
                                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
                gen_tokens = output_ids[0][input_ids.shape[1]:]
                generated = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                has_eos = gen_tokens[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                n_gen = len(gen_tokens)
            except Exception as e:
                generated = f"ERROR: {e}"; has_eos = False; n_gen = 0
            for h in handles: h.remove()
            ce = evaluate_strict_clean(prompt, generated, has_eos, n_gen)
            all_results_5.append({
                "prompt": prompt, "condition": cond_name,
                "generated": generated[:200], "has_eos": has_eos,
                "n_tokens": n_gen, "strict_clean": ce["strict_clean"],
                "is_ascii": ce["is_ascii"], "lang_switched": not ce["is_ascii"],
            })
        log(f"    {pi+1}/{len(JOINT_PROMPTS_FAST)} prompts")

    cond_agg = defaultdict(lambda: {"eos": 0, "clean": 0, "switch": 0, "n": 0, "tokens": []})
    for r in all_results_5:
        c = r["condition"]
        cond_agg[c]["eos"] += int(r["has_eos"])
        cond_agg[c]["clean"] += int(r["strict_clean"])
        cond_agg[c]["switch"] += int(r["lang_switched"])
        cond_agg[c]["n"] += 1
        cond_agg[c]["tokens"].append(r["n_tokens"])

    summary5 = {}
    for c, d in cond_agg.items():
        summary5[c] = {
            "eos_rate": d["eos"] / max(d["n"], 1),
            "strict_clean_rate": d["clean"] / max(d["n"], 1),
            "lang_switch_rate": d["switch"] / max(d["n"], 1),
            "mean_tokens": float(np.mean(d["tokens"])) if d["tokens"] else 0,
        }

    results["task5"] = {
        "task": "task5_joint_intervention", "model": model_name,
        "primary_head": f"L{L0}_H{H0}",
        "summary": summary5, "raw_results": all_results_5,
    }
    (model_dir / "task5_joint_intervention.json").write_text(
        json.dumps(results["task5"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved task5")

    for c in [c[0] for c in conditions]:
        s = summary5.get(c, {})
        log(f"    {c:25s}: eos={s.get('eos_rate', 0):.2f}  clean={s.get('strict_clean_rate', 0):.2f}  "
            f"switch={s.get('lang_switch_rate', 0):.2f}  tokens={s.get('mean_tokens', 0):.1f}")

    log(f"    Sample (prompt 0):")
    for r in all_results_5:
        if r["prompt"] == JOINT_PROMPTS_FAST[0]:
            log(f"      {r['condition']:25s}: eos={r['has_eos']}  clean={r['strict_clean']}  "
                f"text={r['generated'][:80]}")

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    release_model(model)
    log(f"  {model_name} complete")

    save_path = RESULT_DIR / f"{model_name}_result.json"
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"Saved: {save_path}")


if __name__ == "__main__":
    run_glm4()
