#!/usr/bin/env python3
"""Phase 964 DS7B: direct EOS injection (b=20, delayed) + head diff search."""
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
from phase964_forward_diff import (
    log, get_head_dims, evaluate_strict_clean, make_head_hook, make_eos_inject_hook,
    make_channel_hook, EN_PROMPTS_50, MAX_TOKENS, RESULT_DIR,
    LOCK_HEADS, EOS_CHANNELS, EOS_PROMOTER_HEADS
)

model_name = "deepseek7b"

def run():
    log(f"Phase 964 DS7B")
    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)
    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers
    eos_id = tokenizer.eos_token_id
    log(f"  {info.model_class}, {info.n_layers}L  (load: {time.time()-t0:.0f}s)")

    prompts = EN_PROMPTS_50[:5]
    results = {"model": model_name}
    t_start = time.time()

    # ---- Diagnostic ----
    log("  Diagnostic: base EOS logit...")
    for pi, prompt in enumerate(prompts[:3]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
        top5_idx = np.argsort(logits)[-5:][::-1]
        eos_logit = float(logits[eos_id]) if eos_id else 0
        top1 = float(logits[top5_idx[0]])
        log(f"    p{pi}: EOS={eos_logit:.3f}  top1={top1:.3f}  gap={top1-eos_logit:.3f}  "
            f"top3={[(tokenizer.decode([int(i)]), float(logits[i])) for i in top5_idx[:3]]}")
        results.setdefault("diagnostics", []).append({"prompt": prompt, "eos_logit": eos_logit, "top1": top1, "gap": top1-eos_logit})

    # ---- Task 2: Direct EOS injection ----
    log("  Task 2: Direct EOS injection...")
    inject_conditions = [("normal", 0, 0), ("b=10", 10, 0), ("b=20", 20, 0),
                         ("delayed2_b=20", 20, 2), ("delayed3_b=20", 20, 3),
                         ("delayed2_b=15", 15, 2)]
    all_r2 = []
    for pi, prompt in enumerate(prompts):
        for cn, bias, delay in inject_conditions:
            handle = None
            if bias > 0:
                handle = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, bias, delay))
            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                he = gt[-1].item() == eos_id if eos_id else False
                ng = len(gt)
            except Exception as e:
                gen = f"ERROR: {e}"; he = False; ng = 0
            if handle: handle.remove()
            ce = evaluate_strict_clean(prompt, gen, he, ng)
            all_r2.append({"prompt": prompt, "condition": cn, "generated": gen[:200],
                           "has_eos": he, "n_tokens": ng, "strict_clean": ce["strict_clean"],
                           "lang_switched": not ce["is_ascii"]})
            log(f"    p{pi} {cn}: eos={he} clean={ce['strict_clean']} toks={ng} text={gen[:60]}")

    cagg = defaultdict(lambda: {"eos": 0, "clean": 0, "n": 0, "toks": []})
    for r in all_r2:
        c = r["condition"]; cagg[c]["eos"] += int(r["has_eos"])
        cagg[c]["clean"] += int(r["strict_clean"])
        cagg[c]["n"] += 1; cagg[c]["toks"].append(r["n_tokens"])
    s2 = {c: {"eos_rate": d["eos"]/max(d["n"],1), "strict_clean_rate": d["clean"]/max(d["n"],1),
              "mean_tokens": float(np.mean(d["toks"])) if d["toks"] else 0} for c, d in cagg.items()}
    results["task2"] = {"summary": s2, "raw_results": all_r2}
    log(f"    Summary:")
    for c in [c[0] for c in inject_conditions]:
        sv = s2.get(c, {})
        log(f"      {c:20s}: eos={sv.get('eos_rate',0):.2f}  clean={sv.get('strict_clean_rate',0):.2f}  toks={sv.get('mean_tokens',0):.1f}")
    log(f"    Task 2 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Task 1: Head diff (last 3 layers) ----
    log("  Task 1: Head diff (last 3 layers)...")
    search_layers = list(range(max(0, n_layers - 3), n_layers))
    proto_ids = {"EOS": eos_id}
    for ts in [".", " "]:
        toks = tokenizer.encode(ts, add_special_tokens=False)
        if toks: proto_ids[ts] = toks[0]

    diff_results = defaultdict(lambda: defaultdict(list))
    for pi, prompt in enumerate(prompts[:3]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            base_logits = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
        for L in search_layers:
            for H in range(n_heads):
                sc = H * d_head; ec = sc + d_head
                handle = layers[L].self_attn.o_proj.register_forward_pre_hook(make_head_hook(sc, ec, 0.0))
                try:
                    with torch.no_grad():
                        pl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
                except: pl = base_logits.copy()
                handle.remove()
                key = f"L{L}_H{H}"
                for tn, tid in proto_ids.items():
                    if tid < len(pl):
                        diff_results[key][tn].append(float(pl[tid] - base_logits[tid]))
        log(f"    {pi+1}/3 prompts ({time.time()-t_start:.0f}s)")

    head_diff = {k: {tn: float(np.mean(v)) for tn, v in d.items()} for k, d in diff_results.items()}
    eos_promoters = sorted(head_diff.items(), key=lambda x: x[1].get("EOS", 0))[:15]
    eos_suppressors = sorted(head_diff.items(), key=lambda x: -x[1].get("EOS", 0))[:10]
    results["task1"] = {"search_layers": search_layers,
        "top_eos_promoters": [(k, v.get("EOS", 0)) for k, v in eos_promoters if v.get("EOS", 0) < -0.01],
        "top_eos_suppressors": [(k, v.get("EOS", 0)) for k, v in eos_suppressors if v.get("EOS", 0) > 0.01],
        "all_heads": head_diff}
    log(f"    Top EOS promoters:")
    for k, ed in [(k, v.get("EOS", 0)) for k, v in eos_promoters if v.get("EOS", 0) < -0.01][:10]:
        log(f"      {k}: ΔEOS_ablate={ed:.4f}")
    log(f"    Top EOS suppressors:")
    for k, ed in [(k, v.get("EOS", 0)) for k, v in eos_suppressors if v.get("EOS", 0) > 0.01][:5]:
        log(f"      {k}: ΔEOS_ablate={ed:.4f}")
    log(f"    Task 1 done ({time.time()-t_start:.0f}s)")

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    save_path = RESULT_DIR / f"{model_name}_result.json"
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"Saved: {save_path}")

if __name__ == "__main__":
    run()
