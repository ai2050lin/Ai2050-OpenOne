#!/usr/bin/env python3
"""Phase 962 GLM4 task3 re-run (ultra-minimal, 2 prompts only)."""
import sys, json, time
from pathlib import Path
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, get_layers, get_model_info, release_model
from phase951_protocol_atlas import ensure_dir
from phase962_eos_promoter_search import (
    log, get_head_dims, make_head_hook, MODEL_HEADS, EN_PROMPTS_50, MAX_TOKENS, RESULT_DIR
)

model_name = "glm4"

def run():
    log(f"Phase 962 GLM4 task3 ultra-minimal (2 prompts)")
    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)
    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  loaded ({time.time()-t0:.0f}s)")

    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    L0, H0 = MODEL_HEADS[model_name][0]["layer"], MODEL_HEADS[model_name][0]["head"]
    sc, ec = H0 * d_head, (H0 + 1) * d_head

    prompts = EN_PROMPTS_50[:2]
    lambdas = [0.0, 1.0, 2.0]
    all_r = []
    for pi, prompt in enumerate(prompts):
        for lam in lambdas:
            handles = []
            if lam != 0.0:
                handles.append(layers[L0].self_attn.o_proj.register_forward_pre_hook(
                    make_head_hook(sc, ec, 1.0 - lam)))
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
            is_ascii = all(ord(c) < 256 for c in gen)
            all_r.append({"prompt": prompt, "lambda": lam, "generated": gen[:200],
                          "has_eos": he, "n_tokens": ng, "lang_switched": not is_ascii})
            log(f"    p{pi} λ={lam:.1f}: eos={he} switch={not is_ascii} text={gen[:60]}")

    # Summary
    for lam in lambdas:
        lr = [r for r in all_r if r["lambda"] == lam]
        eos = np.mean([r["has_eos"] for r in lr]) if lr else 0
        sw = np.mean([r["lang_switched"] for r in lr]) if lr else 0
        log(f"    λ={lam:.1f}: eos={eos:.2f} switch={sw:.2f}")

    # Save
    result = {"task": "task3_reverse_lock_fixed", "model": model_name,
              "primary_head": f"L{L0}_H{H0}", "lambdas": lambdas,
              "raw_results": all_r}
    (model_dir / "task3_reverse_lock.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved task3")

    # Merge
    existing_path = RESULT_DIR / f"{model_name}_result.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        existing["task3"] = result
        existing["task3_4_fixed"] = True
        existing_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        log(f"Updated: {existing_path}")

    release_model(model)
    log(f"Done ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    run()
