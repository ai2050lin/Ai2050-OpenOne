"""
Phase 440: 属性中介的alpha sweep验证
============================================

验证Qwen3的类别-属性中介是否只在大alpha下出现。

如果mediation在clean switch阈值附近同步出现，
说明层级中介是真机制而非强制重写。
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import os
import time
import json
import numpy as np
import torch
from datetime import datetime
from model_utils import (get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"  [bf16] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  [bf16] {model_name} loaded, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def get_cat_logit(logits, tokenizer, words):
    ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in words]
    return float(np.mean([logits[i] for i in ids]))


def run_phase440(model_name, round_num):
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"Phase 440: {model_name} Round {round_num}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    W_U = get_W_U(model, model_name)

    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        W_E = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    elif hasattr(model, 'get_input_embeddings'):
        W_E = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    else:
        W_E = None

    print(f"  n_layers={n_layers}, d_model={d_model}")

    # Test objects
    test_cases = [
        {
            "obj": "apple", "src_cat": "fruit", "tgt_cat": "animal",
            "src_props": ["red", "sweet", "juicy"],
            "tgt_props": ["fur", "alive", "legs"],
            "cat_words": ["fruit"], "opp_words": ["animal"],
        },
        {
            "obj": "knife", "src_cat": "tool", "tgt_cat": "vehicle",
            "src_props": ["metal", "sharp", "handle"],
            "tgt_props": ["engine", "wheels", "fast"],
            "cat_words": ["tool"], "opp_words": ["vehicle"],
        },
    ]

    # Category direction
    def get_cat_dir(cat_words, opp_words):
        if W_E is not None:
            cat_e = [W_E[tokenizer.encode(w, add_special_tokens=False)[0]] for w in cat_words]
            opp_e = [W_E[tokenizer.encode(w, add_special_tokens=False)[0]] for w in opp_words]
            d = np.mean(cat_e, axis=0) - np.mean(opp_e, axis=0)
        else:
            cat_e = [W_U[:, tokenizer.encode(w, add_special_tokens=False)[0]] for w in cat_words]
            opp_e = [W_U[:, tokenizer.encode(w, add_special_tokens=False)[0]] for w in opp_words]
            d = np.mean(cat_e, axis=1) - np.mean(opp_e, axis=1)
        return d / (np.linalg.norm(d) + 1e-8)

    alphas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    results = {
        "model": model_name, "round": round_num,
        "n_layers": n_layers, "alphas": alphas,
        "timestamp": datetime.now().isoformat(),
        "per_test": {},
    }

    for tc in test_cases:
        obj = tc["obj"]
        src_cat = tc["src_cat"]
        tgt_cat = tc["tgt_cat"]
        print(f"\n  --- {obj}: {src_cat}→{tgt_cat} ---")

        prompt = f"An {obj} is a kind of"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

        obj_pos = None
        for i, t in enumerate(tokens):
            if obj[:3].lower() in t.lower():
                obj_pos = i
                break
        if obj_pos is None:
            obj_pos = 1

        # Baseline logits
        with torch.no_grad():
            base_out = model(input_ids=input_ids, attention_mask=attention_mask)
            base_logits = base_out.logits[0, -1].float().cpu().numpy()

        # Baseline property logits
        base_src_prop = [get_cat_logit(base_logits, tokenizer, [p]) for p in tc["src_props"]]
        base_tgt_prop = [get_cat_logit(base_logits, tokenizer, [p]) for p in tc["tgt_props"]]
        base_cat_gap = get_cat_logit(base_logits, tokenizer, tc["cat_words"]) - \
                       get_cat_logit(base_logits, tokenizer, tc["opp_words"])

        # Category direction for pushing toward tgt_cat
        # Push: src_cat → tgt_cat means we want to move AWAY from src_cat
        d_src = get_cat_dir(tc["cat_words"], tc["opp_words"])
        # Push in direction of tgt_cat (against src_cat)
        d_push = -d_src  # Reverse the src_cat direction to push toward tgt_cat

        d_push_t = torch.tensor(d_push, dtype=torch.bfloat16, device=device)

        alpha_results = {}

        for alpha in alphas:
            # Inject push direction at obj_pos in first layer
            def make_pert_hook(pos, d, a):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        o = out[0].clone()
                        o[0, pos] = o[0, pos] + a * d.to(o.device, o.dtype)
                        return (o,) + out[1:]
                    return out
                return hook

            hook = layers[0].register_forward_hook(make_pert_hook(obj_pos, d_push_t, alpha))
            with torch.no_grad():
                pert_out = model(input_ids=input_ids, attention_mask=attention_mask)
                pert_logits = pert_out.logits[0, -1].float().cpu().numpy()
            hook.remove()

            # Measure property changes
            pert_src_prop = [get_cat_logit(pert_logits, tokenizer, [p]) for p in tc["src_props"]]
            pert_tgt_prop = [get_cat_logit(pert_logits, tokenizer, [p]) for p in tc["tgt_props"]]
            pert_cat_gap = get_cat_logit(pert_logits, tokenizer, tc["cat_words"]) - \
                           get_cat_logit(pert_logits, tokenizer, tc["opp_words"])

            # Source property delta (should decrease)
            src_prop_delta = np.mean(pert_src_prop) - np.mean(base_src_prop)
            # Target property delta (should increase)
            tgt_prop_delta = np.mean(pert_tgt_prop) - np.mean(base_tgt_prop)
            # Category shift
            cat_shift = pert_cat_gap - base_cat_gap
            # Mediation score
            mediation = tgt_prop_delta - src_prop_delta

            alpha_results[str(alpha)] = {
                "alpha": alpha,
                "src_prop_delta": round(float(src_prop_delta), 4),
                "tgt_prop_delta": round(float(tgt_prop_delta), 4),
                "cat_shift": round(float(cat_shift), 4),
                "mediation": round(float(mediation), 4),
                "src_props": {p: round(float(pert_src_prop[i] - base_src_prop[i]), 4)
                             for i, p in enumerate(tc["src_props"])},
                "tgt_props": {p: round(float(pert_tgt_prop[i] - base_tgt_prop[i]), 4)
                             for i, p in enumerate(tc["tgt_props"])},
            }

            print(f"  alpha={alpha}: src_delta={src_prop_delta:.4f}, tgt_delta={tgt_prop_delta:.4f}, "
                  f"cat_shift={cat_shift:.4f}, mediation={mediation:.4f}")

        results["per_test"][f"{obj}_{src_cat}to{tgt_cat}"] = {
            "obj": obj, "src_cat": src_cat, "tgt_cat": tgt_cat,
            "alpha_sweep": alpha_results,
        }

    # Save
    out_dir = "d:/Ai2050/TransformerLens-Project/results/phase440_alpha_sweep"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_name}_phase440_r{round_num}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_phase440(model_name, round_num)
