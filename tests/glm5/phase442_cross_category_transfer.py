"""
Phase 442: 跨类别运输迁移补全
============================================

Phase 438的同类迁移有正结果，但缺少跨类别负对照。
本实验加入:
1. 同类迁移 (same-class transfer)
2. 跨类迁移 (cross-class transfer)
3. 随机方向迁移 (random direction control)
4. 反向类别迁移 (reverse-class transfer)

如果运输方向是类别特异的，则:
- 同类迁移 > 随机迁移 > 跨类迁移 ≈ 反向迁移
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


def compute_cat_logit_gap(logits, tokenizer, cat_words, opp_words):
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_score = float(np.mean([logits[i] for i in cat_ids]))
    opp_score = float(np.mean([logits[i] for i in opp_ids]))
    return cat_score - opp_score


def get_category_direction(tokenizer, W_E, cat_words, opp_words):
    """计算类别方向"""
    cat_embs = [W_E[tokenizer.encode(w, add_special_tokens=False)[0]] for w in cat_words]
    opp_embs = [W_E[tokenizer.encode(w, add_special_tokens=False)[0]] for w in opp_words]
    d = np.mean(cat_embs, axis=0) - np.mean(opp_embs, axis=0)
    return d / (np.linalg.norm(d) + 1e-8)


def run_phase442(model_name, round_num):
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"Phase 442: {model_name} Round {round_num}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    W_U = get_W_U(model, model_name)

    # Get W_E
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        W_E = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    elif hasattr(model, 'get_input_embeddings'):
        W_E = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    else:
        W_E = None

    print(f"  n_layers={n_layers}, d_model={d_model}")

    # Categories
    categories = {
        "fruit": {"words": ["fruit"], "opp": ["animal", "tool", "vehicle"],
                  "objects": ["apple", "orange", "lemon", "grape"]},
        "tool": {"words": ["tool"], "opp": ["fruit", "animal", "vehicle"],
                 "objects": ["knife", "hammer", "spoon", "axe"]},
        "animal": {"words": ["animal"], "opp": ["fruit", "tool", "vehicle"],
                   "objects": ["dog", "cat", "horse", "rabbit"]},
        "vehicle": {"words": ["vehicle"], "opp": ["fruit", "tool", "animal"],
                    "objects": ["car", "train", "bus", "bicycle"]},
    }

    # Category directions
    cat_dirs = {}
    for cat_name, cat_info in categories.items():
        if W_E is not None:
            d = get_category_direction(tokenizer, W_E, cat_info["words"], cat_info["opp"])
        else:
            # Use W_U
            cat_embs = [W_U[:, tokenizer.encode(w, add_special_tokens=False)[0]] for w in cat_info["words"]]
            opp_embs = [W_U[:, tokenizer.encode(w, add_special_tokens=False)[0]] for w in cat_info["opp"]]
            d = np.mean(cat_embs, axis=1) - np.mean(opp_embs, axis=1)
            d = d / (np.linalg.norm(d) + 1e-8)
        cat_dirs[cat_name] = d

    alpha = 1.5  # Perturbation strength

    results = {
        "model": model_name, "round": round_num,
        "n_layers": n_layers, "alpha": alpha,
        "timestamp": datetime.now().isoformat(),
        "same_class": {},
        "cross_class": {},
        "random_control": {},
        "reverse_class": {},
    }

    rng = np.random.RandomState(42)

    # ============== 1. SAME-CLASS TRANSFER ==============
    print(f"\n--- Same-Class Transfer ---")
    same_class_pairs = [
        ("apple", "orange", "fruit"),
        ("knife", "hammer", "tool"),
        ("dog", "cat", "animal"),
        ("car", "train", "vehicle"),
    ]

    for src, tgt, cat in same_class_pairs:
        print(f"  {src}→{tgt} ({cat})")

        # Source: run with category perturbation at obj_pos, get delta at last token
        src_prompt = f"An {src} is a kind of"
        src_inputs = tokenizer(src_prompt, return_tensors="pt", truncation=True, max_length=64)
        src_ids = src_inputs["input_ids"].to(device)
        src_mask = src_inputs["attention_mask"].to(device)
        src_tokens = tokenizer.convert_ids_to_tokens(src_ids[0].cpu().numpy())

        src_obj_pos = None
        for i, t in enumerate(src_tokens):
            if src[:3].lower() in t.lower():
                src_obj_pos = i
                break
        if src_obj_pos is None:
            src_obj_pos = 1
        src_last_pos = len(src_tokens) - 1

        # Baseline for source
        h_src_base = {}
        def cap_src_base(li):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    h_src_base[li] = out[0].detach().float().cpu()
            return hook

        hooks = [layers[li].register_forward_hook(cap_src_base(li)) for li in [n_layers - 1]]
        with torch.no_grad():
            model(input_ids=src_ids, attention_mask=src_mask)
        for h in hooks:
            h.remove()

        # Perturbed for source (inject category direction)
        d_cat_t = torch.tensor(cat_dirs[cat], dtype=torch.bfloat16, device=device)
        h_src_pert = {}

        def make_pert_hook(pos, d, a):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    o = out[0].clone()
                    o[0, pos] = o[0, pos] + a * d.to(o.device, o.dtype)
                    return (o,) + out[1:]
                return out
            return hook

        hooks = [layers[0].register_forward_hook(make_pert_hook(src_obj_pos, d_cat_t, alpha))]
        hooks.append(layers[n_layers - 1].register_forward_hook(
            lambda m, i, o: h_src_pert.update({n_layers - 1: o[0].detach().float().cpu()})
            if isinstance(o, tuple) else None))

        with torch.no_grad():
            model(input_ids=src_ids, attention_mask=src_mask)
        for h in hooks:
            h.remove()

        # Source delta at last position
        if n_layers - 1 in h_src_pert and n_layers - 1 in h_src_base:
            src_delta = h_src_pert[n_layers - 1][0, src_last_pos].numpy() - \
                       h_src_base[n_layers - 1][0, src_last_pos].numpy()
        else:
            src_delta = np.zeros(d_model)

        # Target: inject source delta at obj_pos
        tgt_prompt = f"An {tgt} is a kind of"
        tgt_inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
        tgt_ids = tgt_inputs["input_ids"].to(device)
        tgt_mask = tgt_inputs["attention_mask"].to(device)
        tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_ids[0].cpu().numpy())

        tgt_obj_pos = None
        for i, t in enumerate(tgt_tokens):
            if tgt[:3].lower() in t.lower():
                tgt_obj_pos = i
                break
        if tgt_obj_pos is None:
            tgt_obj_pos = 1
        tgt_last_pos = len(tgt_tokens) - 1

        # Target baseline
        h_tgt_base = {}
        hooks = [layers[n_layers - 1].register_forward_hook(
            lambda m, i, o, li=n_layers - 1: h_tgt_base.update({li: o[0].detach().float().cpu()})
            if isinstance(o, tuple) else None)]
        with torch.no_grad():
            tgt_base_out = model(input_ids=tgt_ids, attention_mask=tgt_mask)
            tgt_base_logits = tgt_base_out.logits[0, -1].float().cpu().numpy()
        for h in hooks:
            h.remove()

        # Target with src delta injected at last layer
        src_delta_t = torch.tensor(src_delta, dtype=torch.bfloat16, device=device)
        h_tgt_trans = {}
        hooks = [layers[n_layers - 1].register_forward_hook(
            make_pert_hook(tgt_last_pos, src_delta_t, 1.0))]
        hooks.append(layers[n_layers - 1].register_forward_hook(
            lambda m, i, o: h_tgt_trans.update({n_layers - 1: o[0].detach().float().cpu()})
            if isinstance(o, tuple) else None))

        with torch.no_grad():
            tgt_trans_out = model(input_ids=tgt_ids, attention_mask=tgt_mask)
            tgt_trans_logits = tgt_trans_out.logits[0, -1].float().cpu().numpy()
        for h in hooks:
            h.remove()

        # Compute transfer score
        cat_info = categories[cat]
        base_gap = compute_cat_logit_gap(tgt_base_logits, tokenizer, cat_info["words"], cat_info["opp"])
        trans_gap = compute_cat_logit_gap(tgt_trans_logits, tokenizer, cat_info["words"], cat_info["opp"])
        transfer_score = trans_gap - base_gap

        results["same_class"][f"{src}→{tgt}"] = {
            "src": src, "tgt": tgt, "category": cat,
            "base_cat_gap": round(float(base_gap), 4),
            "trans_cat_gap": round(float(trans_gap), 4),
            "transfer_score": round(float(transfer_score), 4),
        }
        print(f"    base_gap={base_gap:.4f}, trans_gap={trans_gap:.4f}, "
              f"transfer={transfer_score:.4f}")

    # ============== 2. CROSS-CLASS TRANSFER ==============
    print(f"\n--- Cross-Class Transfer ---")
    cross_class_pairs = [
        ("apple", "knife", "fruit", "tool"),
        ("apple", "dog", "fruit", "animal"),
        ("knife", "car", "tool", "vehicle"),
        ("dog", "apple", "animal", "fruit"),
    ]

    for src, tgt, src_cat, tgt_cat in cross_class_pairs:
        print(f"  {src}({src_cat})→{tgt}({tgt_cat})")

        # Source delta (fruit category direction)
        src_prompt = f"An {src} is a kind of"
        src_inputs = tokenizer(src_prompt, return_tensors="pt", truncation=True, max_length=64)
        src_ids = src_inputs["input_ids"].to(device)
        src_mask = src_inputs["attention_mask"].to(device)
        src_tokens = tokenizer.convert_ids_to_tokens(src_ids[0].cpu().numpy())
        src_obj_pos = 1
        for i, t in enumerate(src_tokens):
            if src[:3].lower() in t.lower():
                src_obj_pos = i
                break
        src_last_pos = len(src_tokens) - 1

        # Baseline and perturbed for source
        h_base = {}
        hooks = [layers[n_layers - 1].register_forward_hook(
            lambda m, i, o: h_base.update({n_layers - 1: o[0].detach().float().cpu()})
            if isinstance(o, tuple) else None)]
        with torch.no_grad():
            model(input_ids=src_ids, attention_mask=src_mask)
        for h in hooks:
            h.remove()

        d_cat_t = torch.tensor(cat_dirs[src_cat], dtype=torch.bfloat16, device=device)
        h_pert = {}
        hooks = [layers[0].register_forward_hook(make_pert_hook(src_obj_pos, d_cat_t, alpha))]
        hooks.append(layers[n_layers - 1].register_forward_hook(
            lambda m, i, o: h_pert.update({n_layers - 1: o[0].detach().float().cpu()})
            if isinstance(o, tuple) else None))
        with torch.no_grad():
            model(input_ids=src_ids, attention_mask=src_mask)
        for h in hooks:
            h.remove()

        if n_layers - 1 in h_pert and n_layers - 1 in h_base:
            src_delta = h_pert[n_layers - 1][0, src_last_pos].numpy() - \
                       h_base[n_layers - 1][0, src_last_pos].numpy()
        else:
            src_delta = np.zeros(d_model)

        # Target baseline and injected
        tgt_prompt = f"An {tgt} is a kind of"
        tgt_inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
        tgt_ids = tgt_inputs["input_ids"].to(device)
        tgt_mask = tgt_inputs["attention_mask"].to(device)
        tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_ids[0].cpu().numpy())
        tgt_last_pos = len(tgt_tokens) - 1

        h_tgt_base = {}
        hooks = [layers[n_layers - 1].register_forward_hook(
            lambda m, i, o: h_tgt_base.update({n_layers - 1: o[0].detach().float().cpu()})
            if isinstance(o, tuple) else None)]
        with torch.no_grad():
            tgt_base_out = model(input_ids=tgt_ids, attention_mask=tgt_mask)
            tgt_base_logits = tgt_base_out.logits[0, -1].float().cpu().numpy()
        for h in hooks:
            h.remove()

        src_delta_t = torch.tensor(src_delta, dtype=torch.bfloat16, device=device)
        hooks = [layers[n_layers - 1].register_forward_hook(
            make_pert_hook(tgt_last_pos, src_delta_t, 1.0))]
        with torch.no_grad():
            tgt_trans_out = model(input_ids=tgt_ids, attention_mask=tgt_mask)
            tgt_trans_logits = tgt_trans_out.logits[0, -1].float().cpu().numpy()
        for h in hooks:
            h.remove()

        # Transfer score: how much does src_cat direction move tgt toward src_cat?
        src_cat_info = categories[src_cat]
        cross_gap = compute_cat_logit_gap(tgt_trans_logits, tokenizer,
                                           src_cat_info["words"], src_cat_info["opp"])
        base_cross_gap = compute_cat_logit_gap(tgt_base_logits, tokenizer,
                                                src_cat_info["words"], src_cat_info["opp"])
        transfer_score = cross_gap - base_cross_gap

        results["cross_class"][f"{src}({src_cat})→{tgt}({tgt_cat})"] = {
            "src": src, "tgt": tgt, "src_cat": src_cat, "tgt_cat": tgt_cat,
            "base_cross_gap": round(float(base_cross_gap), 4),
            "trans_cross_gap": round(float(cross_gap), 4),
            "transfer_score": round(float(transfer_score), 4),
        }
        print(f"    base_cross={base_cross_gap:.4f}, trans_cross={cross_gap:.4f}, "
              f"transfer={transfer_score:.4f}")

    # ============== 3. RANDOM DIRECTION CONTROL ==============
    print(f"\n--- Random Direction Control ---")
    for src, tgt, cat in same_class_pairs[:2]:  # Only test 2 pairs
        print(f"  {src}→{tgt} ({cat}) - random direction")

        # Random direction of same norm as src_delta
        random_dir = rng.randn(d_model).astype(np.float32)
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-8)

        tgt_prompt = f"An {tgt} is a kind of"
        tgt_inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
        tgt_ids = tgt_inputs["input_ids"].to(device)
        tgt_mask = tgt_inputs["attention_mask"].to(device)
        tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_ids[0].cpu().numpy())
        tgt_last_pos = len(tgt_tokens) - 1

        # Baseline
        with torch.no_grad():
            tgt_base_out = model(input_ids=tgt_ids, attention_mask=tgt_mask)
            tgt_base_logits = tgt_base_out.logits[0, -1].float().cpu().numpy()

        # Inject random direction
        rand_t = torch.tensor(random_dir, dtype=torch.bfloat16, device=device)
        hooks = [layers[n_layers - 1].register_forward_hook(
            make_pert_hook(tgt_last_pos, rand_t, 10.0))]  # Larger norm for comparison
        with torch.no_grad():
            tgt_rand_out = model(input_ids=tgt_ids, attention_mask=tgt_mask)
            tgt_rand_logits = tgt_rand_out.logits[0, -1].float().cpu().numpy()
        for h in hooks:
            h.remove()

        cat_info = categories[cat]
        base_gap = compute_cat_logit_gap(tgt_base_logits, tokenizer, cat_info["words"], cat_info["opp"])
        rand_gap = compute_cat_logit_gap(tgt_rand_logits, tokenizer, cat_info["words"], cat_info["opp"])
        random_transfer = rand_gap - base_gap

        results["random_control"][f"{src}→{tgt}_rand"] = {
            "src": src, "tgt": tgt, "category": cat,
            "random_transfer": round(float(random_transfer), 4),
        }
        print(f"    random_transfer={random_transfer:.4f}")

    # ============== 4. REVERSE-CLASS TRANSFER ==============
    print(f"\n--- Reverse-Class Transfer ---")
    reverse_pairs = [
        ("apple", "knife", "fruit", "tool"),  # fruit direction on tool object
        ("dog", "car", "animal", "vehicle"),  # animal direction on vehicle object
    ]

    for src, tgt, src_cat, tgt_cat in reverse_pairs:
        print(f"  {src}({src_cat}) direction → {tgt}({tgt_cat})")

        tgt_prompt = f"An {tgt} is a kind of"
        tgt_inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
        tgt_ids = tgt_inputs["input_ids"].to(device)
        tgt_mask = tgt_inputs["attention_mask"].to(device)
        tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_ids[0].cpu().numpy())
        tgt_obj_pos = 1
        for i, t in enumerate(tgt_tokens):
            if tgt[:3].lower() in t.lower():
                tgt_obj_pos = i
                break

        # Baseline
        with torch.no_grad():
            tgt_base_out = model(input_ids=tgt_ids, attention_mask=tgt_mask)
            tgt_base_logits = tgt_base_out.logits[0, -1].float().cpu().numpy()

        # Inject src_cat direction at obj_pos
        d_cat_t = torch.tensor(cat_dirs[src_cat], dtype=torch.bfloat16, device=device)
        hooks = [layers[0].register_forward_hook(make_pert_hook(tgt_obj_pos, d_cat_t, alpha))]
        with torch.no_grad():
            tgt_rev_out = model(input_ids=tgt_ids, attention_mask=tgt_mask)
            tgt_rev_logits = tgt_rev_out.logits[0, -1].float().cpu().numpy()
        for h in hooks:
            h.remove()

        # Score: does src_cat direction push tgt toward src_cat?
        src_cat_info = categories[src_cat]
        base_gap = compute_cat_logit_gap(tgt_base_logits, tokenizer,
                                           src_cat_info["words"], src_cat_info["opp"])
        rev_gap = compute_cat_logit_gap(tgt_rev_logits, tokenizer,
                                          src_cat_info["words"], src_cat_info["opp"])
        reverse_transfer = rev_gap - base_gap

        results["reverse_class"][f"{src_cat}→{tgt}({tgt_cat})"] = {
            "src_cat": src_cat, "tgt": tgt, "tgt_cat": tgt_cat,
            "base_gap": round(float(base_gap), 4),
            "rev_gap": round(float(rev_gap), 4),
            "reverse_transfer": round(float(reverse_transfer), 4),
        }
        print(f"    base_gap={base_gap:.4f}, rev_gap={rev_gap:.4f}, "
              f"reverse_transfer={reverse_transfer:.4f}")

    # Save
    out_dir = "d:/Ai2050/TransformerLens-Project/results/phase442_cross_category_transfer"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_name}_phase442_r{round_num}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    # Summary
    print(f"\n{'='*40}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*40}")
    print("Same-class transfers:")
    for k, r in results["same_class"].items():
        print(f"  {k}: {r['transfer_score']}")
    print("Cross-class transfers:")
    for k, r in results["cross_class"].items():
        print(f"  {k}: {r['transfer_score']}")
    print("Random controls:")
    for k, r in results["random_control"].items():
        print(f"  {k}: {r['random_transfer']}")
    print("Reverse-class transfers:")
    for k, r in results["reverse_class"].items():
        print(f"  {k}: {r['reverse_transfer']}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_phase442(model_name, round_num)
