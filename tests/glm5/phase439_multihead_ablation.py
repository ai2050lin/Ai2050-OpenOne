"""
Phase 439: 多头联合消融 + 分布式路由验证
============================================

验证Phase 434的单头消融低效是否因为运输是分布式过程。

方法:
1. 在embedding层注入类别扰动(自然运输)
2. 选top-k候选头做联合消融
3. 加random-k对照
4. 测三指标: delta_norm_score, direction_cos, readout_score

关键假设:
- 如果top-k联合消融显著强于random-k, 说明注意力头集合参与类别运输
- 如果top-k效果随k递增, 说明运输是分布式冗余过程
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
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)


def load_model_bf16(model_name):
    """BF16+auto加载"""
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


def get_candidate_heads(model_name):
    """基于Phase 431结果的候选头列表(高last->obj注意力)"""
    candidates = {
        "qwen3": [
            (3, 16), (6, 16), (14, 12), (10, 16), (2, 16),
            (5, 16), (8, 12), (18, 16), (1, 16), (4, 16),
            (12, 16), (15, 12), (20, 16), (7, 16), (9, 16),
            (22, 16), (25, 16), (28, 16), (31, 16), (33, 16),
        ],
        "glm4": [
            (4, 17), (3, 17), (2, 17), (1, 17), (0, 17),
            (4, 16), (3, 16), (2, 16), (1, 16), (5, 17),
            (6, 17), (7, 17), (8, 17), (10, 17), (12, 17),
            (14, 17), (16, 17), (18, 17), (20, 17), (22, 17),
        ],
        "deepseek7b": [
            (27, 12), (27, 10), (26, 12), (26, 10), (25, 12),
            (25, 10), (24, 12), (24, 10), (23, 12), (22, 10),
            (21, 12), (20, 12), (19, 10), (18, 12), (17, 10),
            (16, 12), (15, 10), (14, 12), (13, 10), (12, 12),
        ],
    }
    return candidates.get(model_name, [])


def compute_cat_logit_gap(logits, tokenizer, cat_words, opp_words):
    """计算category logit gap"""
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_score = sum(float(logits[i]) for i in cat_ids) / len(cat_ids)
    opp_score = sum(float(logits[i]) for i in opp_ids) / len(opp_ids)
    return cat_score - opp_score


def run_model_with_hooks(model, input_ids, attention_mask, capture_layers,
                         zero_attn_heads=None, embed_perturb=None):
    """
    Run model with hooks.
    
    Args:
        zero_attn_heads: list of (layer, head) to zero out
        embed_perturb: (position, direction_tensor, alpha) for embedding perturbation
        capture_layers: list of layer indices to capture output
    
    Returns:
        captured: dict of layer_idx -> hidden state tensor
        logits: final logits
    """
    layers = get_layers(model)
    n_layers = len(layers)
    
    # Get model config for head dimensions
    n_heads = getattr(model.config, 'num_attention_heads',
                       getattr(model.config, 'num_heads', 32))
    d_model = layers[0].self_attn.q_proj.weight.shape[1]
    head_dim = d_model // n_heads
    
    captured = {}
    
    # Embedding perturbation hook
    embed_hooks = []
    if embed_perturb is not None:
        pos, direction, alpha = embed_perturb
        
        def make_embed_hook(position, dir_tensor, a):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    out[0, position] = out[0, position] + a * dir_tensor.to(out.device, out.dtype)
                    return (out,) + output[1:]
                return output
            return hook
        
        # Hook on the first layer's input (which is after embedding)
        embed_hooks.append(layers[0].register_forward_hook(make_embed_hook(pos, direction, alpha)))
    
    # Head zeroing hooks
    head_hooks = []
    if zero_attn_heads:
        # Group by layer
        heads_by_layer = {}
        for (l, h) in zero_attn_heads:
            heads_by_layer.setdefault(l, []).append(h)
        
        for layer_idx, heads in heads_by_layer.items():
            sa = layers[layer_idx].self_attn
            
            def make_head_zero_hook(layer_heads, hd, nh):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        # Zero the specified heads in the attention output
                        for h in layer_heads:
                            start = h * hd
                            end = (h + 1) * hd
                            out[0, :, start:end] = 0
                        return (out,) + output[1:]
                    return output
                return hook
            
            head_hooks.append(sa.register_forward_hook(
                make_head_zero_hook(heads, head_dim, n_heads)))
    
    # Capture hooks
    cap_hooks = []
    for li in capture_layers:
        def make_capture(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[layer_idx] = output[0].detach().float().cpu()
            return hook
        cap_hooks.append(layers[li].register_forward_hook(make_capture(li)))
    
    # Forward pass
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1].float().cpu().numpy()
    
    # Remove all hooks
    for h in embed_hooks + head_hooks + cap_hooks:
        h.remove()
    
    return captured, logits


def run_phase439(model_name, round_num):
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"Phase 439: {model_name} Round {round_num}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    n_heads = getattr(model.config, 'num_attention_heads',
                      getattr(model.config, 'num_heads', 32))
    d_model = info.d_model
    head_dim = d_model // n_heads
    W_U = get_W_U(model, model_name)

    # Get W_E
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        W_E = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    elif hasattr(model, 'get_input_embeddings'):
        W_E = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    else:
        W_E = None

    print(f"  n_layers={n_layers}, n_heads={n_heads}, d_model={d_model}, head_dim={head_dim}")

    # Objects and categories
    objects = {
        "apple": {"category": "fruit", "opposing": "animal", "cat_words": ["fruit"], "opp_words": ["animal"]},
        "knife": {"category": "tool", "opposing": "vehicle", "cat_words": ["tool"], "opp_words": ["vehicle"]},
        "dog":   {"category": "animal", "opposing": "fruit", "cat_words": ["animal"], "opp_words": ["fruit"]},
    }

    # Category directions in embedding space
    cat_directions = {}
    if W_E is not None:
        for obj_name, obj_info in objects.items():
            cat_embs = [W_E[tokenizer.encode(w, add_special_tokens=False)[0]] for w in obj_info["cat_words"]]
            opp_embs = [W_E[tokenizer.encode(w, add_special_tokens=False)[0]] for w in obj_info["opp_words"]]
            d_cat = np.mean(cat_embs, axis=0) - np.mean(opp_embs, axis=0)
            d_cat = d_cat / (np.linalg.norm(d_cat) + 1e-8)
            cat_directions[obj_name] = d_cat
    else:
        print("  WARNING: W_E not found, using W_U")
        for obj_name, obj_info in objects.items():
            cat_embs = [W_U[:, tokenizer.encode(w, add_special_tokens=False)[0]] for w in obj_info["cat_words"]]
            opp_embs = [W_U[:, tokenizer.encode(w, add_special_tokens=False)[0]] for w in obj_info["opp_words"]]
            d_cat = np.mean(cat_embs, axis=1) - np.mean(opp_embs, axis=1)
            d_cat = d_cat / (np.linalg.norm(d_cat) + 1e-8)
            cat_directions[obj_name] = d_cat

    candidates = get_candidate_heads(model_name)
    print(f"  Candidate heads: {candidates[:5]}... (total {len(candidates)})")

    alpha = 1.5

    results = {
        "model": model_name, "round": round_num,
        "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
        "timestamp": datetime.now().isoformat(), "alpha": alpha,
        "per_object": {},
    }

    for obj_name, obj_info in objects.items():
        print(f"\n  --- {obj_name} ({obj_info['category']}) ---")
        prompt = f"An {obj_name} is a kind of"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        obj_pos = None
        for i, t in enumerate(tokens):
            if obj_name.lower() in t.lower() or t.lower().startswith(obj_name[:3]):
                obj_pos = i
                break
        if obj_pos is None:
            obj_pos = 1
        last_pos = len(tokens) - 1
        print(f"  prompt='{prompt}', obj_pos={obj_pos}, last_pos={last_pos}")

        # 1. Baseline run (no perturbation, no ablation)
        cap_base, base_logits = run_model_with_hooks(
            model, input_ids, attention_mask, capture_layers=[n_layers - 1])
        base_cat_gap = compute_cat_logit_gap(base_logits, tokenizer,
                                              obj_info["cat_words"], obj_info["opp_words"])

        # 2. Perturbed run (embed perturbation, no ablation)
        d_cat_t = torch.tensor(cat_directions[obj_name], dtype=torch.bfloat16)
        cap_pert, pert_logits = run_model_with_hooks(
            model, input_ids, attention_mask, capture_layers=[n_layers - 1],
            embed_perturb=(obj_pos, d_cat_t, alpha))
        pert_cat_gap = compute_cat_logit_gap(pert_logits, tokenizer,
                                              obj_info["cat_words"], obj_info["opp_words"])

        # Compute original delta
        if n_layers - 1 in cap_pert and n_layers - 1 in cap_base:
            orig_delta = cap_pert[n_layers - 1][0, last_pos].numpy() - cap_base[n_layers - 1][0, last_pos].numpy()
        else:
            orig_delta = np.zeros(d_model)
        orig_delta_norm = np.linalg.norm(orig_delta)

        print(f"  base_gap={base_cat_gap:.4f}, pert_gap={pert_cat_gap:.4f}, "
              f"orig_delta_norm={orig_delta_norm:.4f}")

        # 3. Multi-head joint ablation
        ablation_results = {}
        rng = np.random.RandomState(42)

        for k in [1, 2, 4, 8, 16]:
            top_k = candidates[:k]
            # Random control: same k, different heads
            all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
            rand_idx = rng.choice(len(all_heads), k, replace=False)
            rand_k = [all_heads[i] for i in rand_idx]

            for head_set_name, head_set in [("top_k", top_k), ("rand_k", rand_k)]:
                key = f"k={k}_{head_set_name}"

                cap_abl, abl_logits = run_model_with_hooks(
                    model, input_ids, attention_mask, capture_layers=[n_layers - 1],
                    zero_attn_heads=head_set,
                    embed_perturb=(obj_pos, d_cat_t, alpha))

                abl_cat_gap = compute_cat_logit_gap(abl_logits, tokenizer,
                                                     obj_info["cat_words"], obj_info["opp_words"])

                # Compute ablated delta
                if n_layers - 1 in cap_abl and n_layers - 1 in cap_base:
                    abl_delta = cap_abl[n_layers - 1][0, last_pos].numpy() - cap_base[n_layers - 1][0, last_pos].numpy()
                else:
                    abl_delta = np.zeros(d_model)
                abl_delta_norm = np.linalg.norm(abl_delta)

                # Three scores
                norm_score = 1.0 - abl_delta_norm / orig_delta_norm if orig_delta_norm > 1e-6 else 0.0
                if abl_delta_norm > 1e-6 and orig_delta_norm > 1e-6:
                    direction_cos = float(np.dot(abl_delta, orig_delta) / (abl_delta_norm * orig_delta_norm))
                else:
                    direction_cos = 0.0
                readout_score = pert_cat_gap - abl_cat_gap

                ablation_results[key] = {
                    "k": k, "type": head_set_name,
                    "heads": [f"L{l}_H{h}" for l, h in head_set],
                    "norm_score": round(norm_score, 4),
                    "direction_cos": round(direction_cos, 4),
                    "readout_score": round(readout_score, 4),
                    "abl_delta_norm": round(float(abl_delta_norm), 4),
                    "abl_cat_gap": round(float(abl_cat_gap), 4),
                }

                print(f"  {key}: norm_sc={norm_score:.4f}, dir_cos={direction_cos:.4f}, "
                      f"readout={readout_score:.4f}")

        # 4. Single attention layer ablation (zero all heads in a layer)
        layer_ablation_results = {}
        sample_layers = [0, 1, 2, 3, n_layers // 4, n_layers // 2,
                         3 * n_layers // 4, n_layers - 2, n_layers - 1]

        for li in sample_layers:
            if li >= n_layers:
                continue
            all_heads_in_layer = [(li, h) for h in range(n_heads)]

            cap_la, la_logits = run_model_with_hooks(
                model, input_ids, attention_mask, capture_layers=[n_layers - 1],
                zero_attn_heads=all_heads_in_layer,
                embed_perturb=(obj_pos, d_cat_t, alpha))

            la_cat_gap = compute_cat_logit_gap(la_logits, tokenizer,
                                                obj_info["cat_words"], obj_info["opp_words"])

            if n_layers - 1 in cap_la and n_layers - 1 in cap_base:
                la_delta = cap_la[n_layers - 1][0, last_pos].numpy() - cap_base[n_layers - 1][0, last_pos].numpy()
            else:
                la_delta = np.zeros(d_model)
            la_delta_norm = np.linalg.norm(la_delta)

            la_norm_score = 1.0 - la_delta_norm / orig_delta_norm if orig_delta_norm > 1e-6 else 0.0
            if la_delta_norm > 1e-6 and orig_delta_norm > 1e-6:
                la_dir_cos = float(np.dot(la_delta, orig_delta) / (la_delta_norm * orig_delta_norm))
            else:
                la_dir_cos = 0.0
            la_readout = pert_cat_gap - la_cat_gap

            layer_ablation_results[f"L{li}"] = {
                "layer": li,
                "norm_score": round(la_norm_score, 4),
                "direction_cos": round(la_dir_cos, 4),
                "readout_score": round(la_readout, 4),
                "la_delta_norm": round(float(la_delta_norm), 4),
                "la_cat_gap": round(float(la_cat_gap), 4),
            }

            print(f"  L{li}_all_attn: norm_sc={la_norm_score:.4f}, dir_cos={la_dir_cos:.4f}, "
                  f"readout={la_readout:.4f}")

        results["per_object"][obj_name] = {
            "category": obj_info["category"],
            "base_cat_gap": round(float(base_cat_gap), 4),
            "pert_cat_gap": round(float(pert_cat_gap), 4),
            "orig_delta_norm": round(float(orig_delta_norm), 4),
            "head_ablation": ablation_results,
            "layer_ablation": layer_ablation_results,
        }

        elapsed = time.time() - t_start
        print(f"  {obj_name} done in {elapsed:.0f}s")

    # Summary: compare top-k vs rand-k
    print(f"\n{'='*40}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*40}")
    for obj_name, obj_data in results["per_object"].items():
        print(f"\n  {obj_name}:")
        for k in [1, 2, 4, 8, 16]:
            top_key = f"k={k}_top_k"
            rand_key = f"k={k}_rand_k"
            top_r = obj_data["head_ablation"].get(top_key, {})
            rand_r = obj_data["head_ablation"].get(rand_key, {})
            print(f"    k={k}: top_norm={top_r.get('norm_score', 'N/A'):.3f} vs "
                  f"rand_norm={rand_r.get('norm_score', 'N/A'):.3f} | "
                  f"top_readout={top_r.get('readout_score', 'N/A'):.3f} vs "
                  f"rand_readout={rand_r.get('readout_score', 'N/A'):.3f}")

    # Save
    out_dir = "d:/Ai2050/TransformerLens-Project/results/phase439_multihead_ablation"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_name}_phase439_r{round_num}.json"
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
    run_phase439(model_name, round_num)
