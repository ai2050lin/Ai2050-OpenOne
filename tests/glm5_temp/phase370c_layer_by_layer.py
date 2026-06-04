"""
Phase 370c: Layer-by-Layer Analysis of DS7B L3-L7 Norm Explosion
=================================================================

Phase 370 discovered that DS7B's 1D collapse starts at L5, where
PC1 norm jumps from 2.87(L4) to 112.1(L5) — a 39x increase.

This test does a fine-grained layer-by-layer analysis of L3-L7 to:
1. Pinpoint the exact transition layer
2. Measure per-pair norm distribution changes
3. Check if the PC1 direction rotates or stays stable across the transition
4. Compute the "norm explosion factor" per pair

Models: deepseek7b (primary), qwen3 (control)
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
    },
}

TEST_PAIRS_42 = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("volcano", "hot", "cold"),
    ("furnace", "hot", "cold"), ("glacier", "cold", "hot"),
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("swamp", "wet", "dry"), ("desert", "dry", "wet"),
    ("silk", "smooth", "rough"), ("sandpaper", "rough", "smooth"),
    ("glass", "smooth", "rough"), ("rock", "rough", "smooth"),
    ("velvet", "soft", "hard"), ("diamond", "hard", "soft"),
]

EXTRA_PAIRS = [
    ("elephant", "big", "small"), ("mountain", "big", "small"), ("ant", "small", "big"),
    ("planet", "big", "small"), ("grain", "small", "big"), ("whale", "big", "small"),
    ("boulder", "heavy", "light"), ("feather", "light", "heavy"), ("lead", "heavy", "light"),
    ("balloon", "light", "heavy"), ("steel", "heavy", "light"), ("cotton", "light", "heavy"),
    ("cheetah", "fast", "slow"), ("turtle", "slow", "fast"), ("rocket", "fast", "slow"),
    ("snail", "slow", "fast"), ("lightning", "fast", "slow"), ("sloth", "slow", "fast"),
    ("star", "bright", "dark"), ("cave", "dark", "bright"), ("sun", "bright", "dark"),
    ("shadow", "dark", "bright"), ("lamp", "bright", "dark"), ("night", "dark", "bright"),
    ("flame", "hot", "cold"), ("ice", "cold", "hot"), ("oven", "hot", "cold"),
    ("frost", "cold", "hot"), ("magma", "hot", "cold"), ("winter", "cold", "hot"),
    ("river", "wet", "dry"), ("desert", "dry", "wet"), ("rain", "wet", "dry"),
    ("dust", "dry", "wet"), ("ocean", "wet", "dry"), ("sand", "dry", "wet"),
    ("silk", "soft", "hard"), ("diamond", "hard", "soft"), ("cotton", "soft", "hard"),
    ("iron", "hard", "soft"), ("pillow", "soft", "hard"), ("concrete", "hard", "soft"),
]

TEST_PAIRS = TEST_PAIRS_42 + EXTRA_PAIRS  # 82 pairs

CORRUPTED_BASELINE = "The item"
TEMPLATE = "The {obj} is {attr}."


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl)
            break
        except:
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        return

    log(f"Phase 370c: Layer-by-Layer Norm Explosion Analysis")
    log(f"Model: {model_name}")

    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    n_pairs = len(TEST_PAIRS)

    # Load model
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"Model loaded in {time.time()-t0:.1f}s")

    # Collect hidden states for ALL layers (0 to n_layers)
    log("Collecting hidden states for ALL layers...")
    h_clean_all = {l: np.zeros((n_pairs, d_model), dtype=np.float32) for l in range(n_layers + 1)}
    h_corrupt_all = {l: np.zeros((n_pairs, d_model), dtype=np.float32) for l in range(n_layers + 1)}

    input_device = next(model.parameters()).device

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        if pidx % 10 == 0:
            log(f"  Pair {pidx+1}/{n_pairs}: {obj}-{target}/{competitor}")

        clean_prompt = TEMPLATE.format(obj=obj, attr=target)
        corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)

        clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
        corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)

        with torch.no_grad():
            clean_out = model(input_ids=clean_inputs["input_ids"].to(input_device),
                              attention_mask=clean_inputs["attention_mask"].to(input_device),
                              output_hidden_states=True)
            corrupt_out = model(input_ids=corrupt_inputs["input_ids"].to(input_device),
                                attention_mask=corrupt_inputs["attention_mask"].to(input_device),
                                output_hidden_states=True)

        last_pos_c = clean_inputs["input_ids"].shape[1] - 1
        last_pos_r = corrupt_inputs["input_ids"].shape[1] - 1

        for l in range(n_layers + 1):
            h_clean_all[l][pidx] = clean_out.hidden_states[l][0, last_pos_c].detach().cpu().float().numpy()
            h_corrupt_all[l][pidx] = corrupt_out.hidden_states[l][0, last_pos_r].detach().cpu().float().numpy()

        del clean_out, corrupt_out
        if pidx % 5 == 0:
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("Model released")

    # Per-layer analysis
    log("\n" + "="*70)
    log("Layer-by-Layer Analysis: Norm, PCA, PC1 Stability, Direction Rotation")
    log("="*70)

    results = {}
    prev_pc1_dir = None

    for l in range(n_layers + 1):
        M_clean = h_clean_all[l]
        M_corrupt = h_corrupt_all[l]
        M_dh = M_clean - M_corrupt

        # Per-pair norms
        norms = np.array([np.linalg.norm(M_dh[i]) for i in range(n_pairs)])

        # PCA
        M_centered = M_dh - M_dh.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        except:
            continue
        total_var = np.sum(S**2)
        if total_var < 1e-10:
            results[str(l)] = {"mean_norm": float(np.mean(norms)), "skip": True}
            prev_pc1_dir = None
            continue
        explained = (S**2) / total_var
        eff_rank = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)

        pc1_dir = Vt[0]

        # PC1 direction rotation (cos with previous layer's PC1)
        cos_prev_pc1 = float(np.abs(np.dot(pc1_dir, prev_pc1_dir))) if prev_pc1_dir is not None else None
        prev_pc1_dir = pc1_dir.copy()

        # Per-pair PC1 projection
        pc1_projections = M_dh @ pc1_dir  # (n_pairs,)
        pc1_norms = np.abs(pc1_projections)

        # Residual norms
        residual_norms = np.array([
            np.linalg.norm(M_dh[i] - np.dot(M_dh[i], pc1_dir) * pc1_dir)
            for i in range(n_pairs)
        ])

        # Norm explosion factor: ratio of this layer's mean norm to previous
        # (will compute below)

        # PC1 alignment per pair
        cos_with_pc1 = np.array([
            np.abs(np.dot(M_dh[i], pc1_dir)) / (np.linalg.norm(M_dh[i]) + 1e-10)
            for i in range(n_pairs)
        ])

        results[str(l)] = {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "median_norm": float(np.median(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
            "pc1_explained": float(explained[0]),
            "pc2_explained": float(explained[1]) if len(explained) > 1 else 0,
            "eff_rank_95": eff_rank,
            "mean_pc1_norm": float(np.mean(pc1_norms)),
            "mean_residual_norm": float(np.mean(residual_norms)),
            "pc1_over_residual": float(np.mean(pc1_norms) / (np.mean(residual_norms) + 1e-10)),
            "mean_cos_pc1": float(np.mean(cos_with_pc1)),
            "std_cos_pc1": float(np.std(cos_with_pc1)),
            "cos_with_prev_pc1": cos_prev_pc1,
        }

        if l >= 1 and str(l-1) in results and "skip" not in results[str(l-1)]:
            prev_norm = results[str(l-1)]["mean_norm"]
            norm_explosion = np.mean(norms) / (prev_norm + 1e-10)
            results[str(l)]["norm_explosion_factor"] = float(norm_explosion)
        else:
            results[str(l)]["norm_explosion_factor"] = 1.0

        # Print key info
        r = results[str(l)]
        cos_str = f", cos_prev={cos_prev_pc1:.3f}" if cos_prev_pc1 is not None else ""
        log(f"  L{l:>2}: norm={r['mean_norm']:>8.2f}±{r['std_norm']:.2f}, "
            f"PC1={r['pc1_explained']:.3f}({eff_rank}D), "
            f"PC1/res={r['pc1_over_residual']:.2f}, "
            f"cos(PC1,Δh)={r['mean_cos_pc1']:.3f}, "
            f"explode={r['norm_explosion_factor']:.2f}{cos_str}")

    # Save
    output = {
        "model": model_name,
        "phase": "370c",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": n_pairs,
        "n_layers": n_layers,
        "d_model": d_model,
        "per_layer": results,
    }

    os.makedirs("results/phase370_norm_mask", exist_ok=True)
    out_path = f"results/phase370_norm_mask/{model_name}_phase370c.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    # Summary: find transition layer
    log("\n" + "="*70)
    log("TRANSITION ANALYSIS")
    log("="*70)

    # Find the layer with maximum norm explosion factor
    max_explode_layer = max(
        [(int(l), results[l]["norm_explosion_factor"]) for l in results if "norm_explosion_factor" in results[l]],
        key=lambda x: x[1]
    )
    log(f"Max norm explosion: L{max_explode_layer[0]} = {max_explode_layer[1]:.2f}x")

    # Find first layer where PC1 > 0.9
    first_1d = None
    for l in sorted(results.keys(), key=int):
        if "skip" in results[l]:
            continue
        if results[l].get("pc1_explained", 0) > 0.9:
            first_1d = int(l)
            break
    log(f"First layer with PC1 > 0.9: L{first_1d}")

    # Find first layer where PC1/residual > 1.0
    first_norm_dom = None
    for l in sorted(results.keys(), key=int):
        if "skip" in results[l]:
            continue
        if results[l].get("pc1_over_residual", 0) > 1.0:
            first_norm_dom = int(l)
            break
    log(f"First layer with PC1/residual > 1.0: L{first_norm_dom}")

    log(f"\nPhase 370c complete for {model_name}!")


if __name__ == "__main__":
    main()
