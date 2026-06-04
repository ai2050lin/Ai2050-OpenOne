"""
Phase 368b: Confirmation test — expanded sample size for PCA + increment analysis
=================================================================================

Phase 368 found:
1. DS7B L6-L21 has near-perfect 1D collapse (PC1=0.96-0.98) — MUST confirm with more samples
2. Qwen3/GLM4 are high-dimensional subspace (PC1=0.14-0.28) — is this robust?
3. Qwen3/GLM4 have orthogonal_rewrite increments — not same_direction accumulation
4. DS7B has same_direction layers (L7/L9/L13) + late reverse_calibration

Confirmation: expand from 42 to 80+ pairs by adding new attribute categories:
- size (big/small): 10 pairs
- weight (heavy/light): 10 pairs  
- speed (fast/slow): 10 pairs
- brightness (bright/dark): 10 pairs

This tests: does DS7B's 1D collapse hold with more diverse binding types?
And: is Qwen3/GLM4's high-dimensional structure robust to more categories?
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Model Configs =====
MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
    },
}

# Original 42 pairs + 40 new pairs = 82 total
ORIGINAL_PAIRS = [
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

NEW_PAIRS = [
    # size (big/small) - 10 pairs
    ("elephant", "big", "small"), ("mountain", "big", "small"), ("planet", "big", "small"),
    ("whale", "big", "small"), ("continent", "big", "small"),
    ("ant", "small", "big"), ("grain", "small", "big"), ("needle", "small", "big"),
    ("molecule", "small", "big"), ("droplet", "small", "big"),
    # weight (heavy/light) - 10 pairs
    ("lead", "heavy", "light"), ("boulder", "heavy", "light"), ("tank", "heavy", "light"),
    ("anchor", "heavy", "light"), ("iron", "heavy", "light"),
    ("feather", "light", "heavy"), ("bubble", "light", "heavy"), ("cloud", "light", "heavy"),
    ("snowflake", "light", "heavy"), ("paper", "light", "heavy"),
    # speed (fast/slow) - 10 pairs
    ("cheetah", "fast", "slow"), ("rocket", "fast", "slow"), ("lightning", "fast", "slow"),
    ("bullet", "fast", "slow"), ("falcon", "fast", "slow"),
    ("snail", "slow", "fast"), ("turtle", "slow", "fast"), ("sloth", "slow", "fast"),
    ("glacier_speed", "slow", "fast"), ("caterpillar", "slow", "fast"),
    # brightness (bright/dark) - 10 pairs
    ("star", "bright", "dark"), ("lamp", "bright", "dark"), ("sun", "bright", "dark"),
    ("flashlight", "bright", "dark"), ("candle", "bright", "dark"),
    ("shadow", "dark", "bright"), ("cave", "dark", "bright"), ("midnight", "dark", "bright"),
    ("tunnel", "dark", "bright"), ("abyss", "dark", "bright"),
]

ALL_PAIRS = ORIGINAL_PAIRS + NEW_PAIRS

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
            log(f"  Loaded {model_name} with attn_impl={impl}")
            break
        except Exception as e:
            log(f"  Failed with {impl}: {str(e)[:80]}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


def find_object_positions(tokenizer, prompt, obj_word):
    input_ids = tokenizer.encode(prompt)
    positions = []
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid]).strip().lower()
        if obj_word.lower() in decoded and decoded != '':
            positions.append(i)
    if not positions:
        positions = [1] if len(input_ids) > 1 else [0]
    return positions


def collect_dh_vectors(model, tokenizer, model_name, pairs):
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    n_pairs = len(pairs)

    dh_per_pair = {}

    for pidx, (obj, target, competitor) in enumerate(pairs):
        if pidx % 10 == 0:
            log(f"  Pair {pidx+1}/{n_pairs}: {obj}-{target}/{competitor}")

        clean_prompt = TEMPLATE.format(obj=obj, attr=target)
        corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)

        input_device = next(model.parameters()).device
        clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
        corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)

        with torch.no_grad():
            clean_out = model(input_ids=clean_inputs["input_ids"].to(input_device),
                              attention_mask=clean_inputs["attention_mask"].to(input_device),
                              output_hidden_states=True)
            corrupt_out = model(input_ids=corrupt_inputs["input_ids"].to(input_device),
                                attention_mask=corrupt_inputs["attention_mask"].to(input_device),
                                output_hidden_states=True)

        last_pos_clean = clean_inputs["input_ids"].shape[1] - 1
        last_pos_corrupt = corrupt_inputs["input_ids"].shape[1] - 1

        for l in range(n_layers + 1):
            h_clean = clean_out.hidden_states[l][0, last_pos_clean].detach().cpu().float().numpy()
            h_corrupt = corrupt_out.hidden_states[l][0, last_pos_corrupt].detach().cpu().float().numpy()
            dh = h_clean - h_corrupt
            if l not in dh_per_pair:
                dh_per_pair[l] = np.zeros((n_pairs, d_model), dtype=np.float32)
            dh_per_pair[l][pidx] = dh

        del clean_out, corrupt_out
        torch.cuda.empty_cache()

    return dh_per_pair


def compute_pca_at_key_layers(dh_per_pair, n_layers, n_pairs):
    """PCA at key layers only (for speed)."""
    key_layers = [0, n_layers // 6, n_layers // 3, n_layers // 2, 
                  2 * n_layers // 3, 5 * n_layers // 6, n_layers]
    key_layers = sorted(set([0] + key_layers + [n_layers]))

    results = {}
    for l in key_layers:
        if l not in dh_per_pair:
            continue
        M = dh_per_pair[l]
        M_centered = M - M.mean(axis=0, keepdims=True)

        try:
            U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        except:
            continue

        total_var = np.sum(S ** 2)
        if total_var < 1e-10:
            continue

        explained_ratio = (S ** 2) / total_var
        cumulative = np.cumsum(explained_ratio)
        effective_rank_95 = int(np.searchsorted(cumulative, 0.95) + 1)

        # Per-pair cos with PC1
        pc1 = Vt[0]
        cos_sims = []
        for i in range(n_pairs):
            norm_dh = np.linalg.norm(M[i])
            norm_pc1 = np.linalg.norm(pc1)
            if norm_dh > 1e-8 and norm_pc1 > 1e-8:
                cos_sims.append(float(np.dot(M[i], pc1) / (norm_dh * norm_pc1)))

        results[str(l)] = {
            "pc1_ratio": float(explained_ratio[0]),
            "pc2_ratio": float(explained_ratio[1]) if len(explained_ratio) > 1 else 0,
            "pc3_ratio": float(explained_ratio[2]) if len(explained_ratio) > 2 else 0,
            "eff_rank_95": effective_rank_95,
            "cos_pc1_mean": float(np.mean(cos_sims)) if cos_sims else 0,
            "cos_pc1_std": float(np.std(cos_sims)) if cos_sims else 0,
        }

        log(f"  L{l}: PC1={explained_ratio[0]:.3f}, PC2={explained_ratio[1]:.3f}, "
            f"eff_rank_95={effective_rank_95}, cos(PC1)={np.mean(cos_sims):.3f}")

    return results


def compute_increment_at_key_layers(dh_per_pair, n_layers):
    """Increment decomposition at key layers."""
    key_layers = list(range(1, n_layers + 1, max(1, n_layers // 8)))
    if n_layers not in key_layers:
        key_layers.append(n_layers)

    results = {}
    for l in key_layers:
        if l not in dh_per_pair or (l - 1) not in dh_per_pair:
            continue

        dh_l = dh_per_pair[l]
        dh_prev = dh_per_pair[l - 1]
        delta_l = dh_l - dh_prev

        cos_delta_dh = []
        for i in range(dh_l.shape[0]):
            nd = np.linalg.norm(delta_l[i])
            ndp = np.linalg.norm(dh_prev[i])
            if nd > 1e-8 and ndp > 1e-8:
                cos_delta_dh.append(float(np.dot(delta_l[i], dh_prev[i]) / (nd * ndp)))

        mean_cos = float(np.mean(cos_delta_dh)) if cos_delta_dh else 0
        pos_rate = float(np.mean(np.array(cos_delta_dh) > 0.3)) if cos_delta_dh else 0
        neg_rate = float(np.mean(np.array(cos_delta_dh) < -0.3)) if cos_delta_dh else 0

        inc_type = "same_direction" if mean_cos > 0.3 else ("reverse_calibration" if mean_cos < -0.3 else "orthogonal_rewrite")

        results[str(l)] = {
            "type": inc_type,
            "cos_delta_prev_mean": mean_cos,
            "same_rate": float(pos_rate),
            "reverse_rate": float(neg_rate),
        }

        log(f"  L{l}: {inc_type}, cos(δ,Δh_prev)={mean_cos:.3f}")

    return results


def compare_original_vs_expanded(dh_per_pair, n_layers, n_original=42):
    """Compare PCA structure between original 42 and all 82 pairs."""
    log("\n--- Original vs Expanded PCA comparison ---")
    key_layers = [0, n_layers // 2, n_layers]

    comparison = {}
    for l in key_layers:
        if l not in dh_per_pair:
            continue

        # Original pairs
        M_orig = dh_per_pair[l][:n_original]
        M_orig_c = M_orig - M_orig.mean(axis=0, keepdims=True)
        try:
            _, S_orig, _ = np.linalg.svd(M_orig_c, full_matrices=False)
        except:
            continue
        var_orig = np.sum(S_orig ** 2)
        if var_orig < 1e-10:
            continue
        pc1_orig = float((S_orig[0] ** 2) / var_orig)

        # All pairs
        M_all = dh_per_pair[l]
        M_all_c = M_all - M_all.mean(axis=0, keepdims=True)
        try:
            _, S_all, _ = np.linalg.svd(M_all_c, full_matrices=False)
        except:
            continue
        var_all = np.sum(S_all ** 2)
        if var_all < 1e-10:
            continue
        pc1_all = float((S_all[0] ** 2) / var_all)
        cumulative = np.cumsum((S_all ** 2) / var_all)
        eff_rank_95 = int(np.searchsorted(cumulative, 0.95) + 1)

        comparison[str(l)] = {
            "pc1_original_42": pc1_orig,
            "pc1_expanded_82": pc1_all,
            "eff_rank_95_expanded": eff_rank_95,
        }

        log(f"  L{l}: PC1(42pairs)={pc1_orig:.3f} → PC1(82pairs)={pc1_all:.3f}, "
            f"eff_rank_95={eff_rank_95}")

    return comparison


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log(f"Phase 368b: Confirmation test — {model_name} (82 pairs)")

    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    log(f"Model loaded in {t_load:.0f}s")

    # Collect Δh with expanded pairs
    dh_per_pair = collect_dh_vectors(model, tokenizer, model_name, ALL_PAIRS)
    t_collect = time.time() - t0 - t_load
    log(f"Δh collection done in {t_collect:.0f}s (82 pairs × {n_layers+1} layers)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # PCA at key layers
    log("\n=== PCA (82 pairs) ===")
    pca_results = compute_pca_at_key_layers(dh_per_pair, n_layers, len(ALL_PAIRS))

    # Increment decomposition
    log("\n=== Increment (82 pairs) ===")
    inc_results = compute_increment_at_key_layers(dh_per_pair, n_layers)

    # Original vs expanded comparison
    comparison = compare_original_vs_expanded(dh_per_pair, n_layers)

    # Save
    output = {
        "model": model_name,
        "phase": "368b",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": len(ALL_PAIRS),
        "n_original": len(ORIGINAL_PAIRS),
        "n_new": len(NEW_PAIRS),
        "n_layers": n_layers,
        "pca_results": pca_results,
        "increment_results": inc_results,
        "original_vs_expanded": comparison,
        "total_time_s": round(time.time() - t0, 1),
    }

    os.makedirs("results/phase368_dh_subspace", exist_ok=True)
    out_path = f"results/phase368_dh_subspace/{model_name}_phase368b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"Saved to {out_path}")

    log(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
