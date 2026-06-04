"""
Phase 368: Δh Subspace Analysis — PCA + Increment Decomposition + Embedding Source
==================================================================================

Three critical questions from Phase 365+367:
1. Is binding encoded in a single 1D direction or a low-dimensional subspace?
   (Average Δh cos_sim=0.89 is stable, but individual Δh vectors may spread)
2. What is each layer's increment δ_l = Δh_l - Δh_{l-1} doing?
   (Same-direction amplification? Orthogonal rewrite? Reverse calibration?)
3. Where does the L0 high Post-LN signal come from?
   (Word embedding difference? Or LayerNorm amplification artifact?)

Experiments:
  Part A: Per-pair Δh collection at all layers → PCA → effective rank
  Part B: δ_l increment decomposition → direction relative to Δh and W_U
  Part C: Embedding Δe analysis → compare with final Δh direction

Test pairs: 42 (same as Phase 364/365 for consistency)
Estimated runtime:
  Qwen3: ~4 min | GLM4: ~40 min | DS7B: ~16 min
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

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

# Full test pairs (42) - same as Phase 364/365
TEST_PAIRS = [
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

CORRUPTED_BASELINE = "The item"
TEMPLATE = "The {obj} is {attr}."


# ===== Model Loading =====

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

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        log(f"  GPU={gpu_count} components, CPU={cpu_count}, GPU mem={gpu_mem:.2f}GB")

    return model, tokenizer, next(model.parameters()).device


def get_W_U(model, model_name):
    if hasattr(model, "lm_head"):
        w = model.lm_head.weight
        if not w.is_meta:
            return w.detach().cpu().float().numpy()
    import glob
    from safetensors import safe_open
    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            if 'lm_head.weight' in sf.keys():
                return sf.get_tensor('lm_head.weight').float().numpy()
    raise ValueError(f"Cannot load lm_head for {model_name}")


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers")


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


# ===== Part A: Per-pair Δh collection + PCA =====

def collect_dh_vectors(model, tokenizer, device, model_name):
    """Collect per-pair Δh at last_token position for all layers."""
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    n_pairs = len(TEST_PAIRS)

    # Storage: per-pair Δh at each layer
    # dh_per_pair[layer_idx][pair_idx] = Δh vector (d_model,)
    dh_per_pair = {}
    pair_labels = []

    # Also collect embedding differences (Part C)
    embedding_diffs = []

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        if pidx % 5 == 0:
            log(f"  Pair {pidx+1}/{n_pairs}: {obj}-{target}/{competitor}")

        # Build prompts
        clean_prompt = TEMPLATE.format(obj=obj, attr=target)
        corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)

        # Find object position
        obj_positions = find_object_positions(tokenizer, clean_prompt, obj)

        # Run clean and corrupt with hidden states
        clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
        corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)

        input_device = next(model.parameters()).device
        clean_ids = clean_inputs["input_ids"].to(input_device)
        clean_mask = clean_inputs["attention_mask"].to(input_device)
        corrupt_ids = corrupt_inputs["input_ids"].to(input_device)
        corrupt_mask = corrupt_inputs["attention_mask"].to(input_device)

        with torch.no_grad():
            clean_out = model(input_ids=clean_ids, attention_mask=clean_mask,
                              output_hidden_states=True)
            corrupt_out = model(input_ids=corrupt_ids, attention_mask=corrupt_mask,
                                output_hidden_states=True)

        # Get last token position
        last_pos = clean_ids.shape[1] - 1

        # Extract Δh at last_token position for each layer
        for l in range(n_layers + 1):  # +1 for embedding layer
            h_clean = clean_out.hidden_states[l][0, last_pos].detach().cpu().float().numpy()
            h_corrupt = corrupt_out.hidden_states[l][0, last_pos].detach().cpu().float().numpy()
            dh = h_clean - h_corrupt

            if l not in dh_per_pair:
                dh_per_pair[l] = np.zeros((n_pairs, d_model), dtype=np.float32)
            dh_per_pair[l][pidx] = dh

        # Part C: Embedding difference for the object token
        # Use L0 hidden state at object position as "embedding"
        obj_pos = obj_positions[0] if obj_positions else 1
        # Clean object position at L0
        emb_clean = clean_out.hidden_states[0][0, obj_pos].detach().cpu().float().numpy()
        # Corrupt object position at L0
        last_pos_c = corrupt_ids.shape[1] - 1
        emb_corrupt = corrupt_out.hidden_states[0][0, min(obj_pos, last_pos_c)].detach().cpu().float().numpy()
        embedding_diffs.append(emb_clean - emb_corrupt)

        pair_labels.append(f"{obj}-{target}/{competitor}")

        del clean_out, corrupt_out
        torch.cuda.empty_cache()

    return dh_per_pair, embedding_diffs, pair_labels


def compute_pca_analysis(dh_per_pair, n_layers, n_pairs):
    """Part A: PCA on Δh matrix at each layer."""
    log("\n=== Part A: PCA Analysis ===")

    pca_results = {}
    key_layers = list(range(0, n_layers + 1, max(1, n_layers // 8)))
    if n_layers not in key_layers:
        key_layers.append(n_layers)
    key_layers = sorted(set(key_layers))

    for l in key_layers:
        if l not in dh_per_pair:
            continue

        M = dh_per_pair[l]  # (n_pairs, d_model)

        # Center the data
        M_centered = M - M.mean(axis=0, keepdims=True)

        # SVD
        try:
            U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        except Exception as e:
            log(f"  L{l}: SVD failed: {e}")
            continue

        # Explained variance
        total_var = np.sum(S ** 2)
        if total_var < 1e-10:
            log(f"  L{l}: Near-zero variance, skipping")
            continue

        explained_ratio = (S ** 2) / total_var
        cumulative = np.cumsum(explained_ratio)

        # Effective rank (number of components for 95% variance)
        effective_rank_95 = int(np.searchsorted(cumulative, 0.95) + 1)
        effective_rank_99 = int(np.searchsorted(cumulative, 0.99) + 1)

        # Cosine similarity of individual Δh with first principal direction
        pc1 = Vt[0]  # First principal component direction
        cos_sims_pc1 = []
        for i in range(n_pairs):
            norm_dh = np.linalg.norm(M[i])
            norm_pc1 = np.linalg.norm(pc1)
            if norm_dh > 1e-8 and norm_pc1 > 1e-8:
                cos_sims_pc1.append(float(np.dot(M[i], pc1) / (norm_dh * norm_pc1)))

        result = {
            "explained_ratio_top10": [float(x) for x in explained_ratio[:10]],
            "cumulative_top10": [float(x) for x in cumulative[:10]],
            "effective_rank_95": effective_rank_95,
            "effective_rank_99": effective_rank_99,
            "singular_values_top10": [float(x) for x in S[:10]],
            "cos_sim_with_pc1_mean": float(np.mean(cos_sims_pc1)) if cos_sims_pc1 else 0,
            "cos_sim_with_pc1_std": float(np.std(cos_sims_pc1)) if cos_sims_pc1 else 0,
            "cos_sim_with_pc1_min": float(np.min(cos_sims_pc1)) if cos_sims_pc1 else 0,
            "total_variance": float(total_var),
        }

        pca_results[str(l)] = result

        # Log summary
        top5 = explained_ratio[:5]
        log(f"  L{l}: PC1={top5[0]:.3f}, PC2={top5[1]:.3f}, PC3={top5[2]:.3f}, "
            f"PC4={top5[3]:.3f}, PC5={top5[4]:.3f} | "
            f"eff_rank_95={effective_rank_95}, eff_rank_99={effective_rank_99} | "
            f"cos(PC1)={np.mean(cos_sims_pc1):.3f}±{np.std(cos_sims_pc1):.3f}")

    return pca_results


# ===== Part B: δ_l increment decomposition =====

def compute_increment_decomposition(dh_per_pair, W_U, n_layers, n_pairs):
    """Part B: δ_l = Δh_l - Δh_{l-1}, analyze direction relative to Δh and W_U."""
    log("\n=== Part B: Increment Decomposition ===")

    # W_U direction (target - competitor for each pair would be ideal,
    # but use the mean Δh direction as reference)
    # Also compute W_U projection direction for general analysis

    results = {}
    key_layers = list(range(1, n_layers + 1, max(1, n_layers // 10)))
    if n_layers not in key_layers:
        key_layers.append(n_layers)
    key_layers = sorted(set(key_layers))

    for l in key_layers:
        if l not in dh_per_pair or (l - 1) not in dh_per_pair:
            continue

        dh_l = dh_per_pair[l]      # (n_pairs, d_model)
        dh_prev = dh_per_pair[l-1]  # (n_pairs, d_model)
        delta_l = dh_l - dh_prev     # (n_pairs, d_model) — per-pair increment

        # Mean Δh direction at this layer (reference direction)
        mean_dh = dh_l.mean(axis=0)
        mean_dh_norm = np.linalg.norm(mean_dh)
        if mean_dh_norm < 1e-10:
            continue
        mean_dh_dir = mean_dh / mean_dh_norm

        # Mean increment direction
        mean_delta = delta_l.mean(axis=0)
        mean_delta_norm = np.linalg.norm(mean_delta)
        if mean_delta_norm < 1e-10:
            log(f"  L{l}: Near-zero increment")
            continue
        mean_delta_dir = mean_delta / mean_delta_norm

        # Per-pair analysis
        cos_delta_dh = []      # cos(δ_l, Δh_{l-1}) — is increment in same direction?
        cos_delta_mean = []    # cos(δ_l, mean_dh) — is increment in the common direction?
        norm_delta = []        # ||δ_l||
        norm_dh = []           # ||Δh_l||

        for i in range(n_pairs):
            d = delta_l[i]
            dh_i = dh_l[i]
            dh_prev_i = dh_prev[i]

            nd = np.linalg.norm(d)
            ndh = np.linalg.norm(dh_i)
            ndh_prev = np.linalg.norm(dh_prev_i)

            norm_delta.append(nd)
            norm_dh.append(ndh)

            if nd > 1e-8 and ndh_prev > 1e-8:
                cos_delta_dh.append(float(np.dot(d, dh_prev_i) / (nd * ndh_prev)))
            else:
                cos_delta_dh.append(0.0)

            if nd > 1e-8 and mean_dh_norm > 1e-8:
                cos_delta_mean.append(float(np.dot(d, mean_dh_dir) / nd))
            else:
                cos_delta_mean.append(0.0)

        cos_delta_dh = np.array(cos_delta_dh)
        cos_delta_mean = np.array(cos_delta_mean)
        norm_delta = np.array(norm_delta)
        norm_dh = np.array(norm_dh)

        # Classify increment direction
        mean_cos = float(np.mean(cos_delta_dh))
        pos_rate = float(np.mean(cos_delta_dh > 0.3))
        neg_rate = float(np.mean(cos_delta_dh < -0.3))
        ortho_rate = float(np.mean(np.abs(cos_delta_dh) <= 0.3))

        if mean_cos > 0.3:
            inc_type = "same_direction"
        elif mean_cos < -0.3:
            inc_type = "reverse_calibration"
        else:
            inc_type = "orthogonal_rewrite"

        result = {
            "increment_type": inc_type,
            "cos_delta_prev_dh_mean": float(np.mean(cos_delta_dh)),
            "cos_delta_prev_dh_std": float(np.std(cos_delta_dh)),
            "cos_delta_mean_dh_dir": float(np.mean(cos_delta_mean)),
            "cos_delta_mean_dh_dir_std": float(np.std(cos_delta_mean)),
            "norm_delta_mean": float(np.mean(norm_delta)),
            "norm_delta_std": float(np.std(norm_delta)),
            "norm_dh_mean": float(np.mean(norm_dh)),
            "same_direction_rate": float(pos_rate),
            "reverse_rate": float(neg_rate),
            "orthogonal_rate": float(ortho_rate),
        }
        results[str(l)] = result

        log(f"  L{l}: type={inc_type}, cos(δ,Δh_prev)={np.mean(cos_delta_dh):.3f}±{np.std(cos_delta_dh):.3f}, "
            f"cos(δ,mean_dir)={np.mean(cos_delta_mean):.3f}, "
            f"||δ||={np.mean(norm_delta):.1f}, "
            f"same={pos_rate:.2f} rev={neg_rate:.2f} orth={ortho_rate:.2f}")

    return results


# ===== Part C: Embedding source analysis =====

def compute_embedding_analysis(embedding_diffs, dh_per_pair, W_U, n_layers, n_pairs):
    """Part C: Compare L0 embedding Δe direction with final Δh direction."""
    log("\n=== Part C: Embedding Source Analysis ===")

    # L0 Δh (embedding layer difference at last_token)
    if 0 not in dh_per_pair:
        log("  No L0 data, skipping")
        return {}

    dh_l0 = dh_per_pair[0]   # (n_pairs, d_model) at last_token
    dh_final = dh_per_pair[n_layers]  # final layer

    # Also the object-position embedding differences
    emb_diffs = np.array(embedding_diffs)  # (n_pairs, d_model)

    # Mean directions
    mean_dh_l0 = dh_l0.mean(axis=0)
    mean_dh_final = dh_final.mean(axis=0)
    mean_emb_diff = emb_diffs.mean(axis=0)

    norm_l0 = np.linalg.norm(mean_dh_l0)
    norm_final = np.linalg.norm(mean_dh_final)
    norm_emb = np.linalg.norm(mean_emb_diff)

    # Cosine similarities between directions
    cos_l0_final = float(np.dot(mean_dh_l0, mean_dh_final) / (norm_l0 * norm_final)) if norm_l0 > 1e-8 and norm_final > 1e-8 else 0
    cos_emb_l0 = float(np.dot(mean_emb_diff, mean_dh_l0) / (norm_emb * norm_l0)) if norm_emb > 1e-8 and norm_l0 > 1e-8 else 0
    cos_emb_final = float(np.dot(mean_emb_diff, mean_dh_final) / (norm_emb * norm_final)) if norm_emb > 1e-8 and norm_final > 1e-8 else 0

    # Per-pair: cos(emb_diff_i, dh_final_i)
    per_pair_cos = []
    for i in range(n_pairs):
        n_e = np.linalg.norm(emb_diffs[i])
        n_f = np.linalg.norm(dh_final[i])
        if n_e > 1e-8 and n_f > 1e-8:
            per_pair_cos.append(float(np.dot(emb_diffs[i], dh_final[i]) / (n_e * n_f)))

    # Per-pair: cos(dh_l0_i, dh_final_i)
    per_pair_cos_l0_final = []
    for i in range(n_pairs):
        n_0 = np.linalg.norm(dh_l0[i])
        n_f = np.linalg.norm(dh_final[i])
        if n_0 > 1e-8 and n_f > 1e-8:
            per_pair_cos_l0_final.append(float(np.dot(dh_l0[i], dh_final[i]) / (n_0 * n_f)))

    results = {
        "cos_mean_l0_vs_final": cos_l0_final,
        "cos_mean_emb_vs_l0": cos_emb_l0,
        "cos_mean_emb_vs_final": cos_emb_final,
        "per_pair_cos_emb_final_mean": float(np.mean(per_pair_cos)) if per_pair_cos else 0,
        "per_pair_cos_emb_final_std": float(np.std(per_pair_cos)) if per_pair_cos else 0,
        "per_pair_cos_l0_final_mean": float(np.mean(per_pair_cos_l0_final)) if per_pair_cos_l0_final else 0,
        "per_pair_cos_l0_final_std": float(np.std(per_pair_cos_l0_final)) if per_pair_cos_l0_final else 0,
        "norm_l0_mean": float(np.mean([np.linalg.norm(dh_l0[i]) for i in range(n_pairs)])),
        "norm_final_mean": float(np.mean([np.linalg.norm(dh_final[i]) for i in range(n_pairs)])),
        "norm_emb_mean": float(np.mean([np.linalg.norm(emb_diffs[i]) for i in range(n_pairs)])),
    }

    log(f"  cos(mean_L0, mean_Final) = {cos_l0_final:.4f}")
    log(f"  cos(mean_Emb, mean_L0)   = {cos_emb_l0:.4f}")
    log(f"  cos(mean_Emb, mean_Final)= {cos_emb_final:.4f}")
    log(f"  per-pair cos(emb, final) = {np.mean(per_pair_cos):.4f} ± {np.std(per_pair_cos):.4f}")
    log(f"  per-pair cos(L0, final)  = {np.mean(per_pair_cos_l0_final):.4f} ± {np.std(per_pair_cos_l0_final):.4f}")
    log(f"  ||Δh_L0|| mean = {results['norm_l0_mean']:.2f}")
    log(f"  ||Δh_Final|| mean = {results['norm_final_mean']:.2f}")
    log(f"  ||Δe_emb|| mean = {results['norm_emb_mean']:.2f}")

    return results


# ===== Cross-layer PCA evolution =====

def compute_cross_layer_pca_evolution(dh_per_pair, n_layers, n_pairs):
    """Track how PCA structure evolves across layers."""
    log("\n=== Cross-layer PCA Evolution ===")

    evolution = {}
    for l in range(0, n_layers + 1):
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

        evolution[str(l)] = {
            "pc1_ratio": float(explained_ratio[0]),
            "pc2_ratio": float(explained_ratio[1]) if len(explained_ratio) > 1 else 0,
            "pc3_ratio": float(explained_ratio[2]) if len(explained_ratio) > 2 else 0,
            "effective_rank_95": effective_rank_95,
            "total_variance": float(total_var),
        }

    # Log at key points
    for l in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]:
        ls = str(l)
        if ls in evolution:
            e = evolution[ls]
            log(f"  L{l}: PC1={e['pc1_ratio']:.3f}, eff_rank_95={e['effective_rank_95']}, var={e['total_variance']:.1f}")

    return evolution


# ===== Per-attribute-category PCA =====

def compute_per_category_pca(dh_per_pair, n_layers):
    """PCA within each attribute category (color, temperature, wetness, texture)."""
    log("\n=== Per-category PCA ===")

    # Define categories
    categories = {
        "color": list(range(0, 20)),       # pairs 0-19
        "temperature": list(range(20, 28)), # pairs 20-27
        "wetness": list(range(28, 36)),     # pairs 28-35
        "texture": list(range(36, 42)),     # pairs 36-41
    }

    results = {}
    key_layers = [0, n_layers // 2, n_layers]

    for cat_name, indices in categories.items():
        cat_results = {}
        for l in key_layers:
            ls = str(l)
            if l not in dh_per_pair:
                continue

            M = dh_per_pair[l][indices]  # (n_cat, d_model)
            if M.shape[0] < 3:
                continue

            M_centered = M - M.mean(axis=0, keepdims=True)
            try:
                U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
            except:
                continue

            total_var = np.sum(S ** 2)
            if total_var < 1e-10:
                continue

            explained_ratio = (S ** 2) / total_var
            cat_results[ls] = {
                "pc1_ratio": float(explained_ratio[0]),
                "pc2_ratio": float(explained_ratio[1]) if len(explained_ratio) > 1 else 0,
                "n_samples": len(indices),
            }

            log(f"  {cat_name} L{l}: PC1={explained_ratio[0]:.3f}, PC2={explained_ratio[1]:.3f}" if len(explained_ratio) > 1
                else f"  {cat_name} L{l}: PC1={explained_ratio[0]:.3f}")

        results[cat_name] = cat_results

    return results


# ===== Main =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log(f"Phase 368: Δh Subspace Analysis — {model_name}")
    log(f"=" * 60)

    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    n_pairs = len(TEST_PAIRS)

    # Load model
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    log(f"Model loaded in {t_load:.0f}s")

    # Get W_U
    W_U = get_W_U(model, model_name)
    log(f"W_U shape: {W_U.shape}")

    # Collect Δh vectors
    t0 = time.time()
    dh_per_pair, embedding_diffs, pair_labels = collect_dh_vectors(
        model, tokenizer, device, model_name)
    t_collect = time.time() - t0
    log(f"Δh collection done in {t_collect:.0f}s ({n_pairs} pairs × {n_layers+1} layers)")

    # Free model memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("Model freed, GPU cleared")

    # Part A: PCA analysis
    pca_results = compute_pca_analysis(dh_per_pair, n_layers, n_pairs)

    # Cross-layer PCA evolution (all layers)
    pca_evolution = compute_cross_layer_pca_evolution(dh_per_pair, n_layers, n_pairs)

    # Per-category PCA
    category_pca = compute_per_category_pca(dh_per_pair, n_layers)

    # Part B: Increment decomposition
    increment_results = compute_increment_decomposition(dh_per_pair, W_U, n_layers, n_pairs)

    # Part C: Embedding source analysis
    embedding_results = compute_embedding_analysis(
        embedding_diffs, dh_per_pair, W_U, n_layers, n_pairs)

    # Save results
    output = {
        "model": model_name,
        "phase": "368",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": n_pairs,
        "n_layers": n_layers,
        "d_model": cfg["d_model"],
        "load_time_s": round(t_load, 1),
        "collect_time_s": round(t_collect, 1),
        "pca_key_layers": pca_results,
        "pca_evolution": pca_evolution,
        "category_pca": category_pca,
        "increment_decomposition": increment_results,
        "embedding_analysis": embedding_results,
    }

    os.makedirs("results/phase368_dh_subspace", exist_ok=True)
    out_path = f"results/phase368_dh_subspace/{model_name}_phase368.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    # Print summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    # PCA summary
    log("\n--- PCA: Is binding 1D or subspace? ---")
    for ls in sorted(pca_results.keys(), key=lambda x: int(x)):
        r = pca_results[ls]
        log(f"  L{ls}: PC1={r['explained_ratio_top10'][0]:.3f}, "
            f"eff_rank_95={r['effective_rank_95']}, "
            f"cos(Δh, PC1)={r['cos_sim_with_pc1_mean']:.3f}±{r['cos_sim_with_pc1_std']:.3f}")

    # Increment summary
    log("\n--- Increment: What does each layer add? ---")
    for ls in sorted(increment_results.keys(), key=lambda x: int(x)):
        r = increment_results[ls]
        log(f"  L{ls}: {r['increment_type']}, cos(δ,Δh_prev)={r['cos_delta_prev_dh_mean']:.3f}")

    # Embedding summary
    log("\n--- Embedding: Where does L0 signal come from? ---")
    if embedding_results:
        log(f"  cos(Δe_emb, Δh_final) = {embedding_results.get('per_pair_cos_emb_final_mean', 0):.4f}")
        log(f"  cos(Δh_L0, Δh_final)  = {embedding_results.get('per_pair_cos_l0_final_mean', 0):.4f}")

    total_time = time.time() - t0 + t_load
    log(f"\nTotal time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
