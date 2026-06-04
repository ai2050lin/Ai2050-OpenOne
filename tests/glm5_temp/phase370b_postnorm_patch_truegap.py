"""
Phase 370b: Post-RMSNorm PC Causal Patch with True Target-Competitor Gap
=========================================================================

Fix the Part B recovery calculation from Phase 370:
- Use actual target/competitor token IDs instead of logit norm
- Compute real logit gap recovery: (W_U @ Δh_k)[target] - (W_U @ Δh_k)[competitor]
  divided by (W_U @ Δh_full)[target] - (W_U @ Δh_full)[competitor]

This is the critical test: does post-RMSNorm space need k=1 or k=20+ PCs
to recover the binding gap?

Models: deepseek7b (primary), qwen3, glm4
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
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
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
            log(f"  Loaded with attn_impl={impl}")
            break
        except:
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers")


def get_W_U(model, model_name):
    for attr in ["lm_head", "embed_out"]:
        if hasattr(model, attr):
            w = getattr(model, attr)
            if hasattr(w, "weight"):
                weight = w.weight
                if not weight.is_meta:
                    return weight.detach().cpu().float().numpy()
    import glob
    from safetensors import safe_open
    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            for key in sf.keys():
                if 'lm_head' in key or 'embed_out' in key:
                    return sf.get_tensor(key).float().numpy()
    return None


def rms_norm_pytorch(x, weight, eps=1e-6):
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps)
    return x_normed * weight


def get_layer_norm_weights(model, model_name, n_layers):
    import glob
    from safetensors import safe_open
    layers = get_layers(model)
    ln_weights = {}
    for l in range(n_layers):
        layer = layers[l]
        for attr in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, attr):
                ln = getattr(layer, attr)
                if hasattr(ln, "weight"):
                    w = ln.weight
                    if not w.is_meta:
                        ln_weights[l] = w.detach().cpu().float().numpy()
                        break
                    key = f"model.layers.{l}.{attr}.weight"
                    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
                        with safe_open(sf_file, framework='pt', device='cpu') as sf:
                            if key in sf.keys():
                                ln_weights[l] = sf.get_tensor(key).float().numpy()
                                break
                    if l in ln_weights:
                        break
    return ln_weights


def get_token_id(tokenizer, word):
    """Get the single token ID for a word."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    # Try with space prefix
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    # Return first token
    return ids[0] if ids else None


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        return

    log(f"Phase 370b: Post-RMSNorm PC Causal Patch with True Gap")
    log(f"Model: {model_name}")

    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    n_pairs = len(TEST_PAIRS)

    # 1. Load model
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"Model loaded in {time.time()-t0:.1f}s")

    # 2. Get W_U
    log("Getting W_U...")
    W_U = get_W_U(model, model_name)
    if W_U is not None:
        log(f"  W_U shape: {W_U.shape}")
    else:
        log("  FATAL: W_U not available")
        return

    # 3. Get LN weights
    log("Getting LN weights...")
    ln_weights = get_layer_norm_weights(model, model_name, n_layers)
    log(f"  Got {len(ln_weights)} LN weights")

    # 4. Get target/competitor token IDs
    log("Getting token IDs...")
    target_ids = []
    competitor_ids = []
    for obj, target, competitor in TEST_PAIRS:
        tid = get_token_id(tokenizer, target)
        cid = get_token_id(tokenizer, competitor)
        target_ids.append(tid)
        competitor_ids.append(cid)
    valid_count = sum(1 for t, c in zip(target_ids, competitor_ids) if t is not None and c is not None)
    log(f"  Got {valid_count}/{n_pairs} valid target/competitor token IDs")

    # 5. Collect hidden states
    key_layers = sorted(set(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] +
        list(range(10, n_layers, max(1, n_layers // 6))) +
        [n_layers - 2, n_layers - 1, n_layers]
    ))
    key_layers = [l for l in key_layers if l <= n_layers]
    
    h_clean_all = {l: np.zeros((n_pairs, d_model), dtype=np.float32) for l in key_layers}
    h_corrupt_all = {l: np.zeros((n_pairs, d_model), dtype=np.float32) for l in key_layers}
    
    input_device = next(model.parameters()).device
    
    log("Collecting hidden states...")
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
        
        for l in key_layers:
            h_clean_all[l][pidx] = clean_out.hidden_states[l][0, last_pos_c].detach().cpu().float().numpy()
            h_corrupt_all[l][pidx] = corrupt_out.hidden_states[l][0, last_pos_r].detach().cpu().float().numpy()
        
        del clean_out, corrupt_out
        if pidx % 5 == 0:
            torch.cuda.empty_cache()

    # Release model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("Model released")

    # 6. Compute post-RMSNorm Δh and do PC causal patch
    log("\n" + "="*60)
    log("Post-RMSNorm PC Causal Patch with True Target-Competitor Gap")
    log("="*60)
    
    ones = np.ones(d_model, dtype=np.float32)
    k_values = [1, 3, 5, 10, 20, 50]
    results = {}
    
    for l in key_layers:
        M_clean = h_clean_all[l]
        M_corrupt = h_corrupt_all[l]
        M_dh_raw = M_clean - M_corrupt
        
        # Compute post-RMSNorm Δh
        if (l + 1) in ln_weights and l < n_layers:
            ln_w = ln_weights[l + 1]
        else:
            ln_w = ones
        
        M_dh_postnorm = np.zeros_like(M_dh_raw)
        for i in range(n_pairs):
            normed_clean = rms_norm_pytorch(M_clean[i], ln_w)
            normed_corrupt = rms_norm_pytorch(M_corrupt[i], ln_w)
            M_dh_postnorm[i] = normed_clean - normed_corrupt
        
        # PCA on both raw and post-norm Δh
        def do_pca(M):
            M_c = M - M.mean(axis=0, keepdims=True)
            try:
                U, S, Vt = np.linalg.svd(M_c, full_matrices=False)
                total = np.sum(S**2)
                if total < 1e-10:
                    return None, None, None, None
                explained = (S**2) / total
                eff_rank = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
                return U, S, Vt, {"pc1": float(explained[0]), "eff_rank": eff_rank}
            except:
                return None, None, None, None
        
        _, _, Vt_raw, raw_info = do_pca(M_dh_raw)
        _, S_pn, Vt_pn, pn_info = do_pca(M_dh_postnorm)
        
        if raw_info is None or pn_info is None:
            continue
        
        # Compute per-pair logit gap for both raw and post-norm
        # Gap = (W_U @ Δh)[target] - (W_U @ Δh)[competitor]
        
        layer_result = {
            "raw_pc1": raw_info["pc1"],
            "raw_eff_rank": raw_info["eff_rank"],
            "postnorm_pc1": pn_info["pc1"],
            "postnorm_eff_rank": pn_info["eff_rank"],
        }
        
        # For each k, compute recovery
        for space_name, M_dh, Vt_space, info_key in [
            ("raw", M_dh_raw, Vt_raw, "raw"),
            ("postnorm", M_dh_postnorm, Vt_pn, "pn"),
        ]:
            mean_dh = M_dh.mean(axis=0)
            
            # Base gap: per-pair average gap using full Δh
            base_gaps = []
            for i in range(n_pairs):
                tid = target_ids[i]
                cid = competitor_ids[i]
                if tid is None or cid is None:
                    continue
                logits = W_U @ M_dh[i]
                gap = logits[tid] - logits[cid]
                base_gaps.append(gap)
            
            if len(base_gaps) == 0:
                continue
            
            mean_base_gap = float(np.mean(base_gaps))
            
            if abs(mean_base_gap) < 1e-6:
                # Very small gap, skip to avoid division issues
                layer_result[f"{space_name}_base_gap"] = float(mean_base_gap)
                layer_result[f"{space_name}_skip"] = "tiny_gap"
                continue
            
            layer_result[f"{space_name}_base_gap"] = float(mean_base_gap)
            
            for k in k_values:
                if k > Vt_space.shape[0]:
                    continue
                
                # Reconstruct with top-k PCs
                M_c = M_dh - mean_dh
                coeffs = M_c @ Vt_space[:k].T  # (n_pairs, k)
                reconstructed = coeffs @ Vt_space[:k] + mean_dh  # (n_pairs, d_model)
                
                # Compute gap for reconstructed Δh
                recon_gaps = []
                for i in range(n_pairs):
                    tid = target_ids[i]
                    cid = competitor_ids[i]
                    if tid is None or cid is None:
                        continue
                    logits_recon = W_U @ reconstructed[i]
                    gap_recon = logits_recon[tid] - logits_recon[cid]
                    recon_gaps.append(gap_recon)
                
                if len(recon_gaps) == 0:
                    continue
                
                mean_recon_gap = float(np.mean(recon_gaps))
                recovery = mean_recon_gap / mean_base_gap if abs(mean_base_gap) > 1e-6 else 0
                
                # Also compute per-pair recovery stats
                per_pair_rec = []
                for bg, rg in zip(base_gaps, recon_gaps):
                    if abs(bg) > 1e-6:
                        per_pair_rec.append(rg / bg)
                
                layer_result[f"{space_name}_k{k}_recovery"] = float(recovery)
                layer_result[f"{space_name}_k{k}_per_pair_mean"] = float(np.mean(per_pair_rec)) if per_pair_rec else 0
                layer_result[f"{space_name}_k{k}_per_pair_std"] = float(np.std(per_pair_rec)) if len(per_pair_rec) > 1 else 0
                layer_result[f"{space_name}_k{k}_per_pair_median"] = float(np.median(per_pair_rec)) if per_pair_rec else 0
        
        results[str(l)] = layer_result
        
        # Print summary
        raw_k1 = layer_result.get("raw_k1_recovery", "N/A")
        raw_k5 = layer_result.get("raw_k5_recovery", "N/A")
        pn_k1 = layer_result.get("postnorm_k1_recovery", "N/A")
        pn_k5 = layer_result.get("postnorm_k5_recovery", "N/A")
        pn_k10 = layer_result.get("postnorm_k10_recovery", "N/A")
        pn_k20 = layer_result.get("postnorm_k20_recovery", "N/A")
        
        log(f"  L{l}: raw({raw_info['pc1']:.3f},{raw_info['eff_rank']}D)→pn({pn_info['pc1']:.3f},{pn_info['eff_rank']}D) | "
            f"raw_k1={raw_k1}, pn_k1={pn_k1}, pn_k5={pn_k5}, pn_k10={pn_k10}, pn_k20={pn_k20}")
    
    # 7. Save results
    output = {
        "model": model_name,
        "phase": "370b",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": n_pairs,
        "n_layers": n_layers,
        "d_model": d_model,
        "key_layers": key_layers,
        "pc_causal_patch": results,
    }
    
    os.makedirs("results/phase370_norm_mask", exist_ok=True)
    out_path = f"results/phase370_norm_mask/{model_name}_phase370b.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")
    
    # 8. Summary table
    log("\n" + "="*60)
    log("SUMMARY: Post-RMSNorm PC Causal Patch")
    log("="*60)
    log(f"{'Layer':>5} | {'raw_PC1':>8} | {'raw_rank':>8} | {'pn_PC1':>7} | {'pn_rank':>7} | "
        f"{'raw_k1':>8} | {'pn_k1':>8} | {'pn_k5':>8} | {'pn_k10':>8} | {'pn_k20':>8}")
    log("-" * 95)
    
    for l in sorted(results.keys(), key=int):
        r = results[l]
        raw_pc1 = r.get("raw_pc1", -1)
        raw_rank = r.get("raw_eff_rank", -1)
        pn_pc1 = r.get("postnorm_pc1", -1)
        pn_rank = r.get("postnorm_eff_rank", -1)
        
        def fmt(val):
            if isinstance(val, str):
                return val[:8]
            if val is None or val == "N/A":
                return "N/A"
            return f"{val:.3f}"
        
        raw_k1 = fmt(r.get("raw_k1_recovery", "N/A"))
        pn_k1 = fmt(r.get("postnorm_k1_recovery", "N/A"))
        pn_k5 = fmt(r.get("postnorm_k5_recovery", "N/A"))
        pn_k10 = fmt(r.get("postnorm_k10_recovery", "N/A"))
        pn_k20 = fmt(r.get("postnorm_k20_recovery", "N/A"))
        
        log(f"L{l:>4} | {raw_pc1:.3f}   | {raw_rank:>8} | {pn_pc1:.3f}  | {pn_rank:>7} | "
            f"{raw_k1:>8} | {pn_k1:>8} | {pn_k5:>8} | {pn_k10:>8} | {pn_k20:>8}")
    
    log(f"\nPhase 370b complete for {model_name}!")


if __name__ == "__main__":
    main()
